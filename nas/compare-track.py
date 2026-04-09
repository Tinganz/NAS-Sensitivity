#!/usr/bin/env python3
"""Simulate DNN planners and compare their track traces side-by-side."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from f110_planning.utils.sim_utils import resolve_start_pose, setup_env  # noqa: E402
from f110_planning.utils.waypoint_utils import load_waypoints  # noqa: E402
from f110_planning.visualization.svg_trace import SimTrace  # noqa: E402
from f110_scripts.sim import reactive_planners as sim  # noqa: E402


DEFAULT_MAP = "data/maps/F1/Nuerburgring/Nuerburgring_map"
DEFAULT_WAYPOINTS = "data/maps/F1/Nuerburgring/Nuerburgring_centerline.tsv"
DEFAULT_RUNS = [
    (
        "arch1",
        "data/models/left_wall_dist_arch1.pt",
        "data/models/track_width_arch1.pt",
        "data/models/heading_error_arch1.pt",
    ),
    (
        "arch8",
        "data/models/left_wall_dist_arch8.pt",
        "data/models/track_width_arch8.pt",
        "data/models/heading_error_arch8.pt",
    ),
]
DEFAULT_OUTPUT = (Path(__file__).with_name("compare-map") / "compare_track.png").as_posix()


@dataclass
class ModelRun:
    label: str
    left: str
    track: str
    heading: str


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DNN planners on a track and compare their traces."
    )
    parser.add_argument(
        "--map",
        default=DEFAULT_MAP,
        help="Base path (without extension) to the map YAML/image.",
    )
    parser.add_argument(
        "--map-ext",
        default=".png",
        help="Extension for the map image (e.g. .png, .pgm).",
    )
    parser.add_argument(
        "--waypoints",
        default=DEFAULT_WAYPOINTS,
        help="Waypoint TSV/CSV used for reference poses.",
    )
    parser.add_argument(
        "--laps",
        type=int,
        default=1,
        help="Number of laps to complete before terminating each run.",
    )
    parser.add_argument(
        "--render-mode",
        default="None",
        choices=["human", "human_fast", "None"],
        help="Optional pyglet visualization mode.",
    )
    parser.add_argument(
        "--lookahead",
        type=float,
        default=1.0,
        help="Lookahead gain used by the planner.",
    )
    parser.add_argument(
        "--lateral-gain",
        type=float,
        default=1.0,
        help="Lateral centering gain for the planner.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=5.0,
        help="Maximum planner speed (m/s).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Figure output path (relative paths are resolved next to this script).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    parser.add_argument(
        "--run",
        nargs=4,
        action="append",
        metavar=("LABEL", "LEFT_PT", "TRACK_PT", "HEADING_PT"),
        help="Add a labelled planner run. Pass three .pt files per label.",
    )
    return parser.parse_args(argv)


def _load_map_background(map_path: str, map_ext: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    with open(map_path + ".yaml", encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)

    resolution: float = float(meta["resolution"])
    origin_x: float = float(meta["origin"][0])
    origin_y: float = float(meta["origin"][1])

    img = (
        Image.open(map_path + map_ext)
        .convert("L")
        .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    )
    arr = np.array(img)
    height_m = arr.shape[0] * resolution
    width_m = arr.shape[1] * resolution
    extent = (
        origin_x,
        origin_x + width_m,
        origin_y,
        origin_y + height_m,
    )
    return arr, extent


def _simulate_run(
    base_args: argparse.Namespace,
    run: ModelRun,
    waypoints: np.ndarray,
) -> tuple[SimTrace, dict]:
    args = argparse.Namespace(
        planner="dnn",
        left_wall_model=str(run.left),
        track_width_model=str(run.track),
        heading_model=str(run.heading),
        lookahead=base_args.lookahead,
        lateral_gain=base_args.lateral_gain,
        speed=base_args.speed,
        map=base_args.map,
        map_ext=base_args.map_ext,
        waypoints=base_args.waypoints,
        start_x=0.0,
        start_y=0.0,
        start_theta=None,
        render_mode=base_args.render_mode,
        render_fps=60,
        max_laps=base_args.laps,
        randomize=False,
        camera_tracking=False,
    )
    r_mode = None if args.render_mode == "None" else args.render_mode
    env = setup_env(args, r_mode)
    planner = sim._create_planner(args, waypoints)

    pose = np.array([list(resolve_start_pose(args))])
    obs, _ = env.reset(options={"poses": pose})
    trace = SimTrace()
    _, metrics = sim._run_reactive_sim(
        env,
        obs,
        planner,
        r_mode,
        waypoints,
        trace=trace,
    )
    env.close()
    return trace, metrics


def _build_runs(run_args: list[list[str]] | None) -> list[ModelRun]:
    if not run_args:
        run_args = DEFAULT_RUNS
    runs: list[ModelRun] = []
    for label, left, track, heading in run_args:
        runs.append(
            ModelRun(
                label=label,
                left=left,
                track=track,
                heading=heading,
            )
        )
    return runs


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    runs = _build_runs(args.run)

    waypoints = load_waypoints(args.waypoints)
    if waypoints.size == 0:
        raise RuntimeError(f"Failed to load waypoints at {args.waypoints}")

    traces: list[tuple[ModelRun, SimTrace]] = []
    for run in runs:
        trace, metrics = _simulate_run(args, run, waypoints)
        traces.append((run, trace))
        rmse = metrics.get("crosstrack_rmse_m")
        rmse_txt = f"{rmse:.4f} m" if rmse is not None else "n/a"
        print(f"{run.label}: lap RMSE={rmse_txt}  steps={len(trace.positions)}")

    img, extent = _load_map_background(args.map, args.map_ext)
    fig, ax = plt.subplots(figsize=(14, 14), dpi=320)
    ax.imshow(
        img,
        cmap="gray_r",
        extent=extent,
        origin="lower",
        alpha=0.35,
        zorder=0,
    )

    if waypoints.size > 0:
        ax.plot(
            waypoints[:, 0],
            waypoints[:, 1],
            color="#444",
            linewidth=0.5,
            linestyle="--",
            label="reference",
            zorder=1,
        )

    for run, trace in traces:
        pts = np.asarray(trace.positions)
        if pts.size == 0:
            continue
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            linewidth=0.25,
            label=run.label,
            zorder=3,
        )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    map_name = Path(args.map).stem
    run_labels = ", ".join(run.label for run in runs)
    ax.set_title(f"{map_name} ({run_labels})", fontsize=16)
    if traces:
        ax.legend(loc="upper right")
    ax.axis("off")

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    print(f"Saved comparison plot to {out_path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
