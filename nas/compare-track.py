#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
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
        "arch2",
        "data/models/left_wall_dist_arch2.pt",
        "data/models/track_width_arch2.pt",
        "data/models/heading_error_arch2.pt",
    ),
    (
        "arch3",
        "data/models/left_wall_dist_arch3.pt",
        "data/models/track_width_arch3.pt",
        "data/models/heading_error_arch3.pt",
    ),
    (
        "arch4",
        "data/models/left_wall_dist_arch4.pt",
        "data/models/track_width_arch4.pt",
        "data/models/heading_error_arch4.pt",
    ),
    (
        "arch5",
        "data/models/left_wall_dist_arch5.pt",
        "data/models/track_width_arch5.pt",
        "data/models/heading_error_arch5.pt",
    ),
    (
        "arch6",
        "data/models/left_wall_dist_arch6.pt",
        "data/models/track_width_arch6.pt",
        "data/models/heading_error_arch6.pt",
    ),
    (
        "arch7",
        "data/models/left_wall_dist_arch7.pt",
        "data/models/track_width_arch7.pt",
        "data/models/heading_error_arch7.pt",
    ),
    (
        "arch8",
        "data/models/left_wall_dist_arch8.pt",
        "data/models/track_width_arch8.pt",
        "data/models/heading_error_arch8.pt",
    ),
]
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("compare-map")
DEFAULT_MAP_ROOT = "data/maps"
DEFAULT_RUN_ID = None
DEFAULT_ALL_MAPS = True


@dataclass
class ModelRun:
    label: str
    left: str
    track: str
    heading: str


@dataclass
class MapSpec:
    name: str
    map_base: str
    map_ext: str
    waypoints: str


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
        default=None,
        help="Optional explicit PNG path (single-map mode only).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR.as_posix(),
        help="Directory where compare-map outputs (figures + metrics) are stored.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Override the auto-generated identifier for this compare-track batch.",
    )
    parser.add_argument(
        "--all-maps",
        action="store_true",
        default=DEFAULT_ALL_MAPS,
        help="Iterate over every map with waypoints under --maps-root.",
    )
    parser.add_argument(
        "--maps-root",
        default=DEFAULT_MAP_ROOT,
        help="Root directory to scan for maps when --all-maps is set.",
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
    map_spec: MapSpec,
) -> tuple[SimTrace, dict]:
    args = argparse.Namespace(
        planner="dnn",
        left_wall_model=str(run.left),
        track_width_model=str(run.track),
        heading_model=str(run.heading),
        lookahead=base_args.lookahead,
        lateral_gain=base_args.lateral_gain,
        speed=base_args.speed,
        map=map_spec.map_base,
        map_ext=map_spec.map_ext,
        waypoints=map_spec.waypoints,
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


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _normalize_map_name(map_path: str) -> str:
    stem = Path(map_path).stem
    return stem[:-4] if stem.endswith("_map") else stem


def _discover_map_specs(root: str | Path) -> list[MapSpec]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Map root {root_path} does not exist.")

    specs: list[MapSpec] = []
    for waypoint_file in sorted(root_path.rglob("*_centerline.*")):
        base_name = waypoint_file.stem.rsplit("_centerline", 1)[0]
        map_dir = waypoint_file.parent
        map_base = map_dir / f"{base_name}_map"
        map_yaml = map_base.with_suffix(".yaml")
        if not map_yaml.exists():
            continue
        image_path: Path | None = None
        for ext in (".png", ".pgm", ".jpg", ".jpeg"):
            candidate = map_base.with_suffix(ext)
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue
        rel_dir = map_dir.relative_to(root_path)
        display_name = str(rel_dir / base_name).replace("\\", "/") if rel_dir != Path(".") else base_name
        specs.append(
            MapSpec(
                name=display_name,
                map_base=map_base.as_posix(),
                map_ext=image_path.suffix,
                waypoints=waypoint_file.as_posix(),
            )
        )
    return specs


def _prepare_run_directory(
    output_dir: str | Path, run_id: str | None
) -> tuple[Path, str]:
    base_dir = Path(output_dir).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    identifier = run_id or datetime.now(UTC).strftime("compare_%Y%m%dT%H%M%S")

    run_dir = base_dir / identifier
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = base_dir / f"{identifier}_{suffix}"
    if suffix > 1:
        identifier = f"{identifier}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, identifier


def _build_map_list(args: argparse.Namespace) -> list[MapSpec]:
    if args.all_maps:
        specs = _discover_map_specs(args.maps_root)
        if not specs:
            raise RuntimeError(f"No maps with waypoints found under {args.maps_root}")
        return specs
    single_name = _normalize_map_name(args.map)
    return [
        MapSpec(
            name=single_name,
            map_base=args.map,
            map_ext=args.map_ext,
            waypoints=args.waypoints,
        )
    ]


def _run_map_comparison(
    args: argparse.Namespace,
    runs: list[ModelRun],
    map_spec: MapSpec,
    run_dir: Path,
    run_identifier: str,
) -> dict | None:
    try:
        waypoints = load_waypoints(map_spec.waypoints)
    except FileNotFoundError:
        print(f"[warn] Missing waypoints at {map_spec.waypoints}; skipping {map_spec.name}")
        return None
    if waypoints.size == 0:
        print(f"[warn] No waypoints found at {map_spec.waypoints}; skipping {map_spec.name}")
        return None

    traces: list[tuple[ModelRun, SimTrace]] = []
    run_summaries: list[dict[str, object]] = []
    for run in runs:
        trace, metrics = _simulate_run(args, run, waypoints, map_spec)
        traces.append((run, trace))
        rmse = metrics.get("crosstrack_rmse_m")
        rmse_value = float(rmse) if rmse is not None else None
        rmse_txt = f"{rmse_value:.4f} m" if rmse_value is not None else "n/a"
        print(
            f"{map_spec.name}: {run.label} lap RMSE={rmse_txt}  steps={len(trace.positions)}"
        )
        run_summaries.append(
            {
                "label": run.label,
                "left": run.left,
                "track": run.track,
                "heading": run.heading,
                "rmse": rmse_value,
            }
        )

    img, extent = _load_map_background(map_spec.map_base, map_spec.map_ext)
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
    run_labels = ", ".join(run.label for run in runs)
    ax.set_title(f"{map_spec.name} ({run_labels})", fontsize=16)
    if traces:
        ax.legend(loc="upper right")
    ax.axis("off")

    map_slug = _slugify(map_spec.name.replace("/", "_"))
    image_name = f"{map_slug}_{run_identifier}.png"
    image_path = run_dir / image_name
    fig.savefig(image_path, dpi=600, bbox_inches="tight")
    print(f"Saved comparison plot to {image_path}")

    if args.output and not args.all_maps:
        mirror_path = Path(args.output).expanduser()
        if not mirror_path.is_absolute():
            mirror_path = Path(__file__).resolve().parent / mirror_path
        mirror_path = mirror_path.resolve()
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(mirror_path, dpi=600, bbox_inches="tight")
        print(f"Saved comparison plot to {mirror_path}")

    if args.show:
        plt.show()
    plt.close(fig)

    return {
        "map": map_spec.name,
        "waypoints": map_spec.waypoints,
        "runs": run_summaries,
        "image": image_name,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    runs = _build_runs(args.run)
    map_specs = _build_map_list(args)
    run_dir, run_identifier = _prepare_run_directory(args.output_dir, args.run_id)
    safe_run_id = _slugify(run_identifier)

    records: list[dict] = []
    for map_spec in map_specs:
        record = _run_map_comparison(args, runs, map_spec, run_dir, safe_run_id)
        if record:
            records.append(record)

    if records:
        metrics_path = run_dir / "metrics.jsonl"
        with metrics_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                json.dump(rec, fh)
                fh.write("\n")
        print(f"Wrote RMSE summaries to {metrics_path}")
    else:
        print("No comparison plots were generated.")


if __name__ == "__main__":
    main()
