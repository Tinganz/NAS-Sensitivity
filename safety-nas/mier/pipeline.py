from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent        # safety-nas/mier/
SAFETY_NAS_DIR = BASE_DIR.parent                  # safety-nas/
REPO_ROOT = SAFETY_NAS_DIR.parent                 # NAS-Sensitivity/

for _p in [str(BASE_DIR), str(SAFETY_NAS_DIR), str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


NAS_MAX_PARAMS = [34405, 34405, 137777]   # baseline
DEFAULT_DATASET = "safety-nas/datasets/combined_all.npz"

DEFAULT_DNN_OUTPUT   = BASE_DIR / "output" / "search"
DEFAULT_STAGING_ROOT = BASE_DIR / "output" / "retrain"
DEFAULT_COMPARE_OUT  = BASE_DIR / "output" / "compare"

TRAINING_PROFILES: dict[int, dict[str, Any]] = {
    0: {"label": "arch1-2 (150 ep)",  "max_epochs": 150, "lr": 1e-3,  "weight_decay": 1e-5, "early_stopping_patience": 15, "lr_patience": 10, "optimizer": "adam"},
    1: {"label": "arch3-4 (250 ep)",  "max_epochs": 250, "lr": 5e-4,  "weight_decay": 1e-5, "early_stopping_patience": 20, "lr_patience": 10, "optimizer": "adam"},
    2: {"label": "arch5   (450 ep)",  "max_epochs": 450, "lr": 1e-4,  "weight_decay": 1e-5, "early_stopping_patience": 60, "lr_patience": 20, "optimizer": "adam"},
    3: {"label": "arch6-7 (700 ep)",  "max_epochs": 700, "lr": 5e-5,  "weight_decay": 1e-4, "early_stopping_patience": 60, "lr_patience": 20, "optimizer": "adamw"},
}

# comb1 = arch3 (left, track) + arch5 (heading) 
BASELINE_RUNS = [
    ("comb1",
     "data/models/left_wall_dist_arch3.pt",
     "data/models/track_width_arch3.pt",
     "data/models/heading_error_arch5.pt"),
]

SELECTED_TRACKS = {
    "shanghai", "silverstone", "sochi", "spa",
    "nuerburgring", "monza", "mexicocity",
}

TRACK_METRIC_KEYS: tuple[str, ...] = (
    "crosstrack_rmse_m",
    "crosstrack_mean_m",
    "crosstrack_std_m",
    "crosstrack_max_m",
    "heading_error_rmse_deg",
    "heading_error_max_deg",
    "wall_min_distance_m",
    "steering_rate_mean_rad_s",
    "steering_rate_max_rad_s",
    "steering_rate_std_rad_s",
    "collision",
    "laps_completed",
    "speed_mean_m_s",
    "speed_std_m_s",
)

def cmd_search(args: argparse.Namespace) -> None:
    """Run the Optuna NAS architecture search."""
    import optuna
    import torch
    from cnn import EvaluationTrack, objective

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    track_names = None
    if args.track:
        track_names = [EvaluationTrack[args.track.strip().upper()]]

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: objective(t, track_names=track_names, max_params=NAS_MAX_PARAMS),
        n_trials=args.n_trials,
    )
    best = study.best_trial
    print(f"[search] best trial #{best.number}  value={best.value:.4f}")

def _target_stem(path: Path) -> str:
    """Extract target name (e.g. 'heading_error') from a config/checkpoint path."""
    stem = path.stem
    return stem if "_arch" not in stem else stem.rsplit("_arch", 1)[0]


def _stage(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest


def cmd_retrain(args: argparse.Namespace) -> None:
    """Pick the best NAS trial and retrain it to full epochs."""
    from cnn import TRAIN_EVAL_TRACKS
    from testing import test_cnn_arch
    from training import orchestrate_best_trial, train_from_configs

    if args.trials_file:
        trials_path = Path(args.trials_file).expanduser().resolve()
        if not trials_path.exists():
            raise FileNotFoundError(f"Trials file not found: {trials_path}")
    else:
        candidates = sorted(DEFAULT_DNN_OUTPUT.glob("nas_trials_*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No nas_trials_*.jsonl under {DEFAULT_DNN_OUTPUT}")
        trials_path = candidates[-1]
    print(f"[retrain] trials file: {trials_path}")

    profile = TRAINING_PROFILES[args.profile]
    print(f"[retrain] profile {args.profile}: {profile['label']}")

    session_hash = trials_path.stem.rsplit("_", 1)[-1]
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = DEFAULT_STAGING_ROOT / session_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[retrain] staging to: {output_dir}")

    best_trial, config_paths = orchestrate_best_trial(
        trials_path=str(trials_path),
        dataset_path=args.dataset,
        output_dir=str(output_dir),
        max_epochs=profile["max_epochs"],
        lr=profile["lr"],
        weight_decay=profile["weight_decay"],
        early_stopping_patience=profile["early_stopping_patience"],
        lr_patience=profile["lr_patience"],
        optimizer=profile["optimizer"],
    )
    print(
        f"[retrain] best trial #{best_trial['trial_number']}  "
        f"avg_rmse={best_trial['rmse'][0]['value']:.4f}"
    )
    print(f"[retrain] param_counts: {best_trial.get('param_counts', {})}")

    config_paths = [Path(p) for p in config_paths]
    trained_pts = train_from_configs(config_paths)

    staged: list[Path] = []
    for cfg_path, pt_path in zip(config_paths, trained_pts):
        pt_src = pt_path if pt_path.is_absolute() else (REPO_ROOT / pt_path)
        staged.append(_stage(pt_src, cfg_path.with_suffix(".pt")))
        print(f"[retrain] staged {staged[-1].name}")

    lookup = {_target_stem(p): str(p) for p in staged}
    required = {"left_wall_dist", "track_width", "heading_error"}
    if not required.issubset(lookup):
        print(f"[retrain] warning: missing checkpoints for {required - set(lookup)}")
        return

    track_configs = [(t.map_path, t.waypoints_path) for t in TRAIN_EVAL_TRACKS]
    avg_rmse, _, _, _ = test_cnn_arch(
        left_wall_dist_filepath=lookup["left_wall_dist"],
        track_width_filepath=lookup["track_width"],
        heading_error_filepath=lookup["heading_error"],
        track_configs=track_configs,
    )
    print(f"[retrain] RMSE on {len(track_configs)} training track(s): {avg_rmse:.4f}")
    print(f"[retrain] → run compare with:  --nas-dir {output_dir}")

@dataclass
class _ModelRun:
    label: str
    left: str
    track: str
    heading: str


@dataclass
class _MapSpec:
    name: str
    map_base: str
    map_ext: str
    waypoints: str


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _normalize_map_name(map_path: str) -> str:
    stem = Path(map_path).stem
    return stem[:-4] if stem.endswith("_map") else stem


def _discover_map_specs(root: str | Path) -> list[_MapSpec]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Map root {root_path} does not exist.")
    specs: list[_MapSpec] = []
    for waypoint_file in sorted(root_path.rglob("*_centerline.*")):
        base_name = waypoint_file.stem.rsplit("_centerline", 1)[0]
        map_dir = waypoint_file.parent
        map_base = map_dir / f"{base_name}_map"
        if not map_base.with_suffix(".yaml").exists():
            continue
        image_path: Path | None = None
        for ext in (".png", ".pgm", ".jpg", ".jpeg"):
            if (candidate := map_base.with_suffix(ext)).exists():
                image_path = candidate
                break
        if image_path is None:
            continue
        rel_dir = map_dir.relative_to(root_path)
        display_name = (
            str(rel_dir / base_name).replace("\\", "/")
            if rel_dir != Path(".")
            else base_name
        )
        specs.append(_MapSpec(
            name=display_name,
            map_base=map_base.as_posix(),
            map_ext=image_path.suffix,
            waypoints=waypoint_file.as_posix(),
        ))
    return specs


def _find_nas_model_pts(nas_dir: Path) -> dict[str, str] | None:
    patterns = {
        "left_wall_dist": "*left_wall_dist*.pt",
        "track_width":    "*track_width*.pt",
        "heading_error":  "*heading_error*.pt",
    }
    found: dict[str, str] = {}
    for key, pattern in patterns.items():
        matches = sorted(nas_dir.glob(pattern))
        if not matches:
            return None
        found[key] = str(matches[-1])
    return found


def _label_for_nas_dir(nas_dir: Path) -> str:
    return _slugify(nas_dir.name)


def _load_map_background(map_path: str, map_ext: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    with open(map_path + ".yaml", encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)
    resolution: float = float(meta["resolution"])
    origin_x: float = float(meta["origin"][0])
    origin_y: float = float(meta["origin"][1])
    from PIL import Image
    img = (
        Image.open(map_path + map_ext)
        .convert("L")
        .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    )
    arr = np.array(img)
    height_m = arr.shape[0] * resolution
    width_m  = arr.shape[1] * resolution
    extent = (origin_x, origin_x + width_m, origin_y, origin_y + height_m)
    return arr, extent


def _simulate_run(
    run: _ModelRun,
    waypoints: np.ndarray,
    map_spec: _MapSpec,
    lookahead: float = 1.5,
    lateral_gain: float = 1.0,
    speed: float | None = None,
    render_mode: str = "None",
    max_laps: int = 1,
) -> tuple[Any, dict]:
    from f110_planning.utils.sim_utils import resolve_start_pose, setup_env
    from f110_planning.visualization.svg_trace import SimTrace
    from f110_scripts.sim import reactive_planners as sim

    sim_args = SimpleNamespace(
        planner="dnn",
        left_wall_model=str(run.left),
        track_width_model=str(run.track),
        heading_model=str(run.heading),
        lookahead=lookahead,
        lateral_gain=lateral_gain,
        speed=speed,
        map=map_spec.map_base,
        map_ext=map_spec.map_ext,
        waypoints=map_spec.waypoints,
        start_x=0.0,
        start_y=0.0,
        start_theta=None,
        render_mode=render_mode,
        render_fps=60,
        max_laps=max_laps,
        randomize=False,
        camera_tracking=False,
    )
    r_mode = None if render_mode == "None" else render_mode
    env = setup_env(sim_args, r_mode)
    planner = sim._create_planner(sim_args, waypoints)
    pose = np.array([list(resolve_start_pose(sim_args))])
    obs, _ = env.reset(options={"poses": pose})
    trace = SimTrace()
    _, metrics = sim._run_reactive_sim(env, obs, planner, r_mode, waypoints, trace=trace)
    env.close()
    return trace, metrics


def _extract_track_metrics(metrics: dict) -> dict[str, float | None]:
    return {
        key: float(metrics[key]) if metrics.get(key) is not None else None
        for key in TRACK_METRIC_KEYS
    }


def _run_one_map(
    runs: list[_ModelRun],
    map_spec: _MapSpec,
    run_dir: Path,
    run_id_slug: str,
    save_trace: bool = True,
    save_plot: bool = False,
) -> dict | None:
    from f110_planning.utils.waypoint_utils import load_waypoints

    try:
        waypoints = load_waypoints(map_spec.waypoints)
    except FileNotFoundError:
        print(f"[compare] missing waypoints: {map_spec.waypoints}; skipping")
        return None
    if waypoints.size == 0:
        print(f"[compare] empty waypoints: {map_spec.waypoints}; skipping")
        return None

    traces: list[tuple[_ModelRun, Any]] = []
    run_summaries: list[dict] = []
    for run in runs:
        trace, metrics = _simulate_run(run, waypoints, map_spec)
        traces.append((run, trace))
        track_m = _extract_track_metrics(metrics)
        rmse = track_m["crosstrack_rmse_m"]
        print(
            f"{map_spec.name}: {run.label}  "
            f"RMSE={'n/a' if rmse is None else f'{rmse:.4f} m'}  "
            f"steps={len(getattr(trace, 'positions', []))}"
        )
        run_summaries.append({
            "label": run.label,
            "left": run.left,
            "track": run.track,
            "heading": run.heading,
            "rmse": rmse,
            **track_m,
        })

    map_slug = _slugify(map_spec.name.replace("/", "_"))
    image_name: str | None = None

    if save_plot:
        import matplotlib.pyplot as plt
        img, extent = _load_map_background(map_spec.map_base, map_spec.map_ext)
        fig, ax = plt.subplots(figsize=(14, 14), dpi=320)
        ax.imshow(img, cmap="gray_r", extent=extent, origin="lower", alpha=0.35, zorder=0)
        if waypoints.size > 0:
            ax.plot(waypoints[:, 0], waypoints[:, 1], color="#444",
                    linewidth=0.5, linestyle="--", label="reference", zorder=1)
        for run, trace in traces:
            pts = np.asarray(getattr(trace, "positions", []))
            if pts.size == 0:
                continue
            ax.plot(pts[:, 0], pts[:, 1], linewidth=0.25, label=run.label, zorder=3)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{map_spec.name} ({', '.join(r.label for r in runs)})", fontsize=16)
        if traces:
            ax.legend(loc="upper right")
        ax.axis("off")
        image_name = f"{map_slug}_{run_id_slug}.png"
        fig.savefig(run_dir / image_name, dpi=600, bbox_inches="tight")
        print(f"[compare] saved plot: {image_name}")
        plt.close(fig)

    trace_file: str | None = None
    if save_trace:
        payload: dict[str, np.ndarray] = {}
        for run, trace in traces:
            pts = getattr(trace, "positions", [])
            if pts:
                payload[f"{run.label}_positions"] = np.asarray(pts, dtype=np.float32)
        if payload:
            npz_path = run_dir / f"{map_slug}_{run_id_slug}.npz"
            np.savez(npz_path, **payload)
            trace_file = npz_path.name

    return {
        "map": map_spec.name,
        "waypoints": map_spec.waypoints,
        "runs": run_summaries,
        "image": image_name,
        "trace_file": trace_file,
    }


def cmd_compare(args: argparse.Namespace) -> None:
    """Run multi-track simulation comparisons: NAS result vs baselines."""
    nas_dir = Path(args.nas_dir).expanduser().resolve()
    if not nas_dir.exists():
        raise FileNotFoundError(f"--nas-dir not found: {nas_dir}")

    pts = _find_nas_model_pts(nas_dir)
    if pts is None:
        raise FileNotFoundError(
            f"Could not find left_wall_dist / track_width / heading_error .pt files in {nas_dir}\n"
            "Expected pattern: *left_wall_dist*.pt, *track_width*.pt, *heading_error*.pt"
        )

    nas_label = _label_for_nas_dir(nas_dir)
    runs = (
        [_ModelRun(label=lbl, left=l, track=t, heading=h) for lbl, l, t, h in BASELINE_RUNS]
        + [_ModelRun(label=nas_label, left=pts["left_wall_dist"],
                     track=pts["track_width"], heading=pts["heading_error"])]
    )

    maps_root = Path(args.maps_root) if args.maps_root else REPO_ROOT / "data/maps"
    all_specs = _discover_map_specs(maps_root)
    map_specs = [
        s for s in all_specs
        if _normalize_map_name(s.map_base).lower() in SELECTED_TRACKS
    ]
    if not map_specs:
        raise RuntimeError(
            f"No selected tracks found under {maps_root}.\n"
            f"Looking for: {', '.join(sorted(SELECTED_TRACKS))}"
        )
    print(f"[compare] {len(map_specs)} tracks × {len(runs)} models")

    # Output directory
    out_base = Path(args.output_dir).expanduser().resolve() if args.output_dir else DEFAULT_COMPARE_OUT
    run_id = args.run_id or nas_label
    run_dir = out_base / run_id
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = out_base / f"{run_id}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id_slug = _slugify(run_id)

    records: list[dict] = []
    for map_spec in map_specs:
        rec = _run_one_map(
            runs, map_spec, run_dir, run_id_slug,
            save_trace=True,
            save_plot=args.save_plot,
        )
        if rec:
            records.append(rec)

    if records:
        metrics_path = run_dir / "metrics.jsonl"
        with metrics_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                json.dump(rec, fh)
                fh.write("\n")
        print(f"[compare] metrics → {metrics_path}")

        for run in runs:
            rmses = [
                r["rmse"] for rec in records
                for r in rec["runs"]
                if r["label"] == run.label and r["rmse"] is not None
            ]
            if rmses:
                print(f"[compare] {run.label:20s} avg RMSE = {sum(rmses)/len(rmses):.4f}  ({len(rmses)} tracks)")
    else:
        print("[compare] no records generated.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Safety-NAS pipeline: search / retrain / compare",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # search
    s = sub.add_parser("search", help="Run Optuna NAS search")
    s.add_argument(
        "--track", type=str, default=None,
        help="Single evaluation track (e.g. SEPANG). Default: all TRAIN_EVAL_TRACKS.",
    )
    s.add_argument("--n-trials", type=int, default=120, help="Number of Optuna trials.")

    # retrain
    r = sub.add_parser("retrain", help="Retrain the best NAS trial to full epochs")
    r.add_argument(
        "--trials-file", type=str, default=None,
        help="Path to nas_trials_*.jsonl. Default: newest under dnn-output/.",
    )
    r.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to stage retrained .pt files. Default: test-best-runs-tp0/<hash>/.",
    )
    r.add_argument(
        "--profile", type=int, default=0, choices=list(TRAINING_PROFILES),
        help="Training profile: 0=150ep, 1=250ep, 2=450ep, 3=700ep. Default: 0.",
    )
    r.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET,
        help=f"Dataset path. Default: {DEFAULT_DATASET}.",
    )

    # compare
    c = sub.add_parser("compare", help="Run multi-track simulation comparison")
    c.add_argument(
        "--nas-dir", type=str, required=True,
        help="Directory containing the staged NAS .pt files.",
    )
    c.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to write metrics.jsonl. Default: safety-nas/output-dir/.",
    )
    c.add_argument(
        "--run-id", type=str, default=None,
        help="Subdirectory name for this run. Default: nas_dir name.",
    )
    c.add_argument(
        "--maps-root", type=str, default=None,
        help="Path to maps root. Default: NAS-Sensitivity/data/maps.",
    )
    c.add_argument(
        "--save-plot", action="store_true",
        help="Save a trajectory comparison PNG per track (requires matplotlib + PIL).",
    )

    return p


def main() -> None:
    args = _build_parser().parse_args()
    {"search": cmd_search, "retrain": cmd_retrain, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    main()
