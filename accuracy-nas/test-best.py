#!/usr/bin/env python3
"""retrain the best supervised model for each lidar target."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SAFETY_NAS_DIR = REPO_ROOT / "safety-nas"
if str(SAFETY_NAS_DIR) not in sys.path:
    sys.path.insert(0, str(SAFETY_NAS_DIR))

from cnn import TRAIN_EVAL_TRACKS
from testing import test_cnn_arch
from training import export_trial_configs, train_from_configs

TRAINING_PROFILES = {
    0: {
        "label": "arch1-2",
        "max_epochs": 150,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "early_stopping_patience": 15,
        "lr_patience": 10,
        "optimizer": "adam",
    },
    1: {
        "label": "arch3-4",
        "max_epochs": 250,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "early_stopping_patience": 20,
        "lr_patience": 10,
        "optimizer": "adam",
    },
    2: {
        "label": "arch5",
        "max_epochs": 450,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "early_stopping_patience": 60,
        "lr_patience": 20,
        "optimizer": "adam",
    },
    3: {
        "label": "arch6-7",
        "max_epochs": 700,
        "lr": 5e-5,
        "weight_decay": 1e-4,
        "early_stopping_patience": 60,
        "lr_patience": 20,
        "optimizer": "adamw",
    },
}

TARGET_FILES = {
    "left_wall_dist": "accuracy-nas/dnn-output/standard_trials_left_wall_dist_20260522T041610_2631784_262e4c.jsonl",
    "track_width": "accuracy-nas/dnn-output/standard_trials_track_width_20260522T041610_2631784_262e4c.jsonl",
    "heading_error": "accuracy-nas/dnn-output/standard_trials_heading_error_20260522T041610_2631784_262e4c.jsonl",
}
TRAINING_PROFILE = 0  # 0: arch1-2, 1: arch3-4, 2: arch5, 3: arch6-7
TRAIN_PATH = "safety-nas/datasets/combined_all.npz"
OUTPUT_DIR = "accuracy-nas/dnn-output/compare-map-150"
SKIP_EVAL = False


def _trials_file_id(target: str, trials_file: str | Path) -> str:
    stem = Path(trials_file).stem
    prefix = f"standard_trials_{target}_"
    if not stem.startswith(prefix):
        raise ValueError(f"Trials file {trials_file} must be named {prefix}<run>.jsonl.")
    return stem[len(prefix):].rsplit("_", 1)[-1]


def _composite_id(target_files: dict[str, str]) -> str:
    source_ids = {
        target: _trials_file_id(target, trials_file)
        for target, trials_file in target_files.items()
    }
    if len(set(source_ids.values())) == 1:
        return next(iter(source_ids.values()))
    return "_".join(
        (
            f"left-{source_ids['left_wall_dist']}",
            f"track-{source_ids['track_width']}",
            f"heading-{source_ids['heading_error']}",
        )
    )


def _load_best(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    if not entries:
        raise RuntimeError(f"no trials found in {path}")
    return min(entries, key=lambda entry: float(entry["objective"]["value"]))


def _selected_training_profile() -> dict[str, Any]:
    try:
        return TRAINING_PROFILES[TRAINING_PROFILE]
    except KeyError as exc:
        valid = ", ".join(str(key) for key in sorted(TRAINING_PROFILES))
        raise ValueError(
            f"Unknown TRAINING_PROFILE={TRAINING_PROFILE}. Valid profiles: {valid}."
        ) from exc


def _stage_model(config_path: Path, model_path: Path) -> Path:
    staged = config_path.with_suffix(".pt")
    if staged.resolve() != model_path.resolve():
        shutil.copy2(model_path, staged)
    return staged


def _target_name(path: Path) -> str:
    return path.stem.split("_arch", 1)[0]


def _evaluate_models(model_paths: list[Path]) -> float | None:
    lookup = {_target_name(path): path for path in model_paths}
    required = {"left_wall_dist", "track_width", "heading_error"}
    missing = sorted(required - set(lookup))
    if missing:
        print(f"[warn] missing checkpoints for: {', '.join(missing)}")
        return None

    track_configs = [(track.map_path, track.waypoints_path) for track in TRAIN_EVAL_TRACKS]
    avg_rmse, _, _, _ = test_cnn_arch(
        left_wall_dist_filepath=str(lookup["left_wall_dist"]),
        track_width_filepath=str(lookup["track_width"]),
        heading_error_filepath=str(lookup["heading_error"]),
        track_configs=track_configs,
    )
    print(f"[rmse] average across {len(track_configs)} tracks: {avg_rmse:.4f}")
    return avg_rmse


def main() -> None:
    composite_id = _composite_id(TARGET_FILES)
    output_dir = Path(OUTPUT_DIR).expanduser().resolve() / composite_id
    training_profile = _selected_training_profile()
    print(f"[run] composite_id={composite_id}")
    print(f"[run] output_dir={output_dir}")
    print(f"[run] training_profile={TRAINING_PROFILE} ({training_profile['label']})")
    configs: list[Path] = []
    for target, trials_file in TARGET_FILES.items():
        best = _load_best(trials_file)
        target_configs = export_trial_configs(
            best,
            output_dir,
            dataset_path=TRAIN_PATH,
            max_epochs=training_profile["max_epochs"],
            lr=training_profile["lr"],
            weight_decay=training_profile["weight_decay"],
            early_stopping_patience=training_profile["early_stopping_patience"],
            lr_patience=training_profile["lr_patience"],
            optimizer=training_profile["optimizer"],
        )
        configs.extend(target_configs)
        print(
            f"[best] {target} trial #{best['trial_number']} "
            f"validation_rmse={best['objective']['value']:.6f}"
        )

    models = train_from_configs(configs)
    staged_models: list[Path] = []
    for config_path, model_path in zip(configs, models, strict=True):
        staged = _stage_model(config_path, model_path)
        staged_models.append(staged)
        print(f"[model] {_target_name(staged)} -> {staged}")
    if not SKIP_EVAL:
        _evaluate_models(staged_models)


if __name__ == "__main__":
    main()
