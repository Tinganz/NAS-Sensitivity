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

from nas.training import export_trial_configs, train_from_configs

from evaluation import evaluate_model

TARGET_FILES = {
    "left_wall_dist": "standard-search/dnn-output/standard_trials_left_wall_dist_<run>.jsonl",
    "track_width": "standard-search/dnn-output/standard_trials_track_width_<run>.jsonl",
    "heading_error": "standard-search/dnn-output/standard_trials_heading_error_<run>.jsonl",
}
TRAIN_VALIDATION_PATH = "standard-search/datasets/train_validation.npz"
TEST_PATH = "standard-search/datasets/test.npz"
OUTPUT_DIR = "standard-search/dnn-output/test-best"
MAX_EPOCHS = 150
LR = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 15
LR_PATIENCE = 10
OPTIMIZER = "adam"


def _load_best(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    if not entries:
        raise RuntimeError(f"no trials found in {path}")
    return min(entries, key=lambda entry: float(entry["objective"]["value"]))


def _stage_model(config_path: Path, model_path: Path) -> Path:
    staged = config_path.with_suffix(".pt")
    if staged.resolve() != model_path.resolve():
        shutil.copy2(model_path, staged)
    return staged


def main() -> None:
    output_dir = Path(OUTPUT_DIR).expanduser().resolve()
    configs: list[Path] = []
    for target, trials_file in TARGET_FILES.items():
        best = _load_best(trials_file)
        target_configs = export_trial_configs(
            best,
            output_dir,
            dataset_path=TRAIN_VALIDATION_PATH,
            max_epochs=MAX_EPOCHS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            lr_patience=LR_PATIENCE,
            optimizer=OPTIMIZER,
        )
        configs.extend(target_configs)
        print(
            f"[best] {target} trial #{best['trial_number']} "
            f"validation_rmse={best['objective']['value']:.6f}"
        )

    models = train_from_configs(configs)
    report: dict[str, dict[str, float | int]] = {}
    for config_path, model_path in zip(configs, models, strict=True):
        target = config_path.stem.split("_arch", 1)[0]
        staged = _stage_model(config_path, model_path)
        report[target] = evaluate_model(staged, TEST_PATH, target)
        print(f"[model] {target} -> {staged}")
        print(f"[test]  {target} rmse={report[target]['rmse']:.6f}")

    report_path = output_dir / "test_metrics.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"[report] {report_path}")


if __name__ == "__main__":
    main()
