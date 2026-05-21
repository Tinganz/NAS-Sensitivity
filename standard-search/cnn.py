"""per-target supervised Optuna search for lidar CNNs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
from f110_planning.utils.nn_models import get_architecture

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nas.cnn import DynamicCNN, _build_training_config, _run_training, _summarize_layers

from evaluation import evaluate_model

TARGETS = ("left_wall_dist", "track_width", "heading_error")
OUTPUT_DIR = BASE_DIR / "dnn-output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SESSION_ID = os.getenv("F1_SESSION_ID") or (
    f"{datetime.utcnow():%Y%m%dT%H%M%S}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
)


def objective(
    trial: optuna.trial.Trial,
    target_col: str,
    train_path: str = "standard-search/datasets/train.npz",
    validation_path: str = "standard-search/datasets/validation.npz",
) -> float:
    """Train one candidate and score its validation RMSE."""
    if target_col not in TARGETS:
        raise ValueError(f"unknown target: {target_col}")

    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    model_block = DynamicCNN(trial).to_model_block()
    trial_root = OUTPUT_DIR / "trial_artifacts" / f"{SESSION_ID}_{target_col}_trial{trial.number:05d}"
    cfg = _build_training_config(
        model_block,
        target_col,
        train_path,
        artifact_root=trial_root,
    )
    cfg["training"]["optimizer"] = optimizer

    try:
        model_path = _run_training(cfg)
    except subprocess.CalledProcessError:
        return float("inf")

    metrics = evaluate_model(model_path, validation_path, target_col)
    _log_trial(trial, cfg, model_path, metrics, validation_path)
    return float(metrics["rmse"])


def _log_trial(
    trial: optuna.trial.Trial,
    cfg: dict[str, Any],
    model_path: Path,
    metrics: dict[str, float | int],
    validation_path: str,
) -> None:
    """Write one line shaped for later retraining."""
    model = get_architecture(cfg["model"]["arch_id"], cfg["model"])
    target = cfg["data"]["target_col"]
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "trial_number": trial.number,
        "target_col": target,
        "objective": {
            "name": "validation_rmse",
            "value": metrics["rmse"],
            "dataset": validation_path,
        },
        "validation": metrics,
        "params": trial.params,
        "targets": [
            {
                "target_col": target,
                "arch_id": cfg["model"]["arch_id"],
                "conv_layers": cfg["model"]["dynamic"]["conv_layers"],
                "fc_layers": cfg["model"]["dynamic"]["fc_layers"],
                "activation": cfg["model"]["dynamic"]["activation"],
                "model_path": str(model_path),
                "architecture": repr(model),
                "layers": _summarize_layers(model),
            }
        ],
    }
    path = OUTPUT_DIR / f"standard_trials_{target}_{SESSION_ID}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry))
        handle.write("\n")
