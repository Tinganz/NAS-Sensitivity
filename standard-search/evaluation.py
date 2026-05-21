"""supervised metrics for lidar predictor checkpoints."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from f110_planning.utils.nn_models import load_torchscript
from f110_scripts.train.train_nn import LidarDataset


def evaluate_model(
    model_path: str | Path,
    dataset_path: str | Path,
    target_col: str,
    batch_size: int = 512,
) -> dict[str, float | int]:
    """Evaluate one saved lidar model on one split."""
    dataset = LidarDataset(str(dataset_path), target_col)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model, _ = load_torchscript(model_path, map_location="cpu")
    model.eval()

    squared_error = 0.0
    absolute_error = 0.0
    sample_count = 0
    with torch.no_grad():
        for scans, targets in loader:
            predictions = model(scans)
            errors = predictions - targets
            squared_error += float(torch.sum(errors * errors))
            absolute_error += float(torch.sum(torch.abs(errors)))
            sample_count += int(targets.numel())

    if sample_count == 0:
        raise ValueError(f"{dataset_path} has no samples for {target_col}.")

    mse = squared_error / sample_count
    return {
        "samples": sample_count,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": absolute_error / sample_count,
    }
