#!/usr/bin/env python3
"""split a lidar dataset for the supervised architecture search."""

from __future__ import annotations

from pathlib import Path

import numpy as np

INPUT_PATH = "safety-nas/datasets/combined_all.npz"
OUTPUT_DIR = "accuracy-nas/datasets"
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
SEED = 41


def _write_split(path: Path, arrays: dict[str, np.ndarray], indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: values[indices] for key, values in arrays.items()})


def split_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 41,
) -> dict[str, Path]:
    """Write fixed row splits plus a train-validation retraining split."""
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("train and validation ratios must leave a non-empty test split.")

    with np.load(input_path) as data:
        arrays = {key: data[key] for key in data.files}
    if not arrays:
        raise ValueError(f"{input_path} has no arrays.")

    rows = {values.shape[0] for values in arrays.values()}
    if len(rows) != 1:
        raise ValueError("all dataset arrays must have the same first dimension.")
    total = rows.pop()

    indices = np.random.default_rng(seed).permutation(total)
    train_end = int(total * train_ratio)
    validation_end = train_end + int(total * validation_ratio)
    split_indices = {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
        "train_validation": indices[:validation_end],
    }

    output_root = Path(output_dir)
    paths = {name: output_root / f"{name}.npz" for name in split_indices}
    for name, split in split_indices.items():
        _write_split(paths[name], arrays, split)
    return paths


def main() -> None:
    paths = split_dataset(
        INPUT_PATH,
        OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
        seed=SEED,
    )
    for name, path in paths.items():
        with np.load(path) as data:
            count = data[data.files[0]].shape[0]
        print(f"[{name}] {count} rows -> {path}")


if __name__ == "__main__":
    main()
