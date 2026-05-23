#!/usr/bin/env python3
"""entry point for supervised lidar architecture search."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import optuna

from cnn import objective

TARGETS = ("left_wall_dist", "track_width", "heading_error")
N_TRIALS = 120
TRAIN_PATH = "accuracy-nas/datasets/train.npz"
VALIDATION_PATH = "accuracy-nas/datasets/validation.npz"


def _run_search(target: str) -> None:
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(
            trial,
            target_col=target,
            train_path=TRAIN_PATH,
            validation_path=VALIDATION_PATH,
        ),
        n_trials=N_TRIALS,
    )


def main() -> None:
    with ThreadPoolExecutor(max_workers=len(TARGETS)) as executor:
        futures = [executor.submit(_run_search, target) for target in TARGETS]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
