"""Scriptable entry point for selecting + training the best NAS trial."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nas.testing import test_cnn_arch
from nas.training import orchestrate_best_trial, train_from_configs

# Leave as None to automatically use the newest JSONL under nas/dnn-output/.
TRIALS_FILE: Path | None = None
DATASET_PATH = "nas/datasets/combined_train.npz"

# Override where configs land. Defaults to nas/dnn-output/best_configs/<stem>.
CONFIG_OUTPUT_DIR: Path | None = None

# Set an integer to clamp training.max_epochs in every exported YAML.
MAX_EPOCHS: int | None = 400

# Choose "train" to retrain + evaluate or "test" to evaluate existing .pt files.
MODE = "train"  # valid values: "train", "test"

MODEL_DIR = (REPO_ROOT / "data/models").resolve()


def _target_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_arch") or "_arch" not in stem:
        return stem
    return stem.rsplit("_arch", 1)[0]


def evaluate_models(model_paths: list[Path]) -> float | None:
    """Run the simulator with the provided checkpoints and report RMSE."""
    lookup = {_target_from_path(path): path for path in model_paths}
    required = ("left_wall_dist", "track_width", "heading_error")
    missing = [key for key in required if key not in lookup]
    if missing:
        print(f"[warn] missing checkpoints for: {', '.join(missing)}")
        return None
    rmse = test_cnn_arch(
        left_wall_dist_filepath=str(lookup["left_wall_dist"]),
        track_width_filepath=str(lookup["track_width"]),
        heading_error_filepath=str(lookup["heading_error"]),
    )
    print(f"[rmse] {rmse:.4f}")
    return rmse


def main() -> None:
    best_trial, config_paths = orchestrate_best_trial(
        trials_path=TRIALS_FILE,
        dataset_path=DATASET_PATH,
        output_dir=CONFIG_OUTPUT_DIR,
        max_epochs=MAX_EPOCHS,
    )

    mode = MODE.lower()
    if mode == "train":
        trained_models = train_from_configs(config_paths)
    elif mode == "test":
        trained_models = []
        missing_files: list[str] = []
        for target in best_trial["targets"]:
            ckpt = MODEL_DIR / f"{target['target_col']}_arch{target['arch_id']}.pt"
            if ckpt.exists():
                trained_models.append(ckpt)
            else:
                missing_files.append(str(ckpt))
        if missing_files:
            print("[warn] checkpoints not found:")
            for missing in missing_files:
                print(f"    {missing}")
    else:
        raise ValueError("MODE must be 'train' or 'test'")

    print(f"[best] trial #{best_trial['trial_number']} rmse={best_trial['rmse']:.4f}")
    for tgt in best_trial["targets"]:
        print(f"[arch] {tgt['target_col']} arch{tgt['arch_id']}")
    for cfg in config_paths:
        print(f"[config] {cfg}")
    for model_path in trained_models:
        print(f"[checkpoint] {model_path}")
        
    evaluate_models(trained_models)


if __name__ == "__main__":
    main()