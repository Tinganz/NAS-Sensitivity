from __future__ import annotations

import shutil
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

# Override training.early_stopping_patience (set None to keep NAS defaults).
EARLY_STOPPING_PATIENCE: int | None = 350

# Choose "train" to retrain + evaluate or "test" to evaluate existing .pt files.
MODE = "test"  # valid values: "train", "test"

LEGACY_MODEL_DIRS = (
    (REPO_ROOT / "nas/models").resolve(),
    (REPO_ROOT / "data/models").resolve(),
)


def _copy_checkpoints_to_config_dir(
    config_paths: list[Path], checkpoints: list[Path]
) -> list[Path]:
    """Copy trained checkpoints so they live next to their YAML configs."""
    if len(config_paths) != len(checkpoints):
        raise RuntimeError(
            f"Expected {len(config_paths)} checkpoints, received {len(checkpoints)}."
        )

    staged_paths: list[Path] = []
    for config_path, checkpoint in zip(config_paths, checkpoints, strict=False):
        source = checkpoint if checkpoint.is_absolute() else (REPO_ROOT / checkpoint)
        destination = config_path.with_suffix(".pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        src_abs = source.resolve()
        dest_abs = destination.resolve()
        if src_abs == dest_abs:
            staged_paths.append(destination)
            continue
        shutil.copy2(src_abs, dest_abs)
        staged_paths.append(destination)
    return staged_paths


def _stage_checkpoint_for_config(config_path: Path, checkpoint: Path) -> Path:
    """Ensure a checkpoint sits next to its config and return that path."""
    source = checkpoint if checkpoint.is_absolute() else (REPO_ROOT / checkpoint)
    destination = config_path.with_suffix(".pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    src_abs = source.resolve()
    dest_abs = destination.resolve()
    if src_abs != dest_abs:
        shutil.copy2(src_abs, dest_abs)
    return destination


def _target_summary_for_config(best_trial: dict, config_path: Path) -> dict | None:
    """Locate the NAS summary entry that matches a config path."""
    target_name = _target_from_path(config_path)
    for entry in best_trial.get("targets", []):
        if entry.get("target_col") == target_name:
            return entry
    return None


def _find_checkpoint_for_config(
    config_path: Path, target_summary: dict | None
) -> Path | None:
    """Return the most relevant checkpoint path for a config, if it exists."""
    staged_checkpoint = config_path.with_suffix(".pt")
    if staged_checkpoint.exists():
        return staged_checkpoint
    if not target_summary:
        return None
    model_name = f"{target_summary['target_col']}_arch{target_summary['arch_id']}.pt"
    for legacy_dir in LEGACY_MODEL_DIRS:
        candidate = legacy_dir / model_name
        if candidate.exists():
            return candidate
    return None


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
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )

    mode = MODE.lower()
    if mode == "train":
        trained_models = train_from_configs(config_paths)
        trained_models = _copy_checkpoints_to_config_dir(config_paths, trained_models)
    elif mode == "test":
        trained_models = []
        missing_files: list[str] = []
        for config_path in config_paths:
            summary = _target_summary_for_config(best_trial, config_path)
            checkpoint = _find_checkpoint_for_config(config_path, summary)
            if checkpoint:
                staged_checkpoint = _stage_checkpoint_for_config(
                    config_path, checkpoint
                )
                trained_models.append(staged_checkpoint)
            else:
                staged_path = config_path.with_suffix(".pt")
                fallback_msg = ""
                if summary:
                    fallback_msg = (
                        f" (legacy name: {summary['target_col']}_arch{summary['arch_id']}.pt)"
                    )
                missing_files.append(f"{staged_path}{fallback_msg}")
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
