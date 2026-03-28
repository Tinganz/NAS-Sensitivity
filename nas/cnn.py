import subprocess
import sys
import tempfile
from pathlib import Path

import optuna
import yaml

from testing import test_cnn_arch


class DynamicCNN:
    """
    Generates a dynamic architecture specification for f110's 1-D LiDAR CNNs.
    """

    def __init__(self, trial: optuna.trial.Trial) -> None:
        self.num_layers = trial.suggest_int("num_layers", 1, 5)
        self.activation = trial.suggest_categorical("activation", ["elu", "relu"])
        self.conv_layers: list[dict[str, int]] = []

        in_channels = 1 # store out_channels and assign here
        for idx in range(self.num_layers):
            out_channels = trial.suggest_int(f"out_channels_l{idx}", 8, 128, log=True)
            pool_size = trial.suggest_categorical(f"pool_size_l{idx}", [0, 2, 4])
            
            self.conv_layers.append(
                {
                    "out_channels": out_channels,
                    "kernel_size": 3, # intentionally set
                    "stride": 1, # intentionally set
                    "padding": 0, # intentionally set
                    "pool_size": pool_size,
                }
            )
            in_channels = out_channels
            
        fc_hidden = trial.suggest_int("fc_hidden", 32, 256, log=True)
        self.model_block = {
            "arch_id": 8,
            "dynamic": {
                "in_channels": 1,
                "input_length": 1080,
                "activation": self.activation,
                "conv_layers": self.conv_layers,
                "fc_layers": [fc_hidden],
            },
        }

    def to_model_block(self) -> dict[str, any]:
        return self.model_block


def _run_training(config: dict[str, any]) -> Path:
    """
    Writes a temporary YAML config, runs the training script, and returns
    the trained model path.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(config, tmp)
        cfg_path = Path(tmp.name)

    try:
        subprocess.run(
            [
                sys.executable,
                "packages/f110_scripts/src/f110_scripts/train/train_nn.py",
                "--config",
                str(cfg_path),
            ],
            check=True,
        )
    finally:
        cfg_path.unlink(missing_ok=True)

    model_name = f"{config['data']['target_col']}_arch{config['model']['arch_id']}"
    return Path("data/models", f"{model_name}.pt")


def objective(
    trial: optuna.trial.Trial,
    _unused_loader=None,
    n_epoch: int = 10,
    seed: int = 41,
    target_col: str = "left_wall_dist"
) -> float:
    del n_epoch, seed, _unused_loader  # unused

    architecture = DynamicCNN(trial)
    cfg = {
        "data": {
            "train_path": "data/datasets/combined_all.npz",
            "target_col": target_col,
            "batch_size": 128,
            "num_workers": 8,
            "val_split": 0.1,
            "pin_memory": True,
            "prefetch_factor": 2,
        },
        "training": {
            "max_epochs": 2,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "early_stopping_patience": 8,
            "lr_patience": 5,
            "lr_scheduler_factor": 0.5,
            "optimizer": "adam",
            "scheduler": "reduce_on_plateau",
            "auto_lr_find": False,
            "resume": False,
            "precision": "32",
            "gradient_clip_val": 1.0,
            "profiler": None,
        },
        "model": architecture.to_model_block(),
    }

    cfg["model"]["arch_id"] = 8  # ensure we always use the dynamic factory

    try:
        left_model_path = _run_training(cfg)
    except subprocess.CalledProcessError:
        return float("inf")

    rmse = test_cnn_arch(left_wall_dist_filepath=str(left_model_path))
    return rmse
