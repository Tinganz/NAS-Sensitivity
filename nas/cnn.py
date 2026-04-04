import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import optuna
import yaml
from torch import nn

from f110_planning.utils.nn_models import get_architecture
from testing import test_cnn_arch

# sending output to ./dnn-output
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "dnn-output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SESSION_ID = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
LOG_PATH = OUTPUT_DIR / f"nas_trials_{SESSION_ID}.jsonl"


class DynamicCNN:
    """
    Generates a dynamic architecture specification for f110's 1-D LiDAR CNNs.
    """

    def __init__(self, trial: optuna.trial.Trial) -> None:
        self.num_layers = trial.suggest_int("num_layers", 1, 5)
        self.activation = trial.suggest_categorical("activation", ["elu", "relu"])
        self.conv_layers: list[dict[str, int]] = []

        in_channels = 1  # ensures 1×1080 LiDAR input is accepted
        feature_length = 1080
        curr_channels = in_channels
        kernel_size = 3  # intentionally set
        stride = 1  # intentionally set
        padding = 0  # intentionally set

        for idx in range(self.num_layers):
            out_channels = trial.suggest_int(f"out_channels_l{idx}", 8, 128, log=True)
            pool_size = trial.suggest_categorical(f"pool_size_l{idx}", [2, 4, 8])

            self.conv_layers.append(
                {
                    "out_channels": out_channels,
                    "kernel_size": kernel_size,
                    "stride": stride,
                    "padding": padding,
                    "pool_size": pool_size,
                }
            )
            feature_length = max(
                1,
                (feature_length + 2 * padding - kernel_size) // stride + 1,
            )
            if pool_size and pool_size > 1:
                feature_length = max(1, feature_length // pool_size)
            curr_channels = out_channels

        flattened = max(1, feature_length * max(curr_channels, 1))
        min_hidden = max(32, flattened // 64)
        max_hidden = min(128, max(32, flattened // 4))
        if min_hidden > max_hidden:
            min_hidden = max_hidden

        if min_hidden == max_hidden:
            fc_hidden = min_hidden
        else:
            fc_hidden = trial.suggest_int("fc_hidden", min_hidden, max_hidden, log=True)

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
            "max_epochs": 11,
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
    _log_trial_result(
        trial=trial,
        cfg=cfg,
        rmse=rmse,
        model_path=left_model_path,
    )
    return rmse


def _log_trial_result(
    trial: optuna.trial.Trial,
    cfg: dict[str, any],
    rmse: float,
    model_path: Path,
) -> None:
    """
    Append a structured summary for each NAS trial to ``output/nas_trials.jsonl``.
    """
    try:
        model = get_architecture(cfg["model"]["arch_id"], cfg["model"])
        architecture = repr(model)
        layers = _summarize_layers(model)
    except Exception as exc:  # pragma: no cover - logging best effort
        architecture = f"<error rendering architecture: {exc}>"
        layers = []

    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "trial_number": trial.number,
        "rmse": rmse,
        "params": trial.params,
        "target_col": cfg["data"]["target_col"],
        "arch_id": cfg["model"]["arch_id"],
        "conv_layers": cfg["model"]["dynamic"]["conv_layers"],
        "fc_layers": cfg["model"]["dynamic"]["fc_layers"],
        "activation": cfg["model"]["dynamic"]["activation"],
        "model_path": str(model_path),
        "architecture": architecture,
        "layers": layers,
    }

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry))
        f.write("\n")


def _summarize_layers(model: nn.Module) -> list[dict[str, any]]:
    """
    Convert a Sequential model into a structured list of layer dictionaries for logging.
    """
    layer_summaries: list[dict[str, any]] = []
    for layer in model:
        info: dict[str, any] = {"type": layer.__class__.__name__}
        if isinstance(layer, nn.Conv1d):
            info.update(
                {
                    "in_channels": layer.in_channels,
                    "out_channels": layer.out_channels,
                    "kernel_size": layer.kernel_size[0],
                    "stride": layer.stride[0],
                    "padding": layer.padding[0],
                }
            )
        elif isinstance(layer, nn.MaxPool1d):
            info.update(
                {
                    "kernel_size": layer.kernel_size,
                    "stride": layer.stride,
                    "padding": layer.padding,
                }
            )
        elif isinstance(layer, nn.Linear):
            info.update(
                {
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    "bias": layer.bias is not None,
                }
            )
        elif isinstance(layer, nn.Flatten):
            info.update(
                {
                    "start_dim": layer.start_dim,
                    "end_dim": layer.end_dim,
                }
            )
        elif isinstance(layer, (nn.ELU, nn.ReLU)):
            if isinstance(layer, nn.ELU):
                info["alpha"] = layer.alpha
        else:
            info["repr"] = repr(layer)
        layer_summaries.append(info)
    return layer_summaries
