import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
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
DEFAULT_TARGET_COLS = ("left_wall_dist", "track_width", "heading_error")
LATEST_MODEL_PATHS = {target: None for target in DEFAULT_TARGET_COLS}


class DynamicCNN:
    """
    Generates a dynamic architecture specification for f110's 1-D LiDAR CNNs.
    """

    def __init__(self, trial: optuna.trial.Trial, prefix: str = "") -> None:
        def _key(name: str) -> str:
            return f"{prefix}_{name}" if prefix else name

        self.num_layers = trial.suggest_int(_key("num_layers"), 3, 5)
        self.activation = trial.suggest_categorical(_key("activation"), ["elu", "relu"])
        self.conv_layers: list[dict[str, int]] = []

        in_channels = 1  # ensures 1×1080 LiDAR input is accepted
        feature_length = 1080
        curr_channels = in_channels
        kernel_size = 3  # intentionally set
        stride = 1  # intentionally set
        padding = 0  # intentionally set

        for idx in range(self.num_layers):
            out_key = _key(f"out_channels_l{idx}")
            pool_key = _key(f"pool_size_l{idx}")
            out_channels = trial.suggest_int(out_key, 32, 256, log=True)
            pool_size = trial.suggest_categorical(pool_key, [2, 4, 8])
            # clamp pool_size to 1 (skip) when feature length is less than the selected pool size
            if feature_length < pool_size:
                pool_size = 1

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
            # Guards against pooling sizes that are large than the feature length/kernal size (3)
            if feature_length < 3:
                self.num_layers = len(self.conv_layers)
                break
            curr_channels = out_channels

        flattened = max(1, feature_length * max(curr_channels, 1))
        min_hidden = max(32, flattened // 64)
        max_hidden = min(128, max(32, flattened // 4))
        if min_hidden > max_hidden:
            min_hidden = max_hidden

        if min_hidden == max_hidden:
            fc_hidden = min_hidden
        else:
            fc_hidden = trial.suggest_int(_key("fc_hidden"), min_hidden, max_hidden, log=True)

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


def _build_training_config(model_block: dict[str, any], target_col: str, dataset_pth: str = None) -> dict[str, any]:

    if dataset_pth is None:
        raise ValueError("dataset_path in _build_training_config is not specified.")

    cfg = {
        "data": {
            "train_path": dataset_pth,
            "target_col": target_col,
            "batch_size": 128,
            "num_workers": 8,
            "val_split": 0.1,
            "pin_memory": True,
            "prefetch_factor": 2,
        },
        "training": {
            "max_epochs": 21,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "early_stopping_patience": 11,
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
        "model": deepcopy(model_block),
    }
    return cfg


def objective(
    trial: optuna.trial.Trial,
    _unused_loader=None,
    n_epoch: int = 10,
    seed: int = 41,
    target_cols: tuple[str, ...] = DEFAULT_TARGET_COLS,
    dataset_pth: str = "nas/datasets/combined_train.npz"
) -> float:
    """
    Run one Optuna trial that samples a single arch8 config, trains the left/track/heading
    nets in parallel, waits for all three checkpoints (.pt) to land, then evaluates the trio
    once via test_cnn_arch and returns that combined RMSE.
    """
    del n_epoch, seed, _unused_loader  # unused

    model_blocks = {}
    for target in target_cols:
        architecture = DynamicCNN(trial, prefix=target)
        block = architecture.to_model_block()
        block["arch_id"] = 8
        model_blocks[target] = block

    cfgs = [
        _build_training_config(model_blocks[target], target, dataset_pth)
        for target in target_cols
    ]

    trained_runs: list[tuple[dict[str, any], Path]] = []
    # TODO add specific partition of resources once the specific resource constraints (running on a gpu/cpu) are known
    with ThreadPoolExecutor(max_workers=len(cfgs)) as executor:
        futures = []
        for cfg in cfgs:
            futures.append(executor.submit(_run_training, cfg))
        for cfg, future in zip(cfgs, futures):
            target = cfg["data"]["target_col"]
            try:
                model_path = future.result()
            except subprocess.CalledProcessError:
                return float("inf")
            trained_runs.append((cfg, model_path))

    for cfg, model_path in trained_runs:
        target = cfg["data"]["target_col"]
        LATEST_MODEL_PATHS[target] = model_path

    try:
        rmse = test_cnn_arch(
            left_wall_dist_filepath=str(LATEST_MODEL_PATHS["left_wall_dist"]),
            track_width_filepath=str(LATEST_MODEL_PATHS["track_width"]),
            heading_error_filepath=str(LATEST_MODEL_PATHS["heading_error"]),
        )
    except TypeError as exc:
        raise RuntimeError("Missing trained checkpoints before running test_cnn_arch") from exc

    _log_trial_result(
        trial=trial,
        trained_runs=trained_runs,
        rmse=rmse,
    )
    return rmse


def _log_trial_result(
    trial: optuna.trial.Trial,
    trained_runs: list[tuple[dict[str, any], Path]],
    rmse: float,
) -> None:
    """
    Append a structured summary for each NAS trial to ``output/nas_trials.jsonl``.
    """
    target_summaries: list[dict[str, any]] = []
    for cfg, model_path in trained_runs:
        try:
            model = get_architecture(cfg["model"]["arch_id"], cfg["model"])
            architecture = repr(model)
            layers = _summarize_layers(model)
        except Exception as exc:  # pragma: no cover - logging best effort
            architecture = f"<error rendering architecture: {exc}>"
            layers = []

        target_summaries.append(
            {
                "target_col": cfg["data"]["target_col"],
                "arch_id": cfg["model"]["arch_id"],
                "conv_layers": cfg["model"]["dynamic"]["conv_layers"],
                "fc_layers": cfg["model"]["dynamic"]["fc_layers"],
                "activation": cfg["model"]["dynamic"]["activation"],
                "model_path": str(model_path),
                "architecture": architecture,
                "layers": layers,
            }
        )

    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "trial_number": trial.number,
        "rmse": rmse,
        "params": trial.params,
        "targets": target_summaries,
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
