from pathlib import Path
import subprocess
import sys
import os

os.chdir("/home/tingan/NAS-Sensitivity/safety-nas/")

REPO_ROOT = Path("/home/tingan/NAS-Sensitivity").resolve()

TRAIN_SCRIPT = (
    REPO_ROOT
    / "packages/f110_scripts/src/f110_scripts/train/train_nn.py"
).resolve()

def train_model(config_path):
    config_path = Path(config_path).resolve()
    print(os.getcwd())

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(config_path),
    ]

    print(f"[train] running {config_path}", flush=True)

    subprocess.run(
        cmd,
        cwd=REPO_ROOT,
    )

config_dir = (
    REPO_ROOT
    / "safety-nas/new_baseline/"
)
yaml_files = sorted(config_dir.glob("*.yaml"))

for yaml_file in yaml_files:
    train_model(yaml_file)