import subprocess
import sys

def train_cnn_arch():
    # command instantiation
    TRAIN_CMDS = [
        [
            sys.executable,
            "packages/f110_scripts/src/f110_scripts/train/train_nn.py",
            "--config",
            "packages/f110_scripts/src/f110_scripts/train/config_heading_1.yaml",
        ],
        [
            sys.executable,
            "packages/f110_scripts/src/f110_scripts/train/train_nn.py",
            "--config",
            "packages/f110_scripts/src/f110_scripts/train/config_left_wall_1.yaml",
        ],
        [
            sys.executable,
            "packages/f110_scripts/src/f110_scripts/train/train_nn.py",
            "--config",
            "packages/f110_scripts/src/f110_scripts/train/config_track_width_1.yaml",
        ],
    ]
    
    # run, fail gracefully
    for cmd in TRAIN_CMDS:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            print(
                f"Training command {exc.cmd} failed with code {exc.returncode}",
                file=sys.stderr,
            )
            if exc.stdout:
                print(exc.stdout, file=sys.stderr)
            if exc.stderr:
                print(exc.stderr, file=sys.stderr)
            return float("inf")
        else:
            print(proc.stdout, proc.stderr)