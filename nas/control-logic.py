import subprocess
import sys
from pathlib import Path

import optuna

from cnn import objective

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=120)

    # subprocess.run([sys.executable, str(BASE_DIR / "test-best.py")], check=True)
    # subprocess.run([sys.executable, str(BASE_DIR / "compare-track.py")], check=True)


if __name__ == "__main__":
    main()
