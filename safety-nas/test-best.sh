#!/bin/bash

set -e

cd ~/NAS-Sensitivity || exit 1

export PYTHONPATH="$HOME/NAS-Sensitivity:${PYTHONPATH:-}"

# Load modules only if the system supports them
if command -v module >/dev/null 2>&1; then
    module purge
    module load python/3.12.4
    module load cuda/12.4
fi

# Activate virtual environment
source .venv/bin/activate

echo "===== Environment ====="
echo "Host: $(hostname)"
echo "Python: $(which python)"
python --version

echo
echo "===== GPU ====="
nvidia-smi || true

echo
echo "===== Running test-best.py ====="

python safety-nas/test-best.py "$@"