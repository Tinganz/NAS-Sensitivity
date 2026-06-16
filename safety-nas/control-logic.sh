#!/bin/bash

set -e

cd ~/NAS-Sensitivity || exit 1

export PYTHONPATH="$HOME/NAS-Sensitivity:${PYTHONPATH:-}"

source .venv/bin/activate

echo "===== Environment ====="
echo "Host: $(hostname)"
echo "Python: $(which python)"
python --version

echo
echo "===== GPU ====="
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

mkdir -p logs

python safety-nas/control-logic.py "$@" \
    > logs/control_logic.out \
    2> logs/control_logic.err