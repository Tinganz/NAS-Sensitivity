#!/bin/bash
#SBATCH --job-name=train_nn_debug
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
cd ~/NAS-Sensitivity || exit 1

export PYTHONPATH="$HOME/NAS-Sensitivity:$PYTHONPATH"

if command -v module &>/dev/null; then
    module purge
    module load python/3.12.4
    module load cuda/12.4
fi

# activate venv AFTER module load
source .venv/bin/activate

PYTHON_BIN="/nas/longleaf/home/tingan/NAS-Sensitivity/.venv/bin/python"

CONFIG="/nas/longleaf/home/tingan/NAS-Sensitivity/safety-nas/test-best-runs-tp0/aee7a1/left_wall_dist_arch8_trial6.yaml"

echo "Running training..."
echo "Python: $PYTHON_BIN"
echo "Config: $CONFIG"

$PYTHON_BIN \
  /nas/longleaf/home/tingan/NAS-Sensitivity/packages/f110_scripts/src/f110_scripts/train/train_nn.py \
  --config "$CONFIG"

echo "Job finished at $(date)"