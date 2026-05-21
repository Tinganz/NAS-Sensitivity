#!/bin/bash
#SBATCH --job-name=f1_std_best
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zayah@unc.edu
cd "$SLURM_SUBMIT_DIR" || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
if command -v module &>/dev/null; then
    module purge
    module load python/3.12.4
    module load cuda/12.4
fi
source .venv/bin/activate
python standard-search/test-best.py
