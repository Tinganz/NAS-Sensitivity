#!/bin/bash
#SBATCH --job-name=f1_safety_nas
#SBATCH -p volta-gpu
#SBATCH --qos=hp_volta_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tzhu@unc.edu
cd ~/NAS-Sensitivity || exit 1
export PYTHONPATH="$HOME/NAS-Sensitivity:$PYTHONPATH"
if command -v module &>/dev/null; then
    module purge
    module load python/3.12.4
    module load cuda/12.4
fi

source .venv/bin/activate
echo "PYTHONPATH: $PYTHONPATH"
python --version
which python
python safety-nas/control-logic.py "$@"
