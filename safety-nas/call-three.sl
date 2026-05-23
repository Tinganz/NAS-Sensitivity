#!/bin/bash

set -euo pipefail

# Submit three control-logic SLURM jobs (one per track) in parallel.
sbatch safety-nas/control-logic.sl --track AUSTIN &
sbatch safety-nas/control-logic.sl --track SEPANG &
sbatch safety-nas/control-logic.sl --track MELBOURNE &

wait
