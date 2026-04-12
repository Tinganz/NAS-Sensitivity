#!/bin/bash

set -euo pipefail

# Submit five control-logic SLURM jobs (one per track) in parallel.
sbatch nas/control-logic.sl --track SEPANG &
sleep 2
sbatch nas/control-logic.sl --track YAS_MARINA &
sleep 2
sbatch nas/control-logic.sl --track AUSTIN &
sleep 2
sbatch nas/control-logic.sl --track SAKHIR &
sleep 2
sbatch nas/control-logic.sl --track MELBOURNE &

wait
