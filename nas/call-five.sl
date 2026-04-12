#!/bin/bash

set -euo pipefail

# Submit five control-logic SLURM jobs (one per track) in parallel.
sbatch nas/control-logic.sl --track SEPANG &
sbatch nas/control-logic.sl --track YAS_MARINA &
sbatch nas/control-logic.sl --track AUSTIN &
sbatch nas/control-logic.sl --track SAKHIR &
sbatch nas/control-logic.sl --track MELBOURNE &

wait
