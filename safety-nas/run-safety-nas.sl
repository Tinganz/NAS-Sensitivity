#!/bin/bash

set -euo pipefail

# Submit safety-NAS batch on all training-designated tracks to SLURM in parallel
sbatch safety-nas/control-logic.sl --track AUSTIN &
sbatch safety-nas/control-logic.sl --track SEPANG &
sbatch safety-nas/control-logic.sl --track MELBOURNE &
sbatch safety-nas/control-logic.sl --track BRANDS_HATCH &
sbatch safety-nas/control-logic.sl --track BUDAPEST &
sbatch safety-nas/control-logic.sl --track CATALUNYA &
sbatch safety-nas/control-logic.sl --track HOCKENHEIM &
sbatch safety-nas/control-logic.sl --track IMS &
sbatch safety-nas/control-logic.sl --track MONTREAL &
sbatch safety-nas/control-logic.sl --track MOSCOW_RACEWAY &
sbatch safety-nas/control-logic.sl --track OSCHERSLEBEN &
sbatch safety-nas/control-logic.sl --track SAKHIR &
sbatch safety-nas/control-logic.sl --track SAO_PAULO &
sbatch safety-nas/control-logic.sl --track SPIELBERG &
sbatch safety-nas/control-logic.sl --track YAS_MARINA &
sbatch safety-nas/control-logic.sl --track ZANDVOORT &

wait
