#!/usr/bin/env bash
#SBATCH --job-name=nb-six-smoke
#SBATCH --partition=n
#SBATCH --account=sophont
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --open-mode=append

set -uo pipefail
CAMPAIGN_ROOT=/admin/home/hm/repos/nanobrain
export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri
export WANDB_MODE=disabled
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset BRAINMARKS_SMRI_TEST_SUBSET
attempt=${1:-1}
failed=0
for name in main-regions main-consistency main-coverage ryan-regions ryan-consistency ryan-coverage; do
    "$CAMPAIGN_ROOT/.venv/bin/python" -u "$CAMPAIGN_ROOT/campaign/run.py" "$name" --phase smoke --attempt "$attempt" || failed=1
done
exit "$failed"
