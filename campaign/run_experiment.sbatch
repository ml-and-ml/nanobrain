#!/usr/bin/env bash
#SBATCH --job-name=nb-region
#SBATCH --partition=n
#SBATCH --account=sophont
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120

set -euo pipefail
CAMPAIGN_ROOT=/admin/home/hm/repos/nanobrain
export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri
export WANDB_MODE=disabled
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset BRAINMARKS_SMRI_TEST_SUBSET
exec "$CAMPAIGN_ROOT/.venv/bin/python" -u "$CAMPAIGN_ROOT/campaign/run.py" "$@"
