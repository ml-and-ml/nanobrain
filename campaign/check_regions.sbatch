#!/bin/bash
#SBATCH --job-name=nb-final-regions
#SBATCH --partition=n
#SBATCH --account=sophont
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/admin/home/hm/repos/nanobrain/campaign/logs/final-regions-%j.out

set -euo pipefail
REPO=/admin/home/hm/repos/nanobrain
export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
for variant in main-regions ryan-regions; do
    PYTHONPATH="${REPO}/branches/${variant}/src" "${REPO}/.venv/bin/python" \
        "${REPO}/campaign/check_checkpoint.py" \
        "/data/hassan/nanobrain/anatomical-regions-20261109/${variant}/checkpoint-last.pth"
done
