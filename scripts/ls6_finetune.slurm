#!/bin/bash
#----------------------------------------------------------------------------
# The fine-tune, on Lonestar6. Submitted automatically by scripts/ls6_pipeline.sh
# with a dependency on the baseline job, so it only starts if the baseline
# reproduced.
#
# Resumes itself: if runs/ls6_finetune/last.ckpt exists it is passed as resume=,
# so re-queueing this script continues rather than restarting.
#----------------------------------------------------------------------------

#SBATCH -J mhc1-ft
#SBATCH -o logs/finetune.%j.out
#SBATCH -e logs/finetune.%j.err
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -A MCB26003
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saif.s@utexas.edu

set -euo pipefail
REPO="$WORK/mhc1-boltz"
cd "$REPO"

module load cuda/12.8 2>/dev/null || module load cuda/12.2 || true
source "$REPO/.venv-ls6/bin/activate"

RUN_DIR="$REPO/runs/ls6_finetune"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export BOLTZ_CHUNK_IN_TRAINING=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "node   : $(hostname)"
echo "job    : ${SLURM_JOB_ID:-?}"
echo "gpu    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "started: $(date)"

ARGS=("output=$RUN_DIR")
# Resume from the newest checkpoint if there is one. train.py ignores
# `pretrained` when `resume` is set, which is correct -- we want the optimizer
# state back, not a fresh load of the pretrained weights.
CKPT=$(ls -t "$RUN_DIR"/last.ckpt "$RUN_DIR"/checkpoints/last.ckpt 2>/dev/null | head -1 || true)
if [ -n "$CKPT" ]; then
    echo "RESUMING from $CKPT"
    ARGS+=("resume=$CKPT")
else
    echo "fresh start from the pretrained checkpoint"
fi

cd boltz-src
python scripts/train/train.py ../configs/mhc1_finetune_ls6.yaml "${ARGS[@]}"
echo "finished: $(date)"
