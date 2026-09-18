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
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -A MCB26003
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saif.s@utexas.edu

set -euo pipefail
REPO="$WORK/mhc1-boltz"
cd "$REPO"

# Environment prep (module hygiene, PYTHONPATH, user-site). See scripts/ls6_env.sh.
source "$REPO/scripts/ls6_env.sh"
source "$REPO/.venv-ls6/bin/activate"
ls6_env_report

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
# gpu-a100-small hands out VIRTUAL nodes (v-prefixed hostnames) whose CPU
# flags do not include what the login-node-built numpy wheel needs:
#   NumPy was built with baseline optimizations (X86_V2) but your machine
#   doesn't support (X86_V2)
# Real compute nodes (c-prefixed) are fine. Print the flags either way.
echo "cpu    : $(lscpu | grep -i '^model name' | head -1 | cut -d: -f2- | xargs)"
echo "x86-v2 : $(grep -qE 'sse4_2' /proc/cpuinfo && echo yes || echo NO)"

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
