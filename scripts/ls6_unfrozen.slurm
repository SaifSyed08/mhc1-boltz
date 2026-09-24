#!/bin/bash
#----------------------------------------------------------------------------
# The fine-tune, on Lonestar6. Submitted automatically by scripts/ls6_pipeline.sh
# with a dependency on the baseline job, so it only starts if the baseline
# reproduced.
#
# Resumes itself: if runs/ls6_finetune/last.ckpt exists it is passed as resume=,
# so re-queueing this script continues rather than restarting.
#----------------------------------------------------------------------------

#SBATCH -J mhc1-unfroz
#SBATCH -o logs/unfrozen.%j.out
#SBATCH -e logs/unfrozen.%j.err
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

RUN_DIR="$REPO/runs/ls6_unfrozen"
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
    # A checkpoint carries its optimizer AND LR-SCHEDULER state. Resuming into a
    # config whose learning rate has changed silently restores the OLD schedule
    # and the new max_lr/warmup never take effect -- the run looks like it used
    # the new settings and did not. That is unfalsifiable from the logs, so
    # require it to be asked for explicitly.
    if [ "${ALLOW_RESUME:-0}" = "1" ]; then
        echo "RESUMING from $CKPT (ALLOW_RESUME=1)"
        ARGS+=("resume=$CKPT")
    else
        echo "!! A checkpoint exists at $CKPT but ALLOW_RESUME is not set."
        echo "!! Resuming would restore that checkpoint's LR scheduler state and"
        echo "!! silently override the max_lr/lr_warmup_no_steps in the config."
        echo "!! Either:  rm -rf $RUN_DIR      (start clean, what you usually want)"
        echo "!!     or:  sbatch --export=ALL,ALLOW_RESUME=1 \$0   (continue it)"
        exit 1
    fi
else
    echo "fresh start from the pretrained checkpoint"
fi

cd boltz-src
python scripts/train/train.py ../configs/mhc1_finetune_ls6_unfrozen.yaml "${ARGS[@]}"
echo "finished: $(date)"
