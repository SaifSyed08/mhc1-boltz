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

# Both overridable so scripts/ls6_ddp_smoke.slurm can run this exact launch
# path against a scratch directory with a few batches.
RUN_DIR="${RUN_DIR:-$REPO/runs/ls6_unfrozen}"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
mkdir -p "$RUN_DIR"

# NOT pinning CUDA_VISIBLE_DEVICES here: this run uses all three GPUs on the
# node via DDP (trainer.devices: 3), and pinning to 0 would hide the other two
# and silently drop throughput back to 1x while the config claimed otherwise.
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export BOLTZ_CHUNK_IN_TRAINING=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "node   : $(hostname)"
echo "job    : ${SLURM_JOB_ID:-?}"
echo "gpus   : $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) x $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "started: $(date)"
# gpu-a100-small hands out VIRTUAL nodes (v-prefixed hostnames) whose CPU
# flags do not include what the login-node-built numpy wheel needs:
#   NumPy was built with baseline optimizations (X86_V2) but your machine
#   doesn't support (X86_V2)
# Real compute nodes (c-prefixed) are fine. Print the flags either way.
echo "cpu    : $(lscpu | grep -i '^model name' | head -1 | cut -d: -f2- | xargs)"
echo "x86-v2 : $(grep -qE 'sse4_2' /proc/cpuinfo && echo yes || echo NO)"

ARGS=("output=$RUN_DIR" ${EXTRA[@]+"${EXTRA[@]}"})   # bash<4.4-safe empty expansion
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
        # Was "rm -rf $RUN_DIR (start clean, what you usually want)", which would
        # have deleted the finished 47 h unfrozen run. Starting clean is
        # right only when the LR settings changed; when only parallelism or
        # memory settings did, resuming is. Never suggest deleting results.
        echo "!! A checkpoint exists at $CKPT but ALLOW_RESUME is not set."
        echo "!! Resuming restores that checkpoint's LR scheduler state, overriding"
        echo "!! max_lr/lr_warmup_no_steps in the config. If those did NOT change:"
        echo "!!     sbatch --export=ALL,ALLOW_RESUME=1 scripts/ls6_unfrozen.slurm"
        echo "!! If they did, move the old run aside (do not delete it):"
        echo "!!     mv $RUN_DIR ${RUN_DIR}_\$(date +%m%d)"
        exit 1
    fi
else
    echo "fresh start from the pretrained checkpoint"
fi

cd boltz-src
CONFIG=../configs/mhc1_finetune_ls6_unfrozen.yaml
NGPU=$(awk '$1 == "devices:" { print $2; exit }' "$CONFIG")
case "$NGPU" in ''|*[!0-9]*) echo "!! cannot read trainer.devices from $CONFIG"; exit 1 ;; esac

# Launch through torchrun, NOT plain python. Inside a Slurm job Lightning sees
# SLURM_NTASKS, picks SLURMEnvironment, and assumes srun already started one
# process per GPU (creates_processes_externally=True; world_size=SLURM_NTASKS).
# With `-n 1` that is ONE process: `devices: 3` then trains on GPU 0 alone at a
# third of the effective batch, and nothing errors
# (lightning_fabric/plugins/environments/slurm.py, PL 2.4). torchrun is checked
# BEFORE Slurm (TorchElasticEnvironment has priority) and starts the processes.
# torchrun would otherwise force OMP_NUM_THREADS=1 per process; 16 x 3 ranks
# stays well inside the node's 128 cores.
export OMP_NUM_THREADS=16
echo "launch : torchrun --nproc_per_node=$NGPU"
torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" \
    scripts/train/train.py "$CONFIG" "${ARGS[@]}"
echo "finished: $(date)"
