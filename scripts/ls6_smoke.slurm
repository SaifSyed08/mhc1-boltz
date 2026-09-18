#!/bin/bash
#----------------------------------------------------------------------------
# Lonestar6 smoke test -- does a GPU job run at all, and what do we get?
#
# Deliberately trivial: it allocates a GPU node, prints what it sees, and exits.
# No data, no checkpoint, no training. The point is to separate "can I submit and
# run jobs" from "does my training work", which are two different failures and
# much easier to debug apart.
#
# SUBMIT FROM A LOGIN NODE:
#   mkdir -p logs                       # REQUIRED FIRST -- see below
#   sbatch scripts/ls6_smoke.slurm
#
# The logs/ directory must exist BEFORE submitting. Slurm opens the -o and -e
# files when the job launches, which is before the script body runs, so a
# `mkdir -p logs` inside the script is too late -- the job fails immediately with
# no output, and the absence of output is the only symptom.
#
# BEFORE FIRST USE, set -A below to your allocation name. Find it with:
#   /usr/local/etc/taccinfo
# It prints a "Project" column; that string is what -A wants.
#
# QUEUE CHOICE: gpu-a100-small, not gpu-a100. Per the LS6 queue table:
#
#   gpu-a100         8 nodes max   48 h   3.0 SU/node-hour
#   gpu-a100-small   1 node max    48 h   1.5 SU/node-hour   <- this one
#   gpu-a100-dev     2 nodes max    2 h   3.0 SU/node-hour
#   gpu-h100         1 node max    48 h   6.0 SU/node-hour
#
# gpu-a100 hands you a whole node; an LS6 A100 node has THREE A100s. This job,
# and the training config (trainer.devices: 1), use exactly one GPU. Taking a
# full node would bill double and idle two GPUs someone else is queuing for --
# and the cluster is currently ~97% utilised with 125 jobs waiting. The -small
# queue is both cheaper and the neighbourly choice.
#
# COST: 1.5 SU/node-hour with a 15-minute minimum charge => well under 1 SU.
#
# A tight -t also helps YOU at high load: Slurm's backfill scheduler can slot a
# 10-minute job into gaps that a 48-hour reservation cannot fit into, so a short
# job often starts sooner than the queue depth suggests. Never pad -t "just in
# case" -- it delays your own start.
#----------------------------------------------------------------------------

#SBATCH -J mhc1-smoke                  # job name
#SBATCH -o logs/smoke.%j.out           # stdout  (%j = job id)
#SBATCH -e logs/smoke.%j.err           # stderr
# NOTE: keep every directive line starting with "#SBATCH" at column 0 and put
# any prose on its own "#" line. Slurm stops scanning for directives at the first
# line that is not a comment or blank, so an indented continuation can silently
# drop everything below it -- including -A, which would make the job unchargeable.
#
#SBATCH -p gpu-a100-small
#SBATCH -N 1                           # one node
#SBATCH -n 1                           # one task
#SBATCH -t 00:10:00                    # 10 min; billing rounds up to 15 anyway
#SBATCH -A MCB26003                    # Kozakov group startup allocation
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saif.s@utexas.edu

set -euo pipefail
mkdir -p logs   # harmless; the real one runs before sbatch

echo "=================== node ==================="
hostname
echo "job id      : ${SLURM_JOB_ID:-?}"
echo "partition   : ${SLURM_JOB_PARTITION:-?}"
echo "nodes       : ${SLURM_JOB_NUM_NODES:-?}"
date

echo
echo "=================== gpu ===================="
# LS6 A100 nodes have THREE A100s at 40 GB each; the -small queue gives one.
# The T4 work assumed 16 GB, so 40 GB changes which deviations are still needed.
nvidia-smi || echo "!! nvidia-smi failed -- is this actually a GPU queue?"

echo
echo "=================== modules ================"
module list 2>&1 || true
echo
echo "available python / cuda modules:"
module avail python cuda 2>&1 | head -40 || true

echo
echo "=================== filesystems ============"
# TACC gives three, with very different rules. $HOME is small and backed up,
# $WORK is large and not purged, $SCRATCH is huge and PURGED periodically.
# The 3.8 GB of assets and any checkpoints belong in $WORK or $SCRATCH, never
# $HOME.
for v in HOME WORK SCRATCH; do
    d="${!v:-}"
    if [ -n "$d" ]; then
        printf '  %-8s %s\n' "$v" "$d"
        df -h "$d" 2>/dev/null | tail -1 | sed 's/^/           /'
    else
        printf '  %-8s (unset)\n' "$v"
    fi
done

echo
echo "=================== internet ==============="
# This is the one that matters for our setup. TACC COMPUTE nodes are typically
# firewalled off from the outside world even though login nodes are not. If this
# fails, the 3.8 GB checkpoint download and the `git clone` CANNOT happen inside
# a job -- they have to be done on a login node beforehand, and the job reads
# from $WORK. Better to learn that here than 20 minutes into a training job.
if timeout 15 curl -sI https://huggingface.co >/dev/null 2>&1; then
    echo "  compute node CAN reach the internet"
else
    echo "  compute node CANNOT reach the internet (expected)"
    echo "  -> stage assets and the repo from a LOGIN node into \$WORK first"
fi

echo
echo "=================== done ==================="
date
