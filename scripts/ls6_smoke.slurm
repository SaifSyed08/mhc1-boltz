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
#   sbatch scripts/ls6_smoke.slurm
#
# BEFORE FIRST USE, set -A below to your allocation name. Find it with:
#   /usr/local/etc/taccinfo
# It prints a "Project" column; that string is what -A wants.
#
# COST: GPU nodes are charged 4x SU/hour and every job is charged a minimum of
# 15 minutes regardless of runtime. So this costs ~1 SU. That is the cheapest
# possible way to find out your account works.
#----------------------------------------------------------------------------

#SBATCH -J mhc1-smoke                  # job name
#SBATCH -o logs/smoke.%j.out           # stdout  (%j = job id)
#SBATCH -e logs/smoke.%j.err           # stderr
# NOTE: keep every directive line starting with "#SBATCH" at column 0 and put
# any prose on its own "#" line. Slurm stops scanning for directives at the first
# line that is not a comment or blank, so an indented continuation can silently
# drop everything below it -- including -A, which would make the job unchargeable.
#
# gpu-a100 as advised. gpu-a100-dev schedules faster for short tests if your
# allocation covers it.
#SBATCH -p gpu-a100
#SBATCH -N 1                           # one node
#SBATCH -n 1                           # one task
#SBATCH -t 00:10:00                    # 10 min; billing rounds up to 15 anyway
#SBATCH -A MCB26003                    # Kozakov group startup allocation
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saif.s@utexas.edu

set -euo pipefail
mkdir -p logs

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
