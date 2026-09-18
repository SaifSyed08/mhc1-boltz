#!/bin/bash
# Attach the fine-tuning jobs to an ALREADY-SUBMITTED baseline.
#   bash scripts/ls6_chain.sh <baseline-job-id>
# Use when the baseline submitted but the dependents did not.
set -euo pipefail
BASE="${1:?usage: bash scripts/ls6_chain.sh <baseline-job-id>}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
submit() { sbatch --parsable "$@" | grep -oE '^[0-9]+$' | tail -1; }

FT1=$(submit --dependency=afterok:"$BASE" scripts/ls6_finetune.slurm)
echo "  fine-tune         job $FT1   (afterok $BASE)"
FT2=$(submit --dependency=afterany:"$FT1" scripts/ls6_finetune.slurm)
echo "  fine-tune resume  job $FT2   (afterany $FT1)"
echo
squeue -u "$USER"
