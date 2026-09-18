#!/bin/bash
#----------------------------------------------------------------------------
# Submit the whole experiment as a dependency chain and walk away.
#
#   bash scripts/ls6_pipeline.sh
#
# Slurm runs it unattended:
#
#   1. baseline    validation_only, reproduces val/lddt_protein_protein = 0.8827
#                  EXITS NONZERO if it does not reproduce within 0.02
#   2. fine-tune   starts only `afterok` the baseline, so a failed reproduction
#                  halts the chain instead of producing an incomparable number
#   3. fine-tune   a second instance, `afterany` the first, which resumes from
#      (resume)    last.ckpt. Covers the 48 h queue cap.
#
# Every job emails saif.s@utexas.edu on END or FAIL. Nothing to monitor.
#----------------------------------------------------------------------------
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p logs

BASE=$(sbatch --parsable scripts/ls6_baseline.slurm)
echo "  [1] baseline          job $BASE"

FT1=$(sbatch --parsable --dependency=afterok:"$BASE" scripts/ls6_finetune.slurm)
echo "  [2] fine-tune         job $FT1   (starts only if the baseline reproduces)"

FT2=$(sbatch --parsable --dependency=afterany:"$FT1" scripts/ls6_finetune.slurm)
echo "  [3] fine-tune resume  job $FT2   (continues from last.ckpt)"

cat <<MSG

Submitted. Nothing further to do -- Slurm runs the chain and emails you at each
stage.

  squeue -u $USER                     what is queued or running
  scancel $BASE $FT1 $FT2       stop everything

Results land in:
  runs/ls6_baseline/val_state.json
  runs/ls6_finetune/val_state.json
  logs/baseline.$BASE.out
  logs/finetune.$FT1.out
MSG
