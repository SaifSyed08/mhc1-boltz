#!/bin/bash
#----------------------------------------------------------------------------
# Validation only, on all 99 validation-era records, for each checkpoint below,
# through the same launch path as training (scripts/ls6_unfrozen.slurm with
# CONFIG/RUN_DIR/EXTRA_ARGS), so the numbers line up with training-time
# validations. Per checkpoint, runs/eval_val99/<name>/ gets val_state.json and
# val_peptide_rank*.jsonl -- per-structure MHC-peptide CA RMSD, the metric
# MHC-Fine is judged on, which no run here has measured before.
#
# Why 99 and not subset30: the 69 records outside subset30 were never trained
# on and never used to pick a checkpoint (configs/mhc1_eval_ls6_val99.yaml).
#
# ~20 min per checkpoint on 3 GPUs. A checkpoint whose directory already has a
# finished validation is skipped, so a resubmission only does what is missing.
#----------------------------------------------------------------------------

#SBATCH -J mhc1-eval99
#SBATCH -o logs/eval99.%j.out
#SBATCH -e logs/eval99.%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A MCB26003
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saif.s@utexas.edu

set -uo pipefail
REPO="$WORK/mhc1-boltz"
cd "$REPO"
export CONFIG=../configs/mhc1_eval_ls6_val99.yaml

CKPTS=(
  "pretrained:$REPO/data/assets/boltz1_conf.ckpt"
  "step153:$REPO/runs/ls6_unfrozen/r3_end_step153.ckpt"
)
# Other checkpoints: EVAL_CKPTS="name:path name:path" (paths relative to the repo
# or absolute), e.g. sbatch --export=ALL,EVAL_CKPTS="b9_3e-4_last:runs/ls6_b9_lr3e-4/last.ckpt" ...
if [ -n "${EVAL_CKPTS:-}" ]; then
    read -r -a CKPTS <<< "$EVAL_CKPTS"
    for i in "${!CKPTS[@]}"; do
        n=${CKPTS[$i]%%:*}; p=${CKPTS[$i]#*:}
        case "$p" in /*) ;; *) p="$REPO/$p" ;; esac
        CKPTS[$i]="$n:$p"
    done
fi
rc_all=0
for spec in "${CKPTS[@]}"; do
    name=${spec%%:*}; ckpt=${spec#*:}
    out="$REPO/runs/eval_val99/$name"
    # The per-epoch copy, not val_state.json: that one is rewritten after every
    # batch, so it also exists for a run that died halfway.
    if compgen -G "$out/val_state_epoch*_step*.json" >/dev/null; then
        echo "== $name: already done, skipping"; continue
    fi
    if [ ! -f "$ckpt" ]; then echo "!! $name: no checkpoint at $ckpt"; rc_all=1; continue; fi
    rm -rf "$out"                    # a half-finished earlier attempt, never results
    echo "== $name: $ckpt -> $out"
    RUN_DIR="$out" EXTRA_ARGS="pretrained=$ckpt validation_only=true disable_checkpoint=true" \
        bash scripts/ls6_unfrozen.slurm || { echo "!! $name failed"; rc_all=1; }
done
exit $rc_all
