#!/bin/bash
# Round 6 in one go, run in an LS6 shell:
#   cd $WORK/mhc1-boltz && git pull -q && bash scripts/ls6_submit_round6.sh
# 1. syncs boltz-src to the repo's patch (queued jobs refuse to run otherwise)
# 2. dev queue:  smoke test + pocket/structure eval      (scripts/ls6_round6_dev.slurm)
# 3. gpu-a100:   two MSA-removal arms, fresh from pretrained, the Round 5 best
#                recipe (batch 9 = accumulate 3 x 3 GPUs, max_lr 3e-4), validating
#                at MSA depth 1:
#                  taper   caps 2048,512,128,32,8 for epochs 0-4, then 1 (Ernest's
#                          gradual removal)
#                  abrupt  cap 1 from the first step (tests whether gradual matters)
# Long jobs are NOT gated on the smoke (a dependency-held job earns no queue age).
# If the smoke fails: scontrol hold <taper> <abrupt>, and tell Claude.
set -euo pipefail
cd "$WORK/mhc1-boltz"
mkdir -p logs

git -C boltz-src checkout -- . && git -C boltz-src apply ../patches/boltz-v1.0.0-training-path.patch
if ! diff -q <(git -C boltz-src diff | grep -v '^index ') \
             <(grep -v '^index ' patches/boltz-v1.0.0-training-path.patch) >/dev/null; then
    echo "!! boltz-src still differs from the patch; not submitting"; exit 1
fi
echo "boltz-src synced to $(git log -1 --format=%h)"

id() { grep -E '^[0-9]+$' | tail -1; }    # TACC's sbatch prints a banner before the id
RECIPE="trainer.accumulate_grad_batches=3 model.training_args.max_lr=0.0003"
taper=$(RUN_DIR="$WORK/mhc1-boltz/runs/ls6_msa_taper" MSA_SCHEDULE="2048,512,128,32,8,1" MSA_VAL_DEPTH=1 \
        EXTRA_ARGS="$RECIPE" sbatch --parsable --export=ALL -J mhc1-msa-taper scripts/ls6_unfrozen.slurm | id)
abrupt=$(RUN_DIR="$WORK/mhc1-boltz/runs/ls6_msa_abrupt" MSA_SCHEDULE="1" MSA_VAL_DEPTH=1 \
        EXTRA_ARGS="$RECIPE" sbatch --parsable --export=ALL -J mhc1-msa-abrupt scripts/ls6_unfrozen.slurm | id)
# Last, and allowed to fail: the dev queue takes 3 submitted jobs per user.
dev=$(sbatch --parsable scripts/ls6_round6_dev.slurm | id || true)
echo "dev (smoke + pocket eval): ${dev:-NOT SUBMITTED (dev queue full?) -- resubmit scripts/ls6_round6_dev.slurm later}"
echo "msa taper:  $taper"
echo "msa abrupt: $abrupt"
echo "--- how the Round 5 b9 3e-4 arm was launched (for parity):"
grep -m4 -iE "accumulate|max_lr|EXTRA|ARGS" logs/unfrozen.3499815.out 2>/dev/null || echo "(not in its log)"
squeue -u "$USER" -o "%.10i %.16j %.10P %.2t %.20R" 2>/dev/null
