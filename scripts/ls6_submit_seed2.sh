#!/bin/bash
# Second run of the two Round 6 arms that the email to Ernest calls "a second
# training seed". Run in an LS6 shell:
#   cd $WORK/mhc1-boltz && git pull -q && bash scripts/ls6_submit_seed2.sh
#
# WHY THIS SCRIPT EXISTS
# Jobs 3588216 / 3588217 (submitted 2026-11-16, job names mhc1-abrupt-s2 and
# mhc1-b9-s2) FAILED six seconds in on 2026-11-17. They were submitted with
# RUN_DIR pointing at the SEED-1 directories, so ls6_unfrozen.slurm's resume
# guard refused to start:
#   !! A checkpoint exists at runs/ls6_msa_abrupt/last.ckpt but ALLOW_RESUME is not set.
# The guard was right. ALLOW_RESUME=1 would have RESUMED seed 1 -- a longer
# first run, not a second one -- and silently restored seed 1's LR scheduler
# state on top of it. The fix is a fresh RUN_DIR, which is all this script
# changes.
#
# "SEED" IS A MISNOMER, DELIBERATELY KEPT
# boltz-src/scripts/train/train.py sets NO global seed (no seed_everything, no
# manual_seed), and `data.random_seed` reaches only ValidationDataset
# (data/module/training.py:632) -- the TrainingDataset takes no seed at all.
# So:
#   * there is nothing to increment; an unseeded rerun already differs in data
#     order, new-head init, augmentation and training diffusion noise. That is
#     exactly the variation we want to measure.
#   * data.random_seed STAYS 42. Changing it would re-draw the validation
#     sampling and break the like-for-like comparison against seed 1 -- the
#     opposite of the point.
# Kept the s2 name because that is what Ernest was told on 2026-11-17.
#
# WHAT IT ANSWERS
# Round 6 found the three models separated by less than the ~0.045 A
# draw-to-draw noise (reports/ls6/RESULT_round6.md s3a). This asks whether the
# arm ordering survives a second independent training run, which single-draw
# noise cannot answer.
set -euo pipefail
cd "$WORK/mhc1-boltz"
mkdir -p logs

# Queued jobs refuse to run if boltz-src drifts from the patch.
git -C boltz-src checkout -- . && git -C boltz-src apply ../patches/boltz-v1.0.0-training-path.patch
if ! diff -q <(git -C boltz-src diff | grep -v '^index ') \
             <(grep -v '^index ' patches/boltz-v1.0.0-training-path.patch) >/dev/null; then
    echo "!! boltz-src still differs from the patch; not submitting"; exit 1
fi
echo "boltz-src synced to $(git log -1 --format=%h)"

# Refuse to repeat the 3588216 mistake from the other direction: if these dirs
# already hold a checkpoint, the slurm guard will kill the job 6 s in, so say
# so now instead of burning two queue slots.
for d in runs/ls6_msa_abrupt_s2 runs/ls6_b9_lr3e-4_s2; do
    if [ -e "$d/last.ckpt" ] || [ -e "$d/checkpoints/last.ckpt" ]; then
        echo "!! $d already has a checkpoint. Move it aside (do not delete):"
        echo "!!     mv $d ${d}_\$(date +%m%d)"
        exit 1
    fi
done

id() { grep -E '^[0-9]+$' | tail -1; }    # TACC's sbatch prints a banner before the id
RECIPE="trainer.accumulate_grad_batches=3 model.training_args.max_lr=0.0003"

# abrupt: MSA cap 1 from the first step, validated at depth 1 (Round 6 arm)
abrupt=$(RUN_DIR="$WORK/mhc1-boltz/runs/ls6_msa_abrupt_s2" MSA_SCHEDULE="1" MSA_VAL_DEPTH=1 \
        EXTRA_ARGS="$RECIPE" sbatch --parsable --export=ALL -J mhc1-abrupt-s2 scripts/ls6_unfrozen.slurm | id)
# b9 3e-4: the control -- trained with FULL MSAs, no MSA_SCHEDULE, no
# MSA_VAL_DEPTH. It is scored query-only later via EVAL_MAX_SEQS=1 in
# scripts/ls6_eval_val99.slurm, which is what made it the arm that reframed
# Round 6.
b9=$(RUN_DIR="$WORK/mhc1-boltz/runs/ls6_b9_lr3e-4_s2" \
        EXTRA_ARGS="$RECIPE" sbatch --parsable --export=ALL -J mhc1-b9-s2 scripts/ls6_unfrozen.slurm | id)

echo "abrupt s2: $abrupt  -> runs/ls6_msa_abrupt_s2"
echo "b9 3e-4 s2: $b9  -> runs/ls6_b9_lr3e-4_s2"
echo
echo "Both are 48 h walltime on gpu-a100. When they finish, evaluate BOTH query-only"
echo "on all 99 with the same command shape Round 6 used, so the numbers are poolable:"
echo "  EVAL_MAX_SEQS=1 EVAL_CKPTS=\"abrupt_s2:runs/ls6_msa_abrupt_s2/last.ckpt b9_s2:runs/ls6_b9_lr3e-4_s2/last.ckpt\" \\"
echo "    sbatch --export=ALL scripts/ls6_eval_val99.slurm"
squeue -u "$USER" -o "%.10i %.16j %.10P %.2t %.20R" 2>/dev/null
