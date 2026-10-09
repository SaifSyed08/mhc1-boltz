#!/bin/bash
#----------------------------------------------------------------------------
# How much does pretrained Boltz-1 lose when the MSA is cut at inference?
# Ernest (2026-10-12) expects a drop, and suggests training it away by giving the
# model fewer MSA rows each step until it runs on the query alone. This is that
# experiment's starting point: validation only, all 99 validation-era records,
# pretrained weights, MSA depth capped at each value below. Depth 2048 is the
# existing runs/eval_val99/pretrained row, so it is not repeated.
#
# Three things to know when reading the result:
# * Training already varies MSA depth. Upstream's featurizer draws
#   max_seqs_batch = randint(1, max_seqs + 1) per training sample
#   (featurizer.py ~1174), so every fine-tune here, and Boltz-1 itself, has seen
#   shallow MSAs -- but depth <= 16 in only ~0.8% of draws at max_seqs 2048.
#   Gradual removal is a shift of that distribution, not a new mechanism.
# * "Depth N" is the first N rows of the taxonomy-paired table (paired
#   heavy-chain/b2m rows first, then unpaired), not the top-N hits per chain.
# * No seed is set, so diffusion noise differs from the depth-2048 reference job;
#   lDDT differences of ~0.002 are noise.
#
# data.max_seqs caps the rows fed to the model. Validation downsamples
# deterministically (featurizer: pairing[:max_seqs]) and the first row is always
# the query, so max_seqs=1 is single-sequence mode. Output per depth:
# runs/eval_msa/msa<depth>/ with val_state.json and val_peptide_rank*.jsonl.
# A depth whose directory already holds a finished validation is skipped.
#----------------------------------------------------------------------------

#SBATCH -J mhc1-evalmsa
#SBATCH -o logs/evalmsa.%j.out
#SBATCH -e logs/evalmsa.%j.err
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
ckpt="$REPO/data/assets/boltz1_conf.ckpt"

rc_all=0
for depth in 1 16 128; do
    out="$REPO/runs/eval_msa/msa$depth"
    if compgen -G "$out/val_state_epoch*_step*.json" >/dev/null; then
        echo "== msa$depth: already done, skipping"; continue
    fi
    rm -rf "$out"                    # a half-finished earlier attempt, never results
    echo "== msa$depth: max_seqs=$depth -> $out"
    RUN_DIR="$out" EXTRA_ARGS="pretrained=$ckpt validation_only=true disable_checkpoint=true data.max_seqs=$depth" \
        bash scripts/ls6_unfrozen.slurm || { echo "!! msa$depth failed"; rc_all=1; }
done
exit $rc_all
