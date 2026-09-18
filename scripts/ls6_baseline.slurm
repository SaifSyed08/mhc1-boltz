#!/bin/bash
#----------------------------------------------------------------------------
# Step 1 on Lonestar6: reproduce the pretrained baseline.
#
#   sbatch scripts/ls6_baseline.slurm
#
# This runs `validation_only` against the same 30-sample subset that produced
# val/lddt_protein_protein = 0.8827 on the laptop, with identical
# validation_args. It trains nothing.
#
# WHY THIS FIRST, before any fine-tuning: if the number does not reproduce, the
# environment differs from the one the baseline was measured in -- different
# torch, different CUDA, different arithmetic -- and every fine-tuned number
# measured afterwards would be compared against a baseline from a different
# machine. That is not a comparison. Roughly 0.88 here means the LS6 environment
# is sound and everything downstream is interpretable.
#
# It also gives a wall-clock reading for validation on an A100, which is what
# `check_val_every_n_epoch` should be tuned against.
#
# QUEUE: gpu-a100-small, 1.5 SU/node-hour. One GPU is all this uses.
# TIME:  validation was ~65 min for 30 samples on a T4. An A100 should be 2-5x
#        faster, but -t is set generously because a job killed at the limit
#        wastes everything it did. Slurm bills ACTUAL runtime, not the request --
#        the only cost of a generous -t is a slightly worse backfill position.
#----------------------------------------------------------------------------

#SBATCH -J mhc1-baseline
#SBATCH -o logs/baseline.%j.out
#SBATCH -e logs/baseline.%j.err
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
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

RUN_DIR="$REPO/runs/ls6_baseline"
mkdir -p "$RUN_DIR"

echo "node    : $(hostname)"
echo "job     : ${SLURM_JOB_ID:-?}"
echo "gpu     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "started : $(date)"
echo

# Lightning sees all three GPUs on a full node and the config says devices: 1,
# but pinning removes any ambiguity about which one and keeps DDP from ever
# being considered.
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NOT setting BOLTZ_CHUNK_IN_TRAINING here, deliberately. Chunking is already on
# for validation upstream (the gate is `not self.training`), so the env var only
# affects training -- and leaving it unset keeps this run byte-identical to the
# laptop baseline in every respect that could move the number.

cd boltz-src
python scripts/train/train.py ../configs/mhc1_baseline_subset30.yaml \
    "output=$RUN_DIR"

echo
echo "finished: $(date)"
echo
echo "=== compare against reports/BASELINE.md ==="
echo "  val/lddt_protein_protein   baseline 0.8827"
echo "  val/lddt_intra_protein     baseline 0.9248"
echo "  val/rmsd                   baseline 3.053 A"
echo "  val/best_rmsd              baseline 2.982 A  (best of 3)"
echo
echo "  Do NOT compare val/lddt (the aggregate) -- it is diluted across eight"
echo "  empty categories and caps at 0.3846 here. See reports/BASELINE.md."
echo
python - <<PYEOF
import json, pathlib
p = pathlib.Path("$RUN_DIR/val_state.json")
if not p.exists():
    print("  no val_state.json -- validation did not complete")
    raise SystemExit
m = json.load(p.open())["metrics"]
base = {"lddt.protein_protein": 0.8827, "lddt.intra_protein": 0.9248}
TOL = 0.02          # generous: fp arithmetic differs across architectures, but a
                    # genuine environment problem moves lDDT far more than this
print(f"  {'metric':<28}{'this run':>10}{'laptop':>10}{'delta':>10}")
bad = []
for k, b in base.items():
    v = m.get(k, {})
    if not v.get("weight"):
        bad.append(f"{k}: missing from val_state.json")
        continue
    got = v["mean_value"] / v["weight"]
    print(f"  {k:<28}{got:>10.4f}{b:>10.4f}{got-b:>+10.4f}")
    if abs(got - b) > TOL:
        bad.append(f"{k}: {got:.4f} vs {b:.4f}, |delta| > {TOL}")

if bad:
    print()
    print("  !! BASELINE DID NOT REPRODUCE")
    for line in bad:
        print(f"  !!   {line}")
    print("  !! Exiting nonzero so the dependent fine-tuning job does NOT run.")
    print("  !! A fine-tuned number measured in an environment whose baseline")
    print("  !! does not match is not comparable to anything.")
    raise SystemExit(1)

print()
print("  baseline reproduced within tolerance -- the dependent job will start")
PYEOF
