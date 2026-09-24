#!/bin/bash
#----------------------------------------------------------------------------
# CONTROL: validate the PRETRAINED checkpoint using the FINE-TUNE config.
#
#   sbatch scripts/ls6_control.slurm
#
# Why this exists. Both fine-tuning runs produced the same pattern:
#
#                       val/rmsd   val/best_rmsd   lddt_protein_protein
#   pretrained baseline   3.06         2.98             0.8843
#   Kaggle fine-tune      2.640        1.690            0.8840
#   LS6 fine-tune         2.6401       1.7032           0.8821
#
# RMSD improves by ~0.4 A and best_rmsd by ~1.3 A, in two runs that share almost
# no hyperparameters -- 6 optimizer steps at batch 16 versus ~80 at batch 128,
# multiplicity 4 versus 16, EMA off versus on -- while lDDT does not move at all
# in either. Training that hard and getting an identical RMSD shift is not what
# learning looks like.
#
# The two families of runs differ in more than their weights: the baseline is
# `validation_only: true` (trainer.validate) with mhc1_baseline_subset30.yaml,
# the fine-tunes validate inside trainer.fit() with mhc1_finetune_*.yaml. This
# job holds the WEIGHTS fixed at pretrained and switches only the config and
# code path.
#
#   RMSD comes back ~3.05  ->  the config is not the cause. Training really did
#                              move RMSD without moving lDDT, which is a genuine
#                              and interesting result worth chasing.
#   RMSD comes back ~2.64  ->  the improvement is an artifact of the config or
#                              the validation path, and BOTH fine-tuning RMSD
#                              numbers are meaningless.
#
# Cost: one validation pass, about 4.5 SU. Cheap against reporting a result that
# is not real.
#----------------------------------------------------------------------------

#SBATCH -J mhc1-control
#SBATCH -o logs/control.%j.out
#SBATCH -e logs/control.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A MCB26003
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saif.s@utexas.edu

set -euo pipefail
REPO="$WORK/mhc1-boltz"
cd "$REPO"
source "$REPO/scripts/ls6_env.sh"
source "$REPO/.venv-ls6/bin/activate"
ls6_env_report

RUN_DIR="$REPO/runs/ls6_control"
mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "node : $(hostname)   gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "start: $(date)"
echo
echo "PRETRAINED weights, FINE-TUNE config, validation_only."
echo

cd boltz-src
# validation_only + the finetune config. `pretrained` already points at
# boltz1_conf.ckpt and no `resume` is given, so the weights are untouched.
python scripts/train/train.py ../configs/mhc1_finetune_ls6.yaml \
    "output=$RUN_DIR" validation_only=true

echo
echo "finished: $(date)"
python - <<PYEOF
import json, pathlib
p = pathlib.Path("$RUN_DIR/val_state.json")
if not p.exists():
    print("  no val_state.json"); raise SystemExit
m = json.load(p.open())["metrics"]
ref = {"lddt.protein_protein": (0.8843, 0.8821), "rmsd": (3.0611, 2.6401),
       "best_rmsd": (2.9785, 1.7032)}
print(f"  {'metric':<22}{'control':>10}{'LS6 base':>10}{'LS6 FT':>10}")
for k, (b, ft) in ref.items():
    v = m.get(k, {})
    if v.get("weight"):
        got = v["mean_value"] / v["weight"]
        near = "matches BASELINE" if abs(got-b) < abs(got-ft) else "matches FINE-TUNE"
        print(f"  {k:<22}{got:>10.4f}{b:>10.4f}{ft:>10.4f}   {near}")
print()
print("  rmsd near 3.05 -> training moved it. near 2.64 -> the config did.")
PYEOF
