#!/bin/bash
#----------------------------------------------------------------------------
# Pocket conditioning as a partial template, plus structures for PyMOL.
# Validation only, pretrained Boltz-1, all 99 validation-era records, three runs:
#   plain      no pocket (same as runs/eval_val99/pretrained, a repeat draw);
#              writes structures for the hard cases
#   pseudoseq  binder = peptide, pocket = NetMHCpan's 34 pseudo-sequence
#              positions mapped onto each heavy chain (sequence only: fair)
#   oracle     pocket = heavy-chain residues within 6 A of the crystal peptide
#              (uses the answer: an upper bound for any anchor template)
# Pocket files: data/pockets_{pseudoseq,oracle}.json (src/make_pockets.py).
# Every run writes truth/sample mmCIF + view.pml for DUMP below, to
# runs/eval_pocket/<name>/structures/<id>/. Read with scripts/ls6_evals.sh.
# A run whose directory already holds a finished validation is skipped.
#----------------------------------------------------------------------------

#SBATCH -J mhc1-pocket
#SBATCH -o logs/evalpocket.%j.out
#SBATCH -e logs/evalpocket.%j.err
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
# Drop, difficult and the main can-improve cases of reports/TRIAGE_peptides.md,
# plus well-predicted controls from the same families (7cis, 7ow4, 7sif, 7wt3).
export STRUCTURE_DUMP_IDS="7wj3,7wt3,7t5m,8d5n,8d5q,6zkx,7tud,7cir,7cis,7ciq,7ndq,7ndu,7ow4,7ow5,7ow6,7ejn,7sih,7sif"

rc_all=0
for spec in "plain:" "pseudoseq:$REPO/data/pockets_pseudoseq.json" "oracle:$REPO/data/pockets_oracle.json"; do
    name=${spec%%:*}; pf=${spec#*:}
    out="$REPO/runs/eval_pocket/$name"
    if compgen -G "$out/val_state_epoch*_step*.json" >/dev/null; then
        echo "== $name: already done, skipping"; continue
    fi
    rm -rf "$out"
    echo "== $name: pocket file '${pf:-none}' -> $out"
    if [ -n "$pf" ]; then export VAL_POCKET_FILE="$pf"; else unset VAL_POCKET_FILE; fi
    RUN_DIR="$out" EXTRA_ARGS="pretrained=$ckpt validation_only=true disable_checkpoint=true" \
        bash scripts/ls6_unfrozen.slurm || { echo "!! $name failed"; rc_all=1; }
done
exit $rc_all
