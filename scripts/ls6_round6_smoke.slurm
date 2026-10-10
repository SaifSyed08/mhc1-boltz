#!/bin/bash
#----------------------------------------------------------------------------
# Round 6 smoke test, dev queue, ~30 min. Exercises the three new code paths
# through the real 3-GPU launch (scripts/ls6_unfrozen.slurm) before any 48 h job
# depends on them:
#   1. MSA schedule (src/msa_schedule.py): 2 short epochs at caps 8 then 1,
#      validation at depth 1. Passes only if the batches actually carried
#      depth 8, then 1, then validated at 1 (msa_schedule.jsonl "seen" events).
#   2. Structure dump (src/structure_dump.py): truth + 3 samples + view.pml for
#      the validated records, with a sane atom count.
#   3. Pocket conditioning (VAL_POCKET_FILE): a validation-only pass with the
#      NetMHCpan pseudo-sequence pockets; passes only if the pocket file loaded
#      and the rows are finite.
# Exits non-zero on any failure. If it fails, hold the Round 6 training arms.
#----------------------------------------------------------------------------

#SBATCH -J mhc1-r6smk
#SBATCH -o logs/r6smoke.%j.out
#SBATCH -e logs/r6smoke.%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A MCB26003
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=saif.s@utexas.edu

set -uo pipefail
REPO="$WORK/mhc1-boltz"
cd "$REPO"
S1="$REPO/runs/r6_smoke_train"; S2="$REPO/runs/r6_smoke_pocket"
rm -rf "$S1" "$S2"                   # scratch, never results

echo "== 1+2: MSA schedule 8 -> 1 and structure dump, 2 short epochs"
RUN_DIR="$S1" MSA_SCHEDULE="8,1" MSA_VAL_DEPTH=1 \
STRUCTURE_DUMP_IDS="$(tr '\n' ',' < data/processed/subset30_ids.txt)" \
EXTRA_ARGS="trainer.limit_train_batches=8 trainer.limit_val_batches=1 trainer.max_epochs=2 trainer.max_time=null trainer.accumulate_grad_batches=3 disable_checkpoint=true" \
    bash scripts/ls6_unfrozen.slurm > "$S1.log" 2>&1
rc1=$?; tail -5 "$S1.log"

echo "== 3: pocket-conditioned validation (pseudo-sequence), 1 batch per rank"
CONFIG=../configs/mhc1_eval_ls6_val99.yaml RUN_DIR="$S2" VAL_POCKET_FILE="$REPO/data/pockets_pseudoseq.json" \
EXTRA_ARGS="validation_only=true disable_checkpoint=true trainer.limit_val_batches=1" \
    bash scripts/ls6_unfrozen.slurm > "$S2.log" 2>&1
rc2=$?; tail -5 "$S2.log"

source "$REPO/scripts/ls6_env.sh"
source "$REPO/.venv-ls6/bin/activate"
python - "$S1" "$S2" "$rc1" "$rc2" <<'PY'
import json, pathlib, sys
s1, s2 = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rc1, rc2 = int(sys.argv[3]), int(sys.argv[4])
fail = []
if rc1: fail.append(f"schedule/dump run exited {rc1}")
if rc2: fail.append(f"pocket run exited {rc2}")
try:
    ev = [json.loads(l) for l in (s1 / "msa_schedule.jsonl").read_text().splitlines() if l.strip()]
    for e in ev: print("  ", e)
    seen = {e["epoch"]: e["batch_msa_depth"] for e in ev if e["event"] == "seen"}
    if seen != {0: 8, 1: 1}:
        fail.append(f"train MSA depth by epoch {seen}, want {{0: 8, 1: 1}}")
    vs = [e["batch_msa_depth"] for e in ev if e["event"] == "val_seen"]
    if not vs or any(v != 1 for v in vs):
        fail.append(f"validation MSA depths {vs}, want all 1")
except Exception as e:
    fail.append(f"msa_schedule.jsonl unreadable: {e}")
d = sorted((s1 / "structures").glob("*/view.pml"))
print(f"  structure dumps: {[p.parent.name for p in d]}")
if not d:
    fail.append("no structure dumps")
for p in d[:1]:
    for f in ["truth.cif", "sample0.cif", "sample1.cif", "sample2.cif"]:
        n = sum(l.startswith(("ATOM", "HETATM")) for l in (p.parent / f).read_text().splitlines())
        print(f"  {p.parent.name}/{f}: {n} atoms")
        if n < 1000:
            fail.append(f"{p.parent.name}/{f} has {n} atoms")
log2 = pathlib.Path(str(s2) + ".log").read_text()
if "[val_pocket] 99 pockets" not in log2:
    fail.append("pocket file not loaded ('[val_pocket] 99 pockets' missing from log)")
rows = [json.loads(l) for f in s2.glob("val_peptide_rank*.jsonl") for l in f.read_text().splitlines() if l.strip()]
rows = [r for r in rows if not r.get("sanity")]
for r in rows: print(f"  pocket {r.get('record_id')}: pep_rmsd {r.get('pep_rmsd')} {r.get('reason', '')}")
if len(rows) != 3 or not all(len(r.get("pep_rmsd") or []) == 3 for r in rows):
    fail.append(f"pocket run: {len(rows)} rows, want 3 with 3 samples each")
print("R6 SMOKE " + ("PASS" if not fail else "FAIL: " + "; ".join(fail)))
sys.exit(1 if fail else 0)
PY
