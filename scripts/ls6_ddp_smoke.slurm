#!/bin/bash
#----------------------------------------------------------------------------
# Does the 3-GPU launch actually run 3 GPUs? ~35 min on the dev queue, gating
# the 48 h job behind it (submit that with --dependency=afterok:<this job>).
#
# Runs scripts/ls6_unfrozen.slurm itself -- the same env, same torchrun launch,
# same config -- against a scratch run dir, fresh from the pretrained weights,
# for 50 training batches (2 optimizer steps at accumulate 43: the second one is
# where DDP reports unused parameters) and 2 validation batches per rank.
#
# Exits non-zero unless ALL of these hold, so afterok means something:
#   * training exited 0
#   * val_state.json reports world_size 3            (torchrun really launched 3)
#   * lddt weight == 3 ranks x 2 batches = 6          (the cross-rank sum works)
#   * mem.jsonl has batch records                     (telemetry survived DDP)
#   * mem.jsonl batch records carry train/loss        (the loss curve is recorded)
#   * val_peptide_rank*.jsonl: 6 real-validation rows, distinct record ids,
#     3 finite MHC-peptide RMSDs each                 (peptide scoring works)
# and prints peak memory and s/batch, the two numbers that decide whether
# activation offload can come off, plus each row's peptide RMSD.
#----------------------------------------------------------------------------

#SBATCH -J mhc1-ddpsmk
#SBATCH -o logs/ddpsmoke.%j.out
#SBATCH -e logs/ddpsmoke.%j.err
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

export RUN_DIR="$REPO/runs/ddp_smoke"
rm -rf "$RUN_DIR"                    # scratch, never results
export EXTRA_ARGS="trainer.limit_train_batches=50 trainer.limit_val_batches=2 trainer.max_epochs=1 trainer.max_time=null disable_checkpoint=true"

bash scripts/ls6_unfrozen.slurm
rc=$?

source "$REPO/scripts/ls6_env.sh"     # module hygiene, or TACC python3 leaks into PYTHONPATH
source "$REPO/.venv-ls6/bin/activate"
python - "$RUN_DIR" "$rc" <<'EOF'
import json, pathlib, statistics, sys
run, rc = pathlib.Path(sys.argv[1]), int(sys.argv[2])
fail = []
if rc != 0:
    fail.append(f"training exited {rc}")
try:
    v = json.loads((run / "val_state.json").read_text())
    w = v["metrics"]["lddt.protein_protein"]["weight"]
    print(f"world_size={v.get('world_size')} scope={v.get('scope')} lddt_weight={w:g}")
    if v.get("world_size") != 3:
        fail.append(f"world_size {v.get('world_size')} != 3")
    if w != 6:
        fail.append(f"lddt weight {w:g} != 6 (3 ranks x 2 batches)")
except Exception as e:
    fail.append(f"val_state.json unreadable: {e}")
try:
    recs = [json.loads(l) for l in (run / "mem.jsonl").read_text().splitlines() if l.strip()]
    b = [r for r in recs if r.get("event") == "batch"]
    secs = [r["secs"] for r in b[5:] if r.get("secs")]          # skip warm-up
    print(f"batches={len(b)} peak_gb={max(r['peak_gb'] for r in b):.2f} "
          f"median_s_per_batch={statistics.median(secs):.1f} last_global_step={b[-1]['global_step']}")
    if not b:
        fail.append("no batch records in mem.jsonl")
    losses = [r.get("train/loss") for r in b]
    print(f"train/loss on {sum(l is not None for l in losses)}/{len(b)} batches, "
          f"first {losses[0]} last {losses[-1]}")
    if not all(isinstance(l, float) and l == l for l in losses):
        fail.append("train/loss missing or NaN in mem.jsonl batch records")
except Exception as e:
    fail.append(f"mem.jsonl unreadable: {e}")
try:
    rows = [json.loads(l) for f in sorted(run.glob("val_peptide_rank*.jsonl"))
            for l in f.read_text().splitlines() if l.strip()]
    real = [r for r in rows if not r.get("sanity")]
    for r in real:
        print(f"  rank {r['rank']} {r.get('record_id')}: pep_rmsd {r.get('pep_rmsd')} "
              f"groove {r.get('groove_rmsd')} boltz_rmsd {r.get('boltz_rmsd')} {r.get('reason', '')}")
    ids = [r.get("record_id") for r in real]
    good = [r for r in real if len(r.get("pep_rmsd") or []) == 3
            and all(isinstance(x, float) and 0 <= x < 30 for x in r["pep_rmsd"])]
    if len(real) != 6:
        fail.append(f"{len(real)} peptide rows from the real validation, expected 6")
    if None in ids or len(set(ids)) != len(ids):
        fail.append(f"record ids missing or repeated: {ids}")
    if len(good) != len(real):
        fail.append(f"{len(real) - len(good)} rows without 3 finite peptide RMSDs")
except Exception as e:
    fail.append(f"val_peptide rows unreadable: {e}")
print("SMOKE " + ("PASS" if not fail else "FAIL: " + "; ".join(fail)))
sys.exit(1 if fail else 0)
EOF
