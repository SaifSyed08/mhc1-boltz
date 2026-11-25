#!/bin/bash
#----------------------------------------------------------------------------
# Run scripts/seed2_digest.py as soon as the seed-2 eval job finishes, on LS6,
# so the comparison exists whether or not a laptop is awake to ask for it.
#
# Submit it chained to the eval job:
#   sbatch --dependency=afterany:<evaljobid> scripts/ls6_seed2_digest.slurm
#
# afterany, not afterok: a failed or timed-out eval is exactly when the
# diagnosis below is worth having. The script checks the completion markers
# itself and says what is missing instead of producing a median over whichever
# records happened to be written.
#
# Pure stdlib (json/glob/pathlib/statistics/math), so no venv and no GPU.
#----------------------------------------------------------------------------
#SBATCH -J seed2-digest
#SBATCH -o logs/seed2_digest.%j.out
#SBATCH -e logs/seed2_digest.%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -A MCB26003
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saif.s@utexas.edu

set -uo pipefail
cd "$WORK/mhc1-boltz"
OUT=runs/eval_val99/SEED2_DIGEST.txt      # runs/ is gitignored; commit deliberately
mkdir -p runs/eval_val99

PY=$(command -v python3 || command -v python)
[ -n "$PY" ] || { echo "!! no python on PATH"; exit 1; }

want="abrupt_s2 b9_s2 abrupt_s2_r2 b9_s2_r2"
missing=""
for n in $want; do
    compgen -G "runs/eval_val99/$n/val_state_epoch*_step*.json" >/dev/null || missing="$missing $n"
done

{
  echo "seed2 digest, $(date), host $(hostname)"
  echo "eval arms expected:$want"
  if [ -n "$missing" ]; then
      echo
      echo "!! NOT RUNNING THE DIGEST. These arms have no completion marker:$missing"
      echo "!! val_state.json alone is rewritten after every batch and exists for a run"
      echo "!! that died halfway, so a digest over a partial set would be a median over"
      echo "!! whichever records happened to be written. Fix the eval, then rerun:"
      echo "!!     cd \$WORK/mhc1-boltz && $PY scripts/seed2_digest.py"
      echo
      echo "--- what is on disk:"
      for n in $want; do
          echo "  $n: $(ls runs/eval_val99/$n/*.jsonl 2>/dev/null | wc -l) jsonl, marker=$(compgen -G "runs/eval_val99/$n/val_state_epoch*_step*.json" >/dev/null && echo yes || echo NO)"
      done
      exit 1
  fi
  echo "all four markers present; running the digest"
  echo
  "$PY" scripts/seed2_digest.py
} > "$OUT" 2>&1
rc=$?

echo "wrote $OUT (rc=$rc)"
cat "$OUT"
exit $rc
