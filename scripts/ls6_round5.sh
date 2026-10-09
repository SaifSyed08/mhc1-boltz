#!/bin/bash
# Round 5 digest: what the batch-9 and batch-129 arms did. Paste the output back.
#   cd $WORK/mhc1-boltz && git pull && bash scripts/ls6_round5.sh
# Also bundles every run's per-structure rows into runs/round5_rows.tar.gz so
# they can be copied into the repo in one go.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JOBS=3499813,3499814,3499815,3499581,3499582
RUNS="runs/ls6_b9_lr1e-4 runs/ls6_b9_lr3e-4 runs/ls6_lr3e-4 runs/ls6_lr1e-3"

echo "=================== jobs ==================="
sacct -j "$JOBS" -X --format=JobID%10,JobName%14,State%12,ExitCode%6,Start%20,Elapsed%11 2>/dev/null
squeue -u "$USER" 2>/dev/null | tail -n +1

for j in ${JOBS//,/ }; do
    e=$(ls logs/*."$j".err 2>/dev/null | head -1)
    [ -s "$e" ] || continue
    # Only the lines that say why a job stopped; the full stderr is mostly warnings.
    hits=$(grep -iE "error|traceback|killed|oom|cancel|time limit" "$e" | grep -v "UserWarning" | tail -6)
    [ -n "$hits" ] && { echo "--- $e"; echo "$hits"; }
done

for d in $RUNS; do
    echo
    echo "=================== $d ==================="
    [ -d "$d" ] || { echo "  (no run dir)"; continue; }
    python3 - "$d" <<'PY'
import json, sys, pathlib, statistics as st
d = pathlib.Path(sys.argv[1])

# Training curve: rank-0 micro-batches, averaged per bucket of optimizer steps.
mem = d / "mem.jsonl"
if mem.exists():
    rows = [json.loads(l) for l in mem.open() if l.strip()]
    b = [r for r in rows if r.get("event") == "batch"]
    keys = sorted({k for r in b for k in r if k.startswith("train/")})
    print(f"  train: {len(b)} rank-0 micro-batches, last global_step "
          f"{b[-1]['global_step'] if b else '-'}; keys {keys}")
    if b:
        last = b[-1]["global_step"]
        edges = sorted({0, last // 8, last // 4, last // 2, 3 * last // 4, last})
        for lo, hi in zip(edges, edges[1:] + [last + 1]):
            seg = [r for r in b if lo <= r["global_step"] < hi] if hi > lo else []
            if not seg:
                continue
            msg = f"    steps {lo:>5}-{hi:<5} n={len(seg):<5}"
            for k in ("train/loss", "train/diffusion_loss", "train/distogram_loss"):
                v = [r[k] for r in seg if r.get(k) is not None]
                if v:
                    msg += f" {k.split('/')[1]} {st.mean(v):.4f}"
            lr = [r["lr"] for r in seg if r.get("lr") is not None]
            if lr:
                msg += f"  lr {lr[-1]:.2e}"
            msg += f"  {st.median([r['secs'] for r in seg if r.get('secs')] or [0]):.1f} s/batch"
            print(msg)
else:
    print("  (no mem.jsonl)")

# Boltz's own metrics, one line per validation.
for p in sorted(d.glob("val_state_epoch*.json")):
    m = json.load(p.open())["metrics"]
    def g(k):
        v = m.get(k, {})
        return v["mean_value"] / v["weight"] if v.get("weight") else float("nan")
    print(f"  {p.stem:<24} lddt_pp {g('lddt.protein_protein'):.4f}  "
          f"rmsd {g('rmsd'):.3f}  best_rmsd {g('best_rmsd'):.3f}")

# Peptide RMSD, one line per validation step (sanity pass excluded).
rows = []
for p in sorted(d.glob("val_peptide_rank*.jsonl")):
    rows += [json.loads(l) for l in p.open() if l.strip()]
rows = [r for r in rows if not r.get("sanity")]
if not rows:
    print("  (no peptide rows)"); raise SystemExit
steps = sorted({r["global_step"] for r in rows})
print(f"  peptide CA RMSD (A) per validation: median all samples | sample 0 | "
      f"oracle best | mean | n ok/fail | swaps>5A")
first = {}
for s in steps:
    rs = [r for r in rows if r["global_step"] == s]
    ok = [r for r in rs if r.get("pep_rmsd")]
    allv = [v for r in ok for v in r["pep_rmsd"] if v is not None]
    s0 = [r["pep_rmsd"][0] for r in ok if r["pep_rmsd"][0] is not None]
    best = [r["pep_rmsd_best"] for r in ok]
    swaps = sum(1 for r in ok for v in (r.get("boltz_truth_dev") or []) if v and v > 5)
    if allv:
        print(f"    step {s:>5}  {st.median(allv):.3f} | {st.median(s0):.3f} | "
              f"{st.median(best):.3f} | {st.mean(allv):.3f} | {len(ok)}/{len(rs) - len(ok)} | {swaps}")
    if not first:
        first = {r["record_id"]: r["pep_rmsd_mean"] for r in ok}
    last = {r["record_id"]: r["pep_rmsd_mean"] for r in ok}

# Paired change, first validation -> last, per structure.
common = sorted(set(first) & set(last))
if common and len(steps) > 1:
    diff = [last[k] - first[k] for k in common]
    better = sum(x < 0 for x in diff)
    print(f"  paired step {steps[0]} -> {steps[-1]} (mean of samples): median "
          f"{st.median(diff):+.3f}, mean {st.mean(diff):+.3f}, {better}/{len(common)} improve")
    worst = sorted(common, key=lambda k: -first[k])[:6]
    print("  hardest at the first validation: " +
          ", ".join(f"{k} {first[k]:.2f}->{last[k]:.2f}" for k in worst))
PY
done

echo
echo "=================== smoke ==================="
s=$(ls -t logs/*3499813*.out 2>/dev/null | head -1)
[ -n "$s" ] && tail -8 "$s" || echo "  (no smoke log yet)"

echo
echo "=================== checkpoints ============"
for d in $RUNS; do find "$d" -name "*.ckpt" -printf "  %-55p %6.2f GiB  %TY-%Tm-%Td %TH:%TM\n" 2>/dev/null; done
df -h "$WORK" 2>/dev/null | tail -1 | sed 's/^/  $WORK: /'

tar czf runs/round5_rows.tar.gz runs/eval_val99/*/val_peptide_rank*.jsonl \
    $(for d in $RUNS; do ls "$d"/val_peptide_rank*.jsonl "$d"/val_state_epoch*.json "$d"/mem.jsonl 2>/dev/null; done) 2>/dev/null
echo
echo "rows bundled: runs/round5_rows.tar.gz ($(du -h runs/round5_rows.tar.gz 2>/dev/null | cut -f1))"
