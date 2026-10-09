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
import json, math, sys, pathlib, statistics as st
d = pathlib.Path(sys.argv[1])

def jl(p):
    """Rows of a jsonl file; a truncated line (killed job, full disk) is skipped, not fatal."""
    out, bad = [], 0
    for l in p.open():
        if l.strip():
            try:
                out.append(json.loads(l))
            except ValueError:
                bad += 1
    if bad:
        print(f"  !! {p.name}: {bad} unreadable line(s) skipped")
    return out

def fin(v):
    return [x for x in v if x is not None and math.isfinite(x)]

# Training curve: rank-0 micro-batches, averaged per bucket of optimizer steps.
mem = d / "mem.jsonl"
if mem.exists():
    b = [r for r in jl(mem) if r.get("event") == "batch"]
    keys = sorted({k for r in b for k in r if k.startswith("train/")})
    print(f"  train: {len(b)} rank-0 micro-batches, last global_step "
          f"{b[-1]['global_step'] if b else '-'}; keys {keys}")
    if b:
        last = b[-1]["global_step"]
        edges = sorted({0, last // 8, last // 4, last // 2, 3 * last // 4, last})
        for lo, hi in zip(edges, edges[1:] + [last + 1]):
            seg = [r for r in b if lo <= r["global_step"] < hi]
            if not seg:
                continue
            msg = f"    steps {lo:>5}-{hi:<5} n={len(seg):<5}"
            for k in ("train/loss", "train/diffusion_loss", "train/distogram_loss"):
                raw = [r[k] for r in seg if r.get(k) is not None]
                v = fin(raw)
                if raw:
                    name = k.split("/")[1]
                    msg += f" {name} {st.mean(v):.4f}" if v else f" {name} nan"
                    if len(v) < len(raw):
                        msg += f"(!{len(raw) - len(v)} non-finite)"
            lr = [r["lr"] for r in seg if r.get("lr") is not None]
            if lr:
                msg += f"  lr {lr[-1]:.2e}"
            msg += f"  {st.median(fin([r.get('secs') for r in seg]) or [0]):.1f} s/batch"
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
    rows += jl(p)
rows = [r for r in rows if not r.get("sanity")]
if not rows:
    print("  (no peptide rows)"); raise SystemExit

def means(rs):
    """record -> mean peptide RMSD over its finite samples."""
    out = {}
    for r in rs:
        v = fin(r.get("pep_rmsd") or [])
        if v:
            out[r["record_id"]] = sum(v) / len(v)
    return out

# The reference is pretrained Boltz-1 under the same validation_args
# (runs/eval_val99/pretrained), not this run's first validation: that one comes
# after a whole epoch (step 10 at batch 129, ~144 at batch 9), so it would be a
# different starting point per arm.
pre = {}
for p in pathlib.Path("runs/eval_val99/pretrained").glob("val_peptide_rank*.jsonl"):
    pre.update(means(r for r in jl(p) if not r.get("sanity")))

steps = sorted({r["global_step"] for r in rows})
print("  peptide CA RMSD (A) per validation: median all samples | sample 0 | "
      "oracle best | mean | n ok/fail | non-finite | swaps>5A | vs pretrained "
      "(median paired diff, n improved/n)")
last_m, last_s = {}, None
for s in steps:
    rs = [r for r in rows if r["global_step"] == s]
    ok = [r for r in rs if r.get("pep_rmsd")]
    raw = [v for r in ok for v in r["pep_rmsd"]]
    allv = fin(raw)
    s0 = fin([r["pep_rmsd"][0] for r in ok])
    best = [min(fin(r["pep_rmsd"])) for r in ok if fin(r["pep_rmsd"])]
    swaps = sum(1 for r in ok for v in (r.get("boltz_truth_dev") or []) if v and v > 5)
    m = means(ok)
    common = sorted(set(m) & set(pre))
    diff = [m[k] - pre[k] for k in common]
    paired = (f"{st.median(diff):+.3f}, {sum(x < 0 for x in diff)}/{len(common)}"
              if diff else "-")
    if allv:
        print(f"    step {s:>5}  {st.median(allv):.3f} | {st.median(s0) if s0 else float('nan'):.3f} | "
              f"{st.median(best):.3f} | {st.mean(allv):.3f} | {len(ok)}/{len(rs) - len(ok)} | "
              f"{len(raw) - len(allv)} | {swaps} | {paired}")
    else:
        print(f"    step {s:>5}  no finite samples ({len(ok)} rows, {len(raw)} values)")
    last_m, last_s = m, s

common = sorted(set(last_m) & set(pre))
if common:
    worst = sorted(common, key=lambda k: -pre[k])[:6]
    print(f"  hardest for pretrained -> this run at step {last_s}: " +
          ", ".join(f"{k} {pre[k]:.2f}->{last_m[k]:.2f}" for k in worst))
PY
done

echo
echo "=================== eval_val99 per record (mean of 3 samples) ==="
# Compact enough to paste; feeds the peptide triage (reports/TRIAGE_peptides.md).
python3 - <<'PY'
import json, pathlib
tab = {}
for name in ("pretrained", "step153"):
    for p in pathlib.Path("runs/eval_val99", name).glob("val_peptide_rank*.jsonl"):
        for l in p.open():
            r = json.loads(l)
            if r.get("sanity") or not r.get("pep_rmsd"):
                continue
            v = [x for x in r["pep_rmsd"] if x is not None]
            tab.setdefault(r["record_id"], {})[name] = (sum(v) / len(v), min(v), max(v))
print("record pre_mean pre_min pre_max s153_mean")
for k in sorted(tab, key=lambda k: -tab[k].get("pretrained", (0,))[0]):
    a, b = tab[k].get("pretrained"), tab[k].get("step153")
    print(k, *(f"{x:.2f}" for x in (a or (float("nan"),) * 3)), f"{b[0]:.2f}" if b else "nan")
PY

echo
echo "=================== smoke ==================="
s=$(ls -t logs/*3499813*.out 2>/dev/null | head -1)
[ -n "$s" ] && tail -8 "$s" || echo "  (no smoke log yet)"

echo
echo "=================== checkpoints ============"
for d in $RUNS; do
    find "$d" -name "*.ckpt" -printf "%s %TY-%Tm-%Td %TH:%TM %p\n" 2>/dev/null |
        awk '{printf "  %-55s %6.2f GiB  %s %s\n", $4, $1 / 2^30, $2, $3}'
done
df -h "$WORK" 2>/dev/null | tail -1 | sed 's/^/  $WORK: /'

tar czf runs/round5_rows.tar.gz runs/eval_val99/*/val_peptide_rank*.jsonl runs/eval_msa/*/val_peptide_rank*.jsonl \
    $(for d in $RUNS; do ls "$d"/val_peptide_rank*.jsonl "$d"/val_state_epoch*.json "$d"/mem.jsonl 2>/dev/null; done) 2>/dev/null
echo
echo "rows bundled: runs/round5_rows.tar.gz ($(du -h runs/round5_rows.tar.gz 2>/dev/null | cut -f1))"
