#!/bin/bash
# Every validation-only eval on the 99 validation-era records, side by side,
# paired per structure against pretrained Boltz-1. Paste the output back.
#   cd $WORK/mhc1-boltz && git pull && bash scripts/ls6_evals.sh [--records]
# Covers runs/eval_val99/* (checkpoints), runs/eval_msa/* (MSA depth) and
# runs/eval_pocket/* (pocket conditioning).
# --records adds the per-record table (mean over samples) for every eval.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sacct -X --starttime now-3days --format=JobID%10,JobName%14,State%12,Elapsed%11 2>/dev/null | grep -i eval
python3 - "${1:-}" <<'PY'
import json, math, pathlib, statistics as st, sys

def jl(p):
    out, bad = [], 0
    for l in p.open():
        if l.strip():
            try:
                out.append(json.loads(l))
            except ValueError:
                bad += 1
    if bad:
        print(f"  !! {p}: {bad} unreadable line(s) skipped")
    return out

def fin(v):
    return [x for x in (v or []) if x is not None and math.isfinite(x)]

def load(d):
    """record -> row, last non-sanity scored row per record."""
    out = {}
    for p in sorted(d.glob("val_peptide_rank*.jsonl")):
        for r in jl(p):
            if not r.get("sanity") and r.get("record_id") and fin(r.get("pep_rmsd")):
                out[r["record_id"]] = r
    return out

def mean(v):
    return sum(v) / len(v)

held_in = set(pathlib.Path("data/processed/subset30_ids.txt").read_text().split())
dirs = [d for top in ("runs/eval_val99", "runs/eval_msa", "runs/eval_pocket")
        for d in sorted(pathlib.Path(top).glob("*/")) if d.is_dir()]
evals = {f"{d.parent.name}/{d.name}": load(d) for d in dirs}
pre = evals.get("eval_val99/pretrained", {})
if not pre:
    print("!! no runs/eval_val99/pretrained rows: paired columns will be empty")

def bucket(r):
    n = r.get("pep_len") or 0
    return "<8" if n < 8 else ("8-11" if n <= 11 else "12+")

print("\nper-sample median peptide CA RMSD (A); paired = per-record mean minus pretrained "
      "(median, n improved/n); held-out = the 69 records outside subset30")
print(f"{'eval':<28}{'n':>4} {'median':>7} {'smp0':>6} {'mean':>6} {'lddt_pp':>8}  "
      f"{'paired all':>15} {'paired held-out':>17}  {'<8':>5} {'8-11':>5} {'12+':>5}")
for name, rows in evals.items():
    if not rows:
        print(f"{name:<28}  (no scored rows)"); continue
    allv = [x for r in rows.values() for x in fin(r["pep_rmsd"])]
    s0 = fin([r["pep_rmsd"][0] for r in rows.values()])
    lp = [x for r in rows.values() for x in fin(r.get("lddt_pp"))]
    def paired(keys):
        d = [mean(fin(rows[k]["pep_rmsd"])) - mean(fin(pre[k]["pep_rmsd"])) for k in keys]
        return f"{st.median(d):+.3f} {sum(x < 0 for x in d)}/{len(d)}" if d else "-"
    common = [k for k in rows if k in pre]
    by = {b: [x for r in rows.values() if bucket(r) == b for x in fin(r["pep_rmsd"])]
          for b in ("<8", "8-11", "12+")}
    print(f"{name:<28}{len(rows):>4} {st.median(allv):>7.3f} {st.median(s0):>6.3f} "
          f"{mean(allv):>6.3f} {mean(lp) if lp else float('nan'):>8.4f}  "
          f"{paired(common):>15} {paired([k for k in common if k not in held_in]):>17}  "
          + " ".join(f"{st.median(v):>5.2f}" if v else "    -" for v in by.values()))

if sys.argv[1] == "--records":
    names = list(evals)
    print("\nrecord len " + " ".join(n.split("/")[1][:12] for n in names))
    ids = sorted(pre or next(iter(evals.values())), key=lambda k: -mean(fin(pre[k]["pep_rmsd"])) if k in pre else 0)
    for k in ids:
        L = (pre.get(k) or {}).get("pep_len", "?")
        print(k, L, " ".join(f"{mean(fin(evals[n][k]['pep_rmsd'])):.2f}" if k in evals[n] else "-" for n in names))
PY
