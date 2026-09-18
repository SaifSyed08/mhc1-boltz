#!/bin/bash
# Digest of everything worth knowing about the LS6 runs. Paste the output back.
#   cd $WORK/mhc1-boltz && git pull && bash scripts/ls6_report.sh
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=================== jobs ==================="
sacct --starttime now-7days --format=JobID%14,JobName%14,State%12,ExitCode%8,Elapsed%11,Partition%16 2>/dev/null | head -25

echo
echo "=================== queue =================="
squeue -u "$USER" 2>/dev/null || true

echo
echo "=================== baseline ==============="
b=$(ls -t logs/baseline.*.out 2>/dev/null | head -1)
if [ -n "$b" ]; then echo "($b)"; tail -25 "$b"; else echo "no baseline log"; fi

echo
echo "=================== finetune ==============="
f=$(ls -t logs/finetune.*.out 2>/dev/null | head -1)
if [ -n "$f" ]; then
    echo "($f)"
    grep -E "freeze_trunk|RESUMING|fresh start|^gpu|^started|^finished|Skipping batch|oom-context|Error|Traceback" "$f" | tail -20
    echo "--- last lines ---"
    tail -12 "$f"
else
    echo "no finetune log yet"
fi

echo
echo "=================== metrics ================"
for d in runs/ls6_baseline runs/ls6_finetune; do
    python3 - "$d" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
p = d / "val_state.json"
print(f"  {d}:")
if not p.exists():
    print("    (no val_state.json)"); raise SystemExit
m = json.load(p.open())["metrics"]
base = {"lddt.protein_protein": 0.8827, "lddt.intra_protein": 0.9248,
        "rmsd": 3.053, "best_rmsd": 2.982}
for k, b in base.items():
    v = m.get(k, {})
    if v.get("weight"):
        got = v["mean_value"] / v["weight"]
        print(f"    {k:<24}{got:>9.4f}   laptop {b:<8}  delta {got-b:+.4f}")
PY
done

echo
echo "=================== checkpoints ============"
find runs -name "*.ckpt" -printf "  %-60p %8.2f GiB\n" 2>/dev/null | sed 's/GiB/GiB/' | head
du -sh runs 2>/dev/null | sed 's/^/  total: /'
