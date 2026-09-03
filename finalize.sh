#!/usr/bin/env bash
# Re-run the downstream pipeline over whatever NPZs are present in data/npz_raw.
# Safe to run repeatedly; run again once tar_extract.py has finished to pick up
# the remaining structures.
set -e
cd "$(dirname "$0")"
echo "=== raw NPZs available: $(ls data/npz_raw/*.npz 2>/dev/null | wc -l) ==="
python src/step3_annotate.py   | tail -6
python src/step4_split.py      | tail -8
python src/step5_map_select.py | tail -10
python src/step6_write.py      | tail -14
./.venv/Scripts/python.exe src/step7_validate.py | tail -22
python src/step8_report.py     > /dev/null
echo "=== reports/MILESTONE_SUMMARY.md updated ==="
