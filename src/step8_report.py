"""Step 8: assemble the milestone deliverable summary from the stage reports."""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
D, R = ROOT / "data", ROOT / "reports"
OUT = D / "processed"


def load(p, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def main():
    tcr3d = load(D / "tcr3d_mhc1_raw.json", [])
    meta = load(D / "rcsb_meta.json", {})
    cands = load(D / "candidates.json", [])
    sel = load(D / "chain_selection.json", {})
    split_rep = load(R / "split_report.json", {})
    write_rep = load(R / "write_report.json", {})
    val_rep = load(R / "validation_report.json", {})

    selected = {p: r for p, r in sel.items() if r.get("keep_chain_idx")}
    by_split = collections.Counter(r["split"] for r in cands)

    drop = collections.Counter(d["reason"] for d in split_rep.get("dropped_at_annotation", []))
    leak = split_rep.get("leakage_removed", [])
    leak_by = collections.Counter(x["from_split"] for x in leak)

    lines = []
    a = lines.append
    a("# MHC-I dataset preparation for Boltz-1 -- milestone summary")
    a("")
    a("## 1. Sample counts")
    a("")
    a("| stage | count |")
    a("|---|---|")
    a("| MHC-I entries listed by TCR3D | %d |" % len(tcr3d))
    a("| resolved in RCSB | %d |" % len(meta))
    a("| with an MHC-I heavy chain AND a separate peptide chain | %d |" % len(cands))
    a("| after temporal split + leakage removal | %d |"
      % sum(split_rep.get("split_after_leakage", {}).values()))
    a("| Boltz NPZ present locally so far | %d |" % len(sel))
    a("| chain-mapped and pMHC-I selected | %d |" % len(selected))
    a("| written to the processed dataset | %d |" % write_rep.get("n_written", 0))
    a("")
    a("## 2. Splits (AlphaFold-3 style, RCSB initial_release_date)")
    a("")
    a("train <= 2021-09-30, validation 2021-10-01..2023-01-13, test > 2023-01-13")
    a("")
    a("| split | after annotation | after leakage removal | written |")
    a("|---|---|---|---|")
    after = split_rep.get("split_after_leakage", {})
    wrote = write_rep.get("splits", {})
    for s in ("train", "validation", "test"):
        a("| %s | %d | %d | %d |" % (s, by_split.get(s, 0), after.get(s, 0), wrote.get(s, 0)))
    a("")
    a("## 3. Removals and why")
    a("")
    a("| reason | count |")
    a("|---|---|")
    for k, v in drop.most_common():
        a("| %s | %d |" % (k.replace("_", " "), v))
    for s in ("train", "validation", "test"):
        if leak_by.get(s):
            a("| dropped from %s to break (allele, peptide) leakage | %d |" % (s, leak_by[s]))
    a("")
    a("## 3b. Candidates with no Boltz sample")
    a("")
    have = {p for p in sel}
    miss = [r for r in cands if r["pdb_id"] not in have]
    if miss:
        yrs = collections.Counter(r["release_date"][:4] for r in miss)
        recent = sum(v for k, v in yrs.items() if k >= "2024")
        a("%d of %d post-split candidates have no NPZ in rcsb_processed_targets.tar."
          % (len(miss), len(cands)))
        a("")
        a("**%d of those %d were released in 2024 or later** -- Boltz-1's PDB snapshot"
          % (recent, len(miss)))
        a("effectively ends in early 2024, so they cannot be matched. They must be")
        a("processed from mmCIF with boltz's scripts/process/rcsb.py to be usable.")
        a("")
        a("| release year | missing |")
        a("|---|---|")
        for k, v in sorted(yrs.items()):
            a("| %s | %d |" % (k, v))
    else:
        a("_none_")
    a("")
    a("## 4. Validation results")
    a("")
    if val_rep:
        a("| check | pass | total |")
        a("|---|---|---|")
        for k, v in val_rep.get("checks", {}).items():
            a("| %s | %d | %d |" % (v["description"], v["pass"], v["total"]))
        fails = val_rep.get("failures", {})
        a("")
        a("Failures: %s" % ("none" if not fails
                            else ", ".join("%s=%d" % (k, len(v)) for k, v in fails.items())))
        a("Needs manual review: %d" % len(val_rep.get("needs_manual_review", [])))
    else:
        a("_validation not yet run_")
    a("")
    a("## 5. Chain selection statistics")
    a("")
    if selected:
        kept = collections.Counter(r["n_chains_kept"] for r in selected.values())
        tot = sorted(r["n_chains_total"] for r in selected.values())
        con = sorted(r["mhc_pep_contacts"] for r in selected.values())
        a("- chains present in the original Boltz sample: median %d, max %d"
          % (tot[len(tot) // 2], tot[-1]))
        a("- chains kept: %s" % dict(sorted(kept.items())))
        a("- chains disabled in total: %d" % write_rep.get("stats", {}).get("disabled_chains", 0))
        a("- MHC-peptide heavy-atom contacts (<=4.5 A): min %d, median %d, max %d"
          % (con[0], con[len(con) // 2], con[-1]))
        fl = collections.Counter(f for r in selected.values() for f in r.get("flags", []))
        a("- flags: %s" % (dict(fl) or "none"))
    a("")
    (R / "MILESTONE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nwrote %s" % (R / "MILESTONE_SUMMARY.md"))


if __name__ == "__main__":
    main()
