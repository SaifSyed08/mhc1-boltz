"""Step 4: AlphaFold-3-style temporal split + duplicate/leakage analysis.

Split boundaries (from the brief):
  train      release date <= 2021-09-30
  validation 2021-10-01 .. 2023-01-13
  test       > 2023-01-13

Release dates come from RCSB `initial_release_date` (authoritative), and TCR3D's
own release_date column is cross-checked against it rather than trusted.

Leakage is defined on the (MHC allele, peptide sequence) pair, since that is the
biological identity a model could memorise. Where the same pair appears in more
than one split the copies in the *evaluation* splits are kept and the training
copies are dropped: the test set must stay untouched, and shrinking train is the
cheap side of the trade.
"""
import collections
import json
import pathlib

D = pathlib.Path(__file__).resolve().parents[1] / "data"
R = pathlib.Path(__file__).resolve().parents[1] / "reports"

TRAIN_END = "2021-09-30"
VAL_END = "2023-01-13"

meta = json.loads((D / "rcsb_meta.json").read_text())
ents = json.loads((D / "annotated_entities.json").read_text())
tcr3d = {r["pdbid"].upper(): r for r in json.loads((D / "tcr3d_mhc1_raw.json").read_text())}


def split_of(date):
    if date <= TRAIN_END:
        return "train"
    if date <= VAL_END:
        return "validation"
    return "test"


# ---------------------------------------------------------------- candidate set
cands, dropped = {}, []
for pid, rec in ents.items():
    roles = collections.Counter(e["role"] for e in rec["entities"])
    has_mhc = roles.get("MHC_I_HEAVY", 0) > 0
    has_pep = roles.get("PEPTIDE", 0) > 0
    if not has_mhc and roles.get("MHC_I_FUSION", 0):
        dropped.append((pid, "single_chain_fusion_construct"))
        continue
    if not has_mhc:
        dropped.append((pid, "no_mhc_i_heavy_chain_identified"))
        continue
    if not has_pep:
        pep = (tcr3d.get(pid, {}).get("peptide") or "").strip()
        reason = ("tcr3d_reports_no_peptide" if pep in ("", "None", "---")
                  else "no_separate_peptide_chain_in_structure")
        dropped.append((pid, reason))
        continue
    cands[pid] = rec

print("candidates with MHC-I heavy + peptide chain: %d" % len(cands))
print("dropped at annotation stage: %d" % len(dropped))
for r, n in collections.Counter(x[1] for x in dropped).most_common():
    print("    %-45s %d" % (r, n))

# ------------------------------------------------------------- dates and splits
rows = []
date_mismatch = []
for pid in sorted(cands):
    ai = meta[pid].get("rcsb_accession_info") or {}
    rel = (ai.get("initial_release_date") or "")[:10]
    t_rel = (tcr3d.get(pid, {}).get("release_date") or "")[:10]
    if rel and t_rel and rel != t_rel:
        date_mismatch.append((pid, rel, t_rel, split_of(rel), split_of(t_rel)))
    ei = meta[pid].get("rcsb_entry_info") or {}
    res = (ei.get("resolution_combined") or [None])[0]
    mhc = [e for e in cands[pid]["entities"] if e["role"] == "MHC_I_HEAVY"]
    pep = [e for e in cands[pid]["entities"] if e["role"] == "PEPTIDE"]
    rows.append({
        "pdb_id": pid,
        "release_date": rel,
        "split": split_of(rel),
        "resolution": res,
        "method": ((meta[pid].get("exptl") or [{}])[0] or {}).get("method"),
        "mhc_allele": tcr3d.get(pid, {}).get("mhc_allele_name"),
        "mhc_species": tcr3d.get(pid, {}).get("mhc_species"),
        "tcr_bound": bool(tcr3d.get(pid, {}).get("is_bound")),
        "peptide_seq_tcr3d": (tcr3d.get(pid, {}).get("peptide") or "").strip().upper(),
        "peptide_seq_struct": pep[0]["seq"],
        "n_mhc_entities": len(mhc),
        "n_peptide_entities": len(pep),
        "mhc_entity_ids": [e["entity_id"] for e in mhc],
        "peptide_entity_ids": [e["entity_id"] for e in pep],
        "mhc_seq": mhc[0]["seq"],
        "flags": sorted(set(cands[pid]["flags"])),
    })

by_split = collections.Counter(r["split"] for r in rows)
print("\ntemporal split (RCSB initial_release_date):")
for s in ("train", "validation", "test"):
    print("    %-11s %4d" % (s, by_split[s]))

print("\nrelease-date disagreements TCR3D vs RCSB: %d" % len(date_mismatch))
crossing = [m for m in date_mismatch if m[3] != m[4]]
print("    of which change the split assignment: %d" % len(crossing))
for m in crossing[:15]:
    print("      %s rcsb=%s(%s) tcr3d=%s(%s)" % (m[0], m[1], m[3], m[2], m[4]))

# --------------------------------------------------------------- leakage checks
def norm_allele(a):
    return (a or "").upper().replace(" ", "")


pair_key = lambda r: (norm_allele(r["mhc_allele"]), r["peptide_seq_struct"])
groups = collections.defaultdict(list)
for r in rows:
    groups[pair_key(r)].append(r)

dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
cross = {k: v for k, v in dup_groups.items() if len({x["split"] for x in v}) > 1}
print("\nduplicate (allele, peptide) pairs: %d groups covering %d entries"
      % (len(dup_groups), sum(len(v) for v in dup_groups.values())))
print("pairs spanning more than one split: %d groups" % len(cross))

RANK = {"test": 0, "validation": 1, "train": 2}
leak_removed = []
for k, v in cross.items():
    keep_split = min((x["split"] for x in v), key=lambda s: RANK[s])
    for x in v:
        if x["split"] != keep_split:
            x["leakage_action"] = "dropped_from_%s (pair kept in %s)" % (x["split"], keep_split)
            leak_removed.append((x["pdb_id"], x["split"], keep_split, k[0], k[1]))

kept = [r for r in rows if "leakage_action" not in r]
print("entries dropped to break cross-split leakage: %d" % len(leak_removed))
print("    from train      : %d" % sum(1 for x in leak_removed if x[1] == "train"))
print("    from validation : %d" % sum(1 for x in leak_removed if x[1] == "validation"))
print("    from test       : %d" % sum(1 for x in leak_removed if x[1] == "test"))

final = collections.Counter(r["split"] for r in kept)
print("\nfinal split after leakage removal:")
for s in ("train", "validation", "test"):
    print("    %-11s %4d" % (s, final[s]))
print("    TOTAL       %4d" % len(kept))

# same-split duplicates are fine for training but worth reporting
same = {k: v for k, v in dup_groups.items() if len({x["split"] for x in v}) == 1}
print("\nduplicate pairs entirely within one split (kept): %d groups, %d entries"
      % (len(same), sum(len(v) for v in same.values())))

R.mkdir(exist_ok=True)
# Only the retained rows go downstream -- writing `rows` here would carry the
# leakage-dropped entries into chain selection and back into the dataset.
(D / "candidates.json").write_text(json.dumps(kept, indent=1))
(D / "candidates_dropped_for_leakage.json").write_text(
    json.dumps([r for r in rows if "leakage_action" in r], indent=1))
(R / "split_report.json").write_text(json.dumps({
    "boundaries": {"train_end": TRAIN_END, "validation_end": VAL_END},
    "n_tcr3d_rows": len(tcr3d),
    "n_rcsb_resolved": len(meta),
    "n_candidates": len(rows),
    "dropped_at_annotation": [{"pdb_id": p, "reason": r} for p, r in dropped],
    "date_disagreements": [
        {"pdb_id": m[0], "rcsb": m[1], "tcr3d": m[2],
         "rcsb_split": m[3], "tcr3d_split": m[4]} for m in date_mismatch],
    "split_before_leakage": dict(by_split),
    "split_after_leakage": dict(final),
    "leakage_removed": [
        {"pdb_id": x[0], "from_split": x[1], "pair_kept_in": x[2],
         "allele": x[3], "peptide": x[4]} for x in leak_removed],
    "n_duplicate_pair_groups": len(dup_groups),
    "n_cross_split_pair_groups": len(cross),
}, indent=1))
print("\nwrote %s and %s" % (D / "candidates.json", R / "split_report.json"))
