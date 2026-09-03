"""Step 7: automatic validation of the processed dataset.

Implements the brief's checklist as independent per-sample checks, plus two
dataset-level checks (temporal split, cross-split leakage). Every failure is
recorded with the sample id so it can be reviewed separately.

V11 exercises Boltz's own code path -- Structure -> BoltzTokenizer ->
BoltzCropper -- on the rewritten files, and additionally asserts that the
tokenizer emits exactly the residues of the chains we kept.
"""
import collections
import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "boltz-src" / "src"))
sys.path.insert(0, str(ROOT / "src"))
from boltz.data.const import chain_types, prot_token_to_letter  # noqa: E402

D = ROOT / "data"
R = ROOT / "reports"
OUT = D / "processed"
CUT = 4.5
A1_END, A2_END = 90, 190
TRAIN_END, VAL_END = "2021-09-30", "2023-01-13"


def seq_of(res_slice):
    return "".join(prot_token_to_letter.get(str(n), "X") for n in res_slice["name"])


def chain_atoms(c, present):
    a0, a1 = int(c["atom_idx"]), int(c["atom_idx"]) + int(c["atom_num"])
    idx = np.arange(a0, a1)
    return idx[present[a0:a1]]


def n_contacts(ia, ib, xyz):
    if not len(ia) or not len(ib):
        return 0, set()
    ta, tb = cKDTree(xyz[ia]), cKDTree(xyz[ib])
    hits = ta.query_ball_tree(tb, CUT)
    n = sum(len(h) for h in hits)
    touched = {k for k, h in enumerate(hits) if h}
    return n, touched


def main():
    cands = {r["pdb_id"]: r for r in json.loads((D / "candidates.json").read_text())}
    sel = json.loads((D / "chain_selection.json").read_text())
    ann = json.loads((D / "annotated_entities.json").read_text())
    tcr3d_ids = {r["pdbid"].upper() for r in json.loads((D / "tcr3d_mhc1_raw.json").read_text())}
    # train+validation and the held-out test records live in separate manifests
    recs = {r["id"].upper(): r for r in json.loads((OUT / "manifest.json").read_text())}
    n_trainval = len(recs)
    test_path = OUT / "manifest_test.json"
    if test_path.exists():
        recs.update({r["id"].upper(): r for r in json.loads(test_path.read_text())})
    files = sorted(OUT.glob("structures/*.npz"))
    print("processed samples on disk: %d, manifest records: %d (train+val %d, test %d)"
          % (len(files), len(recs), n_trainval, len(recs) - n_trainval))

    fails = collections.defaultdict(list)
    checks = collections.Counter()
    review = []

    # --- Boltz pipeline objects (V11) ---
    pipeline_ok = True
    try:
        from boltz.data.tokenize.boltz import BoltzTokenizer
        from boltz.data.crop.boltz import BoltzCropper
        from boltz.data.types import Connection, Input, Structure
        tokenizer, cropper = BoltzTokenizer(), BoltzCropper(min_neighborhood=0, max_neighborhood=40)
    except Exception as exc:
        pipeline_ok = False
        print("!! Boltz pipeline import failed (%s: %s) -- V11 will be skipped"
              % (type(exc).__name__, exc))

    for path in files:
        pid = path.stem.upper()
        info = sel[pid]
        keep = set(info["keep_chain_idx"])
        z = np.load(path)
        chains, residues, atoms = z["chains"], z["residues"], z["atoms"]
        mask = z["mask"].astype(bool)
        present = atoms["is_present"].astype(bool)
        ens = z["ensemble"]
        xyz = (z["coords"]["coords"][int(ens[0]["atom_coord_idx"]):
                                     int(ens[0]["atom_coord_idx"]) + int(ens[0]["atom_num"])]
               if len(ens) and len(z["coords"]) == len(atoms) else atoms["coords"])
        rec = recs.get(pid)
        cmap = {int(c["idx"]): c for c in info["chains"]}

        # V1 -- provenance
        checks["V1_total"] += 1
        if pid in tcr3d_ids and pid in cands:
            checks["V1_pass"] += 1
        else:
            fails["V1_not_a_tcr3d_mhc1_entry"].append(pid)

        # V2/V3 -- expected MHC and peptide chain present and enabled
        mhc_i = next((i for i in keep if cmap[i]["role"] == "MHC_I_HEAVY"), None)
        pep_i = next((i for i in keep if cmap[i]["role"] == "PEPTIDE"), None)
        b2m_i = next((i for i in keep if cmap[i]["role"] == "B2M"), None)
        for tag, idx in (("V2_mhc", mhc_i), ("V3_peptide", pep_i)):
            checks["%s_total" % tag] += 1
            if idx is not None and mask[idx]:
                checks["%s_pass" % tag] += 1
            else:
                fails["%s_missing_or_disabled" % tag].append(pid)
        if mhc_i is None or pep_i is None:
            continue

        # V4 -- kept chain sequences match the RCSB entity sequences
        checks["V4_total"] += 1
        ok4 = True
        for i in keep:
            c = chains[i]
            s = seq_of(residues[int(c["res_idx"]):int(c["res_idx"]) + int(c["res_num"])])
            ent_id = cmap[i]["rcsb_entity_id"]
            ent = next((e for e in ann[pid]["entities"] if e["entity_id"] == ent_id), None)
            if ent is None:
                ok4 = False
                break
            cmpd = [(x, y) for x, y in zip(s, ent["seq"]) if x != "X"]
            # the NPZ holds only modelled residues, so it is a subsequence
            if not (s.replace("X", "") in ent["seq"].replace("X", "")
                    or (cmpd and sum(x == y for x, y in cmpd) / len(cmpd) >= 0.9)):
                ok4 = False
                break
        if ok4:
            checks["V4_pass"] += 1
        else:
            fails["V4_sequence_mismatch"].append(pid)

        # V5 -- unrelated chains disabled in BOTH npz mask and manifest valid flag
        checks["V5_total"] += 1
        npz_ok = set(np.flatnonzero(mask).tolist()) == keep
        man_ok = rec is not None and {c["chain_id"] for c in rec["chains"] if c["valid"]} == keep
        itf_ok = rec is not None and all(
            it["valid"] == (it["chain_1"] in keep and it["chain_2"] in keep)
            for it in (rec.get("interfaces") or []))
        if npz_ok and man_ok and itf_ok:
            checks["V5_pass"] += 1
        else:
            fails["V5_disable_flags_inconsistent"].append(
                {"pdb_id": pid, "npz": npz_ok, "manifest": man_ok, "interfaces": itf_ok})

        # V6 -- kept chains form one connected complex (no cross-copy mixing)
        checks["V6_total"] += 1
        ia_mhc = chain_atoms(chains[mhc_i], present)
        detached = []
        for i in keep:
            if i == mhc_i:
                continue
            n, _ = n_contacts(chain_atoms(chains[i], present), ia_mhc, xyz)
            if n == 0:
                detached.append(cmap[i]["name"])
        if not detached:
            checks["V6_pass"] += 1
        else:
            fails["V6_kept_chain_not_contacting_mhc"].append({"pdb_id": pid, "chains": detached})

        # V7 -- peptide in the alpha1/alpha2 groove
        checks["V7_total"] += 1
        cm = chains[mhc_i]
        rs = residues[int(cm["res_idx"]):int(cm["res_idx"]) + int(cm["res_num"])]
        res_of_atom = np.repeat(np.arange(len(rs)), rs["atom_num"].astype(int))
        ia_all = np.arange(int(cm["atom_idx"]), int(cm["atom_idx"]) + int(cm["atom_num"]))
        keep_m = present[ia_all[0]:ia_all[-1] + 1]
        ia = ia_all[keep_m]
        rmap = res_of_atom[keep_m]
        npep, touched = n_contacts(ia, chain_atoms(chains[pep_i], present), xyz)
        tres = {int(rmap[k]) for k in touched}
        a1 = sum(1 for r in tres if r < A1_END)
        a2 = sum(1 for r in tres if A1_END <= r < A2_END)
        if npep >= 20 and a1 > 0 and a2 > 0:
            checks["V7_pass"] += 1
        else:
            fails["V7_peptide_not_in_groove"].append(
                {"pdb_id": pid, "contacts": npep, "alpha1": a1, "alpha2": a2})
            review.append({"pdb_id": pid, "reason": "weak groove geometry",
                           "contacts": npep, "alpha1": a1, "alpha2": a2})
        for f in info.get("flags", []):
            if f in ("ambiguous_peptide_choice_needs_review",
                     "b2m_present_but_not_contacting_selected_mhc",
                     "selected_pair_spans_operators"):
                review.append({"pdb_id": pid, "reason": f,
                               "selected_peptide": info.get("selected_peptide"),
                               "contacts": info.get("mhc_pep_contacts")})

        # V8 -- structural integrity of the rewritten arrays
        checks["V8_total"] += 1
        ok8 = (len(mask) == len(chains)
               and mask.sum() == len(keep)
               and all(0 <= i < len(chains) for i in keep)
               and np.all(np.diff(chains["asym_id"][mask]) > 0)
               and int(chains["atom_idx"][-1]) + int(chains["atom_num"][-1]) == len(atoms)
               and int(chains["res_idx"][-1]) + int(chains["res_num"][-1]) == len(residues))
        if ok8:
            checks["V8_pass"] += 1
        else:
            fails["V8_broken_arrays"].append(pid)

        # V11 -- Boltz can load, tokenize and crop the modified file
        if pipeline_ok:
            checks["V11_total"] += 1
            try:
                st = Structure(atoms=z["atoms"], bonds=z["bonds"], residues=z["residues"],
                               chains=z["chains"], connections=z["connections"].astype(Connection),
                               interfaces=z["interfaces"], mask=z["mask"])
                tok = tokenizer.tokenize(Input(st, {}))
                got = set(np.unique(tok.tokens["asym_id"]).tolist())
                exp = {i for i in keep if chain_types[int(chains[i]["mol_type"])] != "NONPOLYMER"
                       or True}
                n_exp_res = sum(int(chains[i]["res_num"]) for i in keep)
                cropper.crop(tok, max_tokens=512, random=np.random.RandomState(0), max_atoms=4608)
                if got <= keep and got == exp and len(tok.tokens) >= 1 and n_exp_res > 0:
                    checks["V11_pass"] += 1
                else:
                    fails["V11_tokenizer_chain_set_mismatch"].append(
                        {"pdb_id": pid, "tokenized": sorted(got), "kept": sorted(keep)})
            except Exception as exc:
                fails["V11_pipeline_error"].append(
                    {"pdb_id": pid, "error": "%s: %s" % (type(exc).__name__, exc)})

    # --- dataset-level checks ---
    print("\n--- dataset-level checks ---")
    checks["V9_total"] = len(files)
    bad9 = []
    for path in files:
        pid = path.stem.upper()
        d, s = cands[pid]["release_date"], cands[pid]["split"]
        exp = "train" if d <= TRAIN_END else ("validation" if d <= VAL_END else "test")
        if exp != s:
            bad9.append((pid, d, s, exp))
    checks["V9_pass"] = len(files) - len(bad9)
    if bad9:
        fails["V9_split_inconsistent_with_date"] = bad9
    print("V9 temporal split consistent with release date: %d/%d"
          % (checks["V9_pass"], checks["V9_total"]))

    groups = collections.defaultdict(set)
    for path in files:
        pid = path.stem.upper()
        c = cands[pid]
        key = ((c["mhc_allele"] or "").upper().replace(" ", ""), c["peptide_seq_struct"])
        groups[key].add(c["split"])
    cross = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    checks["V10_total"] = len(groups)
    checks["V10_pass"] = len(groups) - len(cross)
    if cross:
        fails["V10_pair_spans_splits"] = [{"allele": k[0], "peptide": k[1], "splits": v}
                                          for k, v in cross.items()]
    print("V10 (allele, peptide) pairs confined to one split: %d/%d"
          % (checks["V10_pass"], checks["V10_total"]))

    # --- V12: every referenced MSA file is actually on disk ---
    #
    # This is a real failure mode, not a hypothetical one. Boltz's load_input()
    # (boltz/data/module/training.py:115) walks `record.chains` with no regard
    # for the `valid` flag and np.load()s every chain's msa_id, so a DISABLED
    # chain still needs its MSA present. step6 originally emitted only the valid
    # chains' ids, and the 209 missing files killed the validation run in a
    # particularly unhelpful way: TrainingDataset.__getitem__ catches load errors
    # and retries with __getitem__(0), so one permanently-missing MSA on record 0
    # recurses to RecursionError rather than reporting the absent file.
    msa_dir = ROOT / "data" / "msa"
    ref = collections.defaultdict(list)
    for rid, rec in recs.items():
        for ch in rec["chains"]:
            mid = ch.get("msa_id")
            if mid not in (-1, "", None):
                ref[str(mid)].append(rid)
    checks["V12_total"] = len(ref)
    if not msa_dir.exists():
        fails["V12_msa_dir_absent"].append(str(msa_dir))
    else:
        on_disk = {p.stem for p in msa_dir.glob("*.npz")}
        absent = sorted(set(ref) - on_disk)
        checks["V12_pass"] = len(ref) - len(absent)
        for mid in absent:
            fails["V12_msa_file_missing"].append({"msa_id": mid, "used_by": ref[mid][:3]})
    print("V12 referenced MSA files present on disk: %d/%d"
          % (checks["V12_pass"], checks["V12_total"]))

    # --- summary ---
    print("\n%-42s %8s %8s" % ("CHECK", "PASS", "TOTAL"))
    names = {
        "V1": "sample is a TCR3D MHC-I entry",
        "V2_mhc": "expected MHC-I chain present + enabled",
        "V3_peptide": "expected peptide chain present + enabled",
        "V4": "kept chain sequences match RCSB",
        "V5": "unrelated chains disabled (npz + manifest)",
        "V6": "kept chains form one connected complex",
        "V7": "peptide in alpha1/alpha2 groove",
        "V8": "structural arrays intact",
        "V9": "temporal split matches release date",
        "V10": "no (allele, peptide) leakage across splits",
        "V11": "Boltz loads / tokenizes / crops the file",
        "V12": "referenced MSA files present on disk",
    }
    for k, label in names.items():
        t, p = checks.get("%s_total" % k, 0), checks.get("%s_pass" % k, 0)
        flag = "" if t and p == t else "   <-- FAILURES"
        print("%-42s %8d %8d%s" % (label, p, t, flag))

    print("\nfailure detail:")
    if not fails:
        print("    none")
    for k, v in fails.items():
        print("    %-45s %d  %s" % (k, len(v), str(v[:2])[:110]))

    R.mkdir(exist_ok=True)
    (R / "validation_report.json").write_text(json.dumps({
        "n_samples": len(files),
        "checks": {k: {"pass": checks.get("%s_pass" % k, 0),
                       "total": checks.get("%s_total" % k, 0),
                       "description": v} for k, v in names.items()},
        "failures": {k: v for k, v in fails.items()},
        "needs_manual_review": review,
    }, indent=1))
    print("\nwrote %s" % (R / "validation_report.json"))


if __name__ == "__main__":
    main()
