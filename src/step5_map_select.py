"""Step 5: map Boltz NPZ chains to biological roles and select the pMHC-I system.

Chain-ID mapping
----------------
Boltz chain names are NOT author chain IDs. `scripts/process/rcsb.py` expands
assembly 1 with `gemmi.HowToNameCopiedChain.AddNumber`, giving names like `A1`
or `C2`. Two facts matter, both verified against RCSB on this dataset:

  * the letter part is the **label_asym_id**, not the auth_asym_id. Measured
    over 1152 protein chains the prefix matched label_asym_id 1152/1152 (100%)
    but auth_asym_id only 1010/1152 (87.7%). TCR3D and the literature quote
    auth IDs -- 1FZK's peptide is auth chain "P" but Boltz chain "C1" -- so
    name-matching on auth IDs silently picks the wrong chain ~12% of the time.
  * the trailing number is the *assembly-generation operator index*, not a copy
    index. In 7RRG, assembly 1 is one 5-chain complex emitted by two operator
    rows (oper 1 -> A,B,E; oper 2 -> C,D), so `A1` and `C2` are genuinely part
    of the same biological complex. Grouping chains by that suffix and keeping
    one group would tear real complexes apart.
  * a genuine second copy of a chain shows up as the same `entity_id` with a
    different `sym_id` (e.g. 3DTX: A1/B1/C1 and A2/B2/C2).

So mapping is done on *sequence*, and the MHC/peptide pairing is decided on
*geometry*. Names are only cross-checked and reported, never trusted.

Selection
---------
For every (MHC-I heavy, peptide) pair we count heavy-atom contacts <= 4.5 A and
require the peptide to touch both the alpha1 and alpha2 halves of the heavy
chain -- the signature of sitting in the binding groove rather than merely
brushing a neighbouring copy. The best-scoring pair wins, then the beta-2
microglobulin chain most in contact with that heavy chain is added. Everything
else is disabled. This makes cross-copy pairing impossible by construction.
"""
import collections
import json
import pathlib
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "boltz-src" / "src"))
sys.path.insert(0, str(ROOT / "src"))
from boltz.data.const import chain_types, prot_token_to_letter  # noqa: E402
from seqlib import sw_align, wildcard_identity  # noqa: E402

D = ROOT / "data"
CONTACT_CUTOFF = 4.5
MIN_GROOVE_CONTACTS = 20
A1_END = 90          # mature MHC-I heavy chain: alpha1 ~1-90, alpha2 ~91-180
A2_END = 190
KEEP_TCR = False     # the target system is pMHC-I; TCR chains are not part of it


def chain_seq(res_slice):
    return "".join(prot_token_to_letter.get(str(n), "X") for n in res_slice["name"])


def model0_coords(z):
    """Coordinates of the first ensemble model, aligned to the atom table."""
    atoms = z["atoms"]
    ens = z["ensemble"]
    if "coords" in z.files and len(z["coords"]) and len(ens):
        start = int(ens[0]["atom_coord_idx"])
        num = int(ens[0]["atom_num"])
        xyz = z["coords"]["coords"][start:start + num]
        if len(xyz) == len(atoms):
            return xyz
    return atoms["coords"]


def load_chains(pid, npz_path, entities):
    """Return per-chain dicts with sequence, role and atom index ranges."""
    z = np.load(npz_path)
    chains, residues, atoms = z["chains"], z["residues"], z["atoms"]
    xyz = model0_coords(z)
    present = atoms["is_present"].astype(bool)

    # sequence -> role, from the RCSB annotation
    by_seq = {}
    for e in entities:
        if e["seq"]:
            by_seq[e["seq"]] = e
    auth_ids = {a for e in entities for a in e["auth_asym_ids"]}

    out = []
    for i, c in enumerate(chains):
        rs = residues[int(c["res_idx"]):int(c["res_idx"]) + int(c["res_num"])]
        seq = chain_seq(rs)
        name = str(c["name"])
        m = re.match(r"^(.*?)(\d+)$", name)
        auth_guess, oper = (m.group(1), m.group(2)) if m else (name, None)
        role, ent, how = "OTHER", None, "unmatched"
        if chain_types[int(c["mol_type"])] != "PROTEIN":
            role, how = "LIGAND_OR_NA", "mol_type=%s" % chain_types[int(c["mol_type"])]
        elif seq in by_seq:
            ent = by_seq[seq]
            role, how = ent["role"], "exact sequence match to entity %s" % ent["entity_id"]
        else:
            # modified residues decode to 'X'; match on the resolvable positions
            wc = [(wildcard_identity(seq, e["seq"]), e) for e in entities
                  if e["polymer_type"] == "Protein" and e["seq"]]
            wc = [(v, e) for v, e in wc if v is not None and v >= 0.9]
            wc.sort(key=lambda t: -t[0])
            if wc:
                ent = wc[0][1]
                role = ent["role"]
                how = ("wildcard match to entity %s (non-X identity=%.2f, %d X positions)"
                       % (ent["entity_id"], wc[0][0], seq.count("X")))
            best, bent = 0.0, None
            for e in entities:
                if e["polymer_type"] != "Protein" or not e["seq"]:
                    continue
                if abs(len(e["seq"]) - len(seq)) > max(60, 0.5 * len(seq)):
                    continue
                _, idt, alen = sw_align(seq[:400], e["seq"][:400])
                cov = alen / max(1, min(len(seq), len(e["seq"])))
                score = idt * cov
                if score > best:
                    best, bent = score, e
            if ent is None and bent is not None and best >= 0.80:
                ent = bent
                role = bent["role"]
                how = "SW alignment to entity %s (score=%.2f)" % (bent["entity_id"], best)
        a0 = int(c["atom_idx"])
        a1 = a0 + int(c["atom_num"])
        out.append({
            "idx": i, "name": name, "auth_guess": auth_guess, "oper": oper,
            "auth_in_rcsb": auth_guess in auth_ids,
            "mol_type": chain_types[int(c["mol_type"])],
            "entity_id_npz": int(c["entity_id"]), "sym_id": int(c["sym_id"]),
            "asym_id": int(c["asym_id"]), "n_res": int(c["res_num"]),
            "seq": seq, "role": role, "match": how,
            "rcsb_entity_id": ent["entity_id"] if ent else None,
            "rcsb_auth_asym_ids": ent["auth_asym_ids"] if ent else [],
            "_a0": a0, "_a1": a1,
            "_res_of_atom": np.repeat(np.arange(len(rs)), rs["atom_num"].astype(int)),
        })
    return z, out, xyz, present


def contacts(ch_a, ch_b, xyz, present):
    """(n_contacts, residue indices of ch_a involved) for heavy atoms <= cutoff."""
    ia = np.arange(ch_a["_a0"], ch_a["_a1"])[present[ch_a["_a0"]:ch_a["_a1"]]]
    ib = np.arange(ch_b["_a0"], ch_b["_a1"])[present[ch_b["_a0"]:ch_b["_a1"]]]
    if not len(ia) or not len(ib):
        return 0, set()
    ta, tb = cKDTree(xyz[ia]), cKDTree(xyz[ib])
    pairs = ta.query_ball_tree(tb, CONTACT_CUTOFF)
    n = 0
    res_a = set()
    rmap = ch_a["_res_of_atom"]
    for k, hits in enumerate(pairs):
        if hits:
            n += len(hits)
            res_a.add(int(rmap[ia[k] - ch_a["_a0"]]))
    return n, res_a


def select(pid, chains, xyz, present):
    """Pick the biologically relevant pMHC-I chain set. Returns (keep, info)."""
    mhcs = [c for c in chains if c["role"] == "MHC_I_HEAVY"]
    peps = [c for c in chains if c["role"] == "PEPTIDE"]
    b2ms = [c for c in chains if c["role"] == "B2M"]
    info = {"n_mhc_chains": len(mhcs), "n_pep_chains": len(peps), "n_b2m_chains": len(b2ms),
            "pair_scores": [], "flags": []}
    if not mhcs or not peps:
        info["flags"].append("no_mhc_or_peptide_chain_in_npz")
        return None, info

    best = None
    for m in mhcs:
        for p in peps:
            n, res_m = contacts(m, p, xyz, present)
            a1 = sum(1 for r in res_m if r < A1_END)
            a2 = sum(1 for r in res_m if A1_END <= r < A2_END)
            groove = a1 > 0 and a2 > 0
            info["pair_scores"].append({
                "mhc": m["name"], "pep": p["name"], "contacts": n,
                "alpha1_res": a1, "alpha2_res": a2, "in_groove": bool(groove),
                "same_oper": m["oper"] == p["oper"],
            })
            cand = (groove, n)
            if best is None or cand > best[0]:
                best = (cand, m, p)
    (groove, ncon), mhc, pep = best
    if ncon < MIN_GROOVE_CONTACTS:
        info["flags"].append("peptide_mhc_contacts_below_threshold(%d)" % ncon)
    if not groove:
        info["flags"].append("peptide_not_spanning_alpha1_alpha2")
    if len(mhcs) > 1 or len(peps) > 1:
        info["flags"].append("multiple_mhc_or_peptide_copies_resolved_by_contact")
        # A runner-up with zero contacts is a different copy and the choice is
        # unambiguous. A runner-up that is also in the groove with a comparable
        # contact count is a real judgement call (two peptides in one groove, or
        # partial occupancy) and belongs on the manual-review list.
        rivals = sorted((s for s in info["pair_scores"]
                         if not (s["mhc"] == mhc["name"] and s["pep"] == pep["name"])),
                        key=lambda s: -s["contacts"])
        if rivals and rivals[0]["in_groove"] and rivals[0]["contacts"] > 0.5 * ncon:
            info["flags"].append("ambiguous_peptide_choice_needs_review")
    if mhc["oper"] != pep["oper"]:
        info["flags"].append("selected_pair_spans_operators(%s,%s)" % (mhc["oper"], pep["oper"]))

    keep = [mhc["idx"], pep["idx"]]
    b2m_pick = None
    if b2ms:
        scored = [(contacts(mhc, b, xyz, present)[0], b) for b in b2ms]
        scored.sort(key=lambda t: -t[0])
        if scored[0][0] > 0:
            b2m_pick = scored[0][1]
            keep.append(b2m_pick["idx"])
        else:
            info["flags"].append("b2m_present_but_not_contacting_selected_mhc")
    else:
        info["flags"].append("no_b2m_chain")

    if KEEP_TCR:
        for c in chains:
            if c["role"].startswith("TCR") and contacts(mhc, c, xyz, present)[0] > 0:
                keep.append(c["idx"])

    info.update({
        "selected_mhc": mhc["name"], "selected_peptide": pep["name"],
        "selected_b2m": b2m_pick["name"] if b2m_pick else None,
        "mhc_pep_contacts": ncon, "mhc_pep_in_groove": bool(groove),
        "peptide_seq": pep["seq"], "mhc_seq": mhc["seq"],
        "n_chains_total": len(chains), "n_chains_kept": len(keep),
    })
    return sorted(keep), info


def main():
    cands = {r["pdb_id"]: r for r in json.loads((D / "candidates.json").read_text())}
    ann = json.loads((D / "annotated_entities.json").read_text())
    npz_dir = D / "npz_raw"
    avail = sorted(p for p in npz_dir.glob("*.npz") if p.stem.upper() in cands)
    print("candidates with an NPZ present: %d / %d" % (len(avail), len(cands)))

    results, name_agree, name_total = {}, 0, 0
    for k, path in enumerate(avail):
        pid = path.stem.upper()
        try:
            z, chains, xyz, present = load_chains(pid, path, ann[pid]["entities"])
            keep, info = select(pid, chains, xyz, present)
        except Exception as exc:  # keep going, record the failure
            results[pid] = {"error": "%s: %s" % (type(exc).__name__, exc)}
            continue
        for c in chains:
            if c["mol_type"] == "PROTEIN" and c["rcsb_auth_asym_ids"]:
                name_total += 1
                if c["auth_guess"] in c["rcsb_auth_asym_ids"]:
                    name_agree += 1
        results[pid] = {
            "pdb_id": pid, "split": cands[pid]["split"], "keep_chain_idx": keep,
            "chains": [{kk: vv for kk, vv in c.items() if not kk.startswith("_")}
                       for c in chains],
            **info,
        }
        if (k + 1) % 50 == 0:
            print("  processed %d/%d" % (k + 1, len(avail)), flush=True)

    ok = [r for r in results.values() if r.get("keep_chain_idx")]
    err = {p: r for p, r in results.items() if "error" in r}
    print("\nmapped and selected : %d" % len(ok))
    print("errors              : %d %s" % (len(err), list(err.items())[:3]))
    print("no selection made   : %d" % sum(1 for r in results.values()
                                           if "error" not in r and not r.get("keep_chain_idx")))
    print("\nauth-ID name cross-check: %d/%d protein chains had a name prefix matching an "
          "RCSB auth_asym_id (%.1f%%)" % (name_agree, name_total,
                                          100.0 * name_agree / max(1, name_total)))
    fl = collections.Counter(f for r in results.values() for f in r.get("flags", []))
    print("\nflags raised:")
    for f, n in fl.most_common():
        print("    %-58s %d" % (f, n))
    ncon = [r["mhc_pep_contacts"] for r in ok]
    if ncon:
        ncon.sort()
        print("\nMHC-peptide contact counts: min=%d p05=%d median=%d max=%d"
              % (ncon[0], ncon[int(0.05 * (len(ncon) - 1))], ncon[len(ncon) // 2], ncon[-1]))
    kept = collections.Counter(r["n_chains_kept"] for r in ok)
    print("chains kept per sample: %s" % dict(sorted(kept.items())))
    tot = collections.Counter(r["n_chains_total"] for r in ok)
    print("chains originally present: median=%d max=%d"
          % (sorted(r["n_chains_total"] for r in ok)[len(ok) // 2], max(tot)))

    (D / "chain_selection.json").write_text(json.dumps(results, indent=1))
    print("\nwrote %s" % (D / "chain_selection.json"))


if __name__ == "__main__":
    main()
