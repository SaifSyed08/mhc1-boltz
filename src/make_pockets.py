"""Pocket conditioning for validation: a NetMHCpan-style partial template.

Why this exists
---------------
MHC-Fine leaned on templates. Boltz-1 v1.0.0 has no template input, but it does
take a pocket: one chain marked BINDER, a set of residues marked POCKET (the
`pocket_feature` the trunk embeds; trained with pocket-conditioned samples). So
the anchor idea can be tested at inference with no retraining. Two pocket sets,
for every validation record, binder = the peptide:

  pseudoseq  NetMHCpan's MHC pseudo-sequence: the 34 heavy-chain positions
             (mature HLA numbering) that contact the peptide across known
             structures. Mapped onto each heavy chain by global alignment to
             HLA-A*02:01. Uses only the sequence -> a fair, deployable input.
  oracle     heavy-chain residues with any atom within 6 A of the peptide in the
             crystal (6 A = Boltz's own binder_pocket_cutoff). Uses the answer ->
             an upper bound on what any pocket/anchor template could give.

Output: data/pockets_{pseudoseq,oracle}.json, record id ->
  {"binder": [peptide asym_id], "pocket": [[heavy-chain asym_id, res_idx], ...]}
with asym_id / res_idx exactly as the tokenizer sees them (structure npz chain
asym_id, 0-based residue index within the chain). Read in validation through
VAL_POCKET_FILE (boltz-src data/module/training.py local patch).

  python src/make_pockets.py
"""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

# HLA-A*02:01 mature sequence (IMGT), positions 1-181. Checked: these positions
# give NetMHCpan's A*02:01 pseudo-sequence YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY.
A0201 = ("GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLG"
         "TLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLR"
         "AYLEGTCVEWLRRYLENGKETLQRT")
PSEUDO = [7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97,
          99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171]
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
         "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
         "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def align(a, b, match=2, mismatch=-1, gap=-3):
    """Global alignment; returns {index in a: index in b} for aligned pairs."""
    n, m = len(a), len(b)
    S = np.zeros((n + 1, m + 1), dtype=np.int32)
    S[:, 0] = gap * np.arange(n + 1)
    S[0, :] = gap * np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if a[i - 1] == b[j - 1] else mismatch
            S[i, j] = max(S[i - 1, j - 1] + s, S[i - 1, j] + gap, S[i, j - 1] + gap)
    i, j, pairs = n, m, {}
    while i > 0 and j > 0:
        s = match if a[i - 1] == b[j - 1] else mismatch
        if S[i, j] == S[i - 1, j - 1] + s:
            pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif S[i, j] == S[i - 1, j] + gap:
            i -= 1
        else:
            j -= 1
    return pairs


def chain_view(s, name):
    ch = next(c for c in s["chains"] if str(c["name"]) == name)
    res = s["residues"][ch["res_idx"]:ch["res_idx"] + ch["res_num"]]
    return ch, res


def main():
    sel = json.loads((ROOT / "data" / "chain_selection.json").read_text())
    ids = (ROOT / "data" / "processed" / "validation_ids.txt").read_text().split()
    pseudo, oracle, report = {}, {}, []
    for rid in ids:
        e = sel[rid.upper()]
        s = np.load(ROOT / "data" / "processed" / "structures" / f"{rid}.npz")
        hc, hres = chain_view(s, e["selected_mhc"])
        pc, pres = chain_view(s, e["selected_peptide"])
        binder = [int(pc["asym_id"])]

        # pseudo-sequence via alignment to A*02:01
        seq = "".join(THREE.get(str(r["name"]), "X") for r in hres)
        pairs = align(A0201, seq)
        pos = [pairs[p - 1] for p in PSEUDO if (p - 1) in pairs]
        ident = sum(A0201[i] == seq[j] for i, j in pairs.items()) / len(A0201)
        pseudo[rid] = {"binder": binder,
                       "pocket": [[int(hc["asym_id"]), int(hres[k]["res_idx"])] for k in pos]}

        # oracle: heavy-chain residues within 6 A of the peptide in the crystal
        atoms = s["atoms"]
        pa = np.concatenate([atoms[r["atom_idx"]:r["atom_idx"] + r["atom_num"]] for r in pres])
        pxyz = pa["coords"][pa["is_present"].astype(bool)]
        near = []
        for r in hres:
            a = atoms[r["atom_idx"]:r["atom_idx"] + r["atom_num"]]
            xyz = a["coords"][a["is_present"].astype(bool)]
            if len(xyz) and len(pxyz) and np.min(np.linalg.norm(xyz[:, None] - pxyz[None], axis=-1)) < 6.0:
                near.append([int(hc["asym_id"]), int(r["res_idx"])])
        oracle[rid] = {"binder": binder, "pocket": near}

        overlap = len({tuple(x) for x in near} & {tuple(x) for x in pseudo[rid]["pocket"]})
        report.append((rid, len(pos), round(ident, 2), len(near), overlap))

    (ROOT / "data" / "pockets_pseudoseq.json").write_text(json.dumps(pseudo))
    (ROOT / "data" / "pockets_oracle.json").write_text(json.dumps(oracle))
    lows = [r for r in report if r[1] < 34 or r[2] < 0.6]
    print(f"{len(report)} records. pseudo positions mapped: min {min(r[1] for r in report)}; "
          f"identity to A*02:01 min {min(r[2] for r in report)}, median "
          f"{sorted(r[2] for r in report)[len(report) // 2]}")
    print(f"oracle pocket size median {sorted(r[3] for r in report)[len(report) // 2]}; "
          f"pseudo-in-oracle overlap median {sorted(r[4] for r in report)[len(report) // 2]}/34")
    for r in lows:
        print("  check:", r)


if __name__ == "__main__":
    main()
