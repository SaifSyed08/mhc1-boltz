"""Export processed samples to PDB text for 3D viewing.

Purpose is not decoration. Two of the pipeline's claims are geometric, and the
only honest way to check them is to look:

  * V7 -- the peptide sits in the MHC-I alpha1/alpha2 groove;
  * section 4 of the brief -- the chains we kept are the biologically relevant
    pMHC-I system and the ones we dropped are crystallographic company.

So each structure is exported with EVERY chain from the original Boltz sample,
and a sidecar JSON says which ones the pipeline kept and what role it assigned.
The viewer greys out what was dropped, which makes a wrong selection obvious at a
glance rather than something you have to take on trust.

Encoding notes for the Boltz NPZ format:
  * `atoms["name"]` is 4 int8s, each `ord(c) - 32`, zero-padded.
  * `atoms["element"]` is the atomic number.
  * Coordinates come from `coords["coords"]` indexed via `ensemble[0]`, not from
    `atoms["coords"]`, whenever an ensemble is present.
"""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "npz_raw"
OUT = ROOT / "data" / "viz"

ELEMENTS = {1: "H", 6: "C", 7: "N", 8: "O", 15: "P", 16: "S",
            12: "MG", 20: "CA", 26: "FE", 30: "ZN", 34: "SE"}


def atom_name(arr):
    return "".join(chr(v + 32) for v in arr if v > 0).strip()


# A PDB chain id is a single character, but Boltz names chains "A1", "A2", ...
# so taking the first character collapses distinct chains together -- fatal when a
# sample has 41 of them. Assign a unique id per chain index instead and publish the
# mapping in index.json.
CHAIN_IDS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789")
BACKBONE = {"N", "CA", "C", "O"}


def to_pdb(npz_path, keep_idx):
    z = np.load(npz_path)
    chains, residues, atoms = z["chains"], z["residues"], z["atoms"]
    present = atoms["is_present"].astype(bool)
    ens = z["ensemble"]
    if len(ens) and len(z["coords"]) == len(atoms):
        c0 = int(ens[0]["atom_coord_idx"])
        xyz = z["coords"]["coords"][c0:c0 + int(ens[0]["atom_num"])]
    else:
        xyz = atoms["coords"]

    lines, serial, cid_map = [], 1, {}
    for ci, ch in enumerate(chains):
        if ci >= len(CHAIN_IDS):
            continue
        cid = CHAIN_IDS[ci]
        cid_map[ci] = cid
        kept = ci in keep_idx
        r0, rn = int(ch["res_idx"]), int(ch["res_num"])
        for ri in range(r0, r0 + rn):
            res = residues[ri]
            resname = str(res["name"])[:3]
            a0, an = int(res["atom_idx"]), int(res["atom_num"])
            for ai in range(a0, a0 + an):
                if not present[ai]:
                    continue
                nm = atom_name(atoms["name"][ai])
                # Discarded chains are only ever drawn as a thin backbone trace,
                # so their side chains are dead weight -- dropping them cuts the
                # 41-chain samples by roughly two thirds.
                if not kept and nm not in BACKBONE:
                    continue
                el = ELEMENTS.get(int(atoms["element"][ai]), "C")
                x, y, zc = (float(v) for v in xyz[ai])
                # Column-exact PDB ATOM record; 3Dmol is tolerant but a
                # misaligned element column silently breaks colouring.
                nm4 = (" " + nm).ljust(4) if len(nm) < 4 else nm[:4]
                lines.append(
                    "ATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s"
                    % (serial, nm4, resname, cid, int(res["res_idx"]) + 1,
                       x, y, zc, el)
                )
                serial += 1
        lines.append("TER")
    lines.append("END")
    return "\n".join(lines), cid_map


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((ROOT / "data" / "chain_selection.json").read_text())
    cands = {r["pdb_id"]: r for r in
             json.loads((ROOT / "data" / "candidates.json").read_text())}
    val_ids = [l.strip() for l in
               open(ROOT / "data" / "processed" / "validation_ids.txt") if l.strip()]

    # Prefer validation structures (the ones the baseline scored), and prefer
    # samples that actually had chains to discard -- those show the selection
    # doing something.
    scored = []
    for pid in val_ids:
        info = sel.get(pid.upper())
        if not info or not (RAW / f"{pid}.npz").exists():
            continue
        dropped = len(info["chains"]) - len(info["keep_chain_idx"])
        scored.append((dropped, pid, info))
    scored.sort(reverse=True)

    picked, manifest = [], []
    for dropped, pid, info in scored:
        if len(picked) >= 6:
            break
        try:
            pdb, cid_map = to_pdb(RAW / f"{pid}.npz", set(info["keep_chain_idx"]))
        except Exception as exc:
            print("  skip %s: %s" % (pid, exc))
            continue
        (OUT / f"{pid}.pdb").write_text(pdb)
        c = cands.get(pid.upper(), {})
        keep = set(info["keep_chain_idx"])
        manifest.append({
            "pdb_id": pid,
            "allele": c.get("mhc_allele"),
            "peptide": c.get("peptide_seq_struct"),
            "released": c.get("release_date"),
            "n_chains_total": len(info["chains"]),
            "n_chains_kept": len(keep),
            "chains": [
                {"name": ch["name"],
                 "pdb_chain": cid_map.get(ch["idx"]),
                 "role": ch["role"] if ch["idx"] in keep else "DROPPED",
                 "kept": ch["idx"] in keep}
                for ch in info["chains"]
            ],
        })
        picked.append(pid)
        print("  %-6s %2d chains, %d kept, %s" %
              (pid, len(info["chains"]), len(keep), c.get("mhc_allele")))

    (OUT / "index.json").write_text(json.dumps(manifest, indent=1))
    print("\nwrote %d structures to %s" % (len(picked), OUT))


if __name__ == "__main__":
    main()
