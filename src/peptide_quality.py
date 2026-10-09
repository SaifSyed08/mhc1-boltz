"""How trustworthy is each record's peptide? wwPDB validation numbers, per record.

Why this exists
---------------
Ernest (2026-10-12): some PDB samples are low quality and may be worth dropping;
sort the peptides into "can improve", "particularly difficult" and "drop". A
prediction can only be judged against a peptide the crystal actually resolved, so
the first input is how well the deposited peptide fits its electron density.

What it fetches (RCSB Data API, GraphQL, read-only)
---------------------------------------------------
Per entry:   resolution, R-free, clashscore, % RSRZ outliers (whole entry).
Per selected peptide chain (the one in data/chain_selection.json):
  rscc        real-space correlation coefficient per residue; < 0.8 is a poor
              fit to density, the usual cut in the wwPDB report
  rsrz        real-space R Z-score per residue; > 2 is a wwPDB outlier
  owab        occupancy-weighted B-factor per residue
  occupancy   residues modelled at partial occupancy
  unobserved  residues in the sequence with no coordinates
  covalent    ligands covalently linked to the peptide (e.g. MYR on 7wj3's
              lipopeptide: the model never sees that lipid, so it cannot place
              a peptide whose position depends on it)
Entries with no structure factors (mostly pre-2008) have no RSCC/RSRZ; those
fields are null, not zero.

  python src/peptide_quality.py            # all 1,084 records -> data/peptide_quality.json
"""
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
GQL = "https://data.rcsb.org/graphql"
OUT = ROOT / "data" / "peptide_quality.json"

Q_ENTRY = """query($ids:[String!]!){ entries(entry_ids:$ids){ rcsb_id
  rcsb_entry_info { resolution_combined }
  refine { ls_R_factor_R_free }
  pdbx_vrpt_summary_geometry { clashscore }
  pdbx_vrpt_summary_diffraction { percent_RSRZ_outliers } } }"""

Q_INST = """query($ids:[String!]!){ polymer_entity_instances(instance_ids:$ids){ rcsb_id
  rcsb_polymer_instance_feature { type name feature_positions { beg_seq_id end_seq_id values } } } }"""


def gql(query, ids, tries=4):
    body = json.dumps({"query": query, "variables": {"ids": ids}}).encode()
    for i in range(tries):
        try:
            req = urllib.request.Request(GQL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
            if d.get("errors"):
                raise RuntimeError(d["errors"][0].get("message"))
            return d["data"]
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            if i == tries - 1:
                raise
            print(f"  retry {i + 1}: {exc}")
            time.sleep(2 ** i)


def first(xs, key):
    return (xs or [{}])[0].get(key) if xs else None


def per_residue(features, ftype):
    """Values of a per-residue feature, in sequence order, or None if absent."""
    vals = []
    for f in features:
        if f["type"] == ftype:
            for p in f["feature_positions"]:
                if p.get("values"):
                    vals += p["values"]
    return vals or None


def summarize(features, n_res):
    rscc = per_residue(features, "RSCC")
    rsrz = per_residue(features, "RSRZ")
    owab = per_residue(features, "OWAB")
    occ = per_residue(features, "AVERAGE_OCCUPANCY")
    unobs = 0
    for f in features:
        if f["type"] == "UNOBSERVED_RESIDUE_XYZ":
            for p in f["feature_positions"]:
                unobs += (p.get("end_seq_id") or p["beg_seq_id"]) - p["beg_seq_id"] + 1
    cov = sorted({f["name"].replace("ligand ", "") for f in features
                  if f["type"] == "LIGAND_COVALENT_LINKAGE"})
    return {
        "rscc": rscc,
        "rscc_mean": round(sum(rscc) / len(rscc), 3) if rscc else None,
        "rscc_min": min(rscc) if rscc else None,
        "n_rscc_below_0_8": sum(v < 0.8 for v in rscc) if rscc else None,
        "rsrz": rsrz,
        "n_rsrz_outliers": sum(v > 2 for v in rsrz) if rsrz else None,
        "owab_mean": round(sum(owab) / len(owab), 1) if owab else None,
        "n_partial_occupancy": sum(v < 1 for v in occ) if occ else None,
        "n_unobserved": unobs,
        "n_res": n_res,
        "covalent_ligands": cov,
    }


def main():
    sel = json.loads((ROOT / "data" / "chain_selection.json").read_text())
    want = {}
    for pdb, e in sel.items():
        if e.get("split") not in ("train", "validation", "test") or not e.get("selected_peptide"):
            continue
        ch = next(c for c in e["chains"] if c["name"] == e["selected_peptide"])
        label = ch["name"][: len(ch["name"]) - len(str(ch.get("oper", "")))] if ch.get("oper") else ch["name"]
        want[pdb] = {"split": e["split"], "pep_chain": ch["name"], "label_asym": label,
                     "pep_seq": ch["seq"], "pep_len": ch["n_res"]}
    ids = sorted(want)
    print(f"{len(ids)} records")

    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        for e in gql(Q_ENTRY, chunk)["entries"]:
            w = want[e["rcsb_id"]]
            rc = (e.get("rcsb_entry_info") or {}).get("resolution_combined")
            w["resolution"] = rc[0] if rc else None
            w["r_free"] = first(e.get("refine"), "ls_R_factor_R_free")
            w["clashscore"] = first(e.get("pdbx_vrpt_summary_geometry"), "clashscore")
            w["pct_rsrz_outliers_entry"] = first(e.get("pdbx_vrpt_summary_diffraction"), "percent_RSRZ_outliers")
        inst = [f"{p}.{want[p]['label_asym']}" for p in chunk]
        got = {x["rcsb_id"]: x for x in gql(Q_INST, inst)["polymer_entity_instances"] or []}
        for p in chunk:
            x = got.get(f"{p}.{want[p]['label_asym']}")
            if x is None:
                want[p]["peptide"] = None
                print(f"  {p}: no instance {want[p]['label_asym']}")
                continue
            want[p]["peptide"] = summarize(x["rcsb_polymer_instance_feature"] or [], want[p]["pep_len"])
        print(f"  {min(i + 50, len(ids))}/{len(ids)}")

    OUT.write_text(json.dumps(want, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
