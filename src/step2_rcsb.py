"""Step 2: fetch authoritative RCSB metadata (release date, entities, chain IDs, assemblies)."""
import json, pathlib, sys, time, urllib.request, urllib.error

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GQL = "https://data.rcsb.org/graphql"

QUERY = """
query($ids:[String!]!){
  entries(entry_ids:$ids){
    rcsb_id
    rcsb_accession_info { initial_release_date deposit_date revision_date }
    exptl { method }
    rcsb_entry_info { resolution_combined deposited_polymer_entity_instance_count
                      deposited_nonpolymer_entity_instance_count assembly_count }
    assemblies {
      rcsb_id
      rcsb_assembly_info { assembly_id polymer_entity_instance_count }
      pdbx_struct_assembly { oligomeric_count details method_details }
      pdbx_struct_assembly_gen { assembly_id asym_id_list oper_expression }
    }
    polymer_entities {
      rcsb_id
      entity_poly { pdbx_seq_one_letter_code_can type rcsb_entity_polymer_type }
      rcsb_polymer_entity { pdbx_description formula_weight }
      rcsb_polymer_entity_container_identifiers { asym_ids auth_asym_ids entity_id }
      rcsb_entity_source_organism { ncbi_scientific_name }
      rcsb_polymer_entity_align { reference_database_accession reference_database_name }
    }
    nonpolymer_entities {
      rcsb_id
      nonpolymer_comp { chem_comp { id name formula } }
      rcsb_nonpolymer_entity_container_identifiers { asym_ids auth_asym_ids entity_id }
    }
  }
}
"""

def post(ids, tries=4):
    body = json.dumps({"query": QUERY, "variables": {"ids": ids}}).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request(GQL, data=body,
                headers={"Content-Type": "application/json", "User-Agent": "mhc1-boltz-prep/1.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read())
            if "errors" in out and not out.get("data"):
                raise RuntimeError(str(out["errors"])[:300])
            return out["data"]["entries"]
        except Exception as e:
            if t == tries - 1: raise
            time.sleep(3 * (t + 1))

rows = json.loads((DATA / "tcr3d_mhc1_raw.json").read_text(encoding="utf-8"))
ids = sorted({r["pdbid"].upper() for r in rows})
print(f"fetching RCSB metadata for {len(ids)} entries")

cache = DATA / "rcsb_meta.json"
meta = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
todo = [i for i in ids if i not in meta]
print(f"already cached: {len(meta)}, to fetch: {len(todo)}")

B = 40
for k in range(0, len(todo), B):
    batch = todo[k:k+B]
    ents = post(batch)
    found = 0
    for e in ents:
        if e: meta[e["rcsb_id"].upper()] = e; found += 1
    cache.write_text(json.dumps(meta), encoding="utf-8")
    print(f"  [{k+len(batch):>5}/{len(todo)}] +{found}", flush=True)

missing = [i for i in ids if i not in meta]
print(f"DONE. cached={len(meta)}  missing={len(missing)} {missing[:20]}")
