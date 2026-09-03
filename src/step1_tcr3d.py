"""Step 1: scrape the TCR3D MHC-I chain list into a normalised table."""
import json, re, sys, pathlib, collections

SCRATCH = pathlib.Path(r"C:\Users\SAIFSY~1\AppData\Local\Temp\claude\C--Users-Saif-Syed\9449de71-3c90-4f48-8af5-dbef96bae4cb\scratchpad")
OUT = pathlib.Path(__file__).resolve().parents[1] / "data"

html = (SCRATCH / "tcr3d.html").read_text(encoding="utf-8", errors="replace")
m = re.search(r"var data = (\[.*?\]);", html, re.S)
assert m, "could not find embedded data array"
rows = json.loads(m.group(1))
print(f"raw TCR3D rows: {len(rows)}")

keys = collections.Counter()
for r in rows: keys.update(r.keys())
print("fields:", dict(keys))

cls = collections.Counter(r.get("mhc_class") for r in rows)
print("mhc_class:", dict(cls))
print("species:", dict(collections.Counter(r.get("mhc_species") for r in rows).most_common(10)))
print("is_bound:", dict(collections.Counter(r.get("is_bound") for r in rows)))

# unique PDB IDs
pdbs = [r["pdbid"].upper() for r in rows]
print(f"unique PDB IDs: {len(set(pdbs))}  (rows {len(pdbs)})")
dupe = {k:v for k,v in collections.Counter(pdbs).items() if v>1}
print(f"PDB IDs with >1 row: {len(dupe)}", dict(list(dupe.items())[:10]))

# peptide sanity
nopep = [r["pdbid"] for r in rows if not r.get("peptide") or r["peptide"] in ("---","")]
print(f"rows with no peptide seq: {len(nopep)} e.g. {nopep[:10]}")
plens = collections.Counter(len(r["peptide"]) for r in rows if r.get("peptide"))
print("peptide lengths:", dict(sorted(plens.items())))

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tcr3d_mhc1_raw.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
print("wrote", OUT / "tcr3d_mhc1_raw.json")
