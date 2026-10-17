import json,glob,statistics as S,collections as C
q=json.load(open("data/peptide_quality.json"))
def nres(i):
    e=q.get(i.upper()) or q.get(i) or {}
    return e.get("pep_len") or (e.get("peptide") or {}).get("n_res")
def load(p):
    out={}
    for f in glob.glob(p+"/*.jsonl"):
        for l in open(f):
            r=json.loads(l)
            if not r.get("sanity"): out[r["record_id"]]=r
    return out
runs={"pretrained":"runs/eval_val99/pretrained","pre_q1":"runs/eval_msa/msa1",
      "b9_full":"runs/eval_val99/b9_3e-4_last","abrupt_q1":"runs/eval_val99/abrupt_last_d1"}
D={k:load(v) for k,v in runs.items()}
ids=sorted(D["pretrained"])
def bucket(n):
    if n is None: return "?"
    if n<=7: return "<=7"
    if n<=9: return "8-9"
    if n<=11: return "10-11"
    return ">=12"
g=C.defaultdict(list)
for i in ids: g[bucket(nres(i))].append(i)
print("median peptide RMSD by peptide length (residues, from peptide_quality.json)")
print("%-7s %4s %9s %9s %9s %9s"%("len","n","pretrain","pre_q1","b9_full","abrupt_q1"))
for b in ["<=7","8-9","10-11",">=12","?"]:
    if not g[b]: continue
    row=[S.median([D[k][i]["pep_rmsd_mean"] for i in g[b] if i in D[k]]) for k in runs]
    print("%-7s %4d %9.3f %9.3f %9.3f %9.3f"%(b,len(g[b]),*row))
print()
print("the long/short tail, per record:")
for i in sorted([x for x in ids if bucket(nres(x)) in ("<=7",">=12")],key=lambda x:(nres(x) or 0)):
    print("  %-6s %2dmer  pretrained %6.2f -> b9_full %5.2f  abrupt_q1 %5.2f"%(
      i,nres(i),D["pretrained"][i]["pep_rmsd_mean"],
      D["b9_full"][i]["pep_rmsd_mean"],D["abrupt_q1"][i]["pep_rmsd_mean"]))
