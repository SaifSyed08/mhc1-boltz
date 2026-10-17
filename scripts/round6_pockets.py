import json,glob,statistics as S,os
from math import comb
def load(p):
    out={}
    for f in glob.glob(p+"/*.jsonl"):
        for l in open(f):
            r=json.loads(l)
            if r.get("sanity"): continue
            out[r["record_id"]]=r
    return out
def p2(k,n):
    if n==0: return 1.0
    o=comb(n,k); return min(1.0,sum(comb(n,i) for i in range(n+1) if comb(n,i)<=o)/2**n)
print("eval_pocket:",sorted(os.listdir("runs/eval_pocket")) if os.path.isdir("runs/eval_pocket") else "none")
base=load("runs/eval_val99/pretrained")
cands={"smoke_pocket_pseudoseq":"runs/r6_smoke_pocket"}
for d in sorted(glob.glob("runs/eval_pocket/*")):
    cands[os.path.basename(d)]=d
for nm,p in cands.items():
    a=load(p)
    if not a: print(nm,"-> no rows"); continue
    ids=[i for i in a if i in base]
    diff=[a[i]["pep_rmsd_mean"]-base[i]["pep_rmsd_mean"] for i in ids]
    imp=sum(x<0 for x in diff)
    print("%-26s n=%2d  median %.3f (base %.3f)  paired %+.3f  improved %d/%d p=%.3f"%(
      nm,len(ids),S.median([a[i]["pep_rmsd_mean"] for i in ids]),
      S.median([base[i]["pep_rmsd_mean"] for i in ids]),S.median(diff),imp,len(ids),p2(imp,len(ids))))
