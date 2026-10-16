import json,glob,pathlib,statistics as S
from math import comb
def load(p):
    out={}
    for f in glob.glob(p+"/*.jsonl"):
        for l in open(f):
            r=json.loads(l)
            if r.get("sanity"): continue
            out[r["record_id"]]=r
    return out
def p_two(k,n):
    if n==0: return 1.0
    obs=comb(n,k); return min(1.0,sum(comb(n,i) for i in range(n+1) if comb(n,i)<=obs)/2**n)
sub=set(pathlib.Path("data/processed/subset30_ids.txt").read_text().split())
runs={
 "pretrained_fullMSA":"runs/eval_val99/pretrained",
 "pretrained_query1":"runs/eval_msa/msa1",
 "b9_3e-4_fullMSA":"runs/eval_val99/b9_3e-4_last",
 "b9_3e-4_query1":"runs/eval_val99/b9_3e-4_last_d1",
 "taper_query1":"runs/eval_val99/taper_last_d1",
 "abrupt_query1":"runs/eval_val99/abrupt_last_d1",
}
D={k:load(v) for k,v in runs.items()}
print("n rows:",{k:len(v) for k,v in D.items()})
print()
print("%-22s %7s %7s %7s %8s"%("run","median","sample0","mean","lddt_pp"))
for k,d in D.items():
    ids=sorted(d)
    print("%-22s %7.3f %7.3f %7.3f %8.4f"%(k,
      S.median([d[i]["pep_rmsd_mean"] for i in ids]),
      S.median([d[i]["pep_rmsd"][0] for i in ids]),
      S.mean([d[i]["pep_rmsd_mean"] for i in ids]),
      S.mean([S.mean(d[i]["lddt_pp"]) for i in ids])))
print()
def cmp(a,b,label):
    A,B=D[a],D[b]; ids=[i for i in A if i in B]
    ho=[i for i in ids if i not in sub]
    for nm,S_ in [("all %d"%len(ids),ids),("held-out %d"%len(ho),ho)]:
        diff=[A[i]["pep_rmsd_mean"]-B[i]["pep_rmsd_mean"] for i in S_]
        imp=sum(x<0 for x in diff)
        print("  %-28s %-14s paired %+.3f  improved %2d/%2d  p=%.4f"%(label,nm,S.median(diff),imp,len(S_),p_two(imp,len(S_))))
for a in ["taper_query1","abrupt_query1","b9_3e-4_query1"]:
    print(a+":")
    cmp(a,"pretrained_query1","vs pretrained query-only")
    cmp(a,"pretrained_fullMSA","vs pretrained FULL MSA")
