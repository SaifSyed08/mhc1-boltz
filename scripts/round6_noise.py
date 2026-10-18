import json,glob,statistics as S,pathlib
from math import comb
def load(p):
    out={}
    for f in glob.glob(p+"/*.jsonl"):
        for l in open(f):
            r=json.loads(l)
            if not r.get("sanity"): out[r["record_id"]]=r
    return out
def p2(k,n):
    if n==0: return 1.0
    o=comb(n,k); return min(1.0,sum(comb(n,i) for i in range(n+1) if comb(n,i)<=o)/2**n)
sub=set(pathlib.Path("data/processed/subset30_ids.txt").read_text().split())
R={"abrupt":("runs/eval_val99/abrupt_last_d1","runs/eval_val99/abrupt_d1_r2"),
   "taper":("runs/eval_val99/taper_last_d1","runs/eval_val99/taper_d1_r2"),
   "b9":("runs/eval_val99/b9_3e-4_last_d1","runs/eval_val99/b9_3e-4_d1_r2")}
print("DRAW-TO-DRAW NOISE (same checkpoint, two independent evals)")
print("%-8s %8s %8s %8s   %s"%("arm","draw1","draw2","delta","|per-record diff| median / p90"))
D={}
for k,(a,b) in R.items():
    A,B=load(a),load(b); ids=[i for i in A if i in B]; D[k]=(A,B,ids)
    d=[abs(A[i]["pep_rmsd_mean"]-B[i]["pep_rmsd_mean"]) for i in ids]
    m1=S.median([A[i]["pep_rmsd_mean"] for i in ids]); m2=S.median([B[i]["pep_rmsd_mean"] for i in ids])
    print("%-8s %8.3f %8.3f %+8.3f   %.3f / %.3f"%(k,m1,m2,m2-m1,S.median(d),sorted(d)[int(.9*len(d))]))
print()
print("POOLED (6 samples per record: both draws) - arms vs each other")
def pool(k):
    A,B,ids=D[k]; return {i:S.mean(A[i]["pep_rmsd"]+B[i]["pep_rmsd"]) for i in ids}
P={k:pool(k) for k in R}
for k in R: print("  %-8s pooled median %.3f"%(k,S.median(list(P[k].values()))))
print()
for a,b in [("abrupt","b9"),("taper","b9"),("abrupt","taper")]:
    ids=[i for i in P[a] if i in P[b]]; ho=[i for i in ids if i not in sub]
    for nm,ss in [("all %d"%len(ids),ids),("held-out %d"%len(ho),ho)]:
        diff=[P[a][i]-P[b][i] for i in ss]; imp=sum(x<0 for x in diff)
        print("  %-16s %-14s paired %+.3f  %s better %2d/%2d  p=%.3f"%(a+" vs "+b,nm,S.median(diff),a,imp,len(ss),p2(imp,len(ss))))
pre=load("runs/eval_val99/pretrained")
print()
print("POOLED arms vs pretrained-with-full-MSA (held-out 69):")
for k in R:
    ho=[i for i in P[k] if i in pre and i not in sub]
    diff=[P[k][i]-pre[i]["pep_rmsd_mean"] for i in ho]; imp=sum(x<0 for x in diff)
    print("  %-8s paired %+.3f  better %2d/%2d  p=%.4f"%(k,S.median(diff),imp,len(ho),p2(imp,len(ho))))
