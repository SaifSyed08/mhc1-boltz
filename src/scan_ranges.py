"""Supplementary scan of explicit byte ranges of the tar.

Used to cover ranges a previous run missed. Workers 8 and 11 of the main run
both started inside a >128 MB member and hit the old resync cap, so their
5.4 GB ranges were never walked.
"""
import argparse
import concurrent.futures as cf
import pathlib
import time

import tar_extract as TE

p = argparse.ArgumentParser()
p.add_argument("--wanted", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--ranges", required=True, help="comma-separated lo-hi byte ranges")
p.add_argument("--split", type=int, default=3, help="sub-ranges per range")
a = p.parse_args()

wanted = {l.strip() for l in open(a.wanted) if l.strip()}
outdir = pathlib.Path(a.outdir)
have = {f.name for f in outdir.glob("*.npz")}
wanted -= have
TE.PROGRESS_PATH = outdir.parent / "extract_progress_supp.json"
N = TE.total_size()
TE.log(f"want {len(wanted)} files still missing; archive {N/1e9:.1f} GB")

jobs, wid = [], 100
for spec in a.ranges.split(","):
    lo, hi = (int(x) for x in spec.split("-"))
    step = (hi - lo) // a.split
    for k in range(a.split):
        s = lo + k * step
        e = hi if k == a.split - 1 else lo + (k + 1) * step - 1
        jobs.append((s, e, wanted, outdir, wid, N - 1))
        wid += 1
TE.log(f"{len(jobs)} sub-ranges: " + ", ".join(f"{j[0]/1e9:.2f}-{j[1]/1e9:.2f}GB" for j in jobs))

t0 = time.time()
tot = 0
with cf.ThreadPoolExecutor(len(jobs)) as ex:
    for got, seen, rs in ex.map(TE.scan_range, jobs):
        tot += len(got)
TE.log(f"extracted {tot} in {time.time()-t0:.0f}s; outdir now {len(list(outdir.glob('*.npz')))} npz")
