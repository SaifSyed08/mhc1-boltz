"""Parallel range-scan extractor for the Boltz rcsb_processed_targets.tar on S3.

Tar headers are 512-byte aligned and self-identifying (POSIX 'ustar' magic at +257
plus a verifiable header checksum), so a worker can start at any 512-aligned offset,
resync to the next valid header, and stream forward extracting only wanted members.
This lets us pull the ~1.5k MHC-I .npz files out of a 65 GB archive without ever
storing the archive, and to parallelise across byte ranges.
"""
import concurrent.futures as cf, hashlib, json, os, pathlib, sys, threading, time
import urllib.request, urllib.error

URL = "https://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_targets.tar"
BLK = 512
CHUNK = 4 * 1024 * 1024

_print_lock = threading.Lock()
def log(*a):
    with _print_lock: print(*a, flush=True)

def total_size():
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])

def valid_header(h):
    """Return (name, size, typeflag) if h is a valid tar header, else None."""
    if len(h) < BLK or h[257:262] != b"ustar":
        return None
    try:
        stored = int(h[148:156].rstrip(b"\0 ").decode() or "-1", 8)
    except ValueError:
        return None
    chk = sum(h[:148]) + sum(b" " * 8) + sum(h[156:512])
    if chk != stored:
        return None
    name = h[0:100].rstrip(b"\0").decode("utf-8", "replace")
    prefix = h[345:500].rstrip(b"\0").decode("utf-8", "replace")
    if prefix:
        name = prefix + "/" + name
    try:
        size = int(h[124:136].rstrip(b"\0 ").decode() or "0", 8)
    except ValueError:
        return None
    typ = chr(h[156]) if h[156] not in (0,) else "0"
    return name, size, typ

class RangeReader:
    """Buffered sequential reader over an HTTP byte range, with retry."""
    def __init__(self, start, end, hard_end=None):
        self.pos, self.end, self.buf, self.bufpos = start, end, b"", 0
        # A member whose header falls inside our range may extend past `end`
        # (notably the ~100 MB manifest.json at the tail), so reads are only
        # capped by the real archive end; `end` just terminates the walk.
        self.hard_end = hard_end if hard_end is not None else end
    def _fetch(self, n):
        for t in range(12):
            try:
                hi = min(self.pos + n - 1, self.hard_end)
                if hi < self.pos: return b""
                req = urllib.request.Request(URL, headers={"Range": f"bytes={self.pos}-{hi}"})
                with urllib.request.urlopen(req, timeout=240) as r:
                    return r.read()
            except Exception:
                if t == 11: raise
                time.sleep(min(60, 2 ** t))
    def read(self, n):
        out = []
        need = n
        while need > 0:
            if self.bufpos >= len(self.buf):
                if self.pos > self.hard_end: break
                self.buf = self._fetch(max(CHUNK, need)); self.bufpos = 0
                if not self.buf: break
                self.pos += len(self.buf)
            take = min(need, len(self.buf) - self.bufpos)
            out.append(self.buf[self.bufpos:self.bufpos + take])
            self.bufpos += take; need -= take
        return b"".join(out)
    def skip(self, n):
        avail = len(self.buf) - self.bufpos
        if n <= avail:
            self.bufpos += n
        else:
            self.pos += n - avail
            self.buf, self.bufpos = b"", 0
    @property
    def abs_pos(self):
        return self.pos - (len(self.buf) - self.bufpos)

PROGRESS_LOCK = threading.Lock()
PROGRESS_PATH = None

def save_progress(wid, off):
    """Record how far this worker has walked so a restart can resume."""
    if PROGRESS_PATH is None: return
    with PROGRESS_LOCK:
        try:
            cur = json.loads(PROGRESS_PATH.read_text()) if PROGRESS_PATH.exists() else {}
        except Exception:
            cur = {}
        cur[str(wid)] = int(off)
        PROGRESS_PATH.write_text(json.dumps(cur))

def scan_range(args):
    """Fault-isolated wrapper: a worker that dies must not abort the whole scan."""
    try:
        return _scan_range(args)
    except Exception as exc:
        log(f"  [w{args[4]}] ABORTED: {type(exc).__name__}: {exc}")
        return [], 0, 0

def _scan_range(args):
    start, end, wanted, outdir, wid, hard_end = args
    start -= start % BLK
    r = RangeReader(start, end, hard_end)
    got, seen, resync = [], 0, 0
    # Build a reusable member index (name -> offset,size) so any later fetch of a
    # specific structure is a single range request instead of another 65 GB walk.
    idx_path = outdir.parent / "tar_index" / ("w%02d.tsv" % wid)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_buf = []
    idx_fh = idx_path.open("a", encoding="utf-8")
    # resync: read blocks until a valid header appears
    while True:
        b = r.read(BLK)
        if len(b) < BLK:
            idx_fh.close(); return got, seen, resync
        hv = valid_header(b)
        if hv: break
        resync += 1
        if resync > 4000000:
            idx_fh.close(); return got, seen, resync
    while True:
        name, size, typ = hv
        seen += 1
        pad = (-size) % BLK
        if typ == "0":
            idx_buf.append("\t".join(
                (os.path.basename(name), str(r.abs_pos), str(size))) + "\n")
            if len(idx_buf) >= 500:
                idx_fh.write("".join(idx_buf)); idx_fh.flush(); idx_buf.clear()
        if typ == "0" and not name.endswith(".npz"):
            base = os.path.basename(name)
            dest = outdir.parent / base
            if dest.exists() and dest.stat().st_size == size:
                r.skip(size + pad)
            else:
                body = r.read(size)
                if len(body) == size:
                    dest.write_bytes(body)
                    log(f"  [w{wid}] + {base} ({size/1e6:.1f} MB)  [non-npz member]")
                r.skip(pad)
        elif typ == "0" and name.endswith(".npz"):
            base = os.path.basename(name)
            if base in wanted:
                body = r.read(size)
                if len(body) == size:
                    (outdir / base).write_bytes(body)
                    got.append(base)
                    log(f"  [w{wid}] + {base} ({size/1024:.0f} KB)")
                r.skip(pad)
            else:
                r.skip(size + pad)
        else:
            r.skip(size + pad)
        if r.abs_pos > end:
            break
        save_progress(wid, r.abs_pos)
        b = r.read(BLK)
        if len(b) < BLK: break
        hv = valid_header(b)
        if not hv:
            # end-of-archive zero blocks or a desync; try to resync a little
            ok = False
            for _ in range(64):
                b = r.read(BLK)
                if len(b) < BLK: break
                hv = valid_header(b)
                if hv: ok = True; break
            if not ok: break
    idx_fh.write("".join(idx_buf)); idx_fh.close()
    return got, seen, resync

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--wanted", required=True, help="text file, one basename per line")
    p.add_argument("--outdir", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", action="store_true",
                   help="continue each worker from its last checkpoint")
    p.add_argument("--sample-gb", type=float, default=0.0,
                   help="if >0, scan only this many GB spread across the archive")
    a = p.parse_args()

    wanted = {l.strip() for l in open(a.wanted) if l.strip()}
    outdir = pathlib.Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    have = {f.name for f in outdir.glob("*.npz")}
    wanted -= have
    log(f"want {len(wanted)} files ({len(have)} already present)")

    N = total_size()
    log(f"archive size {N/1e9:.1f} GB")

    global PROGRESS_PATH
    PROGRESS_PATH = outdir.parent / "extract_progress.json"
    resume = {}
    if a.resume and PROGRESS_PATH.exists():
        resume = json.loads(PROGRESS_PATH.read_text())
        log(f"resuming from checkpoint: {len(resume)} workers")

    if a.sample_gb > 0:
        per = int(a.sample_gb * 1e9 / a.workers)
        stride = N // a.workers
        ranges = [(i * stride, min(i * stride + per, N) - 1, wanted, outdir, i, N - 1)
                  for i in range(a.workers)]
        log(f"SAMPLE mode: {a.workers} x {per/1e6:.0f} MB windows = {a.sample_gb} GB total")
    else:
        stride = N // a.workers
        ranges = []
        for i in range(a.workers):
            lo = i * stride
            hi = (N - 1) if i == a.workers - 1 else (i + 1) * stride - 1
            ck = resume.get(str(i))
            if ck is not None and lo <= ck <= hi:
                lo = ck
            ranges.append((lo, hi, wanted, outdir, i, N - 1))
        log(f"FULL mode: {a.workers} workers x {stride/1e9:.1f} GB")

    t0 = time.time(); allgot = []; allseen = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for got, seen, rs in ex.map(scan_range, ranges):
            allgot += got; allseen += seen
    log(f"\nheaders seen={allseen}  extracted={len(allgot)}  in {time.time()-t0:.0f}s")
    log(f"outdir now has {len(list(outdir.glob('*.npz')))} npz files")

if __name__ == "__main__":
    main()
