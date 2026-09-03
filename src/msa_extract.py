"""Hop-mode extractor for the 107 GB rcsb_processed_msa.tar on S3.

Why this is not just tar_extract.py pointed at a different URL
--------------------------------------------------------------
`tar_extract.py` walks the archive as a buffered *stream*: it fetches 4 MB at a
time and `skip()` only avoids downloading when the jump is larger than what is
already buffered. In rcsb_processed_targets.tar that was fine -- members there are
often hundreds of MB, so skips were real skips. In this archive the mean member is
~490 KB, comfortably inside a 4 MB buffer, so the same code would download
essentially all 107 GB to keep 1,134 files.

The structure of tar makes something much cheaper possible. Every member is a
512-byte header followed by its body padded up to a 512-byte boundary, and the
header states the body's size. So the offset of the *next* header is a pure
function of the current one:

    next = here + 512 + size + (-size % 512)

That means we never have to read a body we do not want. We "hop": range-request
512 bytes, parse, compute `next`, request 512 bytes there. Cost becomes
~220k * 512 B ~= 110 MB of headers plus only the members we actually keep --
roughly a 1000x reduction in bytes moved, at the price of many small requests.

That trade turns the job from bandwidth-bound into latency-bound, so the two
things that matter are now (a) parallelism across byte ranges and (b) not paying
a TLS handshake per hop. Each worker therefore keeps ONE persistent HTTPS
connection and reuses it for every request, reconnecting only on error.

Resync
------
A worker starting at an arbitrary offset almost certainly lands mid-body, so it
must scan forward for the first valid header before it can start hopping. A tar
header is self-identifying: 'ustar' magic at +257 and a checksum over the header
with the checksum field blanked. Two independent conditions make a false positive
very unlikely. Resync reads sequentially (bodies and all) but only until the next
header, so it costs at most one member -- unlike the old code, which paid that
cost for the whole range.
"""
import argparse, concurrent.futures as cf, http.client, json, os, pathlib
import ssl, threading, time

HOST = "boltz1.s3.us-east-2.amazonaws.com"
PATH = "/rcsb_processed_msa.tar"
BLK = 512
RESYNC_CHUNK = 1 << 20        # 1 MB reads while hunting for a header
RESYNC_CAP = 512 << 20        # give up after 512 MB (largest plausible member)

_print_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(*a, flush=True)


class Conn:
    """One keep-alive HTTPS connection, with transparent reconnect + retry.

    Reusing the socket is the whole point: at ~220k hops a fresh TLS handshake
    per request would dominate the runtime.
    """

    def __init__(self):
        self.ctx = ssl.create_default_context()
        self.c = None

    def _connect(self):
        self.c = http.client.HTTPSConnection(HOST, timeout=120, context=self.ctx)

    def get(self, start, end):
        """Return bytes [start, end] inclusive, as HTTP Range semantics define."""
        for attempt in range(10):
            try:
                if self.c is None:
                    self._connect()
                self.c.request("GET", PATH, headers={
                    "Range": "bytes=%d-%d" % (start, end),
                    "Host": HOST,
                    "Connection": "keep-alive",
                })
                r = self.c.getresponse()
                body = r.read()          # must drain to keep the socket reusable
                if r.status not in (200, 206):
                    raise OSError("HTTP %s" % r.status)
                return body
            except Exception:
                try:
                    if self.c:
                        self.c.close()
                except Exception:
                    pass
                self.c = None
                if attempt == 9:
                    raise
                time.sleep(min(30, 2 ** attempt))

    def close(self):
        try:
            if self.c:
                self.c.close()
        except Exception:
            pass


def parse_header(h):
    """(name, size, typeflag) if h is a valid ustar header, else None."""
    if len(h) < BLK or h[257:262] != b"ustar":
        return None
    try:
        stored = int(h[148:156].rstrip(b"\0 ").decode() or "-1", 8)
    except ValueError:
        return None
    # Checksum is computed with the checksum field itself treated as 8 spaces.
    if sum(h[:148]) + sum(b" " * 8) + sum(h[156:512]) != stored:
        return None
    name = h[0:100].rstrip(b"\0").decode("utf-8", "replace")
    prefix = h[345:500].rstrip(b"\0").decode("utf-8", "replace")
    if prefix:
        name = prefix + "/" + name
    try:
        size = int(h[124:136].rstrip(b"\0 ").decode() or "0", 8)
    except ValueError:
        return None
    return name, size, chr(h[156]) if h[156] else "0"


def resync(conn, start, hard_end):
    """First 512-aligned offset >= start holding a valid header, or None."""
    off = start - (start % BLK)
    scanned = 0
    while scanned < RESYNC_CAP and off < hard_end:
        buf = conn.get(off, min(off + RESYNC_CHUNK - 1, hard_end))
        if not buf:
            return None
        for i in range(0, len(buf) - BLK + 1, BLK):
            if parse_header(buf[i:i + BLK]):
                return off + i
        step = (len(buf) // BLK) * BLK
        if step == 0:
            return None
        off += step
        scanned += step
    return None


STATE_LOCK = threading.Lock()


def worker(wid, start, end, hard_end, wanted, outdir, idxdir, statepath):
    conn = Conn()
    idx_path = idxdir / ("m%03d.tsv" % wid)
    idx_fh = idx_path.open("a", encoding="utf-8")
    idx_buf, got, seen, t0 = [], [], 0, time.time()
    try:
        off = resync(conn, start, hard_end)
        if off is None:
            log("  [m%d] no header found in range" % wid)
            return got, seen
        while off < end:
            h = conn.get(off, off + BLK - 1)
            hv = parse_header(h)
            if not hv:
                # Either end-of-archive zero blocks or a desync; try once to recover.
                off2 = resync(conn, off, min(end + (1 << 20), hard_end))
                if off2 is None or off2 <= off:
                    break
                off = off2
                continue
            name, size, typ = hv
            seen += 1
            base = os.path.basename(name)
            body_at = off + BLK
            if typ == "0":
                idx_buf.append("%s\t%d\t%d\n" % (base, body_at, size))
                if len(idx_buf) >= 500:
                    idx_fh.write("".join(idx_buf))
                    idx_fh.flush()
                    idx_buf.clear()
                if base in wanted:
                    dest = outdir / base
                    if not (dest.exists() and dest.stat().st_size == size):
                        body = conn.get(body_at, body_at + size - 1)
                        if len(body) == size:
                            dest.write_bytes(body)
                            got.append(base)
                            log("  [m%d] + %s (%.0f KB)" % (wid, base, size / 1024))
            # The hop: skip header + body + padding without reading the body.
            off = body_at + size + ((-size) % BLK)
            if seen % 2000 == 0:
                rate = seen / max(time.time() - t0, 1e-9)
                log("  [m%d] %d members, %d kept, %.0f hdr/s, at %.1f GB"
                    % (wid, seen, len(got), rate, off / 1e9))
                with STATE_LOCK:
                    try:
                        cur = json.loads(statepath.read_text()) if statepath.exists() else {}
                    except Exception:
                        cur = {}
                    cur[str(wid)] = int(off)
                    statepath.write_text(json.dumps(cur))
    finally:
        idx_fh.write("".join(idx_buf))
        idx_fh.close()
        conn.close()
    return got, seen


def archive_size():
    c = Conn()
    c.c = http.client.HTTPSConnection(HOST, timeout=60, context=c.ctx)
    c.c.request("HEAD", PATH, headers={"Host": HOST})
    r = c.c.getresponse()
    n = int(r.headers["Content-Length"])
    r.read()
    c.close()
    return n


def load_index(idxdir):
    """basename -> (body_offset, size) from the TSVs written by a previous scan."""
    idx = {}
    for f in sorted(idxdir.glob("*.tsv")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    idx[parts[0]] = (int(parts[1]), int(parts[2]))
    return idx


def fetch_by_index(wanted, idx, outdir, workers):
    """Fast path: the offsets are already known, so each file is one range GET.

    This is the payoff for writing the index during the first scan. Fetching a
    late addition costs one request per file instead of another walk of the
    107 GB archive.
    """
    todo = [(b, idx[b]) for b in sorted(wanted) if b in idx]
    unknown = sorted(wanted - set(idx))
    log("index covers %d of %d wanted files" % (len(todo), len(wanted)))
    if unknown:
        log("  NOT in index (needs a scan): %s" % ", ".join(unknown[:10]))

    local = threading.local()

    def one(job):
        base, (off, size) = job
        if not hasattr(local, "conn"):
            local.conn = Conn()
        body = local.conn.get(off, off + size - 1)
        if len(body) != size:
            return None
        (outdir / base).write_bytes(body)
        return base

    got = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, todo):
            if r:
                got.append(r)
    log("fetched %d files by index" % len(got))
    return got, unknown


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wanted", required=True,
                   help="text file: one msa id per line, extension optional")
    p.add_argument("--outdir", required=True)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--use-index", action="store_true",
                   help="fetch straight from data/msa_tar_index/ instead of walking "
                        "the archive; only valid after a full scan has run")
    a = p.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    outdir = pathlib.Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    idxdir = root / "data" / "msa_tar_index"
    idxdir.mkdir(parents=True, exist_ok=True)
    statepath = root / "data" / "msa_extract_progress.json"

    ids = [l.strip() for l in open(a.wanted) if l.strip()]
    allwant = {i if i.endswith(".npz") else i + ".npz" for i in ids}
    have = {f.name for f in outdir.glob("*.npz")}
    wanted = allwant - have
    log("want %d MSA files (%d already present)" % (len(wanted), len(have)))
    if not wanted:
        return

    if a.use_index:
        idx = load_index(idxdir)
        log("loaded index: %d members" % len(idx))
        fetch_by_index(wanted, idx, outdir, a.workers)
        present = {f.name for f in outdir.glob("*.npz")}
        missing = sorted(allwant - present)
        log("have %d/%d; missing %d" % (len(present), len(ids), len(missing)))
        if missing:
            (root / "data" / "msa_missing.txt").write_text("\n".join(missing))
        return

    total = archive_size()
    log("archive %.1f GB, %d workers" % (total / 1e9, a.workers))

    state = {}
    if a.resume and statepath.exists():
        try:
            state = json.loads(statepath.read_text())
        except Exception:
            state = {}

    span = total // a.workers
    jobs = []
    for w in range(a.workers):
        s = w * span
        e = total if w == a.workers - 1 else (w + 1) * span
        s = max(s, int(state.get(str(w), s)))
        if s >= e:
            continue
        jobs.append((w, s, e, total, wanted, outdir, idxdir, statepath))

    t0 = time.time()
    kept, seen = 0, 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(worker, *j) for j in jobs]
        for f in cf.as_completed(futs):
            try:
                g, s = f.result()
                kept += len(g)
                seen += s
            except Exception as exc:
                log("  worker failed: %s: %s" % (type(exc).__name__, exc))
    log("done in %.1f min: %d members walked, %d new files"
        % ((time.time() - t0) / 60, seen, kept))
    present = {f.name for f in outdir.glob("*.npz")}
    missing = sorted(allwant - present)
    log("have %d/%d; missing %d" % (len(present), len(ids), len(missing)))
    if missing:
        (root / "data" / "msa_missing.txt").write_text("\n".join(missing))


if __name__ == "__main__":
    main()
