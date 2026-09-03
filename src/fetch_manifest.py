"""Fetch manifest.json straight out of the tar via range requests.

The member is the last one in the archive; its offset/size were located by
binary-searching the binary->JSON transition (see reports/README).
"""
import concurrent.futures as cf
import pathlib
import time
import urllib.request

URL = "https://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_targets.tar"
START, SIZE = 64176935936, 771587365
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "manifest.json"
W = 4


def get(i):
    chunk = (SIZE + W - 1) // W
    s = START + i * chunk
    e = min(s + chunk, START + SIZE) - 1
    for t in range(12):
        try:
            req = urllib.request.Request(URL, headers={"Range": "bytes=%d-%d" % (s, e)})
            with urllib.request.urlopen(req, timeout=900) as r:
                d = r.read()
            if len(d) == e - s + 1:
                return i, d
            raise IOError("short read %d != %d" % (len(d), e - s + 1))
        except Exception as exc:
            if t == 11:
                raise
            print("  part %d retry %d (%s)" % (i, t, exc), flush=True)
            time.sleep(min(60, 2 ** t))


t0 = time.time()
parts = {}
with cf.ThreadPoolExecutor(W) as ex:
    for i, d in ex.map(get, range(W)):
        parts[i] = d
        print("  part %d ok %.1f MB" % (i, len(d) / 1e6), flush=True)
data = b"".join(parts[i] for i in range(W))
assert len(data) == SIZE, (len(data), SIZE)
OUT.write_bytes(data)
print("wrote %s (%d bytes) in %.0fs" % (OUT, len(data), time.time() - t0))
print("head:", data[:90])
print("tail:", data[-60:])
