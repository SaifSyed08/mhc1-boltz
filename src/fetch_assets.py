"""Fetch the two large single-file assets: the pretrained checkpoint and symmetry.pkl.

Both are plain HTTPS GETs, but Saif's link has dropped mid-transfer before, so each
download is resumable: we keep a partial file and re-request only `bytes=<have>-`
on retry. Servers that honour Range answer 206 with the tail; a server that ignores
Range answers 200 with the whole body, which we detect and restart cleanly.
"""
import pathlib, sys, time, urllib.request, urllib.error

ASSETS = {
    "symmetry.pkl":     "https://boltz1.s3.us-east-2.amazonaws.com/symmetry.pkl",
    "boltz1_conf.ckpt": "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt",
}
DEST = pathlib.Path(__file__).resolve().parent.parent / "data" / "assets"


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0))


def fetch(name, url):
    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / name
    total = remote_size(url)
    if out.exists() and out.stat().st_size == total:
        print(f"[{name}] already complete ({total/1e6:.0f} MB)", flush=True)
        return
    for attempt in range(20):
        have = out.stat().st_size if out.exists() else 0
        if have and have == total:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as r:
                # 200 means Range was ignored -> we must overwrite from zero.
                mode = "ab" if r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                t0, last = time.time(), 0
                with out.open(mode) as fh:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        if have - last > 50 << 20:
                            last = have
                            mbs = (have / 1e6) / max(time.time() - t0, 1e-9)
                            print(f"[{name}] {have/1e6:8.0f}/{total/1e6:.0f} MB "
                                  f"({100*have/total:5.1f}%)  {mbs:.1f} MB/s", flush=True)
        except Exception as exc:
            print(f"[{name}] attempt {attempt}: {type(exc).__name__}: {exc} "
                  f"-- resuming from {have/1e6:.0f} MB", flush=True)
            time.sleep(min(60, 2 ** attempt))
            continue
        if out.exists() and out.stat().st_size >= total:
            break
    got = out.stat().st_size if out.exists() else 0
    ok = "OK" if got == total else "INCOMPLETE"
    print(f"[{name}] {ok}: {got/1e6:.0f}/{total/1e6:.0f} MB -> {out}", flush=True)


if __name__ == "__main__":
    want = sys.argv[1:] or list(ASSETS)
    for n in want:
        fetch(n, ASSETS[n])
