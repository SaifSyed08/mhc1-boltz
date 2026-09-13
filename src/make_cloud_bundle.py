"""Pack the parts of data/ that a cloud GPU session cannot regenerate cheaply.

Three kinds of thing live under data/:

  1. Regenerable in seconds from a URL -- data/assets/ (the 3.6 GB checkpoint and
     the 215 MB symmetry pickle). The notebook re-downloads these; shipping 3.8 GB
     through Drive to save a 5-minute download would be silly.

  2. Regenerable, but only by re-scanning the upstream S3 tars -- data/npz_raw/,
     data/tar_index/, data/msa_tar_index/, data/manifest_source.json. That scan is
     the expensive half of milestone 1 and there is no reason to pay it twice.

  3. The actual dataset -- data/processed/ and data/msa/. ~575 MB. This is the
     output of milestone 1 and the only thing the fine-tune reads.

This script packs (3), and only (3). Upload the result once to Google Drive (for
Colab) or as a Kaggle Dataset; every subsequent session just unpacks it.

    python src/make_cloud_bundle.py
    python src/make_cloud_bundle.py --out D:/somewhere/mhc1-data.tar.gz

The tarball unpacks to `data/processed/...` and `data/msa/...` relative to the
repo root, so the notebook untars it straight over a fresh clone.
"""

import argparse
import hashlib
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# (path relative to repo root, required)
MEMBERS = [
    ("data/processed", True),
    ("data/msa", True),
]


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def tree_size(p: Path) -> tuple[int, int]:
    total = count = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
            count += 1
    return total, count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "mhc1-data-bundle.tar.gz",
        help="output tarball (default: <repo>/mhc1-data-bundle.tar.gz)",
    )
    ap.add_argument(
        "--no-compress",
        action="store_true",
        help="write an uncompressed .tar -- NPZs are already deflated, so gzip "
        "buys ~1%% for several minutes of CPU. Use this if upload speed is not "
        "your bottleneck.",
    )
    args = ap.parse_args()

    missing = [rel for rel, req in MEMBERS if req and not (REPO / rel).exists()]
    if missing:
        print("error: missing required inputs:", file=sys.stderr)
        for rel in missing:
            print(f"  {rel}", file=sys.stderr)
        print(
            "\nRun the milestone-1 pipeline first (see README.md), or fetch the\n"
            "MSA subset with src/msa_extract.py.",
            file=sys.stderr,
        )
        return 1

    print("Packing:")
    grand = grand_n = 0
    for rel, _ in MEMBERS:
        size, n = tree_size(REPO / rel)
        grand += size
        grand_n += n
        print(f"  {rel:<24} {human(size):>10}  ({n:,} files)")
    print(f"  {'total':<24} {human(grand):>10}  ({grand_n:,} files)")

    out = args.out
    if args.no_compress and out.suffix == ".gz":
        out = out.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if args.no_compress else "w:gz"
    print(f"\nWriting {out} ({'uncompressed' if args.no_compress else 'gzip'})...")
    t0 = time.time()
    done = [0]

    def progress(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        done[0] += 1
        if done[0] % 200 == 0:
            pct = 100 * done[0] / grand_n
            print(f"\r  {done[0]:,}/{grand_n:,} files ({pct:4.1f}%)", end="", flush=True)
        return ti

    with tarfile.open(out, mode) as tar:
        for rel, _ in MEMBERS:
            tar.add(REPO / rel, arcname=rel, filter=progress)
    print(f"\r  {grand_n:,}/{grand_n:,} files (100.0%)")

    size = out.stat().st_size
    print(f"\nWrote {human(size)} in {time.time() - t0:.0f}s")

    print("Hashing...")
    h = hashlib.sha256()
    with out.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    print(f"sha256  {digest}")

    sidecar = out.with_suffix(out.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {out.name}\n")
    print(f"wrote   {sidecar.name}")

    print(
        "\nNext:\n"
        "  Colab  -- upload to Google Drive, e.g. MyDrive/mhc1/ , then run\n"
        "            notebooks/mhc1_boltz_t4.ipynb and set BUNDLE to that path.\n"
        "  Kaggle -- create a new Dataset from this file; the notebook finds it\n"
        "            automatically under /kaggle/input/."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
