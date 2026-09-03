"""Step 6: write the processed NPZ dataset, manifest and split lists.

Unrelated chains are *disabled*, not deleted. Boltz consumes the two flags:

  * NPZ `mask[i]`  -- the tokenizer does `chains = struct.chains[struct.mask]`,
    so a masked chain is never tokenised and never reaches the model.
  * manifest `chains[i].valid` / `interfaces[j].valid` -- the samplers
    (ClusterSampler, RandomSampler) only pick valid chains/interfaces as the
    anchor the cropper centres on.

Both are needed: the mask removes the chain from the input, the manifest flag
stops the sampler from anchoring a crop on it.

Deleting chains via `Structure.remove_invalid_chains()` would renumber
`asym_id`, which would break the manifest's `chain_id` -> NPZ `asym_id`
correspondence and the per-chain `msa_id` mapping. Masking keeps every index
stable and is exactly reversible, and the tokenizer still emits a monotonically
increasing `asym_id` sequence (gaps are fine), which is what the featurizer
asserts.

The record is copied verbatim apart from the valid flags, so `msa_id`,
`cluster_id` and `template_id` are preserved for training.
"""
import collections
import copy
import json
import pathlib
import shutil

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / "data"
R = ROOT / "reports"
OUT = D / "processed"
# stable copy: the tar extractor also emits data/manifest.json, so read a
# snapshot it will not overwrite mid-read.
MANIFEST_SRC = D / "manifest_source.json"


# boltz v1.0.0's Connection dtype, field order significant (see below).
CONNECTION_V1 = [("chain_1", "<i4"), ("chain_2", "<i4"), ("res_1", "<i4"),
                 ("res_2", "<i4"), ("atom_1", "<i4"), ("atom_2", "<i4")]


def upgrade_chains(chains):
    """Add the `cyclic_period` field boltz v1.0.0's tokenizer requires.

    The distributed rcsb_processed_targets.tar predates/postdates v1.0.0's Chain
    dtype and ships 9 fields; `tokenize/boltz.py` reads `chain["cyclic_period"]`,
    so tokenising the archive as-shipped raises
    `ValueError: no field of name cyclic_period`. Zero is the correct value for
    every chain here -- 0 means "not cyclic" to `modules/encoders.py` -- so
    zero-filling is a faithful upgrade rather than a guess.
    """
    if "cyclic_period" in (chains.dtype.names or ()):
        return chains, False
    out = np.zeros(len(chains), dtype=chains.dtype.descr + [("cyclic_period", "<i4")])
    for n in chains.dtype.names:
        out[n] = chains[n]
    return out, True


def reorder_connections(conn):
    """Put `connections` fields in v1.0.0 order.

    `load_input` does `structure["connections"].astype(Connection)`, and numpy
    casts structured arrays **by position, not by name**. The archive stores
    (atom_1, atom_2, res_1, res_2, chain_1, chain_2) while v1.0.0's Connection is
    (chain_1, chain_2, res_1, res_2, atom_1, atom_2) -- the exact reverse -- so
    that cast silently loads atom indices into the chain fields. Writing the
    fields in v1.0.0 order makes the positional cast a correct no-op.
    """
    names = list(conn.dtype.names or ())
    want = [f[0] for f in CONNECTION_V1]
    if names == want:
        return conn, False
    out = np.zeros(len(conn), dtype=CONNECTION_V1)
    for n in want:
        if n in names:
            out[n] = conn[n]
    return out, True


def iter_records(path, chunk=8 << 20):
    """Stream a JSON array of records by brace depth.

    The source manifest is ~772 MB; loading it whole would cost several GB, so
    this walks the byte stream and yields one top-level object at a time.
    """
    buf = bytearray()
    depth = 0
    in_str = False
    esc = False
    started = False
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                return
            for b in block:
                if not started:
                    if b == 0x5B:  # '['
                        started = True
                    continue
                if depth == 0 and b in (0x20, 0x09, 0x0A, 0x0D, 0x2C):
                    continue
                if depth == 0 and b == 0x5D:  # ']'
                    return
                buf.append(b)
                if in_str:
                    if esc:
                        esc = False
                    elif b == 0x5C:  # backslash
                        esc = True
                    elif b == 0x22:  # closing quote
                        in_str = False
                    continue
                if b == 0x22:
                    in_str = True
                elif b == 0x7B:  # '{'
                    depth += 1
                elif b == 0x7D:  # '}'
                    depth -= 1
                    if depth == 0:
                        yield json.loads(bytes(buf))
                        buf.clear()


def main():
    cands = {r["pdb_id"]: r for r in json.loads((D / "candidates.json").read_text())}
    sel = json.loads((D / "chain_selection.json").read_text())
    selected = {p: r for p, r in sel.items() if r.get("keep_chain_idx")}
    print("selected samples: %d" % len(selected))

    if not MANIFEST_SRC.exists():
        raise SystemExit("manifest.json not extracted yet: %s" % MANIFEST_SRC)

    wanted = {p.lower() for p in selected}
    records = {}
    total = 0
    for rec in iter_records(MANIFEST_SRC):
        total += 1
        rid = str(rec.get("id", "")).lower()
        if rid in wanted:
            records[rid] = rec
    print("records in source manifest: %d" % total)
    print("of our samples, present in manifest: %d/%d" % (len(records), len(selected)))
    missing_rec = sorted(p for p in selected if p.lower() not in records)
    if missing_rec:
        print("  NOT in Boltz manifest: %s" % missing_rec[:20])

    (OUT / "structures").mkdir(parents=True, exist_ok=True)
    out_records, problems, stats = [], [], collections.Counter()

    for pid in sorted(selected):
        rid = pid.lower()
        if rid not in records:
            problems.append({"pdb_id": pid, "issue": "no_manifest_record"})
            continue
        info = selected[pid]
        keep = set(info["keep_chain_idx"])
        z = np.load(D / "npz_raw" / ("%s.npz" % rid))
        chains = z["chains"]

        rec = copy.deepcopy(records[rid])
        rc = rec.get("chains") or []

        # the manifest chain list must line up 1:1 with the NPZ chain table
        if len(rc) != len(chains):
            problems.append({"pdb_id": pid, "issue": "manifest_npz_chain_count_mismatch",
                             "manifest": len(rc), "npz": len(chains)})
            continue
        bad = [(i, rc[i].get("chain_name"), str(chains[i]["name"]))
               for i in range(len(rc))
               if rc[i].get("chain_id") != i or rc[i].get("chain_name") != str(chains[i]["name"])]
        if bad:
            problems.append({"pdb_id": pid, "issue": "chain_id_or_name_misalignment",
                             "examples": bad[:4]})
            continue

        for i, c in enumerate(rc):
            c["valid"] = bool(i in keep)
        for itf in rec.get("interfaces") or []:
            itf["valid"] = bool(itf.get("chain_1") in keep and itf.get("chain_2") in keep)
        stats["valid_interfaces"] += sum(1 for i in (rec.get("interfaces") or []) if i["valid"])
        stats["invalidated_interfaces"] += sum(
            1 for i in (rec.get("interfaces") or []) if not i["valid"])
        stats["disabled_chains"] += len(chains) - len(keep)
        stats["kept_chains"] += len(keep)

        # write the NPZ with every original key preserved, mask replaced, and the
        # two dtype drifts against boltz v1.0.0 repaired (see upgrade_* above)
        mask = np.zeros(len(chains), dtype=bool)
        mask[sorted(keep)] = True
        payload = {k: z[k] for k in z.files}
        payload["mask"] = mask
        payload["chains"], added_cyc = upgrade_chains(z["chains"])
        payload["connections"], fixed_conn = reorder_connections(z["connections"])
        stats["chains_field_added"] += int(added_cyc)
        stats["connections_reordered"] += int(fixed_conn)
        np.savez_compressed(OUT / "structures" / ("%s.npz" % rid), **payload)
        out_records.append(rec)

    print("\nwrote %d processed NPZ files" % len(out_records))
    print("problems: %d" % len(problems))
    for iss, n in collections.Counter(p["issue"] for p in problems).most_common():
        print("    %-45s %d" % (iss, n))
    print("chains kept: %d   chains disabled: %d" % (stats["kept_chains"], stats["disabled_chains"]))
    print("interfaces valid: %d   invalidated: %d"
          % (stats["valid_interfaces"], stats["invalidated_interfaces"]))

    # Boltz's DatasetConfig.split names the VALIDATION ids and puts *everything
    # else in the manifest* into training -- and its filters apply only to the
    # training side. So test records must be physically absent from the training
    # manifest, not merely filtered: otherwise the held-out set leaks into train.
    ids = {r["id"]: cands[r["id"].upper()]["split"] for r in out_records}
    train_val = [r for r in out_records if ids[r["id"]] != "test"]
    test_only = [r for r in out_records if ids[r["id"]] == "test"]

    (OUT / "manifest.json").write_text(json.dumps(train_val))
    (OUT / "manifest_test.json").write_text(json.dumps(test_only))
    print("wrote %s (%d records: train+validation only)"
          % (OUT / "manifest.json", len(train_val)))
    print("wrote %s (%d records: held out, never seen by training)"
          % (OUT / "manifest_test.json", len(test_only)))

    for s in ("train", "validation", "test"):
        lst = sorted(k for k, v in ids.items() if v == s)
        (OUT / ("%s_ids.txt" % s)).write_text("\n".join(lst) + "\n")
        print("    %-11s %4d -> %s" % (s, len(lst), OUT / ("%s_ids.txt" % s)))

    # The brief asks for our validation ids to be added to boltz's
    # scripts/train/assets/validation_ids.txt. That upstream list has no overlap
    # with an MHC-I-only manifest, so pointing `split:` at it alone would yield an
    # empty validation set; emit the union, which is both literal and correct.
    upstream = ROOT / "boltz-src" / "scripts" / "train" / "assets" / "validation_ids.txt"
    up = ([l.strip().lower() for l in upstream.read_text().splitlines() if l.strip()]
          if upstream.exists() else [])
    val = sorted(k for k, v in ids.items() if v == "validation")
    merged = sorted(set(up) | set(val))
    (OUT / "validation_ids_boltz.txt").write_text("\n".join(merged) + "\n")
    print("    validation_ids_boltz.txt: %d upstream + %d ours = %d"
          % (len(up), len(val), len(merged)))

    # Exactly which MSA files training will need, so the 107 GB MSA archive can
    # be range-scanned for just this subset rather than downloaded whole.
    # NOT filtered on c["valid"]. Boltz's loader does not consult that flag when
    # reading MSAs: load_input() in boltz/data/module/training.py iterates
    # `record.chains` unconditionally and np.load()s every chain's msa_id. A
    # disabled chain is skipped by the *tokenizer* (via the NPZ mask), but its
    # MSA file is opened first, so it still has to exist on disk.
    #
    # Filtering on valid here produced 1134 ids where the manifest actually
    # references 1343, and the missing ones took down the validation run: the
    # dataset's __getitem__ falls back to __getitem__(0) on any load error, so a
    # permanently-missing MSA on record 0 recurses until RecursionError.
    msa_ids = sorted({c["msa_id"] for r in out_records for c in r["chains"]
                      if c["msa_id"] not in (-1, "", None)})
    (OUT / "required_msa_ids.txt").write_text("\n".join(str(m) for m in msa_ids) + "\n")
    print("    required MSA files: %d -> %s" % (len(msa_ids), OUT / "required_msa_ids.txt"))

    R.mkdir(exist_ok=True)
    (R / "write_report.json").write_text(json.dumps({
        "n_selected": len(selected),
        "n_written": len(out_records),
        "n_records_in_source_manifest": total,
        "not_in_boltz_manifest": missing_rec,
        "problems": problems,
        "stats": dict(stats),
        "splits": {s: sum(1 for v in ids.values() if v == s)
                   for s in ("train", "validation", "test")},
    }, indent=1))
    print("wrote %s" % (R / "write_report.json"))


if __name__ == "__main__":
    main()
