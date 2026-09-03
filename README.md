# MHC-I dataset preparation for Boltz-1 fine-tuning

Preprocessing and validation pipeline that turns the TCR3D MHC-I chain list into
a clean, MHC-I-specific training set for continued training of
[Boltz-1 v1.0.0](https://github.com/jwohlwend/boltz/tree/v1.0.0).

Everything here is stage 1 of the brief: data preprocessing and validation. No
GPU is required for any of it.

## Layout

```
src/          pipeline stages, run in order
configs/      mhc1_finetune.yaml + mhc1_baseline.yaml (generated)
data/         inputs, intermediates and the processed dataset
reports/      per-stage reports + MILESTONE_SUMMARY.md
boltz-src/    boltz v1.0.0 checkout (reference for the data format)
.venv/        isolated env with mashumaro/numba so Boltz's own loader can run
.venv-gpu/    training stack: torch+cu128, lightning, hydra, boltz --no-deps
```

## Pipeline

| stage | what it does |
|---|---|
| `step1_tcr3d.py` | scrape the TCR3D MHC-I chain list (embedded Tabulator JSON) |
| `step2_rcsb.py` | fetch RCSB metadata: release dates, entity sequences, auth/label chain IDs, assemblies |
| `step3_annotate.py` | assign a role to every entity (MHC-I heavy / peptide / b2m / TCR / other) |
| `step4_split.py` | temporal split + (allele, peptide) leakage removal |
| `tar_extract.py` | pull our ~1.5k `.npz` out of the 65 GB `rcsb_processed_targets.tar` by range-scanning it |
| `scan_ranges.py` | rescan explicit byte ranges a previous pass missed |
| `fetch_manifest.py` | pull the 772 MB `manifest.json` member out of the same tar |
| `step5_map_select.py` | map NPZ chains to roles by sequence; pick the pMHC-I system by geometry |
| `step6_write.py` | write processed NPZs (mask), manifest (valid flags) and split lists |
| `step7_validate.py` | run the validation checklist, including Boltz's own tokenizer/cropper |
| `step8_report.py` | assemble `reports/MILESTONE_SUMMARY.md` |
| `fetch_assets.py` | resumable download of `boltz1_conf.ckpt` + `symmetry.pkl` |
| `msa_extract.py` | pull the required MSAs out of the 107 GB `rcsb_processed_msa.tar` by *hopping* tar headers |
| `make_baseline_config.py` | derive `configs/mhc1_baseline.yaml` from the fine-tune config |

Run `step7` with `.venv/Scripts/python.exe` so the Boltz pipeline check (V11) is
exercised; the other stages run on the system Python.

## Getting the data without storing 65 GB

The archive is a plain tar, so `data/tar_index/*.tsv` is built as a side effect of
scanning: `basename <TAB> offset <TAB> size` for every member. With it, fetching
any single structure later is one range request rather than another full walk.

Two things bite when range-scanning this archive:

* members are far larger than you would guess -- the biggest seen is 335 MB
  (`6cgr.npz`), and several exceed 300 MB. A worker that starts inside one must
  read past it before it finds a header, so the resync cap has to be generous
  (currently 2 GB). An earlier 102 MB cap silently made two of twelve workers
  give up immediately and skip their entire 5.4 GB range.
* `manifest.json` is the last member and is 772 MB. Its offset was found by
  binary-searching the binary->JSON transition rather than by walking the
  archive; `fetch_manifest.py` hard-codes the result.

## Getting the MSAs: hop the headers, do not stream them

`rcsb_processed_msa.tar` is 107 GB and we need 1,343 files out of it. Reusing
`tar_extract.py` would have been a mistake: it walks the archive as a buffered
stream with a 4 MB chunk, and `skip()` only avoids a download when the jump is
bigger than what is already buffered. The mean member here is ~490 KB, so every
skip would have been served out of a buffer that had already been paid for --
effectively downloading all 107 GB.

`msa_extract.py` exploits the fact that a tar header states its own member's
size, so the next header's offset is arithmetic, not a search:

```
next = here + 512 + size + (-size % 512)
```

So a worker range-requests 512 bytes, reads the size, and jumps straight to the
next header. Bodies are only ever fetched for members we actually want. That
moves ~110 MB of headers instead of 107 GB.

The trade is that the job stops being bandwidth-bound and becomes latency-bound:
~151k members at ~100 ms per round trip is 4+ hours on one connection. Two things
fix that, and both are in the code:

* **one persistent HTTPS connection per worker** (`http.client`, `Connection:
  keep-alive`) -- at this request count a TLS handshake per hop would dominate;
* **48 workers over disjoint byte ranges**, each resyncing to its first valid
  header before it starts hopping.

Measured: **151,410 members walked and all files fetched in 4.2 minutes**, 469 MB
on disk.

As a side effect it writes `data/msa_tar_index/*.tsv` (`basename, offset, size`).
That index earned its keep almost immediately -- when 288 more MSAs turned out to
be needed, `--use-index` fetched them with one range request each in seconds
rather than a second 107 GB walk.

## MSAs are needed for disabled chains too

The subtle one. `required_msa_ids.txt` was originally built from chains with
`valid == true`, on the reasonable assumption that a chain we masked out is a
chain Boltz never reads. That is wrong:

```python
# boltz/data/module/training.py:115, in load_input()
for chain in record.chains:
    msa_id = chain.msa_id
    if msa_id != -1 and msa_id != "":
        msa = np.load(msa_dir / f"{msa_id}.npz")
```

`record.chains` is iterated with no reference to the `valid` flag. Masking stops
a chain at the *tokenizer*, but `load_input` runs earlier and opens every chain's
MSA regardless. The manifest references 1,343 distinct `msa_id`s where the
valid-only rule produced 1,134.

The failure mode is worth remembering because it hides its own cause:
`TrainingDataset.__getitem__` catches load errors and retries with
`self.__getitem__(0)`, so a permanently-missing MSA on record 0 recurses until
`RecursionError: maximum recursion depth exceeded` -- 980 frames of Lightning
traceback and no mention of the absent file.

`step6_write.py` no longer filters on `valid`, and **V12** now asserts that every
`msa_id` referenced by either manifest exists on disk.

## The three findings that shape the design

**1. Boltz chain names come from `label_asym_id`, not `auth_asym_id`.**
Measured over 1152 protein chains: the name prefix matched `label_asym_id`
1152/1152 (100%) and `auth_asym_id` 1010/1152 (87.7%). TCR3D and the literature
quote auth IDs -- 1FZK's peptide is auth chain `P` but Boltz chain `C1` -- so
matching on auth IDs picks the wrong chain about 12% of the time. All mapping
here is done on sequence and only cross-checked against names.

**2. The numeric suffix is an assembly *operator* index, not a copy index.**
`rcsb.py` expands assembly 1 with `gemmi.HowToNameCopiedChain.AddNumber`. In
7RRG, assembly 1 is a single 5-chain complex emitted by two operator rows
(oper 1 -> A,B,E; oper 2 -> C,D), so `A1` and `C2` genuinely belong together.
Grouping by suffix and keeping one group would tear real complexes apart. A
genuine second copy instead shows up as the same `entity_id` with a different
`sym_id` (e.g. 3DTX: A1/B1/C1 and A2/B2/C2). Cross-copy pairing is prevented
geometrically: the MHC/peptide pair is chosen by heavy-atom contact count, and
every kept chain must contact the selected heavy chain.

**3. Chains are disabled, not deleted.**
Boltz reads two flags: the NPZ `mask[i]` (the tokenizer does
`chains = struct.chains[struct.mask]`, so a masked chain never reaches the
model) and the manifest `chains[i].valid` / `interfaces[j].valid` (the samplers
only anchor crops on valid entries). Both are set. Deleting chains via
`Structure.remove_invalid_chains()` would renumber `asym_id` and break the
manifest `chain_id` -> NPZ `asym_id` correspondence and the per-chain `msa_id`
mapping; masking keeps every index stable and is exactly reversible.

## Test-set handling

Boltz's `DatasetConfig.split` names the **validation** ids, and everything else
in the manifest goes to training -- with `filters` applied only to the training
side. So `data/processed/manifest.json` deliberately contains **train +
validation only**; test records are held in `manifest_test.json` and are absent
from anything training reads.

## Boltz's PDB snapshot ends in early 2024 — this caps the test split

Of the post-split candidates with no NPZ in `rcsb_processed_targets.tar`, almost
all are recent: **212 of 217 were released in 2024–2026**. The archive contains
only 19 MHC-I entries released in 2024 and none after, so Boltz-1's PDB snapshot
effectively stops there.

This matters for the AlphaFold-3-style split. The test bucket is "released after
2023-01-13", and TCR3D lists plenty of such structures — but most of them simply
do not exist as Boltz samples, so the test set here is far smaller than the
candidate count suggests. Train and validation are essentially unaffected
(both end well before the cutoff).

To get a full-size test set, those 2024+ structures have to be processed from
mmCIF with Boltz's own `scripts/process/rcsb.py` (see `boltz-src/docs/training.md`
step 7), which needs `ccd.rdb` + redis and a clustering file. That is the single
biggest outstanding item for evaluation, and it is independent of everything
above: the same chain-selection and validation stages here apply unchanged to
NPZs produced that way.

## Training setup (brief section 7)

All of the assets the config needs are now local and every `SET_PATH_HERE` is
filled except the wandb entity:

```
data/assets/boltz1_conf.ckpt   3.60 GB   606.4M params (453.6M once
                                         confidence_prediction: false)
data/assets/symmetry.pkl        215 MB
data/msa/                       469 MB   1,457 MSA .npz
```

`src/fetch_assets.py` downloads the first two and is resumable, because the link
has dropped mid-transfer before.

Two configs, one source of truth:

* `configs/mhc1_finetune.yaml` -- the fine-tuning config, faithful to upstream
  `structure.yaml` apart from the data wiring, `samples_per_epoch: 100`, and the
  `steering_args` block upstream forgot (see `reports/UPSTREAM_PATCHES.md`).
* `configs/mhc1_baseline.yaml` -- **generated** by `src/make_baseline_config.py`
  from the above. The brief's `validation_only` baseline is only meaningful if it
  evaluates identical data with identical validation settings, so it is derived
  rather than maintained by hand; the only permitted deltas are `validation_only`,
  `debug`, `output`, `disable_checkpoint` and `ema`.

`debug: true` is what lets the baseline run without a Weights & Biases account --
`train.py` reads it as "local sanity run" and sets `wandb = None` along with
`devices = 1` and `num_workers = 0`. The fine-tuning run still logs to W&B as the
brief asks.

Run it with:

```
cd boltz-src
../.venv-gpu/Scripts/python.exe scripts/train/train.py ../configs/mhc1_baseline.yaml
```

Note `matmul_precision: null`, kept identical to upstream. Setting it to `high`
would enable TF32 and speed up fp32 matmuls substantially on an Ada card, at a
small precision cost -- a reasonable knob, but it must be set the same way for
the baseline and the fine-tune or the comparison stops meaning anything.

## Still outstanding

* **Fine-tuning on this hardware is not possible.** The 8 GB RTX 4060 cannot hold
  weights + gradients + Adam states + EMA for 453.6M fp32 parameters (9.06 GB
  before any activation). `reports/GPU_REQUIREMENTS.md` has the arithmetic and
  the four things that would change the answer. Per the brief this is documented
  rather than worked around, since the fine-tuning strategy is meant to be agreed
  first.
* **A full-size test set** still needs the 2024+ structures processed from mmCIF
  with Boltz's own `scripts/process/rcsb.py` (needs `ccd.rdb` + redis). Unchanged
  from before, and independent of everything above.
* **The wandb entity** in `configs/mhc1_finetune.yaml`.
* **This directory is not under version control.** For anything lab-facing that
  should be fixed before more work lands on it.
