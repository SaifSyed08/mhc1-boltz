# First fine-tuning result — one epoch on a free Kaggle T4

Session 2026-09-16. Raw artifacts in this directory.

## The number

| metric | pretrained baseline | after 1 epoch | delta |
|---|---|---|---|
| **`val/lddt_protein_protein`** | **0.8827** | **0.8840** | **+0.0013** |
| `val/lddt_intra_protein` | 0.9248 | 0.9243 | −0.0005 |
| `val/complex_lddt_protein_protein` | 0.8825 | 0.8838 | +0.0013 |
| `val/complex_lddt_intra_protein` | 0.9239 | 0.9234 | −0.0005 |
| `val/rmsd` | 3.053 Å | 2.636 Å | −0.417 |
| `val/best_rmsd` | 2.982 Å | 1.694 Å | −1.288 |

Both sides measured on the same 30-sample subset with identical
`validation_args` — `recycling_steps: 3`, `sampling_steps: 200`,
`diffusion_samples: 3`, `symmetry_correction: true`. That last one matters: the
T4 config briefly carried `diffusion_samples: 5`, and `val/lddt` is best-of-N
(`model.py:704`), so it would have beaten the baseline on the metric definition
alone. Corrected before this run.

**Do not report `val/lddt`** (the aggregate). It is diluted across eight empty
categories and caps at 0.3846 on this dataset — see `reports/BASELINE.md`.

## How much training this represents

**Six optimizer steps.**

`samples_per_epoch: 100` at `batch_size: 1` is 100 batches; at
`accumulate_grad_batches: 16` that is 6 optimizer steps before the epoch-end
validation. From `mem.jsonl`, the clean run reached `global_step: 10` across 160
batches total, and validation ran at batch 100.

Six steps at effective batch 16, on a 280 M-parameter structure module. The
honest prior for "what should lDDT do after six steps" is *nothing*, and lDDT did
nothing.

## Reading it

**lDDT is flat.** +0.0013 on `protein_protein` across 30 samples is noise, not
signal. Nothing here says fine-tuning helps, and nothing says it does not — six
steps is not an experiment, it is a smoke test that the loop runs and the metric
is comparable. Both of those now hold, which is the actual deliverable of this
session.

**The RMSD movement is large and unexplained, and should not be reported on its
own.** `best_rmsd` 2.982 → 1.694 is a 43% drop. Two reasons to distrust it until
it is understood:

1. It disagrees with lDDT. If structures were genuinely much better placed, a
   local, per-residue measure should have moved too. It did not move at all.
2. The *relationship* between the two RMSDs changed shape. Baseline `rmsd` 3.053
   vs `best_rmsd` 2.982 is a gap of 0.07; this run is 2.636 vs 1.694, a gap of
   0.94. `best_rmsd` selects the sample with the best lDDT
   (`model.py` `best_idx`), not the lowest RMSD, so a 13x larger gap suggests the
   spread of RMSD across the three diffusion samples changed, not that the
   predictions got better.

RMSD is global and dominated by a few badly-placed outliers; lDDT is local and
averaged. One or two samples flipping from badly-placed to well-placed moves RMSD
a lot and lDDT barely at all. That is the most likely explanation and it is
checkable — per-sample RMSDs would settle it.

**Nothing from this run should go to Ernest as evidence of improvement.** What
should go is: the pipeline runs end-to-end on free-tier hardware, the metric is
now comparable to the baseline, and here is the compute budget for a real run.

## Run conditions

| | |
|---|---|
| GPU | Tesla T4, 14.56 GiB (Kaggle, `GPU T4 x2`, one used) |
| peak VRAM | 11.88 GiB — 2.7 GiB headroom |
| throughput | 61 s/batch median |
| epoch | 3 h 10 m (≈2 h training + 65 min validation) |
| skipped batches | **0** |
| optimizer steps at validation | 6 |

Zero skipped batches is worth stating explicitly: the earlier
`diffusion_multiplicity: 8` attempt skipped **48 of 100** on intermittent OOMs
(`finetune_mult8_48skips.log`), and a skipped batch contributes nothing to the
accumulation. That run's effective batch size varied silently and its output is
not interpretable. It produced no validation and is kept only as a record.

## Deviations from the upstream recipe, for any writeup

1. `ema: false`
2. everything upstream of `structure_module` frozen — 152.1 M of 432.2 M
3. `accumulate_grad_batches: 16` (upstream 128) → effective batch **16**
4. `diffusion_multiplicity: 4` (upstream 16, AlphaFold-3 48)
5. `max_atoms: 3904` (upstream 4608) — crops nothing, pure padding reduction
6. `BOLTZ_CHUNK_IN_TRAINING=1` — mathematically identical, not bitwise

(4) is the one that most needs saying aloud. At `batch_size: 1` it is a plain
M-sample Monte-Carlo estimate of a fixed objective, so lowering M raises gradient
variance without changing the estimand — but `synchronize_sigmas: true` means M
averages noise vectors at a *fixed* sigma rather than thinning the sigma
schedule. Reasoning in `reports/CLOUD_GPU.md` §4.

## What to change before the next session

* **`check_val_every_n_epoch: 2`** (already committed). Validation is 65 min of
  every 3 h 10 m epoch — over a third of the session spent measuring. Validating
  half as often changes nothing about the metric.
* **`max_time` is now computed from the session clock** (already committed). In
  this session, training began 6.5 h in after two failed configurations, so a
  trainer-relative 9 h 30 m budget would have run past Kaggle's 12 h wall.
* **Watch host RAM.** `mem.jsonl` shows `host_free_gb` falling 8.0 → 4.9 across
  the session. If that is a genuine leak rather than caching, a longer run hits a
  host OOM, which Kaggle kills without a useful message.

At 61 s/batch with validation every other epoch, a full 12 h session should yield
roughly 25–30 optimizer steps. Reaching a few hundred means several sessions
chained through `last.ckpt`, or a real GPU.
