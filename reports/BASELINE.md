# Pretrained-model validation baseline

Brief section 7: *"Before fine-tuning, run the pretrained Boltz model on the
validation set using `validation_only` to establish the baseline performance
before training."*

```
cd boltz-src
../.venv-gpu/Scripts/python.exe scripts/train/train.py ../configs/mhc1_baseline_subset30.yaml
```

Pretrained `boltz1_conf.ckpt`, no training, no weight updates. The delivered
result below is 30 of the 99 validation samples -- see "Deviations". The
canonical full-set config is `configs/mhc1_baseline.yaml`.

## Do not report `val/lddt`. It cannot exceed 0.385 on this dataset.

This is the single most important thing to know before comparing a fine-tuned
model against this baseline, and it is easy to get wrong: `val/lddt` is the metric
Lightning puts in the progress bar *and* the one `ModelCheckpoint` monitors
(`monitor="val/lddt"` in `scripts/train/train.py`).

It is not an lDDT. It is a **fixed-weight average over ten modality categories**
(`model.py:1045`):

```python
overall_lddt = sum(avg_lddt[m] * w for (m, w) in const.out_types_weights.items()) \
             / sum(const.out_types_weights.values())
```

and a category with no data contributes **0.0**, not nothing —
`avg_lddt[m] = 0.0 if torch.isnan(avg_lddt[m])` at `model.py:875`. The weights
total 104:

| category | weight | present in an MHC-I / peptide dataset? |
|---|---|---|
| `intra_protein` | 20 | yes |
| `protein_protein` | 20 | yes |
| `ligand_protein` | 20 | no |
| `intra_ligand` | 20 | no |
| `intra_rna` | 8 | no |
| `dna_protein` | 5 | no |
| `rna_protein` | 5 | no |
| `dna_ligand` | 2 | no |
| `rna_ligand` | 2 | no |
| `intra_dna` | 2 | no |

Our dataset is entirely protein, by construction — MHC-I heavy chain, peptide,
and β2-microglobulin. So 40 of 104 weight units carry signal and the other 64
contribute a hard zero. **A perfect model would score 40/104 = 0.3846.**

Verified against the observed run rather than assumed: feeding only the two live
categories through that formula reproduces the logged value exactly.

```
intra_protein   0.9086085557937622  (w=20)
protein_protein 0.8862317204475403  (w=20)
-> weighted mean over 104 = 0.345162
   logged val/lddt          = 0.3451615869998932
```

`val/disto_lddt` and `val/complex_lddt` are aggregated the same way and are
diluted identically.

This does **not** break checkpoint selection — the aggregate is a positive linear
combination of the two live categories, so it still ranks checkpoints correctly.
It breaks *reporting*: 0.345 reads like a broken model when the structures are
actually being predicted well.

### Report these instead

| metric | meaning here |
|---|---|
| `val/lddt_intra_protein` | within-chain geometry (MHC fold, peptide conformation) |
| `val/lddt_protein_protein` | inter-chain geometry — peptide in the groove, β2m interface |
| `val/rmsd`, `val/best_rmsd` | `best_` is the best of `diffusion_samples` draws |

`val/lddt_protein_protein` is the one that matters most for this project: getting
a peptide correctly seated in the MHC-I binding groove is an interface problem.

## RESULT -- pretrained Boltz-1 baseline, 30 validation samples

Completed 2026-09-03 07:39, 30/30 samples, no failures.
Config `configs/mhc1_baseline_subset30.yaml`; raw log `reports/baseline_subset30.log`;
state dump `runs/baseline/val_state.json`.

| metric | value |
|---|---|
| **`val/lddt_protein_protein`** | **0.8827** |
| **`val/lddt_intra_protein`** | **0.9248** |
| `val/complex_lddt_protein_protein` | 0.8825 |
| `val/complex_lddt_intra_protein` | 0.9239 |
| `val/rmsd` | 3.053 A |
| `val/best_rmsd` (best of 3) | 2.982 A |
| `val/lddt` (aggregate -- do not use) | 0.3476 |

`val/lddt_protein_protein = 0.8827` is the number the fine-tune has to beat.

These were produced two independent ways that agree exactly: Lightning's own
`on_validation_epoch_end` logging, and `src/combine_val_states.py` summing the
per-batch `MeanMetric` sufficient statistics written by `src/val_progress.py`.
Agreement to 4+ decimal places is the check that the chunk-merging path is sound,
which matters because that is how any future partial runs will be combined.

### Deviations from `configs/mhc1_baseline.yaml`

Both must be mirrored when the fine-tuned model is evaluated, or the comparison
is invalid:

1. **`diffusion_samples: 3`** (canonical: 5). Reduces cost ~40%. `best_rmsd` is
   therefore best-of-3, which makes it slightly pessimistic relative to the
   canonical recipe. Adopted under time pressure, not for technical reasons.
2. **30 of 99 validation samples**, seeded (`random.Random(42)`), ids in
   `data/processed/subset30_ids.txt`.

`offload_to_cpu: true` is *not* in this list: it changes memory placement only,
never a number.

### Why a single sample was a bad predictor

The n=1 smoke run reported RMSD 1.85 A. The n=30 result is 3.05 A -- 65% higher.
Same for the earlier n=3 partial (1.30 A). Per-sample difficulty varies a lot, so
neither the timing nor the accuracy of one sample generalises. Worth remembering
before quoting any number from a smoke run.

## Timing, and the EMA trap

One validation sample takes **2:00** on the RTX 4060 at the config's
`sampling_steps: 200`, `diffusion_samples: 5`, `symmetry_correction: true`. So 99
samples is ~3.3 hours plus ~5 minutes to load the checkpoint — an overnight run,
not a reason to rent a GPU.

That is only true with `ema: false`. With `ema: true` the same single sample did
**not finish in over 10 minutes**, because the two extra full-parameter clones
pushed the run to the edge of VRAM (7921 MiB of 8188) and the caching allocator
spent its time churning instead of raising `CUDA out of memory`. See
`reports/GPU_REQUIREMENTS.md` for why `ema: false` is provably a no-op for a
fresh checkpoint. Roughly a 5x speedup for zero change in the numbers.

## Note on the torchmetrics warning

```
UserWarning: The ``compute`` method of metric MeanMetric was called before the
``update`` method which may lead to errors, as metric states have not yet been
updated.
```

Expected, not a problem: the eight absent modality categories are never updated,
so `compute()` on them returns NaN, which the code maps to 0.0. It is the same
mechanism as the dilution described above.
