# Running the fine-tune on a free cloud GPU

`reports/GPU_REQUIREMENTS.md` ends with "fine-tuning needs a bigger GPU", and the
plan at that point was TACC. TACC did not work out — there is no route to an
allocation through coursework at the moment. This document is the fallback, and
it is a better fallback than it sounds: the gap between "does not fit" and "fits"
for this model is 8 GB to 16 GB, and 16 GB is exactly what the free tiers hand
out.

Nothing here supersedes `GPU_REQUIREMENTS.md`. That document measures the model;
this one picks a machine to run it on and works out what that machine can and
cannot buy.

## 1. What the free tiers actually give you

| | Google Colab (free) | Kaggle Notebooks (free) |
|---|---|---|
| GPU | T4, 16 GB — *when available* | P100 16 GB, **or T4 x2** (32 GB total) |
| Session cap | 12 h | 12 h |
| Idle disconnect | ~90 min | on commit, none |
| Weekly budget | not published; roughly 15–30 h, varies with account history | **30 h, stated** |
| Runs with the tab closed | no | **yes** — "Save & Run All (Commit)" |
| Persistence | Google Drive (15 GB free) | Kaggle Datasets, `/kaggle/working` |
| Host RAM / vCPU | ~13 GB / 2 | ~30 GB / 4 |

**Kaggle is the primary, Colab the backup.** Two rows decide it. The weekly
budget on Colab is undocumented and demand-dependent, so a plan built on it is a
plan built on a number nobody will tell you. And background execution matters
enormously for a job whose unit of work is hours: on Kaggle you commit the
notebook and close the laptop, on Colab you keep a tab alive for twelve hours and
lose the session if the wifi drops.

**On Kaggle, select `GPU T4 x2`, not `GPU P100`.** Not for the second GPU — this
setup only uses one — but because the P100 is Pascal (sm_60) and recent PyTorch
wheels no longer ship kernels for it. The notebook checks compute capability in
its first cell and says so rather than letting you discover it twenty minutes in.

Using both T4s would mean sharding optimizer state across them with DeepSpeed
ZeRO or FSDP. That is real work, it is not what is blocking the next result, and
16 GB is already enough. Noted as an option, not taken.

## 2. Memory: why 16 GB clears a bar that 8 GB does not

From `GPU_REQUIREMENTS.md`, the frozen-trunk recipe (`msa_module` and
`pairformer_module` frozen, `ema: false`) instantiates 432 M parameters, of which
281.6 M are trainable:

| tensor | bytes/param | applies to | total |
|---|---|---|---|
| weights | 4 | all 432 M | 1.73 GB |
| gradients | 4 | 281.6 M | 1.13 GB |
| Adam `exp_avg` | 4 | 281.6 M | 1.13 GB |
| Adam `exp_avg_sq` | 4 | 281.6 M | 1.13 GB |
| **persistent state** | | | **5.11 GB** |

That state is the same on either card. What differs is what is left over for
activations, and the measured failure on the 4060 says exactly how much was
needed:

```
Skipping batch 1 due to error: CUDA out of memory. Tried to allocate 730.00 MiB.
GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free.
Of the allocated memory 6.52 GiB is allocated by PyTorch,
and 666.27 MiB is reserved by PyTorch but unallocated.
                                              -- reports/ft_final.log
```

So the frozen-trunk step wanted roughly **7.3 GB** at the moment it died, on a
card with 8.00 GiB total and a desktop already holding some of it. It was not far
over — it was *just* over, which is the most annoying place to be.

### Where the memory actually is

Four diagnoses were made here by reasoning from a traceback about what ought to
be resident. Three were wrong. This section is the corrected account; the history
is kept because the way it went wrong is the reusable lesson.

**The answer is `smooth_lddt_loss`** (`model/loss/diffusion.py:97`). It builds
`[batch x multiplicity, n_atoms, n_atoms]` fp32 matrices. At multiplicity 16 and
4608 atoms that is **1.27 GiB each** -- exactly the allocation in the traceback.
About ten are built while constructing the mask, and about **ten stay live for
the backward pass**: `pred_dists`, the `true - pred` difference, four sigmoid
outputs, the broadcast `eps`, and `mask`. Call it ~10 GiB retained plus ~6 GiB
transient, before the score model is counted at all.

It dominates for a structural reason: the loss works over 4608^2 **atom** pairs
while the trunk works over 512^2 **token** pairs. An 81x larger pair dimension.

| | |
|---|---|
| weights (432.21 M x 4 B) | 1.61 GiB |
| gradients (280.07 M x 4 B) | 1.04 GiB |
| Adam, from batch 16 (280.07 M x 8 B) | 2.09 GiB |
| **`smooth_lddt_loss`, retained** | **~10 GiB** |

(Units: this document previously mixed GB and GiB. It is GiB throughout now.
1.73 GB and 1.61 GiB are the same quantity and were presented as two different
measurements.)

### The three wrong diagnoses, and the single mistake behind them

**"It needs ~7.3 GB."** Read off `ft_final.log`, where an 8 GB card ran out.
That is a lower bound on demand, not a measure of it. Worse, the framing said the
process "died early in the trunk forward and never got as far as the structure
module" -- which is backwards. The `try/except` that prints `Skipping batch`
wraps **only** `structure_module.compute_loss`; `out = self(...)` is outside it.
Every "Skipping batch ... out of memory" line is a batch whose forward
**completed**. The OOM was always in the loss.

**"48 blocks retain 6.4 GB of pair representations."** This was *correct* for the
configuration it described, and the retraction of it was wrong. fairscale's
`checkpoint_wrapper` saves each block's inputs unconditionally
(`checkpoint_activations.py:278`) but **detaches** the output when nothing
upstream and no local parameter requires grad (`:297`, `:199`/`:210`), which
releases them. Under the *partial* freeze, `z` was built by
`z_init_*`/`rel_pos`/`token_bonds` from `input_embedder` -- all still trainable --
so nothing detached and all 48 blocks retained `z[1,512,512,128]` = 134 MB each.
Under the *full* freeze, they detach and nothing is retained.

The `None of the inputs have requires_grad=True` warning, which this document
quoted as proof the theory was wrong, fires under *exactly* the condition that
triggers the release. It was proof the mechanism was real and had just been
fixed.

That has a consequence: with the full freeze, **`offload_to_cpu` buys nothing** --
the tensors are released either way -- while still paying a device-to-host copy
of ~6.5 GB per step. It is now off, and that is a plausible slice of the measured
85 s/batch.

**"Memory does not scale with `diffusion_multiplicity`."** 13.61 GiB at 16 versus
13.92 GiB at 8 looked like a refutation. It was not: both figures are
`max_memory_allocated` on batches that **OOM'd**. `training_step` catches the
OOM and returns `None`, no backward runs, and the high-water mark is wherever the
allocator gave up. At multiplicity 16 the tensors are twice as large, so it fails
*earlier*; at 8, more of the smaller tensors fit first, so the mark climbs
*higher*. The inversion is the artefact. The code does scale linearly.

**The single mistake, three times: a number read off a process that had already
died, treated as a measure of demand.** It only ever marks where that process ran
out.

### What is actually being changed, and why

| lever | effect | objective change? |
|---|---|---|
| `max_atoms: 4608 → 3904` | shrinks every atom tensor ~28%, including the `smooth_lddt` matrices | **none** -- valid-chain atoms max out at 3896, so nothing is cropped |
| `offload_to_cpu: true → false` | removes a ~6.5 GB device-to-host copy per step that the full freeze made pointless | none |
| full `freeze_trunk` list | no autograd graph through the trunk at all | it *is* the stated strategy |
| `diffusion_multiplicity` (sweep) | scales `smooth_lddt_loss` linearly | raises gradient variance; see below |
| `add_smooth_lddt_loss: false` | removes the dominant tensor family outright | **yes** -- drops an auxiliary loss term |

**Not** `max_tokens: 384`. Tokens max out at 494, so 512 crops nothing today;
384 would crop **588 of 1013 records (58%)** silently. That was listed as a lever
in an earlier version of this document and in the notebook's OOM hint. It was
wrong and it is removed.

On lowering `diffusion_multiplicity`: at `batch_size: 1` the `eps` averaging in
`smooth_lddt_loss` reduces exactly to the mean of the per-copy lDDTs, so the
objective is a plain M-sample Monte-Carlo estimate of a fixed target -- lowering M
raises variance without changing the estimand. Two caveats belong next to that
when reporting it: it holds *only* at batch_size 1 (the `view`/`repeat_interleave`
ordering would average across different structures otherwise), and
`synchronize_sigmas: true` means M is averaging over noise vectors at a **fixed**
sigma, not over the sigma schedule.

## 3. Wall clock: the constraint that replaced memory

Memory stops being the problem on a T4. Time becomes it, and it is worth being
blunt about the size of the change.

A T4 is Turing. It has **no bf16 and no TF32** — the two cheap precision wins
available on anything Ampere or newer simply do not exist on this card, so
`precision: 32` runs on plain CUDA cores at ~8.1 TFLOPS.

### Measured, 2026-09-15

The first run to get past the trunk did 24 clean batches in 34m06s:

**85 s per training batch**, with chunked triangular attention and the trunk
frozen. Everything below follows from that one number.

| | |
|---|---|
| per optimizer step (`accumulate_grad_batches: 16`) | 22.7 min |
| per epoch (`samples_per_epoch: 100`) | 2.4 h |
| one 12 h Kaggle session | ~5 epochs, **~32 optimizer steps** |
| a 30 h Kaggle week | ~12.7 epochs, **~79 optimizer steps** |

Validation is not in those figures and is expensive on top.

Thirty-two optimizer steps per session is thin, and it is worth saying plainly
what it does and does not support. It is enough to move a structure module that
starts from good pretrained weights, and to see whether validation lDDT moves in
the right direction. It is not enough to train anything to convergence, and any
plot of it will be a short one. The alternative to a thin fine-tune here is no
fine-tune, so it is still worth running — but the sample size belongs in the
writeup next to the result.

### One failure mode to watch for

`train.py` catches a CUDA OOM per batch and prints `Skipping batch N due to
error`, then continues. That is good for robustness and bad for interpretability:
a skipped batch contributes nothing to the accumulation, so a run that limps
through with intermittent skips has a silently varying effective batch size.

If `mem.jsonl` shows skips, the run is not describable as "effective batch 16"
and the right move is to fix the memory rather than let it ride.

fp16 AMP is available on a T4 and is not used. AlphaFold3-style triangular
attention is softmax over large additive biases, which is the canonical way to
overflow fp16, and a fine-tune that silently produces NaNs is worse than one that
is slow. If this ever moves to an L4 or A100, `precision: bf16-mixed` becomes
both safe and a large win.

### The scheduling problem this creates

`samples_per_epoch: 100` with `batch_size: 1` is 100 batches per epoch. The
upstream `accumulate_grad_batches: 128` therefore means the optimizer steps
**about once per epoch** — and on a T4 an epoch is hours. A full 12 h session
would buy a single-digit number of optimizer steps. That is not an experiment;
it is an expensive way to confirm the loss is finite.

So `configs/mhc1_finetune_t4.yaml` sets `accumulate_grad_batches: 16`, giving
~6 optimizer steps per epoch at an effective batch of 16. **This is a deviation
and it must be reported as one** — any result from this config is at effective
batch 16, not the upstream 128.

The notebook's section 7 measures seconds-per-batch on the actual card and prints
what it implies for a 12 h session and a 30 h week. Run it before committing to
anything; the projection is the deliverable, not the eight probe steps.

## 4. Every deviation from the upstream recipe, in one place

Someone reading a result needs this list, so it lives here rather than scattered
through config comments.

| # | change | why | cost |
|---|---|---|---|
| 1 | `ema: false` | saves **1.61 GiB** (432.21 M x 4 B). The 1.81 GB figure used earlier was 453.6 M params — the checkpoint's state_dict minus `confidence_module`, not the model that is actually instantiated with `confidence_prediction: false` | real. Boltz-1 trained with EMA and its released weights are EMA weights |
| 2 | everything upstream of `structure_module` frozen | 152.2 M params; keeps `z.requires_grad` False so no graph is built through the trunk at all | a genuine strategy choice, endorsed by Ernest; freezing only msa+pairformer was a partial version that saved optimizer state but no activations |
| 3 | `offload_to_cpu: false` | with the full freeze those block inputs are never retained, so the offload copies ~6.5 GB host-ward per step for nothing | none; removes a cost |
| 4 | `accumulate_grad_batches: 16` | 128 would mean ~1 optimizer step per epoch | effective batch 16, not 128. **Report this.** |
| 5 | `save_top_k: 1` | a Lightning checkpoint here is ~4 GB (1.73 GB weights + 2.25 GB Adam); `-1` fills a 15 GB Drive in three epochs | keeps best + `last.ckpt`, which is what a resumable run needs |
| 6 | `num_workers: 2` | free Colab has 2 vCPUs | possible dataloader stall; watch for it in the probe |
| 7 | `BOLTZ_CHUNK_IN_TRAINING=1` | upstream disables triangular-attention chunking during training, forcing a 2.00 GiB allocation per call | none — mathematically identical in slices of 128, at some speed cost (not bitwise identical: different cuBLAS kernel and reduction order) |
| 8 | `max_atoms: 3904` | 4608 is upstream's general-PDB size; valid-chain atoms here max out at 3896 | **none** — zero samples cropped, ~28% smaller atom tensors |
| 9 | `max_time: "00:09:30:00"` | a committed Kaggle run that overruns 12 h has its output upload skipped on a best-effort basis | none — bounds the session, not the model |
| 10 | `ckpt_every_minutes: 30` | an epoch is ~2.4 h plus validation; upstream saves once per epoch, so a session could end having written nothing | none |
| 11 | `validation_args.diffusion_samples: 3` | **CORRECTION, not a deviation.** Was 5; the 0.8827 baseline was measured at 3, and `val/lddt` is best-of-N (`model.py:704`). Leaving it at 5 would have beaten the baseline with no model improvement | restores comparability |

Deliberately **not** changed: `max_tokens: 512` (tokens max at 494, so it crops
nothing — and 384 would crop 58% of the set), the crop settings, and the rest of
`validation_args`.

`diffusion_multiplicity` is the next lever and section 7b of the notebook picks
it from measurement; `add_smooth_lddt_loss: false` is the one after that. Both
must be declared with any result.

### The comparability rule this project nearly broke

`val/lddt` is **best-of-N** (`model.py:704`:
`best_idx = all_lddt_dict[key].reshape(-1, n_samples).argmax(dim=1)`), so
`validation_args.diffusion_samples` is part of the metric's *definition*, not a
compute knob. The baseline for this project is
`configs/mhc1_baseline_subset30.yaml` — the config that actually produced
0.8827 — **not** `configs/mhc1_baseline.yaml`, which has never been run and uses
5. Any change to `validation_args` has to be made in both files or the comparison
is meaningless, and it will be meaningless in the flattering direction.

## 5. Surviving the 12 h cap

The run will be interrupted. The design assumes it:

* `save_last: true` is always on in the checkpoint callback, so `last.ckpt` sits
  in the run directory.
* The run directory lives on Drive (Colab) or `/kaggle/working` (Kaggle), not on
  the session filesystem.

### Kaggle's two storage limits, and why the layout changed

`/kaggle/working` is capped at **20 GiB** and is simultaneously the staging area
for a version's saved output. Putting the whole repo there — 3.8 GB of
re-downloadable assets plus a `boltz-src` checkout — spent a fifth of that budget
on files that get re-fetched anyway, and left them competing with ~8 GB of
checkpoints (`best` + `last`, ~4 GB each).

So the repo now goes to `/kaggle/tmp` (~60 GiB, not persisted, exactly right for
things that can be fetched again) and only `runs/t4` stays in `/kaggle/working`.

### Resuming on Kaggle is a manual chain, not a re-run

This is the part that genuinely differs from Colab and is easy to get wrong.
`/kaggle/working` starts **empty in every session**. It is not shared storage; it
is one version's output staging area. `last.ckpt` from a finished 12 h run is not
sitting there waiting for the next one.

Continuing a run means:

1. Confirm `runs/t4/last.ckpt` is in the finished version's **Output** (requires
   "Always save output" on the version).
2. In the next version, **+ Add Input → Your Work → Notebooks**, attaching the
   previous notebook's output. It mounts read-only under `/kaggle/input`.
3. Run normally — section 8 searches `/kaggle/input/**/last.ckpt` and picks it up.

Each 12 h session is one link in a chain and the chain is hand-assembled. At
~32 optimizer steps per session, a week of this is a handful of manual
reattachments, which is worth knowing before planning around it.

Also: **Save & Run All executes every cell from the top**, so an unpinned
committed run repeats the ~20 minute sweep in 7b before it starts training.
`MULTIPLICITY_OVERRIDE` in that cell exists to skip it.
* The notebook's section 8 detects `last.ckpt` and passes it as `resume=`. When
  `resume` is set, `pretrained` is correctly ignored
  (`scripts/train/train.py:130`) — you want the optimizer state back, not a fresh
  load of the pretrained weights.
* `ValProgressDump` (our patch, `src/val_progress.py`) snapshots the running
  validation metrics after every batch. That was a convenience on the laptop; on
  a session that can be killed mid-validation it is the difference between
  partial results and none.

Recovery is: re-run notebook sections 1–5, then section 8.

## 6. Data staging

`data/` splits three ways by how expensive it is to recreate:

| | what | size | how it gets there |
|---|---|---|---|
| regenerate from a URL | `data/assets/` — checkpoint, symmetry pickle | 3.8 GB | notebook runs `src/fetch_assets.py`; faster than pushing it through Drive |
| regenerate, but expensively | `npz_raw/`, `tar_index/`, `msa_tar_index/`, `manifest_source.json` | ~900 MB + a full S3 tar scan | not needed on the GPU box at all |
| **irreplaceable** | `data/processed/`, `data/msa/` | **~575 MB** | `src/make_cloud_bundle.py` → upload once |

`src/make_cloud_bundle.py` packs the third row and prints a sha256. Upload it to
Drive or make a Kaggle Dataset from it; every later session just untars it. The
notebook asserts 1,084 structures after staging, so a truncated upload fails
immediately instead of silently training on a subset.

Two Kaggle-specific things the notebook handles, because both fail confusingly:

* **Kaggle auto-extracts archives on dataset upload.** The dataset may therefore
  contain `data/processed/` and `data/msa/` directly rather than the tarball. The
  staging cell looks for both shapes, and symlinks the extracted tree out of the
  read-only `/kaggle/input` rather than copying 570 MB into the working quota.
* **Internet is off by default on Kaggle notebooks** and requires a phone-verified
  account (right sidebar > Notebook options > Internet). Without it the GitHub
  clone and the 3.8 GB asset download both fail. Cell 1 tests connectivity and
  says exactly which toggle to flip.

## 7. The one real upside of leaving Windows: `trifast`

`trifast` is a Triton kernel for triangular attention. Boltz soft-imports it
(`model/layers/triangular_attention/primitives.py:46`) and it was never
installable on the laptop, because Triton has no Windows build. Triangular
attention is precisely where every 8 GB run died.

On Linux it installs. `use_trifast` reaches the layer through the config, since
`pairformer_args` is splatted straight into `PairformerModule`
(`model/model.py:192`), so `model.pairformer_args.use_trifast=true` on the
command line is all it takes.

It is **off by default**. 16 GB does not need it, and an unvalidated kernel swap
underneath a result you intend to report is a bad trade. It is worth running as a
deliberate A/B — probe with it off, probe with it on, compare peak VRAM and
s/step — and that is how the notebook frames it.

## 8. Where Ernest's two suggestions land

Both of his technical suggestions are memory-and-signal arguments, and moving to
16 GB changes their urgency but not their merit.

**Shrinking the MSA blocks.** His reasoning is that MHC proteins all have similar
MSAs, so the MSA signal is weak here and the block is large. The second half is
worth checking against the actual parameter count before investing in it:
`msa_module` is **3.2 M parameters of 432 M** — under 1% (`GPU_REQUIREMENTS.md`,
"What is actually in the checkpoint"). Shrinking it saves very little *parameter*
memory. What it could still save is activation memory and time, since the MSA
module runs over `max_seqs: 2048` sequences, and that is a real cost. So the
idea is worth testing, but as an *activation* and *throughput* optimisation, not
a parameter one — and it is worth telling him that, because it changes what a
successful result would look like. The trunk is frozen in this recipe anyway,
which already removes its backward pass.

**Peptide templates.** Untouched so far, and unaffected by any of this. It is a
data-pipeline question rather than a GPU one, so it can proceed on the laptop in
parallel with cloud runs rather than waiting on them.

## 9. What this setup can and cannot support

**Can:** a frozen-trunk fine-tune at effective batch 16 that actually takes
optimizer steps, resumes across sessions, and is measured against the pretrained
baseline in `reports/BASELINE.md` on the same 30-sample subset with the same
`validation_args`. That is a real experimental result.

**Cannot:** a faithful reproduction of the upstream Boltz-1 recipe. EMA is off,
the trunk is frozen, and the effective batch is 16 rather than 128. A 40 GB A100
runs `configs/mhc1_finetune.yaml` as written; nothing on a free tier does.

The honest framing for the next conversation with Ernest is that free-tier compute
is enough to answer *"does fine-tuning the structure module on pMHC-I data improve
lDDT and RMSD over pretrained Boltz-1?"*, and not enough to answer *"what does
Boltz-1's own fine-tuning recipe do on this dataset?"*. The first question is the
one worth answering first anyway, and if it answers yes, that is a much stronger
case to bring to a cluster allocation than an estimate is.

## 10. Sources

Free-tier specifications change; these were checked 2026-09-14.

* [Google Colab free tier limits (2026)](https://joshthompson.co.uk/ai/google-colab-2026-guide-free-compute-automations-pro-tips/)
* [Google Colab GPU usage limits and alternatives](https://www.hivenet.com/post/google-colaboratory-gpu-complete-guide-to-free-cloud-gpu-access-and-limitations)
* [Kaggle: efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage)
* [Kaggle notebooks documentation](https://www.kaggle.com/docs/notebooks)
* [Kaggle free GPU/TPU: 30 hours per week (2026)](https://aicreditmart.com/ai-credits-providers/kaggle-free-gpu-tpu-30-hours-week-access-guide-2026/)
