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

### That 7.3 GB figure was wrong, and the way it was wrong is the point

The prediction here was ~7.3 GB of demand against ~15.0 GiB usable, so better
than 2x headroom. **The first T4 run OOM'd at 12.37 GiB allocated of 14.56 GiB.**

7.3 GB was never a measurement of demand. It was the point at which an 8 GB card
happened to run out — a *lower bound*, read off a process that died early in the
trunk forward and never got as far as the structure module. Quoting it as the
requirement was the mistake. A card that dies tells you what it could not do, not
what the job needs.

What the 16 GB card revealed, by getting further:

| | |
|---|---|
| persistent state (weights + grads + Adam) | 5.11 GB |
| 48 pairformer block inputs retained by `checkpoint_wrapper`, `z[1,512,512,128]` fp32 @ 134 MB | **~6.4 GB** |
| working set, one 2.00 GiB triangular-attention allocation on top | ~0.9 GB |
| **total** | **~12.4 GB**, matching the 12.37 GiB reported |

The 6.4 GB is the surprise, and it is nearly pure waste. `activation_checkpointing`
retains each block's *input* so it can recompute the block during backward — but
the trunk is **frozen**, so there is no backward through it and nothing to
recompute. fairscale's wrapper stores the inputs anyway. 48 blocks x 134 MB of
pair representations sit in VRAM to serve a backward pass that never runs.

Two fixes, both of which change no numbers:

* **`offload_to_cpu: true`**, reversing the call made earlier in this document.
  The reasoning for turning it off — a free Colab box has ~13 GB of host RAM, so
  offloading relocates the bottleneck — is true on Colab, false on Kaggle
  (~30 GB), and beside the point either way, because the offload is not optional.
  It moves those 6.4 GB of retained inputs to host RAM. Same tensors, different
  place.
* **`BOLTZ_CHUNK_IN_TRAINING=1`**, our patch. Upstream gates triangular-attention
  chunking on `not self.training`, so training materialises the whole score
  tensor: `1 x 4 x 512 x 512 x 512 x 4 bytes` = exactly 2.00 GiB per allocation.
  That is the number in every OOM traceback in this repo, going back to the 8 GB
  runs. Chunked into slices of 128 it is the same arithmetic at a quarter of the
  peak. See `reports/UPSTREAM_PATCHES.md` section 5.

Neither is a recipe deviation. Both were available all along; neither was applied
because the 8 GB failure was misread as "needs a bigger card" when a good part of
it was "needs the memory knobs upstream already has".

### And a third thing, which was the actual bug

With both of the above applied the run reached batch 24 and then OOM'd at
**13.09 GiB** — higher than before. The growth is explained: Adam allocates
`exp_avg` and `exp_avg_sq` lazily on the *first optimizer step*, which with
`accumulate_grad_batches: 16` is batch 16, not batch 0. That is ~2.25 GB
appearing a third of the way into an epoch, and it is why a run can look healthy
for twenty batches and then die.

But the underlying problem was the freeze itself. **Freezing a module does not
stop its activations being retained.** Autograd keeps an activation if it is
needed for *some* backward pass, not if the local module's weights happen to be
trainable. `z` entering the pairformer is built by `z_init_1`, `z_init_2`,
`rel_pos` and `token_bonds` from `input_embedder` output — all of which were
still trainable. So gradients had to flow back through all 48 frozen blocks to
reach them, and every block's activations were kept to make that possible.

`freeze_trunk: [msa_module, pairformer_module]` therefore saved the optimizer
state for 150.6 M parameters and bought nothing at all on activations, which is
where the memory actually was.

The fix is to freeze everything upstream of the structure module, so that
`z.requires_grad` is False and no graph is built through the trunk in the first
place. It costs 1.6 M more frozen parameters out of 432 M, and it is a closer
match to the strategy as described to Ernest — *freeze the trunk, fine-tune the
structure module*. The original list was a partial version of that, and the gap
between "partial" and "complete" was about 6 GB.

Whether that is enough is now a question for `mem.jsonl` rather than for
arithmetic. Three OOMs on this project were diagnosed by reasoning about what
should be resident and two of those diagnoses were wrong, both times by reading a
number off a process that had already died and treating it as demand.
`src/mem_probe.py` records allocated, peak, reserved and host-free memory per
batch, so the next one is answered from a file.

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
| 1 | `ema: false` | saves 1.81 GB | real. Boltz-1 trained with EMA and its released weights are EMA weights |
| 2 | everything upstream of `structure_module` frozen | 152.2 M params; keeps `z.requires_grad` False so no graph is built through the trunk at all | a genuine strategy choice, endorsed by Ernest; freezing only msa+pairformer was a partial version that saved optimizer state but no activations |
| 3 | `offload_to_cpu: true` | 48 pairformer block inputs (~6.4 GB) are retained for a backward pass a frozen trunk never runs | none numerically; costs PCIe traffic per step |
| 4 | `accumulate_grad_batches: 16` | 128 would mean ~1 optimizer step per epoch | effective batch 16, not 128. **Report this.** |
| 5 | `save_top_k: 1` | a Lightning checkpoint here is ~4 GB (1.73 GB weights + 2.25 GB Adam); `-1` fills a 15 GB Drive in three epochs | keeps best + `last.ckpt`, which is what a resumable run needs |
| 6 | `num_workers: 2` | free Colab has 2 vCPUs | possible dataloader stall; watch for it in the probe |
| 7 | `BOLTZ_CHUNK_IN_TRAINING=1` | upstream disables triangular-attention chunking during training, forcing a 2.00 GiB allocation per call | none — identical math in slices of 128, at some speed cost |

Deliberately **not** changed: `max_tokens: 512`, `diffusion_multiplicity: 16`,
the crop settings, and everything under `validation_args`. Those alter the
objective or the metric, and a number produced under a changed metric cannot be
compared against the pretrained baseline in `reports/BASELINE.md`. If the probe
comes back OOM they are the next levers, in that order, and each one would have
to be declared.

## 5. Surviving the 12 h cap

The run will be interrupted. The design assumes it:

* `save_last: true` is always on in the checkpoint callback, so `last.ckpt` sits
  in the run directory.
* The run directory lives on Drive (Colab) or `/kaggle/working` (Kaggle), not on
  the session filesystem.
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
