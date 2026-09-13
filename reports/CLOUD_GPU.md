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

A 16 GB T4 gives about 15.0 GiB usable after the CUDA context. Against ~7.3 GB of
demand that is better than 2x headroom. This is not a close call, which is why
the notebook probes it in a few steps rather than assuming it, but the arithmetic
says it should pass comfortably.

One consequence: `offload_to_cpu` is **turned off** in the T4 config, reversed
from `mhc1_finetune_8gb.yaml`. Pushing activations to host RAM is the right trade
when VRAM is the binding constraint; on a free Colab box the host has ~13 GB and
2 vCPUs, so it just relocates the bottleneck. Activation checkpointing stays on.

## 3. Wall clock: the constraint that replaced memory

Memory stops being the problem on a T4. Time becomes it, and it is worth being
blunt about the size of the change.

A T4 is Turing. It has **no bf16 and no TF32** — the two cheap precision wins
available on anything Ampere or newer simply do not exist on this card, so
`precision: 32` runs on plain CUDA cores at ~8.1 TFLOPS. The laptop 4060 is Ada
at ~11.6 TFLOPS fp32. Expect the T4 to be roughly **1.4x slower per batch** than
the 4060 was, before considering that the 4060 never completed a clean batch to
measure.

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
| 2 | trunk frozen | saves gradients, Adam moments and retained activations for 150.6 M params | a genuine strategy choice, endorsed by Ernest; not only a memory hack |
| 3 | `offload_to_cpu: false` | host RAM is scarcer than VRAM here | none — pure win at 16 GB |
| 4 | `accumulate_grad_batches: 16` | 128 would mean ~1 optimizer step per epoch | effective batch 16, not 128. **Report this.** |
| 5 | `save_top_k: 1` | a Lightning checkpoint here is ~4 GB (1.73 GB weights + 2.25 GB Adam); `-1` fills a 15 GB Drive in three epochs | keeps best + `last.ckpt`, which is what a resumable run needs |
| 6 | `num_workers: 2` | free Colab has 2 vCPUs | possible dataloader stall; watch for it in the probe |

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
notebook asserts 1,084 structures after unpacking, so a truncated upload fails
immediately instead of silently training on a subset.

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
