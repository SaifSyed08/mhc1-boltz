# LS6 fine-tune, run 1 — lDDT flat, RMSD moved, and the RMSD is suspect

Job 3454273. A100-PCIE-40GB, started 2026-09-22 10:33, stopped 2026-09-24 09:59
on the 47 h `max_time` limit — as designed, so the checkpoint was written and the
resume job could take over. Reached epoch 8, roughly 80 optimizer steps at
effective batch 128.

| metric | laptop base | LS6 base | Kaggle FT | **LS6 FT** |
|---|---|---|---|---|
| `lddt_protein_protein` | 0.8827 | 0.8843 | 0.8840 | **0.8821** |
| `lddt_intra_protein` | 0.9248 | 0.9250 | — | **0.9246** |
| `rmsd` | 3.053 | 3.0611 | 2.640 | **2.6401** |
| `best_rmsd` | 2.982 | 2.9785 | 1.690 | **1.7032** |

## Result 1: lDDT did not move. That is a real finding.

`lddt_protein_protein` went 0.8843 → 0.8821, i.e. **down** 0.0022, against a
measured noise floor of ±0.002 (`reports/ls6/BASELINE_ls6.md`). Flat.

This is no longer the Kaggle excuse of "only six optimizer steps." This run did
~80 steps at the upstream effective batch of 128 — on the order of 10,000 samples
through the model, roughly 100x the Kaggle run — with EMA on and
`diffusion_multiplicity` at the upstream 16. The only remaining deviation is the
frozen trunk.

So, stated plainly: **freezing the trunk and fine-tuning only the structure
module does not improve interface lDDT on this dataset at this scale.**

### What this result is NOT

It is not evidence that fine-tuning Boltz-1 on pMHC-I fails. That has not been
tested. Read the scope carefully, because it is easy to overstate:

* **The frozen trunk was never a scientific choice.** It entered this project as
  a memory workaround — from the 2026-09-03 email, *"I was also considering
  freezing the trunk and only fine-tuning the structure module since that fits
  much more comfortably"* — and Ernest endorsed it as a reasonable idea, not as a
  validated strategy. The brief (`reports/ASSIGNMENT_TASKS.md` §8) specifies
  "compare against pretrained baseline" and says nothing about freezing anything.
* **The constraint that forced it is gone.** It existed because an 8 GB card
  could not hold gradients and Adam state for 432 M parameters. An A100 40 GB
  can: the projection from measured peaks is ~28.5 GiB for the fully unfrozen
  recipe, leaving ~11 GiB spare.

So what has actually been shown is that *one* strategy — the cheap one, chosen
under a hardware constraint — does not work. The obvious experiment has not been
run yet.

### Why freezing the trunk is a plausible reason for the flat result

There is a mechanism, not just an absence of one.

The trunk (MSA module + 48 pairformer blocks) is where the pair representation
`z` of the complex is built. The structure module is a diffusion decoder
*conditioned on* that representation. Freezing the trunk means the conditioning
signal for a given pMHC complex is **identical before and after fine-tuning** —
the decoder can only rearrange geometry within what it is told, and cannot learn
that this family of complexes should be represented differently.

That matters more here than it would for most targets, because of what
`reports/MSA_ANALYSIS.md` measured: **every peptide MSA in this dataset has depth
1.** The peptide contributes no evolutionary signal at all, so whatever the model
knows about where a 9-mer sits in a groove lives in the pairformer's learned
priors — exactly the part that was frozen.

**Next experiment: unfreeze and rerun.** `configs/mhc1_finetune_ls6_unfrozen.yaml`
and `scripts/ls6_unfrozen.slurm`.

## Result 2: the RMSD improvement is probably an artifact

`best_rmsd` 2.98 → 1.70 looks spectacular. It should not be believed yet, for a
reason that only became visible with two runs to compare:

```
                    steps  batch  mult   ema    rmsd    best_rmsd   lddt_pp
Kaggle fine-tune        6     16     4   off   2.640      1.690     0.8840
LS6 fine-tune         ~80    128    16   on    2.6401     1.7032    0.8821
```

Those two runs share almost no hyperparameters. One did six optimizer steps, the
other about eighty; the batch differs 8x, the multiplicity 4x, EMA is off in one
and on in the other. They land on the same `rmsd` to four decimal places.

Training thirteen times longer, on eight times the batch, and arriving at an
identical RMSD is not what learning looks like. It is what a **fixed difference
between two measurement setups** looks like.

The baseline and the fine-tunes differ in more than their weights: the baseline
runs `validation_only: true` through `trainer.validate()` with
`mhc1_baseline_subset30.yaml`; the fine-tunes validate inside `trainer.fit()`
with `mhc1_finetune_*.yaml`. Config diff on everything validation-relevant comes
to `max_atoms` (4608 vs 3904, inert while `crop_validation: false`),
`num_workers`, `samples_per_epoch`, and `ema` — none of which should move RMSD
0.4 Å while leaving lDDT untouched.

### The control that settles it

`scripts/ls6_control.slurm`: validate the **pretrained** checkpoint using the
**fine-tune** config. Weights held fixed, config and code path switched.

* RMSD ≈ 3.05 → the config is innocent, training really did move RMSD without
  moving lDDT, and that divergence becomes the interesting question.
* RMSD ≈ 2.64 → both fine-tuning RMSD numbers are artifacts and neither belongs
  in any writeup.

One validation pass, ~4.5 SU. Cheap against sending a supervisor a 43%
improvement that evaporates.

That RMSD is noisier than lDDT here is already established: `val/rmsd` moved
+0.0081 between two runs of an **unchanged** model (`BASELINE_ls6.md`).

## A bug this run exposed

```
ModelCheckpoint(monitor='val/lddt') could not find the monitored key in the
returned metrics: ['train/distogram_loss', ..., 'epoch', 'step']
```

Self-inflicted. Setting `ckpt_every_minutes: 30` switched `ModelCheckpoint` to
`train_time_interval`, which fires on training-batch boundaries — where only
`train/*` metrics exist. So `monitor="val/lddt"` never matches and `save_top_k`
never selects anything on validation quality. The surviving
`epoch=1-step=11.ckpt` is simply whatever the timer happened to write.

Impact is limited: `save_last` still works, so resume is unaffected, and over
eight epochs with a flat metric there was no meaningful "best" to pick. But
"best checkpoint" currently means nothing, and any longer run needs this fixed —
either drop the monitor and keep only `last`, or checkpoint on epoch boundaries
where `val/lddt` exists.

## Cost

47 h on `gpu-a100` at 3.0 SU/node-hour ≈ **141 SU** of 78,584. The run also
carries ~9.5 GB of checkpoints in `/work`, against a 1 TB quota.
