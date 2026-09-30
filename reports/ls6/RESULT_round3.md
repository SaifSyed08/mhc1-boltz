# Round 3 — the unfrozen run, all 47 hours of it

Job 3469187. It is the same run Round 2 reported as "unfrozen, ~10 steps" — Round 2
was written while it was still in its first day. This is the whole thing.

| val | opt. steps | `lddt_pp` | vs control | `intra_protein` | `rmsd` | `best_rmsd` |
|---|---|---|---|---|---|---|
| control | 0 | 0.8858 | — | | 3.044 | 2.967 |
| epoch 0 (Round 2) | 10 | 0.8791 | −0.0067 | | 3.079 | 3.011 |
| epoch 1 | 20 | 0.8853 | −0.0005 | 0.9253 | 2.617 | 1.679 |
| epoch 2 | 30 | 0.8837 | −0.0021 | 0.9248 | 3.063 | 2.987 |
| epoch 3 | 40 | 0.8864 | +0.0006 | 0.9251 | 3.061 | 2.973 |
| stop (max_time) | 40 | 0.8871 | +0.0013 | 0.9255 | 3.060 | 2.962 |

All 30 validation samples every time. Raw states: `reports/ls6/unfrozen_r1/`.
Noise floor ±0.002 (`BASELINE_ls6.md`).

## What ran

Fresh from the pretrained checkpoint, **one GPU** (`CUDA_VISIBLE_DEVICES: [0]`), code at
1ba819e: LS6 pulled it and submitted this job in the same second (Sep 24 21:02), and
the job then queued until Sep 28 02:12. Trunk unfrozen (432 M trainable), activation
offload **off**, 1280 micro-batches per epoch, accumulate 128 → 10 optimizer steps per
epoch, warmup 10 steps to `max_lr` 1e-4. `max_time` stopped it 97 batches into epoch 4,
short of the 128 needed for another step.

## Results

1. **Flat.** The −0.0067 at 10 steps was a transient: gone by step 20, and steps 20–40
   sit inside the noise band around the control. Round 2's "unfreezing hurt, the first
   result outside noise" describes the 10-step point only and should not be quoted as
   the outcome. Forty steps at 1e-4 neither helps nor hurts.
2. **A free noise measurement.** Epoch 3 and the stop validation scored *identical
   weights* (no optimizer step in between): Δ`lddt_pp` 0.0008, Δ`rmsd` 0.0003,
   Δ`best_rmsd` 0.011. Pure sampling noise, consistent with the ±0.002 floor.
3. **The RMSD excursion again.** Epoch 1: `rmsd` 2.617, `best_rmsd` 1.679, gone at the
   next validation while lDDT did not move. Frozen run 1 showed 2.640 / 1.703 and Kaggle
   2.640. Three runs in different training states landing within 0.03 Å of each other
   is more like something in the evaluation than in the weights. Unexplained. Treat a
   one-epoch RMSD drop as suspect until it is.

## Two Round 2 numbers were wrong

* **Speed: 31.5 s/batch, not 132.** The log shows 1280 batches in 11 h 11 min. 132 was
  47 h ÷ 1280, but the job had been running ~15 h, not 47, when Round 2 was written.
  One GPU therefore does ~42 optimizer steps per 47 h job, not ~10.
* **Memory: 36.36 GiB peak** of 39.49 over 1305 batches with offload off. 0082bc1
  turned offload on from an estimate; on one GPU it was not needed. Under DDP the
  gradient buckets add 432 M × 4 B = 1.6 GiB, which would leave ~1 GiB, so offload
  stays ON for the 3-GPU run until the smoke test shows what it costs per batch.

`val_state_epoch000.json` and `mem.jsonl` are not in `runs/ls6_unfrozen` although the
log shows both were produced; epoch 0 survives only as Round 2's table row.

## The 3-GPU commit (d906e80) would not have run

1. It deleted `_snapshot()` from `src/val_progress.py` but kept the call: `NameError`
   at the sanity-check validation, minutes into any job, so a resume queued on that
   code would have died before training a single batch.
2. Plain `python` inside a Slurm job cannot use 3 GPUs. Lightning sees `SLURM_NTASKS`,
   picks `SLURMEnvironment`, and assumes srun started one process per GPU. With `-n 1`
   that is one process on GPU 0 with a third of the batch, and nothing errors
   (`lightning_fabric/plugins/environments/slurm.py`, PL 2.4). Fixed by launching
   through `torchrun`, which Lightning detects ahead of Slurm.
3. Under DDP each rank validates a third of the set, so rank 0's file would have held a
   10-sample mean. The epoch-end snapshot now sums all ranks first (tested: two-process
   gloo run, 9/9 checks).

`scripts/ls6_ddp_smoke.slurm` now gates any 3-GPU job: 50 batches on the dev queue,
fails unless it sees world size 3 and validation weight 6.

## Next

Resume the run from its end-of-run checkpoint on 3 GPUs (`ALLOW_RESUME=1` is safe:
`max_lr` and warmup are unchanged since 1ba819e), chained `afterok` on the smoke test.
At 3× the step rate, one job would take it from 40 to ~165 steps, enough to tell flat
from slow — less whatever offload costs, which the smoke test prints as s/batch.
