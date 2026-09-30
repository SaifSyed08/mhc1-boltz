# Round 2 — the control, the resumed frozen run, and the unfrozen run

> **Corrected by `RESULT_round3.md`.** The unfrozen row is that run's epoch 0 only,
> written while the job was ~15 h in. Its −0.0067 was gone by step 20, and the run
> finished flat at 40 steps. "One epoch in 47 hours / 132 s/batch" is wrong: the log
> shows 31.5 s/batch, 11 h per epoch. Sections 3–5 below rest on one or both.

Three jobs, and between them they close the RMSD question, kill a tempting false
positive, and produce the first result that is outside the noise band.

| run | config | opt. steps | `lddt_pp` | vs control | `rmsd` | `best_rmsd` |
|---|---|---|---|---|---|---|
| LS6 baseline | baseline | 0 | 0.8843 | −0.0015 | 3.061 | 2.978 |
| **control** | fine-tune | 0 | **0.8858** | — | 3.044 | 2.967 |
| frozen, run 1 | fine-tune | ~80 | 0.8821 | −0.0037 | 2.640 | 1.703 |
| frozen, resumed | fine-tune | ~160 | 0.8839 | −0.0019 | 3.064 | 2.983 |
| **unfrozen** | fine-tune | ~10 | **0.8791** | **−0.0067** | 3.079 | 3.011 |

Noise floor ±0.002, from the environment-only comparison in `BASELINE_ls6.md`.
All five runs covered all 30 validation samples.

## 1. The control changes which baseline is correct

Pretrained weights, fine-tune config, `validation_only`: **0.8858**, against the
baseline config's **0.8843** on identical weights. The config alone is worth
+0.0015 — inside the noise band, but systematically in the flattering direction.

**So the fine-tuned runs must be compared to 0.8858, not 0.8843.** Anything else
credits the config to the training. The table above uses the control throughout.

## 2. The RMSD improvement is dead, twice over

`best_rmsd` 2.982 → 1.703 was the most exciting number this project produced. It
is not real:

* **The control rules out the config.** Pretrained weights under the fine-tune
  config give `rmsd` 3.044, not 2.640. The excursion came from the weights.
* **But it did not survive more training.** The same run, same settings, at ~160
  steps instead of ~80: `rmsd` 3.065, `best_rmsd` 2.983. Back to baseline.

A real improvement does not reverse itself by training longer at unchanged
settings. Combined with Kaggle landing on the same 2.640 from entirely different
hyperparameters, the coherent reading is a transient state early in training that
happened to align coordinates well, not anything learned.

`rmsd` moved ±0.026 across five runs whose lDDT stayed within ±0.007. It is the
noisier metric here by roughly 4x, exactly as `BASELINE_ls6.md` warned.

## 3. Unfreezing made it worse, and that is the first signal outside noise

−0.0067 on `lddt_pp` against the control, −0.0026 on `intra_protein`, and both
RMSDs up. Every metric degraded, consistently, which is not what noise looks
like.

**But it only managed ~10 optimizer steps.** One epoch in 47 hours — about
132 s/batch against the frozen run's ~16.5, an 8x slowdown from backpropagating
through 48 pairformer blocks plus offloading their activations to host RAM.

Ten steps at `max_lr: 1e-4` reaching full rate exactly at step 10. So the honest
reading is **not** "unfreezing does not work". It is:

> Unfreezing at 1e-4 degrades the pretrained weights within the first ten
> optimizer steps, and the run is too slow to tell whether it recovers.

Two explanations, and they need different fixes:

* **1e-4 is too high for the trunk.** Plausible: it is a reasonable rate for a
  decoder head, less so for 147 M parameters of pretrained pairformer. Standard
  practice would be a discriminative rate — trunk at 1e-5 or lower, structure
  module at 1e-4.
* **Ten steps is just early disruption** that more training would recover from.
  Cannot be distinguished from the above without more steps.

## 4. Where this leaves the project

Nothing tested so far improves interface lDDT. Stated with the scope it has
earned:

* **Frozen trunk, ~160 steps at a ramping LR: flat** (−0.0019, inside noise).
* **Unfrozen, ~10 steps at 1e-4: worse** (−0.0067, outside noise).
* Every RMSD improvement seen has evaporated under a control or under more
  training.

That is a legitimate interim result. It is not "fine-tuning Boltz-1 on pMHC-I
does not work" — the frozen run never trained at a meaningful rate, and the
unfrozen run has had ten steps.

## 5. The blocker is now throughput, not memory

At 132 s/batch and 128 micro-batches per optimizer step, 100 optimizer steps is
~470 GPU-hours — ten 47-hour sessions. That is not a viable experiment loop.

Three levers, cheapest first:

1. **`matmul_precision: high`** (TF32). Ampere only, never available on the T4,
   and untried here. Potentially several-fold on the matmul-heavy trunk. It
   perturbs fp32 matmul arithmetic, so it needs its own control run against
   0.8858 before being trusted — but that is one validation pass.
2. **`offload_to_cpu: false`**, if the unfrozen peak in `mem.jsonl` allows it.
   Offload was enabled defensively on a projection of 31.65 GiB of 40. The run's
   own measurement should now say whether it was needed.
3. **Fewer validations.** `check_val_every_n_epoch` is 1 and validation is
   ~65 min. With epochs this long that is minor, but it compounds.

Only after throughput is fixed does a discriminative learning rate become worth
testing, because at ten steps per session nothing can be distinguished.
