# Second training run: the Round 6 arm ordering does not survive, and why

Training jobs ran 2026-11-17 → 2026-11-19 (~47 h each, three A100s). Eval jobs
3615394 and 3615395 ran 2026-11-27, 38 min each, on `gpu-a100-dev`; the chained
digest job 3615396 produced `runs/eval_val99/SEED2_DIGEST.txt`.

Round 6 left the three arms separated by less than the draw-to-draw diffusion
noise, so the ordering could have been noise. This asks the one question a second
draw cannot answer: **does the ordering survive retraining from scratch?**

Both Round 6 arms were retrained with no other change:

* **abrupt:** MSA cap 1 from the first step, validated at depth 1;
* **b9 3e-4:** the control, trained with full MSAs, scored query-only.

`scripts/ls6_submit_seed2.sh` documents why "seed 2" is a misnomer kept for
continuity: `boltz-src/scripts/train/train.py` sets no global seed, and
`data.random_seed` reaches only `ValidationDataset`
(`data/module/training.py:632`). So an unseeded rerun already differs in data
order, new-head init, augmentation and training diffusion noise, which is exactly
the variation being measured. `data.random_seed` stayed 42 so the validation
sampling is identical to run 1 and the numbers stay poolable.

Each checkpoint was evaluated **twice**. Pooling is a plain mean over all 6
samples per record (`seed2_digest.py:63`), identical for all four checkpoints, so
every comparison below is on equal footing. Submitting two draws for run 2 was
deliberate: scoring one draw against run 1's two pooled draws would have loaded
all the draw noise onto one side.

Metric throughout is peptide Cα RMSD after superposing on the α1/α2 groove.
"Held-out 69" is the 99 minus subset30, which played no part in training or in
picking checkpoints. Both arms stopped at the same epoch as their run-1
counterpart off the same wall clock (abrupt epoch 11 both, b9 epoch 10 both), so
this is a matched replicate, not a longer or shorter run.

## The verdict criterion was fixed before the data

`scripts/seed2_digest.py` was written and committed (`3be3ced`) before either
eval finished. Its closing lines:

> A sign that FLIPS, or an effect under the noise in either seed, means the arms
> are not separated by this experiment. Say that plainly if so.

It flips. Both halves fail.

## The numbers

Pooled medians, 6 samples per record:

| arm | run | all 99 | held-out 69 | mean (all 99) | p90 (all 99) |
|---|---|---|---|---|---|
| abrupt | 1 | 0.744 | 0.728 | 0.989 | 1.737 |
| abrupt | 2 | 0.779 | 0.727 | 0.992 | 1.663 |
| b9 3e-4 | 1 | 0.763 | 0.749 | 0.984 | 1.834 |
| b9 3e-4 | 2 | 0.823 | 0.784 | 1.018 | 1.815 |

Arm effect, paired per-record within each run, two-sided sign test. Negative
means abrupt is better:

| run | all 99 | held-out 69 |
|---|---|---|
| 1 | +0.016, 41/99, p = 0.107 | +0.014, 29/69, p = 0.228 |
| 2 | −0.011, 52/99, p = 0.688 | −0.012, 37/69, p = 0.630 |

The sign is consistent across the two subsets within each run, and opposite
between runs. Held-out 69 is a subset of the 99, so those two columns are one
observation shown twice, not corroboration.

Seed effect, paired, run 2 minus run 1:

| arm | all 99 | held-out 69 |
|---|---|---|
| abrupt | −0.006, 52/99, p = 0.688 | −0.029, 42/69, p = 0.091 |
| b9 3e-4 | −0.003, 51/99, p = 0.841 | −0.007, 38/69, p = 0.470 |

Draw-to-draw noise, same checkpoint evaluated twice:

| arm | run | per-record median &#124;diff&#124; |
|---|---|---|
| abrupt | 1 | 0.053 |
| abrupt | 2 | 0.054 |
| b9 3e-4 | 1 | 0.062 |
| b9 3e-4 | 2 | 0.088 |

## 1. The arms are not separated

The effect is +0.014 Å in run 1 and −0.012 Å in run 2 on the held-out 69. No
sign test anywhere reaches p < 0.1. The pre-committed criterion is failed on the
sign alone.

**This is a statement about the experiment's reach, not about the recipes.** The
arm effects are 0.011 to 0.016 Å. Retraining the same recipe moved the pooled
median by 0.035 Å (abrupt, all 99) and 0.060 Å (b9, all 99). Training-seed noise
is two to four times the effect being tested, so a real difference of 0.02 Å or
less could not have been detected by this design. Differences above roughly 0.03
to 0.05 Å would likely have shown.

The right conclusion is not "the schedules are equivalent." It is **"any schedule
effect is smaller than the seed noise in this setup."**

## 2. Correcting the yardstick Round 6 used, and this script inherited

`SEED2_DIGEST.txt` compares the arm effect against 0.058 Å, the median per-record
draw noise. That is the wrong quantity. The effect is a median over 69 to 99
records, whose sampling error is far smaller than the per-record noise, so the
comparison is conservative in a way that makes small real effects look like
nothing. The verdict survives because the sign flips and the sign tests are
non-significant, which is the argument that should be cited.

The yardstick that actually matters here is the **seed-to-seed** shift (up to
0.029 Å paired on held-out, up to 0.060 Å on pooled medians), not the
draw-to-draw shift. Round 6 s3a framed its own caution against draw noise for the
same reason, and that framing was too generous to itself.

## 3. Reproducibility is not established, and "within 0.006" was a misreading

The digest's `−0.006` is the **paired median of per-record differences**, not the
gap between the two runs' medians. Those gaps:

* all 99: abrupt 0.744 vs 0.779 (0.035); b9 0.763 vs 0.823 (0.060)
* held-out 69: abrupt 0.728 vs 0.727 (0.001); b9 0.749 vs 0.784 (0.035)

Per-record the two runs track each other closely. The headline median does not.
Both are true and they say different things: the model makes similar predictions
record by record, while the summary statistic moves around.

No seed effect reaches p < 0.05, but non-significance with n=2 runs is absence of
evidence, not equivalence. No equivalence margin was set in advance. Bounding
seed spread needs a third run per arm, or a bootstrap CI on the paired median
against a margin fixed beforehand.

## 4. Unresolved: the median is fragile on this distribution

Draw-to-draw **median** shifts are +0.045, −0.030, +0.051 and +0.035, three of
four positive, while the per-record noise is ~0.05 and the **means** move by
0.005 or less. For 99 records of iid sampling noise the median shift should be
nearer 0.01 to 0.015.

Both eval draws ran through `scripts/ls6_eval_val99.slurm` with the same
environment and no seed argument, so they differ only in inference sampling.
Nothing in the configuration explains the drift. The likely cause is that peptide
RMSD is strongly right-skewed (median ~0.75, mean ~1.0, p90 ~1.7), so the median
sits where small shifts in a dense pile move it, while the mean is anchored by the
tail. That would explain both this and section 3's per-record vs median split
with one mechanism.

Not treated as settled. It is a reason to report medians and means together, with
CIs, in every future comparison.

## 5. b9 run 2 is the noisiest checkpoint

Its draw-to-draw noise is 0.088 against 0.053 to 0.062 for the other three, 45 to
60% higher, and it also has the worst pooled median (0.823 all 99, 0.784
held-out). A less stable sampler together with a worse median is consistent with a
less converged run. Pooling six samples dilutes variance but not bias, so part of
run 2's arm effect may be a bad-run effect rather than a recipe effect. That makes
the sign flip less diagnostic, which strengthens "not separated" rather than
weakening it.

## What this changes

* **The Round 6 arm ordering is not a finding.** abrupt vs taper vs b9 cannot be
  ranked by this data. Any writing that ranks them should be corrected.
* **The MSA-free result itself is untouched.** Removing the MSA from a fine-tuned
  model is the Round 6 result that rests on p < 0.0001 effects of 0.1 to 0.24 Å,
  an order above the seed noise measured here. All four checkpoints land between
  0.727 and 0.823 Å median query-only, against 1.119 Å for pretrained Boltz-1
  query-only. That gap is real and this run confirms it is not seed-specific.
* **The detection floor for this setup is ~0.05 Å with two runs per arm.** Any
  future question finer than that needs more runs, not more draws. Draws were
  already at diminishing returns; seeds are the binding constraint.
* **Designate held-out 69 as the primary number in advance** from here on, and
  report all 99 as secondary, so the two stop being cited as independent.
