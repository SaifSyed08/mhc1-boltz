# Round 4 — steps 43 → 153 on three GPUs: structures flat, distogram learning

Job 3491007, `gpu-a100`, three A100s through torchrun, resumed from step 43
(`ALLOW_RESUME=1`; `max_lr` 1e-4 and warmup unchanged since 1ba819e). It ran the whole
session this time: started Oct 7 04:45, `Time limit reached. Elapsed time is 1 day,
23:00:18`, ended Oct 9 03:57 (1-23:12:02), exit 0. The Timer patch (a590380) is what
changed: the previous resume stopped after 70 minutes. 110 optimizer steps in one
session, 10 per epoch, an epoch every ~4 h 15 min including a ~5 min validation;
~35 s/batch averaged over an epoch, peak 36.98 GiB with offload on. No NCCL errors, no OOM, no tracebacks.

Every validation of the unfrozen run, so the two distogram columns have their history:

| val | step | `lddt_pp` | vs control | `complex_pp` | `intra` | `disto_pp` | `disto_intra` | `rmsd` | `best_rmsd` |
|---|---|---|---|---|---|---|---|---|---|
| control (pretrained) | 0 | 0.8858 | — | 0.8855 | 0.9245 | 0.9053 | 0.9635 | 3.044 | 2.967 |
| r1 epoch 0 (Round 2) | 10 | 0.8791 | −0.0067 | | | | | 3.079 | 3.011 |
| r1 epoch 1 | 20 | 0.8853 | −0.0005 | 0.8849 | 0.9253 | 0.9053 | 0.9641 | 2.617 | 1.679 |
| r1 epoch 2 | 30 | 0.8837 | −0.0021 | 0.8835 | 0.9248 | 0.9061 | 0.9643 | 3.063 | 2.987 |
| r1 epoch 3 | 40 | 0.8864 | +0.0006 | 0.8859 | 0.9251 | 0.9066 | 0.9645 | 3.061 | 2.973 |
| r1 stop (max_time) | 40 | 0.8871 | +0.0013 | 0.8867 | 0.9255 | 0.9066 | 0.9646 | 3.060 | 2.962 |
| r2 stop (max_time) | 43 | 0.8833 | −0.0025 | 0.8828 | 0.9259 | 0.9067 | 0.9646 | 3.059 | 2.981 |
| epoch 4 | 51 | 0.8864 | +0.0006 | 0.8862 | 0.9264 | 0.9077 | 0.9646 | 1.737 | 1.658 |
| epoch 5 | 61 | 0.8825 | −0.0033 | 0.8822 | 0.9260 | 0.9080 | 0.9648 | 3.079 | 2.984 |
| epoch 6 | 71 | 0.8847 | −0.0011 | 0.8842 | 0.9268 | 0.9076 | 0.9649 | 2.613 | 1.681 |
| epoch 7 | 81 | 0.8863 | +0.0005 | 0.8862 | 0.9263 | 0.9079 | 0.9650 | 3.050 | 2.981 |
| epoch 8 | 91 | 0.8844 | −0.0014 | 0.8839 | 0.9269 | 0.9081 | 0.9651 | 2.621 | 1.688 |
| epoch 9 | 101 | 0.8824 | −0.0034 | 0.8820 | 0.9266 | 0.9083 | 0.9652 | 3.047 | 2.965 |
| epoch 10 | 111 | 0.8840 | −0.0018 | 0.8837 | 0.9259 | 0.9080 | 0.9653 | 2.519 | 1.380 |
| epoch 11 | 121 | 0.8862 | +0.0004 | 0.8859 | 0.9257 | 0.9083 | 0.9654 | 2.614 | 1.690 |
| epoch 12 | 131 | 0.8871 | +0.0013 | 0.8866 | 0.9267 | 0.9087 | 0.9655 | 3.049 | 2.989 |
| epoch 13 | 141 | 0.8874 | +0.0016 | 0.8872 | 0.9269 | 0.9088 | 0.9655 | 3.038 | 2.977 |
| epoch 14 | 151 | 0.8868 | +0.0010 | 0.8866 | 0.9266 | 0.9086 | 0.9656 | 3.040 | 2.971 |
| stop (max_time) | 153 | 0.8855 | −0.0003 | 0.8851 | 0.9267 | 0.9086 | 0.9657 | 3.028 | 2.954 |

All 30 subset samples every time (weight 30, `scope: all`, world size 3 from step 51).
Raw states: `reports/ls6/unfrozen_r3/` (new), `unfrozen_r1/`, `unfrozen_r2/`,
`control/` (new: the control's own file, which the repo did not have).
Noise floor for `lddt_pp` ±0.002 (`BASELINE_ls6.md`).

## 1. Structure accuracy is flat through 153 steps

The twelve validations of this session span 0.8824–0.8874, mean **0.8853** against
the control's **0.8858**. None is above the control by more than the noise floor; two
are below it by more (0.8825 at step 61, 0.8824 at step 101). A line through the twelve
rises +0.0022 per 100 steps, p = 0.15 — not distinguishable from flat. Across all
eighteen validations, rank correlation with step is 0.22 (p = 0.39). `complex_pp`
tracks `lddt_pp` to within 0.0005 everywhere and says the same thing.

`best-epoch=13-step=141.ckpt` (0.8874, +0.0016) is the largest of twelve noisy reads
of a flat curve. It is not a better model, and should not be reported as one.

## 2. The trunk is learning — it just does not reach the structures

`disto_lddt` scores the trunk's distance predictions directly, with no diffusion
sampling, so it barely moves on identical weights (step 40 validated twice:
Δ 0.0000 / 0.0001). It rises steadily from the control onward:

* `disto_intra` 0.9635 → 0.9657 (+0.0022), Spearman ρ 1.00 against step
* `disto_pp` 0.9053 → 0.9086–0.9088 (+0.0035), ρ 0.96

That is 20–35× its own repeat noise, so training does change the model, consistently,
in the right direction. What it does not do is change the structures the diffusion
module builds from those representations: structure lDDT stays where it started.

## 3. Why this may be the expected answer

> **Weakened by `MHCFINE_COMPARISON.md`.** MHC-Fine's training set also predates
> its base model's cutoff, and it still gained ~0.9 Å of peptide RMSD. So the
> argument below is at most part of the story; the measurement (no peptide metric)
> and the batch size (~16× fewer updates) are better-supported candidates.

Every training structure was released on or before 2021-09-30 (`split_report.json`:
`train_end`), which is Boltz-1's own training cutoff, and the training config applies
upstream's own `DateFilter` and `ResolutionFilter`. So the pretrained model was almost
certainly trained on these same structures already. Fine-tuning here re-weights the
model toward MHC-I; it shows it no structure it has not seen. Validation
(2021-10 → 2023-01) and test (after 2023-01) are new to it. A flat result is what
"more emphasis on data it already fits" should produce, and it says nothing yet about
whether *new* MHC-I structures would help.

The only structures in this dataset that are new to Boltz-1 and not used for
evaluation are the **69 validation-era records outside subset30**. Today they are used
by nothing: `split` sends them to the training side and the `DateFilter` then drops
them. Training on them, after an (allele, peptide) leakage check against subset30, is
the experiment that tests new data rather than more emphasis. It changes the split
design, so it is a decision for Saif and Ernest, not a default.

## 4. The RMSD excursion: most likely one structure, not a chain-selection bug

`rmsd` sits near 3.04 in seven validations and drops in five (steps 51, 71, 91, 111,
121); steps 71/91/121 land on 2.613/2.621/2.614 with `best_rmsd` 1.681/1.688/1.690,
the same values as r1's step 20 (2.617/1.679). The drop in `best_rmsd` is three times
the drop in `rmsd` (1.29 vs 0.43 Å). `rmsd` averages 90 values (30 structures × 3
samples) and `best_rmsd` 30, so a 3:1 ratio is exactly what one structure produces when
one of its three samples comes in ~39 Å closer. The simplest reading: one complex is
~40 Å off in all three samples most of the time, and is worth ~1.3 Å of the mean
`best_rmsd` on its own. `lddt_pp` does not move between the two modes, so whatever is
wrong with that structure is something distance-based lDDT cannot see. A mirror-image
prediction has exactly that signature (a reflection preserves every distance); that is
a candidate, not a finding.

Ruled out: chain selection. Every valid chain in all 1,084 records touches another
valid chain (≤ 6 Å), and in all 30 subset records the peptide lies along the groove
(≥ 60% of its residues within 4 Å of the heavy chain) and β2m, where present, has a
normal-sized interface (`8d5q` has none: a 180-residue α1α2 heavy chain and a 10-mer).
Which structure it is cannot be read from summed metric states; per-sample validation
output (below) will name it. Until then RMSD stays out of every conclusion, as before.

Side result of that check, for the dataset rather than this run — nine records whose
valid chains are not the usual heavy chain + β2m + peptide: `5iue`, `5knm` (two
heavy-chain-length chains, no β2m), `5f1n`, `7zqi`, `7zqj` (a valid chain whose length
fits none of heavy chain, β2m or peptide), `4uq2` (4 of 7 peptide residues within 4 Å of the heavy chain),
`1tmc`, `5e6i`, `8fjb` (43–59 β2m atoms within 4 Å of the heavy chain; median 84).
Six are training, three test; none is in subset30. Not yet examined one by one.

## 5. Next: is it the learning rate?

Submitted 2026-10-09, both fresh from the pretrained weights, same code and config as
3491007 except `model.training_args.max_lr`:

| job | `max_lr` | run dir |
|---|---|---|
| 3499581 | 3e-4 | `runs/ls6_lr3e-4` |
| 3499582 | 1e-3 | `runs/ls6_lr1e-3` |

Each should reach ~110 steps in one session. Read against the 1e-4 trajectory above:
if `disto_lddt` climbs faster but `lddt_pp` stays flat, the structures are not
rate-limited and new data (section 3) is the lever; if `lddt_pp` falls, the rate is
too high for pretrained weights; if it rises, 1e-4 was the limit. The test set stays
unread: nothing has beaten the control on validation, so there is nothing to confirm.

Two things these runs still cannot show, to be added after them (they run
already-tested code on purpose): **the training loss** — with `wandb: null` there is
no logger, so every `self.log("train/...")` value Boltz produces is discarded and no
run so far has a loss curve — and **per-sample validation scores**, which would name
the RMSD structure and show which complexes move.

## Bridge bug that corrupted a saved result

`reports/ls6/unfrozen_r2/val_state_epoch004_step00043.json` was missing its final `}`.
A file with no trailing newline ends on a line that the bridge's done-marker gets
appended to, and the marker filter dropped that whole line. Restored from LS6's backup
copy (`runs/ls6_unfrozen/r2_ddp_step43/`); the content is otherwise identical.
`scripts/ls6_bridge.sh` now strips the marker instead of the line.
