# Round 5: batch 9 vs batch 129 — a small, consistent gain, the same for every arm

Four arms, each one 47 h session on 3 × A100, fresh from pretrained Boltz-1, code
822e470. Every validation is on subset30 and records peptide RMSD per structure
(`src/peptide_rmsd.py`). Digest from `scripts/ls6_round5.sh`, run 2026-10-12.

| job | arm | optimizer steps | state |
|---|---|---|---|
| 3499814 | batch 9, `max_lr` 1e-4 | 1,482 | completed (time limit) |
| 3499815 | batch 9, `max_lr` 3e-4 | 1,441 at digest time | still running, ends ~16:35 Oct 12 |
| 3499581 | batch 129, 3e-4 | 110 | completed (time limit) |
| 3499582 | batch 129, 1e-3 | 111 | completed (time limit) |
| 3499813 | DDP smoke test | 50 batches | **SMOKE PASS** (00:35 Oct 10, before any arm started) |

The smoke test confirmed that the training loss and finite per-structure peptide
rows come out of the real 3-GPU path. Its six rows reproduce the eval's hard cases
(`6zkx` 2.5–3.3 Å, `7cir` 2.4–2.6 Å).

## Training loss falls in every arm; batch 9 at 3e-4 falls furthest

Mean total `train/loss` over rank-0 micro-batches (first bucket → last bucket):

| arm | loss | diffusion | distogram |
|---|---|---|---|
| b9 1e-4 | 1.064 → 1.024 | 0.263 → 0.253 | 0.461 → 0.438 |
| b9 3e-4 | 1.061 → 1.007 | 0.262 → 0.248 | 0.457 → 0.430 |
| b129 3e-4 | 1.057 → 1.033 | 0.261 → 0.255 | 0.473 → 0.443 |
| b129 1e-3 | 1.058 → 1.020 | 0.261 → 0.252 | 0.473 → 0.441 |

* No arm diverged: no non-finite losses, not even at 1e-3.
* The LR stayed at `max_lr` for the whole session in every arm; it never decayed.
* Batch 9 runs at the same 34 s per micro-batch. It just steps 13× more often.

## Validation (subset30)

Pretrained reference on these 30 records:
* peptide RMSD, mean of per-record means: 1.458 Å; median 0.945 Å (from
  `runs/eval_val99/pretrained`);
* Boltz's `lddt_pp`: 0.8858 (control run).

Columns, averaged over each arm's validations:
* **lddt_pp:** Boltz's protein-protein lDDT.
* **paired Δ:** per-structure peptide RMSD minus pretrained, median over the 30.
* **improved:** how many of the 30 structures beat pretrained.
* **mean pep RMSD:** the mean of per-record means, second half of the run.

| arm | lddt_pp (all / 2nd half) | paired Δ (all / 2nd half) | improved, typical | mean pep RMSD, 2nd half |
|---|---|---|---|---|
| b9 1e-4 | 0.8881 / 0.8898 | −0.032 / −0.040 Å | 17–19 | 1.39 Å |
| **b9 3e-4** | **0.8926 / 0.8941** | −0.035 / −0.043 Å | 16–20 | **1.19 Å** |
| b129 3e-4 | 0.8887 / 0.8910 | −0.038 / −0.049 Å | 17–21 | 1.22 Å |
| b129 1e-3 | 0.8900 / 0.8908 | −0.034 / −0.044 Å | 15–21 | 1.28 Å |

What this shows:
1. **Every arm ends a little better than pretrained.** The paired peptide Δ is
   negative at 40 of 44 validations, about −0.04 Å, with 17–20 of 30 structures
   improving. Those 44 points are not independent replicates: neighbouring
   checkpoints are correlated, and all of them are scored on the same 30 records
   that select the checkpoints. Each arm's mean `lddt_pp` is above the control
   (0.888–0.893 vs 0.8858), but six single validations fall below it. Round 4
   ranged 0.8824–0.8874. Suggestive, not established: the held-out eval below
   decides it.
2. **The gain is small and nearly arm-independent.** Batch 9 took 13× more steps
   than batch 129 and ended in the same place on the paired Δ. So "too few updates"
   was not why Round 4 looked flat. This matches the eval: typical peptides are
   already at ~0.8 Å.
3. **Batch 9 at 3e-4 is the best candidate.** It has the highest mean `lddt_pp`
   (0.8926; the others 0.888–0.890), though most per-validation gaps are within
   the ±0.002 noise. It also has the lowest mean peptide RMSD, driven by the hard
   cases (below).
4. **The movement is in the hard cases.** Pretrained → end of run:

   | record | what it is | pretrained | b9 1e-4 | b9 3e-4 | b129 3e-4 | b129 1e-3 |
   |---|---|---|---|---|---|---|
   | `7t5m` | 20-mer | 7.56 | 8.77 | **3.56** | **2.93** | **3.83** |
   | `8d5q` | 10-mer | 2.20 | 1.30 | **0.66** | **0.73** | **0.67** |
   | `7ndq` | 9-mer | 2.58 | 2.06 | 1.58 | 1.80 | 1.78 |
   | `6zkx` | 12-mer | 2.84 | 1.89 | 2.25 | 3.26 | 2.50 |
   | `8d5n` | 16-mer | 3.66 | 3.62 | 3.70 | 3.20 | 4.39 |
   | `7cir` | phosphopeptide | 1.98 | 2.57 | 2.63 | 2.52 | 2.23 |

   `8d5q` is fixed in three of four arms (1.30 Å in b9 1e-4). `7t5m` halves in three. `7cir` gets worse in
   every arm. `7cir` carries a phosphoserine (`RRF[SEP]RSPIRR`); its sibling `7ciq`
   is the unphosphorylated form.

## Caveats

* **Subset30 also picks the checkpoints**, and a single validation is noisy: the
  median over all samples moves ±0.1 Å between neighbouring validations. A claim
  needs the 69 records that played no part in selection: run the 99-record eval on
  the candidates.
* **The ~2.6 Å `rmsd` / ~1.65 Å `best_rmsd` validations of b129 3e-4 are the
  `7ndt` chain swap** (`UPSTREAM_PATCHES.md` §8), not a better model. The swap
  detector fires on the same 3 samples at every validation. At the step-110 line,
  there were two validations at the same step: the one at epoch end and the one at
  the time-limit stop.

## Next

The 99-record eval of the candidates (`scripts/ls6_eval_val99.slurm` with
`EVAL_CKPTS`), queued behind 3499815:
* b9 3e-4 final, and its best checkpoint (step 1008, picked by `val/lddt`). If the
  final validation beats it, the file is replaced and the eval skips it;
* b9 1e-4 final;
* b129 3e-4 final.

The comparison is paired against `runs/eval_val99/pretrained`, on all 99 and on
the 69 held-out records.
