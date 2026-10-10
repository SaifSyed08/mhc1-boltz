# Held-out eval of the Round 5 checkpoints, and MSA depth

Two validation-only jobs on all 99 validation-era records:
* **3506252:** four Round 5 checkpoints;
* **3506180:** pretrained Boltz-1 with the MSA cut to 1, 16 or 128 rows.

Every number is MHC-Fine's metric: peptide Cα RMSD after superposing on the α1/α2
groove, against the real complex. Each run is paired per structure with
`runs/eval_val99/pretrained`, which used the same `validation_args`.

The 69 held-out records are the ones outside subset30. They played no part in
training or in picking checkpoints.

The p-values are exact two-sided sign tests on "improved vs not". Digest:
`scripts/ls6_evals.sh`, 2026-10-12.

## Fine-tuning: batch 9 at 3e-4 is a real, small gain

| eval | median, all samples | sample 0 | mean | `lddt_pp` | paired, all 99 | paired, held-out 69 |
|---|---|---|---|---|---|---|
| pretrained | 0.804 | 0.819 | 1.213 | 0.8842 | — | — |
| Round 4, step 153 (b129, 1e-4) | 0.747 | 0.752 | 1.186 | 0.8882 | −0.012, 54/99 | −0.008, 38/69 |
| b129 3e-4, final | 0.748 | 0.743 | 0.973 | 0.8923 | −0.032, 55/99 (p 0.31) | −0.016, 36/69 |
| b9 1e-4, final | 0.785 | 0.770 | 1.062 | 0.8900 | −0.021, 55/99 (p 0.31) | −0.019, 37/69 |
| b9 3e-4, step 1008 (best by `val/lddt`) | 0.704 | 0.672 | 0.952 | 0.8927 | −0.043, 61/99 (p 0.03) | −0.051, 44/69 (p 0.03) |
| **b9 3e-4, final (~1,480 steps)** | **0.691** | 0.707 | **0.932** | **0.8952** | **−0.070, 62/99 (p 0.015)** | **−0.066, 43/69 (p 0.05)** |

1. **Batch 9 at 3e-4 is the first fine-tune that beats pretrained on structures
   it never saw.**
   * The median peptide RMSD goes 0.80 → 0.69 Å.
   * The paired change is −0.07 Å, on all 99 and on the 69 held out alike, so
     selection on subset30 did not inflate it.
   * `lddt_pp` rises by 0.011.
   * MHC-Fine reports 0.66 Å on its own test set, so the two are now within a few
     hundredths of an ångström. The test sets differ, and their number is one
     pLDDT-picked model per target.
2. **Learning rate matters, batch size by itself does not.** Batch 9 at 1e-4 is
   no better than batch 129 at 3e-4. The arm with both more updates and the
   higher rate is the one that moves.
3. **The final checkpoint beats the "best" one.** Its training loss was still
   falling and the LR never decayed, so the run had not finished.
   Picking the best checkpoint by subset30 `val/lddt` chose an earlier, worse
   checkpoint. Next: a second session of the same run.
4. **The mean falls more than the median** (1.21 → 0.93 Å), so the hard cases
   move most. That matches Round 5's in-training rows (`8d5q`, `7t5m`). The
   per-record table (`ls6_evals.sh --records`) is needed to name them.

## MSA depth: 16 sequences are almost enough; the query alone is not

| depth (rows) | median | mean | `lddt_pp` | paired vs 2048 |
|---|---|---|---|---|
| 2048 (pretrained) | 0.804 | 1.213 | 0.8842 | — |
| 128 | 0.778 | 1.165 | 0.8859 | +0.008, 48/99 improve (p 0.84) |
| 16 | 0.787 | 1.180 | 0.8830 | +0.033, 41/99 (p 0.11) |
| **1 (query only)** | **1.057** | **1.879** | **0.8556** | **+0.107, 32/99 (p 0.0006)** |

* **Ernest's expectation holds.** With the MSA removed at inference, the median
  peptide RMSD worsens by 0.25 Å, the mean by 0.67 Å, and `lddt_pp` drops by 0.029.
  67 of 99 structures get worse.
* **The loss is all in the last step.** At 16 rows, nothing is significantly
  lost. The MHC fold needs only a shallow MSA. So the target for gradual removal
  is depth 1; depth 16 is already almost free.
* There is no seed, so each row is one diffusion draw. Differences of about
  0.02 Å or 0.002 lDDT are noise.
* The MSA training arms (3506808 taper, 3506809 abrupt) validate at depth 1. Their
  baseline is the depth-1 row here: median 1.057, `lddt_pp` 0.8556.

## Correction to the digest's length buckets

The run printed "<8 / 8–11 / 12+" medians that are wrong. The rows' `pep_len`
counts tokens, and modified residues tokenize per atom. Fixed: the digest now uses
residue counts from `data/peptide_quality.json`. By residue count, the 99 split
3 / 93 / 3, so the <8 and 12+ columns are three-record anecdotes.

## Queue (2026-10-12)

* 3506808 (MSA taper) and 3506809 (MSA abrupt): gpu-a100, pending.
* 3506810: dev, Round 6 smoke test plus the pocket/structure eval, pending.
