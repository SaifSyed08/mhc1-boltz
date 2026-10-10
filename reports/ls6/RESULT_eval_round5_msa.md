# Held-out eval of the Round 5 checkpoints, and MSA depth

Two validation-only jobs on all 99 validation-era records:
* **3506252:** four Round 5 checkpoints;
* **3506180:** pretrained Boltz-1 with the MSA cut to 1, 16 or 128 rows.

Every number is MHC-Fine's metric: peptide Cα RMSD after superposing on the α1/α2
groove, against the real complex. Each run is paired per structure with
`runs/eval_val99/pretrained`, which used the same `validation_args`.

The 69 held-out records are the ones outside subset30. They played no part in
training or in picking checkpoints.

The p-values are exact two-sided sign tests on "improved vs not" (a tie counts as
not improved). "Paired" is the median of the per-record differences. Digest:
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

1. **Batch 9 at 3e-4 is the first fine-tune with a gain that survives on
   structures it never saw — suggestive, not established.**
   * The median peptide RMSD goes 0.80 → 0.69 Å, and `lddt_pp` rises by 0.011.
   * The paired median change is −0.07 Å on all 99 and −0.066 Å on the 69 held
     out, so the effect does not shrink when the checkpoint-selection records are
     removed. The final checkpoint was not itself selected on subset30 (it is
     simply where the session's time limit fell), though the *arm* was chosen by
     looking at subset30.
   * Significance is borderline. The all-99 sign test is p = 0.015, but those 99
     include the 30 used for selection. On the 69 held-out records it is 43/69,
     **p = 0.053**, and five checkpoints were tested with no multiplicity
     correction. `s1008` is the one that clears 0.05 held out (44/69, p = 0.030).
     Read this as "two checkpoints of the same arm both lean the same way",
     not as a demonstrated effect.
   * MHC-Fine reports 0.66 Å on its own test set, so the two are now within a few
     hundredths of an ångström. The test sets differ, and their number is one
     pLDDT-picked model per target.
2. **Neither more updates nor the higher rate is enough alone; only the
   combination moves.** Batch 9 at 1e-4 (−0.021, 55/99, p = 0.31) and batch 129 at
   3e-4 (−0.032, 55/99, p = 0.31) are both indistinguishable from pretrained, as
   is batch 129 at 1e-4 (step 153: 54/99, p = 0.42). Only batch 9 at 3e-4 moves.
   Two factors are confounded here: batch 9 also took ~1,480 optimizer steps
   against ~110, so "small batch" and "13× more updates" cannot be separated by
   these four arms.
3. **The final checkpoint and the "best" one are indistinguishable; the run had
   not finished either way.** Final leads on the median (0.691 vs 0.704) and the
   all-99 paired change, `s1008` leads on sample 0 (0.672 vs 0.707) and on the
   held-out count and p-value. Those gaps are inside this eval's ~0.02 Å
   single-draw noise, so neither is "better". What both say is that training loss
   was still falling and the LR never decayed, so the arm was still improving when
   the 47 h session ended. Next: a second session of the same run.
4. **The mean falls much more than the median** (1.21 → 0.93 Å), which means the
   tail moved. How much of that is one record is not yet known: `7wj3` alone would
   contribute ~0.14 Å of the 0.28 Å mean drop if it fell from 15.3 Å to ~1 Å, and
   the `< 8` column below shows at least one of its samples did move a long way in
   the four Round 5 checkpoints. **Do not claim "the hard cases move most" until
   the per-record table (`scripts/ls6_evals.sh --records`) is in.**

## MSA depth: 16 sequences are almost enough; the query alone is not

| depth (rows) | median | mean | `lddt_pp` | paired vs 2048 |
|---|---|---|---|---|
| 2048 (pretrained) | 0.804 | 1.213 | 0.8842 | — |
| 128 | 0.778 | 1.165 | 0.8859 | +0.008, 48/99 improve (p 0.84) |
| 16 | 0.787 | 1.180 | 0.8830 | +0.033, 41/99 (p 0.11) |
| **1 (query only)** | **1.057** | **1.879** | **0.8556** | **+0.107, 32/99 (p 0.0006)** |

* **Ernest's expectation holds.** With the MSA removed at inference, the median
  peptide RMSD worsens by 0.25 Å, the mean by 0.67 Å, and `lddt_pp` drops by 0.029.
  Only 32 of 99 structures improve (p = 0.0006); the other 67 are not improved.
* **The loss looks concentrated in the last step down.** At 128 rows there is no
  sign of a cost (48/99, p = 0.84); at 16 rows the shift is small and not
  significant (41/99, p = 0.11), which with n = 99 means "no large effect", not
  "no effect". Nothing was run between 1 and 16, so where the cliff starts is
  unmeasured. Either way the target for gradual removal is depth 1.
* There is no seed, so each row is one diffusion draw. Differences of about
  0.02 Å or 0.002 lDDT are noise.
* The MSA training arms (3506808 taper, 3506809 abrupt) validate at depth 1. Their
  baseline is the depth-1 row here: median 1.057, `lddt_pp` 0.8556.

## The length buckets in that run, and what the `< 8` column really was

The digest bucketed by the rows' `pep_len`, which counts **tokens**: a modified
residue tokenizes per atom, so `7wt3` (MYR-GANF, 5 residues) counted as 19 tokens.
17 validation records reach 12 tokens but only 3 have 12 residues, so **14** short
peptides were pushed into `12+` — 7 of them phosphoserine, the rest carrying QM8,
QMB, DPN, E7P, OSE or (for `7wt3`) MYR. Fixed: the digest now takes residue counts
from `data/peptide_quality.json`. By residue the 99 split 3 / 93 / 3, so `< 8` and
`12+` are three-record anecdotes either way.

The `< 8` value of 8.47 Å for pretrained is **not** an artefact, though. Under
token counting that bucket held exactly two records, `7wj3` (4 tokens) and `7tud`
(6); the next smallest is 8. The median of their six pooled samples is
(1.80 + 15.15) / 2 = 8.475 — the printed number.

**What the fine-tuned columns do and do not prove.** The digest pools *samples*,
so the column is the mean of the 3rd and 4th of six values. A column of 1.15 forces
x(4) ≤ 2.30, i.e. at least four of the six are ≤ 2.30 Å; `7tud` supplies only
three, so **at least one of `7wj3`'s three samples is ≤ 2.30 Å**. It does *not*
follow that all three fell, nor that its mean is ~1 Å: {0.9, 15.0, 15.4} against a
well-placed `7tud` prints the same column. Even so, every pretrained `7wj3` sample
was 15.15–15.42 Å, so one sample at ≤ 2.3 Å is a real change. The columns and the
bound each implies: b9 3e-4 final 1.15 (≤ 2.30), b129 3e-4 1.17 (≤ 2.34), b9 3e-4
s1008 1.35 (≤ 2.70), b9 1e-4 1.54 (≤ 3.08).

**Not "every arm".** Round 4's step 153 reads 8.07 in that column, and its `7wj3`
mean is 15.02 against 15.26 pretrained (`eval_val99/per_record_mean.tsv`). The three
MSA-depth runs read 8.48–9.00. So this is the four Round 5 checkpoints only. The
column values come from the 2026-10-12 terminal run; the raw rows are not yet in
the repo.

If the per-record table bears it out, it is the most interesting result here.
`7wj3` is the N-myristoylated lipopeptide whose MYR group our pipeline masks
(`reports/TRIAGE_peptides.md`). Myristoyl is on six peptides in the dataset, in two
different forms:

| form | train | validation |
|---|---|---|
| MYR masked as a separate covalent ligand (what `7wj3` is) | `4zfz`, `7byd` | `7wj3` |
| MYR kept as the peptide's first residue, so the model sees it | `6iwg`, `6iwh` | `7wt3` |

So training shows the model four myristoylated peptides, two of them in `7wj3`'s
exact configuration. That is a plausible route to a 15 Å → ~2 Å move, and it argues
against dropping `7wj3` from the set. It is a hypothesis, not a measurement: four
training records is thin, and nothing here tests the mechanism.

## Queue (2026-10-12)

* 3506808 (MSA taper) and 3506809 (MSA abrupt): gpu-a100, pending.
* 3506810: dev, Round 6 smoke test plus the pocket/structure eval, pending.

## One review claim checked and rejected

A review of this file reported that `src/peptide_rmsd.py` scores the wrong chain
for `7tud`, because that entry has a 2-residue chain `D1` (GL) shorter than its
6-residue peptide `C1`, and the scorer takes the shortest protein chain as the
peptide (`peptide_rmsd.py:167`).

It does not. `D1` is masked in `data/processed/structures/7tud.npz`
(`mask[6] = False`), and the tokenizer iterates `struct.chains[struct.mask]`
(`boltz-src/src/boltz/data/tokenize/boltz.py:60`), so a masked chain never
produces a token and cannot be in `batch`. The scorer sees only the three kept
chains — `A1` (274), `B1` (100), `C1` (6) — and picks `C1`. The row's `pep_len`
of 6 confirms it. Every `7tud` number here is against the real peptide.

The shortest-chain rule is still worth a guard if an entry ever keeps a protein
fragment shorter than its peptide; no record in the 99 does.
