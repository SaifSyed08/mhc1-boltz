# Round 6: training the MSA away, and the control that reframes it

Jobs 3506808 (taper) and 3506809 (abrupt) ran 2026-10-13 22:01 → 2026-10-15 21:13
(1 d 23 h each, three A100s). Held-out eval job 3587742 ran 2026-11-16, 55 min.

Both arms start **fresh from the pretrained checkpoint** with the Round 5 winning
recipe (batch 9, max_lr 3e-4), and differ only in how many MSA rows training is
allowed to use:

* **taper:** 2048 → 512 → 128 → 32 → 8 → 1, one step down per epoch, then query-only;
* **abrupt:** 1 row (the query) from the first step.

Both validate at depth 1 throughout. Every number below is peptide Cα RMSD after
superposing on the α1/α2 groove, over all 99 validation-era records, per-record
mean over 3 diffusion samples.

## The numbers

| run | MSA in training | MSA at eval | median | sample 0 | mean | `lddt_pp` |
|---|---|---|---|---|---|---|
| pretrained Boltz-1 | — | full (2048) | 0.847 | 0.819 | 1.213 | 0.8842 |
| pretrained Boltz-1 | — | **query only** | **1.119** | 1.057 | 1.879 | 0.8556 |
| b9 3e-4 (Round 5) | full | full | 0.689 | 0.707 | 0.932 | 0.8952 |
| **b9 3e-4 (Round 5)** | **full** | **query only** | **0.749** | 0.760 | 0.961 | 0.8937 |
| taper | 2048→1 | query only | 0.725 | 0.725 | 0.956 | 0.8937 |
| **abrupt** | **1** | **query only** | **0.713** | 0.688 | 0.970 | 0.8932 |

Paired per-record, two-sided sign tests. "Held-out 69" is the 99 minus subset30,
which played no part in training or in picking checkpoints.

| arm (all query-only at eval) | vs pretrained query-only, all 99 | held-out 69 | vs pretrained **full MSA**, all 99 | held-out 69 |
|---|---|---|---|---|
| taper | −0.207, 74/99, p < 0.0001 | −0.129, 49/69, p = 0.0006 | −0.034, 60/99, p = 0.044 | −0.034, 41/69, p = 0.148 |
| abrupt | −0.238, 74/99, p < 0.0001 | −0.134, 50/69, p = 0.0002 | −0.069, 65/99, p = 0.0024 | −0.049, 43/69, p = 0.053 |
| b9 3e-4 (no MSA schedule) | −0.126, 80/99, p < 0.0001 | −0.105, 55/69, p < 0.0001 | −0.036, 60/99, p = 0.044 | −0.029, 41/69, p = 0.148 |

## 1. What is solid: a fine-tuned model does not need the MSA

Removing the MSA from **pretrained** Boltz-1 costs 0.847 → 1.119 Å median (+0.27)
and 0.0286 lDDT. That is Ernest's prediction, confirmed on all 99.

After fine-tuning, that cost is gone. Every fine-tuned arm, scored on the query
sequence alone, lands at 0.71–0.75 Å — **better than pretrained Boltz-1 with its
full 2048-row MSA** (0.847). Against the query-only baseline the effect is large
and survives the held-out split (49–55 of 69, p ≤ 0.0006). `lddt_pp` recovers from
0.8556 to 0.893–0.894, against 0.8842 for pretrained with its MSA.

So for pMHC the MSA is not carrying information the model cannot get elsewhere.
That is consistent with what the MSA is for here: the peptide's own MSA is
essentially empty, and what the alignment carries is the MHC fold — which 914 MHC
structures apparently teach directly.

## 2. What the control says: the schedule is not what did it

`b9_3e-4_last` is the Round 5 checkpoint, trained with **full** MSAs, scored here
query-only for the first time. It reaches 0.749 Å — already past pretrained-with-MSA
and 0.37 Å better than pretrained query-only.

The two MSA arms beat it by 0.024 (taper) and 0.036 Å (abrupt). This eval's
single-draw noise is about 0.02 Å, and neither arm was repeated with a second seed.

**So the headline is not "gradual removal works".** It is that ordinary fine-tuning
on this dataset already produces a model that barely needs its MSA, and training
the MSA away adds little on top. The gradual taper, specifically, is not necessary:
abrupt (query-only from step 0) matches or beats it, which also means no schedule
has to be designed.

Upstream's featurizer already draws a training depth uniformly from 1 to `max_seqs`
for every sample (`featurizer.py` ~1174), so even the "full MSA" arm saw shallow
alignments in ~0.8% of draws and depth-1 draws occasionally. That is the most
likely mechanism, and it is a better explanation than anything the schedule adds.

## 3. What is still borderline

Against pretrained **with its full MSA** on the held-out 69, only `abrupt` comes
close to significance (43/69, p = 0.053), and that is the same borderline figure
Round 5 produced. Three arms were compared here with no multiplicity correction.

Claim what the data supports: **an MSA-free fine-tuned model matches pretrained
Boltz-1 that has its MSA.** Do not claim it beats it.

## 4. Where the error actually is: peptide length

Median peptide RMSD by peptide length in **residues** (`data/peptide_quality.json`
`pep_len`, never the eval rows' token count — modified residues tokenize per atom):

| length | n | pretrained, full MSA | pretrained, query only | b9 3e-4, full MSA | abrupt, query only |
|---|---|---|---|---|---|
| ≤ 7 | 3 | 1.519 | 1.268 | 1.094 | 1.206 |
| 8–9 | 78 | 0.727 | 0.988 | 0.656 | 0.674 |
| 10–11 | 15 | 1.277 | 1.989 | 0.817 | 0.924 |
| ≥ 12 | 3 | 3.655 | 3.215 | 3.366 | 3.539 |

* The canonical 8–9mers are 78 of the 99 and are already near the crystals' own
  precision; fine-tuning moves them 0.73 → 0.66 Å.
* **Fine-tuning helps most at 10–11 residues: 1.28 → 0.82 Å.**
* **≥ 12 residues does not improve at all** (3.66 → 3.37/3.54). n = 3, and the
  training set holds only 4 peptides of 15+ residues, so this is as likely a data
  limit as a modelling one.
* **MSA removal costs the non-canonical lengths most** in the pretrained model:
  8–9mers +0.26 Å, 10–11mers +0.71 Å. Fine-tuning recovers both.

Per record, the short and long tail:

| record | residues | pretrained | b9 3e-4 full | abrupt query-only |
|---|---|---|---|---|
| 7wj3 | 4 | 15.26 | 1.31 | 1.21 |
| 7wt3 | 5 | 1.32 | 0.75 | 0.62 |
| 7tud | 6 | 1.52 | 1.09 | 1.56 |
| 6zkx | 12 | 2.84 | 2.22 | 2.57 |
| 8d5n | 16 | 3.66 | 3.99 | 3.54 |
| 7t5m | 20 | 7.56 | 3.37 | 5.15 |

**7wj3 settles the open question from `RESULT_eval_round5_msa.md`.** That file could
only bound one of its three samples; the per-record means here show 15.26 → 1.21 Å.
It remains a hypothesis that this is the four myristoylated training peptides (two
masked exactly as 7wj3 is) being learned, not the lipid being placed for the right
reason. Nothing here tests the mechanism.

## 5. Pocket conditioning does not work, and the oracle says why that matters

`scripts/ls6_eval_pocket.slurm`, all 99, paired against `eval_val99/pretrained`:

| run | median | paired | improved | p |
|---|---|---|---|---|
| plain (no pocket, same code path) | 0.847 | +0.014 | 42/99 | 0.159 |
| NetMHCpan pseudo-sequence (34 positions) | 0.793 | +0.012 | 46/99 | 0.547 |
| oracle (HC residues within 6 Å of the true peptide) | 0.808 | −0.016 | 57/99 | 0.159 |

* The `plain` run reproduces the baseline median exactly, so the conditioning
  harness itself changes nothing. That is the control that makes the rest readable.
* Neither pocket helps. **Even the oracle pocket — perfect information about which
  heavy-chain residues the real peptide touches — does nothing** (57/99, p = 0.16).
* Note the unpaired medians (0.793, 0.808) look better than 0.847 while the paired
  medians say no. The distribution shifts; per structure it is a wash. Trust the
  paired column.
* Caveat already in `PLAN_round6.md` §6: training marks ~4 pocket tokens per
  conditioned sample and these mark 34 or 43, so the conditioning is out of
  distribution and a well-matched pocket is not ruled out. But the oracle result
  argues pocket constraints are not the vehicle for MHC-Fine's template idea.
  The untried route is threading peptides onto same-length structures for real or
  partial templates, or moving to Boltz-2, which has a template module.

## 6. 7cir: fine-tuning makes the phosphopeptide worse, every time

| checkpoint | 7cir |
|---|---|
| pretrained, full MSA | **1.979** |
| pretrained, query only | 1.830 |
| taper, query only | 2.215 |
| b9 3e-4, query only | 2.296 |
| b9 3e-4 s1008 | 2.328 |
| b129 3e-4 | 2.347 |
| b9 1e-4 | 2.397 |
| step 153 | 2.457 |
| abrupt, query only | 2.540 |
| b9 3e-4, full MSA | 2.597 |

Nine fine-tuned checkpoints, all worse than pretrained, spread 2.21–2.60 Å against
1.98. This is consistent and not a single-draw artefact. Phospho-Tune (Kozakov
group, 2024) fine-tunes for phosphopeptides specifically — the open question for
Ernest.

## 7. What would settle it

1. A second seed for `abrupt` and for `b9_3e-4`, scored query-only. 0.036 Å apart
   with ~0.02 Å noise is not a difference yet.
2. The held-out comparison that matters for a paper is arm vs pretrained-with-MSA,
   and it needs more than 69 records or a repeated draw to clear 0.05 honestly.
3. Wall-clock: the practical claim is that dropping the MSA removes the MSA search
   from the pipeline. That has not been timed here.

## Reading note

All six runs use identical `validation_args`, so `diffusion_samples` is constant
and the medians are comparable (see the best-of-N trap in `BASELINE.md`). Rows are
per-record means over 3 samples, never best-of-3.
