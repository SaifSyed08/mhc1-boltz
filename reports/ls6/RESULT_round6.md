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

## 4. What would settle it

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
