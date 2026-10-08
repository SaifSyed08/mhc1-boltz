# MHC-Fine did this and it worked. What is different here?

MHC-Fine (Glukhov et al., *Biophys J* 123:2902–2909, 2024; bioRxiv
10.1101/2023.11.29.569310) fine-tuned AlphaFold-Multimer v2.2 on class I
pMHC crystal structures. On its test set the median **MHC-peptide Cα RMSD**
(superpose on the MHC chain, RMSD over the peptide's Cα) went from **1.59 Å**
(base model, best of its five models by pLDDT) to **0.66 Å** (Pandora: 1.31 Å).
Our fine-tune of Boltz-1 is flat through 153 steps (`RESULT_round4.md`). Side by
side, from the published paper:

| | MHC-Fine | here, Rounds 1–4 |
|---|---|---|
| base model | AlphaFold-Multimer v2.2, PDB cutoff 2018-04-30 | Boltz-1, PDB cutoff 2021-09-30 |
| training split vs that cutoff | 1992-10-15 → 2016-12-07: entirely before it | ≤ 2021-09-30: entirely before it |
| validation / test | 2016-12 → 2020-06 / 2020-06 → 2023-08 | 2021-10 → 2023-01 / after 2023-01 |
| data | 919 X-ray structures < 3.5 Å, peptides 8–11 | 1,084 from TCR3D (914 train) |
| what the model is given | MHC **α1 + α2 only** + peptide; no α3, no β2m | full heavy chain + β2m + peptide |
| templates | yes, with MHC–peptide interaction; "crucial", "significantly accelerated the training convergence" | none — Boltz-1 has no template input |
| architecture | **+2 Evoformer blocks**, "one of the most important parameters" | unchanged |
| loss | peptide residues 1.4×, though "similar levels of accuracy could be achieved without" | unweighted |
| learning rate | 3e-4, cosine annealing | 1e-4, constant after a 10-step warmup |
| batch | **8** (8 GPUs, no accumulation), "proved beneficial" | 128–129 (gradient accumulation) |
| optimizer updates | not stated; ~70 per pass over ~550 training structures | **153 in total** |
| headline metric | MHC-peptide Cα RMSD | `lddt_pp`: every inter-chain contact pooled |
| base model on the headline metric | 1.59 Å | **never measured** |

## What it changes

1. **"The model has already seen the training structures" does not explain it.**
   MHC-Fine's training set also predates its base model's cutoff, and it still
   gained ~0.9 Å. `RESULT_round4.md` §3 offered that as the likely reason; it is
   at most part of it.

2. **We have not been measuring what MHC-Fine improved.** Its gain is peptide
   placement. `lddt_pp` is dominated by the heavy chain–β2m interface, which every
   model already gets right, so a better peptide could sit inside a flat `lddt_pp`.
   And how much room Boltz-1 even has is unknown: AlphaFold-Multimer v2.2 started at
   1.59 Å, while Boltz-1 is an AlphaFold-3-class model trained on 3.5 more years of
   PDB. If it already starts near 0.7 Å, there was little left to gain and a flat
   curve is the right answer. Measuring that comes first.

   **Compare like with like.** MHC-Fine's 1.59 and 0.66 Å come from one model per
   target, picked by pLDDT without seeing the answer. Boltz-1 runs here without a
   confidence head, so nothing can be ranked. The comparable number is the median
   over all diffusion samples, or over sample 0 alone. The best of 3 is picked by
   looking at the answer, so it is biased low. It is reported only as "oracle
   best-of-N", never against 1.59 Å. MHC-Fine's peptides are 8–11-mers; ours run
   4–20. Non-standard residues, in 14 of the 99 validation peptides, are tokenized
   per atom, so they have no Cα frame and drop out of the peptide set. The
   comparison therefore also needs an 8–11-mer subset.

3. **Far fewer optimizer updates.** At batch 129, one 47 h session is ~110 updates;
   MHC-Fine trained at batch 8, which it says "proved beneficial". Per structure
   seen, we take ~16× fewer steps. Our batch size was copied from Boltz's
   from-scratch recipe, which is the wrong regime for fine-tuning on ~900 structures.

4. **Templates, extra blocks, α1α2-only inputs** are larger changes. Boltz-1 cannot
   take templates at all (Boltz-2 can), and new pairformer blocks or a groove-only
   input change the model or the task. Those are for Saif and Ernest to decide on
   once (2) and (3) are answered.

## Round 5: what is set up (2026-10-09)

* **Peptide RMSD everywhere.** `src/peptide_rmsd.py` writes one row per validated
  structure: MHC-peptide Cα RMSD per diffusion sample (superposed on the α1/α2
  groove), peptide shape and groove RMSD, a mirror-image check, and Boltz's own
  per-sample RMSD and lDDT. Every run below records it at every validation.
  Training loss is now recorded too (`mem.jsonl`). Since 822e470 it scores against
  the real complex, never Boltz's chain-swapped truth (`UPSTREAM_PATCHES.md` §8).
  `boltz_truth_dev` marks the samples where Boltz's own numbers measured a swapped
  complex.
* **Eval on all 99 validation-era records** — pretrained vs step 153
  (`scripts/ls6_eval_val99.slurm`). 69 of them were never trained on or used for
  selection. This gives Boltz-1's starting point on MHC-Fine's metric.
* **Batch 9** (`accumulate_grad_batches: 3` × 3 GPUs) at `max_lr` 1e-4 and 3e-4,
  fresh from pretrained: ~1,500 updates per session instead of ~110.
* The batch-129 arms at 3e-4 and 1e-3 (3499581, 3499582) stay queued, re-chained behind
  the same smoke test, as the large-batch half of the comparison.

The jobs were first chained behind one dev-queue smoke test. It fails unless the
training loss and six finite per-structure peptide rows come out of the real
3-GPU path. At ~18:05 the same day Saif removed the chain:
* On LS6 a job held on a dependency earns no queue-age priority. Age is the
  largest priority factor there.
* `gpu-a100` was ~2 days deep: 98 jobs pending, and `sbatch --test-only` gave
  Oct 11 18:46 for any new job.
* So the gate was costing queue position while protecting nothing. The smoke test
  reports days before any arm can start. If it fails, the arms are held and the
  code is fixed before they start; they read the code at start.
* The smoke test was also given `nice` 200, so the eval takes the one dev slot
  first.

| job | what | queue (2026-10-09 18:10) |
|---|---|---|
| 3499816 | eval, all 99: pretrained vs step 153, `runs/eval_val99/` | dev, est. start 19:50 |
| 3499813 | smoke test, dev queue | after the eval |
| 3499814 | batch 9, `max_lr` 1e-4, `runs/ls6_b9_lr1e-4` | gpu-a100, pending |
| 3499815 | batch 9, `max_lr` 3e-4, `runs/ls6_b9_lr3e-4` | gpu-a100, pending |
| 3499581 | batch 129, `max_lr` 3e-4 (submitted earlier the same day) | gpu-a100, pending |
| 3499582 | batch 129, `max_lr` 1e-3 (submitted earlier the same day) | gpu-a100, pending |

All of them run 822e470, pulled on LS6 at ~18:20, before any of them had started.

Step 153's weights are kept as `runs/ls6_unfrozen/r3_end_step153.ckpt` (a copy of
that run's `last.ckpt`, written at 03:36, after step 153 and before the stop).
