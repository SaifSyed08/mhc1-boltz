# Eval on all 99 validation-era records: Boltz-1 already starts near MHC-Fine's endpoint

Job 3499816 (`scripts/ls6_eval_val99.slurm`, `gpu-a100-dev`) ran on 2026-10-09 and
completed in 39 min 30 s. It ran validation only on two checkpoints:
* pretrained Boltz-1 (`boltz1_conf.ckpt`)
* the Round 4 fine-tune at step 153 (`r3_end_step153.ckpt`)

Both used the code at 822e470, where peptide scoring is against the real complex
(`UPSTREAM_PATCHES.md` §8). Every one of the 99 records was scored, with no failure
`reason` rows and no `Skipping` lines in the log (audit MINOR-2). 69 of the 99 were
never trained on and never used to pick a checkpoint.

The summary was computed on LS6 from `runs/eval_val99/{pretrained,step153}/
val_peptide_rank*.jsonl`. Copying the raw rows into the repo is still to do.

## MHC-peptide Cα RMSD (superpose on the α1/α2 groove, RMSD over peptide Cα)

| | pretrained | step 153 |
|---|---|---|
| median, all 3 samples per structure (the number to compare) | **0.80 Å** | **0.75 Å** |
| median, sample 0 only | 0.82 Å | 0.75 Å |
| median, oracle best of 3 (not comparable; picked with the answer) | 0.66 Å | 0.63 Å |
| mean, all samples (tail-driven) | 1.21 Å | 1.19 Å |
| 8–11-mers with all-standard residues (n = 78), median | 0.79 Å | 0.73 Å |
| ... share of samples under 1 Å / under 2 Å | 0.62 / 0.94 | 0.65 / 0.94 |
| groove Cα RMSD, median | 0.49 Å | 0.47 Å |
| all-chain Cα RMSD, median | 0.91 Å | 0.87 Å |
| Boltz `lddt_pp`, mean over samples | 0.8842 | 0.8882 |

For reference, MHC-Fine on its own test set (2020-06 → 2023-08): AlphaFold-Multimer
v2.2 had 1.59 Å, MHC-Fine 0.66 Å, and Pandora 1.31 Å. Those are medians of one
pLDDT-picked model per target.

**Paired, per structure** (mean of 3 samples, step 153 minus pretrained, n = 99):
the median difference is −0.012 Å and the mean −0.027 Å. 54 of 99 improve. A sign
test puts 54/99 at p ≈ 0.4, so the paired change is indistinguishable from zero.
The 0.80 → 0.75 gap between the two medians is larger than the paired change and
should not be read as a gain.

## What it means

1. **Boltz-1 starts where MHC-Fine ended.** Out of the box, its median peptide RMSD
   on our validation era is 0.80 Å (0.79 Å on 8–11-mers). That is far below
   AlphaFold-Multimer's 1.59 Å and close to MHC-Fine's fine-tuned 0.66 Å. The test
   sets differ (ours is 2021-10 → 2023-01; theirs is 2020-06 → 2023-08 and overlaps
   it), and their number is a confidence-picked model. So "close to" is the claim,
   not "equal". Either way, the ~0.9 Å MHC-Fine gained is mostly not available here:
   the typical peptide is already placed within about 1 Å.
2. **153 steps of fine-tuning did not measurably move the peptide**, matching the
   flat `lddt_pp` of Round 4.
3. **The room left is in the tail.** The mean (1.21 Å) is well above the median.
   The worst pretrained structures are the unusual ligands:

   | record | peptide | mean RMSD |
   |---|---|---|
   | `7wj3` | a 4-mer (GAAL) | 15.3 Å |
   | `7t5m` | a 20-mer | 7.6 Å |
   | `8d5n` | a 16-mer | 3.7 Å |
   | `6zkx` | a 12-mer | 2.8 Å |
   | `7ndq` | a 9-mer | 2.6 Å |
   | `7ow5` | a 10-mer | 2.4 Å |

   Anything that moves the headline will have to come from long, short or
   non-canonical peptides and the few hard 9–10-mers. The batch-9 arms
   (3499814/5) will show whether more frequent updates help there. Their
   per-structure rows track exactly these records.

## The 7ndt explanation is confirmed

On 7ndt, `boltz_truth_dev` is 48.9 Å in all three samples for both checkpoints,
and Boltz's own RMSD is ~49.2 Å. Our peptide RMSD against the real complex is
0.39–0.68 Å. So the model predicts 7ndt well; Boltz scores it against the masked
twin copy. That is the "high mode" of `RESULT_round4.md` §4, exactly as predicted.

`7bh8` shows the ~10 Å swap level, 9.85 Å. Step 153 has it in one sample and has
two samples correctly scored (dev 0.0, Boltz RMSD ~1.2 Å). That is the swap
flipping between samples, as §8 described. Its peptide RMSD is 0.37–0.59 Å
throughout. No other record exceeds 5 Å.

## Still open

* The smoke test (3499813) is still pending in the dev queue. The four training
  arms are pending on `gpu-a100`.
* The raw rows need copying into `reports/ls6/eval_val99/`.
