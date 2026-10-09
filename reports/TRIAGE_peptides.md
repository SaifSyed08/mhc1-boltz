# Peptide triage, validation era (99 records): improve / difficult / drop

Ernest asked for the peptides to be sorted into cases we can improve, cases that
are particularly difficult, and cases to drop because the PDB entry is poor.

First pass, built from two tables:
* the pretrained Boltz-1 eval, `reports/ls6/eval_val99/per_record_mean.tsv`
  (peptide CA RMSD, mean of 3 samples);
* the wwPDB density fit of the deposited peptide, `data/peptide_quality.json`
  from `src/peptide_quality.py`.

Rules (first pass, to be checked in PyMOL):
* **Drop:** the deposited peptide cannot be trusted as ground truth, or the input
  leaves out what positions it. That means a covalently attached ligand on the
  peptide, or a peptide whose mean RSCC is below 0.8.
* **Difficult:** the length is outside 8–11, or the peptide has a modified
  residue (X in the sequence) and pretrained Boltz-1 is above 1.2 Å. Modified
  residues on their own are not difficult. 14 validation peptides carry one, and
  11 of them are at or below 1.2 Å (see "Modified residues" below).
* **Can improve:** a canonical 8–11-mer with a good density fit, where pretrained
  Boltz-1 is above 1.2 Å.

## Drop (5)

| record | peptide | pretrained Å | why |
|---|---|---|---|
| `7wj3` | GAAL (4) | 15.26 | N-myristoylated lipopeptide. MYR is covalently linked to Gly1 and masked by our pipeline, so the model never sees what positions the peptide |
| `7ejn` | MYVKWPWYV | 1.54 | peptide mean RSCC 0.68, min 0.38 |
| `7ejm` | TYIKWPWWV | 1.72 | mean RSCC 0.78, min 0.50 |
| `7s8r` | SALEWIKNK | 1.34 | mean RSCC 0.79, min 0.58; resolution 2.95 Å |
| `7sih` | NPDIVIYQY | 1.05 | mean RSCC 0.77, min 0.53. The same peptide in `7sif` is well resolved (0.94) |

`7ejm` and `7ejn` "improved" most at step 153 (1.72 → 1.02, 1.54 → 0.69). Against a
poorly resolved peptide, that is not evidence of anything. In training, 28
records have peptide mean RSCC < 0.8 and 4 carry a covalent ligand on the peptide
(5 across all splits: 3 MYR, 1 NAG, 1 3IB). Dropping them from training is a separate decision.

## Difficult (7)

| record | peptide | len | pretrained Å | note |
|---|---|---|---|---|
| `7t5m` | FLPTPEELGLLGPPRPQVLA | 20 | 7.56 | 3 residues unobserved; halves (≈3 Å) in three of four Round 5 arms |
| `8d5n` | HPGSVNEFDFGCGGSG | 16 | 3.66 | the C-terminally extended form of `8d5q`'s peptide; no arm moves it |
| `6zkx` | RLPAKAPLLGCG | 12 | 2.84 | |
| `7tud` | EEFGRC | 6 | 1.52 | |
| `7wt3` | XGANF | 5 | 1.32 | modified N-terminus |
| `7cir` | RRF[SEP]RSPIRR | 10 | 1.98 | phosphoserine; worse in every Round 5 arm (2.2–2.6) |
| `7ndu` | RMYSPVSIX | 9 | 1.62 | non-standard C-terminal residue |

## Can improve (canonical 8–11-mers, good density, > 1.2 Å) (17)

`7ndq` 2.58 · `7ow5` 2.45 · `7ow6` 2.25 · `8d5q` 2.20 · `7pb2` 2.04 (resolution
3.41 Å) · `7r7y` 1.96 · `7s8s` 1.89 · `7ciq` 1.86 · `7emd` 1.55 · `7su9` 1.54 ·
`8gqv` 1.51 · `7wbi` 1.34 · `7sif` 1.29 · `7ow3` 1.28 · `7u21` 1.26 · `7s8q` 1.24 ·
`7r7v` 1.22

Two families stand out:
* **KRAS peptides**: wild-type 7–16 `VVVGAGGVGK` (`7ow3`, `7ow5`), G12D 7–16
  `VVVGADGVGK` (`7ow6`, `7pb2`), and the G12D 10–18 region `GADGVGKSL` (`7su9`).
  They sit at 1.3–2.5 Å. `7ow4` has the same G12D 7–16 peptide but sits at 0.72 Å.
  That contrast is weaker than it looks: `7ow4`'s peptide density is mediocre
  (mean RSCC 0.82, min 0.53, 2 residues unobserved).
* **`8d5q`** (HPGSVNEFDF) is fixed by fine-tuning: 2.20 → 0.66–0.73 in three of
  four Round 5 arms.

## The rest (70)

These 70 records are 8–11-mers at or below 1.2 Å. They are not drop cases:
* 11 of them carry a modified residue (below);
* 1 (`7phr`) has no density data to judge.

There is little room here; this group's median is Boltz-1's floor.

## Modified residues (14 records)

From the RCSB canonical sequences:
* **Phosphoserine (SEP), 7 records:**
  * the RRF…PIRR family: `7cir` RRF[pS]RSPIRR (singly phosphorylated);
    `7dyn` RRF[pS]R[pS]PIRR; `7cis` RRF[pS]R[pS]PIR (9-mer);
  * EPR[pS]PSHSM: `7s8e`, `7s8a`, `7rzj`;
  * LPA[pS]PAHQL: `7s7f`.

  All but `7cir` are at or below 1.2 Å, so phosphorylation as such is handled.
  `7cir` is the outlier, and it gets worse with fine-tuning.
* **Other serine-site modifications in the EPR family:** `7s7d` carries
  O-sulfo-serine (OSE) and `7s79` carries E7P. Both are fine (≤ 0.41 Å).
* **Modified C-terminal residues:** `7ndt` and `7ndu` end in QM8 (canonical Leu);
  `6zky` ends in QMB. `7ndu` is at 1.62 Å; `7ndt` is at 0.50 Å.
* **Modified core residue:** `7p0a`, Phe6 → D-phenylalanine (DPN) (0.32 Å).
* **Myristoylated N-terminus:** `7wt3` (MYR-GANF). This is the same chemistry as
  the dropped `7wj3`, but deposited differently. In `7wt3` the myristoyl group is a
  residue of the peptide chain, so Boltz sees it. In `7wj3` it is a separate
  ligand, which our pipeline masks. With the lipid in the input, Boltz places
  `7wt3` at 1.32 Å; without it, `7wj3` lands at 15 Å. That is direct evidence that
  `7wj3` fails because of the missing lipid. It also suggests a fix: keep
  covalently linked ligands as part of the peptide.

Which modification each X is (for example, how Boltz tokenizes SEP) is the thing
to check in PyMOL for `7cir` vs `7cis`.

## Next

* Look at the drop and difficult cases in PyMOL. Write predicted coordinates for
  them (`reports/PLAN_round6.md` §2) to overlay.
* Run the eval medians again without the drop group.
* Apply the same rules to the 914 training records: decide whether the 28 + 4
  poor-density or modified peptides leave training.
