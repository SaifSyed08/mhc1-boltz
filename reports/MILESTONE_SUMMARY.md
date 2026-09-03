# MHC-I dataset preparation for Boltz-1 -- milestone summary

## 1. Sample counts

| stage | count |
|---|---|
| MHC-I entries listed by TCR3D | 1515 |
| resolved in RCSB | 1514 |
| with an MHC-I heavy chain AND a separate peptide chain | 1296 |
| after temporal split + leakage removal | 1296 |
| Boltz NPZ present locally so far | 1084 |
| chain-mapped and pMHC-I selected | 1084 |
| written to the processed dataset | 1084 |

## 2. Splits (AlphaFold-3 style, RCSB initial_release_date)

train <= 2021-09-30, validation 2021-10-01..2023-01-13, test > 2023-01-13

| split | after annotation | after leakage removal | written |
|---|---|---|---|
| train | 914 | 914 | 914 |
| validation | 99 | 99 | 99 |
| test | 283 | 283 | 71 |

## 3. Removals and why

| reason | count |
|---|---|
| single chain fusion construct | 30 |
| tcr3d reports no peptide | 27 |
| no separate peptide chain in structure | 12 |
| no mhc i heavy chain identified | 2 |
| dropped from train to break (allele, peptide) leakage | 128 |
| dropped from validation to break (allele, peptide) leakage | 19 |

## 3b. Candidates with no Boltz sample

212 of 1296 post-split candidates have no NPZ in rcsb_processed_targets.tar.

**212 of those 212 were released in 2024 or later** -- Boltz-1's PDB snapshot
effectively ends in early 2024, so they cannot be matched. They must be
processed from mmCIF with boltz's scripts/process/rcsb.py to be usable.

| release year | missing |
|---|---|
| 2024 | 68 |
| 2025 | 72 |
| 2026 | 72 |

## 4. Validation results

| check | pass | total |
|---|---|---|
| sample is a TCR3D MHC-I entry | 1084 | 1084 |
| expected MHC-I chain present + enabled | 1084 | 1084 |
| expected peptide chain present + enabled | 1084 | 1084 |
| kept chain sequences match RCSB | 1084 | 1084 |
| unrelated chains disabled (npz + manifest) | 1084 | 1084 |
| kept chains form one connected complex | 1084 | 1084 |
| peptide in alpha1/alpha2 groove | 1084 | 1084 |
| structural arrays intact | 1084 | 1084 |
| temporal split matches release date | 1084 | 1084 |
| no (allele, peptide) leakage across splits | 766 | 766 |
| Boltz loads / tokenizes / crops the file | 1084 | 1084 |

Failures: none
Needs manual review: 15

## 5. Chain selection statistics

- chains present in the original Boltz sample: median 4, max 43
- chains kept: {2: 21, 3: 1063}
- chains disabled in total: 3363
- MHC-peptide heavy-atom contacts (<=4.5 A): min 82, median 335, max 548
- flags: {'multiple_mhc_or_peptide_copies_resolved_by_contact': 18, 'ambiguous_peptide_choice_needs_review': 15, 'no_b2m_chain': 21}
