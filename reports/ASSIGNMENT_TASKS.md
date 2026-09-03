# Assignment tracker — *MHC-I Dataset Preparation for Boltz-1 Fine-Tuning*

Every task below is a line from the brief. Status is evidence-based: a task is
only DONE if something in this repo demonstrates it.

Legend: **DONE** · **PARTIAL** · **BLOCKED** · **TODO**

---

## 1. Understand the Boltz training pipeline

| # | Task | Status | Evidence |
|---|---|---|---|
| 1.1 | Review training instructions | DONE | `boltz-src/docs/training.md` |
| 1.2 | Download preprocessed RCSB structures | DONE | 1,286 NPZ via `src/tar_extract.py` |
| 1.3 | Download preprocessed RCSB MSAs | DONE | 1,457 files, `src/msa_extract.py`, 4.2 min |
| 1.4 | Download ligand symmetry file | DONE | `data/assets/symmetry.pkl`, 215 MB |
| 1.5 | Understand `manifest.json`, NPZ, configs, chain/residue repr | DONE | `README.md` "three findings" |
| 1.6 | Resume from pretrained weights, not from scratch | DONE | `pretrained:` set; `resume: null` |

## 2. Identify the MHC-I dataset

| # | Task | Status | Evidence |
|---|---|---|---|
| 2.1 | Use TCR3D MHC-I chain list as starting point | DONE | `src/step1_tcr3d.py`, 1,515 entries |
| 2.2 | Identify which entries exist as Boltz samples | DONE | 1,084 mapped |
| 2.3 | Map PDB ID / MHC chain / peptide chain / extra chains / release date / NPZ | DONE | `data/chain_selection.json` |
| 2.4 | Record entries that cannot be mapped reliably | DONE | `MILESTONE_SUMMARY.md` §3 |

## 3. Temporal train / validation / test split

| # | Task | Status | Evidence |
|---|---|---|---|
| 3.1 | Split by PDB release date (AF3-style boundaries) | DONE | `src/step4_split.py` — 914 / 99 / 71 |
| 3.2 | Keep test set untouched during development | DONE | test lives in `manifest_test.json`, absent from training manifest |
| 3.3 | Find identical MHC/peptide pairs across splits | DONE | V10 — 766 groups, 0 spanning |
| 3.4 | Document duplicates and handling | DONE | `data/candidates_dropped_for_leakage.json` |

## 4. Select biologically relevant chains

| # | Task | Status | Evidence |
|---|---|---|---|
| 4.1 | Identify correct MHC-I chain | DONE | V2 — 1084/1084 |
| 4.2 | Identify peptide chain | DONE | V3 — 1084/1084 |
| 4.3 | Decide which extra chains are necessary (β2m) | DONE | `src/step5_map_select.py` |
| 4.4 | Remove/disable unrelated chains in manifest **and** NPZ | DONE | V5 — mask + `valid` flags |
| 4.5 | Handle biological assembly / spurious cross-assembly contacts | DONE | README finding #2; V6 |

## 5. Resolve chain-ID mapping

| # | Task | Status | Evidence |
|---|---|---|---|
| 5.1 | Reproducible mapping across auth/label/assembly/Boltz ids | DONE | README finding #1 — `label_asym_id` matched 1152/1152 vs auth 1010/1152 |
| 5.2 | Do not rely on chain names alone | DONE | sequence-based; names only cross-checked |
| 5.3 | Use sequence/length/polymer type/alignment as fallback | DONE | `src/seqlib.py` |
| 5.4 | Automate and flag ambiguous cases | PARTIAL | 15 samples flagged in `validation_report.json` — **never reviewed by hand** |

## 6. Validate the processed dataset

| # | Task | Status | Evidence |
|---|---|---|---|
| 6.1–6.11 | All 11 required checks | DONE | V1–V11, `reports/validation_report.json` |
| 6.12 | *(added)* referenced MSA files present on disk | DONE | V12 — 1457/1457 |
| 6.13 | Validation summary counts | DONE | `MILESTONE_SUMMARY.md` |
| 6.14 | List of ambiguous samples for separate review | PARTIAL | list exists; review outstanding (see 5.4) |

**12/12 checks pass, 0 failures.**

## 7. Prepare the Boltz training configuration

| # | Task | Status | Evidence |
|---|---|---|---|
| 7.1 | Point config at structures / MSAs / symmetry / weights / manifest | DONE | `configs/mhc1_finetune.yaml` |
| 7.2 | Add validation samples to a `validation_ids.txt` | DONE | `validation_ids_boltz.txt` (upstream 552 + our 99) |
| 7.3 | `samples_per_epoch: 100` | DONE | set |
| 7.4 | Run pretrained model with `validation_only` for a baseline | **DONE** | 30/30 samples, `lddt_protein_protein` **0.8827**; `reports/BASELINE.md` |
| 7.5 | wandb entity | TODO | only remaining placeholder — **needs Saif** |

## 8. Fine-tuning stage

| # | Task | Status | Evidence |
|---|---|---|---|
| 8.1 | Load processed dataset correctly | DONE | proven by the baseline run reading it |
| 8.2 | Resume from pretrained weights | DONE | checkpoint loads (confidence keys correctly unused) |
| 8.3 | Complete multiple epochs without errors | **BLOCKED (tested)** | 3 configs run, all OOM; frozen-trunk fails in triangular attention forward — `GPU_REQUIREMENTS.md` |
| 8.4 | Record metrics in W&B | BLOCKED | depends on 8.3 and 7.5 |
| 8.5 | Compare against pretrained baseline | BLOCKED | depends on 7.4 + 8.3 |

---

## Milestone-1 deliverables (brief, final page)

| # | Deliverable | Status |
|---|---|---|
| 1 | Total MHC-I samples from TCR3D | DONE — 1,515 |
| 2 | Number matched to Boltz samples | DONE — 1,084 |
| 3 | Train / validation / test counts | DONE — 914 / 99 / 71 |
| 4 | Number removed + reasons | DONE |
| 5 | How TCR3D chain IDs were mapped to Boltz chains | DONE |
| 6 | How assemblies / spurious contacts were handled | DONE |
| 7 | Validation check results | DONE — 12/12 |
| 8 | Remaining ambiguous / unresolved samples | PARTIAL — 15 listed, unreviewed |
| 9 | Final `manifest.json` | DONE |
| 10 | NPZ dataset | DONE — 1,084 |
| 11 | Train/val/test sample lists | DONE |
| 12 | Results look reasonable -> proceed to fine-tuning | DONE -- baseline established |

---

## Honest gaps

Three things a reviewer would catch, listed so they are not discovered *for* us:

1. **The 15 ambiguous samples were never manually reviewed.** The brief asks for
   a list that "can be reviewed separately" — the list exists, the review does
   not. This is the most likely thing to be asked about.
2. **The test split is 71, not 283.** Boltz-1's PDB snapshot ends in early 2024,
   so most post-2023 MHC-I structures have no Boltz sample. Getting a full-size
   test set means processing 2024+ mmCIFs with `scripts/process/rcsb.py`
   (needs `ccd.rdb` + redis). Documented in `README.md`, not done.
3. **The baseline is 30 of 99 validation samples**, seeded subset (ids in
   `data/processed/subset30_ids.txt`). The fine-tuned model must be scored on
   *that same subset*, or use `src/combine_val_states.py` to merge additional
   chunks correctly.

## Not in the brief, but done because it was necessary

* Two upstream Boltz v1.0.0 defects found and handled — `UPSTREAM_PATCHES.md`.
  Without the first, **no training or validation run works at all**.
* `src/val_progress.py` + `src/combine_val_states.py` so an interrupted
  validation keeps its completed samples and chunks merge exactly.
