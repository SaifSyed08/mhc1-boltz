# Audit 2026-10-09: Round 5 code, before any job ran it

The audit ran on 2026-10-09 while the Round 5 jobs sat in the queue, under the standing
rule that every batch of changes gets an independent auditor. The auditor was a fresh,
read-only reviewer with no torch and no cluster access. It audited commits 500f121,
8913283 and b17d4d3, plus the submission script. Its report is reproduced in full below.
Every claim acted on was checked again first-hand, as noted in the table.

## What was done about each finding

| finding | disposition |
|---|---|
| CRITICAL-1: peptide scored against Boltz's chain-swapped truth | **Fixed in 822e470**, before any job had started. Every CA metric is now scored against `batch["coords"]`, and `boltz_truth_dev` reports per sample how far Boltz's truth is from the real one. Checked first-hand: `get_chain_symmetries` checks `chain_in_crop` only for the replaced chain; `load_input` keeps `mask`; 7ndt's chains F1/G1/H1 are masked twins of A1/B1/C1 (same entity and atom count), 102/120/67 Å away. The unit test (`tests/test_peptide_rmsd.py`, 19/19 on LS6's venv) has a swapped-truth case. |
| MAJOR-2: the same swap in Boltz's own val/rmsd and val/lddt | **Documented, not patched** (`UPSTREAM_PATCHES.md` §8): fixing it breaks comparability with Rounds 1–4, so it is a decision for Saif and Ernest. `RESULT_round4.md` §4 now attributes the excursion to it. Training loss is not affected: training uses the symmetry correction only for the confidence head, which is off (`model.py` training_step). |
| MAJOR-1: best-of-3 is an oracle pick | **Fixed in 822e470**: the summary line headlines the median over all samples and labels best-of-N as an oracle. `MHCFINE_COMPARISON.md` says which number is compared with 1.59 Å, and that an 8–11-mer subset is needed. |
| MINOR-1: groove window shifted by unresolved leaders | **Fixed in 822e470**: the window starts at the first resolved heavy-chain Cα, recorded as `groove_start`. |
| MINOR-2: a load failure relabels a row | **Analysis step**: grep the eval log for `Skipping`, and check each row's `hc_len` / `pep_len` against its record's processed structure. |
| MINOR-3: smoke gate `pep_rmsd < 30` | **Moot**: the smoke's six records have no twins, and peptide scoring no longer uses the swapped truth. |
| MINOR-4: `train/grad_norm` in mem.jsonl | **Analysis note**: it is logged before backward, so it is the previous step's post-clip norm on the first micro-batch of each accumulation window and a partial sum otherwise. Read only rows with `batch % accumulate == 0`, as "previous step". |
| MINOR-5: eval rows carry global_step 0 | **Analysis note**: the directory (`runs/eval_val99/{pretrained,step153}`) identifies the checkpoint. |
| MINOR-6: `_pairs` fields are 0/1; non-standard peptide residues | **Docstring fixed in 822e470**; the length and residue caveats are in `MHCFINE_COMPARISON.md`. |
| MINOR-7: smoke does not exercise batch 9 | **Watch**: the first hour of each b9 arm's `mem.jsonl` (`train/loss`, `lr`). |
| MINOR-8: re-chained b129 jobs not auto-cancelled | **Moot**: every arm's dependency was removed on 2026-10-09 (`MHCFINE_COMPARISON.md`). |

---

# Round 5 audit (peptide RMSD stash, smoke gate, val99 eval, b9 arms)

BLOCKER: peptide_rmsd scores against Boltz's chain-swapped "truth", so 7ndt (subset30, every training-arm validation) and 7bh8 (val99) get garbage peptide RMSDs -- 2-line fix in src/peptide_rmsd.py only (no patch change), apply before the jobs start (CRITICAL-1). Nothing else blocks.

Scope: outer commits 500f121, 8913283, b17d4d3; boltz-src working tree vs
patches/boltz-v1.0.0-training-path.patch; scratchpad ls6_round5_submit.sh and
test_peptide_rmsd.py. Read-only; no torch imported, no bridge used. PL internals
were checked against the local copy of pytorch_lightning 2.4.0
(.venv-gpu/Lib/site-packages/pytorch_lightning), the same version as LS6.

## Findings

### CRITICAL-1. Symmetry correction swaps kept chains with masked twin copies; the stash hands that "truth" to peptide_rmsd
- Evidence:
  * boltz-src/src/boltz/data/feature/symmetry.py:459-470 (`get_chain_symmetries`): for every
    in-crop chain, swap candidates are ALL chains of `structure.chains` with the same entity_id
    and atom count -- `chain_in_crop` is checked for chain i but not for chain2 j. The structure
    here is the full NPZ: load_input (training.py:103-112) keeps `mask`, and the tokenizer only
    skips masked chains when tokenizing (tokenize/boltz.py:60); step6_write.py masks unrelated
    chains on purpose instead of deleting them.
  * symmetry.py:263-286 (`minimum_lddt_symmetry_coords`) keeps whichever combination gives the
    highest all-atom lDDT against the prediction. lDDT only scores pairs within 15 A IN THE TRUTH,
    so replacing a kept chain by a twin that is far away deletes all of that chain's inter-chain
    pairs from the score. Inter-chain pairs are predicted worse than intra-chain ones (your own
    table: lddt_pp ~0.886 vs intra ~0.925), so the swap routinely WINS.
  * model.py:886-889 (the new stash) passes that swapped `true_coords` / mask to
    src/peptide_rmsd.py:152, which uses it for pep_rmsd, groove_rmsd, ca_rmsd, ca_rmsd_mirror.
  * Affected records (scanned all 99 val NPZs for same-entity, same-atom-count twins of kept
    chains): 7ndt (subset30 index 9 -> rank 0, batch 3 in every training-arm validation;
    val99 index 28) with a whole second pMHC copy F1/G1/H1 that is 41-80 A from every kept chain
    (centroid shifts 102 / 120 / 67 A); 7bh8 (val99 index 2) with twins C1/D1/J1 that pack
    against the kept chains (2.9 A).
- Reproduced with numpy on 7ndt's real coordinates (prediction = truth with b2m and peptide
  rigidly nudged 3-5 deg / ~1 A plus 0.3 A noise): Boltz's criterion ranks truth
  [A1, B1, H1] (peptide replaced by the far copy) at lDDT 0.9246 ABOVE the real truth
  [A1, B1, C1] at 0.9219, so it is selected; all-atom RMSD vs the selected truth is 10.2 A
  instead of 0.7 A, and the groove-superposed peptide is compared with a peptide ~67 A away.
  Swaps of b2m (G1) give ~49 A all-atom RMSD -- the size of the "~40 A excursion" in
  RESULT_round4.md section 4. That excursion is very likely this artefact on 7ndt (the only subset30 record with twin chains), not a
  model failure and not a mirror image.
- Failure scenario in the queued jobs: every validation in all four training arms writes a 7ndt
  row whose pep_rmsd is tens of A whenever the swap wins (round 4: "most of the time"), with
  ca_rmsd and ca_rmsd_mirror both large, so it reads as "the model fails on this complex" (and the
  mirror check does not flag it). Same for 7ndt/7bh8 in the val99 eval. The predictions are not
  saved, so these rows cannot be recomputed afterwards. Medians over 30/99 rows are robust to it;
  means and the per-structure story are not.
- Smallest fix (only src/peptide_rmsd.py, read at runtime, so a `git pull` on LS6 before the jobs
  start is enough; no patch change, Boltz's own metrics untouched): score CA geometry against the
  unswapped ground truth already in the batch. In `score_structure`, replace line 152's
  `true, tmask` with
      true = batch["coords"][b, 0]
      tmask = batch["atom_resolved_mask"][b].bool()
  (compute once, outside the sample loop; keep `pred = stash["pred"][i]`). batch["coords"] is
  (B, 1, N_atoms, 3) after collate (training.py:139-166), centred but otherwise the raw truth,
  same atom indexing as the prediction, and nothing in validation_step modifies it. CA atoms have
  no intra-residue symmetry and the kept chains are three distinct entities, so the symmetry
  correction contributes nothing legitimate to CA metrics. Keep `boltz_rmsd` / `lddt_*` as
  Boltz's own numbers; after the fix, a sample with boltz_rmsd >> ca_rmsd is exactly a sample
  where Boltz picked a swapped truth, which is a free diagnostic worth printing.

### MAJOR-2. The round-4 "RMSD excursion" is this artefact, and it also leaks into val/lddt
- Same mechanism as CRITICAL-1, in Boltz's own metrics (pre-existing, not introduced by this
  batch): for 7ndt, val/rmsd and val/best_rmsd are measured against a disassembled complex
  whenever a swap wins, and 7ndt's val/lddt contribution (lddt_pp included) is computed on
  whichever truth maximises lDDT -- the swap removes precisely the worst-predicted interface
  pairs, so a mispositioned peptide or b2m is hidden from lddt_pp for that structure, and the
  winner flips between validations (extra noise, ~1/30 of the mean).
- Interpretation impact: RESULT_round4.md section 4's leading candidate ("a mirror image") is
  very likely wrong; the 7ndt rows will show it after the CRITICAL-1 fix (boltz_rmsd large,
  ca_rmsd small, ca_rmsd_mirror not smaller).
- Fix, separate decision because it changes val/lddt and val/rmsd and breaks comparability with
  rounds 1-4: in get_chain_symmetries also require `chain_in_crop[j]` for the swap target, or
  drop masked chains before validation. Not needed for the queued jobs if CRITICAL-1 is fixed.

### MAJOR-1. "Best-of-3" peptide RMSD is an oracle pick; MHC-Fine's 1.59 A is a confidence-ranked top-1
- Evidence: src/peptide_rmsd.py:172 `out["pep_rmsd_best"] = min(out["pep_rmsd"])`, and the
  console summary at :229-231 prints "median best-of-N MHC-peptide CA RMSD". The config has
  `confidence_prediction: false` (configs/mhc1_finetune_ls6_unfrozen.yaml:231), so there is no
  pLDDT to rank by; "best" is chosen by looking at the answer.
  reports/ls6/MHCFINE_COMPARISON.md quotes MHC-Fine's base model as 1.59 A "best of its five
  models by pLDDT" (a top-1 by confidence, no access to the truth) and makes a decision rule out of
  the comparison: "If it already starts near 0.7 A, there was little left to gain and a flat curve
  is the right answer."
- Failure scenario: the val99 eval prints, e.g., "median best-of-N ... 0.9 A" for the pretrained
  model; that number is biased low by the oracle min over 3 diffusion samples, so it overstates
  how close Boltz-1 already is to MHC-Fine's 0.66 A and pushes toward the "little left to gain"
  branch for the wrong reason. (Same trap the project already documented for val/lddt.)
- Smallest fix: headline the median over ALL per-sample values (or sample 0 only, i.e. one
  random sample, the closest analogue of a confidence pick when no confidence head exists) and
  report best-of-3 only as a labelled secondary ("oracle best of 3"). Change the print at :229 to
  show both, and add one sentence to MHCFINE_COMPARISON.md saying which one is compared to 1.59 A.
  Nothing is lost on disk: every row keeps all three per-sample values, so this is purely a
  reporting fix and can be made after the jobs run.

### MINOR-1. Groove = "first 180 heavy-chain tokens" is not alpha1/alpha2 for 6 of the 99 records
- Evidence: src/peptide_rmsd.py:144 `groove_tok = chains[hc_a][:GROOVE_RESIDUES]`. Checked the
  processed NPZs: 7rtd, 7rtr, 7tlt carry the full 365-residue precursor whose first 24 residues
  are the unresolved signal peptide (MAVMAPRTLVLLLSGALALTQTWA, then GSHSMRYF...); 7phr has 12,
  7ejm/7ejn 7 leading unresolved residues. Unresolved CAs are dropped by the mask, so for the three
  signal-peptide records the "groove" is mature residues 1-156 (the C-terminal third of the alpha2
  helix, 157-180, is missing from the superposition); for the others it is 1-168/1-173.
  None of these is in subset30 (training-time validations unaffected); all are in val99.
- Effect: slightly different superposition frame on 6/99 rows; medians essentially unaffected.
- Smallest fix: start the window at the first heavy-chain token whose CA is resolved in the truth
  (`first = first index in chains[hc_a] with frame_ok & tmask[frames[...,1]]`, then take 180 tokens
  from there), or record `groove_start` so these rows can be filtered.

### MINOR-2. A dataset load failure silently relabels a row with the wrong record id
- Evidence: boltz-src/src/boltz/data/module/training.py:390-446 -- any exception in load_input /
  tokenizer / cropper / featurizer prints "... Skipping." and returns `self.__getitem__(0)`, i.e.
  record 0's features. src/peptide_rmsd.py:196-198 assigns the id from the sampler position, so
  that row would carry record X's id with record 0's (6zkx's) structure. 69 of the 99 eval records
  have never gone through ValidationDataset on LS6 (BASELINE_ls6.md and every LS6 run used
  subset30); a missing MSA file on LS6 for one of them is the realistic trigger. No such message
  appears in any log in the repo, so probability is low.
- Smallest fix (pick one): (a) after the eval, `grep -c "Skipping" logs/eval99.<job>.out` must be 0
  before rows are trusted; (b) in record_ids also keep each record's valid-chain residue counts
  from the manifest and flag `record_id_mismatch` when the scored heavy-chain token count differs
  (peptide token count can legitimately exceed residues for modified residues).

### MINOR-3. Smoke gate `0 <= pep_rmsd < 30` tests prediction quality, not the code path
- Evidence: scripts/ls6_ddp_smoke.slurm:86-87. RESULT_round4.md section 4 documents one subset30
  structure that is ~40 A off (all-atom) in all three samples in most validations, present in the
  pretrained control too. The smoke validates subset30 indices 0-5 (rank r gets r and r+3):
  6zkx, 7ciq, 7cir (laptop baseline chunk 1 shows these at ~1.3 A all-atom: fine) and 7em9,
  7emb, 7emc. Per CRITICAL-1/MAJOR-2 the outlier is very likely 7ndt (index 9), which the
  smoke does not validate, so the practical risk is now low. But the gate still couples "the code
  works" to "the model predicts well": a genuinely bad sample >= 30 A would fail the smoke and
  --kill-on-invalid-dep would kill all five jobs although the code is correct.
- Smallest fix: gate on `math.isfinite(x) and x >= 0` only; keep printing the values. (Only worth
  doing if the smoke has not already passed.)

### MINOR-4. `train/grad_norm` now recorded in mem.jsonl is not the step's gradient norm
- Evidence: boltz-src model.py:536/540 logs `train/grad_norm` from inside training_step, i.e.
  BEFORE this micro-batch's backward. PL 2.4 closure order is step_fn -> zero_grad_fn -> backward
  (loops/optimization/automatic.py; zero_grad only when `batch_idx % accumulate == 0`). So the
  value is: on the first micro-batch of a window, the previous step's post-clip gradient norm
  (grads are not cleared until after training_step); otherwise a partial accumulation. At b9
  that pattern repeats every 3 rows, at b129 every 43.
- Effect: reading mem.jsonl's grad_norm as "gradient norm per step" (e.g. to judge clipping at
  b9 vs b129) is wrong. train/loss and lr are fine (lr = the LR the next step will use).
- Fix: none needed in code; filter to rows with `batch % accumulate == 0` and label it
  "previous step, post-clip", or log the norm in `on_before_optimizer_step` if it matters.

### MINOR-5. Eval rows do not say which checkpoint produced them
- Evidence: `pretrained=` loads weights only, so trainer.global_step is 0 for both checkpoints;
  rows carry `global_step: 0`, and the val_state files are named `..._epoch000_step00000.json`
  in both directories. Only the directory (runs/eval_val99/{pretrained,step153}) tells them apart.
- Fix: keep the directories separate when pooling, or add the checkpoint name to each row.

### MINOR-6. Field and doc details in peptide_rmsd
- `lddt_pp_pairs` / `lddt_intra_pairs` (peptide_rmsd.py:216-218, docstring :31 "pair totals")
  are 0/1 indicators: factored_lddt_loss converts totals with `(total > 0).float()` unless
  cardinality_weighted (validation.py:194-196). The unit test feeds 100.0, which cannot occur.
- 14 of the 99 val peptides contain non-standard residues (SEP, QM8, QMB, E7P, OSE, DPN, MYR).
  Boltz tokenizes those per atom (res_type UNK, frame mask False), so their CA is excluded from
  pep_rmsd and pep_len counts atoms, not residues (7cis: 9 residues -> 27 tokens; max is 28,
  under MAX_PEPTIDE=30, so no record is misidentified). MHC-Fine's peptides are 8-11-mers;
  ours span 4-20 (7wj3 GAAL, 7wt3 MYR-GANF, 7t5m 20-mer). Worth a filter when comparing.
- If `devices` ever stops dividing the val set (99 or 30), DistributedSampler pads by repeating
  the first indices: a duplicate row for record 0 (and Boltz double-counts it). Not the case for
  3 GPUs; a one-line warning in record_ids would catch it.

### MINOR-7. The smoke does not exercise the b9 path
- The smoke runs at accumulate 43 (2 optimizer steps, LR 0 then 1e-5). The b9 arms reach full
  LR after 11 steps; a divergence at b9 / 3e-4 is not covered. Same code path, so low risk;
  watch the first hour of the b9 jobs' mem.jsonl (train/loss, lr).

### MINOR-8. Re-chained b129 jobs may not be auto-cancelled if the smoke fails
- ls6_round5_submit.sh:24 re-chains 3499581/3499582 with `scontrol update ... Dependency=`;
  `--kill-on-invalid-dep=yes` was passed only to the new sbatch calls. Unless LS6 sets
  kill_invalid_depend globally (UNVERIFIED), a failed smoke leaves those two pending forever
  (DependencyNeverSatisfied). Check squeue after the smoke.

## Verified OK (with evidence)

1. Stash names and shapes. Every name the stash uses exists on every path that reaches it:
   `out` (model.py:622), `true_coords`, `rmsds`, `true_coords_resolved_mask` (:668-675),
   `all_lddt_dict`, `all_total_dict` (:677-683); both OOM exits (:630-637, :684-691) return before
   the stash, after `self._val_stash = None` at :618, so a skipped batch gives a "no predictions"
   row. `out["sample_atom_coords"]` is (B*n, N_atoms, 3), sample index b*n+s
   (diffusion.py `atom_mask.repeat_interleave(multiplicity, 0)`; fk_steering false in config).
   `true_coords` is per sample in both branches: symmetry branch loops idx*n+rep
   (model.py:393-413) and pads each to the pred atom count (symmetry.py:356-377; truth aligned by
   a proper rotation, weighted_rigid_align det fix); non-symmetry branch `repeat_interleave(n, 0)`
   (model.py:415-421). Same b*n+s order the callback uses (:151). `rmsds` is a list of floats
   (symmetry branch) or a tensor; `_floats` handles both. lddt dicts are (B*n,).
2. CA index. frames_idx[t] = [N, CA, C] for protein tokens with res_type in ref_atoms, as
   `ref_atoms[res].index(...) + atom_idx`, atom_idx being the running offset into the same
   per-example atom array the model predicts (featurizer.py:741-791); padding is appended at the
   end (:858-879), so indices stay valid. frame_resolved_mask requires N, CA, C present and
   non-collinear; the callback also requires the CA resolved in the truth.
3. Kabsch math (peptide_rmsd.py:62-79): R = V diag(1,1,d) U^T from H = P^T T, applied as
   (Q - pc) R^T + tc; reflection fix on the smallest singular value; allow_reflection gives the
   optimal orthogonal map. Correct.
4. Chain identification on the real data: all 99 val records have one heavy chain (180-365
   residues), peptide 4-20 residues (<= 28 tokens), b2m in 98/99; no TCRs (step5 KEEP_TCR=False).
   Longest/shortest picks the right chains for all 99.
5. record_id mapping. PL 2.4 `_process_dataloader` injects DistributedSampler(shuffle=False) for
   val (data_connector.py:228-242, 468-484); `trainer.val_dataloaders` returns the processed
   loader (trainer.py:1496-1502 + CombinedLoader.flattened setter); `setup_data()` runs before
   `on_validation_epoch_start` (evaluation_loop.py:108-115, 239-245). val_batch_size is asserted 1
   (training.py:482); ValidationDataset maps a flat index exactly as record_ids walks it
   (training.py:378-387, overfit null). 99 and 30 divide by 3: no padding. Rank r gets r, r+3, ...
   in order (worker order preserved).
6. Smoke counts. `train/loss` is logged with default on_step=True/on_epoch=False in
   training_step (model.py:535), so the key is exactly `train/loss`, on every micro-batch;
   callback_metrics is recomputed lazily from the running results while the batch is in flight
   and includes step values in training (logger_connector.py:230-241, result.py:468-492);
   MemProbe's on_train_batch_end runs before `on_batch_end`. Sanity rows are flagged via
   trainer.sanity_checking; limit_val_batches=2 x 3 ranks gives exactly 6 real rows. The first 6
   subset30 records (6zkx, 7ciq, 7cir, 7em9, 7emb, 7emc) have no twin chains, so CRITICAL-1 does
   not touch the smoke.
7. Eval plumbing. ls6_unfrozen.slurm puts `output=$RUN_DIR` first in ARGS (line 68), so each
   checkpoint writes to its own dir; RUN_DIR is fresh (rm -rf), so the resume guard never fires;
   CONFIG is honoured (line 102); the skip test uses the per-epoch file that only a finished,
   non-sanity validation writes. Config diff vs unfrozen (CR stripped): header, output, split.
   validation_ids.txt: 99 ids, all in manifest.json, all released 2021-10-13..2023-01-11, so the
   training DateFilter (<= 2021-09-30) keeps all of them out of every training arm; subset30's 30
   manifest matches are a subset (its other 552 lines are upstream ids that match nothing).
   No limit_val_batches in the eval path. Timing: round 4 measured ~5 min per 30-record
   validation on 3 GPUs, so 99 records x 2 checkpoints fits the 2 h dev-queue limit.
8. pretrained loading and EMA. train.py:156-158 `load_from_checkpoint(strict=False, **cfg
   hparams)`; PL 2.4 calls on_load_checkpoint before load_state_dict and warns on missing or
   unexpected keys (core/saving.py:171-197). Boltz1.on_load_checkpoint restores the EMA shadow
   when compatible (model.py:1258-1269); on_validation_start -> prepare_eval copies the EMA into
   the weights (:1288-1299). So step153 is evaluated with its EMA weights, the same thing every
   training-time validation scored. Caveat: the copy is the 03:36 last.ckpt; EMA updates every
   micro-batch, so the 03:57 stop-time validation used an EMA ~36 micro-batches newer (tiny).
   The pretrained eval takes the identical path as the round-4 control, so it is comparable to
   it whether or not boltz1_conf.ckpt's EMA is compatible.
9. LR at b9 (AlphaFoldLRScheduler, interval "step", stepped only on optimizer steps): step 1 at
   LR 0, then k/10 * max_lr, max_lr from step 11 to 50,000 (no decay reached). b9: full LR after
   ~30 micro-batches per rank (~18 min); b129: after one epoch (~4 h). The CLI dotlist reaches
   Boltz1 via hydra and `**model_module.hparams` in load_from_checkpoint; values are passed as
   0.0001 / 0.0003 (floats). PL accumulation boundaries reset each epoch
   (training_epoch_loop.py:147-171, 327-343; automatic.py:218), consistent with the zero_grad
   rule: at accumulate 3 an epoch of 430 micro-batches/rank is 143 full steps plus a 1-batch
   step; nothing lost or double-counted. Validation stays once per 430-batch epoch (~4 h 15 min,
   unchanged); ~1,580 steps per 47 h at b9 matches round 4's throughput. Checkpointing unchanged
   (30-min last.ckpt + best-by-val/lddt at epoch end). EMA decay is per micro-batch, so its
   horizon in samples is the same at b9 and b129.
10. Patch consistency: `diff <(git -C boltz-src diff | grep -v '^index ') <(grep -v '^index '
    patches/boltz-v1.0.0-training-path.patch)` is empty (exact, LF); HEAD b17d4d3 == origin/master.
11. Telemetry cannot hang a rank: no collectives in PeptideRMSD or the new MemProbe code;
    callback_metrics in on_step mode does not compute()/sync.

## UNVERIFIED
- Whether 7bh8's swap actually wins (its twins pack against the kept chains at 2.9 A, which adds
  new truth pairs and may keep identity on top). The CRITICAL-1 fix makes it irrelevant.
- Whether boltz1_conf.ckpt carries a compatible "ema" entry (settles which weights the
  pretrained eval scores; comparability with the control holds either way). Comparing
  len(ckpt["ema"]["shadow_params"]) with the model's trainable tensor count would settle it.
- Whether any of the 69 never-validated records fails to load on LS6 (MINOR-2): grep the eval
  log for "Skipping".

## On the unit test (scratchpad/test_peptide_rmsd.py; imports torch, so not run)
It drives the real callback and kabsch code with synthetic inputs and known answers, so the
geometry checks are meaningful, not tautological. What it cannot see: it builds `stash["true"]`
itself, so it never touches validation_step / get_true_coordinates, which is exactly where
CRITICAL-1 lives; frames_idx is synthetic (the real CA index was checked by reading the
featurizer); record_ids gets `sampler=[0]` and one rank; `lddt_total` is 100.0, a value Boltz
never produces.

