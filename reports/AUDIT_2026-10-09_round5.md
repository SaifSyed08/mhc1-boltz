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


---

# Second audit (same day): the fix, 822e470 and 0c472cf

Verdict: no blockers; the code fix is correct. Dispositions:

| finding | disposition |
|---|---|
| MAJOR-1: round-4 swap explanation wrong in detail (four modes; 7ndt ~49 / ~10 Å; ~1.6 Å, not ~1.3) | **Fixed in docs**: `RESULT_round4.md` §4 and `UPSTREAM_PATCHES.md` §8 now carry the four-mode table and the constant 1.46 / 1.38 Å for the other 29, worded as a hypothesis. The sums were re-derived from the table first-hand. |
| MINOR: best-checkpoint picks within ~0.002 | **Noted** in `UPSTREAM_PATCHES.md` §8: re-score close picks without 7ndt. |
| MINOR: docstring said 85-120 Å | **Fixed** (67-120 Å). Docstring only, so LS6 keeps 822e470 until the next pull. |
| MINOR: `groove_start` non-zero for 25 of 99, not 6 | **Accepted**: most start 1-2 residues in, the effect is negligible, and the field is recorded. |
| MINOR: test uses one mask for both truths, all frames resolved | **Open**: extend the test at the next code change. |
| MINOR: overstated wording | **Fixed**, as part of MAJOR-1. |
| Unverified: LS6 at 822e470 | **Verified**: the pull printed `822e470` and ran the tests there. First eval rows to be checked for `groove_start`. |

NO BLOCKERS. The fix in 822e470 is correct: batch["coords"] is the unswapped truth, indexed like the prediction, when the callback runs. One MAJOR documentation finding (MAJOR-1): the round-4 swap story is quantitatively stronger than written but wrong in detail, and every subset30 Boltz rmsd, the "good" mode included, carries a 7ndt artefact. Five MINOR.

# Audit of the Round 5 fix (822e470) and docs (0c472cf)

Started 18:20 CDT 2026-10-09. Read-only; numpy only; no torch, no bridge.

## Findings (appended as confirmed)

### MAJOR-1 (docs, interpretation; does not affect any job). The swap story is right, but RESULT_round4 section 4 and UPSTREAM_PATCHES section 8 describe it wrongly in detail, and they miss that the "good" validations are contaminated too
Evidence:
* numpy on data/processed/structures/7ndt.npz. All 8 chain-swap combinations Boltz can offer for the kept chains A1/B1/C1 (twins F1/G1/H1: same entity, same atom count, identical atom-name arrays). All-atom Kabsch RMSD of each swapped truth against the real truth: none 0.0; peptide only 10.2; b2m only 48.5; b2m+peptide 49.2; HC only 49.2; HC+peptide 48.6; HC+b2m 10.3; all three (the other NCS copy) 1.2 A. **No swap gives ~40 A. The possible levels are about 1, 10 and 49 A.**
* RESULT_round4.md table (lines 14-29), turned into sums (rmsd x 90 samples, best_rmsd x 30 structures). There are four modes, not two:
  - high (control and 10 others): sum rmsd ~275, sum best ~89.3
  - steps 20/71/91/121: 235.5 / 50.5. Drop 39 in both, so ONE sample is 39 A lower.
  - step 51: 156.3 / 49.7. rmsd drops 118 = 3 x 39.3, best 39.3, so ALL THREE samples are 39 A lower.
  - step 111: 226.7 / 41.4. Drop 47.7 in both, so ONE sample is 48 A lower.
* Fit the swap levels to 7ndt and keep the other 29 structures fixed. In high mode all 3 samples are at ~49 A. In the 20/71/91/121 mode one sample is at ~10 A, a peptide swap: 49 - 10 = 39. At step 51 all three are at ~10. At step 111 one sample is at ~1.5 A, actually scored right: 49 - 1.5 = 47.5. Take 7ndt out and the other 29 come to rmsd 127/87 = 1.46 A and best_rmsd 40/29 = 1.38 A, the same to +/-0.01 in every one of the 16 validations (control: (274.0 - 147)/87 = 1.46, (89.0 - 49)/29 = 1.38; step 111: (226.7 - 99.5)/87 = 1.46, (41.4 - 1.5)/29 = 1.38).
What the docs get wrong:
1. RESULT_round4 section 4 says "one of its three samples comes in ~39 A closer" and "one complex is ~40 A off". Under the swap account, which the numbers above fit at every step, 7ndt sits at ~49 A in high mode and its *low* mode is still a ~10 A peptide-swap artefact. Only step 111 has a correctly scored 7ndt sample. Step 51 (all three samples) and step 111 (a 48 A drop) do not fit the 3:1 reasoning the section uses, and the note does not mention them.
2. UPSTREAM_PATCHES section 8 says the swap is "worth ~1.3 A of the subset30 mean RMSD". That is the gap between two modes, not 7ndt's share. Its share in high mode is ~1.63 A of BOTH val/rmsd (147/90) and val/best_rmsd (49/30). In the "good" mode it is still ~1.2 A of rmsd and ~0.33 A of best_rmsd. So every Boltz rmsd/best_rmsd ever recorded on subset30 carries a 7ndt artefact. That includes the r1 step 20 and 71/91/121 values the docs treat as the clean mode, and the pretrained baseline.
3. Missing, and the most useful result: with 7ndt removed, Boltz's own RMSD on the other 29 structures has not moved by more than ~0.01 A in any validation, control included. That fits "training has not changed the structures" (consistent with flat lddt_pp). It belongs in the docs, labelled as inferred from sums.
4. "A swapped b2m gives ~49 A". An HC swap also gives ~49. The ~40 A that the docs then match it to is a *difference* (49 -> 10), not a level.
5. Wording: the RESULT_round4 note says "So the excursion is a metric artefact, not a model failure, and the mirror-image candidate below is very likely wrong", and section 8 says "almost certainly". That is still a hypothesis until the 7ndt rows land. The quantitative fit above makes it much stronger than the docs show, but say "fits every validation quantitatively; confirmed when boltz_truth_dev on 7ndt reads ~49/~10".
Smallest fix: replace the 3:1 paragraph's interpretation with the four-mode table above. Change "worth ~1.3 A" to "7ndt alone is ~1.6 A of val/rmsd and val/best_rmsd in high mode, ~1.2/0.33 A in the 'good' mode". Add the 29-structure constants (1.46 / 1.38 A). For whoever reads the eval rows, say what values boltz_truth_dev should take on 7ndt (next item).

Expected boltz_truth_dev values, so the eval rows can be read without guessing (numpy, CA-only Kabsch of each possible swapped truth against the real truth):
* 7ndt: no swap 0.00; whole twin complex (F1,G1,H1) 0.54, a harmless NCS-copy swap; peptide-only or HC+b2m 10.7; any b2m-only or HC-only combination 48.1-49.0.
* 7bh8 (val99 only; twins A1/C1, B1/D1, I1/J1): none 0.0; all three 1.1; peptide or HC+b2m ~10; the others 16.6-19.2.
So "boltz_truth_dev is large on exactly the samples where Boltz scored a swapped truth" (UPSTREAM_PATCHES section 8, peptide_rmsd docstring) is true only for *harmful* swaps. A whole-copy swap is a real swap that reads 0.5-1.1 A. Use a threshold of ~5 A, and say so.

## MINOR

### MINOR-1. Checkpoint selection is where the 7ndt flip matters, and the docs call it "noise, not a trend"
UPSTREAM_PATCHES section 8, 2nd bullet. The four training arms keep `best-*.ckpt` by `val/lddt` (train.py patch, the second ModelCheckpoint). Round 4's "best" was +0.0016 over the control, "the largest of twelve noisy reads" (RESULT_round4.md:49). Which truth wins changes the set of 7ndt pairs that count, and that set enters `val/lddt` at 1/30 weight. The size is unmeasured; a shift of 0.03-0.05 in 7ndt's lddt_pp moves the mean by 0.001-0.0017, the same size as the selection margin. Fix (docs only): say that best-checkpoint picks within ~0.002 are not to be trusted until they are re-scored from the per-structure rows without 7ndt (the rows carry lddt_pp per sample).

### MINOR-2. The docstring distance is wrong
src/peptide_rmsd.py:43-44 says "a second complete pMHC copy 85-120 A away". The centroid distances, recomputed with numpy, are A1-F1 102.4, B1-G1 119.5 and C1-H1 67.0 A, which match the UPSTREAM_PATCHES table. Fix: "67-120 A".

### MINOR-3. groove_start is non-zero for 25 of 99 records, not 6
Real data (numpy, first heavy-chain residue with is_present): 25/99 heavy chains start at index 1-2. Examples: 6zkx 2, 7ndt 2, 7n9j 2, 7wj3 2, 7dzn 2 and 16 more at 1. 7ejm/7ejn are at 7, 7phr at 12, 7rtd/7rtr/7tlt at 24. For the 1-2 cases the 180-token window now runs one or two residues into the alpha3 linker. Before, it was one or two residues short at the N-terminus. The effect on the superposition is negligible, and `groove_start` is recorded. The commit message and docstring ("a few records") undersell how many rows changed definition. Nine of these are subset30 records, but no rows exist from the old code (it never ran), so comparability is not affected. No fix needed; note it.

### MINOR-4. The test cannot tell the stash mask from the batch mask
tests/test_peptide_rmsd.py:51 passes the same `mask` as the batch's `atom_resolved_mask` and as both of the stash's `true_mask`, and `frame_resolved_mask` is all True (:49). If the code used `stash["true_mask"]` for `ca()`, or ignored `frame_ok` in `hc_ok`, the test would still pass. It does discriminate the main fix: on the old code, pep_rmsd sample 0 would be ~60 A and `groove_start` would be missing, so the test fails. The expected values are correct and not tautological: groove_start 5, n_groove_ca 179 (tokens 5-184 less token 10), and pep_rmsd 1.0. Fix: in the swapped sample, make the stash mask differ (e.g. mark the swapped peptide's CA unresolved in `true_mask[0]`) and assert n_pep_ca == 9. Set one heavy-chain token's frame_resolved_mask to False and assert groove_start skips it.

### MINOR-5. Overstatement
RESULT_round4.md section 4 note: "So the excursion is a metric artefact, not a model failure". UPSTREAM_PATCHES section 8: "almost certainly". The MAJOR-1 arithmetic makes the hypothesis much stronger than the docs argue, but nothing has observed it per structure yet. Phrase it as "fits all 16 validations quantitatively (MAJOR-1); the eval's 7ndt rows confirm it".

## Verified OK (evidence)

1. **`batch["coords"][b, 0]` is the unswapped truth, indexed like the prediction, when the callback runs.**
   * featurizer.py:737-739, 803, 814 build `coords` from `data.structure.atoms["coords"]` token by token. The atom table is the same one `frames_idx` (:774-776, `+ atom_idx`) and `atom_to_token` index, so `frames[t, 1]` is that token's CA in `coords`.
   * :840-842 centre it. That is a translation only, harmless under Kabsch.
   * :846 random augmentation touches `ref_pos` only.
   * :866 pads `coords` at the END (dim 1), so indices stay valid. The shape is (1, N, 3) per example; collate (training.py:125-165) stacks to (B, 1, N, 3), so `[b, 0]` is right. B = 1 here.
   * `sample_atom_coords` has the same N (it is sampled over `atom_pad_mask`).
   * `crop_validation: false` in both configs (unfrozen.yaml:169, eval_val99.yaml:179), so there is no crop and no max-token/max-atom padding (featurizer is only padded when crop_validation).
   * Nothing in validation modifies `batch["coords"]` or `atom_resolved_mask`:
     - the only write, `feats["coords"] = ...` (diffusion.py:717), is in the training forward, reached only under `self.training` (model.py:325);
     - `get_true_coordinates` clones (`all_coords.clone()`, symmetry.py:269) or builds new tensors (`repeat_interleave`, model.py:416);
     - weighted_rigid_align and weighted_minimum_rmsd do no in-place work on their inputs (grep);
     - the other callbacks (val_progress, mem_probe) do not touch the batch.
   * PL 2.4 passes the device-moved batch to `on_validation_batch_end` (evaluation_loop.py:370, 409-410).
   * `precision: 32` in both configs, so there is no half-precision input cast.
2. **Same notion of "resolved".** The featurizer's `atom_resolved_mask` = `atom_data["is_present"]` (:815). The stash's mask comes from `structure.atoms["is_present"]` via `crop_to_all_atom_map` (symmetry.py:424, 453-458), in the same token-major atom order. With no swap the two are identical, and the stash truth is a rigid transform of `coords`, so boltz_truth_dev is ~0 (float error).
3. **Atom-level symmetries do not touch CA, so they cannot create a false boltz_truth_dev.** The amino-acid symmetry swaps are side-chain atoms. Non-standard residues (the 14 peptides: SEP, QM8, QMB, DPN, E7P, OSE, MYR) are per-atom tokens with frame mask False (featurizer.py:743-745), so they are excluded from `a`.
4. **Twins are real, and the swap levels are as stated.** 7ndt: F1/G1/H1 have the same entity, atom count and atom-name arrays as A1/B1/C1. Centroid distances are 102.4 / 119.5 / 67.0 A (UPSTREAM_PATCHES table correct). A b2m swap gives 48.5 A all-atom ("~49" correct). In val99 only 7ndt and 7bh8 have masked twins. Every kept chain in all 99 records is a protein of a distinct entity, so "CA loses nothing" holds.
5. **7ndt is in subset30.** The split file has 582 lines, but only 30 match manifest ids, and 7ndt is one of them. It is in val99 (validation_ids.txt:29).
6. **"Training loss is not affected" is true.**
   * training_step (model.py:432-520) calls get_true_coordinates only under `self.confidence_prediction`, which is false (unfrozen.yaml:231, eval:241).
   * The diffusion loss uses `feats["coords"]` with weighted_rigid_align and no chain symmetry (diffusion.py:703-830).
   * `return_train_symmetries: false`.
7. **The 14/99 non-standard peptides and the 4-20 residue range are right.** The peptide token count is at most 28 (<= MAX_PEPTIDE). No heavy chain has a non-standard residue, so the groove window is 180 real residues.
8. **None-safety and crash paths.**
   * `boltz_truth_dev` None is only `round`ed when not None (:206). It never reaches min/sum/_median, which read only pep_rmsd, never None.
   * The smoke test reads only pep_rmsd (ls6_ddp_smoke.slurm:86-87).
   * `int(hc_ok[0])`, `torch.nonzero` and the slicing work on CUDA. All index tensors and stash tensors share the batch's device; `frames` is int64.
   * 8d5q (180-residue HC, start 1) gives a 179-token window and no error. The first resolved HC residue in any of the 99 records is at most index 24, so the far-in alpha1 case does not occur.
   * on_validation_batch_end is fully wrapped. on_validation_epoch_end is not, but it is only reached with a non-empty `_all` of floats.
9. **The test discriminates the fix.** See MINOR-4: it would fail on the old code, and its expected values are correct.
10. **The tree is clean and HEAD 0c472cf is on origin/master.** 822e470 changes only src/ and tests/, so the boltz-src patch and LS6's patched checkout are unaffected.
11. **The four-mode arithmetic (MAJOR-1)** comes from RESULT_round4.md:14-29 via sums: rmsd x 90 = 274-277 / 235.5 / 226.7 / 156.3; best x 30 = 89-90 / 50.5 / 41.4 / 49.7.

## UNVERIFIED (and what settles it)
- **Whether LS6's checkout is actually at 822e470 before the 19:50 eval.** The doc says it was pulled at ~18:20. I cannot look (no bridge). If it is not, the eval scores 7ndt and 7bh8 against the swapped truth. Settle with `git -C $WORK/mhc1-boltz rev-parse HEAD` on LS6, or look for `groove_start` in the first eval row.
- **That peptide_rmsd runs at all on real Boltz batches.** It never has: the smoke test now runs after the eval. Any exception becomes a row with `reason: "scoring failed: ..."`, not a crash, but the eval could then spend its slot with null peptide rows. Settle by checking the first rows of runs/eval_val99/pretrained/val_peptide_rank0.jsonl a few minutes after 19:50, before the step153 pass.
- **The 0.9246 vs 0.9219 lDDT reproduction.** It is not re-run here: the pair matrices for 3,156 atoms are too large for the free RAM. The swap levels do not depend on it.
- **The queue claims** (age factor weight, Oct 11 18:46 estimate, nice 200) and MHC-Fine's paper details (pLDDT pick of 5, 8-11-mers) are not checkable offline.
