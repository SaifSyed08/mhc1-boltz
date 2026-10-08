# Changes made to the vendored Boltz-1 v1.0.0 checkout

`boltz-src/` is a clean checkout of
[jwohlwend/boltz](https://github.com/jwohlwend/boltz) at tag `v1.0.0`
(commit `34cf560`). Several defects there stop the documented training entry
point from running at all, and one design choice makes fine-tuning OOM on any
card short of an A100. All are recorded here so the delta from upstream is never
implicit.

The source edits are kept as `patches/boltz-v1.0.0-training-path.patch` and
reapply to a fresh checkout with
`git -C boltz-src apply ../patches/boltz-v1.0.0-training-path.patch`. Item 2 is
worked around in our own config and needs no source change.

The patch touches **six** files: `scripts/train/train.py`,
`data/feature/featurizer.py`, `model/model.py`, `model/modules/diffusion.py`,
`model/modules/trunk.py` and `model/optim/scheduler.py`.

Line references below are against **pristine upstream**, which is right for
describing an upstream defect but means a few are off by a handful of lines in
the patched tree you actually have checked out.

---

## 1. `UnboundLocalError` on every training and validation sample

**File:** `src/boltz/data/feature/featurizer.py`, in `BoltzFeaturizer.process`.

As shipped:

```python
residue_constraint_features = {}
if compute_constraint_features:
    residue_constraint_features = process_residue_constraint_features(data)
    chain_constraint_features = process_chain_feature_constraints(data)

return {
    ...
    **residue_constraint_features,
    **chain_constraint_features,      # only bound inside the `if`
}
```

`residue_constraint_features` is initialised to `{}` before the branch;
`chain_constraint_features` is not, but both are splatted unconditionally into
the returned dict.

`compute_constraint_features` defaults to `False`
(`featurizer.py:1150`) and is passed as `True` from exactly one place —
`src/boltz/data/module/inference.py:202`. The training and validation paths
(`src/boltz/data/module/training.py:282` and `:426`) do not pass it at all.

So on the training path the branch never runs and the return statement raises:

```
cannot access local variable 'chain_constraint_features'
where it is not associated with a value
```

for **every sample**. Boltz v1.0.0 cannot produce a single training or
validation batch as shipped. The inference path is unaffected, which is
presumably why it survived release.

**Fix:** initialise `chain_constraint_features = {}` alongside
`residue_constraint_features`.

### Why this took a while to see

The error is raised inside `TrainingDataset.__getitem__`, which catches any
exception, logs `Featurizer failed on <id> ... Skipping.`, and retries with
`self.__getitem__(0)`. Since record 0 fails deterministically for the same
reason, that is unbounded recursion.

How it terminates depends on where the stack runs out first, which is why the
same bug produced two different-looking crashes:

* once as `RecursionError: maximum recursion depth exceeded`, after ~980 frames
  of Lightning traceback;
* once as a bare **`Windows fatal exception: access violation`** — process exit
  code 139, no Python traceback at all, because the C stack was exhausted before
  CPython's recursion limit tripped.

Neither mentions `chain_constraint_features`. Running the data pipeline on its
own with `faulthandler.enable()` and `num_workers=0` is what surfaced the real
message — with workers on, the crash happens in a spawned process and the
message is lost entirely.

---

## 2. Upstream training configs cannot be instantiated

**Files:** `scripts/train/configs/structure.yaml` and `full.yaml`.

`Boltz1.__init__` takes `steering_args: dict[str, Any]` as a **required**
argument (`src/boltz/model/model.py:59`), but neither shipped training config
defines it. `hydra.utils.instantiate` therefore fails before training starts:

```
hydra.errors.InstantiationException: Error in call to target 'boltz.model.model.Boltz1':
TypeError("Boltz1.__init__() missing 1 required positional argument: 'steering_args'")
```

Not patched upstream — `configs/mhc1_finetune.yaml` simply defines the block,
which is where it belongs anyway. Values mirror `boltz predict --no_potentials`
(`fk_steering: false`, `guidance_update: false`), since steering is an
inference-time search procedure rather than part of the model, and enabling it
during validation would both change the metric and triple the diffusion batch
via `num_particles: 3`.

---

## 3. `AlphaFoldLRScheduler` passes `verbose=` to a torch that removed it

`optim/scheduler.py:75` calls

```python
super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)
```

torch deprecated `LRScheduler`'s `verbose` argument in 2.4 and has since removed
it, so on a modern torch this is

```
TypeError: LRScheduler.__init__() got an unexpected keyword argument 'verbose'
```

Patched to try the upstream call and fall back without `verbose`. `self.verbose`
is still assigned a few lines above, so `state_dict()` is unchanged.

This fires only on the **training** path: `configure_optimizers()` is never
called under `validation_only`, which is why the baseline reproduced fine and
this only appeared on the first fine-tuning attempt.

---

## 4. `NameError: center_random_augmentation` in the training forward

`modules/diffusion.py:716` calls `center_random_augmentation`, but the module
never imports it. The function is right there in `modules/utils.py:61`; only the
import line is missing.

Inference never reaches line 716, which is why this shipped. It kills every
training step immediately.

One added import, no behaviour change.

---

## 5. Triangular-attention chunking is disabled during training

Not a bug upstream -- a deliberate speed/memory trade -- but it is the single
thing that makes fine-tuning not fit.

`MSAModule.forward` and `PairformerModule.forward` both gate chunking on
`not self.training`:

```python
if not self.training:
    if z.shape[1] > const.chunk_size_threshold:   # 384
        chunk_size_tri_attn = 128
    ...
else:
    chunk_size_tri_attn = None                    # materialise everything
```

So training allocates the entire triangular-attention score tensor in one go. At
our 512-token crop with 4 pairwise heads (`trunk.py:129` `pairwise_num_heads`)
that is

```
[1, 512, 4, 512, 512] fp32 = 536,870,912 elements x 4 B = exactly 2.00 GiB
```

and each call materialises **two** of them -- `a = matmul(q, k)`
(`primitives.py:219`) and the softmax output (`primitives.py:203`).

Every `PairformerLayer` calls `tri_att_start` *and* `tri_att_end`
(`trunk.py:642, :650`), and every `MSALayer` does the same (`trunk.py:413, :421`),
so a single trunk pass makes **96** such calls in the pairformer and **8** in the
MSA module -- sequentially, not concurrently -- and the trunk runs
`recycling_steps + 1` times.

Two corrections to what an earlier version of this document claimed:

* **It is the MSA module that actually dies, not the pairformer.** All four fatal
  tracebacks in `reports/ft_*.log` go through `MSAModule.forward` → `MSALayer`,
  and the pairformer never appears. The MSA module runs *first* in the trunk
  (`model.py:304`) and carries `m [1, 2048, 512, 64]` on top of `z`, so it
  exhausts the card before the pairformer is reached.
* **The logs do not all name 2.00 GiB.** They name 730 MiB (`ft_final`),
  1024 MiB (`ft_chunk`, an outer-product-mean tensor at
  `outer_product_mean.py:92` -- a different tensor in a different layer),
  2.00 GiB (`ft_msa256`, `ft_msa256_freeze`) and 864 MiB (`ft_mult`, inside the
  *chunked* path). Chunking helps; it is not sufficient on an 8 GB card.

One limit worth knowing: `chunk_layer` is not itself gradient-checkpointed, so
when triangular attention runs with grad enabled -- during fairscale's backward
recomputation -- all chunks' intermediates are live at once and chunking saves
nothing. It helps the forward, and it helps unconditionally only when the trunk
is fully frozen.

Chunking is **mathematically identical** -- `chunk_layer`
(`triangular_attention/utils.py:258`) partitions the row dimension and each slice
does full-length attention, an exact partition rather than an online-softmax
approximation. It is not guaranteed *bitwise* identical: 128 rows instead of 512
can select a different cuBLAS kernel and reduction order. The patch adds an opt-in:

```python
chunk_in_training = os.environ.get("BOLTZ_CHUNK_IN_TRAINING") == "1"
if (not self.training) or chunk_in_training:
```

Unset, behaviour is byte-identical to upstream. `BOLTZ_CHUNK_IN_TRAINING=1` is
set on every training subprocess launched by `notebooks/mhc1_boltz_t4.ipynb`.

Upstream would presumably take a config flag rather than an environment
variable; the env var keeps the diff to two lines per call site and avoids
threading a new argument through `Boltz1.__init__`.

---

## 6. The OOM handler discards the only useful information

`model.py` wraps `structure_module.compute_loss` in `try/except Exception` and
prints `f"Skipping batch {batch_idx} due to error: {e}"`. For a CUDA OOM that is
a requested size and a total, with no location.

That matters more than it looks, because of what the `try` does **not** cover.
`out = self(...)` -- the input embedder, the MSA module, all 48 pairformer
blocks, and the diffusion score model -- is outside it, as is `distogram_loss`.
So any batch that prints "Skipping batch ... out of memory" is a batch whose
**forward pass completed**. The OOM is in the diffusion loss, every time.

Four memory diagnoses on this project were made by inferring a location from that
one-line string. Three were wrong, and one of them inverted this exact fact,
describing a `compute_loss` failure as having "died early in the trunk forward".

The patch prints the traceback, the effective `diffusion_multiplicity`, the atom
count, and the resulting `[mult, n_atoms, n_atoms]` tensor size. On a path that
is already failing, that costs nothing.

## 7. Validation cannot be traced to a structure, and never scores the peptide

Not a bug, a missing capability. `validation_step` feeds every result into
`MeanMetric`s that are summed over structures before anything reaches disk, so
no number can be traced back to the complex that produced it. And none of them
scores the peptide alone: `lddt_pp` pools every inter-chain contact, most of
them heavy chain-β2m. MHC-Fine (Glukhov et al., Biophys J 2024), the
AlphaFold fine-tune this project parallels, is judged on exactly that peptide
number: superpose on the MHC, RMSD over peptide Cα.

The patch adds two things to `model.py`: `self._val_stash = None` at the top of
`validation_step` (so a batch skipped on OOM cannot leave stale predictions
behind), and at the end a dict of references (no copies) to the predicted
coordinates, the symmetry-corrected true coordinates and mask, and Boltz's own
per-sample RMSD and lDDT. `src/peptide_rmsd.py` (registered in `train.py`) reads
it in `on_validation_batch_end`, clears it, and writes one row per structure.
It scores against the batch's own `coords`, not the stash's symmetry-corrected
truth, because of §8; the stash's truth is used only to report how far it is
from the real one (`boltz_truth_dev`).

## 8. Symmetry correction can swap a kept chain for a copy the dataset masked out

**Not patched**: fixing it changes `val/lddt` and `val/rmsd`, so Rounds 1–4 would no
longer compare. That decision belongs to Saif and Ernest. Found by the 2026-10-09 audit
(`reports/AUDIT_2026-10-09_round5.md`).

`get_chain_symmetries` (`data/feature/symmetry.py`) builds the chain swaps that
validation's symmetry correction tries. For every chain in the crop it lists
*every* chain of the full structure with the same `entity_id` and atom count. It
checks `chain_in_crop` for the chain being replaced but not for its replacement.
`load_input` keeps the structure's `mask` and the tokenizer skips masked chains,
so chains our pipeline masked out (`step6_write.py` masks unrelated chains rather
than deleting them) are still offered as replacements. `minimum_lddt_symmetry_coords`
then keeps whichever truth gives the highest all-atom lDDT against the prediction.
lDDT only scores pairs within 15 Å *in the truth*, so swapping in a far-away copy
removes that chain's interface pairs from the score. Interfaces are predicted worse
than chain interiors, so the swap can win.

`7ndt` (subset30, so in every training-time validation) is a TCR–pMHC crystal with
two complexes in the asymmetric unit:

| kept chain | masked twin | centroid distance |
|---|---|---|
| A1 heavy chain (entity 0, 2,249 atoms) | F1 | 102 Å |
| B1 β2m (entity 1, 836 atoms) | G1 | 120 Å |
| C1 peptide (entity 2, 71 atoms) | H1 | 67 Å |

The audit reproduced the swap on 7ndt's real coordinates with numpy. With the
prediction set to the truth with β2m and the peptide nudged 3–5° / ~1 Å, the truth
with the peptide swapped for H1 scores lDDT 0.9246, above the real truth at 0.9219,
so the swap is selected. Swapping β2m gives an all-atom RMSD of ~49 Å. `7bh8` (val99
only) also has twins, packed against the kept chains.

Consequences, all pre-existing in Boltz's own metrics:
* 7ndt's `val/rmsd` and `val/best_rmsd` are measured against a disassembled complex
  whenever a swap wins: ~49 Å with β2m or the heavy chain swapped, ~10 Å with the
  peptide swapped. These levels fit all four modes of the round-4 table
  (`RESULT_round4.md` §4) to within ~0.04 Å, leaving the other 29 structures at
  `rmsd` 1.44–1.50 Å and `best_rmsd` 1.37–1.43 Å. So 7ndt is in every subset30
  RMSD recorded, the baseline included: ~1.6 Å of both in the usual mode, less in
  the others. This is the leading explanation; 7ndt's
  per-structure rows (`boltz_truth_dev` > ~5 Å) will confirm or refute it.
* 7ndt's share of `val/lddt` (`lddt_pp` included) is computed without the interface
  the model got most wrong, and which truth wins flips between validations. Best
  checkpoints are picked on `val/lddt` margins of ~0.002, about the size of this
  term, so re-score close picks from the per-structure rows without 7ndt. That is
  one structure in 30: extra noise, not a trend.

**Confirmed** by the 99-record eval (`reports/ls6/RESULT_eval99.md`): on 7ndt,
Boltz's RMSD is ~49.2 Å with `boltz_truth_dev` 48.9 Å in every sample, while the
peptide is within 0.4–0.7 Å of the real complex. 7bh8 flips between a 9.85 Å swap
and the correct truth.

Root-cause fix, if wanted: also require `chain_in_crop[j]` for the replacement (or
drop masked chains before validation). The per-structure rows make the effect
measurable without the fix: `boltz_truth_dev` is above ~5 Å on exactly the samples
where Boltz scored a harmful swap. A whole-copy swap of 7ndt or 7bh8 reads only
0.5–1.1 Å.

---

## Not patched, but worth knowing

* **`expandable_segments` is a no-op on Windows.** Setting
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` prints
  `expandable_segments not supported on this platform` and changes nothing. On
  an 8 GB card that allocator would have helped with fragmentation; on Windows
  it is simply unavailable.
* **`num_workers > 0` needs the `__main__` guard on Windows.** Windows has no
  `fork`, so DataLoader workers are spawned and re-import the entry module.
  `scripts/train/train.py` guards correctly; any ad-hoc script that builds the
  data module does not, and fails with
  `An attempt has been made to start a new process before the current process
  has finished its bootstrapping phase`. `debug: true` sidesteps this by forcing
  `num_workers = 0`.
