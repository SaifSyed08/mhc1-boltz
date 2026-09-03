# Changes made to the vendored Boltz-1 v1.0.0 checkout

`boltz-src/` is a clean checkout of
[jwohlwend/boltz](https://github.com/jwohlwend/boltz) at tag `v1.0.0`
(commit `34cf560`). Two defects there stop the documented training entry point
from running at all. Both are recorded here so the delta from upstream is never
implicit.

Only the first is an edit to vendored source; it is kept as
`patches/boltz-v1.0.0-training-path.patch` and can be reapplied to a fresh
checkout with `git -C boltz-src apply ../patches/boltz-v1.0.0-training-path.patch`.
The second is worked around in our own config and requires no source change.

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
