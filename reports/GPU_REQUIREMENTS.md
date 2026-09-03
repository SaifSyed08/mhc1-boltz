# GPU requirements for the Boltz-1 MHC-I fine-tune

The brief says: *"While GPU access is being arranged, focus on all preprocessing
and validation steps that can be completed without a GPU. If a particular step
requires GPU resources, document it and continue with the remaining tasks."*

This is that document. Every number below is measured from the actual checkpoint
and read out of the actual model code, not estimated from the paper.

## The hardware on hand

```
NVIDIA GeForce RTX 4060 Laptop GPU
8.59 GB total (8188 MiB reported by nvidia-smi)
compute capability 8.9 (Ada), driver 592.82
torch 2.11.0+cu128
```

Usable is roughly 8.1 GB: the CUDA context itself takes ~0.4-0.6 GB before a
single tensor is allocated.

## What is actually in the checkpoint

`data/assets/boltz1_conf.ckpt` is 3.60 GB on disk and holds 6,611 tensors:

| module | params |
|---|---|
| `structure_module` (diffusion) | 301.4 M |
| `confidence_module` | 152.8 M |
| `pairformer_module` | 147.4 M |
| `msa_module` | 3.2 M |
| `input_embedder` | 1.1 M |
| everything else (`s_init`, `z_init_*`, `rel_pos`, `s_recycle`) | 0.5 M |
| **total** | **606.4 M** |

`configs/mhc1_finetune.yaml` sets `confidence_prediction: false`, and
`model.py:231` only constructs `confidence_module` when that flag is true. So the
confidence weights are loaded from disk but never instantiated:

**trainable parameters = 606.4 M − 152.8 M = 453.6 M.**

## Fine-tuning does not fit, and it is not close

`precision: 32` with Adam and `ema: true`. Per trainable parameter that is four
fp32 copies plus the gradient:

| tensor | bytes/param | total |
|---|---|---|
| weights | 4 | 1.81 GB |
| gradients | 4 | 1.81 GB |
| Adam `exp_avg` | 4 | 1.81 GB |
| Adam `exp_avg_sq` | 4 | 1.81 GB |
| EMA `shadow_params` (`utils.py:134`) | 4 | 1.81 GB |
| **subtotal** | **20** | **9.06 GB** |

9.06 GB of persistent state against ~8.1 GB usable — over budget **before a
single activation is allocated**, and before the CUDA context, the 512-token
crop, or the diffusion batch (`diffusion_multiplicity: 16`,
`diffusion_samples: 2`).

Note what does *not* rescue this:

* **`activation_checkpointing: true`** is already on for the msa, pairformer and
  score modules. It trades compute for activation memory and does nothing about
  the 9.06 GB above, which is all parameter-shaped state.
* **`batch_size: 1`** is already the minimum. The overflow is not batch-related.
* **bf16 mixed precision** does not help either, which is the counter-intuitive
  part: standard AMP keeps fp32 master weights *and* fp32 optimizer states and
  merely casts activations. It cuts activation memory, not the 9.06 GB.

### What would actually fit

Roughly in order of how much they distort the recipe:

1. **Use a bigger GPU.** A 40 GB A100 or 80 GB H100 runs the config as written.
   This is the honest answer and the one the config is set up for.
2. **8-bit Adam** (`bitsandbytes`): optimizer states drop from 3.63 GB to
   0.91 GB → 6.34 GB subtotal. Fits, but leaves very little for activations, and
   changes the optimizer.
3. **`ema: false`**: saves 1.81 GB. Changes the training recipe — Boltz-1 was
   trained with EMA and the reported weights are EMA weights.
4. **Freeze the trunk** (`msa_module` + `pairformer_module`, 150.6 M) and
   fine-tune only `structure_module`: trainable drops to 303 M → 6.06 GB. This is
   a defensible fine-tuning strategy in its own right, not just a memory hack,
   since the pMHC-I task is mostly about the geometry of a peptide in a groove.

None of these are chosen here. Per the brief, the fine-tuning *strategy* is to be
discussed after the data is validated, so the config is left faithful to the
upstream recipe and the constraint is documented instead.

## The validation baseline does fit — after one fix

`validation_only: true` builds no optimizer and holds no gradients, so the
persistent cost is just the 1.81 GB of weights. But EMA still bites, and in a way
that is pure waste here.

`model.py:1246 prepare_eval()` runs on `on_validation_start` and does three
things:

```python
if self.use_ema and self.ema is None:
    self.ema = ExponentialMovingAverage(parameters=self.parameters(), ...)
self.ema.store(self.parameters())      # -> collected_params, a second clone
self.ema.copy_to(self.parameters())
```

`ExponentialMovingAverage.__init__` does
`shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]`
(`modules/utils.py:134`), and `store()` clones again. That is **two extra full
copies, 3.63 GB**, taking the baseline from 1.81 GB to 5.44 GB of persistent
state.

And it buys nothing. The checkpoint's top-level keys are:

```
epoch, global_step, pytorch-lightning_version, state_dict, loops,
callbacks, optimizer_states, lr_schedulers, hparams_name, hyper_parameters
```

There is **no `ema` key**, so `on_load_checkpoint` (`model.py:1217`, guarded by
`if self.use_ema and "ema" in checkpoint`) never fires. `self.ema` is therefore
`None` at validation start and gets built *from the current weights* — meaning
`shadow_params` is a copy of the weights and `copy_to` writes the weights back
onto themselves.

**So `ema: false` in the baseline config is numerically identical and saves
3.63 GB.** That is why `src/make_baseline_config.py` overrides it, and why it
overrides nothing under `model.validation_args` — those would change the measured
number, this cannot.

### Measured, and why the two numbers differ

| baseline run | steady-state `nvidia-smi` | outcome |
|---|---|---|
| `ema: true` | 7921 MiB / 8188 | never completed one batch in >10 min |
| `ema: false` | 5954 MiB / 8188 | runs |

The observed drop is 1.92 GB, not the 3.63 GB the two-clone arithmetic predicts.
That is not a contradiction — it is the interesting part. 7921 MiB is essentially
the card's ceiling once the desktop's ~430 MiB is accounted for, so that figure is
a **cap, not a measure of demand**. True demand with EMA on is about
5954 + 3630 ≈ 9.6 GB, which does not fit.

What that looks like in practice is worth recognising, because it is not what you
expect: PyTorch's caching allocator did not raise `CUDA out of memory`. It kept
freeing and re-fetching blocks to stay under the limit, so the run simply
*degraded* — GPU pinned at 100% utilisation, making almost no forward progress,
and a single validation sample never finished. **A job that is silently 50x too
slow is a plausible symptom of being just barely over VRAM, not only of being
compute-bound.** A clean OOM is the friendlier failure.


## Measured: fine-tuning was attempted and does not fit

The arithmetic above predicted this, but predictions are cheap. Three
configurations were actually run on the 4060 (2026-09-03), each stripping more
memory than the last. All three died with `CUDA error: out of memory`.

| # | configuration | trainable | result |
|---|---|---|---|
| 1 | stock recipe | 453.6M | not attempted -- 9.06 GB of state alone exceeds the card |
| 2 | `ema: false` + `offload_to_cpu: true` | 453.6M | **OOM on the first training step**, peak 7637 MiB |
| 3 | as #2 + trunk frozen (`msa_module`, `pairformer_module`) | 281.6M | **OOM**, in `triangular_attention/attention.py:127` |

Configuration 3 is the informative one. Freezing the trunk removes its gradients
*and* the activations autograd would retain for its backward pass, cutting
trainable parameters by a third. It still failed, and the failure is in the
**forward** pass of triangular attention.

That is the crux: triangular attention over a 512-token crop is expensive whether
or not it is being trained. Validation survives it because no backward graph is
retained; training does not, because gradients and Adam moments for the structure
module (281.6M x 12 bytes = 3.4 GB) occupy the space the trunk forward needs.

So the remaining levers are not memory tricks -- they are changes to the training
objective itself: `diffusion_multiplicity: 16` (the structure module processes 16
noisy copies per sample), `max_tokens: 512`, or the crop size. Turning those down
far enough to fit would no longer be fine-tuning Boltz-1 under its own recipe, and
reporting the result as such would be misleading.

**Conclusion: fine-tuning needs a bigger GPU.** A 40 GB A100 runs
`configs/mhc1_finetune.yaml` as written. This is now an empirical result rather
than an estimate, which is a stronger thing to bring to a supervisor.

`configs/mhc1_finetune_8gb.yaml` and the `freeze_trunk` option in `train.py` are
kept because they are correct and will be useful on real hardware -- freezing the
trunk is a reasonable fine-tuning strategy for pMHC-I on its merits, not only as a
memory workaround.

## Upstream bug worth knowing about

Boltz v1.0.0 made `steering_args` a required positional argument of
`Boltz1.__init__` (`model/model.py:59`) but never updated
`scripts/train/configs/structure.yaml` or `full.yaml`. Instantiating either as
shipped fails with:

```
hydra.errors.InstantiationException: Error in call to target 'boltz.model.model.Boltz1':
TypeError("Boltz1.__init__() missing 1 required positional argument: 'steering_args'")
```

`configs/mhc1_finetune.yaml` adds the missing block, with steering disabled to
match `boltz predict --no_potentials`. This is also a memory decision:
`model.py:338` calls `structure_module.sample()` whenever the module is not in
training mode, and `fk_steering` with `num_particles: 3` would triple the
diffusion batch during validation.

## Environment note

Two virtualenvs, deliberately:

* `.venv` — the milestone-1 pipeline. Python 3.13, numpy 2.5.2, CPU-only torch.
  Enough to run Boltz's tokenizer/cropper for validation check V11.
* `.venv-gpu` — the training stack. torch 2.11.0+cu128, pytorch-lightning 2.4.0,
  hydra 1.3.2, and `boltz` installed with `--no-deps`.

They are separate because `boltz`'s `pyproject.toml` pins `numpy==1.26.3`, which
has no Python 3.13 wheel; installing it in place would either fail to build or
downgrade numpy under the validated milestone-1 pipeline. Three of boltz's pins
are dropped in `.venv-gpu` and none are on the training path:

* `dm-tree==0.1.8` — never imported anywhere in `src/boltz/`.
* `biopython==1.84` — only `data/parse/fasta.py`; no py3.13 wheel.
* `trifast>=0.1.11` — a soft import guarded by
  `importlib.util.find_spec` (`triangular_attention/primitives.py:46`), and
  `use_trifast` defaults to `False` on the training path. It needs Triton, which
  has no Windows build.

`fairscale` is *not* droppable — `modules/trunk.py` and `modules/transformers.py`
import `checkpoint_wrapper` from it, which is what `activation_checkpointing`
uses.
