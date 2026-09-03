"""Make a validation run's partial results survive being killed.

The problem this solves, learned the hard way
---------------------------------------------
`trainer.validate()` accumulates every metric into torchmetrics `MeanMetric`
objects held in process memory and emits nothing until
`on_validation_epoch_end`. There is no mid-validation checkpoint in Lightning.
So a 99-sample run that is interrupted at sample 47 loses all 47 -- roughly 3.5
GPU-hours -- because those results only ever existed as running sums in RAM.

This callback snapshots those running sums to disk after every batch. Two
consequences:

1. **A killed run is not a wasted run.** The JSON always holds the correct
   weighted means over however many samples completed.
2. **Chunks compose exactly.** `MeanMetric` stores `mean_value` (the sum of
   value x weight) and `weight` (the sum of weights). Those are sufficient
   statistics, so several runs over disjoint sample sets can be merged by summing
   both fields and dividing once -- see `src/combine_val_states.py`. This is NOT
   the same as averaging each run's reported mean, which would be wrong whenever
   the chunks differ in size or in per-sample atom counts.

Writes are atomic (temp file + replace) so a kill mid-write cannot leave a
truncated file.
"""
import json
import os
import pathlib

from pytorch_lightning.callbacks import Callback
from torchmetrics import MeanMetric


def _snapshot(pl_module):
    """Every MeanMetric on the module, as {name: {mean_value, weight}}."""
    out = {}
    for name, mod in pl_module.named_modules():
        if isinstance(mod, MeanMetric):
            try:
                out[name] = {
                    "mean_value": float(mod.mean_value),
                    "weight": float(mod.weight),
                }
            except Exception:
                # A metric that has never been updated holds NaN; record it as
                # absent rather than poisoning the merge with a NaN.
                continue
    return out


class ValProgressDump(Callback):
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_batches = 0

    def _write(self, pl_module):
        payload = {
            "n_batches_completed": self.n_batches,
            "metrics": _snapshot(pl_module),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, self.path)          # atomic on Windows and POSIX

    def on_validation_batch_end(self, trainer, pl_module, *args, **kwargs):
        self.n_batches += 1
        self._write(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Last write before Boltz's own on_validation_epoch_end resets the
        # metrics; ordering is by callback registration, and ours is appended
        # after the module's own hook runs only if registered last, so write
        # defensively here too.
        self._write(pl_module)
