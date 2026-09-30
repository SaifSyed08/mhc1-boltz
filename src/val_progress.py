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

Under DDP
---------
Lightning shards the validation set across ranks (DistributedSampler), so each
rank's MeanMetric holds only ITS samples: with 3 GPUs and 30 samples, rank 0
sees 10. Writing rank 0's local state would silently report a 10-sample mean as
if it were the 30-sample metric. So at epoch end every rank sums its state into
the others (a collective -- all ranks must take part, which is why the sync runs
before the rank-zero check) and rank 0 writes the total. The per-batch writes in
between stay rank-0-local, because a collective per batch would stall every
rank on the slowest; the `scope` field says which one a file holds.
"""
import json
import os
import pathlib

import torch
from pytorch_lightning.callbacks import Callback
from torchmetrics import MeanMetric


def _is_rank_zero() -> bool:
    """Under DDP there is one process per GPU. Without this guard all three
    ranks write the same telemetry file and interleave into nonsense."""
    return (int(os.environ.get("LOCAL_RANK", 0)) == 0
            and int(os.environ.get("NODE_RANK", 0)) == 0)


def _metrics(pl_module):
    """Every MeanMetric on the module, in a fixed order (named_modules is
    deterministic, so every rank walks the same list -- required for the sync)."""
    return [(name, mod) for name, mod in pl_module.named_modules()
            if isinstance(mod, MeanMetric)]


def _read(mod):
    return {"mean_value": float(mod.mean_value), "weight": float(mod.weight)}


def _snapshot(pl_module):
    """This rank's own state, as {name: {mean_value, weight}}. No communication."""
    return {name: _read(mod) for name, mod in _metrics(pl_module)}


def _snapshot_all_ranks(pl_module):
    """Every rank's state summed. COLLECTIVE: call on all ranks or none.

    sync_context sums the states across ranks (MeanMetric reduces both fields
    with "sum") and restores the local ones on exit, so Boltz's own compute()
    afterwards still sees local state and does not double-count. Outside DDP it
    is a no-op and this equals _snapshot.
    """
    out = {}
    for name, mod in _metrics(pl_module):
        with mod.sync_context(should_sync=True, should_unsync=True):
            out[name] = _read(mod)
    return out


class ValProgressDump(Callback):
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_batches = 0

    def _write(self, payload):
        if not _is_rank_zero():
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, self.path)          # atomic on Windows and POSIX

    def on_validation_batch_end(self, trainer, pl_module, *args, **kwargs):
        self.n_batches += 1
        if _is_rank_zero():
            self._write({
                "n_batches_completed": self.n_batches,
                "scope": "rank 0 only" if trainer.world_size > 1 else "all",
                "metrics": _snapshot(pl_module),
            })

    def on_validation_epoch_end(self, trainer, pl_module):
        # Callbacks run before the module's own on_validation_epoch_end
        # (evaluation_loop._on_evaluation_epoch_end), so the metrics have not
        # been reset yet. Both of these are collectives: ALL ranks, before any
        # rank-zero check.
        metrics = _snapshot_all_ranks(pl_module)
        n_total = int(trainer.strategy.reduce(
            torch.tensor(float(self.n_batches), device=pl_module.device),
            reduce_op="sum").item())
        self.n_batches = 0            # reset so the count is per-epoch

        self._write({
            "n_batches_completed": n_total,
            "scope": "all",
            "world_size": trainer.world_size,
            "global_step": trainer.global_step,
            "metrics": metrics,
        })

        # ALSO keep a per-epoch copy. self.path is a fixed filename, so without
        # this each epoch silently overwrites the last and a multi-epoch run
        # yields a single final number instead of a curve -- which is exactly
        # what you need to tell "it never learned" apart from "it learned and
        # then overfit". Not for the sanity check: its 2 batches would sit in
        # the epoch file until the real validation replaced them, and a run
        # that stopped first would leave a 2-sample "result" behind.
        if not _is_rank_zero() or trainer.sanity_checking:
            return
        try:
            ep = int(getattr(trainer, "current_epoch", -1))
            snap = self.path.with_name(f"{self.path.stem}_epoch{ep:03d}.json")
            snap.write_text(self.path.read_text())
        except Exception as exc:      # telemetry must never break a run
            print(f"[val_progress] per-epoch snapshot failed: "
                  f"{type(exc).__name__}: {exc}")
