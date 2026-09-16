"""Per-batch GPU memory telemetry for the fine-tuning run.

Why this exists: three separate OOMs on this project were diagnosed by reading a
traceback and reasoning about what *should* be resident. Two of those diagnoses
were wrong, in both cases because a number read off a process that had already
died was treated as a measure of demand rather than as the point where it ran
out. This callback records what is actually resident, per batch, so the next
question is answered from a file instead of from an argument.

Writes one JSON object per line to <output>/mem.jsonl and prints a compact line
every `print_every` batches:

    [mem] b=16 alloc=13.09 peak=13.41 resv=13.38 host_free=22.1

  alloc      torch.cuda.memory_allocated  -- live tensors
  peak       torch.cuda.max_memory_allocated since the last reset
  resv       torch.cuda.memory_reserved -- what the caching allocator holds
  host_free  system RAM free, which matters when offload_to_cpu is on

The gap between `alloc` and `resv` is fragmentation. A large gap with an OOM
means the allocator is holding memory it cannot hand out, and
expandable_segments is worth trying; a small gap means the job genuinely needs
that much and the fix has to be a real one.

The step at which `alloc` jumps by ~2.25 GB is the first optimizer step, where
Adam allocates exp_avg and exp_avg_sq. With accumulate_grad_batches: 16 that is
batch 16, not batch 0, which is why a run can survive 20+ batches and then die.
"""

import json
import time
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

GB = 1024**3


def _host_free_gb() -> float:
    """Free system RAM, without requiring psutil."""
    try:
        import psutil

        return psutil.virtual_memory().available / GB
    except ImportError:
        pass
    try:  # Linux
        with Path("/proc/meminfo").open() as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / GB
    except OSError:
        pass
    return float("nan")


class MemProbe(Callback):
    """Record GPU memory per training batch."""

    def __init__(self, path: Path, print_every: int = 4) -> None:
        self.path = Path(path)
        self.print_every = print_every
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self._t0 = None
        self._peak_seen = 0.0

    def _open(self):
        if self._fh is None:
            self._fh = self.path.open("a", buffering=1)  # line-buffered: survives a kill
        return self._fh

    def on_train_start(self, trainer, pl_module) -> None:  # noqa: ANN001
        if not torch.cuda.is_available():
            return
        torch.cuda.reset_peak_memory_stats()
        props = torch.cuda.get_device_properties(0)
        rec = {
            "event": "start",
            "gpu": props.name,
            "capability": f"sm_{props.major}{props.minor}",
            "total_gb": round(props.total_memory / GB, 2),
            "alloc_gb": round(torch.cuda.memory_allocated() / GB, 2),
            "trainable_m": round(
                sum(p.numel() for p in pl_module.parameters() if p.requires_grad) / 1e6, 1
            ),
            "frozen_m": round(
                sum(p.numel() for p in pl_module.parameters() if not p.requires_grad) / 1e6, 1
            ),
        }
        self._open().write(json.dumps(rec) + "\n")
        print(f"[mem] start {rec}")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:  # noqa: ANN001
        self._t0 = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: ANN001
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / GB
        peak = torch.cuda.max_memory_allocated() / GB
        resv = torch.cuda.memory_reserved() / GB
        rec = {
            "event": "batch",
            "batch": batch_idx,
            "global_step": int(trainer.global_step),
            "alloc_gb": round(alloc, 2),
            "peak_gb": round(peak, 2),
            "reserved_gb": round(resv, 2),
            "frag_gb": round(resv - alloc, 2),
            "host_free_gb": round(_host_free_gb(), 1),
            "secs": round(time.time() - self._t0, 1) if self._t0 else None,
        }
        self._open().write(json.dumps(rec) + "\n")

        # Always announce a new high-water mark -- that is the line you want when
        # reading back a log that ends in an OOM.
        new_peak = peak > self._peak_seen + 0.05
        self._peak_seen = max(self._peak_seen, peak)
        if new_peak or batch_idx % self.print_every == 0:
            print(
                f"[mem] b={batch_idx} alloc={alloc:.2f} peak={peak:.2f} "
                f"resv={resv:.2f} frag={resv - alloc:.2f} "
                f"host_free={rec['host_free_gb']:.1f} {rec['secs']}s"
                + ("  <- new peak" if new_peak else "")
            )

    def on_train_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        if self._fh is not None:
            self._fh.close()
            self._fh = None
