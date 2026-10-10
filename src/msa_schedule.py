"""Gradual MSA removal: shrink the MSA depth cap epoch by epoch.

Why this exists
---------------
Ernest (2026-10-12): cutting the MSA at inference alone will cost accuracy; train
it away instead, giving the model fewer MSA rows each step until it runs on the
query alone. If quality holds without the MSA, that is a result.

Upstream already varies depth in training: the featurizer draws
max_seqs_batch = randint(1, max_seqs + 1) per sample (featurizer.py ~1174), so
the model has always seen some shallow MSAs. This callback lowers the CAP
(`max_seqs`) per epoch, which moves that whole distribution down; at cap 1 every
sample is the query sequence alone.

Configured by environment, so a run without it is untouched:
  MSA_SCHEDULE="2048,512,128,32,8,1"   cap for epoch 0, 1, 2, ...; the last value
                                       holds for every later epoch
  MSA_VAL_DEPTH=1                      validation depth (deterministic: the
                                       first N rows; row 0 is the query)

When MSA_SCHEDULE is set, failure is LOUD: an exception here stops the run
rather than letting it train at full depth while claiming otherwise. Every
epoch's cap and the MSA depth actually seen in the first batch go to
<output>/msa_schedule.jsonl (with pad_to_max_seqs the batch's MSA dimension
equals the cap, so a mismatch means the workers did not pick it up).

Timing (checked against PL 2.4): FitLoop.run() builds the train iterator -- and
forks the workers -- in setup_data(), BEFORE on_train_start fires, and on a
resume it does not rebuild it for the restored epoch. So the cap must already be
on the dataset by then:
  * setup()               -> cap(0), for a fresh run;
  * on_load_checkpoint()  -> cap(checkpoint epoch), for a resume (runs during
                             checkpoint restore, before setup_data);
  * on_train_epoch_end()  -> cap(e + 1). Workers are not persistent and train.py
                             reloads dataloaders every epoch, so epoch e+1's
                             workers are forked after this.
"""
import json
import os
from pathlib import Path

from pytorch_lightning import Callback


class MSASchedule(Callback):
    def __init__(self, output_dir=None):
        raw = os.environ.get("MSA_SCHEDULE", "").strip()
        self.sched = [int(x) for x in raw.split(",")] if raw else None
        v = os.environ.get("MSA_VAL_DEPTH", "").strip()
        self.val_depth = int(v) if v else None
        self.out = Path(output_dir) / "msa_schedule.jsonl" if output_dir else None
        self._checked = set()
        if self.sched:
            assert all(x >= 1 for x in self.sched), f"MSA_SCHEDULE must be >= 1: {raw}"
            print(f"[msa_schedule] caps per epoch {self.sched}, validation depth {self.val_depth}")

    def cap(self, epoch):
        return self.sched[min(epoch, len(self.sched) - 1)]

    def _log(self, trainer, rec):
        print(f"[msa_schedule] {rec}")
        if self.out is not None and trainer.global_rank == 0:
            with self.out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def _set_train(self, trainer, epoch):
        ds = trainer.datamodule._train_set
        ds.max_seqs = self.cap(epoch)
        self._log(trainer, {"event": "set", "epoch": epoch, "train_cap": ds.max_seqs,
                            "global_step": int(trainer.global_step)})

    def _set_val(self, trainer):
        if self.val_depth is not None:
            trainer.datamodule._val_set.max_seqs = self.val_depth

    # --- hooks -------------------------------------------------------------
    def setup(self, trainer, pl_module, stage=None):
        if self.sched or self.val_depth is not None:
            if self.out is None:
                self.out = Path(trainer.default_root_dir) / "msa_schedule.jsonl"
            self._set_val(trainer)
        if self.sched and stage == "fit":
            self._set_train(trainer, 0)

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if self.sched:
            self._set_train(trainer, int(checkpoint.get("epoch", 0)))

    def on_train_epoch_end(self, trainer, pl_module):
        if self.sched:
            self._set_train(trainer, trainer.current_epoch + 1)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.sched and ("train", trainer.current_epoch) not in self._checked:
            self._checked.add(("train", trainer.current_epoch))
            seen = int(batch["msa"].shape[1])
            want = self.cap(trainer.current_epoch)
            self._log(trainer, {"event": "seen", "epoch": trainer.current_epoch,
                                "train_cap": want, "batch_msa_depth": seen})
            if seen != want:
                raise RuntimeError(f"[msa_schedule] epoch {trainer.current_epoch}: batch MSA "
                                   f"depth {seen} != cap {want}; workers did not pick it up")

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if self.val_depth is not None and batch_idx == 0:
            seen = int(batch["msa"].shape[1])
            key = ("val", trainer.global_step, trainer.sanity_checking)
            if key not in self._checked:
                self._checked.add(key)
                self._log(trainer, {"event": "val_seen", "global_step": int(trainer.global_step),
                                    "val_depth": self.val_depth, "batch_msa_depth": seen})
                if seen != self.val_depth:
                    raise RuntimeError(f"[msa_schedule] validation MSA depth {seen} != {self.val_depth}")
