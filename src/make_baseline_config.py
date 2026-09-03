"""Generate configs/mhc1_baseline.yaml from configs/mhc1_finetune.yaml.

Brief section 7 asks for a `validation_only` run of the *pretrained* model before
any training, so the fine-tuned model has something to be compared against. That
baseline is only meaningful if it evaluates the identical data and the identical
validation settings as the fine-tuning run -- so the baseline config is generated
from the fine-tune config rather than maintained separately, and the only fields
that may differ are listed in OVERRIDES below.

Anything under `model.validation_args`, `data.*` or the manifest/split paths is
deliberately NOT overridable here: changing those would make the baseline
incomparable, which defeats the point of measuring it.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "configs" / "mhc1_finetune.yaml"
DST = ROOT / "configs" / "mhc1_baseline.yaml"

# (exact line to find, replacement). Kept as whole-line matches so a drifting
# source config fails loudly instead of silently generating a wrong baseline.
OVERRIDES = [
    ("output: ../runs/mhc1_finetune         # run output directory",
     "output: ../runs/baseline              # run output directory"),
    ("disable_checkpoint: false",
     "disable_checkpoint: true              # nothing is trained, nothing to save"),
    # Provably a no-op here, and it buys back 3.63 GB on an 8 GB card.
    # on_validation_start -> prepare_eval() (model.py:1246) builds an
    # ExponentialMovingAverage if self.ema is None and then calls store(); that is
    # two full fp32 clones of all 453.6M trainable params. self.ema IS None here,
    # because on_load_checkpoint only restores it under `if "ema" in checkpoint`
    # and boltz1_conf.ckpt has no "ema" key -- so shadow_params is initialised
    # from the current weights and copy_to() writes them back onto themselves.
    # Identical numbers, 3.63 GB cheaper. See reports/GPU_REQUIREMENTS.md.
    ("  ema: true",
     "  ema: false                          # no-op for a fresh checkpoint; saves 3.63 GB"),
]

# Appended rather than substituted: these keys are absent from the fine-tune
# config because they take their dataclass defaults there.
APPEND = """
# ---------------------------------------------------------------------------
# Baseline-only settings (see src/make_baseline_config.py)
#
# validation_only: run trainer.validate() instead of trainer.fit(). No optimizer
#   is constructed and no gradients are held, which is the only reason this fits
#   on an 8 GB card at all -- see reports/GPU_REQUIREMENTS.md.
#
# debug: train.py treats this as "local sanity run": it forces devices=1,
#   num_workers=0, and sets wandb to None. That last part is what lets this run
#   without a Weights & Biases account; the fine-tuning run still logs to W&B as
#   the brief asks.
# ---------------------------------------------------------------------------
validation_only: true
debug: true
"""

HEADER = """# GENERATED FILE -- do not edit by hand.
# Regenerate with:  python src/make_baseline_config.py
# Source of truth:  configs/mhc1_finetune.yaml
#
"""


def main():
    text = SRC.read_text()
    for old, new in OVERRIDES:
        if text.count(old) != 1:
            sys.exit("mhc1_finetune.yaml drifted: cannot find unique line %r" % old)
        text = text.replace(old, new)
    DST.write_text(HEADER + text + APPEND)
    print("wrote %s" % DST)


if __name__ == "__main__":
    main()
