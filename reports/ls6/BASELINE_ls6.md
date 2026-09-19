# Baseline reproduced on Lonestar6

Job 3454272, `gpu-a100`, NVIDIA A100-PCIE-40GB. `validation_only` against the
same 30-sample subset with identical `validation_args`.

| metric | LS6 A100 | laptop RTX 4060 | delta |
|---|---|---|---|
| `val/lddt_protein_protein` | **0.8843** | 0.8827 | +0.0016 |
| `val/lddt_intra_protein` | **0.9250** | 0.9248 | +0.0002 |
| `val/rmsd` | 3.0611 Å | 3.053 Å | +0.0081 |
| `val/best_rmsd` | 2.9785 Å | 2.982 Å | −0.0035 |

The environment is sound. Windows/RTX 4060/torch 2.11+cu128 versus
Linux/A100/torch+cu128 on a cluster, and lDDT agrees to within 0.0016. Anything
the fine-tune produces here is comparable to `reports/BASELINE.md`.

## The more useful result: an empirical noise floor

This run changed **nothing about the model**. Same weights, same data, same
validation settings — only the hardware and the software stack differ. So the
deltas above are pure measurement noise, and that gives a number this project did
not previously have.

```
environment-only change      lddt_protein_protein  +0.0016
Kaggle fine-tune, 6 steps    lddt_protein_protein  +0.0013
```

**The Kaggle "improvement" is smaller than the noise floor.** That run was already
reported as flat in `reports/t4/RESULT_epoch0.md` on the grounds that six
optimizer steps cannot do much — this is the same conclusion reached
independently, and more rigorously: a change known to be zero produces a larger
delta than the change being tested did.

Concretely, for anything measured from here on:

> An improvement in `val/lddt_protein_protein` below about **0.002** is not
> distinguishable from swapping the machine it ran on.

That bar belongs in any result sent to Ernest, and it applies to the running
fine-tune too.

`val/rmsd` moved +0.0081 on an unchanged model, which is a further reason to
distrust the `best_rmsd` 2.982 → 1.694 excursion in the T4 run: RMSD is visibly
noisier than lDDT here, being global and outlier-sensitive rather than local and
averaged.

## Two warnings in the log, neither a problem

* `val_dataloader does not have many workers ... consider num_workers=127` —
  Lightning noticing an LS6 node has 128 cores. Validation is GPU-bound at
  `sampling_steps: 200`, so this is not the bottleneck. Worth revisiting only if
  a profile says otherwise.
* `compute was called before update on MeanMetric` — emitted once at startup
  before the first validation batch. Harmless; `ValProgressDump` snapshots
  metric state after every batch and reads an empty metric on the first call.
