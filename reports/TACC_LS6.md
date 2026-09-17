# Moving to Lonestar6

TACC access came through (2026-09-17). This is the setup path and what changes
now that the 16 GB constraint is gone.

## Why none of this can be automated from the laptop

TACC requires multi-factor authentication on every login: a TACC password plus a
6-digit token code, entered interactively. There are no SSH keys on this machine
and the first login is the thing that would place one. So job submission is
something you run; these scripts exist so it is copy-paste rather than authoring.

And it should stay that way: see step 2 for why setting up SSH keys to automate
this is a bad trade on LS6 specifically.

## Step 1 — find your allocation name

On a login node:

```
/usr/local/etc/taccinfo
```

It prints a table with a **Project** column. That string goes in `-A` in every
job script. Without it the job is rejected at submit time, which is the friendly
failure; the unfriendly one is an indented `#SBATCH` line silently truncating the
directive block so `-A` never gets read at all (see the note at the top of
`scripts/ls6_smoke.slurm`).

## Step 2 — do NOT set up SSH keys the usual way

An earlier version of this document said to run `ssh-keygen` locally and then
`ssh-copy-id` to LS6. **Do not do that.** From the Lonestar6 user guide:

> Do not run the `ssh-keygen` command on Lonestar6. This command will create and
> configure a key pair that will interfere with the execution of job scripts in
> the batch system.

LS6 auto-generates a key pair in `~/.ssh` that the **batch system itself uses**.
`ssh-copy-id` writes into that same directory. It does not run `ssh-keygen`
remotely, so it is not the exact thing the warning names — but it modifies the
one directory TACC tells you to leave alone, and the failure mode is job scripts
breaking for reasons unrelated to your code, which is a miserable thing to debug.

Not worth it here. Submitting a job is a handful of commands a few times a day;
the convenience does not justify touching that directory.

**If you have already broken it**, TACC documents the recovery:

```
mv .ssh dot.ssh.old        # on LS6
# log out, then log back in -- the system regenerates a correct key pair
```

### Other connection notes from the user guide

```
ssh username@ls6.tacc.utexas.edu               # rotates across login1-3
ssh username@login2.ls6.tacc.utexas.edu        # pin a specific login node
ssh -X username@ls6.tacc.utexas.edu            # X11, for GUI applications
ssh -vvv username@ls6.tacc.utexas.edu          # verbose, for a help ticket
```

Pinning a login node is worth knowing: if you stage 3.8 GB of assets in a
`screen`/`tmux` session on login2, you need login2 again to get back to it.

## Step 3 — the smoke test

`scripts/ls6_smoke.slurm` allocates a GPU node, prints what it sees, and exits.
It does no training and touches no data, deliberately: "can I run jobs" and "does
my training work" are different failures and much easier to debug apart.

```
ssh ls6
cd $WORK
git clone https://github.com/SaifSyed08/mhc1-boltz.git
cd mhc1-boltz
sed -i 's/SET_YOUR_ALLOCATION_HERE/<your-project>/' scripts/ls6_smoke.slurm
sbatch scripts/ls6_smoke.slurm
squeue -u $USER
```

Cost: GPU nodes are charged **4x SU/hour** and every job is billed a **minimum of
15 minutes** regardless of runtime, so this is ~1 SU.

The output answers four things worth knowing before a real run:

1. **Which GPUs.** LS6 A100 nodes have three A100s at **40 GB** each. Confirm what
   you actually got.
2. **Which modules** are available for python/CUDA.
3. **`$HOME` / `$WORK` / `$SCRATCH`** and their space. `$HOME` is small and backed
   up; `$WORK` is large and not purged; `$SCRATCH` is huge and **purged
   periodically**. The 3.8 GB of assets and any checkpoints belong in `$WORK`.
4. **Whether compute nodes have internet.** This is the one that matters most for
   our setup. TACC compute nodes are typically firewalled from the outside even
   though login nodes are not. If so, the 3.8 GB checkpoint download and the
   `git clone` **cannot happen inside a job** — they must be staged from a login
   node into `$WORK` first, and the job reads from there. The Kaggle notebook does
   all of that inside the job, so that part does not port.

## What changes with 40 GB instead of 16

Most of the deviations forced by the T4 exist only because of memory. On an A100
40 GB they can come off, and each one removed is one less caveat in the writeup.

| deviation | why it existed | on a 40 GB A100 |
|---|---|---|
| `diffusion_multiplicity: 4` | `smooth_lddt_loss` builds `[mult, n_atoms, n_atoms]` matrices | **restore 16** — at 4608 atoms that is ~10 GiB retained, which fits |
| `ema: false` | saves 1.61 GiB | **restore `true`** — it is what Boltz-1's released weights use |
| trunk frozen | removes gradients, Adam state and the backward graph | **optional now** — keep it as a deliberate strategy, or unfreeze for the full recipe |
| `accumulate_grad_batches: 16` | 128 meant ~1 optimizer step per epoch at T4 speed | **revisit** — an A100 is much faster per batch, so 128 may be affordable |
| `max_atoms: 3904` | 4608 was 18% padding on this dataset | **keep** — costs nothing, crops nothing |
| `BOLTZ_CHUNK_IN_TRAINING=1` | upstream disables chunking in training | **keep or drop** — it is a speed/memory trade, and with 40 GB the memory is not needed |

The one thing that does **not** change: `validation_args.diffusion_samples` must
stay at **3** to remain comparable to the 0.8827 baseline. `val/lddt` is
best-of-N (`model.py:704`), so that field is part of the metric's definition, not
a compute knob. See `reports/CLOUD_GPU.md` §4.

Do not change several of these at once. The point of going to LS6 is a clean
result, and a run that differs from the baseline in six ways is not much easier to
interpret than the T4 one was.

## Suggested order once the smoke test passes

1. **Reproduce the baseline on LS6.** `configs/mhc1_baseline_subset30.yaml`,
   `validation_only`. It should land near 0.8827. If it does not, the environment
   differs and nothing after this is interpretable.
2. **One config change at a time**, starting with `diffusion_multiplicity: 16`,
   then `ema: true`.
3. **Then the real fine-tune**, with enough steps to mean something —
   `reports/CLOUD_GPU.md` has the arithmetic on what "enough" costs.

## Sources

* [Lonestar6 user guide](https://docs.tacc.utexas.edu/hpc/lonestar6/)
* [Running jobs on Lonestar6](https://docs.tacc.utexas.edu/hpc/6lonestar/running/)
* [TACC multi-factor authentication](https://docs.tacc.utexas.edu/basics/mfa/)
