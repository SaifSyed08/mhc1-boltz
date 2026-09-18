#!/bin/bash
#----------------------------------------------------------------------------
# One-time Lonestar6 setup: environment, assets, sanity checks.
#
# RUN THIS ON A LOGIN NODE, not in a job:
#   cd $WORK/mhc1-boltz && bash scripts/ls6_setup.sh
#
# The smoke test (job 3452558) confirmed compute nodes CAN reach the internet on
# LS6, so this could run inside a job -- but there is no reason to burn SUs on a
# pip install, and login nodes are the right place for setup work. Keep anything
# heavy off the login nodes; this is downloads and installs, which is fine.
#
# What it does NOT do: fetch data/processed and data/msa. Those are ~575 MB, are
# not in git, and cannot be regenerated without re-scanning the upstream S3 tars.
# Copy them from the laptop instead -- see the instructions this script prints
# at the end, or reports/TACC_LS6.md.
#----------------------------------------------------------------------------
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
echo "repo: $REPO"

if [[ "$REPO" == /home1/* ]]; then
    echo "!! This repo is under \$HOME, which has a 10 GB quota and is 89% full"
    echo "!! at the filesystem level. Move it to \$WORK:"
    echo "!!    mv $REPO \$WORK/ && cd \$WORK/mhc1-boltz"
    exit 1
fi

#----------------------------------------------------------------------------
# 1. Modules
#----------------------------------------------------------------------------
echo
echo "=== modules ==="
source "$REPO/scripts/ls6_env.sh"
module list 2>&1 | sed 's/^/  /' 

#----------------------------------------------------------------------------
# 2. Virtualenv
#
# Built in $WORK, not $HOME: a torch install is several GB and $HOME has 10 GB.
#----------------------------------------------------------------------------
VENV="$REPO/.venv-ls6"
echo
echo "=== virtualenv: $VENV ==="
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    echo "  created"
else
    echo "  exists, reusing"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
ls6_env_report
python -m pip install -q --upgrade pip

#----------------------------------------------------------------------------
# 3. Dependencies
#
# Same set that worked on Kaggle (python 3.12), minus the three boltz pins that
# are droppable -- numpy==1.26.3 (stale), dm-tree (never imported), biopython
# (not on the training path). fairscale is NOT droppable: it is what
# activation_checkpointing calls.
#
# torch comes from the cu128 index to match the cuda/12.8 module and the 570.x
# driver the smoke test reported.
#----------------------------------------------------------------------------
echo
echo "=== dependencies ==="
python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -q \
    "hydra-core==1.3.2" "pytorch-lightning==2.4.0" "fairscale==0.4.13" \
    "omegaconf==2.3.1" "einops==0.8.0" "mashumaro==3.14" "modelcif==1.8" \
    numba rdkit scipy pandas psutil

#----------------------------------------------------------------------------
# 4. Boltz checkout + our patch
#----------------------------------------------------------------------------
echo
echo "=== boltz-src ==="
if [ ! -d boltz-src ]; then
    git clone -q https://github.com/jwohlwend/boltz.git boltz-src
    git -C boltz-src checkout -q v1.0.0
    git -C boltz-src apply ../patches/boltz-v1.0.0-training-path.patch
    echo "  cloned v1.0.0 and applied the patch"
else
    echo "  exists, leaving alone"
fi

# Verify, on both branches. A directory that exists is not a patch that applied:
# if a previous run cloned and then failed to patch, every later run would train
# on unpatched Boltz. This is how the cloud once ended up with a worse Boltz than
# the laptop.
declare -A SENTINEL=(
  ["src/boltz/model/modules/trunk.py"]="BOLTZ_CHUNK_IN_TRAINING"
  ["src/boltz/model/modules/diffusion.py"]="center_random_augmentation"
  ["src/boltz/data/feature/featurizer.py"]="chain_constraint_features = {}"
  ["src/boltz/model/optim/scheduler.py"]="except TypeError"
  ["scripts/train/train.py"]="ckpt_every_minutes"
  ["src/boltz/model/model.py"]="oom-context"
)
missing=0
for f in "${!SENTINEL[@]}"; do
    grep -q -- "${SENTINEL[$f]}" "boltz-src/$f" || { echo "  !! NOT PATCHED: $f"; missing=1; }
done
[ "$missing" -eq 0 ] && echo "  patch verified: ${#SENTINEL[@]}/${#SENTINEL[@]} sentinels present" \
                     || { echo "  Delete boltz-src/ and re-run."; exit 1; }

python -m pip install -q --no-deps -e boltz-src

#----------------------------------------------------------------------------
# 5. Assets (3.8 GB, re-downloadable)
#----------------------------------------------------------------------------
echo
echo "=== assets ==="
if [ -f data/assets/boltz1_conf.ckpt ] && [ -f data/assets/symmetry.pkl ]; then
    echo "  present"
else
    python src/fetch_assets.py
fi
du -sh data/assets 2>/dev/null | sed 's/^/  /'

#----------------------------------------------------------------------------
# 6. Verify the environment actually imports what train.py needs
#----------------------------------------------------------------------------
echo
echo "=== import check ==="
python - <<'PYEOF'
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  built for: {torch.cuda.get_arch_list()}")
import boltz.model.model, boltz.data.module.training, fairscale, numba  # noqa
import pytorch_lightning as pl
print(f"  pytorch-lightning {pl.__version__}")
print("  imports ok")
PYEOF

#----------------------------------------------------------------------------
# 7. Data -- the one thing this script cannot do for you
#----------------------------------------------------------------------------
echo
echo "=== dataset ==="
n_struct=$(ls data/processed/structures/*.npz 2>/dev/null | wc -l)
n_msa=$(ls data/msa/*.npz 2>/dev/null | wc -l)
echo "  structures: $n_struct (want 1084)    msa: $n_msa (want 1457)"

if [ "$n_struct" -ne 1084 ]; then
    cat <<EOF

  Dataset not staged. data/processed and data/msa are ~575 MB, are not in git,
  and cannot be regenerated without re-scanning the upstream S3 tars. Copy the
  bundle from the laptop, in a LOCAL PowerShell window:

      scp "C:\\Users\\Saif Syed\\mhc1-boltz\\mhc1-data-bundle.tar.gz" \\
          saifsyed@ls6.tacc.utexas.edu:$REPO/

  then back here:

      cd $REPO && tar xzf mhc1-data-bundle.tar.gz && rm mhc1-data-bundle.tar.gz

  (If scp hits the same "Corrupted MAC" bug as ssh, add -o MACs=hmac-sha2-512.)

EOF
    exit 1
fi

echo
echo "=== ready ==="
echo "  next: sbatch scripts/ls6_baseline.slurm"
