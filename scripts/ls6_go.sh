#!/bin/bash
#----------------------------------------------------------------------------
# One command to go from a fresh LS6 clone to a submitted baseline job.
#
#   cd $WORK/mhc1-boltz && git pull && bash scripts/ls6_go.sh
#
# Idempotent: every step checks whether it has already been done, so re-running
# after a failure picks up where it stopped rather than redoing the 3.8 GB
# download.
#----------------------------------------------------------------------------
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
echo "repo: $REPO"
echo

#----------------------------------------------------------------------------
# 1. Unpack the dataset bundle, if it is sitting here from the scp
#----------------------------------------------------------------------------
n_struct() { ls data/processed/structures/*.npz 2>/dev/null | wc -l; }

if [ "$(n_struct)" -eq 1084 ]; then
    echo "[1/4] dataset already staged (1084 structures)"
elif [ -f mhc1-data-bundle.tar.gz ]; then
    echo "[1/4] unpacking mhc1-data-bundle.tar.gz ..."
    # Verify before trusting it -- a truncated scp is the likely failure here and
    # it would otherwise surface much later as a confusing dataset error.
    if [ -f mhc1-data-bundle.tar.gz.sha256 ]; then
        sha256sum -c mhc1-data-bundle.tar.gz.sha256 || {
            echo "  !! checksum FAILED -- the transfer was truncated or corrupted."
            echo "  !! Delete it here and re-run 'bash ~/s.sh' on the laptop."
            exit 1
        }
    else
        echo "  (no .sha256 alongside; skipping integrity check)"
    fi
    tar xzf mhc1-data-bundle.tar.gz
    rm -f mhc1-data-bundle.tar.gz
    echo "  unpacked: $(n_struct) structures"
else
    cat <<EOF
[1/4] DATASET MISSING

  Expected either data/processed/structures/ (1084 .npz files) or
  mhc1-data-bundle.tar.gz in $REPO.

  Send it from the laptop, in Git Bash:

      bash ~/s.sh

  then re-run this script.
EOF
    exit 1
fi

#----------------------------------------------------------------------------
# 2. Environment: venv, deps, boltz + patch, assets
#----------------------------------------------------------------------------
echo
echo "[2/4] environment ..."
bash scripts/ls6_setup.sh

#----------------------------------------------------------------------------
# 3. Submit the baseline
#----------------------------------------------------------------------------
echo
echo "[3/4] submitting the pipeline ..."
mkdir -p logs
bash scripts/ls6_pipeline.sh


echo
