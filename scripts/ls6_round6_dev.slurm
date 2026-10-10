#!/bin/bash
#----------------------------------------------------------------------------
# One dev-queue job for Round 6, because the dev queue allows only 3 submitted
# jobs per user: the Round 6 smoke test (scripts/ls6_round6_smoke.slurm), then
# the pocket / structure-dump eval (scripts/ls6_eval_pocket.slurm) whether or
# not the smoke passed -- the eval does not depend on the MSA schedule.
#----------------------------------------------------------------------------

#SBATCH -J mhc1-r6dev
#SBATCH -o logs/r6dev.%j.out
#SBATCH -e logs/r6dev.%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A MCB26003
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saif.s@utexas.edu

set -uo pipefail
cd "$WORK/mhc1-boltz"
bash scripts/ls6_round6_smoke.slurm; rc1=$?
bash scripts/ls6_eval_pocket.slurm; rc2=$?
echo "smoke rc=$rc1  pocket eval rc=$rc2"
exit $(( rc1 || rc2 ))
