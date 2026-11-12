#!/bin/bash -l
# cProfile of one end-to-end run (PDF §14.3 item 1):
#   sbatch slurm/profile.sbatch CONFIG
# Output: $HELIOSWARM_RUNS_DIR/profile/<config stem>-<jobid>/ (profile.pstats, profile_top.txt,
# profile_summary.json with stage wall times and peak RSS).
#SBATCH -J helioswarm-profile
#SBATCH -p test
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 0-00:15
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.err

source "${HELIOSWARM_REPO:-/n/home05/ianpoon/Project-Dyson-Sphere}/slurm/common.sh"
export JAX_PLATFORMS=cpu
CONFIG="${1:?usage: sbatch slurm/profile.sbatch CONFIG}"
NAME="$(basename "$CONFIG" .yaml)"
/usr/bin/time -v uv run --locked python scripts/profile_run.py "$CONFIG" \
    "$HELIOSWARM_RUNS_DIR/profile/${NAME}-${SLURM_JOB_ID}"
