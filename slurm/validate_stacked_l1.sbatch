#!/bin/bash -l
# Stacked-L1 validation on heterogeneous orbits (M7 review): one year at 3896 s, six dissimilar
# satellites stacked in one DOP853 solve vs solo runs vs a tighter-tolerance solo reference, with
# the configured max_step and without it (scripts/validate_stacked_l1.py):
#   sbatch slurm/validate_stacked_l1.sbatch [OUT_JSON]
# (default $HELIOSWARM_RUNS_DIR/validate_stacked_l1/<jobid>.json). Sizing (measured, job
# 60291568, docs/milestones/M7.md D): 9 min 37 s wall, peak RSS 344 MB (/usr/bin/time), CPU
# efficiency 98.6 % -> 1 core, 400M (peak + 16 %), 12 min (+25 %).
#SBATCH -J helioswarm-stackval
#SBATCH -p test
#SBATCH -c 1
#SBATCH --mem=400M
#SBATCH -t 0-00:12
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.err

source "${HELIOSWARM_REPO:-/n/home05/ianpoon/Project-Dyson-Sphere}/slurm/common.sh"
export JAX_PLATFORMS=cpu
OUT="${1:-$HELIOSWARM_RUNS_DIR/validate_stacked_l1/${SLURM_JOB_ID}.json}"
/usr/bin/time -v uv run --locked python scripts/validate_stacked_l1.py "$OUT"
