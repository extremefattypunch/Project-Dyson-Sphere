#!/bin/bash -l
# ERA5 download of a download configuration (M8a):
#   mkdir -p /n/netscratch/hekstra_lab/Lab/ian_poon/logs
#   sbatch slurm/era5_download.sbatch configs/weather/era5_sites.yaml [N_PARALLEL]
# One job, all sites: N_PARALLEL (default 4) site downloads run concurrently, each submitting its
# requests one after the other, so at most N_PARALLEL requests sit in the CDS queue at a time.
# One job (not an array) because the test partition allows 5 submitted jobs per user. With
# --array=I-J each task downloads only site I..J (one site per task) instead.
# Files + provenance JSON -> $HELIOSWARM_WEATHER_DIR (netscratch). Idempotent: a resubmitted
# job skips verified files. Network-bound (CDS queue): 1 core, small memory, test partition
# (free, < 12 h). Needs ~/.cdsapirc (never printed). Sizing: docs/milestones/M8a.md (D).
#SBATCH -J helioswarm-era5
#SBATCH -p test
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 0-08:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.err

source "${HELIOSWARM_REPO:-/n/home05/ianpoon/Project-Dyson-Sphere}/slurm/common.sh"
CONFIG="${1:?usage: sbatch slurm/era5_download.sbatch ERA5_CONFIG [N_PARALLEL]}"
NPAR="${2:-4}"
uv run --locked --extra weather python -c "import helioswarm; print('helioswarm from', helioswarm.__file__)"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    uv run --locked --extra weather helioswarm weather download "$CONFIG" --site-index "$SLURM_ARRAY_TASK_ID"
    exit
fi
NSITES="$(uv run --locked --extra weather python -c "from helioswarm.era5 import load_download_config as l; print(len(l('$CONFIG').sites))")"
mkdir -p "$TMP/site_logs"
# Each site's output goes to its own log; the job fails if any site fails.
seq 0 $((NSITES - 1)) | xargs -P "$NPAR" -I{} bash -c \
    "uv run --locked --extra weather helioswarm weather download '$CONFIG' --site-index {} > '$TMP/site_logs/site_{}.log' 2>&1; rc=\$?; echo \"site {} exit \$rc\"; cat '$TMP/site_logs/site_{}.log'; exit \$rc"
