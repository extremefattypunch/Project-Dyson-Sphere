# Sourced by every slurm/*.sbatch (hekstra-cluster contract): repository, environment,
# node-local TMPDIR with cleanup, outputs on netscratch. Not a job script itself.
set -euo pipefail
REPO="${HELIOSWARM_REPO:-/n/home05/ianpoon/Project-Dyson-Sphere}"
cd "$REPO"
# shellcheck source=/dev/null
source scripts/cluster_env.sh
TMP="/scratch/${USER}_${SLURM_JOB_ID}"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT
export TMPDIR="$TMP"
mkdir -p "$HELIOSWARM_RUNS_DIR"
echo "job ${SLURM_JOB_ID} (${SLURM_JOB_NAME}) on $(hostname) partition ${SLURM_JOB_PARTITION:-?}" \
     "cpus ${SLURM_CPUS_PER_TASK:-?} start $(date -u +%FT%TZ) git $(git rev-parse --short HEAD)"
