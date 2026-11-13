# Sourced by every slurm/*.sbatch (hekstra-cluster contract): repository, environment,
# node-local TMPDIR with cleanup, outputs on netscratch. Not a job script itself.
set -euo pipefail
REPO="${HELIOSWARM_REPO:-/n/home05/ianpoon/Project-Dyson-Sphere}"
cd "$REPO"
# shellcheck source=/dev/null
source scripts/cluster_env.sh
# Optional environment overlay sourced after cluster_env.sh (default: unset, no effect). A
# parallel worktree with its own venv exports e.g.
#   HELIOSWARM_REPO=/path/to/worktree HELIOSWARM_ENV_OVERLAY=/path/to/overlay.sh
# before sbatch so that jobs use that checkout's venv and caches, not the main checkout's.
if [[ -n "${HELIOSWARM_ENV_OVERLAY:-}" ]]; then
    # shellcheck source=/dev/null
    source "$HELIOSWARM_ENV_OVERLAY"
    echo "environment overlay: $HELIOSWARM_ENV_OVERLAY (UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT)"
fi
TMP="/scratch/${USER}_${SLURM_JOB_ID}"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT
export TMPDIR="$TMP"
mkdir -p "$HELIOSWARM_RUNS_DIR"
echo "job ${SLURM_JOB_ID} (${SLURM_JOB_NAME}) on $(hostname) partition ${SLURM_JOB_PARTITION:-?}" \
     "cpus ${SLURM_CPUS_PER_TASK:-?} start $(date -u +%FT%TZ) git $(git rev-parse --short HEAD)"
