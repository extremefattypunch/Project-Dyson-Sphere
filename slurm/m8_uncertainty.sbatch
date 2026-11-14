#!/bin/bash -l
# M8 uncertainty study (helioswarm.uncertainty): every configured part (nested Monte Carlo,
# Sobol design, scenarios) evaluated by $SLURM_CPUS_PER_TASK worker processes, then the
# collation (statistics, Sobol indices with bootstrap CIs and convergence, scenario ranges):
#   sbatch slurm/m8_uncertainty.sbatch configs/uncertainty.yaml $HELIOSWARM_RUNS_DIR/uncertainty_helio
#   PARTS="mc scenarios" sbatch --export=ALL slurm/m8_uncertainty.sbatch ...   (a subset)
# A wide study may also be split over array tasks: --array=0-K with SPLIT=K+1 exported (each
# task writes part_<i>_of_<n>), then MODE=collate.
#SBATCH -J helioswarm-m8unc
#SBATCH -p test
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH -t 0-03:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%A_%a.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%A_%a.err

source "${HELIOSWARM_REPO:-/n/home05/ianpoon/Project-Dyson-Sphere}/slurm/common.sh"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STUDY="${1:?usage: sbatch slurm/m8_uncertainty.sbatch STUDY_YAML OUT_DIR}"
OUT="${2:?usage: sbatch slurm/m8_uncertainty.sbatch STUDY_YAML OUT_DIR}"
W="${SLURM_CPUS_PER_TASK:-1}"
if [[ "${MODE:-run}" == "collate" ]]; then
    uv run --locked helioswarm uncertainty "$STUDY" --out "$OUT" --collate
    exit $?
fi
IDX="${SLURM_ARRAY_TASK_ID:-0}"
N="${SPLIT:-1}"
PART_ARGS=()
for p in ${PARTS:-}; do PART_ARGS+=(--part "$p"); done
/usr/bin/time -v uv run --locked helioswarm uncertainty "$STUDY" --out "$OUT" "${PART_ARGS[@]}" \
    --array-index "$IDX" --array-count "$N" --workers "$W"
if [[ -n "${PARTS:-}" && "$N" == "1" ]]; then  # without PARTS the command collates itself
    uv run --locked helioswarm uncertainty "$STUDY" --out "$OUT" --collate
fi
