#!/bin/bash -l
# Constellation stage A (link candidates) as a job array: task i evaluates the contiguous block
# of satellites [floor(i N/n), floor((i+1) N/n)) and writes CANDIDATES_DIR/linkset_<i>_of_<n>.npz.
# Then ONE dependent merge-and-schedule job (slurm/constellation_merge.sbatch) runs stage B:
#   CFG=configs/constellation_demo_n1000.yaml
#   CAND=$HELIOSWARM_RUNS_DIR/constellation_demo_n1000_links
#   JID=$(sbatch --parsable --array=0-4 slurm/constellation_links_array.sbatch "$CFG" "$CAND")
#   sbatch --dependency=afterok:$JID slurm/constellation_merge.sbatch "$CFG" "$CAND"
# Tasks are independent and deterministic (random phases are drawn from project.seed in every
# task), so a failed task can be re-run alone; export CHUNK_COUNT=<n> when re-running a subset
# (e.g. --array=3 of 5) so the split is kept. The test partition's QOS allows 5 submitted jobs per
# user (array tasks count individually): use 5 tasks there, or -p shared for wider arrays.
# Sizing (measured, array 60272021, docs/milestones/M7.md D): 200 satellites x 3 sites, one year
# at 3896 s, numpy beam reference: 62-72 min wall, peak RSS 1.14 GB, CPU efficiency 99 % (single
# threaded; the beam stage dominates) -> 1 core, 1300M (peak + 14 %), 1 h 25 min (+15 %). Scale
# the time with satellites per task and epochs (~20 s per satellite-year with 3 sites).
#SBATCH -J helioswarm-clinks
#SBATCH -p test
#SBATCH -c 1
#SBATCH --mem=1300M
#SBATCH -t 0-01:25
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%A_%a.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%A_%a.err

source "${HELIOSWARM_REPO:-/n/home05/ianpoon/Project-Dyson-Sphere}/slurm/common.sh"
export JAX_PLATFORMS=cpu
CONFIG="${1:?usage: sbatch --array=0-N slurm/constellation_links_array.sbatch CONFIG CANDIDATES_DIR [args]}"
CAND="${2:?usage: sbatch --array=0-N slurm/constellation_links_array.sbatch CONFIG CANDIDATES_DIR [args]}"
shift 2
mkdir -p "$CAND"
/usr/bin/time -v uv run --locked helioswarm run "$CONFIG" --candidates-dir "$CAND" \
    --chunk-index "${SLURM_ARRAY_TASK_ID:?submit with --array}" \
    --chunk-count "${CHUNK_COUNT:-${SLURM_ARRAY_TASK_COUNT:?submit with --array}}" "$@"
