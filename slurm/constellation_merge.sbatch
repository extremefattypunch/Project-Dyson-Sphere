#!/bin/bash -l
# Constellation stage B: merge the link files of an array (slurm/constellation_links_array.sbatch),
# schedule every epoch and write the run directory $HELIOSWARM_RUNS_DIR/<config stem>/:
#   sbatch --dependency=afterok:<array job id> slurm/constellation_merge.sbatch CONFIG CANDIDATES_DIR [args]
# Also used without CANDIDATES_DIR ("-") for a complete single-process constellation run:
#   sbatch slurm/constellation_merge.sbatch configs/constellation_demo.yaml -
# Sizing (measured, job 60272040, docs/milestones/M7.md D): the N = 1000 merge (5 link files,
# 8101 epochs x 1000 satellites x 3 sites held in memory, greedy scheduling 29 s) took 61 s wall
# with peak RSS 3.47 GB -> 1 core, 4G (peak + 15 %), 5 min. A single-process run ("-") does stage
# A as well: ~20 s per satellite-year with 3 sites (N = 100: 19 min, 0.84 GB) -> raise -t/--mem.
#SBATCH -J helioswarm-cmerge
#SBATCH -p test
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 0-00:05
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.err

source "${HELIOSWARM_REPO:-/n/home05/ianpoon/Project-Dyson-Sphere}/slurm/common.sh"
export JAX_PLATFORMS=cpu
CONFIG="${1:?usage: sbatch slurm/constellation_merge.sbatch CONFIG CANDIDATES_DIR|- [args]}"
CAND="${2:?usage: sbatch slurm/constellation_merge.sbatch CONFIG CANDIDATES_DIR|- [args]}"
shift 2
NAME="$(basename "$CONFIG" .yaml)"
EXTRA=()
if [[ "$CAND" != "-" ]]; then EXTRA=(--candidates-dir "$CAND"); fi
# OUT_DIR (environment) overrides the run directory (default $HELIOSWARM_RUNS_DIR/<config stem>).
OUT="${OUT_DIR:-$HELIOSWARM_RUNS_DIR/$NAME}"
/usr/bin/time -v uv run --locked helioswarm run "$CONFIG" --out "$OUT" "${EXTRA[@]}" "$@"
