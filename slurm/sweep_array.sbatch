#!/bin/bash -l
# Parameter sweep as a job array (deterministic round-robin split of the grid):
#   SWEEP=configs/optical_1au_sweep.yaml
#   OUT=$HELIOSWARM_RUNS_DIR/optical_1au_sweep
#   JID=$(sbatch --parsable --array=0-4 slurm/sweep_array.sbatch "$SWEEP" "$OUT")
#   sbatch --dependency=afterany:$JID --export=ALL,COLLATE_ONLY=1 slurm/sweep_array.sbatch "$SWEEP" "$OUT"
# Task i runs grid points p with p mod SLURM_ARRAY_TASK_COUNT == i (helioswarm.sweep); to re-run
# a subset of tasks (e.g. --array=1-4 of 5) export SWEEP_ARRAY_COUNT=5 so the split is kept. The
# dependent COLLATE_ONLY job rebuilds sweep_summary.parquet from every point directory.
# The test partition's QOS allows 5 submitted jobs per user (array tasks count individually), so
# the 45-point optical sweep runs as 5 tasks of 9 points; submit the collate job once the array
# has left the queue (or use -p shared for wider arrays). Sizing (measured, arrays 60112261 and
# 60127591, 9 points per task): 11.8-14.8 min wall, peak RSS 436-489 MB, CPU efficiency 97-98 %
# -> 1 core, 560 MB (489 MB + 10 %), 17 min. The COLLATE_ONLY job needs ~2 min and 512 MB (pass
# -t 0-00:02 --mem=512M). Microwave sweeps need more memory (ITU-R maps): configs/microwave_1au_sweep.yaml
# as 5 tasks of 3 points (array 60141210, M6b) took 2:13-2:17 wall with peak RSS 632-704 MB, so
# submit it with --mem=768M -t 0-00:12 (README).
#SBATCH -J helioswarm-sweep
#SBATCH -p test
#SBATCH -c 1
#SBATCH --mem=560M
#SBATCH -t 0-00:17
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%A_%a.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%A_%a.err

source "${HELIOSWARM_REPO:-/n/home05/ianpoon/Project-Dyson-Sphere}/slurm/common.sh"
export JAX_PLATFORMS=cpu
SWEEP="${1:?usage: sbatch --array=0-N slurm/sweep_array.sbatch SWEEP_YAML OUT_DIR [args]}"
OUT="${2:?usage: sbatch --array=0-N slurm/sweep_array.sbatch SWEEP_YAML OUT_DIR [args]}"
shift 2
if [[ "${COLLATE_ONLY:-0}" == "1" ]]; then
    uv run --locked helioswarm sweep "$SWEEP" --out "$OUT" --collate-only
else
    /usr/bin/time -v uv run --locked helioswarm sweep "$SWEEP" --out "$OUT" \
        --array-index "${SLURM_ARRAY_TASK_ID:?submit with --array}" \
        --array-count "${SWEEP_ARRAY_COUNT:-${SLURM_ARRAY_TASK_COUNT:?submit with --array}}" "$@"
fi
