#!/bin/bash -l
# M8: several sweeps in ONE multi-core job (instead of one array task per point: the test QOS
# allows 5 submitted jobs per user). Each sweep is split round-robin over $SLURM_CPUS_PER_TASK
# concurrent `helioswarm sweep --array-index i --array-count n` processes (one core each), then
# collated. Arguments: OUT_ROOT SWEEP_YAML [SWEEP_YAML ...]; each sweep goes to
# OUT_ROOT/<sweep name>. Used for the ERA5 comparison (configs/comparison/era5/*_sweep.yaml) and
# the L3 SRP sensitivity (configs/scenarios/sensitivity/srp_area_to_mass_sweep.yaml):
#   sbatch slurm/m8_sweeps.sbatch $HELIOSWARM_RUNS_DIR/comparison_era5 configs/comparison/era5/*_sweep.yaml
#SBATCH -J helioswarm-m8sweeps
#SBATCH -p test
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH -t 0-02:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.err

source "${HELIOSWARM_REPO:-/n/home05/ianpoon/Project-Dyson-Sphere}/slurm/common.sh"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT="${1:?usage: sbatch slurm/m8_sweeps.sbatch OUT_ROOT SWEEP_YAML [...]}"
shift
N="${SLURM_CPUS_PER_TASK:-1}"
status=0
for SWEEP in "$@"; do
    NAME=$(uv run --locked python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$SWEEP")
    OUT="$ROOT/$NAME"
    echo "sweep $SWEEP -> $OUT ($N processes) $(date -u +%FT%TZ)"
    pids=()
    for ((i = 0; i < N; i++)); do
        uv run --locked helioswarm sweep "$SWEEP" --out "$OUT" --array-index "$i" \
            --array-count "$N" --summary-only > "$TMP/${NAME}_$i.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    cat "$TMP/${NAME}"_*.log
    uv run --locked helioswarm sweep "$SWEEP" --out "$OUT" --collate-only || status=1
done
exit $status
