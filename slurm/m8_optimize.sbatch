#!/bin/bash -l
# M8 design optimisation (helioswarm.optimization): one NSGA-II seed per array task, the
# population of each generation evaluated by $SLURM_CPUS_PER_TASK worker processes (spawn; each
# chunk loads the SPICE kernels once). Then a dependent/serial collation and the high-fidelity
# re-evaluation:
#   STUDY=configs/optimization.yaml OUT=$HELIOSWARM_RUNS_DIR/optimization
#   JID=$(sbatch --parsable --array=0-3 slurm/m8_optimize.sbatch "$STUDY" "$OUT")       # seeds
#   sbatch --export=ALL,MODE=collate     -c 1 --mem=2G -t 0-00:10 slurm/m8_optimize.sbatch "$STUDY" "$OUT"
#   sbatch --export=ALL,MODE=reevaluate  slurm/m8_optimize.sbatch "$STUDY" "$OUT"
#   sbatch --export=ALL,MODE=sanity      slurm/m8_optimize.sbatch "$STUDY" "$OUT"
#   sbatch --export=ALL,MODE=all         slurm/m8_optimize.sbatch "$STUDY" "$OUT"   # all seeds + collate
# (test partition QOS: 5 submitted jobs per user, array tasks counted individually; submit the
# collation once the seeds have left the queue.) Sizing: see docs/milestones/M8.md D (measured
# with seff for every job): requests below are measured peak + margin.
#SBATCH -J helioswarm-m8opt
#SBATCH -p test
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH -t 0-02:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%A_%a.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%A_%a.err

source "${HELIOSWARM_REPO:-/n/home05/ianpoon/Project-Dyson-Sphere}/slurm/common.sh"
export JAX_PLATFORMS=cpu
# Worker processes are single-threaded numpy: one BLAS thread each (common.sh sets the job-wide
# value to the core count, which would oversubscribe 8 workers x 8 threads).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STUDY="${1:?usage: sbatch slurm/m8_optimize.sbatch STUDY_YAML OUT_DIR}"
OUT="${2:?usage: sbatch slurm/m8_optimize.sbatch STUDY_YAML OUT_DIR}"
shift 2
W="${SLURM_CPUS_PER_TASK:-1}"
case "${MODE:-seed}" in
    seed)
        /usr/bin/time -v uv run --locked helioswarm optimize "$STUDY" --out "$OUT" \
            --array-index "${SLURM_ARRAY_TASK_ID:?submit with --array (one task per seed)}" \
            --workers "$W" "$@" ;;
    all)  # every configured seed in turn, then the collation (small studies: one QOS slot)
        /usr/bin/time -v uv run --locked helioswarm optimize "$STUDY" --out "$OUT" \
            --workers "$W" "$@" ;;
    collate)
        uv run --locked helioswarm optimize "$STUDY" --out "$OUT" --collate ;;
    reevaluate)
        /usr/bin/time -v uv run --locked helioswarm optimize "$STUDY" --out "$OUT" --reevaluate \
            --workers "$W" "$@" ;;
    sanity)
        /usr/bin/time -v uv run --locked helioswarm optimize "$STUDY" --out "$OUT" --sanity \
            --workers "$W" "$@" ;;
    *) echo "unknown MODE=$MODE" >&2; exit 2 ;;
esac
