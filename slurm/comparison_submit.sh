#!/bin/bash
# Submit the M7b harmonised comparison (configs/comparison/*_sweep.yaml) and the bounded
# GEO-optical flux-limit sensitivity (configs/comparison/sensitivity/) as Slurm job arrays and
# print the job ids. Run from the repository after `source scripts/cluster_env.sh`:
#   bash slurm/comparison_submit.sh            # -> $HELIOSWARM_RUNS_DIR/comparison/<sweep name>
# then, after the arrays have finished (in the session, seconds):
#   uv run helioswarm compare configs/comparison/comparison.yaml \
#       --runs-dir $HELIOSWARM_RUNS_DIR/comparison --out $HELIOSWARM_RUNS_DIR/comparison/report
# The sweeps are resumable: re-running this script re-submits every task, and each task skips
# the points that already have a summary (or a recorded failure) for the same configuration.
#
# Sizing, from the measured M7b arrays (docs/milestones/M7b.md D; seff/sacct, 1 core each;
# BLAS threads give no speed-up, slurm/common.sh pins them to the allocation). Memory = measured
# peak RSS + ~15 %, time = longest measured task + 20-50 % (Job Defense Shield: memory > 80 %
# of the request, elapsed > 50 % of the limit for the longest task):
#   geo_optical       per point 0.5 min (the 10 km receivers fail the Fresnel paraxial check
#                     at once) to 15 min (p0004), and 1.8-2.5 h for p0007 (D = 100 m, R_rx = 1 km:
#                     ~34 Fresnel-Hankel evaluations of 268 s); peak RSS <= 657 MB.
#                     -> one point per task; p0007 alone with 3 h, the rest 30 min; 750M.
#   helio_1au_optical per point 1.5-20.5 min (the 10 km receivers: per-epoch Airy reference
#                     over a horizontal receiver; p0008 20:13); peak RSS <= 624 MB.
#                     -> one point per task, 720M, 30 min (the first submission as 3 tasks
#                     of 3 points with 15 min timed out).
#   helio_1au_microwave 3 points per task 5:50-8:17; peak RSS 787-850 MB (ITU-R maps; seff
#                     781 MB of an 800 MB request). -> 3 tasks of 3 points, 960M, 12 min.
#   geo_microwave     3 points per task 1:44-2:12; peak RSS 737-771 MB. -> 3 tasks, 880M, 5 min.
#   sensitivity       3 GEO-optical points with a 100 m receiver (D = 100 m: ~34 evaluations of
#                     12.5 s, ~7 min). -> one point per task, 750M, 15 min.
# All on the shared partition (the test partition's QOS allows 5 submitted array tasks per
# user). A task exits 1 when one of its points fails (recorded as error.json): sacct then shows
# FAILED for the expected Fresnel-paraxial failures of the three GEO-optical 10 km points.
set -euo pipefail
OUT="${HELIOSWARM_RUNS_DIR:?source scripts/cluster_env.sh first}/comparison"
mkdir -p "$OUT" /n/netscratch/hekstra_lab/Lab/ian_poon/logs
submit() {  # name array mem time sweep-path (relative to configs/comparison) [extra sbatch args]
    local base
    base="$(basename "$5")"
    sbatch --parsable -J "helioswarm-cmp-$1" -p shared --array="$2" -c 1 --mem="$3" -t "$4" \
        "${@:6}" slurm/sweep_array.sbatch "configs/comparison/$5" "$OUT/cmp_${base%_sweep.yaml}"
}
# One point per task: SWEEP_ARRAY_COUNT=9 keeps the 9-way split when only some indices are
# submitted (helioswarm.sweep: task i runs the points p with p mod 9 == i).
NINE=(--export=ALL,SWEEP_ARRAY_COUNT=9)
echo "geo_optical $(submit geoopt 0-6,8 750M 0-00:30 geo_optical_sweep.yaml "${NINE[@]}")"
echo "geo_optical p0007 $(submit geoopt 7 750M 0-03:00 geo_optical_sweep.yaml "${NINE[@]}")"
echo "helio_1au_optical $(submit helioopt 0-8 720M 0-00:30 helio_1au_optical_sweep.yaml)"
echo "helio_1au_microwave $(submit heliomw 0-2 960M 0-00:12 helio_1au_microwave_sweep.yaml)"
echo "geo_microwave $(submit geomw 0-2 880M 0-00:05 geo_microwave_sweep.yaml)"
echo "geo_optical_flux_limit $(submit geofluxlim 0-2 750M 0-00:15 sensitivity/geo_optical_flux_limit_sweep.yaml)"
