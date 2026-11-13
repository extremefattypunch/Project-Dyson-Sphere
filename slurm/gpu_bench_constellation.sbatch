#!/bin/bash -l
# CPU vs GPU benchmark of the batched capture kernel on the N = 1000 constellation workload
# (the points with geometric access in the stage-A link files; scripts/bench_constellation_capture.py):
#   sbatch slurm/gpu_bench_constellation.sbatch configs/constellation_demo_n1000.yaml \
#          $HELIOSWARM_RUNS_DIR/constellation_demo_n1000_links
# gpu_test (free, <= 12 h, <= 8 cores per GPU, 2 jobs per user), exactly 1 GPU (MIG slice). The
# GPU is used only by the JAX kernel; check utilisation afterwards with `seff` and `jobstats`.
# Results: $HELIOSWARM_RUNS_DIR/gpu_bench/constellation_<jobid>.json. Sizing (measured, job
# 60286071 with --sizes 10000 100000 1000000; job 60284257 ran out of memory at 8G): 11 min 55 s
# wall, peak RSS 9.85 GB host (the numpy batch of 1e6 points), 2.0 GB GPU memory, CPU efficiency
# 96 % -> 1 core, 11G (peak + 12 %), 15 min.
#SBATCH -J helioswarm-cgpubench
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=11G
#SBATCH -t 0-00:15
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.err

source "${HELIOSWARM_REPO:-/n/home05/ianpoon/Project-Dyson-Sphere}/slurm/common.sh"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
CONFIG="${1:?usage: sbatch slurm/gpu_bench_constellation.sbatch CONFIG CANDIDATES_DIR [--sizes ...]}"
CAND="${2:?usage: sbatch slurm/gpu_bench_constellation.sbatch CONFIG CANDIDATES_DIR [--sizes ...]}"
shift 2
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
mkdir -p "$HELIOSWARM_RUNS_DIR/gpu_bench"
/usr/bin/time -v uv run --locked --extra gpu python scripts/bench_constellation_capture.py \
    "$CONFIG" "$CAND" "$HELIOSWARM_RUNS_DIR/gpu_bench/constellation_${SLURM_JOB_ID}.json" "$@"
