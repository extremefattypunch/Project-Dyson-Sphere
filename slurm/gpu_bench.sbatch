#!/bin/bash -l
# CPU vs GPU benchmark of the batched capture kernel + the real-device JAX test:
#   sbatch slurm/gpu_bench.sbatch
# gpu_test (free, <= 12 h, <= 8 cores per GPU), exactly 1 GPU (MIG slice). The GPU is used
# only by the JAX kernel; check utilisation afterwards with `seff` and `jobstats`. Results:
# $HELIOSWARM_RUNS_DIR/gpu_bench/<jobid>.json. Sizing (job 60108172, measured): 6 min 47 s wall,
# peak RSS 2.6 GB host / 1.1 GB GPU memory, CPU efficiency 24 % of 4 cores (the numpy reference is
# single-threaded) -> 1 core, 3 GB (peak + 10 %, rounded), 8 min.
#SBATCH -J helioswarm-gpubench
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=3G
#SBATCH -t 0-00:08
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.err

source "${HELIOSWARM_REPO:-/n/home05/ianpoon/Project-Dyson-Sphere}/slurm/common.sh"
export JAX_PLATFORMS=cuda
# Keep the JAX compilation cache off and XLA from pre-allocating 75 % of the slice.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
mkdir -p "$HELIOSWARM_RUNS_DIR/gpu_bench"
uv run --locked --extra gpu pytest -q -m gpu tests/unit/test_capture_batch.py -rs
/usr/bin/time -v uv run --locked --extra gpu python scripts/bench_capture.py \
    "$HELIOSWARM_RUNS_DIR/gpu_bench/${SLURM_JOB_ID}.json" --sizes 1000 10000 100000
