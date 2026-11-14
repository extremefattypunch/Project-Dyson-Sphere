#!/bin/bash -l
# M8 GPU benchmark (gpu_test, free; <= 8 cores per GPU; 2 jobs per user): screening evaluations
# of configs/optimization.yaml with the batched capture kernel on the CPU (numpy) and on the GPU
# (JAX), per design and per stage (scripts/bench_m8_evaluation.py). The CUDA build is installed
# inside the job only (`--extra gpu`). Record `seff` and `jobstats` afterwards (GPU utilisation).
#   sbatch slurm/m8_gpu_bench.sbatch [STUDY] [N]
#SBATCH -J helioswarm-m8gpu
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=6G
#SBATCH -t 0-00:45
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.err

source "${HELIOSWARM_REPO:-/n/home05/ianpoon/Project-Dyson-Sphere}/slurm/common.sh"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
mkdir -p "$HELIOSWARM_RUNS_DIR/gpu_bench_m8"
STUDY="${1:-configs/optimization.yaml}"
N="${2:-8}"
/usr/bin/time -v uv run --locked --extra gpu python scripts/bench_m8_evaluation.py \
    "$HELIOSWARM_RUNS_DIR/gpu_bench_m8/${SLURM_JOB_ID}.json" --study "$STUDY" --n "$N"
