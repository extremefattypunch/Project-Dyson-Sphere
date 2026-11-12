#!/bin/bash -l
# One end-to-end run:  sbatch slurm/run.sbatch CONFIG [extra `helioswarm run` args]
# Output: $HELIOSWARM_RUNS_DIR/<config stem>/ (refuses to overwrite; pass --overwrite).
# Sizing (measured, docs/milestones/M6.md D; jobs 60110148/51/54): one year at 3600 s takes
# 52-67 s job wall, peak RSS 359 MB (optical) / 464 MB (microwave, ITU-R maps), single-threaded
# -> test partition (free, <12 h), 1 core, 512 MB, 2 min. Longer windows or finer steps scale the
# time roughly linearly (orbit integration ~86 % of a run); raise -t accordingly. Use -p shared for
# runs that need more than 12 h. Logs: create the directory first
#   mkdir -p /n/netscratch/hekstra_lab/Lab/ian_poon/logs
#SBATCH -J helioswarm-run
#SBATCH -p test
#SBATCH -c 1
#SBATCH --mem=512M
#SBATCH -t 0-00:02
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.err

source "${HELIOSWARM_REPO:-/n/home05/ianpoon/Project-Dyson-Sphere}/slurm/common.sh"
export JAX_PLATFORMS=cpu
CONFIG="${1:?usage: sbatch slurm/run.sbatch CONFIG [helioswarm run args]}"
shift
NAME="$(basename "$CONFIG" .yaml)"
/usr/bin/time -v uv run --locked helioswarm run "$CONFIG" --out "$HELIOSWARM_RUNS_DIR/$NAME" "$@"
