#!/bin/bash -l
# Programmatic model validation (PDF §7.4, §8.5, §9, §10.5, §13.1-13.3; 37 checks):
#   sbatch slurm/validate_models.sbatch [REPORT]      (default docs/validation_report.md)
# Writes the generated block of the report in the repository (a small text file) and a copy
# into $HELIOSWARM_RUNS_DIR/validate_models/<jobid>.md. Sizing (measured): job 60118119 (31
# checks) 52 s, peak RSS 1.10 GB; job 60127176 (37 checks, after the M6 review) 44 s command /
# 80 s job wall, peak RSS 1.09 GB (/usr/bin/time) -> 1 core, 1200 MB, 2 min.
#SBATCH -J helioswarm-validate
#SBATCH -p test
#SBATCH -c 1
#SBATCH --mem=1200M
#SBATCH -t 0-00:02
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.err

source "${HELIOSWARM_REPO:-/n/home05/ianpoon/Project-Dyson-Sphere}/slurm/common.sh"
export JAX_PLATFORMS=cpu
REPORT="${1:-docs/validation_report.md}"
mkdir -p "$HELIOSWARM_RUNS_DIR/validate_models"
status=0
/usr/bin/time -v uv run --locked helioswarm validate-models --report "$REPORT" || status=$?
cp "$REPORT" "$HELIOSWARM_RUNS_DIR/validate_models/${SLURM_JOB_ID}.md"
exit "$status"
