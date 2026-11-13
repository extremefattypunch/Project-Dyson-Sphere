# shellcheck shell=bash
# Source this file (do not execute it) on the Harvard FASRC Cannon cluster:
#
#     source scripts/cluster_env.sh
#
# It points uv, pre-commit, matplotlib and helioswarm at the Hekstra-lab storage tiers:
#   - interpreter + project venv on holylabs (software tier; never in /n/home05),
#   - regenerable caches, SPICE kernels and run outputs on netscratch (90-day purge).
# Outside the cluster none of this is needed: `uv sync --locked --extra dev --extra xcheck
# --extra itur --extra jax` (as in the README; not --all-extras, which pulls ~4.4 GB of CUDA
# wheels) creates a local .venv and helioswarm falls back to the repo-relative data/kernels/
# and runs/ directories.

export MAMBA_ROOT_PREFIX=/n/holylabs/hekstra_lab/Everyone/ianpoon/micromamba
export HELIOSWARM_BASE=/n/holylabs/hekstra_lab/Everyone/ianpoon/envs/helioswarm-base
export PATH="$HELIOSWARM_BASE/bin:$PATH"

export UV_PROJECT_ENVIRONMENT=/n/holylabs/hekstra_lab/Everyone/ianpoon/envs/helioswarm
export UV_PYTHON="$HELIOSWARM_BASE/bin/python"
export UV_PYTHON_DOWNLOADS=never
export UV_LINK_MODE=copy
export UV_CACHE_DIR=/n/netscratch/hekstra_lab/Lab/ian_poon/cache/uv
export PRE_COMMIT_HOME=/n/netscratch/hekstra_lab/Lab/ian_poon/cache/pre-commit

export HELIOSWARM_KERNEL_DIR=/n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/kernels
export HELIOSWARM_RUNS_DIR=/n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/runs
export MPLCONFIGDIR=/n/netscratch/hekstra_lab/Lab/ian_poon/cache/matplotlib

# Tool caches that would otherwise be written inside the repository (i.e. under /n/home05).
# These are additions to the required list above; all are regenerable.
# mypy's cache is per checkout (keyed by the checkout's absolute path): two clones must not
# share one incremental cache (M6b: a fresh-clone job that shared the cache with the main
# checkout hit a one-off mypy INTERNAL ERROR; not reproduced, kept apart as a precaution).
_helioswarm_checkout="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export MYPY_CACHE_DIR="/n/netscratch/hekstra_lab/Lab/ian_poon/cache/mypy/$(printf '%s' "$_helioswarm_checkout" | sha256sum | cut -c1-16)"
unset _helioswarm_checkout
export RUFF_CACHE_DIR=/n/netscratch/hekstra_lab/Lab/ian_poon/cache/ruff
export PYTHONPYCACHEPREFIX=/n/netscratch/hekstra_lab/Lab/ian_poon/cache/pycache
export PYTEST_ADDOPTS="-o cache_dir=/n/netscratch/hekstra_lab/Lab/ian_poon/cache/pytest"
# virtualenv (used by pre-commit to build hook environments) otherwise caches its seed wheels
# and interpreter info in ~/.cache/virtualenv.
export VIRTUALENV_OVERRIDE_APP_DATA=/n/netscratch/hekstra_lab/Lab/ian_poon/cache/virtualenv
