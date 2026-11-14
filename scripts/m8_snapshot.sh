#!/bin/bash
# Freeze the package source, configurations and kernel manifest for M8 Slurm studies:
#   SNAP=$(scripts/m8_snapshot.sh)            # prints the snapshot directory
#   HELIOSWARM_SRC_SNAPSHOT=$SNAP sbatch --export=ALL slurm/m8_optimize.sbatch $SNAP/configs/optimization.yaml ...
# slurm/common.sh puts $SNAP/src first on PYTHONPATH (the uv environment of the checkout supplies
# the dependencies); every run manifest then records the snapshot's source-tree SHA-256.
# SNAPSHOT.json records the checkout's git commit, dirty flag, the SHA-256 of `git diff HEAD`
# and the source-tree SHA-256 (helioswarm.manifest.source_tree_sha256 of the copy).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source scripts/cluster_env.sh
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SNAP="/n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/m8_snapshots/$STAMP"
mkdir -p "$SNAP/data/kernels"
cp -r src configs "$SNAP/"
cp data/kernels/kernels.json "$SNAP/data/kernels/"
find "$SNAP/src" -name __pycache__ -prune -exec rm -rf {} +
DIFF_SHA=$(git diff HEAD | sha256sum | cut -d' ' -f1)
TREE_SHA=$(PYTHONPATH="$SNAP/src" uv run --locked python -c \
    "from helioswarm.manifest import source_tree_sha256; print(source_tree_sha256())")
cat > "$SNAP/SNAPSHOT.json" <<JSON
{"created_utc": "$STAMP", "repo": "$REPO", "git_commit": "$(git rev-parse HEAD)",
 "git_dirty": $([ -n "$(git status --porcelain)" ] && echo true || echo false),
 "git_diff_head_sha256": "$DIFF_SHA", "source_tree_sha256": "$TREE_SHA"}
JSON
chmod -R a-w "$SNAP"
echo "$SNAP"
