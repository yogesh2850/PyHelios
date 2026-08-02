#!/usr/bin/env bash
# Phase 9 — PyHelios -> USD -> Isaac Sim bridge, end to end.
#
# Two envs on purpose: the producer (helios) has no pxr and doesn't need it; the consumer
# (env_isaaclab, Isaac Sim 5.1.0) is where the real Kit runtime + pxr live. The USD file is
# the contract between them.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
HELIOS_PY="$HOME/anaconda3/envs/helios/bin/python"
ISAAC_PY="$HOME/anaconda3/envs/env_isaaclab/bin/python"

N_TREES="${1:-1}"
AGE_DAYS="${2:-365}"
SEED="${3:-9001}"

echo "=== T9.1  Build tree in PyHelios + export USD (helios env) ==="
cd "$REPO"
PYTHONPATH="$REPO" "$HELIOS_PY" yogesh_dev/phase9/usd_export.py \
    --n-trees "$N_TREES" --age-days "$AGE_DAYS" --seed "$SEED" 2>&1 \
    | grep -vE "Advancing time|BVH not cached"

echo
echo "=== T9.3  Open the USD in a real Isaac Sim session (env_isaaclab) ==="
cd "$HERE"
"$ISAAC_PY" open_in_isaacsim.py --render-frames 90 2>&1 \
    | grep -vE "^\[[0-9]" | tail -40

echo
echo "Artifacts in $HERE/output:"
ls -la "$HERE/output"/*.usda "$HERE/output"/*.json "$HERE/output"/rgb_*.png 2>/dev/null || true
