#!/usr/bin/env bash
# Phase 9.5 — textured 45-tree apple orchard -> USD -> Isaac Sim, end to end.
#
# Producer env `helios` (no pxr) builds fully-grown trees and writes a texture-accurate,
# instanced orchard.usda. Consumer env `env_isaaclab` (Isaac Sim 5.1) opens it, round-trip
# validates every instance, confirms the bark/leaf/fruit textures resolve, and renders it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
HELIOS_PY="$HOME/anaconda3/envs/helios/bin/python"
ISAAC_PY="$HOME/anaconda3/envs/env_isaaclab/bin/python"

ROWS="${1:-3}"
PER_ROW="${2:-15}"
AGE_DAYS="${3:-1825}"      # fully grown (~4.4 m)
IN_ROW="${4:-4.5}"         # m, semi-dwarf apple in-row spacing
ROW_SPACING="${5:-6.0}"    # m, tractor alley between rows
N_UNIQUE="${6:-3}"         # distinct prototype trees to instance

echo "=== Build textured fully-grown orchard in PyHelios (helios env) ==="
cd "$HERE"
PYTHONPATH="$HERE:$REPO" "$HELIOS_PY" orchard_export.py \
    --rows "$ROWS" --per-row "$PER_ROW" --age-days "$AGE_DAYS" \
    --in-row-spacing "$IN_ROW" --row-spacing "$ROW_SPACING" \
    --n-unique "$N_UNIQUE" 2>&1 \
    | grep -vE "Advancing time|BVH not cached"

echo
echo "=== Open + validate + render the orchard in Isaac Sim (env_isaaclab) ==="
"$ISAAC_PY" open_in_isaacsim.py \
    --usd "$HERE/output/orchard.usda" \
    --report "$HERE/output/orchard_export_report.json" \
    --screenshot "$HERE/output/isaacsim_orchard.png" \
    --render-frames 120 --width 1600 --height 900 \
    --cam-x 0.28 --cam-y -0.52 --cam-z 0.10 2>&1 \
    | grep -vE "^\[[0-9]|deprecated" | tail -40

echo
echo "Artifacts in $HERE/output:"
ls -la "$HERE/output/orchard.usda" "$HERE/output"/orchard_*.json "$HERE/output"/rgb_*.png 2>/dev/null || true
