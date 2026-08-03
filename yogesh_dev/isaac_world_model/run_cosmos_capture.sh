#!/usr/bin/env bash
# Cosmos-Transfer data capture from the phase9 orchard: wait for the 256px
# world-model pipeline to release the GPU -> smoke clip + validation ->
# full 14-clip capture -> output verification.
#
# Launched DETACHED (setsid nohup); watch_cosmos.sh greps this script's log
# for COSMOS_DONE / COSMOS_FAIL and sends the single Slack message.
set -u
cd /home/yogesh/PyHelios

ISAAC=/home/yogesh/isaacsim/python.sh
IWM=yogesh_dev/isaac_world_model
OUT=yogesh_dev/world_model/output/cosmos_isaac
SMOKE_OUT=yogesh_dev/world_model/output/cosmos_isaac_smoke
MSG_FILE=$IWM/cosmos_slack_msg.txt

say()  { echo "[$(date +%H:%M:%S)] $*"; }
fail() { say "COSMOS_FAIL: $*"; exit 1; }

say "=== step 0/4: wait for the isaac_r2 (256px) pipeline to release the GPU ==="
# Guard on the training process (as instructed) AND the parent pipeline script,
# because run_isaac_256.sh runs GPU-hungry eval/recon steps after training.
# [.] so the pattern cannot match this script's own cmdline text.
while pgrep -f "world_model[.]train .*--tag isaac_r2" >/dev/null \
   || pgrep -f "isaac_world_model/run_isaac_256[.]sh" >/dev/null; do
  sleep 60
done
say "isaac_r2 pipeline finished; GPU free"

[ -f yogesh_dev/phase9/output/orchard_scene.usda ] || fail "orchard_scene.usda missing"

say "=== step 1/4: smoke clip (12 frames) + validation ==="
rm -rf "$SMOKE_OUT"
$ISAAC $IWM/cosmos_capture.py --smoke --out "$SMOKE_OUT" \
  || fail "smoke validation (see $SMOKE_OUT/smoke_report.json)"

say "=== step 2/4: full capture (14 clips x 121 frames @ 1280x720) ==="
$ISAAC $IWM/cosmos_capture.py --out "$OUT" \
  || fail "full capture (see log above and $OUT/validate_clip_*.json)"

say "=== step 3/4: verify output counts ==="
N_CLIPS=$(find "$OUT" -maxdepth 1 -type d -name 'clip_*' | wc -l)
N_MP4=$(find "$OUT" -name '*.mp4' | wc -l)
N_RGB=$(find "$OUT" -path '*/rgb/*.png' | wc -l)
say "clips=$N_CLIPS mp4s=$N_MP4 rgb_pngs=$N_RGB"
[ "$N_CLIPS" -eq 14 ] || fail "expected 14 clip dirs, got $N_CLIPS"
[ "$N_RGB" -eq 1694 ] || fail "expected 1694 rgb frames (14x121), got $N_RGB"
[ "$N_MP4" -eq 70 ] || fail "expected 70 mp4s (14 clips x 5 modalities), got $N_MP4"

say "=== step 4/4: write Slack summary ==="
SIZE=$(du -sh "$OUT" | cut -f1)
cat > "$MSG_FILE" <<EOF
Cosmos-Transfer capture DONE: 14 clips x 121 frames @ 1280x720 (ZED X Mini 110deg HFOV, 1.2 m robot POV) from the phase9 orchard -> $OUT ($SIZE). 70 mp4s (rgb/segmentation/shaded_seg/depth/edges per clip) + PNG frames. Smoke + per-clip validation passed (soil/leaf/fruit/shoot/hills/sky all present). Next: feed rgb.mp4 + control videos to Cosmos-Transfer1 -- see yogesh_dev/isaac_world_model/COSMOS_DATA.md.
EOF

say "COSMOS_DONE"
