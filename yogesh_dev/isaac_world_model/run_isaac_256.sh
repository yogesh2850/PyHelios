#!/usr/bin/env bash
# Isaac world-model 256px redo: wait for isaac_r1 -> capture 512->256 ->
# train isaac_r2 -> W6 eval -> full-res recon comparison -> summary.
#
# Launched DETACHED (setsid nohup); watch_isaac_256.sh greps this script's log
# for PIPELINE_DONE / PIPELINE_FAIL and sends the single Slack message.
set -u
cd /home/yogesh/PyHelios

ISAAC=/home/yogesh/isaacsim/python.sh
GSPLAT=/home/yogesh/anaconda3/envs/gsplat/bin/python
IWM=yogesh_dev/isaac_world_model
WM=yogesh_dev/world_model
DATA=$WM/output/dataset_isaac_256
W6OUT=$WM/output/isaac_r2_w6
SCENE=yogesh_dev/phase9/output/orchard_scene.usda
TRAIN_LOG=$WM/output/train_isaac_r2_stdout.txt

say()  { echo "[$(date +%H:%M:%S)] $*"; }
fail() { say "PIPELINE_FAIL: $*"; exit 1; }

say "=== step 0/8: wait for the isaac_r1 (128px) run to release the GPU ==="
# [.] not . so the pattern cannot match a process whose cmdline contains the
# pattern text itself. Also waits out run_isaac.sh's own eval/summarize steps
# (run_isaac[.]sh does not match run_isaac_256.sh).
while pgrep -f "world_model[.]train .*--tag isaac_r1" >/dev/null \
   || pgrep -f "isaac_world_model/run_isaac[.]sh" >/dev/null; do
  sleep 60
done
say "isaac_r1 pipeline finished; GPU free for the 256px run"

[ -f "$SCENE" ] || fail "scene $SCENE missing (same scene as the 128px dataset is reused on purpose)"

say "=== step 1/8: smoke validation at 256 (render 512, 48 subframes) ==="
$ISAAC $IWM/capture.py --smoke --render-res 512 --out-res 256 --rt-subframes 48 \
  --out "$DATA" || fail "256 smoke validation (see $DATA/smoke/smoke_report.json)"

say "=== step 2/8: full capture (48 episodes x 32 steps, 512 -> 256) ==="
$ISAAC $IWM/capture.py --render-res 512 --out-res 256 --rt-subframes 48 \
  --out "$DATA" || fail "capture"

say "=== step 3/8: dataset sanity gate at 256 ==="
$GSPLAT $IWM/check_dataset.py --root "$DATA" --res 256 || fail "dataset check"

say "=== step 4/8: probe largest batch that fits next to the skrl job ==="
$GSPLAT $IWM/probe_batch.py --image-size 256 > "$WM/output/probe_batch256.txt" 2>&1 \
  || fail "batch probe (tail: $(tail -3 "$WM/output/probe_batch256.txt" | tr '\n' ' '))"
BS=$(grep -oP 'PROBE_BATCH \K[0-9]+' "$WM/output/probe_batch256.txt" | tail -1)
[ -n "$BS" ] || fail "batch probe produced no PROBE_BATCH line"
say "probe picked batch size $BS ($(grep 'free VRAM' "$WM/output/probe_batch256.txt" | head -1))"

say "=== step 5/8: train isaac_r2 (isaac_r1 recipe at 256) ==="
try_train() {
  $GSPLAT -m yogesh_dev.world_model.train --data "$DATA" --steps 30000 \
    --image-size 256 --batch-size "$1" --seq-len 32 \
    --growth-fraction 0.0 \
    --free-bits 12.0 --kl-dyn 0.2 --kl-rep 0.04 \
    --sem-class-weights auto --depth-loss l1 \
    --log-every 250 --val-every 500 --val-batches 8 --ckpt-every 2000 \
    --tag isaac_r2 > "$TRAIN_LOG" 2>&1
}
attempt=$BS
while :; do
  say "training isaac_r2 at batch $attempt"
  if try_train "$attempt"; then break; fi
  if grep -qE "CUDA out of memory|OutOfMemoryError" "$TRAIN_LOG"; then
    # The probe measured free VRAM at probe time; the skrl job's usage moves.
    [ "$attempt" -le 1 ] && fail "training OOM even at batch 1"
    say "OOM at batch $attempt; restarting fresh at batch $((attempt / 2))"
    attempt=$((attempt / 2))
    rm -rf "$WM/output/train/isaac_r2"
  else
    fail "training (tail: $(tail -3 "$TRAIN_LOG" | tr '\n' ' '))"
  fi
done
say "training done: $(grep -c '' "$TRAIN_LOG") log lines"

say "=== step 6/8: W6 evaluation + rollout strips (held-out test) ==="
mkdir -p "$W6OUT"
$GSPLAT -m yogesh_dev.world_model.evaluate \
  --ckpt "$WM/output/train/isaac_r2/ckpt_best.pt" \
  --data "$DATA" --split test --context 5 --seq-len 32 \
  --batch-size 8 --n-batches 8 --qualitative 4 --out "$W6OUT" \
  > "$W6OUT/w6_stdout.txt" 2>&1 \
  || fail "evaluation (tail: $(tail -3 "$W6OUT/w6_stdout.txt" | tr '\n' ' '))"

say "=== step 7/8: full-res GT-vs-recon comparison (r2 vs r1) ==="
$GSPLAT $IWM/recon_compare.py > "$W6OUT/recon_compare_stdout.txt" 2>&1 \
  || fail "recon compare (tail: $(tail -3 "$W6OUT/recon_compare_stdout.txt" | tr '\n' ' '))"

say "=== step 8/8: summary + Slack message file ==="
$GSPLAT $IWM/summarize_256.py || fail "summarize_256"

say "PIPELINE_DONE"
