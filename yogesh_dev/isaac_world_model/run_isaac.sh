#!/usr/bin/env bash
# Isaac world-model pipeline: scene -> smoke -> capture -> train -> eval.
#
# Designed to be launched DETACHED (setsid nohup) so it survives the worker
# process exiting -- Round 3's training was SIGHUP-killed exactly this way.
# Emits PIPELINE_DONE / "PIPELINE_FAIL: <step>" markers that watch_isaac.sh
# greps to send the single Slack completion/failure message.
set -u
cd /home/yogesh/PyHelios

ISAAC=/home/yogesh/isaacsim/python.sh
GSPLAT=/home/yogesh/anaconda3/envs/gsplat/bin/python
IWM=yogesh_dev/isaac_world_model
WM=yogesh_dev/world_model
DATA=$WM/output/dataset_isaac
W6OUT=$WM/output/isaac_r1_w6

say()  { echo "[$(date +%H:%M:%S)] $*"; }
fail() { say "PIPELINE_FAIL: $*"; exit 1; }

say "=== step 1/7: rebuild scene (orchard_scene.usda) ==="
$ISAAC $IWM/build_scene.py || fail "build_scene"

say "=== step 2/7: smoke validation ==="
$ISAAC $IWM/capture.py --smoke || fail "smoke validation (see smoke_report.json)"

say "=== step 3/7: full capture (48 episodes x 32 steps) ==="
$ISAAC $IWM/capture.py || fail "capture"

say "=== step 4/7: dataset sanity gate (gsplat env, data.py loader) ==="
$GSPLAT $IWM/check_dataset.py || fail "dataset check"

say "=== step 5/7: train isaac_r1 (r3_final recipe, static/view-only) ==="
# r3_final recipe minus the growth knobs: this dataset is view-only, and
# SequenceSampler already forces growth_fraction to 0 when no growth episodes
# exist. Action dim stays 5 (a_view 4 + a_grow/10). 30k steps: the dataset is
# ~1.3k frames vs Helios's ~135k, so best-ckpt-on-val does the stopping.
$GSPLAT -m yogesh_dev.world_model.train --data "$DATA" --steps 30000 \
  --image-size 128 --batch-size 24 --seq-len 32 \
  --growth-fraction 0.0 \
  --free-bits 12.0 --kl-dyn 0.2 --kl-rep 0.04 \
  --sem-class-weights auto --depth-loss l1 \
  --log-every 250 --val-every 500 --val-batches 8 --ckpt-every 2000 \
  --tag isaac_r1 \
  > "$WM/output/train_isaac_r1_stdout.txt" 2>&1 \
  || fail "training (tail: $(tail -3 "$WM/output/train_isaac_r1_stdout.txt" | tr '\n' ' '))"
say "training done: $(grep -c '' "$WM/output/train_isaac_r1_stdout.txt") log lines"

say "=== step 6/7: W6 evaluation + rollout strips (held-out test) ==="
mkdir -p "$W6OUT"
$GSPLAT -m yogesh_dev.world_model.evaluate \
  --ckpt "$WM/output/train/isaac_r1/ckpt_best.pt" \
  --data "$DATA" --split test --context 5 --seq-len 32 \
  --batch-size 8 --n-batches 8 --qualitative 4 --out "$W6OUT" \
  > "$W6OUT/w6_stdout.txt" 2>&1 \
  || fail "evaluation (tail: $(tail -3 "$W6OUT/w6_stdout.txt" | tr '\n' ' '))"

say "=== step 7/7: summary ==="
$GSPLAT $IWM/summarize.py || fail "summarize"

say "PIPELINE_DONE"
