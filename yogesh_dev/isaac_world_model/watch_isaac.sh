#!/usr/bin/env bash
# Watches the detached pipeline log and sends EXACTLY ONE Slack message on
# completion, failure, or 16 h timeout.
set -u
cd /home/yogesh/PyHelios
LOG=yogesh_dev/isaac_world_model/isaac_pipeline.log
MSG_FILE=yogesh_dev/world_model/output/isaac_r1_w6/slack_msg.txt
CH=slack; TARGET='C0BM19894JJ'; THREAD='1785484527.424139'

send() {
  openclaw message send --channel "$CH" --target "$TARGET" \
    --message "$1" --account default --thread-id "$THREAD"
}

deadline=$(( $(date +%s) + 57600 ))   # 16 h
while true; do
  if grep -q "PIPELINE_DONE" "$LOG" 2>/dev/null; then
    msg=$(cat "$MSG_FILE" 2>/dev/null \
      || echo "Isaac world-model pipeline finished, but the summary file is missing -- see $LOG")
    send "$msg"
    exit 0
  fi
  if grep -q "PIPELINE_FAIL" "$LOG" 2>/dev/null; then
    reason=$(grep -m1 "PIPELINE_FAIL" "$LOG" | cut -c1-500)
    send "Isaac world-model pipeline FAILED: $reason"
    exit 0
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    last=$(tail -3 "$LOG" 2>/dev/null | tr '\n' ' ' | cut -c1-400)
    send "Isaac world-model pipeline TIMEOUT after 16h. Last log: $last"
    exit 0
  fi
  sleep 60
done
