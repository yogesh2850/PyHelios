#!/usr/bin/env bash
# Watches the detached Cosmos capture pipeline log and sends EXACTLY ONE Slack
# message on completion, failure, silent death, or 48 h timeout (the pipeline
# first waits for the multi-hour isaac_r2 training+eval to release the GPU).
set -u
cd /home/yogesh/PyHelios
LOG=yogesh_dev/isaac_world_model/cosmos_capture.log
MSG_FILE=yogesh_dev/isaac_world_model/cosmos_slack_msg.txt
CH=slack; TARGET='C0BM19894JJ'; THREAD='1785484527.424139'

send() {
  openclaw message send --channel "$CH" --target "$TARGET" \
    --message "$1" --account default --thread-id "$THREAD"
}

deadline=$(( $(date +%s) + 172800 ))   # 48 h
dead_checks=0
while true; do
  if grep -q "COSMOS_DONE" "$LOG" 2>/dev/null; then
    msg=$(cat "$MSG_FILE" 2>/dev/null \
      || echo "Cosmos capture pipeline finished, but the summary file is missing -- see $LOG")
    send "$msg"
    exit 0
  fi
  if grep -q "COSMOS_FAIL" "$LOG" 2>/dev/null; then
    reason=$(grep -m1 "COSMOS_FAIL" "$LOG" | cut -c1-500)
    send "Cosmos-Transfer orchard capture FAILED: $reason (log: $LOG)"
    exit 0
  fi
  if ! pgrep -f "isaac_world_model/run_cosmos_capture[.]sh" >/dev/null; then
    dead_checks=$((dead_checks + 1))
    if [ "$dead_checks" -ge 3 ]; then
      last=$(tail -3 "$LOG" 2>/dev/null | tr '\n' ' ' | cut -c1-400)
      send "Cosmos-Transfer orchard capture DIED without a DONE/FAIL marker. Last log: $last"
      exit 0
    fi
  else
    dead_checks=0
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    last=$(tail -3 "$LOG" 2>/dev/null | tr '\n' ' ' | cut -c1-400)
    send "Cosmos-Transfer orchard capture TIMEOUT after 48h. Last log: $last"
    exit 0
  fi
  sleep 60
done
