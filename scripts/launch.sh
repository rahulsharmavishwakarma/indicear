#!/usr/bin/env bash
# Run a script in the background under nohup, logging to $LOGS/<name>.log; survives SSH and
# notebook disconnects. WAIT_FOR="file ..." makes the job wait until those files exist (e.g.
# results/v1.json), so dependent steps can be queued at once. A waited-for file ending in DONE
# whose sibling ...FAILED appears instead (runs/<v>/DONE vs FAILED, pilot/<stage>.DONE vs .FAILED)
# aborts the wait with exit 1, so a failed step does not leave its successors waiting forever.
# WAIT_POLL sets the poll interval in seconds (default 60).
#   WAIT_FOR="$ROOT/results/v1.json" bash scripts/launch.sh robustness scripts/robustness.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
name="$1"; shift
script=$(printf '%q ' "$@")
wait_loop=""
poll="${WAIT_POLL:-60}"
for f in ${WAIT_FOR:-}; do
  failed=""
  [[ "$f" == *DONE ]] && failed="${f%DONE}FAILED"
  check=""
  [ -n "$failed" ] && check="if [ -e $(printf '%q' "$failed") ]; then echo \"abort: $failed\"; exit 1; fi; "
  wait_loop+="until [ -e $(printf '%q' "$f") ]; do ${check}sleep $poll; done; echo \"ready: $f\"; "
done
nohup bash -c "${wait_loop}bash $script" >> "$LOGS/$name.log" 2>&1 &
echo $! > "$LOGS/$name.pid"
echo "started $name (pid $!)${WAIT_FOR:+, waiting for: $WAIT_FOR}; log: $LOGS/$name.log"
