#!/usr/bin/env bash
# Progress at a glance: running jobs, last log lines, GPU samples, results present.
source "$(dirname "$0")/env.sh"
for pid in "$LOGS"/*.pid; do [ -e "$pid" ] && { n=$(basename "$pid" .pid); kill -0 "$(cat "$pid")" 2>/dev/null && echo "running: $n" || echo "stopped: $n"; }; done
for f in "$LOGS"/*.log "$LOGS"/*/*.log; do [ -e "$f" ] && { echo "== $f"; tail -n 3 "$f"; }; done
for g in $(ls -t "$ROOT"/runs/*/gpu.csv "$LOGS"/*/gpu.csv 2>/dev/null | head -2); do echo "== $g"; tail -n 3 "$g"; done
ls "$ROOT"/results 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
