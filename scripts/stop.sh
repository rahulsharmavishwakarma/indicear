#!/usr/bin/env bash
# Ask a launched job to stop: SIGTERM makes training write a final checkpoint first.
source "$(dirname "$0")/env.sh"
pid=$(cat "$LOGS/$1.pid"); pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; echo "sent SIGTERM to $1 ($pid)"
