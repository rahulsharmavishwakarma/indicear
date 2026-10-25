#!/usr/bin/env bash
# Pull a run's checkpoints and results from the VM to this machine (the primary off-VM copy).
#
#   bash scripts/sync_runs.sh [--once] [--interval 90] [--keep-all] [--drive-copy DIR] SESSION REMOTE_RUN_DIR [LOCAL_DEST]
#
# Every poll reads the remote LATEST.json and pulls ONLY the files it lists as complete, each to a
# temporary name; the sha256 is verified before the file is moved into place, and the local
# LATEST.json advances only when every listed file verified. Files the manifest does not list
# (a save in progress) are never pulled. Keeps the last 2 checkpoints plus best.pt (--keep-all: every
# checkpoint it pulled, e.g. for averaging or an open-weights release); logs, CSVs,
# command.sh and DONE/FAILED are pulled best effort. Exits after a final pull once the run
# writes DONE or FAILED. LOCAL_DEST defaults to ~/indicear-runs/<run>/ (outside any repo).
set -euo pipefail
source "$(dirname "$0")/lib/transport.sh"

once=0
interval=90
drive_copy=""
keep_all=0
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --once) once=1; shift ;;
    --interval) interval="$2"; shift 2 ;;
    --drive-copy) drive_copy="$2"; shift 2 ;;
    --keep-all) keep_all=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ $# -ge 2 ]] || { sed -n '2,12p' "$0"; exit 2; }
session="$1"
remote="${2%/}"
run="$(basename "$remote")"
dest="${3:-$HOME/workspace/indicear-runs/$run}"
dest="${dest%/}"
[[ -n "$dest" && "$dest" != "/" ]] || { echo "refusing destination '$dest'" >&2; exit 2; }
incoming="$dest/.incoming"
mkdir -p "$incoming"
transport_init "$session"

log() { echo "[$(date '+%H:%M:%S')] sync $run: $*"; }

# One syncer per destination: a pid lock file (never pkill by pattern).
lock="$dest/.sync.lock"
if [[ -f "$lock" ]] && kill -0 "$(cat "$lock")" 2>/dev/null; then
  log "another sync (pid $(cat "$lock")) is running for $dest"
  exit 1
fi
echo $$ > "$lock"
trap 'rm -f -- "${dest:?}/.sync.lock"' EXIT

prune() {
  [[ "$keep_all" == 1 ]] && return 0
  [[ -f "$dest/LATEST.json" ]] || return 0
  local keep path name
  keep="$(manifest_field "$dest/LATEST.json" keep)"
  for path in "$dest"/ckpt-*.pt; do
    [[ -e "$path" ]] || continue
    name="$(basename "$path")"
    if ! grep -qxF "$name" <<< "$keep"; then
      rm -f -- "${dest:?}/${name:?}"
      log "pruned $name"
    fi
  done
}

poll_once() {
  if ! pull "$remote/LATEST.json" "$incoming/LATEST.json"; then
    log "no LATEST.json yet (or the VM is unreachable)"
    return 1
  fi
  local ok=1 file sha kind got
  while IFS=$'\t' read -r file sha kind; do
    if [[ -f "$dest/$file" && "$(sha256_of "$dest/$file")" == "$sha" ]]; then
      continue
    fi
    if ! pull "$remote/$file" "$incoming/$file.part"; then
      log "could not pull $file (will retry)"
      ok=0
      continue
    fi
    got="$(sha256_of "$incoming/$file.part")"
    if [[ "$got" == "$sha" ]]; then
      mv -f "$incoming/$file.part" "$dest/$file"
      log "pulled $file ($kind)"
    else
      rm -f -- "${incoming:?}/${file:?}.part"
      log "REJECTED $file: sha256 $got does not match the manifest"
      ok=0
    fi
  done < <(manifest_entries "$incoming/LATEST.json")
  if [[ "$ok" == 1 ]]; then
    mv -f "$incoming/LATEST.json" "$dest/LATEST.json"
    prune
  fi
  pull_matching "$remote" "$dest" "*.csv" "*.log" "command.sh" "init_from.json" "DONE" "FAILED"
  # The tokenizer lives in <ROOT>/tok beside <ROOT>/runs/<run>. Without it the weights cannot be decoded,
  # so copy it once: a lost VM must never strand a run's checkpoints.
  if [[ ! -f "$dest/tok/aggregate.json" ]]; then
    mkdir -p "$dest/tok"
    pull_matching "$remote/../../tok" "$dest/tok" "aggregate.json" "*.model" "*.vocab"
    [[ -f "$dest/tok/aggregate.json" ]] && log "pulled the tokenizer from $remote/../../tok"
  fi
  if [[ -n "$drive_copy" ]]; then
    mkdir -p "$drive_copy/$run"
    rsync -a --exclude ".incoming" --exclude ".sync.lock" "$dest/" "$drive_copy/$run/" || log "drive copy failed"
  fi
  [[ "$ok" == 1 ]]
}

log "remote $remote -> $dest (session $session, every ${interval}s)"
while true; do
  poll_once || true
  if [[ -f "$dest/DONE" || -f "$dest/FAILED" ]]; then
    poll_once || true
    if [[ -f "$dest/DONE" ]]; then log "run finished: DONE"; else log "run finished: FAILED"; fi
    break
  fi
  [[ "$once" == 1 ]] && break
  sleep "$interval"
done
