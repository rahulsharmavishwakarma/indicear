#!/usr/bin/env bash
# Upload a finished run (the warm-start source) to a VM: its final weights, configs and tokenizer.
#
#   bash scripts/push_init.sh SESSION [LOCAL_RUN] [REMOTE_DIR]
#
# LOCAL_RUN defaults to ~/indicear-runs/pilot/v4+lid (synced by sync_runs.sh, with tok/);
# REMOTE_DIR defaults to /content/full/init/v4+lid, the INIT_FROM of scripts/full_warm.sh with
# ROOT=/content/full. Uploads heads.safetensors, adapters.safetensors, variant.json, lid.json,
# result.json and tok/ (aggregate.json, *.model, *.vocab), then checks every file's sha256 on the
# VM. SESSION=local copies on this machine (tests).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/lib/transport.sh"
[[ $# -ge 1 ]] || { sed -n '2,10p' "$0"; exit 2; }
session="$1"
local_run="${2:-$HOME/workspace/indicear-runs/pilot/v4+lid}"
remote="${3:-/content/full/init/v4+lid}"
transport_init "$session"
files=()
for f in heads.safetensors adapters.safetensors variant.json lid.json result.json; do
  [[ -f "$local_run/$f" ]] && files+=("$f")
done
[[ -f "$local_run/heads.safetensors" && -f "$local_run/variant.json" ]] || { echo "$local_run is not a finished run" >&2; exit 1; }
[[ -f "$local_run/tok/aggregate.json" ]] || { echo "$local_run/tok is missing: run fetch_tok.sh or sync_runs.sh first" >&2; exit 1; }
for f in "$local_run"/tok/*; do files+=("tok/$(basename "$f")"); done
for f in "${files[@]}"; do
  push "$local_run/$f" "$remote/$f"
done
bad=0
for f in "${files[@]}"; do
  want="$(sha256_of "$local_run/$f")"
  got="$(rssh "sha256sum '$remote/$f' 2>/dev/null || shasum -a 256 '$remote/$f'" | cut -d' ' -f1)"
  if [[ "$got" != "$want" ]]; then echo "MISMATCH $f" >&2; bad=1; fi
done
[[ "$bad" == 0 ]] || exit 1
echo "uploaded ${#files[@]} files to $remote (sha256 verified); INIT_FROM=$remote"
