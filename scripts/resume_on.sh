#!/usr/bin/env bash
# Resume a run on a fresh Colab VM from the verified local copy.
#
#   bash scripts/resume_on.sh [--local DIR] [--remote-dir DIR] [--no-code] [--no-launch] SESSION RUN
#
# 1. Verifies the local checkpoints and best.pt against LATEST.json (sha256).
# 2. Uploads the code (the repository, without virtualenvs) to the CODE path recorded in command.sh,
#    then the checkpoints, LATEST.json, command.sh and train.csv to the original run directory
#    (rsync over the shared SSH socket, else `colab upload`).
#    The run's tokenizer (<local>/tok, pulled by sync_runs.sh) goes to <ROOT>/tok when the VM has
#    none, so prep does not retrain one that no longer matches the checkpoints (--no-tok skips it).
# 3. Relaunches command.sh under nohup: it rebuilds the environment, data and layer-6 cache
#    (each step skips finished shards) and reruns the training command, which auto-resumes.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/lib/transport.sh"

local_dir=""
remote_dir=""
push_code=1
push_tok=1
launch=1
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --local) local_dir="$2"; shift 2 ;;
    --remote-dir) remote_dir="$2"; shift 2 ;;
    --no-code) push_code=0; shift ;;
    --no-tok) push_tok=0; shift ;;
    --no-launch) launch=0; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ $# -eq 2 ]] || { sed -n '2,13p' "$0"; exit 2; }
session="$1"
run="$2"
local_dir="${local_dir:-$HOME/workspace/indicear-runs/$run}"
manifest="$local_dir/LATEST.json"
[[ -f "$manifest" ]] || { echo "no verified local copy: $manifest is missing" >&2; exit 1; }
remote_dir="${remote_dir:-$(manifest_field "$manifest" run_dir)}"
[[ -n "$remote_dir" ]] || { echo "LATEST.json has no run_dir; pass --remote-dir" >&2; exit 1; }
transport_init "$session"

uploads=()
while IFS=$'\t' read -r file sha kind; do
  [[ "$kind" == "artifact" ]] && continue
  if [[ ! -f "$local_dir/$file" ]]; then
    echo "skip $file: not present locally"
    continue
  fi
  if [[ "$(sha256_of "$local_dir/$file")" != "$sha" ]]; then
    echo "local $file fails its sha256 check; refusing to upload it" >&2
    exit 1
  fi
  uploads+=("$file")
done < <(manifest_entries "$manifest")
[[ ${#uploads[@]} -gt 0 ]] || { echo "no verified checkpoint to upload" >&2; exit 1; }

if [[ "$push_code" == 1 && -f "$local_dir/command.sh" ]]; then
  code_remote="$(sed -n "s/^export CODE=//p" "$local_dir/command.sh" | tr -d "'")"
  code_local="$(cd "$here/.." && pwd)"
  echo "uploading code $code_local -> $code_remote"
  excludes=(--exclude ".venv" --exclude "__pycache__" --exclude ".pytest_cache" --exclude ".ruff_cache")
  if [[ "$session" == "local" ]]; then
    rsync -a "${excludes[@]}" "$code_local/" "$code_remote/"
  else
    rssh "mkdir -p '$code_remote'"
    rsync -a -e "ssh -F $SSH_CONF" "${excludes[@]}" "$code_local/" "$SSH_HOST:$code_remote/"
  fi
fi

for file in "${uploads[@]}" LATEST.json command.sh train.csv; do
  [[ -f "$local_dir/$file" ]] || continue
  push "$local_dir/$file" "$remote_dir/$file"
  echo "uploaded $file"
done

if [[ "$push_tok" == 1 && -f "$local_dir/tok/aggregate.json" && -f "$local_dir/command.sh" ]]; then
  root_remote="$(sed -n "s/^export ROOT=//p" "$local_dir/command.sh" | tr -d "'")"
  if [[ -n "$root_remote" ]] && ! rssh "test -f '$root_remote/tok/aggregate.json'" 2>/dev/null; then
    for f in "$local_dir"/tok/*; do
      push "$f" "$root_remote/tok/$(basename "$f")"
    done
    echo "uploaded the run's tokenizer to $root_remote/tok"
  fi
fi

if [[ "$launch" == 1 ]]; then
  rssh "chmod +x '$remote_dir/command.sh' && nohup bash '$remote_dir/command.sh' >> '$remote_dir/resume.log' 2>&1 < /dev/null & echo started pid \$!"
  echo "relaunched $run; follow it with: bash $here/sync_runs.sh $session $remote_dir"
fi
