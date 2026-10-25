#!/usr/bin/env bash
# Copy a training ROOT's tokenizer (<ROOT>/tok) from the VM to this machine.
#
#   bash scripts/fetch_tok.sh SESSION ROOT DEST
#   bash scripts/fetch_tok.sh $S /content/indicear ~/indicear-tok/pilot
#
# scripts/sync_runs.sh pulls run directories only; the tokenizer lives once per ROOT, and every
# run trained in that ROOT uses it. Fetch it once per ROOT and pass DEST to
# `python -m indicear.export --tok`. Pulls aggregate.json, then the SentencePiece model of every
# script it lists (plus the .vocab files, best effort), into DEST/.incoming, and moves the set
# into place only when every model arrived. SESSION=local reads ROOT as a local path.
set -euo pipefail
source "$(dirname "$0")/lib/transport.sh"

[[ $# -eq 3 ]] || { sed -n '2,11p' "$0"; exit 2; }
session="$1"
remote="${2%/}/tok"
dest="${3%/}"
[[ -n "$dest" && "$dest" != "/" ]] || { echo "refusing destination '$dest'" >&2; exit 2; }
incoming="$dest/.incoming"
rm -rf -- "${incoming:?}"
mkdir -p "$incoming"
transport_init "$session"

pull "$remote/aggregate.json" "$incoming/aggregate.json" \
  || { echo "could not pull $remote/aggregate.json (wrong ROOT, or prep has not trained the tokenizer yet)" >&2; exit 1; }
scripts="$("$PYTHON" -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))["order"]))' "$incoming/aggregate.json")"
for script in $scripts; do
  pull "$remote/$script.model" "$incoming/$script.model" || { echo "could not pull $remote/$script.model" >&2; exit 1; }
  [[ -s "$incoming/$script.model" ]] || { echo "$script.model is empty" >&2; exit 1; }
  pull "$remote/$script.vocab" "$incoming/$script.vocab" 2>/dev/null || true
done
for file in "$incoming"/*; do
  mv -f "$file" "$dest/"
done
rmdir "$incoming"
echo "tokenizer: $(wc -w <<< "$scripts" | tr -d ' ') scripts from $remote -> $dest"
