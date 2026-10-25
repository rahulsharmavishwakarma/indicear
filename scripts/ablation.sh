#!/usr/bin/env bash
# Train and score ablation variants. Finished variants (results/<v>.json) are skipped and
# checkpointed ones resume, so rerunning this script continues the ablation.
#   VARIANTS="v1 v2 v3 v4" PARALLEL=1 bash scripts/ablation.sh     # one at a time, whole GPU
#   VARIANTS="v2 v3 v4"    PARALLEL=3 bash scripts/ablation.sh     # three jobs share the GPU
set -euo pipefail
source "$(dirname "$0")/env.sh"
VARIANTS="${VARIANTS:-v1 v2 v3 v4}"
PARALLEL="${PARALLEL:-1}"
common=(--root "$ROOT" train --aug-views "${AUG_VIEWS:-0}" --epochs "${EPOCHS:-10}" $(sync_args))
if [ "$PARALLEL" -le 1 ]; then
  "$PY" -m indicear.ablate "${common[@]}" --variants "$(echo $VARIANTS | tr ' ' ,)" "$@"
else
  share=$("$PY" -c "print(round(1/$PARALLEL, 3))")
  jobs=()
  for v in $VARIANTS; do
    jobs+=(--job "$v" "$PY -m indicear.ablate ${common[*]} --variants $v --vram-share $share --cache-placement mmap $*")
  done
  "$PY" -m indicear.runtime.parallel --parallel "$PARALLEL" --logs "$LOGS/ablation" "${jobs[@]}"
fi
"$PY" -m indicear.ablate --root "$ROOT" report
[ -n "$SYNC" ] && mkdir -p "$SYNC/results" && cp -r "$ROOT/results/." "$SYNC/results/" || true
