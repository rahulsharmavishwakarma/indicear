#!/usr/bin/env bash
# Frozen-encoder robustness probe with the v1 heads (run after v1 has trained).
# Conditions are split across PARALLEL processes on one GPU; finished conditions are skipped.
set -euo pipefail
source "$(dirname "$0")/env.sh"
VARIANT="${VARIANT:-v1}"
PARALLEL="${PARALLEL:-3}"
conditions=$("$PY" -c "from indicear.robustness_probe import DEFAULT_CONDITIONS as c; print(' '.join(c))")
read -ra list <<< "$conditions"
jobs=()
for ((i = 0; i < PARALLEL; i++)); do
  subset=$(for ((j = i; j < ${#list[@]}; j += PARALLEL)); do printf '%s,' "${list[j]}"; done)
  jobs+=(--job "robust$i" "$PY -m indicear.robustness_probe --root $ROOT --variant $VARIANT --conditions ${subset%,} --workers $(( $(nproc) / PARALLEL )) ${NOISE_DIR:+--noise-dir $NOISE_DIR}")
done
"$PY" -m indicear.runtime.parallel --parallel "$PARALLEL" --logs "$LOGS/robustness" "${jobs[@]}"
"$PY" -m indicear.robustness_probe --root "$ROOT" --variant "$VARIANT" --summarize
[ -n "$SYNC" ] && mkdir -p "$SYNC/results" && cp -r "$ROOT/results/." "$SYNC/results/" || true
