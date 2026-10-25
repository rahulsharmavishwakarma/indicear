#!/usr/bin/env bash
# Layers 1-6 of the frozen encoder, once (resumable per part; df is checked before writing).
set -euo pipefail
source "$(dirname "$0")/env.sh"
# CACHE_WORKERS: feature/augmentation DataLoader workers (default: every core); each worker runs
# with OMP/OpenBLAS/MKL limited to one thread.
"$PY" -m indicear.ablate --root "$ROOT" cache --aug-views "${AUG_VIEWS:-0}" --storage "${STORAGE:-bf16}" \
  --workers "${CACHE_WORKERS:-$(nproc)}" ${NOISE_DIR:+--noise-dir "$NOISE_DIR"} "$@"
touch "$ROOT/cache/layer6/cache.done"   # lets queued jobs (WAIT_FOR) start
