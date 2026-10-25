#!/usr/bin/env bash
# Manifests (FLEURS by default; PREP_ARGS="--mixture pilot --stream" for the streamed mixture), tokenizer,
# synthetic code-switch dev, negatives, label rates.
# Extra options via PREP_ARGS (recorded in each run's command.sh), e.g. PREP_ARGS="--langs hi,en".
set -euo pipefail
source "$(dirname "$0")/env.sh"
read -ra prep_args <<< "${PREP_ARGS:-}"
indicear indicear.ablate --root "$ROOT" prep ${NOISE_DIR:+--noise-dir "$NOISE_DIR"} ${prep_args[@]+"${prep_args[@]}"} "$@"
