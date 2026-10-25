#!/usr/bin/env bash
# Baselines (indicear.baselines) on the eval sets prepared by prep --stream. They run in a separate
# environment built from this project's `baselines` dependency group on standard CPython 3.13
# (onnxruntime has no free-threaded wheels), not the free-threaded training venv; indicear itself is
# imported from $CODE. Finished (system, set) results are skipped.
#   bash scripts/baselines.sh [--systems indicconformer-ctc,...] [--sets 'kathbath_*'] [...]
set -euo pipefail
source "$(dirname "$0")/env.sh"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${BASELINE_VENV:-$ROOT/venv-baselines}"
uv sync --locked --project "$CODE" --python 3.13 --group baselines --no-dev --no-install-project
# PYTHON_GIL=0 is for the free-threaded build only; a standard build refuses it.
env -u PYTHON_GIL PYTHONPATH="$CODE" "$UV_PROJECT_ENVIRONMENT/bin/python" -m indicear.baselines --root "$ROOT" "$@"
