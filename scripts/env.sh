# Sourced by every script: one place for paths and GPU-saturation settings.
# Override any variable in the environment, e.g. ROOT=/mnt/nvme/indicear bash scripts/cache.sh
export CODE="${CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROOT="${ROOT:-/content/indicear}"                 # local NVMe: data, layer-6 cache, working runs
export SYNC="${SYNC:-}"                                # persistent copy of runs/results, e.g. /content/drive/MyDrive/indicear
export VENV="${VENV:-/content/venv314t}"
export PY="${PY:-$VENV/bin/python}"
export LOGS="${LOGS:-$ROOT/logs}"
export PYTHON_GIL=0                                    # free-threaded 3.14t: keep the GIL off
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT" "$LOGS"
cd "$CODE"
sync_args() { if [ -n "$SYNC" ]; then echo "--drive-copy $SYNC"; fi; }
# indicear MODULE ARGS...: python -m MODULE, or with SMOKE=1 the CPU smoke harness (indicear.smoke: tiny
# tower, synthetic shards, no download) running the same entry point.
indicear() {
  if [ -n "${SMOKE:-}" ]; then "$PY" -m indicear.smoke "$@"; else local module="$1"; shift; "$PY" -m "$module" "$@"; fi
}
