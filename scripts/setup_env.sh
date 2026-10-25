#!/usr/bin/env bash
# Free-threaded Python 3.14t environment with uv (pinned by requirements-colab.txt, generated
# with `uv export` from pyproject.toml/uv.lock). Safe to rerun.
set -euo pipefail
source "$(dirname "$0")/env.sh"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.14t
[ -x "$PY" ] || uv venv "$VENV" --python 3.14t
uv pip install -q --python "$PY" -r "$CODE/requirements-colab.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h "$ROOT" | tail -1
"$PY" - <<'PY'
import sys, torch, transformers, sentencepiece, soundfile, jiwer, safetensors
print(sys.version.split()[0], "GIL enabled:", sys._is_gil_enabled(), "| torch", torch.__version__,
      "CUDA", torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0),
      "capability", torch.cuda.get_device_capability(0), "bf16", torch.cuda.is_bf16_supported())
from indicear.augment import available_codecs
print("ffmpeg codecs:", sorted(available_codecs()) or "none (codec conditions will be skipped)")
PY
