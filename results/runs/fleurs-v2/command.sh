#!/usr/bin/env bash
# Original command for run v2; written by sangam.ablate. Safe to rerun: every step resumes.
set -euo pipefail
export CODE=/content/sangam-asr/train
export AUG_VIEWS=0
export ROOT=/content/sangam
export VENV=/content/venv314t
bash "$CODE/scripts/setup_env.sh"
bash "$CODE/scripts/prep.sh"
bash "$CODE/scripts/cache.sh"
source "$CODE/scripts/env.sh"
exec "$PY" -m sangam.ablate --root /content/sangam train --aug-views 0 --epochs 10 --variants v2 --vram-share 0.333 --cache-placement mmap
