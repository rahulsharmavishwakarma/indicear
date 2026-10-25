#!/usr/bin/env bash
# Historical record: the command this run was launched with, kept unchanged below. The project was
# then called Sangam (module `sangam`, now `indicear`) and the paths are the original training VM's.
# To reproduce the run, use the commands in docs/training-recipe.md of the IndicEAR repository.
# Original command for run v1; written by sangam.ablate. Safe to rerun: every step resumes.
set -euo pipefail
export CODE=/content/sangam-asr/train
export ROOT=/content/sangam
export VENV=/content/venv314t
bash "$CODE/scripts/setup_env.sh"
bash "$CODE/scripts/prep.sh"
bash "$CODE/scripts/cache.sh"
source "$CODE/scripts/env.sh"
exec "$PY" -m sangam.ablate --root /content/sangam train --aug-views 0 --epochs 10 --variants v1
