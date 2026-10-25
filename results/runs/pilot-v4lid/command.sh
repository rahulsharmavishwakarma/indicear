#!/usr/bin/env bash
# Original command for run v4+lid; written by sangam.ablate. Safe to rerun: every step resumes.
set -euo pipefail
export CODE=/content/sangam-asr-pilot/train
export PREP_ARGS='--mixture pilot --stream  '
export ROOT=/content/pilot
export VENV=/content/venv314t
bash "$CODE/scripts/setup_env.sh"
bash "$CODE/scripts/prep.sh"

source "$CODE/scripts/env.sh"
sangam sangam.ablate --root /content/pilot train --lower live --variants v4+lid --train-hours 4700
