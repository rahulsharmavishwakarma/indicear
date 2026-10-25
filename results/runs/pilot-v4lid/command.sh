#!/usr/bin/env bash
# Historical record: the command this run was launched with, kept unchanged below. The project was
# then called Sangam (module `sangam`, now `indicear`) and the paths are the original training VM's.
# To reproduce the run, use the commands in docs/training-recipe.md of the IndicEAR repository.
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
