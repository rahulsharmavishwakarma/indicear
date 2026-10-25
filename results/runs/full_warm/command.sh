#!/usr/bin/env bash
# Original command for run v4+lid; written by sangam.ablate. Safe to rerun: every step resumes.
set -euo pipefail
export CODE=/content/sangam-asr/train
export PREP_ARGS='--mixture full_warm --stream --tok-from /content/full/init/v4+lid/tok --tok-keep deva,latn,taml,knda  '
export ROOT=/content/full
export VENV=/content/venv314t
bash "$CODE/scripts/setup_env.sh"
bash "$CODE/scripts/prep.sh"

source "$CODE/scripts/env.sh"
sangam sangam.ablate --root /content/full train --lower live --variants v4+lid --init-from /content/full/init/v4+lid --train-hours 8000 --packing 0.75 --buffer-seconds 10800 --peak-lr 6e-4 --final-lr 5e-5 --warmup-steps 300 --new-lr-mult 3 --inter-weight 0.15 --lid-frame-weight 0.2 --augment reverb=0.3,reverb_pointnoise=0.175,noise=0.2,babble=0.15,speed=0.125,tel8k=0.05 --new-script-bias -2 --cache-gb 100 --zero-streams kathbath/ml,kathbath/or,kathbath/ur,kathbath/gu --stream-scale indicvoices/hi=2.24,vaani/hi=2.24
