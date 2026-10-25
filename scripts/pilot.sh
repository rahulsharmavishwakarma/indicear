#!/usr/bin/env bash
# The pilot on one GPU, end to end: setup -> prep --stream -> live training of v2, v3, v4 and v4+lid
# (one process, one stream: layers 1-6 and the audio decode run once per batch for all four)
# -> eval on the fixed sets -> baselines -> report.
#
#   bash scripts/pilot.sh                 # queue every unfinished stage with launch.sh; returns at once
#   bash scripts/pilot.sh --foreground    # run the unfinished stages one after another in this shell
#   bash scripts/pilot.sh --stage NAME    # run one stage now (what each queued job runs)
#   bash scripts/status.sh                # progress; logs in $LOGS/pilot-<stage>.log
#
# Queued mode starts one nohup job per stage (scripts/launch.sh); each waits (WAIT_FOR) for the
# previous stage's $ROOT/pilot/<stage>.DONE and gives up if that stage writes <stage>.FAILED.
# Rerunning pilot.sh skips DONE stages, clears FAILED markers and re-queues the rest; every
# stage resumes its own work (prep skips finished shards, training restarts from the last
# checkpoint, eval and baselines skip finished (system, set) results). A stage stopped with
# scripts/stop.sh exits 143 and is marked FAILED; rerun pilot.sh to resume it.
# Each stage samples nvidia-smi every 5 s to $ROOT/pilot/<stage>.gpu.csv (training also writes
# runs/<v>/gpu.csv). Results stay under $ROOT/results and runs under $ROOT/runs for sync_runs.sh.
#
# Settings (environment): MIXTURE (pilot), VARIANTS (v2,v3,v4,v4+lid), TRAIN_HOURS (6500 =
# 650 h x 10 epochs), ALLOW_REVIEW (empty: MUCS stays eval-only; "mucs" trains on it after
# counsel review), BASELINES (indicear.baselines systems), NOISE_DIR, SYNC, and extra arguments in
# PREP_EXTRA, TRAIN_EXTRA, EVAL_EXTRA, BASELINE_EXTRA. HF_TOKEN must be in the job environment
# for the gated sources and baselines.
#
#   SMOKE=1 ROOT=/tmp/pilot-smoke PY=$(uv run python -c 'import sys; print(sys.executable)') \
#     bash scripts/pilot.sh --foreground
# runs the same chain on the CPU with the tiny tower, synthetic shards and a few steps
# (indicear.smoke); without --foreground it queues it the same way.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/env.sh"
STAGES=(setup prep train eval baselines report)
export MARK="$ROOT/pilot"
export MIXTURE="${MIXTURE:-pilot}"
export VARIANTS="${VARIANTS:-v2,v3,v4,v4+lid}"
export TRAIN_HOURS="${TRAIN_HOURS:-6500}"
export ALLOW_REVIEW="${ALLOW_REVIEW:-}"
export BASELINES="${BASELINES:-indicconformer-ctc,indicconformer-rnnt,ep-hindi-asr-mixed,ep-hindi-asr-native,dulu-malayalam,kothagemma,dulu-malayalam-ctc-778h}"
export PREP_EXTRA="${PREP_EXTRA:-}" TRAIN_EXTRA="${TRAIN_EXTRA:-}" EVAL_EXTRA="${EVAL_EXTRA:-}" BASELINE_EXTRA="${BASELINE_EXTRA:-}"
if [ -n "${SMOKE:-}" ]; then
  export SMOKE MIXTURE="$ROOT/smoke-mixture.json" TRAIN_HOURS=0 BASELINES=smoke-echo
  PREP_EXTRA="--negatives 4 --negatives-dev 2 --cs-per-pair 2 --decode-workers 2 $PREP_EXTRA"
  TRAIN_EXTRA="--max-steps 4 --frame-budget 300 --eval-every 2 --no-compile --stream-workers 2 --buffer-seconds 20 $TRAIN_EXTRA"
  EVAL_EXTRA="--robust-per-lang 2 --workers 2 $EVAL_EXTRA"
  BASELINE_EXTRA="--robust-per-lang 2 --workers 2 $BASELINE_EXTRA"
fi
# Exported so each run's command.sh (written by the train stage) repeats this prep on a fresh VM.
export PREP_ARGS="--mixture $MIXTURE --stream ${ALLOW_REVIEW:+--allow-review $ALLOW_REVIEW} $PREP_EXTRA"
mkdir -p "$MARK"

stage_command() {
  case "$1" in
    setup)
      if [ -n "${SMOKE:-}" ]; then "$PY" -m indicear.smoke mixture "$MIXTURE"; else bash "$here/setup_env.sh"; fi ;;
    prep)
      bash "$here/prep.sh" ;;
    train)
      # shellcheck disable=SC2046,SC2086
      indicear indicear.ablate --root "$ROOT" train --lower live --variants "$VARIANTS" \
        $( [ "$TRAIN_HOURS" != 0 ] && echo --train-hours "$TRAIN_HOURS" ) ${NOISE_DIR:+--noise-dir "$NOISE_DIR"} \
        $(sync_args) $TRAIN_EXTRA ;;
    eval)
      # shellcheck disable=SC2086
      indicear indicear.ablate --root "$ROOT" eval --variants "$VARIANTS" ${NOISE_DIR:+--noise-dir "$NOISE_DIR"} $EVAL_EXTRA ;;
    baselines)
      # shellcheck disable=SC2086
      if [ -n "${SMOKE:-}" ]; then
        indicear indicear.baselines --root "$ROOT" --systems "$BASELINES" $BASELINE_EXTRA
      else
        bash "$here/baselines.sh" --systems "$BASELINES" ${NOISE_DIR:+--noise-dir "$NOISE_DIR"} $BASELINE_EXTRA
      fi ;;
    report)
      indicear indicear.ablate --root "$ROOT" report
      if [ -n "$SYNC" ]; then mkdir -p "$SYNC/results" && cp -r "$ROOT/results/." "$SYNC/results/"; fi ;;
    *) echo "unknown stage $1 (${STAGES[*]})" >&2; return 2 ;;
  esac
}

run_stage() {
  local name="$1" status=0 sampler=""
  if [ -e "$MARK/$name.DONE" ]; then echo "pilot $name: already done"; return 0; fi
  rm -f "$MARK/$name.FAILED"
  echo "pilot $name: started $(date '+%F %T')"
  if command -v nvidia-smi >/dev/null 2>&1; then
    [ -e "$MARK/$name.gpu.csv" ] || echo "time,util_percent,memory_mib" > "$MARK/$name.gpu.csv"
    ( while true; do
        echo "$(date +%s),$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
        sleep 5
      done >> "$MARK/$name.gpu.csv" ) &
    sampler=$!
  fi
  stage_command "$name" || status=$?
  if [ -n "$sampler" ]; then kill "$sampler" 2>/dev/null || true; fi
  if [ "$status" = 0 ]; then
    date '+%F %T' > "$MARK/$name.DONE"
    echo "pilot $name: DONE"
  else
    echo "exit $status at $(date '+%F %T'); see $LOGS/pilot-$name.log" > "$MARK/$name.FAILED"
    echo "pilot $name: FAILED (exit $status)"
  fi
  return "$status"
}

case "${1:-}" in
  --stage) run_stage "$2" ;;
  --foreground)
    for name in "${STAGES[@]}"; do run_stage "$name"; done ;;
  "")
    previous=""
    for name in "${STAGES[@]}"; do
      if [ -e "$MARK/$name.DONE" ]; then echo "pilot $name: done"; previous=""; continue; fi
      rm -f "$MARK/$name.FAILED"
      pid_file="$LOGS/pilot-$name.pid"
      if [ -e "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "pilot $name: already queued or running (pid $(cat "$pid_file"))"
      else
        WAIT_FOR="${previous:+$MARK/$previous.DONE}" bash "$here/launch.sh" "pilot-$name" "$here/pilot.sh" --stage "$name"
      fi
      previous="$name"
    done ;;
  *) sed -n '2,32p' "$0"; exit 2 ;;
esac
