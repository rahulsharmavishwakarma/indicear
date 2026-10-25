#!/usr/bin/env bash
# The continued ("full") run on one GPU: warm start from the finished pilot, all 12 Indian
# languages plus English (docs/PLAN.md, "Full warm-started run").
#
#   bash scripts/full_warm.sh                 # queue every unfinished stage with launch.sh; returns at once
#   bash scripts/full_warm.sh --foreground    # run the unfinished stages one after another in this shell
#   bash scripts/full_warm.sh --stage NAME    # run one stage now (what each queued job runs)
#   bash scripts/status.sh                    # progress; logs in $LOGS/full_warm-<stage>.log
#
# Stages: setup -> prep -> train -> eval -> report. Each writes $ROOT/full_warm/<stage>.DONE or
# .FAILED; a queued stage waits (WAIT_FOR) for the previous one's DONE and gives up on its FAILED.
# Rerunning skips DONE stages and resumes the rest (prep skips finished work, training restarts
# from its last checkpoint, eval skips finished sets).
#
# - prep: with PREP_PACK (an archive from `python -m indicear.prep_pack pack`), restores it first, so
#   a restart skips the text pass and the tokenizer (and, for a --with-audio pack, every download).
#   Then `ablate prep --mixture full_warm --stream --tok-from $INIT_FROM/tok`: the pilot's deva,
#   latn, taml and knda SentencePiece models are kept unchanged; the other scripts are trained on
#   this mixture's text. With RIR_SETS, scripts/rirs.sh fetches the real RIRs. PACK_OUT writes a
#   pack of the prepared ROOT afterwards (PACK_AUDIO=1 adds the fixed sets' audio).
# - train: `ablate train --lower live --variants v4+lid --init-from $INIT_FROM` (indicear.warmstart):
#   a new run (new config hash, fresh AdamW and schedule) whose adapters, phonetic, LID and shared
#   heads come from the pilot, with the output rows of the kept scripts copied.
# - eval: every eval set prep wrote (Kathbath test, IndicVoices valid, Vaani test, FLEURS dev,
#   IndicContextEval, MUCS, LibriSpeech, ...) and the robustness subset, for all 12 languages.
#
# Settings (environment; defaults in brackets):
#   INIT_FROM [$ROOT/init/v4+lid]  the finished pilot run on this host: heads.safetensors,
#       adapters.safetensors, variant.json, lid.json and tok/ (upload it with scripts/push_init.sh)
#   MIXTURE [full_warm]  VARIANTS [v4+lid]  TOK_KEEP [deva,latn,taml,knda]
#   TRAIN_HOURS [auto: 7000 on an A100, 8000 otherwise]  PACKING [0.75: measured by simulating the
#       planner with a 3 h shuffle buffer; the pilot's 0.5 h buffer packed 0.48]  BUFFER_SECONDS [10800]
#   PEAK_LR [6e-4]  FINAL_LR [5e-5]  WARMUP [300]  NEW_LR_MULT [3]  INTER_WEIGHT [0.15]
#   LID_FRAME_WEIGHT [0.2]  AUGMENT [reverb=0.3,reverb_pointnoise=0.175,noise=0.2,babble=0.15,speed=0.125,tel8k=0.05]
#   RIR_SETS [empty: synthetic RIRs only; "air,RVB2014" after their licence review]  REAL_RIR_SHARE [0.5]
#   PREP_PACK, PACK_OUT, PACK_AUDIO, ALLOW_REVIEW [empty], NOISE_DIR, SYNC,
#   PREP_EXTRA, TRAIN_EXTRA, EVAL_EXTRA. HF_TOKEN must be in the job environment.
#
#   SMOKE=1 ROOT=/tmp/full-smoke PY=$(uv run python -c 'import sys; print(sys.executable)') \
#     bash scripts/full_warm.sh --foreground
# runs the same chain on the CPU with the tiny tower and synthetic shards (indicear.smoke): setup
# trains a 3-step smoke "pilot" to warm-start from.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/env.sh"
STAGES=(setup prep train eval report)
export MARK="$ROOT/full_warm"
export MIXTURE="${MIXTURE:-full_warm}"
export VARIANTS="${VARIANTS:-v4+lid}"
export INIT_FROM="${INIT_FROM:-$ROOT/init/v4+lid}"
export TOK_KEEP="${TOK_KEEP:-deva,latn,taml,knda}"
export PACKING="${PACKING:-0.75}" BUFFER_SECONDS="${BUFFER_SECONDS:-10800}"
export PEAK_LR="${PEAK_LR:-6e-4}" FINAL_LR="${FINAL_LR:-5e-5}" WARMUP="${WARMUP:-300}" NEW_LR_MULT="${NEW_LR_MULT:-3}"
export INTER_WEIGHT="${INTER_WEIGHT:-0.15}" LID_FRAME_WEIGHT="${LID_FRAME_WEIGHT:-0.2}"
export AUGMENT="${AUGMENT:-reverb=0.3,reverb_pointnoise=0.175,noise=0.2,babble=0.15,speed=0.125,tel8k=0.05}"
export RIR_SETS="${RIR_SETS:-}" REAL_RIR_SHARE="${REAL_RIR_SHARE:-0.5}"
export ALLOW_REVIEW="${ALLOW_REVIEW:-}" PREP_PACK="${PREP_PACK:-}" PACK_OUT="${PACK_OUT:-}" PACK_AUDIO="${PACK_AUDIO:-}"
export PREP_EXTRA="${PREP_EXTRA:-}" TRAIN_EXTRA="${TRAIN_EXTRA:-}" EVAL_EXTRA="${EVAL_EXTRA:-}"
if [ -z "${TRAIN_HOURS:-}" ]; then
  # 10-12 GPU-hours at the measured ~620 h of audio per A100-hour; the RTX PRO 6000 (G4) is
  # estimated at 1.15-1.55x that (unmeasured), so 8000 h takes about 8.3-11.2 h there.
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  case "$gpu" in *A100*) TRAIN_HOURS=7000 ;; *) TRAIN_HOURS=8000 ;; esac
fi
export TRAIN_HOURS
if [ -n "${SMOKE:-}" ]; then
  export SMOKE MIXTURE="$ROOT/smoke-full-mixture.json" TRAIN_HOURS=0 BUFFER_SECONDS=20
  export INIT_FROM="$ROOT/init/v4+lid"
  PREP_EXTRA="--negatives 4 --negatives-dev 2 --cs-per-pair 2 --decode-workers 2 $PREP_EXTRA"
  TRAIN_EXTRA="--max-steps 4 --frame-budget 300 --eval-every 2 --no-compile --stream-workers 2 $TRAIN_EXTRA"
  EVAL_EXTRA="--robust-per-lang 2 --workers 2 $EVAL_EXTRA"
fi
# Exported so each run's command.sh (written by the train stage) repeats this prep on a fresh VM.
export PREP_ARGS="--mixture $MIXTURE --stream --tok-from $INIT_FROM/tok --tok-keep $TOK_KEEP ${ALLOW_REVIEW:+--allow-review $ALLOW_REVIEW} $PREP_EXTRA"
mkdir -p "$MARK"

smoke_setup() {
  # A smoke "pilot" to start from: prep + 3 live steps of v4+lid on the smoke mixture, in its own
  # ROOT; its run directory plus tok/ is INIT_FROM. Then the warm mixture: the smoke mixture with the
  # full mixture's new options (loanword normaliser, stream weights, disjoint code-mixed pools).
  local src="$ROOT/init-src"
  if [ ! -e "$INIT_FROM/heads.safetensors" ]; then
    "$PY" -m indicear.smoke mixture "$src/smoke-mixture.json"
    "$PY" -m indicear.smoke ablate --root "$src" prep --mixture "$src/smoke-mixture.json" --stream --negatives 4 \
      --negatives-dev 2 --cs-per-pair 2 --decode-workers 2
    "$PY" -m indicear.smoke ablate --root "$src" train --lower live --variants v4+lid --max-steps 3 --frame-budget 300 \
      --eval-every 0 --no-compile --stream-workers 2 --buffer-seconds 20
    mkdir -p "$(dirname "$INIT_FROM")"
    rm -rf "$INIT_FROM"
    cp -R "$src/runs/v4+lid" "$INIT_FROM"
    cp -R "$src/tok" "$INIT_FROM/tok"
  fi
  "$PY" -m indicear.smoke mixture "$ROOT/smoke-base-mixture.json"
  "$PY" - "$ROOT/smoke-base-mixture.json" "$MIXTURE" <<'PY'
import json, sys
mixture = json.load(open(sys.argv[1]))
mixture.update(name="smoke_full_warm", disjoint_code_mixed_pools=True, loanwords={"sources": ["kathbath"]})
for entry in mixture["train"]:
    if entry["source"] == "kathbath":
        entry["weight"] = 2.0
json.dump(mixture, open(sys.argv[2], "w"), indent=1)
PY
}

check_init() {
  for f in heads.safetensors variant.json tok/aggregate.json; do
    [ -e "$INIT_FROM/$f" ] || { echo "INIT_FROM=$INIT_FROM has no $f: upload the pilot run (scripts/push_init.sh)" >&2; return 1; }
  done
}

stage_command() {
  case "$1" in
    setup)
      if [ -n "${SMOKE:-}" ]; then smoke_setup; else bash "$here/setup_env.sh"; fi
      check_init ;;
    prep)
      check_init
      if [ -n "$PREP_PACK" ] && [ ! -e "$ROOT/tok/aggregate.json" ]; then
        "$PY" -m indicear.prep_pack restore --root "$ROOT" --archive "$PREP_PACK"
      fi
      bash "$here/prep.sh"
      if [ -n "$RIR_SETS" ]; then RIR_SETS="$RIR_SETS" bash "$here/rirs.sh"; fi
      if [ -n "$PACK_OUT" ]; then
        "$PY" -m indicear.prep_pack pack --root "$ROOT" --out "$PACK_OUT" ${PACK_AUDIO:+--with-audio}
      fi ;;
    train)
      check_init
      # shellcheck disable=SC2046,SC2086
      indicear indicear.ablate --root "$ROOT" train --lower live --variants "$VARIANTS" --init-from "$INIT_FROM" \
        $( [ "$TRAIN_HOURS" != 0 ] && echo --train-hours "$TRAIN_HOURS" ) --packing "$PACKING" \
        --buffer-seconds "$BUFFER_SECONDS" --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" --warmup-steps "$WARMUP" \
        --new-lr-mult "$NEW_LR_MULT" --inter-weight "$INTER_WEIGHT" --lid-frame-weight "$LID_FRAME_WEIGHT" \
        --augment "$AUGMENT" ${RIR_SETS:+--rir-dir "$ROOT/rirs" --real-rir-share "$REAL_RIR_SHARE"} \
        ${NOISE_DIR:+--noise-dir "$NOISE_DIR"} $(sync_args) $TRAIN_EXTRA ;;
    eval)
      # shellcheck disable=SC2086
      indicear indicear.ablate --root "$ROOT" eval --variants "$VARIANTS" ${NOISE_DIR:+--noise-dir "$NOISE_DIR"} $EVAL_EXTRA ;;
    report)
      indicear indicear.ablate --root "$ROOT" report
      if [ -n "$SYNC" ]; then mkdir -p "$SYNC/results" && cp -r "$ROOT/results/." "$SYNC/results/"; fi ;;
    *) echo "unknown stage $1 (${STAGES[*]})" >&2; return 2 ;;
  esac
}

run_stage() {
  local name="$1" status=0 sampler=""
  if [ -e "$MARK/$name.DONE" ]; then echo "full_warm $name: already done"; return 0; fi
  rm -f "$MARK/$name.FAILED"
  echo "full_warm $name: started $(date '+%F %T')"
  if command -v nvidia-smi >/dev/null 2>&1; then
    [ -e "$MARK/$name.gpu.csv" ] || echo "time,util_percent,memory_mib" > "$MARK/$name.gpu.csv"
    ( while true; do
        echo "$(date +%s),$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
        sleep 5
      done >> "$MARK/$name.gpu.csv" ) &
    sampler=$!
  fi
  local started; started=$(date +%s)
  stage_command "$name" || status=$?
  if [ -n "$sampler" ]; then kill "$sampler" 2>/dev/null || true; fi
  # A stopped prep can exit 0 (graceful SIGTERM): count it as done only if it wrote a fresh report.
  if [ "$name" = prep ] && [ "$status" = 0 ]; then
    if [ ! -s "$ROOT/prep_report.json" ] || [ "$(stat -c %Y "$ROOT/prep_report.json" 2>/dev/null || stat -f %m "$ROOT/prep_report.json")" -lt "$started" ]; then
      status=3; echo "full_warm prep: exited without a fresh prep_report.json; not marking DONE"
    fi
  fi
  if [ "$status" = 0 ]; then
    date '+%F %T' > "$MARK/$name.DONE"
    echo "full_warm $name: DONE"
  else
    echo "exit $status at $(date '+%F %T'); see $LOGS/full_warm-$name.log" > "$MARK/$name.FAILED"
    echo "full_warm $name: FAILED (exit $status)"
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
      if [ -e "$MARK/$name.DONE" ]; then echo "full_warm $name: done"; previous=""; continue; fi
      rm -f "$MARK/$name.FAILED"
      pid_file="$LOGS/full_warm-$name.pid"
      if [ -e "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "full_warm $name: already queued or running (pid $(cat "$pid_file"))"
      else
        WAIT_FOR="${previous:+$MARK/$previous.DONE}" bash "$here/launch.sh" "full_warm-$name" "$here/full_warm.sh" --stage "$name"
      fi
      previous="$name"
    done ;;
  *) sed -n '2,46p' "$0"; exit 2 ;;
esac
