#!/usr/bin/env bash
# Example for one Apple-silicon workstation (an 18-core M5 Pro; adjust the defaults): resume the
# pilot's live run (v4+lid) with Apple MPS, writing straight into the local
# run directory, so no sync is needed (the Colab A100 that started it is gone).
#
#   bash scripts/resume_local.sh                # start in the background: nohup, nice 10, caffeinate -i
#   bash scripts/resume_local.sh --follow       # start, then follow the log (Ctrl-C stops following only)
#   bash scripts/resume_local.sh --stop         # SIGTERM to the trainer: it checkpoints after the step
#   bash scripts/resume_local.sh --status       # pid, last log lines, LATEST.json step, host memory
#
# Same arguments as the run's original command.sh (`train --lower live --variants v4+lid
# --train-hours 4700`, pilot.sh defaults), plus host-only options that never enter the config
# hash: --device mps, --precision, --splits, --stream-workers, --no-compile, --keep-shards,
# --lenient-stream-restore. The checkpoint restores the frame budget (67045), the step count
# (7423), the optimizer, the scheduler, the loss scale and the stream position.
#
# The workstation stays usable: MPS memory is capped (MPS_MEM_GB), torch and the stream use CPU_THREADS of
# the 18 cores, the job runs at nice 10 (CPU priority only; the GPU is not throttled), and the
# trainer adds a micro-batch split when swap passes SWAP_LIMIT_GB or memory pressure turns critical.
#
# Settings (environment):
#   ROOT=~/indicear-pilot-root   prepared data (manifests, stream_plan.json, dev/eval audio, tok/)
#   RUNS=~/indicear-runs/pilot   run directories; $ROOT/runs is made a symlink to it
#   RUN=v4+lid  TRAIN_HOURS=4700         the run and its original --train-hours
#   PRECISION=fp16                       fp16 (float16 autocast + GradScaler), bf16 or fp32. Measured on
#                                        the real model: fp16 matches fp32 gradients (cosine >= 0.96), bf16
#                                        does not (adapter gradients cosine 0.1-0.9), at the same speed
#   COMPILE=1                            torch.compile the tower layers and heads (0: --no-compile, eager);
#                                        fp16 + compile ~13.7k padded frames/s vs fp32 eager ~3.3k
#   MPS_MEM_GB=42                        MPS allocation cap in GiB (of 51.8 recommended): sets
#                                        PYTORCH_MPS_HIGH_WATERMARK_RATIO; past it the allocator raises an
#                                        out-of-memory error and the trainer doubles its micro-batches
#   SPLITS=0                             micro-batches per step; 0 = derived from MPS_MEM_GB (fp16 + compile: 3)
#   CPU_THREADS=14                       CPU threads for the job (18 cores: 4 left free): STREAM_WORKERS
#   STREAM_WORKERS=4                     decode/codec threads (ffmpeg -threads 1), the rest torch threads
#                                        (log-mel and augmentation on the CPU)
#   SWAP_LIMIT_GB=8                      add a micro-batch split when swap exceeds this (or pressure critical)
#   KEEP_SHARDS=1                        keep the stream pool on disk (68 GiB in $ROOT/stream-cache) instead
#                                        of re-downloading it every stream epoch; PREFETCH=1 downloads it
#   PREFETCH=1                           before training (files present are skipped); training is otherwise
#                                        download-bound here (5-40 MB/s from the Hub vs ~550 s of audio/s)
#   LENIENT_RESTORE=1                    --lenient-stream-restore: a rebuilt prep root can carve a
#                                        different dev set, so a row the A100 had buffered now belongs to an
#                                        excluded (dev) speaker; it is left out and logged (0: stop instead)
#   EXTRA=""                             more `ablate train` arguments
#   ALLOW_CODEC_MISMATCH=1               start even when ffmpeg lacks a codec the run used (GSM)
#
# Logs: $RUN_DIR/resume_local.log (stdout), train.log / train.csv (per step, peak_gib from MPS; host
# memory every 5 min), gpu.csv (in-process: Apple GPU utilization and memory via ioreg), host.csv
# (this script: GPU utilization, GPU memory, trainer RSS, swap, every 10 s). The pid is in
# $LOGS/pilot-train-local.pid.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export CODE="${CODE:-$(cd "$here/.." && pwd)}"
export ROOT="${ROOT:-$HOME/workspace/indicear-pilot-root}"
RUNS="${RUNS:-$HOME/workspace/indicear-runs/pilot}"
RUN="${RUN:-v4+lid}"
TRAIN_HOURS="${TRAIN_HOURS:-4700}"
PRECISION="${PRECISION:-fp16}"
COMPILE="${COMPILE:-1}"
MPS_MEM_GB="${MPS_MEM_GB:-42}"
SPLITS="${SPLITS:-0}"
CPU_THREADS="${CPU_THREADS:-14}"
STREAM_WORKERS="${STREAM_WORKERS:-4}"
SWAP_LIMIT_GB="${SWAP_LIMIT_GB:-8}"
KEEP_SHARDS="${KEEP_SHARDS:-1}"
PREFETCH="${PREFETCH:-1}"
LENIENT_RESTORE="${LENIENT_RESTORE:-1}"
EXTRA="${EXTRA:-}"
export VENV="${VENV:-$CODE/.venv}"
export PY="${PY:-$VENV/bin/python}"
export PREP_ARGS="${PREP_ARGS:---mixture pilot --stream  }"  # as the original command.sh, which this run rewrites
# Before torch is imported: unsupported MPS ops fall back to the CPU.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export INDICEAR_DEVICE=mps HF_HUB_DISABLE_TELEMETRY=1
# Plain HTTP downloads with read timeouts: an hf_xet shard download hung here with no progress and no
# error (24 Oct 2026, a VoxPopuli shard during the buffer restore). HF_HUB_DISABLE_XET=0 to undo.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
# A stop (SIGTERM) exits right after its checkpoint instead of waiting for in-flight downloads.
export INDICEAR_FAST_EXIT=1 INDICEAR_SWAP_LIMIT_GB="$SWAP_LIMIT_GB" INDICEAR_MPS_MEM_GB="$MPS_MEM_GB"
torch_threads=$(( CPU_THREADS - STREAM_WORKERS > 1 ? CPU_THREADS - STREAM_WORKERS : 1 ))
export INDICEAR_TORCH_THREADS="$torch_threads" OMP_NUM_THREADS="$torch_threads" MKL_NUM_THREADS="$torch_threads" \
  VECLIB_MAXIMUM_THREADS="$torch_threads"
source "$here/env.sh"   # PYTHON_GIL=0, PYTHONPATH, LOGS, cd $CODE
RUN_DIR="$RUNS/$RUN"
PID_FILE="$LOGS/pilot-train-local.pid"
OUT="$RUN_DIR/resume_local.log"

running() { [ -e "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }

case "${1:-}" in
  --stop)
    if running; then
      pid=$(cat "$PID_FILE")
      if ps -o command= -p "$pid" | grep -q "indicear.ablate.*--prefetch-pool\|^bash"; then
        pkill -TERM -P "$pid" 2>/dev/null || true; kill -TERM "$pid"  # still downloading the pool: no training state
        echo "stopped the pool download ($pid); finished files stay, partial ones resume next time"
      else
        kill -TERM "$pid"  # the trainer only: never its ffmpeg children
        echo "sent SIGTERM to the trainer ($pid); it checkpoints after the current step, then exits"
      fi
    else
      echo "not running"
    fi
    exit 0 ;;
  --status)
    if running; then echo "running (pid $(cat "$PID_FILE"))"; else echo "not running"; fi
    grep -E "step=|pool:|host memory" "$OUT" 2>/dev/null | tail -n 3 || true
    "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print('LATEST.json:', d['status'], 'step', d['step'], d['checkpoint']['file'])" \
      "$RUN_DIR/LATEST.json"
    "$PY" -c "from indicear.runtime.gpu import host_memory; print('host memory:', host_memory())"
    exit 0 ;;
  ""|--follow) ;;
  *) sed -n '2,46p' "$0"; exit 2 ;;
esac

# ---------------------------------------------------------------- checks
if running; then echo "already running (pid $(cat "$PID_FILE"))" >&2; exit 1; fi
[ -x "$PY" ] || { echo "no Python at $PY: run 'uv sync --locked' in $CODE" >&2; exit 1; }
[ -e "$ROOT/manifests/stream_plan.json" ] || { echo "$ROOT is not a prepared stream root" >&2; exit 1; }
[ -e "$RUN_DIR/LATEST.json" ] || { echo "$RUN_DIR has no LATEST.json" >&2; exit 1; }
if [ -e "$ROOT/results/$RUN.json" ]; then echo "$ROOT/results/$RUN.json exists: the run is finished" >&2; exit 1; fi
if [ -e "$RUN_DIR/FAILED" ]; then  # as pilot.sh does: a rerun clears the marker and resumes
  echo "clearing $RUN_DIR/FAILED ($(cat "$RUN_DIR/FAILED"))"; rm -f "$RUN_DIR/FAILED"
fi
# The run directory lives outside ROOT: ablate writes to $ROOT/runs/<run>.
if [ -L "$ROOT/runs" ]; then
  [ "$(cd "$ROOT/runs" && pwd -P)" = "$(cd "$RUNS" && pwd -P)" ] || { echo "$ROOT/runs points elsewhere" >&2; exit 1; }
elif [ -e "$ROOT/runs" ]; then
  echo "$ROOT/runs exists and is not a symlink to $RUNS; move it aside first" >&2; exit 1
else
  ln -s "$RUNS" "$ROOT/runs"
fi
# The tokenizer must be the run's own (ids are baked into the heads).
if [ -d "$RUN_DIR/tok" ] && ! diff -rq "$RUN_DIR/tok" "$ROOT/tok" >/dev/null; then
  echo "$ROOT/tok differs from the run's tokenizer $RUN_DIR/tok" >&2; exit 1
fi
# The newest checkpoint must match its manifest entry.
"$PY" - "$RUN_DIR" <<'EOF'
import hashlib, json, sys
from pathlib import Path
run = Path(sys.argv[1])
latest = json.loads((run / "LATEST.json").read_text())["checkpoint"]
digest = hashlib.sha256((run / latest["file"]).read_bytes()).hexdigest()
if digest != latest["sha256"]:
    sys.exit(f"{latest['file']} does not match LATEST.json (sha256 {digest})")
print(f"checkpoint {latest['file']} (step {latest['step']}) verified")
EOF
missing=$("$PY" -c "from indicear.augment import WORKER_CODECS, available_codecs; print(' '.join(sorted(set(WORKER_CODECS) - available_codecs())))")
if [ -n "$missing" ]; then
  echo "WARNING: ffmpeg cannot encode: $missing. The A100 run drew GSM and Opus; here codec draws go to the rest."
  echo "         An ffmpeg built with libgsm (and libopencore-amrnb) on PATH restores the original codec mix."
  if [ -z "${ALLOW_CODEC_MISMATCH:-}" ]; then echo "set ALLOW_CODEC_MISMATCH=1 to start anyway" >&2; exit 1; fi
fi
# MPS memory cap -> allocator watermarks (ratios of torch.mps.recommended_max_memory()).
read -r high low < <("$PY" -c "
import sys, torch
r = float(sys.argv[1]) * 2**30 / torch.mps.recommended_max_memory()
print(f'{min(r, 1.0):.3f} {max(min(r, 1.0) - 0.1, 0.1):.3f}')" "$MPS_MEM_GB")
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-$high}"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-$low}"

args=(--root "$ROOT" train --lower live --variants "$RUN" --train-hours "$TRAIN_HOURS"
      --device mps --precision "$PRECISION" --stream-workers "$STREAM_WORKERS")
[ "$SPLITS" != 0 ] && args+=(--splits "$SPLITS")
[ "$COMPILE" = 0 ] && args+=(--no-compile)
[ "$KEEP_SHARDS" = 1 ] && args+=(--keep-shards)
[ "$LENIENT_RESTORE" = 1 ] && args+=(--lenient-stream-restore)
# shellcheck disable=SC2206
args+=(${EXTRA})
prefetch=""
if [ "$KEEP_SHARDS" = 1 ] && [ "$PREFETCH" = 1 ]; then
  prefetch=$(printf '%q ' "$PY" -m indicear.ablate --root "$ROOT" train --lower live --variants "$RUN" \
    --train-hours "$TRAIN_HOURS" --prefetch-pool)
fi
{
  echo "$(date '+%F %T') resume_local: MPS cap ${MPS_MEM_GB} GiB (watermarks $PYTORCH_MPS_HIGH_WATERMARK_RATIO/$PYTORCH_MPS_LOW_WATERMARK_RATIO), torch threads $torch_threads + $STREAM_WORKERS stream workers, nice 10, swap limit ${SWAP_LIMIT_GB} GiB"
  [ -n "$prefetch" ] && echo "$(date '+%F %T') resume_local: first $prefetch"
  echo "$(date '+%F %T') resume_local: $PY -m indicear.ablate ${args[*]}"
} | tee -a "$OUT"

# Always detached: a terminal Ctrl-C would also kill the trainer's ffmpeg children mid-codec; use --stop.
# The wrapper execs the trainer, so the recorded pid is the trainer's once the pool download is done.
nohup nice -n 10 bash -c "${prefetch:+$prefetch || exit \$?; }exec $(printf '%q ' "$PY" -m indicear.ablate "${args[@]}")" \
  >> "$OUT" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"
nohup caffeinate -i -w "$pid" >/dev/null 2>&1 &   # keep the Mac awake (lid open or on power) while it runs
# Host sampler: Apple GPU utilization and memory (ioreg), trainer RSS and swap, every 10 s.
nohup bash -c '
  pid=$1; out=$2
  [ -e "$out" ] || echo "time,gpu_util_percent,gpu_memory_mib,trainer_rss_mib,swap_used_mib" > "$out"
  while kill -0 "$pid" 2>/dev/null; do
    stats=$(ioreg -r -d 1 -w 0 -c IOAccelerator)
    util=$(printf "%s" "$stats" | grep -o "\"Device Utilization %\"=[0-9]*" | head -1 | cut -d= -f2)
    mem=$(printf "%s" "$stats" | grep -o "\"In use system memory\"=[0-9]*" | head -1 | cut -d= -f2)
    rss=$(ps -o rss= -p "$pid" | tr -d " ")
    swap=$(sysctl -n vm.swapusage | sed -E "s/.*used = ([0-9.]+)M.*/\1/")
    echo "$(date +%s),${util:-},$(( ${mem:-0} / 1048576 )),$(( ${rss:-0} / 1024 )),${swap:-}" >> "$out"
    sleep 10
  done' _ "$pid" "$RUN_DIR/host.csv" >/dev/null 2>&1 &
echo "started (pid $pid) under nice 10 and caffeinate; log: $OUT"
echo "stop with: bash $here/resume_local.sh --stop"
if [ "${1:-}" = "--follow" ]; then tail -n +1 -f "$OUT"; fi
