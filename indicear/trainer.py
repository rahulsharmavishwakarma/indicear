"""Training loop on cached layer-6 states (layers 7-12 run every step).

- Dynamic batches: length-bucketed, filled to a padded-frame budget. A fresh run sizes the
  budget from free VRAM (`vram_capacity`) by growing it until a real worst-case step peaks at
  `vram_target`; a resumed run reuses the stored budget. CUDA or MPS OOM halves the micro-batch
  (`OomBackoff`) and retries; the split count is checkpointed with the device type.
- Micro-batches (`splits`): a batch split into micro-batches accumulates gradients, and every
  loss term of a micro-batch is normalized by the WHOLE batch's denominator (`DesignA.loss_norms`:
  target tokens, phonetic units, frames, LID weights), so the accumulated gradient is the
  one-piece gradient (up to float rounding and the per-micro-batch SpecAugment draw). The split
  count is a runtime choice, never hashed: `TrainConfig.splits` (--splits) sets it, else a resumed
  run keeps the stored count on the same device type and re-derives it on another
  (`auto_splits`: 1 on CUDA, from torch.mps.recommended_max_memory() on MPS).
- Devices: CUDA (bf16 autocast, TF32, fused AdamW, torch.compile), Apple MPS (float32, no
  autocast, no TF32, eager, fused AdamW; the log-mel and the main-process augmentation run on the
  CPU, see indicear.encoder.feature_device and indicear.gpu_augment.augment_device) and the CPU.
  `precision="fp16"` (MPS only) runs layers 1-12 under float16 autocast with a dynamic
  GradScaler (its state is checkpointed as "grad_scaler"); weights, optimizer, log-softmax and
  CTC stay float32. Precision is a runtime choice and never enters the config hash. With
  `compile_heads` (the default; --no-compile turns it off) MPS compiles every tower layer and the
  CTC heads (`compile_tower_layers`), pads live micro-batches to MPS_FRAME_BUCKET frames and marks
  the batch dimension dynamic; evaluation runs eager (`eager()`), so its shapes never compile.
- GPU saturation: a background thread (free-threaded Python: no GIL contention) gathers the next
  batch from the cache and copies it non-blocking; bf16 autocast; TF32; heads compiled with
  `torch.compile` when it works on this Python/torch (eager fallback otherwise); gpu.csv at 5 s.
- Resumable: atomic checkpoints every `ckpt_every_steps` and every `ckpt_every_seconds`, plus on
  SIGTERM/SIGINT; rerunning the same command continues from the latest valid checkpoint.
  `INDICEAR_TEST_KILL_AT_STEP` sends SIGTERM to the process after that step (used by the tests).
- Two lower-stack modes, `Trainer(lower=...)`: "cache" (default; examples index the layer-6
  cache) and "live" (streamed waveforms; the GPU log-mel and layers 1-6 run every step under
  torch.no_grad, see indicear.stream_train for the stream protocol). Live runs last
  `TrainConfig.max_steps` steps, size the budget on a synthetic `live_max_frames` worst case, and
  checkpoint the stream's position (`sampler.stream`) instead of the bucket sampler's. A live run
  either has a fixed `max_steps` or a target of `train_hours` of audio, converted to steps once the
  frame budget is fixed (`live_steps`: an estimate, since a streamed batch fills only part of
  the budget; the summary reports the hours actually trained and the measured packing).
- `LiveGroup`: several live Trainers (one per variant) fed by ONE stream in one process. Each
  batch is decoded, augmented and run through the log-mel and layers 1-6 once; each member then
  runs its own layers 7-12 (its adapters are attached only while it runs), heads, backward and
  optimizer step. Every member keeps its own run directory, checkpoints, train.csv and gpu.csv,
  with exactly the checkpoint format and config hash of a solo live run, so a member can also be
  resumed alone. A single live Trainer runs as a group of one.
- Optional LID heads (indicear.lid): batches carry each row's `lang` and `word_langs`, from which
  the LID targets are built; the transcript losses never depend on them.
"""

from __future__ import annotations

import contextlib
import csv
import math
import os
import queue
import signal
import threading
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from indicear.common import FRAME_RATE, GpuSampler
from indicear.lid import make_targets
from indicear.runtime.batching import (
    BucketFrameSampler,
    OomBackoff,
    StepSchedule,
    autotune_frame_budget,
    scale_lr,
    vram_capacity,
    worst_case_batch,
)
from indicear.runtime.checkpoint import (
    CheckpointManager,
    StopFlag,
    config_hash,
    git_revision,
    rng_state,
    set_rng_state,
)
from indicear.stream_train import (
    LiveLower,
    augment_live,
    ctc_feasible,
    prepare,
    slice_live,
    synthetic_batch,
)

LOWER_MODES = ("cache", "live")
MPS_FRAME_BUCKET = 48  # encoder frames (1.92 s; a multiple of the attention chunk, 12): compiled-shape bucket
PRECISIONS = ("auto", "fp32", "bf16", "fp16")


def autocast_dtype(device: torch.device, dtype: torch.dtype, precision: str = "auto") -> torch.dtype | None:
    """The autocast dtype of layers 1-12 and the heads on MPS (of layers 7-12 and the heads on
    CUDA), or None for none: CUDA autocasts to the run dtype (bf16, as before); MPS runs float32
    unless precision is "bf16" or "fp16" (half-precision autocast over float32 weights); the CPU
    runs float32."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision {precision!r} not in {PRECISIONS}")
    if device.type == "cuda":
        if precision in ("bf16", "fp16"):
            raise ValueError(f'precision "{precision}" is for MPS; CUDA autocasts to the run dtype (bf16)')
        return dtype
    if precision in ("bf16", "fp16"):
        if device.type != "mps":
            raise ValueError(f'precision "{precision}" needs MPS')
        return torch.bfloat16 if precision == "bf16" else torch.float16
    return None


@dataclass
class Example:
    keys: tuple[str, ...]  # cache key per view; view 0 is clean audio
    tokens: torch.Tensor
    units: torch.Tensor
    frames: int
    lang: str
    weight: float = 1.0  # expected draws per epoch (temperature sampling); 1.0 = once
    word_langs: tuple[str, ...] | None = None  # per-word language of a code-mixed row (LID heads)


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    ref_frames: int = 20_000  # LR is scaled by sqrt(budget / ref_frames)
    warmup_steps: int = 300
    min_lr_ratio: float = 0.05
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    seed: int = 0
    num_buckets: int = 30
    frame_budget: int = 0  # 0: size from free VRAM at startup
    vram_share: float = 1.0  # fraction of the GPU this job may use (concurrent jobs)
    vram_target: float = 0.85
    tune_start: int = 4_000
    tune_max: int = 400_000
    cpu_budget: int = 2_000
    log_every: int = 25
    eval_every: int = 1_000
    ckpt_every_steps: int = 500
    ckpt_every_seconds: float = 600.0
    compile_heads: bool = True
    max_steps: int = 0  # lower="live": optimizer steps in the run (streams have no epochs); hashed
    train_hours: float = 0.0  # lower="live", max_steps 0: audio hours to train on (-> steps, see live_steps)
    packing: float = 0.85  # assumed real frames / frame budget of a streamed batch (step estimate only)
    live_max_frames: int = 750  # lower="live": longest row (frames, 25 fps) for VRAM autotuning
    splits: int = 0  # micro-batches per step; 0: the stored count, or auto_splits on a new device (not hashed)
    peak_lr: float = 0.0  # > 0: the peak LR itself, without the sqrt(budget / ref_frames) scaling of `lr`
    final_lr: float = 0.0  # > 0: the cosine floor as an absolute LR (else min_lr_ratio x the peak)
    new_lr_mult: float = 1.0  # LR multiplier of the rows and parameters a warm start left fresh (indicear.warmstart)
    extra: dict = field(default_factory=dict)  # variant config and data fingerprint (hashed)

    def semantic(self) -> dict:
        """Fields that define the run; logging/checkpoint cadence and VRAM sizing do not. The LR
        options added for warm starts enter only when set, so earlier runs keep their hashes."""
        keys = ("epochs", "lr", "ref_frames", "warmup_steps", "min_lr_ratio", "weight_decay",
                "grad_clip", "seed", "num_buckets", "extra")
        out = {k: getattr(self, k) for k in keys}
        defaults = {"peak_lr": 0.0, "final_lr": 0.0, "new_lr_mult": 1.0}
        out.update({k: getattr(self, k) for k, v in defaults.items() if getattr(self, k) != v})
        return out

    def peak(self, budget: int) -> float:
        """The run's peak LR: `peak_lr` when set, else `lr` scaled by sqrt(budget / ref_frames)."""
        return self.peak_lr if self.peak_lr > 0 else scale_lr(self.lr, budget, self.ref_frames)


class RowLr:
    """A per-row LR multiplier inside shared parameters: the rows a warm start added to an output
    layer (new scripts' token and class rows, new embeddings) live in the same tensor as the
    copied rows, so they cannot get their own optimizer group. AdamW's update of an element is
    lr x (its Adam direction + decoupled weight decay), so scaling that element's update by `mult`
    after the step is exactly a step with `mult` x lr for those rows (the optimizer state is the
    same either way). Rows are `[[start, end), ...]` along dim 0 per parameter name."""

    def __init__(self, model: nn.Module, rows: dict[str, list[list[int]]], mult: float):
        params = dict(model.named_parameters())
        self.mult = mult
        self.items = []
        for name, spans in rows.items():
            if name not in params or not params[name].requires_grad:
                continue
            index = torch.cat([torch.arange(a, b) for a, b in spans]).to(params[name].device)
            self.items.append((params[name], index))

    def __bool__(self) -> bool:
        return bool(self.items) and self.mult != 1.0

    def before(self) -> list[torch.Tensor]:
        return [param.detach()[index].clone() for param, index in self.items]

    @torch.no_grad()
    def after(self, snapshot: list[torch.Tensor]) -> None:
        for (param, index), old in zip(self.items, snapshot, strict=True):
            new = param[index]
            param[index] = old + self.mult * (new - old)


def live_steps(train_hours: float, frame_budget: int, packing: float) -> int:
    """Optimizer steps for `train_hours` of audio at `frame_budget` padded frames per step, of
    which a share `packing` holds real audio frames (25 fps). An estimate: the stream packs batches
    by duration, so real packing varies a little; the run reports what it actually trained."""
    if train_hours <= 0 or frame_budget <= 0 or not 0 < packing <= 1:
        raise ValueError("live_steps needs train_hours > 0, frame_budget > 0 and 0 < packing <= 1")
    return max(1, math.ceil(train_hours * 3600 * FRAME_RATE / (frame_budget * packing)))


def setup_torch(device: str | torch.device = "cuda") -> None:
    """TF32 matmuls on CUDA; MPS and the CPU keep full float32."""
    if torch.device(device).type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def initial_splits(cfg: TrainConfig, device: torch.device, budget: int, state: dict | None,
                   log: Callable[[str], None], precision: str = "auto") -> int:
    """Micro-batches per step at the start of a run: `cfg.splits` when set; else the checkpoint's
    count when it was written on this device type with this precision (checkpoints without a
    device are CUDA runs); else `auto_splits` for this device, precision and budget. Never part
    of the config hash."""
    from indicear.runtime.gpu import auto_splits

    same_host_kind = state is not None and (state.get("device", "cuda"), state.get("precision", "auto")) == (
        device.type, precision)
    if cfg.splits > 0:
        splits, source = cfg.splits, "requested"
    elif same_host_kind:
        return int(state["splits"])
    else:
        splits = auto_splits(device.type, budget, precision, compiled=cfg.compile_heads)
        source = f"auto for {device.type} {precision}{' compiled' if cfg.compile_heads and device.type == 'mps' else ''}"
    stored = (f"; the checkpoint has {state['splits']} ({state.get('device', 'cuda')}, "
              f"{state.get('precision', 'auto')})") if state is not None else ""
    log(f"micro-batches: {splits} per step of {budget} frames ({source}{stored}); losses normalized by the whole "
        "batch, so the accumulated gradient equals the one-piece gradient")
    return splits


def eager():
    """Run compiled modules eagerly in this block (evaluation and decoding: arbitrary shapes)."""
    return torch.compiler.set_stance("force_eager")


def try_compile(module: nn.Module, example: Callable[[], object], log: Callable[[str], None],
                device: str | torch.device = "cuda") -> None:
    """Compile in place (state_dict keys unchanged); fall back to eager if it fails here. CUDA
    only: MPS and the CPU run eager."""
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        return
    try:
        module.compile(dynamic=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            example()
        log(f"torch.compile: {type(module).__name__} compiled")
    except Exception as error:  # noqa: BLE001 - dynamo support varies by Python build
        module._compiled_call_impl = None
        log(f"torch.compile unavailable for {type(module).__name__} ({type(error).__name__}); eager")


class Trainer:
    def __init__(
        self,
        model,
        cache,
        examples: Sequence[Example],
        config: TrainConfig,
        run_dir: str | Path,
        *,
        copy_dir: str | Path | None = None,
        evaluate: Callable[[nn.Module], float] | None = None,
        force_new_run: bool = False,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        log: Callable[[str], None] = print,
        lower: str = "cache",
        precision: str = "auto",
        stream=None,
        featurizer=None,
        text_encoder: Callable[[str], tuple] | None = None,
        augment: Callable | None = None,
        warm=None,
    ):
        """`warm` (indicear.warmstart.WarmStart): the rows and parameters a warm start left fresh,
        which train at `TrainConfig.new_lr_mult` x the LR.

        lower="live": `cache` and `examples` are unused (pass None and []); `stream` follows
        indicear.stream_train.StreamSource; `text_encoder(text) -> (tokens, phonetic units)`;
        `featurizer` defaults to a "gpu"-backend `Featurizer` in the tower's dtype; `augment`
        (indicear.gpu_augment.GpuAugment) corrupts each batch on the device before the log-mel."""
        if lower not in LOWER_MODES:
            raise ValueError(f"lower {lower!r} not in {LOWER_MODES}")
        self.live = lower == "live"
        self.stream, self.text_encoder, self.augment = stream, text_encoder, augment
        self.warm = warm
        self.row_lr: RowLr | None = None
        self.dropped_rows = 0
        self.model, self.cache, self.examples, self.cfg = model, cache, list(examples or []), config
        self.run_dir = Path(run_dir)
        self.evaluate, self.force_new_run = evaluate, force_new_run
        self.run_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.run_dir / "train.log"

        def tee(message: str) -> None:  # every run keeps its own log beside its checkpoints
            log(message)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")

        self.log = tee
        self.device, self.dtype = torch.device(device), dtype
        from indicear.runtime.gpu import DeviceMemory

        self.memory = DeviceMemory(self.device)
        self.precision = precision
        self.autocast_dtype = autocast_dtype(self.device, dtype, precision)
        self.scaler = None
        half = self.device.type == "mps" and self.autocast_dtype in (torch.bfloat16, torch.float16)
        if half:
            from indicear.encoder import half_safe_tower

            half_safe_tower(model.staged.tower, self.autocast_dtype)
        if self.autocast_dtype == torch.float16:  # bf16 has float32's range: no loss scaling
            self.scaler = torch.amp.GradScaler(self.device.type, init_scale=2.0**14)
        self.skipped_steps = 0
        self.manager = CheckpointManager(
            self.run_dir, copy_dir, every_steps=config.ckpt_every_steps,
            every_seconds=config.ckpt_every_seconds, log=self.log,
        )
        if self.live:
            self._setup_live(featurizer)
            identity = stream.fingerprint() if hasattr(stream, "fingerprint") else type(stream).__name__
            fingerprint = {"lower": "live", "max_steps": config.max_steps, "stream": identity}
            if not config.max_steps:  # hours-based runs only, so fixed-step runs keep their hash
                fingerprint.update(train_hours=config.train_hours, packing=config.packing)
        else:
            fingerprint = {"examples": len(self.examples), "frames": sum(e.frames for e in self.examples),
                           "views": len(self.examples[0].keys) if self.examples else 0}
            if any(e.weight != 1.0 for e in self.examples):  # unweighted runs keep their old hash
                fingerprint["weighted_frames"] = round(sum(e.weight * e.frames for e in self.examples), 3)
        self.hash = config_hash({**config.semantic(), "data": fingerprint})
        self.lengths = [e.frames for e in self.examples]
        self.frame_budget = config.frame_budget or config.cpu_budget

    def _setup_live(self, featurizer) -> None:
        if self.stream is None:
            raise ValueError('lower="live" needs a stream (indicear.stream_train.StreamSource)')
        if self.cfg.max_steps <= 0 and self.cfg.train_hours <= 0:
            raise ValueError('lower="live" needs TrainConfig.max_steps > 0 or train_hours > 0')
        if self.text_encoder is None and getattr(self.model, "final", None) is not None:
            raise ValueError('lower="live" needs a text_encoder for the transcript heads')
        if featurizer is None:
            from indicear.encoder import Featurizer

            tower_dtype = next(self.model.staged.tower.parameters()).dtype
            featurizer = Featurizer("gpu", self.device, tower_dtype)
        # MPS half precision: layers 1-6 under the same autocast (CUDA keeps them in the tower dtype)
        self.live_lower = LiveLower(self.model.staged, featurizer,
                                    self.autocast_dtype if self.device.type == "mps" else None,
                                    MPS_FRAME_BUCKET if self.device.type == "mps" and self.cfg.compile_heads else 0)

    @property
    def has_lid(self) -> bool:
        return getattr(self.model, "lid", None) is not None

    # ------------------------------------------------------------ batches

    def view_key(self, index: int, epoch: int) -> str:
        keys = self.examples[index].keys
        if len(keys) == 1:
            return keys[0]
        return keys[zlib.crc32(f"{self.cfg.seed}:{epoch}:{index}".encode()) % len(keys)]

    def gather(self, indices: Sequence[int], epoch: int) -> dict:
        keys = [self.view_key(i, epoch) for i in indices]
        hidden, lengths = self.cache.batch(keys, self.device, self.dtype)
        return {
            "indices": list(indices),
            "hidden": hidden,
            "lengths": lengths,
            "tokens": [self.examples[i].tokens for i in indices],
            "units": [self.examples[i].units for i in indices],
            "langs": [self.examples[i].lang for i in indices],
            "word_langs": [self.examples[i].word_langs for i in indices],
        }

    @staticmethod
    def slice_batch(batch: dict, rows: Sequence[int]) -> dict:
        if "waveforms" in batch:
            return slice_live(batch, rows)
        rows_t = torch.tensor(list(rows), device=batch["lengths"].device)
        lengths = batch["lengths"][rows_t]
        longest = int(lengths.max())
        return {
            "hidden": batch["hidden"][rows_t, :longest],
            "lengths": lengths,
            "tokens": [batch["tokens"][r] for r in rows],
            "units": [batch["units"][r] for r in rows],
            "langs": [batch["langs"][r] for r in rows] if "langs" in batch else None,
            "word_langs": [batch["word_langs"][r] for r in rows] if "word_langs" in batch else None,
        }

    def prefetch(self, schedule: StepSchedule, start_step: int):
        """Background thread: gather batch n+1 while step n runs."""

        def produce():
            step = start_step
            for indices in schedule:
                yield step, self.gather(indices, schedule.epoch_of(step))
                step += 1

        return self._prefetch(produce())

    def prefetch_live(self, start_step: int, total_steps: int):
        """Background thread: pull, tokenize and pin stream batch n+1 while step n runs. Each
        batch carries the stream state captured right after it was pulled; rows whose
        transcript cannot fit their frames under CTC are dropped here (`dropped` counts them)."""
        pin = self.device.type == "cuda"

        def produce():
            step = start_step
            if step >= total_steps:
                return
            for raw in self.stream:
                batch = prepare(raw, self.text_encoder, pin)
                batch["stream_state"] = raw["state"] if "state" in raw else self.stream.state_dict()
                feasible = ctc_feasible(batch["tokens"], batch["frames"].tolist())
                keep = [i for i, ok in enumerate(feasible) if ok and not batch["rejected"][i]]
                dropped = len(batch["tokens"]) - len(keep)
                if dropped:
                    batch = {**slice_live(batch, keep), "epoch": batch["epoch"], "stream_state": batch["stream_state"]}
                if not keep:
                    self.dropped_rows += dropped
                    continue
                batch["dropped"] = dropped
                yield step, batch
                step += 1
                if step >= total_steps:
                    return

        return self._prefetch(produce())

    def _prefetch(self, items):
        out: queue.Queue = queue.Queue(maxsize=2)
        stop = threading.Event()

        def worker():
            try:
                for item in items:
                    if stop.is_set():
                        return
                    out.put(item)
            except BaseException as error:  # noqa: BLE001 - re-raised in the main thread
                out.put(error)
                return
            out.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            while True:
                item = out.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()

    # ------------------------------------------------------------ sizing

    def micro_step(self, batch: dict, weight: float, logs: dict, norms: dict | None = None) -> None:
        if self.live:  # layers 1-6 in the tower's dtype, outside autocast, as the cache build runs them
            hidden, lengths = self.live_lower(batch)
            hidden = hidden.to(self.dtype)
        else:
            hidden, lengths = batch["hidden"], batch["lengths"]
        self.loss_step(hidden, lengths, batch, weight, logs, norms)

    def loss_norms(self, batch: dict) -> dict | None:
        """The whole batch's loss denominators (`DesignA.loss_norms`), or None for a model
        without them (then a split batch weights micro-batches by their share of rows)."""
        norms = getattr(self.model, "loss_norms", None)
        if norms is None:
            return None
        lid = make_targets(batch["langs"], batch["word_langs"]) if self.has_lid else None
        return norms(batch["tokens"], batch["units"], batch["frames"] if "frames" in batch else batch["lengths"], lid)

    def loss_step(self, hidden: torch.Tensor, lengths: torch.Tensor, batch: dict, weight: float, logs: dict,
                  norms: dict | None = None) -> None:
        """Layers 7-12, heads, losses and backward on layer-6 states (cached or live). With
        `norms` (a micro-batch of a split batch) every term is normalized by the whole batch and
        the micro-batch weight is 1: the micro-batch losses then add up to the batch's."""
        kwargs = {}
        if self.has_lid:
            kwargs["lid"] = make_targets(batch["langs"], batch["word_langs"])
        if norms is not None:
            kwargs["norms"], weight = norms, 1.0
        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype or torch.float32,
                            enabled=self.autocast_dtype is not None):
            out = self.model.losses(hidden, lengths, batch["tokens"], batch["units"], **kwargs)
        self.memory.note()
        if out["loss"].requires_grad:
            loss = out["loss"] * weight
            (self.scaler.scale(loss) if self.scaler is not None else loss).backward()
        self.memory.note()
        for key, value in out.items():
            logs[key] = logs.get(key, 0.0) + weight * float(value.detach())

    def optimizer_step(self, optimizer, scheduler) -> None:
        """Clip, step, schedule, zero the gradients. A step with inf/nan gradients is skipped and
        counted: by the GradScaler under fp16, else by the gradient-norm check."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip or float("inf"))
        if self.scaler is None and not torch.isfinite(norm):
            # bf16/fp32 have no GradScaler to skip a bad step: one inf/nan gradient would be
            # written into the weights (and both rolling checkpoints within minutes).
            self.skipped_steps += 1
            self.log(f"non-finite gradient norm ({float(norm)}): optimizer step skipped "
                     f"({self.skipped_steps} so far)")
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            return
        snapshot = self.row_lr.before() if self.row_lr else None
        if self.scaler is not None:
            scale = self.scaler.get_scale()
            self.scaler.step(optimizer)
            self.scaler.update()
            if self.scaler.get_scale() < scale:
                self.skipped_steps += 1
                self.log(f"fp16: non-finite gradients, optimizer step skipped; loss scale {self.scaler.get_scale():g}")
        else:
            optimizer.step()
        if snapshot is not None:
            self.row_lr.after(snapshot)
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    def restore_scaler(self, state: dict) -> None:
        if self.scaler is not None and state.get("grad_scaler"):
            self.scaler.load_state_dict(state["grad_scaler"])

    def autotune(self) -> int:
        if self.device.type != "cuda":
            return self.cfg.cpu_budget
        capacity = vram_capacity(self.cfg.vram_share)
        self.log(f"autotune: capacity {capacity / 2**30:.1f} GiB (share {self.cfg.vram_share})")

        def measure(budget: int) -> float:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if self.live:
                rows = max(1, budget // self.cfg.live_max_frames)
                batch = synthetic_batch(self.model, rows, self.cfg.live_max_frames)
            else:
                batch = self.gather(worst_case_batch(self.lengths, budget), 0)
            try:
                self.micro_step(batch, 1.0, {})
            finally:
                self.model.zero_grad(set_to_none=True)
                del batch
            return torch.cuda.max_memory_allocated() / capacity

        max_budget = self.cfg.tune_max if self.live else min(self.cfg.tune_max, sum(self.lengths))
        return autotune_frame_budget(measure, self.cfg.tune_start, max_budget, self.cfg.vram_target, log=self.log)

    # ------------------------------------------------------------ loop

    def optimizer(self, lr: float, total_steps: int):
        """AdamW with decay and no-decay groups; a warm start's fresh parameters get their own
        groups at `new_lr_mult` x lr and its fresh rows a `RowLr` (the groups of a run without a
        warm start are unchanged, so its checkpoints load as before)."""
        mult = self.cfg.new_lr_mult
        fresh = set(self.warm.fresh_params) if self.warm is not None and mult != 1.0 else set()
        decay, no_decay, fresh_decay, fresh_no_decay = [], [], [], []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                plain = parameter.ndim < 2 or "norm" in name
                if name in fresh:
                    (fresh_no_decay if plain else fresh_decay).append(parameter)
                else:
                    (no_decay if plain else decay).append(parameter)
        groups = [{"params": decay, "weight_decay": self.cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        if fresh:
            groups += [{"params": fresh_decay, "weight_decay": self.cfg.weight_decay, "lr": lr * mult},
                       {"params": fresh_no_decay, "weight_decay": 0.0, "lr": lr * mult}]
        optimizer = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.98), fused=self.device.type in ("cuda", "mps"))
        self.row_lr = RowLr(self.model, self.warm.new_rows, mult) if self.warm is not None else None
        warmup = min(self.cfg.warmup_steps, max(1, total_steps // 10))
        floor = min(1.0, self.cfg.final_lr / lr) if self.cfg.final_lr > 0 else self.cfg.min_lr_ratio

        def schedule(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    def _csv(self, resumed_step: int) -> tuple[Path, bool]:
        path = self.run_dir / "train.csv"
        if path.exists():  # drop rows past the checkpoint so a resumed log has no duplicates
            rows = list(csv.reader(path.open()))
            kept = [rows[0]] + [r for r in rows[1:] if r and int(r[0]) <= resumed_step] if rows else []
            with path.open("w", newline="") as handle:
                csv.writer(handle).writerows(kept)
            return path, bool(kept)
        return path, False

    def run(self) -> dict:
        if self.live:
            return LiveGroup([self]).run()[0]
        setup_torch(self.device)
        stop = StopFlag(self.log)
        try:
            return self._run(stop)
        finally:
            stop.restore()

    def compile_heads(self) -> None:
        if not self.cfg.compile_heads:
            return
        model = self.model
        if self.device.type == "mps":
            self.compile_mps()
            return
        dummy = torch.randn(2, 16, model.staged.hidden_size, device=self.device, dtype=self.dtype)
        heads = [m for m in (model.final, *(model.inter.values() if model.inter is not None else [])) if m is not None]
        for module in heads:  # the transducer's custom loss path stays eager
            try_compile(module, lambda m=module: m(dummy), self.log, self.device)

    def compile_mps(self) -> None:
        """MPS: compile the tower's layers (once per shared tower) and this model's CTC heads; the
        graphs compile on first use, per frame bucket (see MPS_FRAME_BUCKET)."""
        from indicear.encoder import compile_tower_layers

        model, staged = self.model, self.model.staged
        heads = tuple(m for m in (model.final, *(model.inter.values() if model.inter is not None else []))
                      if m is not None)
        if not staged.dynamic_batch:  # the tower is shared by a LiveGroup's members: compile it once
            compile_tower_layers(staged.tower)
            staged.dynamic_batch = True
        for head in heads:
            head.compile()
        self.log(f"torch.compile (MPS): {len(staged.tower.layers)} tower layers and {len(heads)} heads, static frame "
                 f"buckets of {MPS_FRAME_BUCKET} frames, dynamic batch; first steps per bucket include compilation")

    @property
    def loss_keys(self) -> list[str]:
        return ["loss", "ctc", "phonetic", "rnnt", "consistency"] + (["lid_utt", "lid_frame"] if self.has_lid else [])

    @property
    def csv_columns(self) -> list[str]:
        return ["step", "epoch", *self.loss_keys, "lr", "frames_per_s", "peak_gib", "splits", "elapsed_s"]

    def checkpoint_state(self, optimizer, scheduler, budget: int, splits: int, sampler_state: dict, **extra) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "grad_scaler": self.scaler.state_dict() if self.scaler is not None else None,  # fp16 on MPS only
            "precision": self.precision,  # not hashed
            "rng": rng_state(),
            "frame_budget": budget,
            "splits": splits,
            "device": self.device.type,  # the splits are this device's; not hashed
            "sampler": sampler_state,
            "config": {**asdict(self.cfg)},
            "config_hash": self.hash,
            "git": git_revision(),
            # a warm-started run carries its row map, so a VM without the source can resume it
            **({"warm_start": self.warm.to_json()} if self.warm is not None else {}),
            **extra,
        }

    def log_row(self, writer, handle, step: int, epoch: int, logs: dict, scheduler, frames_per_s: float,
                splits: int, elapsed: float) -> None:
        peak = self.memory.peak_bytes() / 2**30
        row = [step, epoch, *(round(logs.get(k, float("nan")), 4) for k in self.loss_keys),
               f"{scheduler.get_last_lr()[0]:.3e}", round(frames_per_s), round(peak, 2), splits, round(elapsed)]
        writer.writerow(row)
        handle.flush()
        self.log(" ".join(f"{c}={v}" for c, v in zip(self.csv_columns, row, strict=True)))

    def _run(self, stop: StopFlag) -> dict:
        """The cached-lower loop (live runs go through LiveGroup)."""
        model, cfg = self.model, self.cfg
        state = self.manager.restore(self.hash, self.force_new_run)
        if state is not None:
            if state["sampler"].get("lower") == "live":
                raise SystemExit("the checkpoint was written by a live-lower run, not a cached one")
            budget = state["frame_budget"]
        else:
            model.train()
            budget = cfg.frame_budget or self.autotune()
            torch.manual_seed(cfg.seed)
        splits = initial_splits(cfg, self.device, budget, state, self.log, self.precision)
        self.frame_budget = budget
        weights = [e.weight for e in self.examples]
        sampler = BucketFrameSampler(self.lengths, budget, cfg.num_buckets, seed=cfg.seed,
                                     weights=weights if any(w != 1.0 for w in weights) else None)
        total_steps = StepSchedule(sampler, cfg.epochs).total_steps
        lr = cfg.peak(budget)
        optimizer, scheduler = self.optimizer(lr, total_steps)
        step = 0
        if state is not None:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            self.restore_scaler(state)
            set_rng_state(state["rng"], self.log)
            step = state["step"]
            if state["sampler"]["order"] != self._order_fingerprint(sampler, state["sampler"]["epoch"]):
                raise SystemExit("data order differs from the checkpoint: data or sampler changed")
        self.log(
            f"run {self.run_dir.name}: {len(self.examples)} examples, budget {budget} frames, lr {lr:.2e}, "
            f"{total_steps} steps, starting at {step}, config {self.hash}"
        )
        self.compile_heads()
        backoff = OomBackoff(splits, log=self.log)
        csv_path, has_header = self._csv(step)
        handle = csv_path.open("a", newline="")
        writer = csv.writer(handle)
        if not has_header:
            writer.writerow(self.csv_columns)
        schedule = StepSchedule(sampler, cfg.epochs, start_step=step)
        model.train()
        started = time.time()
        window_frames, window_start = 0, time.time()

        def checkpoint(metric: float | None = None) -> None:
            epoch = schedule.epoch_of(max(0, step - 1))
            sampler_state = {"epoch": epoch, "batch_in_epoch": step - sum(schedule.epoch_sizes[:epoch]),
                             "order": self._order_fingerprint(sampler, epoch)}
            self.manager.save(self.checkpoint_state(optimizer, scheduler, budget, backoff.splits, sampler_state),
                              step, metric)

        with GpuSampler(self.run_dir / "gpu.csv") as gpu:
            for batch_step, batch in self.prefetch(schedule, step):
                assert batch_step == step
                epoch = schedule.epoch_of(step)
                logs: dict[str, float] = {}
                rows_in_batch = len(batch["tokens"])
                norms: list = []

                def run_chunk(rows, weight, batch=batch, logs=logs, norms=norms, rows_in_batch=rows_in_batch):
                    if len(rows) == rows_in_batch:  # one piece: the batch's own denominators
                        self.micro_step(self.slice_batch(batch, rows), weight, logs)
                        return
                    if not norms:
                        norms.append(self.loss_norms(batch))
                    self.micro_step(self.slice_batch(batch, rows), weight, logs, norms[0])

                def reset(logs=logs):
                    optimizer.zero_grad(set_to_none=True)
                    logs.clear()

                backoff.run(list(range(len(batch["tokens"]))), run_chunk, on_retry=reset)
                self.optimizer_step(optimizer, scheduler)
                step += 1
                window_frames += int(batch["lengths"].sum())
                if step % cfg.log_every == 0 or step == total_steps:
                    now = time.time()
                    self.log_row(writer, handle, step, epoch, logs, scheduler,
                                 window_frames / max(now - window_start, 1e-6), backoff.splits, now - started)
                    window_frames, window_start = 0, now
                metric = None
                if self.evaluate is not None and cfg.eval_every and step % cfg.eval_every == 0:
                    with eager():
                        metric = self.evaluate(model)
                    model.train()
                    self.log(f"step {step}: dev metric {metric:.3f}")
                if os.environ.get("INDICEAR_TEST_KILL_AT_STEP") == str(step):
                    os.kill(os.getpid(), signal.SIGTERM)
                if stop.stop or metric is not None or step == total_steps or self.manager.due(step):
                    checkpoint(metric)
                if stop.stop:
                    handle.close()
                    raise SystemExit(143)
            self.log(gpu.summary())
        handle.close()
        if state is not None and step == state["step"] and step >= total_steps:
            self.log("run already complete")
        return {"steps": step, "total_steps": total_steps, "frame_budget": budget, "lr": lr,
                "splits": backoff.splits, "config_hash": self.hash, "seconds": round(time.time() - started)}

    @staticmethod
    def _order_fingerprint(sampler: BucketFrameSampler, epoch: int) -> int:
        batches = sampler.batches_for_epoch(epoch)
        return zlib.crc32(repr(batches[:8]).encode())


# ---------------------------------------------------------------- live lower: one stream, several models


class LiveGroup:
    """Live-lower training of one or more Trainers on ONE stream (see the module docstring).

    Members must share the stream, the staged tower and every TrainConfig field except `extra`
    (the variant). With several members, every model's adapters are detached from the shared
    tower and attached only while that model runs, so the members never see each other's
    adapters; layers 1-6 carry no adapters, so their output is shared exactly.

    Resume: each member restores its own latest checkpoint. Members are saved one after another,
    so a VM lost mid-save can leave them one checkpoint apart; the group then restores every
    member at the newest step they all have (the manager keeps the last two). A member with no
    checkpoint while others have one restarts the whole group from step 0 (the others' files are
    moved aside), because the shared stream cannot be at two positions.
    """

    def __init__(self, trainers: Sequence[Trainer]):
        if not trainers:
            raise ValueError("LiveGroup needs at least one trainer")
        lead = trainers[0]
        shared_cfg = {k: v for k, v in asdict(lead.cfg).items() if k != "extra"}
        for trainer in trainers:
            if not trainer.live:
                raise ValueError('LiveGroup members must use lower="live"')
            if trainer.stream is not lead.stream or trainer.model.staged is not lead.model.staged:
                raise ValueError("LiveGroup members must share one stream and one staged tower")
            if {k: v for k, v in asdict(trainer.cfg).items() if k != "extra"} != shared_cfg:
                raise ValueError("LiveGroup members may differ only in TrainConfig.extra")
        if len({t.run_dir.resolve() for t in trainers}) != len(trainers):
            raise ValueError("LiveGroup members need distinct run directories")
        self.members, self.lead, self.cfg = list(trainers), lead, lead.cfg
        self.shared = len(self.members) > 1
        self.log = lead.log

    # ------------------------------------------------------------ helpers

    def _attached(self, trainer: Trainer):
        if not self.shared:
            return contextlib.nullcontext()
        return trainer.model.adapters_attached()

    def micro_step(self, batch: dict, weight: float, logs: Sequence[dict],
                   norms: Sequence[dict | None] | None = None) -> None:
        """Layers 1-6 once, then each member's layers 7-12, heads and backward. `norms`: each
        member's whole-batch denominators when `batch` is a micro-batch of a split batch."""
        hidden, lengths = self.lead.live_lower(batch)
        hidden = hidden.to(self.lead.dtype)
        norms = norms if norms is not None else [None] * len(self.members)
        for trainer, member_logs, member_norms in zip(self.members, logs, norms, strict=True):
            with self._attached(trainer):
                trainer.loss_step(hidden, lengths, batch, weight, member_logs, member_norms)

    def autotune(self) -> int:
        lead, cfg = self.lead, self.cfg
        if lead.device.type != "cuda":
            return cfg.cpu_budget
        capacity = vram_capacity(cfg.vram_share)
        self.log(f"autotune: capacity {capacity / 2**30:.1f} GiB (share {cfg.vram_share}), "
                 f"{len(self.members)} model(s) on one stream")
        token_model = next((t.model for t in self.members if t.model.final is not None), lead.model)

        def measure(budget: int) -> float:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            rows = max(1, budget // cfg.live_max_frames)
            batch = synthetic_batch(token_model, rows, cfg.live_max_frames)
            try:
                self.micro_step(batch, 1.0, [{} for _ in self.members])
            finally:
                for trainer in self.members:
                    trainer.model.zero_grad(set_to_none=True)
                del batch
            return torch.cuda.max_memory_allocated() / capacity

        return autotune_frame_budget(measure, cfg.tune_start, cfg.tune_max, cfg.vram_target, log=self.log)

    def _restore(self) -> list[dict | None]:
        states = [t.manager.restore(t.hash, t.force_new_run) for t in self.members]
        steps = [s["step"] if s is not None else None for s in states]
        if len(set(steps)) > 1:
            if None in steps:
                for trainer, state in zip(self.members, states, strict=True):
                    if state is not None:
                        self.log(f"{trainer.run_dir.name}: another member has no checkpoint; restarting the group")
                        trainer.manager._move_aside()
                states = [None] * len(states)
            else:
                target = min(steps)
                for index, trainer in enumerate(self.members):
                    if steps[index] != target:
                        states[index] = trainer.manager.restore(trainer.hash, trainer.force_new_run, step=target)
                        if states[index] is None:
                            raise SystemExit(f"{trainer.run_dir.name} has no checkpoint at step {target}, which the "
                                             "other members of this group resume from; run it alone or start over")
                self.log(f"group members were saved at steps {steps}; all resume at step {target}")
        for trainer, state in zip(self.members, states, strict=True):
            if state is not None and state["sampler"].get("lower") != "live":
                raise SystemExit(f"{trainer.run_dir.name}: the checkpoint was written by a cached-lower run")
        budgets = {s["frame_budget"] for s in states if s is not None}
        if len(budgets) > 1:
            raise SystemExit(f"group members were checkpointed with different frame budgets {sorted(budgets)}; "
                             "resume them separately")
        return states

    # ------------------------------------------------------------ loop

    def run(self) -> list[dict]:
        setup_torch(self.lead.device)
        stop = StopFlag(self.log)
        try:
            return self._run(stop)
        finally:
            stop.restore()

    def _run(self, stop: StopFlag) -> list[dict]:
        lead, cfg, members = self.lead, self.cfg, self.members
        if self.shared:
            for trainer in members:
                trainer.model.detach_adapters()
        states = self._restore()
        first = next((s for s in states if s is not None), None)
        if first is not None:
            budget = first["frame_budget"]
        else:
            for trainer in members:
                trainer.model.train()
            budget = cfg.frame_budget or self.autotune()
            torch.manual_seed(cfg.seed)
        splits = initial_splits(cfg, lead.device, budget, first, self.log, lead.precision)
        if first is not None and first.get("total_steps"):
            total_steps = int(first["total_steps"])
        elif cfg.max_steps:
            total_steps = cfg.max_steps
        else:
            total_steps = live_steps(cfg.train_hours, budget, cfg.packing)
            self.log(f"steps: {cfg.train_hours:g} h of audio at {budget} frames per step and an assumed packing "
                     f"of {cfg.packing:g} -> {total_steps} steps (an estimate; the summary reports the real hours)")
        lr = cfg.peak(budget)
        optimizers = []
        for trainer, state in zip(members, states, strict=True):
            trainer.frame_budget = budget
            optimizer, scheduler = trainer.optimizer(lr, total_steps)
            if state is not None:
                trainer.model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                trainer.restore_scaler(state)
            optimizers.append((optimizer, scheduler))
        step = first["step"] if first is not None else 0
        audio_seconds = float(first.get("audio_seconds", 0.0)) if first is not None else 0.0
        trained_frames = int(first.get("trained_frames", 0)) if first is not None else 0
        if first is not None:
            set_rng_state(first["rng"], self.log)
        if hasattr(lead.stream, "set_frame_budget"):
            lead.stream.set_frame_budget(budget)
        if first is not None:
            lead.stream.load_state_dict(first["sampler"]["stream"])
        stream_state = lead.stream.state_dict()
        names = ", ".join(t.run_dir.name for t in members)
        for trainer in members:
            trainer.log(f"run {trainer.run_dir.name}: streamed data (live lower; group: {names}), budget {budget} "
                        f"frames, lr {lr:.2e}, {total_steps} steps, starting at {step}, config {trainer.hash}")
            with self._attached(trainer):
                trainer.compile_heads()
        backoff = OomBackoff(splits, log=self.log)
        from indicear.runtime.gpu import HostMemoryGuard

        guard = HostMemoryGuard(float(os.environ.get("INDICEAR_SWAP_LIMIT_GB", "8")), log=self.log)
        guard.enabled = guard.enabled and lead.device.type == "mps"
        outputs = []
        for trainer in members:
            csv_path, has_header = trainer._csv(step)
            handle = csv_path.open("a", newline="")
            writer = csv.writer(handle)
            if not has_header:
                writer.writerow(trainer.csv_columns)
            outputs.append((writer, handle))
        for trainer in members:
            trainer.model.train()
        started = time.time()
        window_frames, window_start = 0, time.time()

        def checkpoint(metrics: Sequence[float | None]) -> None:
            for trainer, (optimizer, scheduler), metric in zip(members, optimizers, metrics, strict=True):
                state = trainer.checkpoint_state(
                    optimizer, scheduler, budget, backoff.splits, {"lower": "live", "stream": stream_state},
                    total_steps=total_steps, audio_seconds=audio_seconds, trained_frames=trained_frames)
                trainer.manager.save(state, step, metric)

        def close() -> None:
            for _, handle in outputs:
                handle.close()

        start_step = step
        with contextlib.ExitStack() as samplers:
            gpus = [samplers.enter_context(GpuSampler(t.run_dir / "gpu.csv")) for t in members]
            for batch_step, batch in lead.prefetch_live(step, total_steps):
                assert batch_step == step
                epoch = batch["epoch"]
                logs: list[dict[str, float]] = [{} for _ in members]
                lead.live_lower.featurizer.check_once(batch, self.log)  # on the clean CPU batch
                lead.dropped_rows += batch["dropped"]
                if lead.augment is not None:  # whole batch, before any OOM split: splits never change it
                    batch, dropped = augment_live(batch, lead.augment)
                    lead.dropped_rows += dropped

                rows_in_batch = len(batch["tokens"])
                norms: list = []

                def run_chunk(rows, weight, batch=batch, logs=logs, norms=norms, rows_in_batch=rows_in_batch):
                    if len(rows) == rows_in_batch:  # one piece: each loss uses the batch's own denominators
                        self.micro_step(slice_live(batch, rows), weight, logs)
                        return
                    if not norms:  # a split batch: every micro-batch is normalized by the whole batch
                        norms.extend(t.loss_norms(batch) for t in members)
                    self.micro_step(slice_live(batch, rows), weight, logs, norms)

                def reset(logs=logs):
                    for optimizer, _ in optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    for member_logs in logs:
                        member_logs.clear()

                backoff.run(list(range(len(batch["tokens"]))), run_chunk, on_retry=reset)
                for trainer, (optimizer, scheduler) in zip(members, optimizers, strict=True):
                    trainer.optimizer_step(optimizer, scheduler)
                if guard.check() and backoff.splits < backoff.max_splits:  # swap or critical memory pressure on the host
                    backoff.splits += 1
                    from indicear.runtime.gpu import empty_device_cache

                    empty_device_cache()
                    self.log(f"host memory pressure: {backoff.splits} micro-batches per step from now on")
                step += 1
                stream_state = batch["stream_state"]
                frames = int(batch["frames"].sum())
                window_frames += frames
                trained_frames += frames
                audio_seconds += float(batch["lengths"].sum()) / 16_000
                if step % cfg.log_every == 0 or step == total_steps:
                    now = time.time()
                    for trainer, (writer, handle), member_logs, (_, scheduler) in zip(
                            members, outputs, logs, optimizers, strict=True):
                        trainer.log_row(writer, handle, step, epoch, member_logs, scheduler,
                                        window_frames / max(now - window_start, 1e-6), backoff.splits, now - started)
                    window_frames, window_start = 0, now
                metrics: list[float | None] = [None] * len(members)
                if cfg.eval_every and step % cfg.eval_every == 0:
                    for index, trainer in enumerate(members):
                        if trainer.evaluate is None:
                            continue
                        with self._attached(trainer), eager():
                            metrics[index] = trainer.evaluate(trainer.model)
                        trainer.model.train()
                        trainer.log(f"step {step}: dev metric {metrics[index]:.3f}")
                if os.environ.get("INDICEAR_TEST_KILL_AT_STEP") == str(step):
                    os.kill(os.getpid(), signal.SIGTERM)
                evaluated = any(m is not None for m in metrics)
                if stop.stop or evaluated or step == total_steps or lead.manager.due(step):
                    checkpoint(metrics)
                if stop.stop:
                    close()
                    raise SystemExit(143)
            if step < total_steps and step > start_step:
                self.log(f"stream ended at step {step} of {total_steps}")
                checkpoint([None] * len(members))
            for trainer, gpu in zip(members, gpus, strict=True):
                trainer.log(gpu.summary())
        close()
        if step == start_step and step >= total_steps:
            self.log("run already complete")
        summary = {"steps": step, "total_steps": total_steps, "frame_budget": budget, "lr": lr,
                   "splits": backoff.splits, "seconds": round(time.time() - started), "lower": "live",
                   "dropped_rows": lead.dropped_rows, "audio_hours": round(audio_seconds / 3600, 3),
                   "packing_measured": round(trained_frames / (step * budget), 4) if step else None,
                   "group": [t.run_dir.name for t in members]}
        if not cfg.max_steps:
            summary["step_estimate"] = {"train_hours": cfg.train_hours, "packing_assumed": cfg.packing}
        return [{**summary, "config_hash": t.hash} for t in members]
