"""Live-encoder training for streamed data (`Trainer(lower="live")`).

Streamed rows cannot be cached, so each step runs the whole encoder:

    waveforms --GpuLogMel (Featurizer)--> log-mel --layers 1-6, torch.no_grad, tower dtype-->
    h6 --layers 7-12 + adapters + heads, bf16 autocast, trained as on the cache path--> losses

Layers 1-6 run exactly as `indicear.cache.build_cache` runs them (no autocast, the tower's own
dtype: bf16 on a GPU VM), and padding frames are zeroed as `RowCache.batch` does, so a live step
sees the same layer-6 states as a cached one (tests/test_stream_train.py checks this). The cached
path stays the default (`lower="cache"`).

## The stream protocol

The trainer consumes any object with this shape (`StreamSource`):

    iter(stream)               -> iterator of batch dicts (called once per run)
    stream.state_dict()        -> dict: the position *after* the most recently yielded batch
    stream.load_state_dict(s)  -> resume so the next batch yielded is the one after `s`
    stream.set_frame_budget(n) -> optional: pack batches so len(batch) * longest <= n encoder
                                  frames (`samples_to_frames`); called before load_state_dict
    stream.fingerprint()       -> optional: JSON-able identity of the data (mixture, seed, ...);
                                  it joins the run's config hash, so a resumed run refuses a
                                  different stream

`state_dict()` is called from the trainer's prefetch thread right after each `next()`, before
the next one, and stored with that batch; the checkpoint after training step n holds the state
captured with batch n, so batches the prefetcher pulled but never trained are replayed on
resume. The state must be picklable (it goes into the torch checkpoint).

Each batch is a dict:

    "waveforms"  float32 [B, N] CPU, 16 kHz mono, zero-padded (the gpu-backend FeatureBatches
                 item; `indicear.features.pad_waveforms` builds it)
    "lengths"    int64 [B]: samples per row
    "text"       list[str] of B normalized transcripts ("" for non-speech)
    "lang"       list[str] of B: one of indicear.lid.LID_LANGS, or "none" for non-speech
    "word_langs" optional list of B (list[str] | None): per-word language of code-mixed rows
    "epoch"      optional int: logged in train.csv
    "key"        optional list[str]: ignored by the trainer
    "state"      optional: the stream position right after this batch; when present the trainer
                 uses it instead of calling state_dict()

Instead of "text"/"lang"/"word_langs" lists a batch may carry "items" (B row dicts with those
keys), as `indicear.stream.StreamLoader` yields; `LoaderSource` adapts that loader to the protocol.

Rows are tokenized in the prefetch thread with the trainer's `text_encoder(text) -> (tokens,
phonetic units)`; an encoder may return None to reject a row (e.g. a transcript that does not
encode). Rejected rows and rows whose transcript cannot fit their frames under CTC are dropped
before layer 6, and counted.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Protocol, runtime_checkable

import torch

HOP = 160  # log-mel hop (samples)
SUBSAMPLE = 4  # encoder frames per log-mel frame (two stride-2 convolutions)
MAX_SAMPLES = 480_000  # the Gemma 4 extractor truncates at 30 s
REQUIRED_KEYS = ("waveforms", "lengths", "text", "lang")
TextEncoder = Callable[[str], tuple[Sequence[int], Sequence[int]]]


@runtime_checkable
class StreamSource(Protocol):
    def __iter__(self) -> Iterator[dict]: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...


def samples_to_frames(samples: int | torch.Tensor):
    """Encoder frames for a waveform of `samples` (exact for the Gemma 4 extractor + tower:
    ceil(((n - 1) // 160) / 4), with the extractor's 30 s truncation)."""
    if isinstance(samples, torch.Tensor):
        mel = (samples.clamp(1, MAX_SAMPLES) - 1) // HOP
        return (mel + SUBSAMPLE - 1) // SUBSAMPLE
    mel = (min(max(int(samples), 1), MAX_SAMPLES) - 1) // HOP
    return (mel + SUBSAMPLE - 1) // SUBSAMPLE


def frames_to_samples(frames: int) -> int:
    """The shortest waveform with `frames` encoder frames."""
    return (SUBSAMPLE * (frames - 1) + 1) * HOP + 1 if frames > 0 else 0


def _from_items(batch: dict) -> dict:
    if "text" in batch or "items" not in batch:
        return batch
    items = batch["items"]
    return {**batch, "text": [i["text"] for i in items], "lang": [i["lang"] for i in items],
            "word_langs": [i.get("word_langs") for i in items], "key": [i.get("key") for i in items],
            "view": [int(i.get("view", 0)) for i in items]}


def validate_batch(batch: dict) -> int:
    missing = [k for k in REQUIRED_KEYS if k not in batch]
    if missing:
        raise ValueError(f"stream batch lacks {missing}; see indicear.stream_train for the protocol")
    rows = len(batch["text"])
    if batch["waveforms"].ndim != 2 or batch["waveforms"].shape[0] != rows or len(batch["lengths"]) != rows \
            or len(batch["lang"]) != rows or (batch.get("word_langs") is not None and len(batch["word_langs"]) != rows):
        raise ValueError("stream batch fields disagree on the number of rows")
    return rows


def prepare(batch: dict, text_encoder: TextEncoder | None, pin: bool) -> dict:
    """Prefetch-thread half of a live step: validate, tokenize, pin. Waveforms stay on the CPU
    (the featurizer copies them non-blocking; `Featurizer.check_once` reads them)."""
    batch = _from_items(batch)
    rows = validate_batch(batch)
    waveforms = batch["waveforms"].to(torch.float32)
    if pin:
        waveforms = waveforms.pin_memory()
    tokens, units, rejected = [], [], []
    for text in batch["text"]:
        encoded = ((), ()) if text_encoder is None else text_encoder(text)
        rejected.append(encoded is None)
        t, u = encoded if encoded is not None else ((), ())
        tokens.append(torch.as_tensor(list(t), dtype=torch.long))
        units.append(torch.as_tensor(list(u), dtype=torch.long))
    lengths = torch.as_tensor(batch["lengths"], dtype=torch.long)
    word_langs = batch.get("word_langs")
    return {
        "waveforms": waveforms,
        "lengths": lengths,
        "frames": samples_to_frames(lengths),
        "tokens": tokens,
        "units": units,
        "langs": list(batch["lang"]),
        "word_langs": list(word_langs) if word_langs is not None else [None] * rows,
        "rejected": rejected,
        "keys": list(batch["key"]) if batch.get("key") is not None else [None] * rows,
        "views": [int(v) for v in batch["view"]] if batch.get("view") is not None else [0] * rows,
        "epoch": int(batch.get("epoch", 0)),
    }


def slice_live(batch: dict, rows: Sequence[int]) -> dict:
    index = torch.tensor(list(rows), dtype=torch.long)
    lengths = batch["lengths"][index]
    longest = int(lengths.max()) if len(rows) else 0
    return {
        "waveforms": batch["waveforms"][index, :longest],
        "lengths": lengths,
        "frames": batch["frames"][index],
        "tokens": [batch["tokens"][r] for r in rows],
        "units": [batch["units"][r] for r in rows],
        "langs": [batch["langs"][r] for r in rows],
        "word_langs": [batch["word_langs"][r] for r in rows],
        **{k: [batch[k][r] for r in rows] for k in ("keys", "views", "conditions") if k in batch},
    }


def augment_live(batch: dict, augment) -> tuple[dict, int]:
    """Main-process augmentation of a prepared batch (indicear.gpu_augment.GpuAugment): waveforms
    move to the augment's device; lengths and frames follow speed perturbation; rows that no
    longer fit their transcript under CTC are dropped (returned count). Rows need "keys" and
    "views" (from the stream's items); without keys nothing is augmented."""
    if any(k is None for k in batch["keys"]):
        return batch, 0
    speech = [len(t) > 0 for t in batch["tokens"]]
    for attempt in range(2):
        try:
            waveforms, lengths, conditions = augment(batch["waveforms"], batch["lengths"], batch["keys"],
                                                     batch["views"], speech)
            break
        except RuntimeError as error:  # cuFFT or device memory pressure: free caches and retry, else train clean
            if "cuFFT" not in str(error) and "out of memory" not in str(error):
                raise
            from indicear.runtime.gpu import empty_device_cache

            empty_device_cache()
            if torch.cuda.is_available():
                torch.backends.cuda.cufft_plan_cache.clear()
            if attempt:
                print(f"augmentation skipped for one batch after a GPU error: {error}", flush=True)
                return batch, 0
    out = {**batch, "waveforms": waveforms, "lengths": lengths, "frames": samples_to_frames(lengths),
           "conditions": conditions}
    keep = [i for i, ok in enumerate(ctc_feasible(out["tokens"], out["frames"].tolist())) if ok]
    if len(keep) == len(conditions):
        return out, 0
    if not keep:  # never leave an empty step: train this batch clean instead
        return batch, 0
    extra = {k: out[k] for k in ("epoch", "stream_state", "dropped") if k in out}
    return {**slice_live(out, keep), **extra}, len(conditions) - len(keep)


def pad_features(features: torch.Tensor, mask: torch.Tensor, frame_bucket: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad log-mel features (zeros, mask False) so the encoder sees a multiple of
    `frame_bucket` frames (SUBSAMPLE log-mel frames each). Exact for the valid frames: the tower
    has no right context and a causal convolution (tests/test_encoder.py), and every loss reads
    only each row's own frames. It bounds the shapes a compiled tower sees."""
    step = SUBSAMPLE * frame_bucket
    target = -(-max(1, features.shape[1]) // step) * step
    extra = target - features.shape[1]
    if extra <= 0:
        return features, mask
    return (torch.nn.functional.pad(features, (0, 0, 0, extra)), torch.nn.functional.pad(mask, (0, extra)))


class LiveLower:
    """Featurizer + layers 1..6 under torch.no_grad in the tower's dtype (as the cache build
    runs them); padding frames zeroed (as the cache loader returns them). `autocast_dtype`
    (bf16/fp16 on MPS) runs layers 1-6 under autocast instead; `frame_bucket` > 0 pads the frames
    to a multiple of it (a compiled tower then sees a few static shapes)."""

    def __init__(self, staged, featurizer, autocast_dtype: torch.dtype | None = None, frame_bucket: int = 0):
        self.staged, self.featurizer, self.autocast_dtype = staged, featurizer, autocast_dtype
        self.frame_bucket = frame_bucket

    @torch.no_grad()
    def __call__(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        features, mask = self.featurizer({"waveforms": batch["waveforms"], "lengths": batch["lengths"]})
        if self.frame_bucket:
            features, mask = pad_features(features, mask, self.frame_bucket)
        with torch.autocast(device_type=features.device.type, dtype=self.autocast_dtype or torch.float32,
                            enabled=self.autocast_dtype is not None):
            hidden, valid = self.staged.lower(features, mask)
        hidden = hidden * valid[..., None].to(hidden.dtype)
        return hidden, valid.sum(-1)


def ctc_feasible(tokens: Sequence[torch.Tensor], frames: Sequence[int]) -> list[bool]:
    """len(tokens) + adjacent repeats <= frames (the cache path's make_examples filter)."""
    out = []
    for target, n in zip(tokens, frames, strict=True):
        repeats = int((target[1:] == target[:-1]).sum()) if len(target) > 1 else 0
        out.append(len(target) + repeats <= n)
    return out


def synthetic_batch(model, rows: int, frames: int, seed: int = 0) -> dict:
    """A worst-case prepared live batch for VRAM autotuning: `rows` noise waveforms of `frames`
    encoder frames each, a dense transcript (one token per 4 frames, in the largest script
    range) and phonetic units (one per 3 frames)."""
    generator = torch.Generator().manual_seed(seed)
    samples = frames_to_samples(frames)
    waveforms = 0.1 * torch.randn(rows, samples, generator=generator)
    lengths = torch.full((rows,), samples, dtype=torch.long)
    if model.final is not None:
        start, end = max(model.script_ranges, key=lambda r: r[1] - r[0])
        tokens = [torch.randint(start, end, (max(1, frames // 4),), generator=generator) for _ in range(rows)]
        units = [torch.randint(2, max(3, model.num_units), (max(1, frames // 3),), generator=generator) for _ in range(rows)]
    else:
        tokens = units = [torch.zeros(0, dtype=torch.long) for _ in range(rows)]
    return {"waveforms": waveforms, "lengths": lengths, "frames": samples_to_frames(lengths), "tokens": tokens,
            "units": units, "langs": ["hi"] * rows, "word_langs": [None] * rows, "epoch": 0}


class ListStream:
    """A `StreamSource` over an in-memory list of batches (tests, small local runs). Its state
    is the number of batches yielded; `set_frame_budget` is recorded, not applied."""

    def __init__(self, batches: Sequence[dict], name: str = "list"):
        self.batches, self.name = list(batches), name
        self.position = 0
        self.frame_budget: int | None = None
        self.yielded: list[int] = []  # batch indices handed out (for tests)

    def __iter__(self) -> Iterator[dict]:
        while self.position < len(self.batches):
            index = self.position
            self.position += 1
            self.yielded.append(index)
            yield self.batches[index]

    def state_dict(self) -> dict:
        return {"position": self.position}

    def load_state_dict(self, state: dict) -> None:
        self.position = int(state["position"])

    def set_frame_budget(self, frames: int) -> None:
        self.frame_budget = frames

    def fingerprint(self) -> dict:
        return {"name": self.name, "batches": len(self.batches)}


class LoaderSource:
    """Adapts `indicear.stream.StreamLoader` (batches carry "items" and their own "state"; the
    planner restores with `load_state`) to `StreamSource`. Duck-typed: nothing is imported."""

    def __init__(self, loader, fingerprint: dict | None = None):
        self.loader, self._fingerprint = loader, fingerprint

    def __iter__(self) -> Iterator[dict]:
        return iter(self.loader)

    def state_dict(self) -> dict:
        return self.loader.state if self.loader.state is not None else self.loader.planner.state()

    def load_state_dict(self, state: dict) -> None:
        self.loader.planner.load_state(state)
        self.loader.state = state

    def set_frame_budget(self, frames: int) -> None:
        self.loader.planner.frame_budget = frames

    def fingerprint(self) -> dict | str:
        return self._fingerprint if self._fingerprint is not None else type(self.loader).__name__
