"""Layer-6 cache: run the frozen encoder's layers 1-6 once per utterance (and per augmented view).

On disk: `<dir>/part-NNNNN.npy` (bf16 stored as int16 bits, or int8 rows), `part-NNNNN.scale.npy`
(int8 only: per-row absmax / 127, float32) and `part-NNNNN.jsonl` (key, offset, frames), plus
`cache.json`. A part is written to a temporary name and renamed when complete, so an interrupted
build resumes by skipping finished parts.

Storage, completion markers and loading are the generic `indicear.runtime.shards` (RowCache); this
module holds what is specific to the encoder: which items, how they are encoded, and the call-path
check on real weights.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from indicear.common import CACHE_LAYER, EG2, FRAME_RATE, GpuSampler, write_json
from indicear.runtime.shards import (  # noqa: F401 (re-exported for callers)
    MmapRowCache,
    RowCache,
    bytes_per_row,
    check_disk,
    dequantize,
    plan_parts,
    quantize_int8,
    to_storage,
    write_part,
)

Layer6Cache = RowCache  # the layer-6 cache is a RowCache keyed by utterance (and view)


def group_by_padded_seconds(items: Sequence[dict], batch_seconds: float) -> list[list[dict]]:
    """Encoder batches: sorted by length, len(batch) * longest <= batch_seconds."""
    order = sorted(items, key=lambda item: item["seconds"])
    groups: list[list[dict]] = []
    current: list[dict] = []
    for item in order:
        longest = max([item["seconds"]] + [i["seconds"] for i in current[-1:]])
        if current and longest * (len(current) + 1) > batch_seconds:
            groups.append(current)
            current = []
        current.append(item)
    if current:
        groups.append(current)
    return groups


def build_cache(
    items: Sequence[dict],
    out_dir: str | Path,
    staged,
    *,
    storage: str = "bf16",
    part_frames: int = 400_000,
    batch_seconds: float = 600.0,
    workers: int = 8,
    transform: Callable | None = None,
    features: str | None = None,
    verify_first_batch: bool = True,
    gpu_augment: Callable | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Encode `items` (dicts with key, audio, seconds[, view]) through layers 1..6. Resumable.

    `transform(audio, rng, item)` runs in the worker processes (it must be picklable); the
    augmentation recipe uses `item["view"]` to decide how to corrupt each cached view.
    `features` ("cpu" or "gpu", default: gpu when CUDA is available) only chooses where the
    log-mel is computed; it is not part of the cache identity. `gpu_augment`
    (indicear.gpu_augment.GpuAugment, "gpu" features only) corrupts the views with view > 0 on the
    device before the log-mel, seeded per (key, view); `transform` then only runs the codecs.
    """
    from indicear.encoder import (
        FeatureBatches,
        Featurizer,
        default_feature_backend,
        feature_loader,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = plan_parts(items, part_frames, frame_estimate)
    pending = [i for i in range(len(parts)) if not (out_dir / f"part-{i:05d}.jsonl").exists()]
    meta = {
        "layer": CACHE_LAYER,
        "dim": staged.hidden_size,
        "storage": storage,
        "frame_rate": FRAME_RATE,
        "encoder": f"{EG2[0]}@{EG2[1]}",
        "parts": len(parts),
    }
    write_json(out_dir / "cache.json", meta)
    features = features or default_feature_backend()
    if gpu_augment is not None and features != "gpu":
        raise ValueError("GPU augmentation needs the gpu feature backend (waveform batches)")
    log(f"cache: {len(parts)} parts, {len(pending)} to build, storage {storage}, features on {features}")
    if not pending:
        return meta
    need = sum(estimate_bytes(parts[p], staged.hidden_size, storage) for p in pending)
    check_disk(out_dir, need, log)
    device = next(staged.tower.parameters()).device
    dtype = next(staged.tower.parameters()).dtype
    # Batches never cross a part, so a part is complete when its last batch arrives.
    groups: list[list[dict]] = []
    owner: list[int] = []
    for part in pending:
        for group in group_by_padded_seconds(parts[part], batch_seconds):
            groups.append(group)
            owner.append(part)
    remaining = {part: owner.count(part) for part in pending}
    buffers: dict[int, list[tuple[str, np.ndarray, np.ndarray | None]]] = {p: [] for p in pending}
    dataset = FeatureBatches(groups, transform=transform, backend=features)
    featurizer = Featurizer(features, device, dtype)
    started = time.time()
    audio_seconds = 0.0
    with GpuSampler(out_dir / "gpu.csv") as sampler:
        for index, batch in feature_loader(dataset, workers):
            if verify_first_batch:
                featurizer.check_once(batch, log)
            if gpu_augment is not None:
                group = groups[index]
                waveforms, lengths, _ = gpu_augment(
                    batch["waveforms"], batch["lengths"], [i["key"] for i in group],
                    [int(i.get("view", 0)) for i in group], [bool(i.get("text")) for i in group])
                batch = {"waveforms": waveforms, "lengths": lengths}
            inputs, mask = featurizer(batch)
            with torch.no_grad():
                hidden, valid = staged.lower(inputs, mask)
                if verify_first_batch:
                    verify_call_path(staged, inputs, mask, hidden, log)
                    verify_first_batch = False
            lengths = valid.sum(-1).tolist()
            part = owner[index]
            for row, item in enumerate(groups[index]):
                data, scale = to_storage(hidden[row, : lengths[row]], storage)
                buffers[part].append((item["key"], data, scale))
                audio_seconds += item["seconds"]
            remaining[part] -= 1
            if remaining[part] == 0:
                write_part(out_dir, part, buffers.pop(part), storage)
                elapsed = time.time() - started
                log(
                    f"cache: part {part} done; {audio_seconds / 3600:.1f} h of audio in "
                    f"{elapsed / 60:.1f} min = {audio_seconds / max(elapsed, 1e-6):.0f} h of audio per hour"
                )
    log(sampler.summary())
    return meta


def frame_estimate(item: dict) -> int:
    return math.ceil(item["seconds"] * FRAME_RATE) + 1


def estimate_bytes(items: Sequence[dict], dim: int, storage: str) -> int:
    return sum(frame_estimate(i) for i in items) * bytes_per_row(dim, storage)


@torch.no_grad()
def verify_call_path(staged, inputs, mask, hidden, log, tolerance: float = 1e-3) -> float:
    """Compare the staged lower stack with the tower's own forward (hooked layer 6) on real
    weights; logs the max abs difference on valid frames (expected 0)."""
    captured = {}
    layer = staged.tower.layers[staged.split - 1]
    handle = layer.register_forward_hook(lambda _m, _i, out: captured.setdefault("h", out))
    try:
        output = staged.tower(inputs, mask)
    finally:
        handle.remove()
    valid = output.attention_mask
    diff = float(((captured["h"] - hidden).abs() * valid[..., None]).max())
    log(f"cache: staged layer-{staged.split} output vs tower.forward max |diff| = {diff:.3g}")
    if diff > tolerance:
        raise SystemExit(f"staged call path differs from Gemma4AudioModel.forward by {diff:.3g}; not caching")
    return diff


# ---------------------------------------------------------------- load
