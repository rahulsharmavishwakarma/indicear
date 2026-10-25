"""Sharded row cache with completion markers (model-agnostic).

On disk: `<dir>/part-NNNNN.npy` (bf16 stored as int16 bits, or int8 rows), `part-NNNNN.scale.npy`
(int8 only: per-row absmax / 127, float32), `part-NNNNN.jsonl` (key, offset, frames) and
`cache.json` ({dim, storage, parts, ...}). Each part is written to temporary names and renamed;
the index file is renamed last and is the completion marker, so a rerun skips finished parts and
rebuilds a partial one. `RowCache` gathers padded batches [B, T, dim] from VRAM, pinned RAM or
memory maps (`MmapRowCache`, shared page cache for concurrent jobs).
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch


def _view(array: np.ndarray) -> torch.Tensor:
    """Tensor view of a read-only memory-map slice; it is only ever copied from."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.from_numpy(np.asarray(array))


# ---------------------------------------------------------------- storage codecs


def quantize_int8(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row absmax int8. Returns (int8 [N, D], scale float32 [N]) with x ~= q * scale."""
    rows = rows.float()
    scale = rows.abs().amax(-1).clamp_min(1e-8) / 127.0
    q = torch.round(rows / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (q.float() * scale.float().unsqueeze(-1)).to(dtype)


def to_storage(rows: torch.Tensor, storage: str) -> tuple[np.ndarray, np.ndarray | None]:
    if storage == "int8":
        q, scale = quantize_int8(rows)
        return q.cpu().numpy(), scale.cpu().numpy()
    if storage == "bf16":
        return rows.to(torch.bfloat16).cpu().view(torch.int16).numpy(), None
    raise ValueError(storage)


def plan_parts(items: Sequence[dict], part_rows: int, rows: Callable[[dict], int]) -> list[list[dict]]:
    """Deterministic partition of items (in the given order) into parts of ~part_rows rows;
    `rows(item)` estimates an item's row count."""
    parts: list[list[dict]] = []
    current: list[dict] = []
    frames = 0
    for item in items:
        estimate = rows(item)
        if current and frames + estimate > part_rows:
            parts.append(current)
            current, frames = [], 0
        current.append(item)
        frames += estimate
    if current:
        parts.append(current)
    return parts


def check_disk(out_dir: Path, need_bytes: int, log: Callable[[str], None] = print, margin: float = 1.15) -> None:
    """Refuse to start when `need_bytes` (plus a margin) will not fit on the cache disk (df)."""
    import shutil

    need = int(margin * need_bytes)
    free = shutil.disk_usage(out_dir).free
    log(f"cache: needs ~{need / 2**30:.1f} GiB, {free / 2**30:.1f} GiB free on {out_dir}")
    if need > free:
        raise SystemExit(
            f"not enough disk for the shard cache: need ~{need / 2**30:.1f} GiB, have "
            f"{free / 2**30:.1f} GiB. Use --storage int8, fewer --aug-views, or a larger disk."
        )


def write_part(out_dir: Path, part: int, rows: list, storage: str) -> None:
    """rows: (key, data [n, dim], scale [n] | None). Data first, the index last (the marker)."""
    data = np.concatenate([r[1] for r in rows], axis=0)
    stem = out_dir / f"part-{part:05d}"
    np.save(f"{stem}.tmp.npy", data)
    os.replace(f"{stem}.tmp.npy", f"{stem}.npy")
    if storage == "int8":
        np.save(f"{stem}.scale.tmp.npy", np.concatenate([r[2] for r in rows]))
        os.replace(f"{stem}.scale.tmp.npy", f"{stem}.scale.npy")
    offset = 0
    with open(f"{stem}.jsonl.tmp", "w", encoding="utf-8") as handle:
        for key, rows_, _ in rows:
            handle.write(json.dumps({"key": key, "offset": offset, "frames": len(rows_)}) + "\n")
            offset += len(rows_)
    os.replace(f"{stem}.jsonl.tmp", f"{stem}.jsonl")  # the index is the completion marker


def bytes_per_row(dim: int, storage: str) -> int:
    return dim * (1 if storage == "int8" else 2) + (4 if storage == "int8" else 0)


class RowCache:
    """All cached rows in one tensor + key -> (offset, frames)."""

    def __init__(self, data: torch.Tensor, scale: torch.Tensor | None, index: dict[str, tuple[int, int]]):
        self.data, self.scale, self.index = data, scale, index

    @classmethod
    def load(
        cls,
        directory: str | Path,
        device: str = "cuda",
        vram_fraction: float = 0.45,
        keys: set[str] | None = None,
        log: Callable[[str], None] = print,
        placement: str = "auto",
    ) -> RowCache:
        """Load the selected keys into one preallocated tensor.

        placement: auto (VRAM if it fits within `vram_fraction` of free memory, else pinned RAM),
        vram, pinned, or mmap (no copy: parts stay memory-mapped, so concurrent jobs on one
        machine share the OS page cache instead of each pinning a private copy).
        """
        directory = Path(directory)
        meta = json.loads((directory / "cache.json").read_text())
        int8 = meta["storage"] == "int8"
        indexes = []
        total = 0
        for part in range(meta["parts"]):
            path = directory / f"part-{part:05d}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing: the cache build did not finish")
            rows = [json.loads(line) for line in path.open(encoding="utf-8")]
            rows = [r for r in rows if keys is None or r["key"] in keys]
            indexes.append(rows)
            total += sum(r["frames"] for r in rows)
        dim = meta["dim"]
        element = 1 if int8 else 2
        size = total * dim * element + (total * 4 if int8 else 0)
        place = "cpu"
        if placement == "mmap":
            log(f"cache: {sum(len(r) for r in indexes)} keys, {total} rows memory-mapped (shared page cache)")
            return MmapRowCache(directory, meta, indexes)
        if device == "cuda" and torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            fits = size <= vram_fraction * free
            place = {"auto": "cuda" if fits else "pinned", "vram": "cuda", "pinned": "pinned"}[placement]
        storage_dtype = torch.int8 if int8 else torch.int16
        target = "cuda" if place == "cuda" else "cpu"
        data = torch.empty(total, dim, dtype=storage_dtype, device=target, pin_memory=place == "pinned")
        scale = torch.empty(total, dtype=torch.float32, device=target, pin_memory=place == "pinned") if int8 else None
        index: dict[str, tuple[int, int]] = {}
        offset = 0
        for part, rows in enumerate(indexes):
            if not rows:
                continue
            stem = directory / f"part-{part:05d}"
            array = np.load(f"{stem}.npy", mmap_mode="r")
            scales = np.load(f"{stem}.scale.npy", mmap_mode="r") if int8 else None
            for row in rows:
                start, frames = row["offset"], row["frames"]
                data[offset : offset + frames].copy_(_view(array[start : start + frames]))
                if int8:
                    scale[offset : offset + frames].copy_(_view(scales[start : start + frames]))
                index[row["key"]] = (offset, frames)
                offset += frames
        if not int8:
            data = data.view(torch.bfloat16)
        log(f"cache: {len(index)} keys, {total} rows, {size / 2**30:.1f} GiB in {place}")
        return cls(data, scale, index)

    def frames(self, key: str) -> int:
        return self.index[key][1]

    def batch(
        self, keys: Sequence[str], device: torch.device | str, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Padded hidden states [B, T, D] (dtype, on device) and lengths [B]."""
        spans = [self.index[key] for key in keys]
        lengths = torch.tensor([frames for _, frames in spans])
        longest = int(lengths.max())
        positions = torch.arange(longest)
        starts = torch.tensor([offset for offset, _ in spans])
        rows = starts[:, None] + positions[None, :]
        rows = torch.minimum(rows, (starts + lengths - 1)[:, None])  # clamp padding to a valid row
        rows = rows.to(self.data.device)
        gathered = self.data[rows]
        if self.scale is not None:
            gathered = dequantize(gathered, self.scale[rows], dtype)
        gathered = gathered.to(device, non_blocking=True).to(dtype)
        valid = (positions[None, :] < lengths[:, None]).to(gathered.device)
        gathered = gathered * valid[..., None].to(gathered.dtype)
        return gathered, lengths.to(gathered.device)


class MmapRowCache:
    """Memory-mapped parts; batches are gathered on the CPU into pinned memory and copied
    non-blocking. Used when several jobs share one GPU."""

    def __init__(self, directory: Path, meta: dict, indexes: list[list[dict]]):
        self.directory, self.meta = directory, meta
        self.int8 = meta["storage"] == "int8"
        self.index: dict[str, tuple[int, int, int]] = {}
        for part, rows in enumerate(indexes):
            for row in rows:
                self.index[row["key"]] = (part, row["offset"], row["frames"])
        self._arrays: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}

    def __getstate__(self):
        return {**self.__dict__, "_arrays": {}}

    def _part(self, part: int):
        if part not in self._arrays:
            stem = self.directory / f"part-{part:05d}"
            array = np.load(f"{stem}.npy", mmap_mode="r")
            scale = np.load(f"{stem}.scale.npy", mmap_mode="r") if self.int8 else None
            self._arrays[part] = (array, scale)
        return self._arrays[part]

    def frames(self, key: str) -> int:
        return self.index[key][2]

    def batch(self, keys: Sequence[str], device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        spans = [self.index[k] for k in keys]
        lengths = torch.tensor([s[2] for s in spans])
        longest = int(lengths.max())
        pin = torch.cuda.is_available()
        storage = torch.int8 if self.int8 else torch.int16
        data = torch.zeros(len(keys), longest, self.meta["dim"], dtype=storage, pin_memory=pin)
        scale = torch.zeros(len(keys), longest, dtype=torch.float32, pin_memory=pin) if self.int8 else None
        for row, (part, offset, frames) in enumerate(spans):
            array, scales = self._part(part)
            data[row, :frames] = _view(array[offset : offset + frames])
            if scale is not None:
                scale[row, :frames] = _view(scales[offset : offset + frames])
        data = data.to(device, non_blocking=True)
        if self.int8:
            hidden = dequantize(data, scale.to(device, non_blocking=True), dtype)
        else:
            hidden = data.view(torch.bfloat16).to(dtype)
        return hidden, lengths.to(hidden.device)


# ---------------------------------------------------------------- generic shard markers


def marker_path(stem: str | Path) -> Path:
    """`<stem>.done.json`: written last, after a shard's outputs, so its presence means complete."""
    return Path(f"{stem}.done.json")


def read_marker(stem: str | Path) -> dict | None:
    """The completion marker of a shard, or None when the shard never finished."""
    path = marker_path(stem)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None  # a torn marker counts as unfinished; the shard is rebuilt


def write_marker(stem: str | Path, info: dict) -> None:
    """Atomically write the completion marker (temporary name, then rename)."""
    path = marker_path(stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
