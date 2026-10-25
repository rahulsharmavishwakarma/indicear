"""Streaming training data: one stream per (source, language), read shard by shard from the Hub.

A Colab VM has ~190 GB free and the full mixture is 0.7-1.4 TB, so training audio is never
materialized. Only the small fixed sets are downloaded and decoded once (`prepare_fixed`: eval
sets, the speaker-disjoint dev set, and tar-only sources such as MUCS); everything else streams.

    fixed  = prepare_fixed(root, mixture)                       # eval/dev/local manifests + exclusions
    plan   = build_planner(root, mixture, frame_budget=...)     # streams + temperature interleave
    loader = StreamLoader(plan, transform=AugmentationRecipe(...), workers=8)
    for batch in loader:            # {"waveforms" [B,N] f32, "lengths" [B] i64, "items", "state"}
        ...                         # same waveforms/lengths as encoder.FeatureBatches(backend="gpu")
        checkpoint["data_state"] = batch["state"]   # resume: planner.load_state(state)

Pieces:
- `ShardCache`: rolling local window of downloaded shards, bounded by `max_bytes` (default 40 GiB).
  Background threads prefetch each stream's next shard; a shard is deleted as soon as its stream has
  read it (its rows then live in the shuffle buffer, in memory). A blocking request evicts unopened
  prefetches first and may overshoot the bound by one shard rather than deadlock. Logs MB/s.
  `keep=True` (a host with disk but little bandwidth, e.g. a workstation) keeps every finished shard on
  disk instead, across epochs and restarts: the pool is downloaded once (the pilot's: 68 GiB), and
  `hub_fetch` returns a complete local copy without touching the network. `download_pool` fills it
  ahead of training.
- `SourceStream`: a fixed *pool* of shards per stream (enough for the mixture's hours, seeded), reshuffled
  every stream-epoch; rows are filtered exactly like the download path (`ingest.parse_row`: markup,
  unusable rows, durations, code-mixed-only) plus the exclusion set (eval and dev speakers, groups,
  keys). Position = (epoch, shard position, row index).
- `StreamPlanner`: interleaves streams with temperature sampling (p_group ~ hours^alpha with caps and
  floors, then within a group in proportion to hours), keeps a shuffle buffer sized in seconds of
  audio, and forms batches by duration: a random pivot in the buffer plus its nearest neighbours in
  duration, up to the frame budget (padded frames at 25 fps). Its state (RNG, every stream's
  position and the references of the rows in the buffer) is attached to every batch; restoring it
  re-reads exactly those rows, so a resumed run continues with the same keys, without repeats or gaps.
- `StreamLoader`: a producer thread plans batches and a thread pool decodes them to 16 kHz float32,
  checks each utterance's audio hash against the eval/dev hashes, applies the augmentation
  transform (`transform(audio, rng, item)`, item["view"] > 0 = augmented draw), and pads.
"""

from __future__ import annotations

import json
import math
import os
import queue
import random
import threading
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from indicear.common import FRAME_RATE, read_jsonl, write_json, write_jsonl
from indicear.ingest import (
    Context,
    DiskGuard,
    Task,
    context_options,
    decode_bytes,
    hf_token,
    list_hf_shards,
    parse_row,
    source_columns,
    to_16k,
)
from indicear.markup import word_languages
from indicear.mixture import (
    CODE_SWITCHED,
    KEEP_RATIO,
    LeakError,
    assert_no_leak,
    group_of,
    load_mixture,
    plan_tasks,
    temperature_shares,
)
from indicear.sources import SOURCES, allowed
from indicear.text import LANG_SCRIPT

GIB = 2**30
STATE_VERSION = 1


# ---------------------------------------------------------------- specs and pools


@dataclass(frozen=True)
class StreamSpec:
    name: str
    source: str
    lang: str
    splits: tuple[str, ...]
    hours: float
    group: str  # sampling group: the language, or "cs" for code-mixed-only streams
    code_mixed_only: bool = False
    weight: float = 1.0  # within its group a stream is drawn in proportion to hours x weight

    @property
    def draw_hours(self) -> float:
        return self.hours * self.weight


def stream_specs(mixture: dict, allow_review: set[str] = frozenset()) -> tuple[list[StreamSpec], list[Task], list[str]]:
    """Streams for hf-parquet training entries; tar-only training entries become local tasks."""
    _, tasks, skipped = plan_tasks(mixture, allow_review)
    merged: dict[tuple, dict] = {}
    local: list[Task] = []
    for task in tasks:
        if SOURCES[task.source].kind != "hf-parquet":
            local.append(task)
            continue
        key = (task.source, task.lang, task.code_mixed_only)
        entry = merged.setdefault(key, {"splits": [], "hours": 0.0, "weight": task.weight})
        if task.weight != entry["weight"]:
            raise ValueError(f"{task.source}/{task.lang}: entries of one stream need the same weight")
        if task.split not in entry["splits"]:
            entry["splits"].append(task.split)
        entry["hours"] += task.hours
    specs = []
    for (source, lang, cm), entry in merged.items():
        name = f"{source}/{lang}" + ("/cm" if cm else "")
        specs.append(StreamSpec(name, source, lang, tuple(entry["splits"]), entry["hours"],
                                CODE_SWITCHED if cm else lang, cm, entry["weight"]))
    return specs, local, skipped


def select_pool(spec: StreamSpec, shards: list[dict], seed: int, avoid: set[str] | None = None) -> list[dict]:
    """Seeded subset of shards whose estimated hours cover the stream's hours (rows are filtered,
    so the estimate is divided by KEEP_RATIO). The same pool is used every epoch. Shards in
    `avoid` (shard ids another stream of the same source and language already reads) are taken
    only after every other shard, so a code-mixed mining stream does not repeat the main stream's rows."""
    source = SOURCES[spec.source]
    order = list(shards)
    random.Random(f"{seed}:pool:{spec.name}").shuffle(order)
    if avoid:
        order = [s for s in order if shard_id(s) not in avoid] + [s for s in order if shard_id(s) in avoid]
    want = spec.hours / KEEP_RATIO / (0.3 if spec.code_mixed_only else 1.0)
    pool, hours = [], 0.0
    for shard in order:
        if hours >= want:
            break
        pool.append(shard)
        hours += (shard.get("size") or 0) / 1e6 / max(source.mb_per_hour, 1.0)
    return pool


# ---------------------------------------------------------------- rolling shard cache


def local_copy(shard: dict, directory: Path) -> Path:
    """Where `hub_fetch` puts a shard (hf_hub_download's local_dir layout)."""
    return directory / shard["repo"].replace("/", "__") / shard["path"]


def hub_fetch(shard: dict, directory: Path) -> Path:
    """Download a pinned shard; a complete local copy (the listed size) is returned without any
    request. hf_hub_download writes the final path only once the file is complete."""
    from huggingface_hub import hf_hub_download

    local = local_copy(shard, directory)
    if shard.get("size") and local.is_file() and local.stat().st_size == int(shard["size"]):
        return local
    return Path(hf_hub_download(shard["repo"], shard["path"], repo_type="dataset", revision=shard["revision"],
                                token=hf_token(), local_dir=directory / shard["repo"].replace("/", "__")))


def shard_id(shard: dict) -> str:
    return f"{shard['repo']}@{shard['revision']}:{shard['path']}"


@dataclass
class _Entry:
    shard: dict
    future: Future | None = None
    path: Path | None = None
    size: int = 0
    opened: bool = False
    admitted: bool = False
    urgent: bool = False  # a consumer is waiting for it: admission may evict prefetches or overshoot


class ShardCache:
    def __init__(self, directory: Path, max_bytes: int = 40 * GIB, fetch: Callable[[dict, Path], Path] | None = None,
                 workers: int = 2, log: Callable[[str], None] = print, log_every: float = 60.0, keep: bool = False):
        self.keep = keep  # finished shards stay on disk (outside the bound) for later epochs and restarts
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.fetch = fetch or hub_fetch
        self.log = log
        self.log_every = log_every
        self.entries: dict[str, _Entry] = {}
        self.used = 0
        self.cond = threading.Condition()
        self.pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="shard-fetch")
        self.downloaded = 0
        self.fetch_seconds = 0.0
        self.started = time.time()
        self.last_log = 0.0
        self.peak = 0

    # -- admission: wait for room; a blocking request may evict prefetches or overshoot by one shard
    def _admit(self, entry: _Entry, blocking: bool) -> bool:
        size = entry.shard.get("size") or 0
        with self.cond:
            while self.used + size > self.max_bytes and self.used > 0:
                if blocking or entry.urgent:
                    victims = [e for e in self.entries.values() if e.admitted and not e.opened and e is not entry
                               and e.future is not None and e.future.done() and e.path is not None]
                    if victims:
                        self._evict(victims[0])
                        continue
                    self.log(f"stream cache: over the {self.max_bytes / GIB:.1f} GiB bound by one shard "
                             f"({size / GIB:.2f} GiB) rather than wait")
                    break
                if entry.shard.get("_cancel"):
                    return False
                self.cond.wait(timeout=5.0)
            self.used += size
            entry.admitted = True
            self.peak = max(self.peak, self.used)
            return True

    def _download(self, entry: _Entry, blocking: bool) -> Path | None:
        if not self._admit(entry, blocking):
            return None
        started = time.time()
        try:
            path = self.fetch(entry.shard, self.dir)
        except BaseException:
            with self.cond:
                self.used -= entry.shard.get("size") or 0
                entry.admitted = False
                self.entries.pop(shard_id(entry.shard), None)
                self.cond.notify_all()
            raise
        size = Path(path).stat().st_size
        with self.cond:
            self.used += size - (entry.shard.get("size") or 0)
            entry.size = size
            entry.path = Path(path)
            self.downloaded += size
            self.fetch_seconds += time.time() - started
        self.maybe_log()
        return entry.path

    def prefetch(self, shard: dict) -> None:
        key = shard_id(shard)
        with self.cond:
            if key in self.entries:
                return
            entry = self.entries[key] = _Entry(dict(shard))
            entry.future = self.pool.submit(self._download, entry, False)

    def open(self, shard: dict) -> Path:
        """Block until the shard is local (download now if it was not prefetched)."""
        key = shard_id(shard)
        with self.cond:
            entry = self.entries.get(key)
            if entry is None:
                entry = self.entries[key] = _Entry(dict(shard))
            entry.opened = True
            entry.urgent = True
            self.cond.notify_all()
            future, path = entry.future, entry.path
        if path is not None and Path(path).exists():
            return Path(path)
        if future is not None:
            result = future.result()
            if result is not None:
                return result
        path = self._download(entry, True)
        if path is None:
            raise RuntimeError(f"shard {key} was cancelled")
        return path

    def done(self, shard: dict) -> None:
        """The stream has read every row of this shard: delete it (keep=True: leave it on disk)."""
        with self.cond:
            entry = self.entries.get(shard_id(shard))
            if entry is not None:
                self._evict(entry, delete=not self.keep)

    def _evict(self, entry: _Entry, delete: bool = True) -> None:  # caller holds the lock
        if delete and entry.path is not None:
            Path(entry.path).unlink(missing_ok=True)
        if entry.admitted:
            self.used -= entry.size or (entry.shard.get("size") or 0)
        self.entries.pop(shard_id(entry.shard), None)
        self.cond.notify_all()

    def stats(self) -> dict:
        elapsed = max(time.time() - self.started, 1e-6)
        return {"downloaded_mb": round(self.downloaded / 1e6, 1),
                "mb_per_s_wall": round(self.downloaded / 1e6 / elapsed, 1),
                "mb_per_s_fetching": round(self.downloaded / 1e6 / max(self.fetch_seconds, 1e-6), 1),
                "cached_gb": round(self.used / GIB, 2), "peak_gb": round(self.peak / GIB, 2),
                "max_gb": round(self.max_bytes / GIB, 1), "shards": len(self.entries)}

    def maybe_log(self, extra: str = "") -> None:
        now = time.time()
        if now - self.last_log >= self.log_every:
            self.last_log = now
            s = self.stats()
            self.log(f"stream: {s['downloaded_mb']:.0f} MB fetched, {s['mb_per_s_wall']:.1f} MB/s (wall), "
                     f"{s['mb_per_s_fetching']:.1f} MB/s (while fetching), cache {s['cached_gb']:.1f}/{s['max_gb']:.0f} GiB"
                     + (f", {extra}" if extra else ""))

    def close(self) -> None:
        with self.cond:
            for entry in self.entries.values():
                entry.shard["_cancel"] = True
            self.cond.notify_all()
        self.pool.shutdown(wait=True, cancel_futures=True)
        with self.cond:
            for entry in list(self.entries.values()):
                self._evict(entry, delete=not (self.keep and entry.path is not None and entry.future is not None
                                               and entry.future.done()))


# ---------------------------------------------------------------- one stream


class SourceStream:
    """Rows of one (source, language) pool, epoch after epoch. Not thread-safe (the planner owns it);
    with `enable_readahead` its own reader thread runs ahead, and only that thread touches the
    reader position, drop counters and shard handles until `load_state`/`close` stops it."""

    def __init__(self, spec: StreamSpec, pool: list[dict], cache: ShardCache, ctx: Context, exclude: dict,
                 seed: int = 0, prefetch: int = 1, batch_rows: int = 32):
        self.spec, self.pool, self.cache, self.ctx = spec, pool, cache, ctx
        self.source = SOURCES.get(spec.source)  # None for local rows of no registered source (negatives)
        self.exclude = exclude
        self.seed, self.prefetch, self.batch_rows = seed, prefetch, batch_rows
        self.epoch = self.pos = self.row = 0
        self.drops: Counter = Counter()
        self.notes: Counter = Counter()
        self._reader: _ShardReader | None = None
        self._orders: dict[int, list[int]] = {}
        self._dry_shards = 0  # shards finished since the last accepted row

    def order(self, epoch: int) -> list[int]:
        cached = self._orders.get(epoch)
        if cached is None:
            cached = list(range(len(self.pool)))
            random.Random(f"{self.seed}:{self.spec.name}:epoch{epoch}").shuffle(cached)
            if len(self._orders) > 4:
                self._orders.clear()
            self._orders[epoch] = cached
        return cached

    def shard(self, epoch: int, pos: int) -> dict:
        return self.pool[self.order(epoch)[pos]]

    # -- read-ahead: a background thread runs the sequential reader into a bounded queue, so the
    # (single-threaded) planner never waits on parquet reads, row parsing or header probes. Each
    # stream's row sequence is unchanged; `state()` is the position after the last row handed out.
    def enable_readahead(self, depth: int) -> None:
        self._ra_depth = max(0, int(depth))

    def _ra_start(self) -> None:
        import queue
        import threading

        q: queue.Queue = queue.Queue(maxsize=self._ra_depth)
        stop = threading.Event()
        self._ra_queue, self._ra_stop, self._ra_state = q, stop, self._sync_state()

        def put(entry: tuple) -> None:
            while not stop.is_set():
                try:
                    q.put(entry, timeout=0.5)
                    return
                except queue.Full:
                    continue

        def run() -> None:
            while not stop.is_set():
                try:
                    item = self._sync_next()
                except BaseException as error:  # noqa: BLE001 - re-raised in the consumer's next()
                    put(("error", error, None))
                    return
                put(("item", item, self._sync_state()))
                if item is None:
                    return

        self._ra_thread = threading.Thread(target=run, name=f"readahead-{self.spec.name}", daemon=True)
        self._ra_thread.start()

    def _ra_halt(self) -> None:
        """Stop the reader thread and drop what it read ahead; the stream is then at the reader's
        position, so callers reset it with load_state."""
        thread = getattr(self, "_ra_thread", None)
        if thread is None:
            return
        self._ra_stop.set()
        while thread.is_alive():
            try:
                self._ra_queue.get_nowait()
            except Exception:  # noqa: BLE001 - queue.Empty
                thread.join(timeout=0.1)
        self._ra_thread = None

    def close(self) -> None:
        state = self.state()
        self._ra_halt()
        self._close()
        self.epoch, self.pos, self.row, self._dry_shards = state["epoch"], state["pos"], state["row"], state["dry"]

    def state(self) -> dict:
        if getattr(self, "_ra_thread", None) is not None:
            return dict(self._ra_state)
        return self._sync_state()

    def _sync_state(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos, "row": self.row, "dry": self._dry_shards}

    def load_state(self, state: dict) -> None:
        self._ra_halt()
        self._ra_exhausted = False
        self._close()
        self.epoch, self.pos, self.row = int(state["epoch"]), int(state["pos"]), int(state["row"])
        self._dry_shards = int(state.get("dry", 0))  # absent in states written before the guard

    def _close(self) -> None:
        self._reader = None

    def _prefetch_next(self) -> None:
        epoch, pos = self.epoch, self.pos
        for _ in range(self.prefetch):
            pos += 1
            if pos >= len(self.pool):
                epoch, pos = epoch + 1, 0
            self.cache.prefetch(self.shard(epoch, pos))

    def start_readahead(self) -> None:
        """Start the reader thread now (the planner starts every stream's at once, so their first
        shard opens overlap) instead of at this stream's first draw. No-op without read-ahead."""
        if self.pool and getattr(self, "_ra_depth", 0) and getattr(self, "_ra_thread", None) is None \
                and not getattr(self, "_ra_exhausted", False):
            self._ra_start()

    def next(self) -> dict | None:
        """The next accepted row, or None when the pool is empty."""
        if not self.pool:
            return None
        if getattr(self, "_ra_depth", 0):
            if getattr(self, "_ra_exhausted", False):
                return None
            if getattr(self, "_ra_thread", None) is None:
                self._ra_start()
            kind, item, state = self._ra_queue.get()
            if kind == "error":
                self.close()
                raise item
            self._ra_state = state
            if item is None:  # the reader thread has exited; no restart until load_state
                self._ra_thread, self._ra_exhausted = None, True
                self.epoch, self.pos, self.row, self._dry_shards = (state["epoch"], state["pos"], state["row"],
                                                                    state["dry"])
            return item
        return self._sync_next()

    def _sync_next(self) -> dict | None:
        while True:
            if self.pos >= len(self.pool):
                self.epoch, self.pos, self.row = self.epoch + 1, 0, 0
            shard = self.shard(self.epoch, self.pos)
            if self._reader is None or self._reader.shard is not shard:
                self._reader = _ShardReader(shard, self.cache.open(shard), source_columns(self.source),
                                            self.batch_rows)
                self._prefetch_next()
            raw = self._reader.row(self.row)
            if raw is None:  # shard finished
                self.cache.done(shard)
                self._reader = None
                self.pos, self.row = self.pos + 1, 0
                self._dry_shards += 1
                if self._dry_shards >= len(self.pool):
                    # A whole pass of the pool accepted nothing (e.g. every speaker excluded): the
                    # stream would loop and re-download its shards forever. None marks it dead.
                    self.ctx.log(f"stream {self.spec.name}: no row accepted in a full pass of its "
                                 f"{len(self.pool)} shards ({dict(self.drops)}); stream stopped")
                    return None
                continue
            ref = [self.epoch, self.pos, self.row]
            self.row += 1
            item = self.accept(raw, ref)
            if item is not None:
                self._dry_shards = 0
                return item

    def accept(self, raw: dict, ref: list[int]) -> dict | None:
        task = Task(self.spec.source, self.spec.lang, self.split_of(ref), math.inf,
                    code_mixed_only=self.spec.code_mixed_only)
        pending = parse_row(self.source, task, raw, self.ctx, self.drops, self.notes)
        if pending is None:
            return None
        meta = pending.meta
        if meta.get("speaker") and meta["speaker"] in self.exclude["speaker"]:
            self.drops["excluded_speaker"] += 1
            return None
        if meta.get("group") and meta["group"] in self.exclude["group"]:
            self.drops["excluded_group"] += 1
            return None
        if pending.key in self.exclude["key"]:
            self.drops["excluded_key"] += 1
            return None
        seconds = pending.seconds_hint
        if seconds is None:
            seconds = _header_seconds(pending.data)
            if seconds is None:
                self.drops["unreadable_header"] += 1
                return None
        if not self.ctx.min_seconds <= seconds <= self.ctx.max_seconds:
            self.drops["too_short" if seconds < self.ctx.min_seconds else "too_long"] += 1
            return None
        cleaned = pending.cleaned
        return {"key": pending.key, "text": cleaned.text, "lang": self.spec.lang, "word_langs": cleaned.word_langs,
                "code_mixed": cleaned.code_mixed, "source": self.spec.source, "seconds": round(seconds, 3),
                "speaker": meta.get("speaker"), "group": meta.get("group"), "stream": self.spec.name,
                "ref": ref, "audio_bytes": pending.data}

    def split_of(self, ref: list[int]) -> str:
        path = self.shard(ref[0], ref[1])["path"]
        for split in self.spec.splits:
            if SOURCES[self.spec.source].splits[split].format(config=SOURCES[self.spec.source].langs[self.spec.lang]) in path:
                return split
        return self.spec.splits[0]

    def fetch(self, refs: Sequence[list[int]], strict: bool = True) -> dict[tuple, dict]:
        """Re-read specific rows (for restoring the shuffle buffer). Returns {tuple(ref): item}.
        A row this host's filters now reject (e.g. a speaker in a rebuilt dev set) stops the run,
        or with strict=False is left out (counted in drops["restore_rejected"])."""
        out = {}
        by_shard: dict[tuple[int, int], list[int]] = defaultdict(list)
        for epoch, pos, row in refs:
            by_shard[(epoch, pos)].append(row)
        for (epoch, pos), rows in by_shard.items():
            shard = self.shard(epoch, pos)
            reader = _ShardReader(shard, self.cache.open(shard), source_columns(self.source), self.batch_rows)
            for row in sorted(rows):
                raw = reader.row(row)
                item = self.accept(raw, [epoch, pos, row]) if raw is not None else None
                if item is None:
                    if not strict:
                        self.drops["restore_rejected"] += 1
                        continue
                    raise RuntimeError(f"{self.spec.name}: buffered row {(epoch, pos, row)} no longer reads back")
                out[(epoch, pos, row)] = item
            current = (self.epoch, self.pos)
            if (epoch, pos) != current:
                self.cache.done(shard)
        return out


def _header_seconds(data: bytes | None) -> float | None:
    import io

    import soundfile as sf

    if not data:
        return None
    try:
        info = sf.info(io.BytesIO(data))
        return info.frames / info.samplerate
    except Exception:  # noqa: BLE001 - e.g. m4a: fall back to decoding
        try:
            audio, rate = decode_bytes(data)
            return len(audio) / rate
        except Exception:  # noqa: BLE001
            return None


class _ShardReader:
    """Sequential reader with cheap forward skips (row groups before the target are not read)."""

    def __init__(self, shard: dict, path: Path, columns: list[str], batch_rows: int):
        import pyarrow.parquet as pq

        self.shard = shard
        self.file = pq.ParquetFile(path)
        names = self.file.schema_arrow.names
        self.columns = [c for c in columns if c in names]
        self.batch_rows = batch_rows
        self.starts = []
        start = 0
        for g in range(self.file.metadata.num_row_groups):
            self.starts.append(start)
            start += self.file.metadata.row_group(g).num_rows
        self.total = start
        self._iter: Iterator | None = None
        self._buffer: list[dict] = []
        self._first = 0  # global index of self._buffer[0]

    def _seek(self, index: int) -> None:
        group = max(g for g, s in enumerate(self.starts) if s <= index) if self.starts else 0
        groups = list(range(group, len(self.starts)))
        self._iter = self.file.iter_batches(batch_size=self.batch_rows, row_groups=groups, columns=self.columns)
        self._buffer, self._first = [], self.starts[group]

    def row(self, index: int) -> dict | None:
        if index >= self.total:
            return None
        if self._iter is None or index < self._first:
            self._seek(index)
        while index >= self._first + len(self._buffer):
            self._first += len(self._buffer)
            self._buffer = next(self._iter).to_pylist()
        return self._buffer[index - self._first]


# ---------------------------------------------------------------- planner


def _encode_rng(rng: random.Random) -> list:
    version, internal, gauss = rng.getstate()
    return [version, list(internal), gauss]


def _decode_rng(state: list) -> tuple:
    return (state[0], tuple(state[1]), state[2])


class StreamPlanner:
    def __init__(self, streams: Sequence[SourceStream], shares: dict[str, float], frame_budget: int,
                 buffer_seconds: float = 1800.0, max_items: int | None = None, seed: int = 0,
                 augment_prob: float = 0.0, frame_rate: int = FRAME_RATE, eval_index: dict | None = None,
                 log: Callable[[str], None] = print):
        self.streams = list(streams)
        self.by_name = {s.spec.name: s for s in self.streams}
        group_hours: dict[str, float] = defaultdict(float)
        for s in self.streams:
            group_hours[s.spec.group] += s.spec.draw_hours
        self.weights = [shares.get(s.spec.group, 0.0) * s.spec.draw_hours / group_hours[s.spec.group]
                        if group_hours[s.spec.group] else 0.0 for s in self.streams]
        self.frame_budget, self.buffer_seconds, self.max_items = frame_budget, buffer_seconds, max_items
        self.augment_prob, self.frame_rate = augment_prob, frame_rate
        self.rng = random.Random(seed)
        self.buffer: list[dict] = []
        self.batches = 0
        self.dead: set[int] = set()
        self.eval_index = eval_index or {"speaker": set(), "group": set(), "audio": set()}
        self.log = log
        # False: buffered rows this host's filters reject are dropped on restore instead of stopping
        # the run (a prep root rebuilt elsewhere, whose dev carve-out excludes other speakers).
        self.strict_restore = True

    def probabilities(self) -> dict[str, float]:
        total = sum(self.weights)
        return {s.spec.name: w / total for s, w in zip(self.streams, self.weights, strict=True)} if total else {}

    def buffered_seconds(self) -> float:
        return sum(r["seconds"] for r in self.buffer)

    def _fill(self) -> None:
        for i, stream in enumerate(self.streams):  # streams need only next(); read-ahead is optional
            if i not in self.dead and self.weights[i] > 0 and hasattr(stream, "start_readahead"):
                stream.start_readahead()
        seconds = self.buffered_seconds()
        while seconds < self.buffer_seconds:
            live = [i for i in range(len(self.streams)) if i not in self.dead and self.weights[i] > 0]
            if not live:
                return
            index = self.rng.choices(live, weights=[self.weights[i] for i in live])[0]
            item = self.streams[index].next()
            if item is None:
                self.dead.add(index)
                continue
            self.buffer.append(item)
            seconds += item["seconds"]

    def _frames(self, seconds: float) -> int:
        return math.ceil(seconds * self.frame_rate)

    def next_batch(self) -> tuple[list[dict], dict]:
        self._fill()
        if not self.buffer:
            raise StopIteration
        order = sorted(range(len(self.buffer)), key=lambda i: (self.buffer[i]["seconds"], self.buffer[i]["key"]))
        pivot = self.rng.randrange(len(order))
        lo = hi = pivot
        longest = self._frames(self.buffer[order[pivot]]["seconds"])
        while True:
            count = hi - lo + 1
            if self.max_items and count >= self.max_items:
                break
            options = []
            if lo > 0:
                options.append(("lo", self._frames(self.buffer[order[lo - 1]]["seconds"])))
            if hi < len(order) - 1:
                options.append(("hi", self._frames(self.buffer[order[hi + 1]]["seconds"])))
            if not options:
                break
            side, frames = min(options, key=lambda o: abs(o[1] - longest))
            if (count + 1) * max(longest, frames) > self.frame_budget:
                break
            longest = max(longest, frames)
            lo, hi = (lo - 1, hi) if side == "lo" else (lo, hi + 1)
        chosen = sorted(order[lo:hi + 1])
        rows = [self.buffer[i] for i in chosen]
        taken = set(chosen)
        self.buffer = [r for i, r in enumerate(self.buffer) if i not in taken]
        for row in rows:
            self._check_leak(row)
            row["view"] = 0 if self.rng.random() >= self.augment_prob else 1 + self.batches
        self.batches += 1
        return rows, self.state()

    def _check_leak(self, row: dict) -> None:
        if (row.get("speaker") and row["speaker"] in self.eval_index["speaker"]) or (
                row.get("group") and row["group"] in self.eval_index["group"]):
            raise LeakError(f"eval/dev speaker reached a training batch: {row['key']}")

    def state(self) -> dict:
        return {"version": STATE_VERSION, "batches": self.batches, "rng": _encode_rng(self.rng),
                "streams": {s.spec.name: s.state() for s in self.streams},
                "buffer": [[r["stream"], *r["ref"]] for r in self.buffer], "dead": sorted(self.dead)}

    def load_state(self, state: dict) -> None:
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"stream state version {state.get('version')} != {STATE_VERSION}")
        missing = set(state["streams"]) - set(self.by_name)
        if missing:
            raise ValueError(f"checkpointed streams not in this mixture: {sorted(missing)}")
        self.rng.setstate(_decode_rng(state["rng"]))
        self.batches = int(state["batches"])
        self.dead = set(state.get("dead", []))
        refs: dict[str, list[list[int]]] = defaultdict(list)
        for name, *ref in state["buffer"]:
            refs[name].append(ref)
        before = {s.spec.name: Counter(s.drops) for s in self.streams}

        def restore(name: str) -> dict[tuple, dict]:  # streams are independent: re-read them in parallel
            stream = self.by_name[name]
            stream.load_state(state["streams"][name])
            return stream.fetch(refs[name], strict=self.strict_restore)

        with ThreadPoolExecutor(max_workers=max(1, min(32, len(refs))), thread_name_prefix="stream-restore") as pool:
            fetched = dict(zip(refs, pool.map(restore, list(refs)), strict=True))
        for name, stream in self.by_name.items():
            stream.load_state(state["streams"][name])
        self.buffer = [fetched[name][tuple(ref)] for name, *ref in state["buffer"]
                       if fetched[name].get(tuple(ref)) is not None]
        dropped = len(state["buffer"]) - len(self.buffer)
        if dropped:
            reasons = Counter()
            for s in self.streams:
                delta = Counter(s.drops)
                delta.subtract(before[s.spec.name])
                reasons.update({f"{s.spec.name}:{k}": v for k, v in delta.items() if v > 0 and k != "restore_rejected"})
            self.log(f"stream restore: {dropped} of {len(state['buffer'])} buffered rows are rejected by this host's "
                     f"filters and left out (by stream and reason: {dict(reasons)}); later batches differ from the "
                     "original host's")

    def report(self) -> dict:
        drops = Counter()
        for s in self.streams:
            drops.update({f"{s.spec.name}:{k}": v for k, v in s.drops.items()})
        return {"probabilities": self.probabilities(), "batches": self.batches,
                "buffered_seconds": round(self.buffered_seconds(), 1),
                "epochs": {s.spec.name: s.epoch for s in self.streams}, "drops": dict(drops)}


# ---------------------------------------------------------------- decoding loader


def decode_batch(rows: Sequence[dict], state: dict, transform: Callable | None, eval_audio: set[str],
                 seed: int = 0, pool: Executor | None = None) -> dict:
    """Decode to 16 kHz float32, drop eval/dev audio duplicates, augment, pad. A transform with
    `uses_batch = True` (augment.AugmentPolicy) is also given the batch's other clean waveforms
    (`batch=`, `index=`), e.g. for babble; all rows are decoded before any is augmented. A
    transform with `all_rows = True` (augment.CodecAugment) also sees the clean draws (view 0).
    With `pool`, the rows are decoded and then augmented in parallel; the result is the same."""
    import hashlib

    from indicear.features import pad_waveforms

    run = pool.map if pool is not None else map

    def decode(row: dict) -> tuple[dict, np.ndarray] | None:
        if row.get("audio_bytes") is not None:
            audio, rate = decode_bytes(row["audio_bytes"])
            audio = to_16k(audio, rate)
        else:
            from indicear.encoder import load_audio

            audio = load_audio(row["audio"])
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
        if eval_audio and hashlib.sha1(pcm.tobytes()).hexdigest() in eval_audio:
            return None
        return {k: v for k, v in row.items() if k not in ("audio_bytes", "ref")}, np.asarray(audio, dtype=np.float32)

    decoded = list(run(decode, rows))
    dropped = sum(d is None for d in decoded)
    items = [d[0] for d in decoded if d is not None]
    audios = [d[1] for d in decoded if d is not None]
    if transform is not None:
        clean = list(audios)
        every_row = getattr(transform, "all_rows", False)  # augment.CodecAugment: clean draws too

        def augment(index: int) -> np.ndarray:
            item = items[index]
            if not item.get("view", 0) and not every_row:
                return audios[index]
            parts = (seed, zlib.crc32(item["key"].encode()), int(item.get("view", 0)))
            rng = np.random.default_rng((*parts, 1) if every_row else parts)
            if getattr(transform, "uses_batch", False):
                audio = transform(clean[index], rng, item, batch=clean, index=index)
            else:
                audio = transform(clean[index], rng, item)
            return np.asarray(audio, dtype=np.float32)

        audios = list(run(augment, range(len(items))))
    waveforms, lengths = pad_waveforms(audios)
    return {"waveforms": waveforms, "lengths": lengths, "items": items, "state": state, "dropped": dropped}


class StreamLoader:
    """Iterator of decoded batches. Planning happens in one producer thread (the planner is
    sequential and deterministic); each batch's rows are decoded and augmented in parallel across
    `workers` threads (free-threaded Python: no GIL), `prefetch` batches ahead. `batch["state"]` restores the stream right after
    that batch."""

    def __init__(self, planner: StreamPlanner, transform: Callable | None = None, workers: int | None = None,
                 prefetch: int = 8, eval_audio: set[str] | None = None, seed: int = 0, cache: ShardCache | None = None,
                 log: Callable[[str], None] = print, log_every: float = 60.0):
        self.planner, self.transform = planner, transform
        self.workers = workers or max(1, (os.cpu_count() or 4) - 1)
        self.prefetch, self.eval_audio, self.seed = prefetch, eval_audio or set(), seed
        self.cache, self.log, self.log_every = cache, log, log_every
        self.state: dict | None = None
        self.stats = {"batches": 0, "seconds": 0.0, "dropped": 0, "started": time.time(), "waits": 0.0}

    def __iter__(self) -> Iterator[dict]:
        # Batches are assembled `prefetch` at a time; their rows are decoded and augmented across all
        # `workers` (a separate pool, so a batch waiting on its rows never holds a row worker).
        pool = ThreadPoolExecutor(max_workers=self.prefetch, thread_name_prefix="stream-batch")
        rows_pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="stream-decode")
        pending: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()
        failure: list[BaseException] = []

        def produce() -> None:
            try:
                while not stop.is_set():
                    try:
                        rows, state = self.planner.next_batch()
                    except StopIteration:
                        break
                    future = pool.submit(decode_batch, rows, state, self.transform, self.eval_audio, self.seed,
                                         rows_pool)
                    while not stop.is_set():
                        try:
                            pending.put(future, timeout=0.5)
                            break
                        except queue.Full:
                            continue
            except BaseException as error:  # noqa: BLE001 - surfaced to the consumer
                failure.append(error)
            finally:
                while True:  # the end marker; never block a consumer that already stopped
                    try:
                        pending.put(None, timeout=0.5)
                        break
                    except queue.Full:
                        if stop.is_set():
                            break

        producer = threading.Thread(target=produce, name="stream-plan", daemon=True)
        producer.start()
        last_log = time.time()
        try:
            while True:
                waited = time.time()
                future = pending.get()
                if future is None:
                    if failure:
                        raise failure[0]
                    return
                batch = future.result()
                self.stats["waits"] += time.time() - waited
                self.state = batch["state"]
                self.stats["batches"] += 1
                self.stats["seconds"] += float(batch["lengths"].sum()) / 16000
                self.stats["dropped"] += batch["dropped"]
                if time.time() - last_log >= self.log_every:
                    last_log = time.time()
                    self.log(self.status())
                yield batch
        finally:
            stop.set()
            producer.join(timeout=30)
            pool.shutdown(wait=True, cancel_futures=True)
            rows_pool.shutdown(wait=True, cancel_futures=True)

    def status(self) -> str:
        elapsed = max(time.time() - self.stats["started"], 1e-6)
        line = (f"stream: {self.stats['batches']} batches, {self.stats['seconds'] / 3600:.2f} h of audio "
                f"({self.stats['seconds'] / elapsed:.0f}x real time), buffer "
                f"{self.planner.buffered_seconds():.0f}/{self.planner.buffer_seconds:.0f} s, consumer waited "
                f"{self.stats['waits']:.0f} s, eval-audio drops {self.stats['dropped']}")
        if self.cache is not None:
            s = self.cache.stats()
            line += f"; network {s['mb_per_s_wall']:.1f} MB/s, cache {s['cached_gb']:.1f}/{s['max_gb']:.0f} GiB"
        return line


# ---------------------------------------------------------------- fixed sets and assembly


def prepare_fixed(root: Path, mixture: dict, allow_review: set[str] = frozenset(), workers: int = 4,
                  decode_workers: int | None = None, disk_budget_gb: float | None = None,
                  log: Callable[[str], None] = print, runner: Callable | None = None) -> dict:
    """Download and decode the small fixed sets once: eval sets, a speaker-disjoint dev set per
    stream (whole speaker groups from the end of the shard order), and tar-only training sources.
    Writes manifests/{eval_*,dev,local_train}.jsonl and manifests/stream_exclude.json."""
    from indicear.ingest import run_tasks, used_bytes
    from indicear.mixture import dedupe, remove_eval_overlap, split_dev

    root = Path(root)
    manifests = root / "manifests"
    marker = manifests / "stream_fixed.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    evals, _, _ = plan_tasks(mixture, allow_review)
    specs, local_tasks, skipped = stream_specs(mixture, allow_review)
    dev_cfg = mixture.get("dev", {})
    lang_hours: dict[str, float] = defaultdict(float)
    for spec in specs:
        lang_hours[spec.lang] += spec.hours
    dev_tasks, dev_targets = [], {}
    for spec in specs:
        if spec.code_mixed_only:
            continue
        target = min(max(float(dev_cfg.get("fraction", 0.02)) * lang_hours[spec.lang],
                         float(dev_cfg.get("min_hours_per_lang", 0.0))), float(dev_cfg.get("max_hours_per_lang", 2.0)))
        share = target * spec.hours / lang_hours[spec.lang]
        task = Task(spec.source, spec.lang, spec.splits[0], round(4 * share, 4), role="dev", reverse=True)
        dev_tasks.append(task)
        dev_targets[task.label] = share * 3600
    guard = DiskGuard(root, int(disk_budget_gb * GIB) if disk_budget_gb else None, used_bytes=used_bytes(root), log=log)
    ctx = Context(root, guard, seed=int(mixture.get("seed", 0)), min_seconds=float(mixture.get("min_seconds", 0.5)),
                  max_seconds=float(mixture.get("max_seconds", 20.0)), log=log, **context_options(mixture))
    if decode_workers:
        ctx.decode_workers = decode_workers
    runner = runner or run_tasks
    try:
        eval_results = runner(evals, ctx, workers)
        dev_results = runner(dev_tasks, ctx, workers)
        local_results = runner(local_tasks, ctx, workers)
    finally:
        ctx.close()
    unavailable = [f"{r.task.eval_name or r.task.label}: {r.error}" for r in eval_results + dev_results + local_results
                   if r.error]
    eval_sets: dict[str, list[dict]] = defaultdict(list)
    for r in eval_results:
        if not r.error:
            eval_sets[r.task.eval_name].extend({**row, "eval_set": r.task.eval_name} for row in r.rows)
    eval_rows = [row for rows in eval_sets.values() for row in rows]
    drops = Counter()
    dev: list[dict] = []
    for r in dev_results:
        if r.error:
            continue
        target = dev_targets[r.task.label]
        rows = remove_eval_overlap(dedupe(r.rows, drops), eval_rows, drops)
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[group_of(row)].append(row)
        taken = 0.0
        for group in sorted(groups, key=lambda g: hash_key(g, ctx.seed)):
            if taken >= target:
                break
            dev.extend(groups[group])
            taken += sum(row["seconds"] for row in groups[group])
    chosen = {row["audio"] for row in dev}
    for r in dev_results:  # decoded but not chosen (or an eval speaker's): the streams serve the rest
        for row in r.rows:
            if row["audio"] not in chosen:
                Path(row["audio"]).unlink(missing_ok=True)
    local_pool = remove_eval_overlap(dedupe([row for r in local_results if not r.error for row in r.rows], drops),
                                     eval_rows, drops)
    local_train, local_dev = split_dev(local_pool, float(dev_cfg.get("fraction", 0.02)),
                                       float(dev_cfg.get("max_hours_per_lang", 2.0)), 0.0, ctx.seed)
    dev += local_dev
    for row in dev:
        row["split"] = "dev"
    assert_no_leak(dev, eval_rows)
    assert_no_leak(local_train, eval_rows + dev)
    held = eval_rows + dev
    exclude = {"speaker": sorted({r["speaker"] for r in held if r.get("speaker")}),
               "group": sorted({r["group"] for r in held if r.get("group")}),
               "key": sorted({r["key"] for r in dev}),
               "audio": sorted({r["audio_sha1"] for r in held})}
    for name, rows in sorted(eval_sets.items()):
        write_jsonl(manifests / f"eval_{name}.jsonl", rows)
    write_jsonl(manifests / "local_train.jsonl", local_train)
    write_json(manifests / "stream_exclude.json", exclude)
    write_jsonl(manifests / "dev.jsonl", dev)
    report = {"mixture": mixture["name"], "streams": [{**spec.__dict__, "splits": list(spec.splits)} for spec in specs],
              "skipped": skipped,
              "unavailable": unavailable, "dev_hours": round(sum(r["seconds"] for r in dev) / 3600, 2),
              "local_train_hours": round(sum(r["seconds"] for r in local_train) / 3600, 2),
              "eval_hours": {k: round(sum(r["seconds"] for r in v) / 3600, 2) for k, v in eval_sets.items()},
              "fixed_drops": dict(drops), "excluded": {k: len(v) for k, v in exclude.items()}}
    write_json(marker, report)
    return report


def hash_key(value: str, seed: int) -> int:
    import hashlib

    return int(hashlib.sha1(f"{seed}:{value}".encode()).hexdigest()[:12], 16)


@dataclass
class StreamSetup:
    planner: StreamPlanner
    cache: ShardCache
    exclude: dict
    shares: dict[str, float]
    pools: dict[str, list[dict]] = field(default_factory=dict)

    def close(self) -> None:
        """Stop the streams' reader threads, then the shard cache (whose files they read)."""
        for stream in self.planner.streams:
            stream.close()
        self.cache.close()


def build_planner(root: Path, mixture: dict | str, frame_budget: int, allow_review: set[str] = frozenset(),
                  buffer_seconds: float = 1800.0, cache_gb: float = 40.0, prefetch: int = 1, fetch_workers: int = 4,
                  augment_prob: float = 0.0, max_items: int | None = None, lister: Callable | None = None,
                  fetch: Callable | None = None, log: Callable[[str], None] = print,
                  extra_rows: Sequence[dict] = (), fixed_shares: dict[str, float] | None = None,
                  keep_shards: bool = False, readahead: int = 0, zero_streams: Sequence[str] = (),
                  stream_scale: dict[str, float] | None = None) -> StreamSetup:
    """Streams for the mixture's hf-parquet training entries plus the local (tar) training rows,
    interleaved by temperature sampling. Requires `prepare_fixed` to have run (exclusions).

    `extra_rows` are more local rows (e.g. the non-speech negatives, lang "none"), streamed like the
    tar sources; `fixed_shares` pins sampling groups to a share (e.g. {"none": 0.02}) instead of
    their tempered share.

    `readahead` > 0 gives each hf-parquet stream a reader thread that keeps up to that many accepted
    rows ready, so the planner's sequential draws never wait on one source's parquet reads, parsing
    or header probes. The rows, their order and the checkpointed state are the same as without it.

    `zero_streams` names streams never drawn from (weight 0), e.g. streams whose filters leave too
    few rows; `stream_scale` ({name: factor}) multiplies named streams' sampling weights (draws are
    relative to the total, so the other streams' shares shrink in proportion). Both leave the plan
    unchanged, so checkpointed stream states still load."""
    root = Path(root)
    mixture = load_mixture(mixture) if isinstance(mixture, str) else mixture
    exclude_path = root / "manifests" / "stream_exclude.json"
    if not exclude_path.exists():
        raise SystemExit("run prepare_fixed first (eval/dev sets and exclusions): ablate prep --mixture X --stream")
    raw = json.loads(exclude_path.read_text(encoding="utf-8"))
    exclude = {k: set(v) for k, v in raw.items()}
    specs, _, _ = stream_specs(mixture, allow_review)
    ctx = Context(root, DiskGuard(root, None, reserve_bytes=0), seed=int(mixture.get("seed", 0)),
                  min_seconds=float(mixture.get("min_seconds", 0.5)), max_seconds=float(mixture.get("max_seconds", 20.0)),
                  log=log, **context_options(mixture))
    cache = ShardCache(root / "stream-cache", int(cache_gb * GIB), fetch=fetch, workers=fetch_workers, log=log,
                       keep=keep_shards)
    lister = lister or list_hf_shards
    streams, pools = [], {}
    disjoint = bool(mixture.get("disjoint_code_mixed_pools"))  # off for the pilot: its pools stay as they were
    for spec in sorted(specs, key=lambda s: s.code_mixed_only) if disjoint else specs:
        shards = []
        for split in spec.splits:
            shards += lister(SOURCES[spec.source], spec.lang, split, root)
        avoid = None
        if disjoint and spec.code_mixed_only:
            avoid = {shard_id(s) for other in specs if other.source == spec.source and other.lang == spec.lang
                     and not other.code_mixed_only for s in pools.get(other.name, [])}
        pools[spec.name] = select_pool(spec, shards, ctx.seed, avoid)
    pools = {spec.name: pools[spec.name] for spec in specs}
    for spec in specs:  # streams in the mixture's order: the planner's draws depend on it
        streams.append(SourceStream(spec, pools[spec.name], cache, ctx, exclude, seed=ctx.seed, prefetch=prefetch))
        if readahead:
            streams[-1].enable_readahead(readahead)
    local = root / "manifests" / "local_train.jsonl"
    if local.exists() and local.stat().st_size:
        streams += local_streams(read_jsonl(local), cache, ctx, exclude, ctx.seed)
    if extra_rows:
        streams += local_streams(list(extra_rows), cache, ctx, exclude, ctx.seed)
    sampling = mixture.get("sampling", {})
    hours = defaultdict(float)
    for stream in streams:
        hours[stream.spec.group] += stream.spec.hours
    fixed = dict(fixed_shares or {})
    shares = temperature_shares(hours, float(sampling.get("alpha", 0.5)), {**(sampling.get("max_share") or {}), **fixed},
                                {**(sampling.get("min_share") or {}), **fixed})
    planner = StreamPlanner(streams, shares, frame_budget, buffer_seconds, max_items, seed=ctx.seed,
                            augment_prob=augment_prob,
                            eval_index={"speaker": exclude["speaker"], "group": exclude["group"], "audio": set()}, log=log)
    stream_scale = dict(stream_scale or {})
    unknown = (set(zero_streams) | set(stream_scale)) - set(planner.by_name)
    if unknown:
        raise SystemExit(f"--zero-streams/--stream-scale: no such stream {sorted(unknown)}; "
                         f"streams: {sorted(planner.by_name)}")
    for i, stream in enumerate(planner.streams):
        if stream.spec.name in zero_streams:
            planner.weights[i] = 0.0
        elif stream.spec.name in stream_scale:
            planner.weights[i] *= stream_scale[stream.spec.name]
    if zero_streams:
        log(f"stream plan: not drawing from {sorted(zero_streams)} (weight 0)")
    if stream_scale:
        log(f"stream plan: weights scaled {stream_scale}")
    log("stream plan: " + ", ".join(f"{k} {v:.3f}" for k, v in planner.probabilities().items()))
    return StreamSetup(planner, cache, exclude, shares, pools)


class LocalStream(SourceStream):
    """A stream over decoded local rows (tar-only sources such as MUCS), in chunks of `chunk` rows."""

    def __init__(self, spec: StreamSpec, rows: list[dict], cache: ShardCache, ctx: Context, exclude: dict,
                 seed: int = 0, chunk: int = 256):
        pool = [{"repo": "local", "revision": "local", "path": f"{spec.name}#{i}", "size": 0,
                 "rows": rows[i:i + chunk]} for i in range(0, len(rows), chunk)]
        super().__init__(spec, pool, cache, ctx, exclude, seed)

    def next(self) -> dict | None:
        if not self.pool:
            return None
        while True:
            if self.pos >= len(self.pool):
                self.epoch, self.pos, self.row = self.epoch + 1, 0, 0
            rows = self.shard(self.epoch, self.pos)["rows"]
            if self.row >= len(rows):
                self.pos, self.row = self.pos + 1, 0
                continue
            ref = [self.epoch, self.pos, self.row]
            self.row += 1
            item = self._local(rows[ref[2]], ref)
            if item is not None:
                return item

    def _local(self, row: dict, ref: list[int]) -> dict | None:
        if (row.get("speaker") and row["speaker"] in self.exclude["speaker"]) or (
                row.get("group") and row["group"] in self.exclude["group"]) or row["key"] in self.exclude["key"]:
            self.drops["excluded"] += 1
            return None
        words = row.get("word_langs")
        if words is None and row["text"] and row["lang"] in LANG_SCRIPT:
            words = word_languages(row["text"], row["lang"])
        return {**{k: row[k] for k in ("key", "text", "lang", "source", "seconds", "audio")},
                "code_mixed": bool(row.get("code_mixed")), "word_langs": words,
                "speaker": row.get("speaker"), "group": row.get("group"), "stream": self.spec.name, "ref": ref,
                "audio_bytes": None}

    def fetch(self, refs: Sequence[list[int]], strict: bool = True) -> dict[tuple, dict]:
        out = {}
        for ref in refs:
            item = self._local(self.shard(ref[0], ref[1])["rows"][ref[2]], list(ref))
            if item is None:
                if strict:
                    raise RuntimeError(f"{self.spec.name}: buffered row {tuple(ref)} no longer reads back")
                self.drops["restore_rejected"] += 1
                continue
            out[tuple(ref)] = item
        return out


def local_streams(rows: list[dict], cache: ShardCache, ctx: Context, exclude: dict, seed: int) -> list[LocalStream]:
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by[(row["source"], row["lang"])].append(row)
    out = []
    for (source, lang), items in sorted(by.items()):
        cm = all(r.get("code_mixed") for r in items)
        spec = StreamSpec(f"{source}/{lang}/local", source, lang, ("train",), sum(r["seconds"] for r in items) / 3600,
                          CODE_SWITCHED if cm else lang, cm)
        out.append(LocalStream(spec, items, cache, ctx, exclude, seed))
    return out


def open_remote(shard: dict):
    """A seekable remote file: parquet column chunks are fetched with HTTP range requests."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=hf_token())
    return fs.open(f"datasets/{shard['repo']}@{shard['revision']}/{shard['path']}", "rb", block_size=1 << 20)


def text_pass(setup: StreamSetup, root: Path, max_rows_per_stream: int | None = None,
              log: Callable[[str], None] = print) -> list[dict]:
    """Transcripts of every stream's pool without downloading audio: only the text, markup and
    duration columns are read (HTTP range reads of the column chunks). Feeds the tokenizer and the
    label-rate report. Rows carry text, lang, seconds (None when the source has no duration column)."""
    import pyarrow.parquet as pq

    if not max_rows_per_stream:
        return _text_pass_parallel(setup, log)
    out = []
    for stream in setup.planner.streams:
        if isinstance(stream, LocalStream):
            out += [{"text": r["text"], "lang": r["lang"], "seconds": r["seconds"]}
                    for shard in stream.pool for r in shard["rows"]]
            continue
        schema = stream.source.schema
        columns = [c for c in (schema.text, schema.duration, schema.unusable) if c]
        gloss = stream.ctx.indicvoices_gloss
        count = 0
        for shard in stream.pool:
            with open_remote(shard) as handle:
                table = pq.ParquetFile(handle).read(columns=columns).to_pylist()
            for row in table:
                from indicear.markup import apply_loanwords, clean, unusable

                if schema.unusable and unusable(row.get(schema.unusable)):
                    continue
                cleaned = clean(row.get(schema.text), stream.spec.lang, stream.source.markup, indicvoices_gloss=gloss)
                if stream.ctx.loanwords is not None and stream.spec.source in stream.ctx.loanword_sources:
                    cleaned = apply_loanwords(cleaned, stream.spec.lang, stream.ctx.loanwords)
                if cleaned.drop or (stream.spec.code_mixed_only and not cleaned.code_mixed):
                    continue
                seconds = row.get(schema.duration) if schema.duration else None
                out.append({"text": cleaned.text, "lang": stream.spec.lang,
                            "seconds": float(seconds) * schema.duration_scale if seconds is not None else None})
                count += 1
                if max_rows_per_stream and count >= max_rows_per_stream:
                    break
            if max_rows_per_stream and count >= max_rows_per_stream:
                break
        log(f"text pass {stream.spec.name}: {count} transcripts")
    return out


def _text_pass_parallel(setup: StreamSetup, log: Callable[[str], None]) -> list[dict]:
    """`text_pass` with every remote shard read concurrently (the pass is network-latency bound).
    Rows come back in the same stream and shard order as the sequential pass, so the tokenizer
    trained from them is unchanged."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq

    from indicear.markup import apply_loanwords, clean, unusable

    def shard_rows(stream, shard) -> list[dict]:
        schema = stream.source.schema
        columns = [c for c in (schema.text, schema.duration, schema.unusable) if c]
        with open_remote(shard) as handle:
            table = pq.ParquetFile(handle).read(columns=columns).to_pylist()
        rows = []
        for row in table:
            if schema.unusable and unusable(row.get(schema.unusable)):
                continue
            cleaned = clean(row.get(schema.text), stream.spec.lang, stream.source.markup,
                            indicvoices_gloss=stream.ctx.indicvoices_gloss)
            if stream.ctx.loanwords is not None and stream.spec.source in stream.ctx.loanword_sources:
                cleaned = apply_loanwords(cleaned, stream.spec.lang, stream.ctx.loanwords)
            if cleaned.drop or (stream.spec.code_mixed_only and not cleaned.code_mixed):
                continue
            seconds = row.get(schema.duration) if schema.duration else None
            rows.append({"text": cleaned.text, "lang": stream.spec.lang,
                         "seconds": float(seconds) * schema.duration_scale if seconds is not None else None})
        return rows

    workers = max(4, min(32, os.cpu_count() or 8))
    # The pass feeds the tokenizer and label-rate report, for which a sample of each stream is
    # enough: INDICEAR_TEXT_PASS_SHARDS caps the shards read per stream (0 = all).
    cap = int(os.environ.get("INDICEAR_TEXT_PASS_SHARDS", "0")) or None
    out: list[dict] = []
    with ThreadPoolExecutor(workers) as pool:
        jobs = [(stream, None if isinstance(stream, LocalStream)
                 else [pool.submit(shard_rows, stream, shard) for shard in stream.pool[:cap]])
                for stream in setup.planner.streams]
        for stream, futures in jobs:
            if futures is None:
                out += [{"text": r["text"], "lang": r["lang"], "seconds": r["seconds"]}
                        for shard in stream.pool for r in shard["rows"]]
                continue
            rows = [r for future in futures for r in future.result()]
            out += rows
            log(f"text pass {stream.spec.name}: {len(rows)} transcripts")
    return out


def check_allowed(names: Sequence[str], allow_review: set[str]) -> None:
    for name in names:
        ok, why = allowed(SOURCES[name], allow_review)
        if not ok:
            raise SystemExit(why)


def download_pool(setup: StreamSetup, workers: int = 4, log: Callable[[str], None] = print) -> dict:
    """Download every shard of every stream's pool into the cache directory (kept), so a
    `keep_shards` run reads them locally; finished files are skipped. Returns counts and GiB."""
    shards = [shard for stream in setup.planner.streams if type(stream) is SourceStream for shard in stream.pool]
    total = sum(int(s.get("size") or 0) for s in shards)
    fetch = setup.cache.fetch
    done = {"files": 0, "bytes": 0}
    lock = threading.Lock()
    started = time.time()

    def one(shard: dict) -> None:
        fetch(shard, setup.cache.dir)
        with lock:
            done["files"] += 1
            done["bytes"] += int(shard.get("size") or 0)
            if done["files"] % 10 == 0 or done["files"] == len(shards):
                rate = done["bytes"] / 1e6 / max(time.time() - started, 1e-6)
                log(f"pool: {done['files']}/{len(shards)} shards, {done['bytes'] / GIB:.1f}/{total / GIB:.1f} GiB, "
                    f"{rate:.1f} MB/s (including files already present)")

    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="pool-fetch") as pool:
        list(pool.map(one, shards))
    return {"shards": len(shards), "gib": round(total / GIB, 1)}
