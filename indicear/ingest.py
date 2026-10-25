"""Download, decode and transcribe-clean source shards into manifest rows (resumable).

One *task* is (source, lang, split, hours). A task walks its source's shards in a seeded order and
stops once the kept audio reaches the requested hours, so only the hours asked for are fetched
(rounded up to one shard). Each shard is processed once:

1. disk check (free space and the optional `--disk-budget-gb`), before anything is fetched;
2. download: `hf_hub_download` of one parquet shard at the pinned revision, or an HTTP download
   with Range resume for the OpenSLR / object-store tars (size checked against the pin);
3. transcript cleaning (`indicear.markup`) before any audio is decoded, so dropped rows cost nothing;
4. parallel decode (thread pool sized to the CPUs; soundfile/soxr release the GIL and the venv is
   free-threaded) to 16 kHz mono FLAC under ROOT/audio/<source>/<lang>/<split>/, with a SHA-1 of the
   16-bit PCM for leak checks;
5. rows to `ROOT/sources/<source>/<lang>/<split>/<shard>.jsonl`, then the completion marker
   (`runtime.shards.write_marker`, written last). A rerun reuses finished shards; a shard that was
   cut at an hours cap is redone only when a larger cap needs more of it.

The HF token comes from the HF_TOKEN environment variable, else from huggingface_hub's saved login. It is
never printed or written. Downloaded shards are deleted after decoding unless `keep_downloads`.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from indicear.common import SAMPLE_RATE, read_jsonl, write_jsonl
from indicear.markup import apply_loanwords, clean, native_reference, unusable
from indicear.runtime.shards import read_marker, write_marker
from indicear.sources import SOURCES, Source
from indicear.text import LANG_SCRIPT

FLAC_MB_PER_HOUR = 70.0  # 16 kHz mono 16-bit FLAC of speech (planning estimate, ~60% of PCM)
GIB = 2**30
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")



# ffmpeg's audio libraries start an OpenMP pool sized to every core per process; with dozens of
# concurrent decodes that oversubscribes the CPU, so each subprocess gets one OpenMP thread.
FFMPEG_ENV = {**os.environ, "OMP_NUM_THREADS": "1"}

class BudgetExceeded(SystemExit):
    """Clean stop: the next shard would not fit on disk or in the disk budget."""


class SourceUnavailable(RuntimeError):
    """Gated without access, missing token, moved, or failed a pin check."""


def hf_token() -> str | None:
    """HF_TOKEN from the environment, else None so huggingface_hub uses the saved login
    (~/.cache/huggingface/token, root-only on the VM)."""
    return os.environ.get("HF_TOKEN") or None


# ---------------------------------------------------------------- disk guard


class DiskGuard:
    """Refuses a shard when free space (minus a reserve) or the byte budget would be exceeded.

    The same df check as `runtime.shards.check_disk`, plus a budget across the whole ingest and
    reservations so concurrent shards cannot oversubscribe the disk.
    """

    def __init__(self, root: Path, budget_bytes: int | None = None, reserve_bytes: int = 2 * GIB,
                 used_bytes: int = 0, log: Callable[[str], None] = print,
                 free_bytes: Callable[[], int] | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)  # df needs an existing path
        self.budget = budget_bytes
        self.reserve = reserve_bytes
        self.used = used_bytes
        self.pending = 0
        self.log = log
        self._free = free_bytes or (lambda: shutil.disk_usage(self.root).free)
        self._lock = threading.Lock()

    def acquire(self, need: int, label: str) -> int:
        with self._lock:
            free = self._free() - self.pending
            if need + self.reserve > free:
                raise BudgetExceeded(
                    f"stopping before {label}: needs ~{need / GIB:.2f} GiB but only {free / GIB:.2f} GiB is free "
                    f"(keeping {self.reserve / GIB:.1f} GiB in reserve). Finished shards are kept; free space or "
                    "use a larger disk, then rerun the same command to continue.")
            if self.budget is not None and self.used + self.pending + need > self.budget:
                raise BudgetExceeded(
                    f"stopping before {label}: the disk budget of {self.budget / GIB:.1f} GiB would be exceeded "
                    f"({(self.used + self.pending) / GIB:.2f} GiB used or reserved, ~{need / GIB:.2f} GiB more needed). "
                    "Finished shards are kept; raise --disk-budget-gb or lower the hours, then rerun.")
            self.pending += need
            return need

    def release(self, reserved: int, written: int) -> None:
        with self._lock:
            self.pending -= reserved
            self.used += written


# ---------------------------------------------------------------- audio


def decode_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Any container soundfile reads (wav, flac, ogg, mp3); else ffmpeg (m4a/aac)."""
    import soundfile as sf

    try:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001 - fall back to ffmpeg for containers libsndfile lacks
        return ffmpeg_decode(data), SAMPLE_RATE
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, rate


def ffmpeg_decode(data: bytes) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed to decode this audio (m4a/aac) but is not installed")
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", "pipe:0", "-f", "f32le", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "pipe:1"], input=data, capture_output=True, check=True, env=FFMPEG_ENV).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate != SAMPLE_RATE:
        import soxr

        audio = soxr.resample(audio, rate, SAMPLE_RATE)
    return np.asarray(audio, dtype=np.float32)


def write_flac(path: Path, audio16k: np.ndarray) -> tuple[str, float, int]:
    """Write 16-bit FLAC atomically. Returns (sha1 of the PCM, seconds, bytes on disk)."""
    import soundfile as sf

    pcm = (np.clip(audio16k, -1.0, 1.0) * 32767.0).round().astype("<i2")
    digest = hashlib.sha1(pcm.tobytes()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    sf.write(tmp, pcm, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    os.replace(tmp, path)
    return digest, len(pcm) / SAMPLE_RATE, path.stat().st_size


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_id(value: str) -> str:
    stem = _SAFE.sub("_", value).strip("._") or "utt"
    if len(stem) > 120 or stem != value:
        stem = f"{stem[:100]}-{hashlib.sha1(value.encode()).hexdigest()[:8]}"
    return stem


# ---------------------------------------------------------------- speaker / group ids

_VAANI_FILE = re.compile(r"IISc_VaaniProject_[A-Z]_[^_]+_[^_]+_([^_]+)_")
_KATHBATH_FILE = re.compile(r"^\d+-(\d+)-[mfMF]")


def speaker_and_group(source: Source, row: dict, audio_path: str) -> tuple[str | None, str | None]:
    """(authoritative speaker id, grouping id for splits). Both namespaced by the source family."""
    ns = source.namespace()
    speaker = None
    if source.schema and source.schema.speaker and row.get(source.schema.speaker) not in (None, "", "NA"):
        speaker = str(row[source.schema.speaker])
    if ns == "kathbath":  # file names carry the speaker in the same form as the test tars
        name = str(row.get("fname") or Path(audio_path).name)
        match = _KATHBATH_FILE.match(Path(name).name)
        if match:
            speaker = match.group(1)
    group = speaker
    if group is None and source.speaker_ids == "filename":
        match = _VAANI_FILE.search(audio_path)
        group = f"fn:{match.group(1)}" if match else None
    if group is None and source.speaker_ids == "recording":
        ident = row.get(source.schema.id) if source.schema and source.schema.id else None
        stem = Path(str(ident or audio_path)).stem
        if "_chunk" in stem:
            group = "rec:" + stem.split("_chunk")[0]
        elif "_SLASH_" in stem:
            group = "rec:" + stem.split("_SLASH_")[0]
        elif stem:
            group = "rec:" + stem.rsplit("_", 1)[0]
    return (f"{ns}:{speaker}" if speaker else None), (f"{ns}:{group}" if group else None)


def style_tag(source: Source) -> str:
    return source.style.split()[0].strip(",;")


# ---------------------------------------------------------------- tasks and context


@dataclass
class Task:
    source: str
    lang: str
    split: str
    hours: float  # math.inf: the whole split
    role: str = "train"  # train | eval
    code_mixed_only: bool = False
    reverse: bool = False  # walk shards from the end (mining tasks stay disjoint from the main task)
    eval_name: str | None = None
    weight: float = 1.0  # streaming: sampling weight within its language group, per hour (mixture "weight")

    @property
    def label(self) -> str:
        extra = " code-mixed" if self.code_mixed_only else ""
        return f"{self.source}/{self.lang}/{self.split}{extra}"


@dataclass
class Context:
    root: Path
    guard: DiskGuard
    seed: int = 0
    min_seconds: float = 0.5
    max_seconds: float = 20.0
    decode_workers: int = max(1, os.cpu_count() or 4)
    keep_downloads: bool = False
    indicvoices_gloss: bool = True
    loanwords: object | None = None  # indicear.loanwords.LoanwordNormaliser (mixture "loanwords")
    loanword_sources: frozenset[str] = frozenset()  # sources whose transcripts it rewrites
    log: Callable[[str], None] = print
    stopping: threading.Event = field(default_factory=threading.Event)
    _pool: ThreadPoolExecutor | None = None

    @property
    def pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.decode_workers, thread_name_prefix="decode")
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


@functools.lru_cache(maxsize=4)
def _normaliser(resources: str | None):
    from indicear.loanwords import LoanwordNormaliser

    return LoanwordNormaliser.load(Path(resources) if resources else None)


def context_options(mixture: dict) -> dict:
    """Context fields a mixture sets: the IndicVoices gloss rewrite, and the loanword normaliser
    with the sources it applies to (`"loanwords": {"sources": [...], "resources": DIR?}`)."""
    options = {"indicvoices_gloss": bool(mixture.get("indicvoices_gloss", True))}
    loan = mixture.get("loanwords")
    if loan:
        options["loanwords"] = _normaliser(loan.get("resources"))
        options["loanword_sources"] = frozenset(loan["sources"])
    return options


@dataclass
class TaskResult:
    task: Task
    rows: list[dict] = field(default_factory=list)
    drops: Counter = field(default_factory=Counter)
    notes: Counter = field(default_factory=Counter)
    shards: int = 0
    seconds: float = 0.0
    error: str | None = None

    @property
    def shortfall_hours(self) -> float:
        if self.task.hours == float("inf"):
            return 0.0
        return max(0.0, self.task.hours - self.seconds / 3600)


def shard_dir(root: Path, task: Task) -> Path:
    tag = "cm" if task.code_mixed_only else task.split
    if task.role == "dev":  # streaming dev picks: unchosen rows' audio is deleted, so never shared
        tag = f"dev-{task.split}"
    return Path(root) / "sources" / task.source / task.lang / tag


def audio_dir(root: Path, task: Task) -> Path:
    return Path(root) / "audio" / task.source / task.lang / task.split


# ---------------------------------------------------------------- row building (shared)


@dataclass
class Pending:
    """A row whose transcript passed; audio still to decode."""

    key: str
    raw: str
    cleaned: object
    meta: dict
    seconds_hint: float | None
    load: Callable[[], tuple[np.ndarray, int]]
    data: bytes | None = None  # the encoded audio, when it came from a parquet row


def _decode_job(pending: Pending, path: Path) -> tuple[Pending, Path, str | None, tuple | None]:
    try:
        audio, rate = pending.load()
        audio = to_16k(audio, rate)
        if not len(audio):
            return pending, path, "empty_audio", None
        return pending, path, None, write_flac(path, audio)
    except Exception as error:  # noqa: BLE001 - one bad file must not stop a shard
        return pending, path, f"decode_error:{type(error).__name__}", None


def finish_rows(pendings: Iterable[Pending], task: Task, ctx: Context, budget_seconds: float,
                drops: Counter) -> tuple[list[dict], bool, int]:
    """Decode in parallel, then accept rows in order until the budget is met (deterministic).
    Returns (rows, capped, bytes written). Files decoded past the cap are removed."""
    out_dir = audio_dir(ctx.root, task)
    rows: list[dict] = []
    seconds = 0.0
    written = 0
    capped = False
    batch: list[Pending] = []

    def flush() -> bool:
        nonlocal seconds, written, capped
        jobs = [ctx.pool.submit(_decode_job, p, out_dir / f"{safe_id(p.key.rsplit('/', 1)[-1])}.flac")
                for p in batch]
        for job in jobs:
            pending, path, error, result = job.result()
            if capped:
                if result is not None:
                    path.unlink(missing_ok=True)
                continue
            if error:
                drops[error] += 1
                continue
            digest, secs, size = result
            if secs < ctx.min_seconds or secs > ctx.max_seconds:
                drops["too_short" if secs < ctx.min_seconds else "too_long"] += 1
                path.unlink(missing_ok=True)
                continue
            cleaned = pending.cleaned
            rows.append({
                "key": pending.key, "audio": str(path), "text": cleaned.text, "raw": pending.raw,
                "lang": task.lang, "script": LANG_SCRIPT[task.lang], "seconds": round(secs, 3),
                "source": task.source, "split": task.split, "code_mixed": cleaned.code_mixed,
                "glosses": cleaned.glosses, "word_langs": cleaned.word_langs, "audio_sha1": digest, **pending.meta,
            })
            seconds += secs
            written += size
            if seconds >= budget_seconds:
                capped = True
        batch.clear()
        return capped

    for pending in pendings:
        if pending.seconds_hint is not None and not ctx.min_seconds <= pending.seconds_hint <= ctx.max_seconds:
            drops["too_short" if pending.seconds_hint < ctx.min_seconds else "too_long"] += 1
            continue
        batch.append(pending)
        if len(batch) >= 4 * ctx.decode_workers and flush():
            break
    if batch and not capped:
        flush()
    return rows, capped, written


def clean_row(source: Source, task: Task, raw: str, ctx: Context, drops: Counter, notes: Counter):
    cleaned = clean(raw, task.lang, source.markup, indicvoices_gloss=ctx.indicvoices_gloss)
    if ctx.loanwords is not None and source.name in ctx.loanword_sources:
        cleaned = apply_loanwords(cleaned, task.lang, ctx.loanwords, primary=task.role != "eval")
    notes.update(cleaned.notes)
    if cleaned.drop:
        drops[cleaned.drop.split(":")[0] if cleaned.drop.startswith("foreign_script") else cleaned.drop] += 1
        return None
    if task.code_mixed_only and not cleaned.code_mixed:
        drops["not_code_mixed"] += 1
        return None
    return cleaned


# ---------------------------------------------------------------- HF parquet shards


def list_hf_shards(source: Source, lang: str, split: str, root: Path) -> list[dict]:
    """Parquet files of (lang, split) at the pinned revision(s); cached for offline resume."""
    cache = Path(root) / "sources" / source.name / f"listing-{lang}-{split}.json"
    if cache.exists():
        return read_jsonl(cache)
    from huggingface_hub import HfApi

    prefix_tmpl = source.splits[split]
    config = source.langs[lang]
    api = HfApi()
    shards = []
    for repo, revision in source.repo_for(lang):
        prefix = prefix_tmpl.format(config=config)
        directory = prefix.rsplit("/", 1)[0] if "/" in prefix else ""
        try:
            entries = api.list_repo_tree(repo, path_in_repo=directory or None, revision=revision,
                                         repo_type="dataset", token=hf_token())
            for entry in entries:
                path = getattr(entry, "path", "")
                if path.startswith(prefix) and path.endswith(".parquet"):
                    shards.append({"repo": repo, "revision": revision, "path": path, "size": entry.size})
        except Exception as error:
            raise SourceUnavailable(_explain(source, repo, error)) from error
    if not shards:
        raise SourceUnavailable(f"{source.name}: no parquet files under {prefix_tmpl.format(config=config)!r}")
    shards.sort(key=lambda s: (s["repo"], s["path"]))
    write_jsonl(cache, shards)
    return shards


def _explain(source: Source, repo: str, error: Exception) -> str:
    kind = type(error).__name__
    if "Gated" in kind or "401" in str(error) or "403" in str(error):
        has = "set" if os.environ.get("HF_TOKEN") else "not set"
        return (f"{source.name}: {repo} is gated and not accessible (HF_TOKEN is {has}). Accept the terms at "
                f"https://huggingface.co/datasets/{repo} with the account whose token is in HF_TOKEN.")
    if "NotFound" in kind or "404" in str(error):
        return f"{source.name}: {repo} or its pinned revision was not found (moved or rewritten?)"
    return f"{source.name}: {repo}: {kind}"


def order_shards(shards: list[dict], task: Task, seed: int) -> list[dict]:
    order = list(shards)
    random.Random(f"{seed}:{task.source}:{task.lang}:{task.split}").shuffle(order)
    return order[::-1] if task.reverse else order


def iter_parquet_rows(path: Path, columns: list[str], required: Iterable[str] = (), batch_rows: int = 64) -> Iterator[dict]:
    """Rows of one parquet file in small batches (bounded memory even for 500 KB audio rows).
    Raises SourceUnavailable when a required column is missing (the schema changed upstream)."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    missing = [c for c in required if c not in names]
    if missing:
        raise SourceUnavailable(f"{Path(path).name}: columns {missing} not found (have {names})")
    present = [c for c in columns if c in names]
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=present):
        yield from batch.to_pylist()


def source_columns(source: Source) -> list[str]:
    schema = source.schema
    columns = [schema.audio, schema.text, *schema.refs, *(c for c, _ in schema.require), *schema.keep]
    columns += [c for c in (schema.speaker, schema.duration, schema.id, schema.unusable) if c]
    if source.namespace() == "kathbath":
        columns.append("fname")
    return list(dict.fromkeys(columns))


def parse_row(source: Source, task: Task, row: dict, ctx: Context, drops: Counter, notes: Counter,
              seen: Counter | None = None) -> Pending | None:
    """One parquet row -> a Pending (transcript cleaned, audio still encoded), or None (counted drop).
    Shared by the download path (`parquet_pendings`) and the streaming path (`indicear.stream`)."""
    schema = source.schema
    if any(row.get(column) != value for column, value in schema.require):
        drops["not_gold" if source.name == "voxpopuli" else "filtered"] += 1
        return None
    audio = row.get(schema.audio) or {}
    audio_path = str(audio.get("path") or "")
    raw = row.get(schema.text)
    if schema.unusable and unusable(row.get(schema.unusable)):
        drops["unintelligible"] += 1
        return None
    cleaned = clean_row(source, task, raw, ctx, drops, notes)
    if cleaned is None:
        return None
    data = audio.get("bytes")
    if not data:
        drops["no_audio_bytes"] += 1
        return None
    ident = str(row.get(schema.id) or Path(audio_path).stem or hashlib.sha1(data[:4096]).hexdigest())
    if Path(ident).suffix.lower() in AUDIO_SUFFIXES:
        ident = Path(ident).stem
    if seen is not None:
        seen[ident] += 1
        if seen[ident] > 1:
            ident = f"{ident}~{seen[ident]}"
    speaker, group = speaker_and_group(source, row, audio_path)
    meta = {"speaker": speaker, "group": group, "style": style_tag(source)}
    meta.update({k: row.get(k) for k in schema.keep if row.get(k) is not None})
    refs = [clean(row.get(r) or "", task.lang, source.markup).text for r in schema.refs]
    if cleaned.glosses:  # English left in native script: the second reference of a script-normalised WER
        refs.append(native_reference(raw, task.lang, source.markup))
    refs += cleaned.extra_refs  # loanword normalisation: the other script's form
    refs = [r for r in refs if r]
    if refs:
        meta["refs"] = refs
    hint = row.get(schema.duration) if schema.duration else None
    hint = float(hint) * schema.duration_scale if hint is not None else None
    return Pending(f"{task.source}/{task.lang}/{task.split}/{ident}", raw, cleaned, meta, hint,
                   lambda data=data: decode_bytes(data), data)


def parquet_pendings(source: Source, task: Task, path: Path, ctx: Context, drops: Counter,
                     notes: Counter) -> Iterator[Pending]:
    seen: Counter = Counter()
    schema = source.schema
    for row in iter_parquet_rows(path, source_columns(source), required=(schema.audio, schema.text)):
        pending = parse_row(source, task, row, ctx, drops, notes, seen)
        if pending is not None:
            yield pending


def download_hf(shard: dict, source: Source, root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(shard["repo"], shard["path"], repo_type="dataset",
                                    revision=shard["revision"], token=hf_token(),
                                    local_dir=Path(root) / "downloads" / source.name / shard["repo"].replace("/", "__")))
    except Exception as error:
        raise SourceUnavailable(_explain(source, shard["repo"], error)) from error


def run_parquet_task(task: Task, ctx: Context, downloader: Callable | None = None,
                     lister: Callable | None = None) -> TaskResult:
    source = SOURCES[task.source]
    downloader, lister = downloader or download_hf, lister or list_hf_shards
    result = TaskResult(task)
    budget = task.hours * 3600
    for shard in order_shards(lister(source, task.lang, task.split, ctx.root), task, ctx.seed):
        remaining = budget - result.seconds
        if remaining <= 0 or ctx.stopping.is_set():
            break
        stem = shard_dir(ctx.root, task) / Path(shard["path"]).stem
        if len(source.repo_for(task.lang)) > 1:
            stem = stem.with_name(f"{shard['repo'].split('/')[-1]}__{stem.name}")
        rows = _reuse(stem, remaining)
        if rows is None:
            rows = _process_parquet_shard(source, task, shard, stem, remaining, ctx, downloader)
        marker = read_marker(stem) or {}
        result.drops.update(marker.get("drops", {}))
        result.notes.update(marker.get("notes", {}))
        rows = _trim(rows, remaining)
        result.rows.extend(rows)
        result.seconds += sum(r["seconds"] for r in rows)
        result.shards += 1
    return result


def _reuse(stem: Path, remaining: float) -> list[dict] | None:
    marker = read_marker(stem)
    if marker is None:
        return None
    if marker.get("capped") and marker.get("seconds", 0.0) < remaining - 1e-6:
        return None  # cut at a smaller cap last time and more is needed now: redo the shard
    return read_jsonl(Path(f"{stem}.jsonl"))


def _trim(rows: list[dict], remaining: float) -> list[dict]:
    out, seconds = [], 0.0
    for row in rows:
        if seconds >= remaining:
            break
        out.append(row)
        seconds += row["seconds"]
    return out


def _process_parquet_shard(source: Source, task: Task, shard: dict, stem: Path, remaining: float,
                           ctx: Context, downloader: Callable) -> list[dict]:
    hours = (shard["size"] or 0) / 1e6 / max(source.mb_per_hour, 1.0)
    need = int((shard["size"] or 0) + min(hours, remaining / 3600) * FLAC_MB_PER_HOUR * 1e6 * 1.1)
    reserved = ctx.guard.acquire(need, f"{task.label} {shard['path']}")
    written = 0
    try:
        started = time.time()
        path = downloader(shard, source, ctx.root)
        drops, notes = Counter(), Counter()
        rows, capped, written = finish_rows(parquet_pendings(source, task, path, ctx, drops, notes),
                                            task, ctx, remaining, drops)
        write_jsonl(Path(f"{stem}.jsonl"), rows)
        write_marker(stem, {"source": source.name, "repo": shard["repo"], "revision": shard["revision"],
                            "file": shard["path"], "rows": len(rows), "seconds": round(sum(r["seconds"] for r in rows), 3),
                            "capped": capped, "bytes": written, "drops": dict(drops), "notes": dict(notes)})
        ctx.log(f"{task.label}: {shard['path']} -> {len(rows)} rows, "
                f"{sum(r['seconds'] for r in rows) / 3600:.2f} h in {time.time() - started:.0f}s"
                f"{' (cap reached)' if capped else ''}")
        if not ctx.keep_downloads:
            Path(path).unlink(missing_ok=True)
        return rows
    finally:
        ctx.guard.release(reserved, written)


# ---------------------------------------------------------------- URL tars (MUCS, Kathbath tests)


_DOWNLOAD_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_LOCK = threading.Lock()


def _path_lock(path: Path) -> threading.Lock:
    with _LOCKS_LOCK:
        return _DOWNLOAD_LOCKS.setdefault(str(Path(path).resolve()), threading.Lock())


def download_url(spec: dict, dest: Path, guard: DiskGuard | None = None, label: str = "") -> Path:
    """HTTP download with Range resume; the final size must equal the pinned size when one is set.
    Concurrent tasks asking for the same file wait for one download."""
    with _path_lock(dest):
        return _download_url(spec, Path(dest), guard, label)


def _download_url(spec: dict, dest: Path, guard: DiskGuard | None, label: str) -> Path:
    size = spec.get("size")
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    reserved = guard.acquire(max(0, (size or 0) - (part.stat().st_size if part.exists() else 0)),
                             label or dest.name) if guard else 0
    errors = []
    try:
        for url in spec.get("urls") or [spec["url"]]:
            try:
                have = part.stat().st_size if part.exists() else 0
                request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
                with urllib.request.urlopen(request, timeout=60) as response, part.open("ab" if have else "wb") as out:
                    if have and response.status != 206:  # server ignored Range: start over
                        out.truncate(0)
                    shutil.copyfileobj(response, out, 8 << 20)
                if size is not None and part.stat().st_size != size:
                    raise SourceUnavailable(f"{url}: got {part.stat().st_size} bytes, pinned {size}")
                os.replace(part, dest)
                return dest
            except (urllib.error.URLError, OSError, TimeoutError) as error:
                errors.append(f"{url}: {type(error).__name__}")
        raise SourceUnavailable("download failed: " + "; ".join(errors))
    finally:
        if guard:
            guard.release(reserved, dest.stat().st_size if dest.exists() else 0)


def _kaldi_map(text: str, maxsplit: int = 1) -> dict[str, list[str]]:
    out = {}
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=maxsplit)
        if len(parts) == maxsplit + 1:
            out[parts[0]] = parts[1:]
    return out


def mucs_pendings(archive: Path, task: Task, ctx: Context, drops: Counter, notes: Counter) -> Iterator[Pending]:
    """MUCS Kaldi layout: <split>/transcripts/{text,segments,utt2spk} + <split>/<recording>.wav."""
    source = SOURCES["mucs"]
    meta_files: dict[str, str] = {}
    with tarfile.open(archive, "r:*") as tar:  # pass 1: transcripts (they may sit after the audio)
        for member in tar:
            name = Path(member.name)
            if member.isfile() and name.parent.name == "transcripts" and name.name in ("text", "segments", "utt2spk"):
                meta_files[name.name] = tar.extractfile(member).read().decode("utf-8", "replace")
    missing = {"text", "segments"} - set(meta_files)
    if missing:
        raise SourceUnavailable(f"mucs: {archive.name} has no transcripts/{sorted(missing)}")
    texts = {k: v[0] for k, v in _kaldi_map(meta_files["text"]).items()}
    speakers = {k: v[0] for k, v in _kaldi_map(meta_files.get("utt2spk", "")).items()}
    by_recording: dict[str, list[tuple[str, float, float]]] = {}
    for utt, fields in _kaldi_map(meta_files["segments"], maxsplit=3).items():
        by_recording.setdefault(fields[0], []).append((utt, float(fields[1]), float(fields[2])))
    with tarfile.open(archive, "r:*") as tar:  # pass 2: audio, in archive order
        for member in tar:
            if not (member.isfile() and member.name.endswith(".wav")):
                continue
            segments = by_recording.get(Path(member.name).stem)
            if not segments:
                continue
            audio, rate = decode_bytes(tar.extractfile(member).read())
            for utt, start, end in sorted(segments, key=lambda s: s[1]):
                raw = texts.get(utt)
                cleaned = clean_row(source, task, raw, ctx, drops, notes)
                if cleaned is None:
                    continue
                piece = audio[int(start * rate): int(end * rate)].copy()
                speaker = speakers.get(utt)
                meta = {"speaker": f"mucs:{speaker}" if speaker else None, "group": f"mucs:{speaker or Path(member.name).stem}",
                        "style": "spontaneous", "pair": f"{task.lang}+en"}
                yield Pending(f"mucs/{task.lang}/{task.split}/{utt}", raw, cleaned, meta, end - start,
                              lambda piece=piece, rate=rate: (piece, rate))


def kathbath_test_pendings(transcripts: Path, audio_tar: Path, task: Task, ctx: Context, drops: Counter,
                           notes: Counter) -> Iterator[Pending]:
    """Transcripts: */<language>/<split>/transcription_n2w.txt ('<file>\\t<text>'); audio: */<language>/*/<file>.m4a."""
    source = SOURCES["kathbath_test"]
    language = source.langs[task.lang]
    texts: dict[str, str] = {}
    with tarfile.open(transcripts, "r:*") as tar:
        for member in tar:
            parts = Path(member.name).parts
            if member.isfile() and parts[-1] == "transcription_n2w.txt" and language in parts:
                for line in tar.extractfile(member).read().decode("utf-8", "replace").splitlines():
                    name, _, text = line.partition("\t")
                    if name and text:
                        texts[Path(name).name] = text
    with tarfile.open(audio_tar, "r:*") as tar:
        for member in tar:
            parts = Path(member.name).parts
            if not (member.isfile() and member.name.endswith(".m4a") and language in parts):
                continue
            name = parts[-1]
            raw = texts.get(name)
            if raw is None:
                drops["no_transcript"] += 1
                continue
            cleaned = clean_row(source, task, raw, ctx, drops, notes)
            if cleaned is None:
                continue
            match = _KATHBATH_FILE.match(name)
            speaker = f"kathbath:{match.group(1)}" if match else None
            data = tar.extractfile(member).read()
            meta = {"speaker": speaker, "group": speaker, "style": "read"}
            if cleaned.extra_refs:
                meta["refs"] = list(cleaned.extra_refs)
            yield Pending(f"kathbath_test/{task.lang}/{task.split}/{Path(name).stem}", raw, cleaned, meta, None,
                          lambda data=data: decode_bytes(data))


def run_tar_task(task: Task, ctx: Context, fetch: Callable | None = None) -> TaskResult:
    """One archive (or archive pair) per task; same marker and cap rules as parquet shards."""
    source = SOURCES[task.source]
    fetch = fetch or download_url
    result = TaskResult(task)
    budget = task.hours * 3600
    stem = shard_dir(ctx.root, task) / "archive"
    rows = _reuse(stem, budget)
    if rows is None:
        downloads = Path(ctx.root) / "downloads" / source.name
        drops, notes = Counter(), Counter()
        if task.source == "mucs":
            name = f"{source.langs[task.lang]}_{task.split}" if task.split != "blind" else "blindtest"
            spec = source.files[name]
            archive = fetch(spec, downloads / f"{name}.tar.gz", ctx.guard, task.label)
            pendings = mucs_pendings(archive, task, ctx, drops, notes)
            local = [archive]
        elif task.source == "kathbath_test":
            noise = task.split.split("_", 1)[0]
            kind = "testkn" if task.split.endswith("known") and not task.split.endswith("unknown") else "testunk"
            transcripts = fetch(source.files[f"{noise}_transcripts"], downloads / f"{noise}_transcripts_n2w.tar",
                                ctx.guard, task.label)
            audio = fetch(source.files[f"{noise}_{kind}"], downloads / f"{noise}_{kind}_audio.tar", ctx.guard, task.label)
            pendings = kathbath_test_pendings(transcripts, audio, task, ctx, drops, notes)
            local = []  # shared by every language: kept until the ingest ends
        else:
            raise ValueError(f"{task.source} is not a tar source")
        archive_bytes = sum(Path(p).stat().st_size for p in (local or [audio]))
        share = len(source.langs) if task.source == "kathbath_test" else 1
        estimate_hours = archive_bytes / 1e6 / max(source.mb_per_hour, 1.0) / share
        reserved = ctx.guard.acquire(int(min(budget / 3600, estimate_hours) * FLAC_MB_PER_HOUR * 1e6 * 1.1),
                                     task.label)
        written = 0
        try:
            rows, capped, written = finish_rows(pendings, task, ctx, budget, drops)
            write_jsonl(Path(f"{stem}.jsonl"), rows)
            write_marker(stem, {"source": source.name, "rows": len(rows), "capped": capped, "bytes": written,
                                "seconds": round(sum(r["seconds"] for r in rows), 3), "drops": dict(drops),
                                "notes": dict(notes)})
        finally:
            ctx.guard.release(reserved, written)
        if not ctx.keep_downloads:
            for path in local:
                Path(path).unlink(missing_ok=True)
    marker = read_marker(stem) or {}
    result.drops.update(marker.get("drops", {}))
    result.notes.update(marker.get("notes", {}))
    result.rows = _trim(rows, budget)
    result.seconds = sum(r["seconds"] for r in result.rows)
    result.shards = 1
    return result


def run_task(task: Task, ctx: Context) -> TaskResult:
    source = SOURCES[task.source]
    if source.kind == "hf-parquet":
        return run_parquet_task(task, ctx)
    if source.kind == "url-tar":
        return run_tar_task(task, ctx)
    raise ValueError(f"{task.source}: kind {source.kind} has no loader")


def run_tasks(tasks: list[Task], ctx: Context, workers: int = 4) -> list[TaskResult]:
    """Run tasks concurrently (downloads overlap; decoding shares ctx.pool). A BudgetExceeded stop
    lets running shards finish their marker, then re-raises; SourceUnavailable is recorded per task."""
    results: list[TaskResult | None] = [None] * len(tasks)
    stop: list[BaseException] = []

    def one(index: int) -> None:
        if stop:
            return
        try:
            results[index] = run_task(tasks[index], ctx)
        except SourceUnavailable as error:
            results[index] = TaskResult(tasks[index], error=str(error))
            ctx.log(f"UNAVAILABLE {tasks[index].label}: {error}")
        except BaseException as error:  # noqa: BLE001 - BudgetExceeded and real failures stop the ingest
            ctx.stopping.set()
            stop.append(error)

    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="task") as pool:
        list(pool.map(one, range(len(tasks))))
    if stop:
        raise stop[0]
    return [r for r in results if r is not None]


def used_bytes(root: Path) -> int:
    """Bytes of audio already written by finished shards (from their markers)."""
    total = 0
    base = Path(root) / "sources"
    for marker in base.rglob("*.done.json") if base.exists() else ():
        try:
            total += int(json.loads(marker.read_text(encoding="utf-8")).get("bytes", 0))
        except (OSError, ValueError):
            continue
    return total
