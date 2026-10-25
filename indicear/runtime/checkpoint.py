"""Resumable runs: atomic checkpoints, a LATEST.json manifest, automatic resume, signals.

- `save()` writes `ckpt-<step>.pt` through a temporary file, fsync, then `os.replace`; keeps the
  last `keep` checkpoints plus `best.pt`; then rewrites `LATEST.json` atomically. The manifest
  lists every complete file (checkpoints, best, published artifacts) with step, sha256, size and
  whether it is the best, so an off-VM puller (scripts/sync_runs.sh) copies only finished files
  and verifies them. `publish()` adds final artifacts; `mark("DONE" | "FAILED")` ends the run.
- `copy_dir` (e.g. a Google Drive mount) is an optional second copy, never the primary: the
  primary off-VM copy is a workstation, pulled over SSH (`scripts/sync_runs.sh`).
- `restore()` picks the newest checkpoint that loads (a truncated or corrupt file falls back to
  the previous one), first pulling checkpoints back from `copy_dir` when the run directory is
  empty. It refuses a checkpoint whose config hash differs unless `force_new_run`, in which
  case old checkpoints are moved aside.
- `StopFlag` turns SIGTERM/SIGINT into a flag the training loop checks after each step, so the
  final checkpoint is written before exit.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

CKPT_GLOB = "ckpt-*.pt"
SYNC_FILES = ("*.csv", "*.json", "*.jsonl", "*.safetensors", "*.log", "*.sh", "DONE", "FAILED")
MANIFEST = "LATEST.json"


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def git_revision(path: str | Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path or Path(__file__).parent,
            capture_output=True, text=True, timeout=10, check=False,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=path or Path(__file__).parent,
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
        if out.returncode == 0:
            return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    # Code copied to a VM without .git carries a REVISION file written at upload time.
    for parent in Path(path or Path(__file__).parent).resolve().parents:
        if (parent / "REVISION").is_file():
            return (parent / "REVISION").read_text().strip()
    return "unknown"


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def rng_state() -> dict:
    """Python, numpy and torch CPU generators, plus every device generator present (CUDA, MPS)."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if _mps_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def derived_seed(state: dict) -> int:
    """A deterministic 32-bit seed from a checkpoint's CPU torch generator state: the seed for a
    device generator the checkpoint has no state for (a CUDA run resumed on MPS)."""
    import zlib

    return zlib.crc32(bytes(torch.as_tensor(state["torch"]).numpy().tobytes()))


def set_rng_state(state: dict, log: Callable[[str], None] | None = None) -> list[str]:
    """Restore the generators of `rng_state()`. Device generators restore only on the same device
    type: a CUDA state cannot be loaded into MPS (or the CPU), and is skipped. When this host has
    MPS but the checkpoint carries no MPS state (it was written on CUDA), the MPS generator is
    seeded with `derived_seed(state)` so a resume of the same checkpoint is still reproducible.
    Returns (and logs) a note for every generator that was not restored exactly."""
    notes: list[str] = []
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda = state.get("cuda") or []
    if cuda and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda)
    elif cuda:
        notes.append(f"rng: the checkpoint's CUDA generator state ({len(cuda)} device(s)) cannot be restored "
                     "without CUDA; device-side randomness (SpecAugment masks) differs from the CUDA run")
    if _mps_available():
        if state.get("mps") is not None:
            torch.mps.set_rng_state(state["mps"])
        else:
            seed = derived_seed(state)
            torch.mps.manual_seed(seed)
            notes.append(f"rng: no MPS generator state in the checkpoint; MPS generator seeded with {seed} "
                         "(derived from the checkpoint's CPU generator state, so this resume is reproducible)")
    elif state.get("mps") is not None:
        notes.append("rng: the checkpoint's MPS generator state cannot be restored without MPS")
    if log is not None:
        for note in notes:
            log(note)
    return notes


def atomic_torch_save(obj: object, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        torch.save(obj, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _step_of(path: Path) -> int:
    return int(path.stem.split("-")[1])


class CheckpointManager:
    def __init__(
        self,
        run_dir: str | Path,
        copy_dir: str | Path | None = None,
        keep: int = 2,
        every_steps: int = 500,
        every_seconds: float = 600.0,
        log: Callable[[str], None] = print,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.sync_dir = Path(copy_dir) if copy_dir else None
        self._sha: dict[str, tuple[int, float, str]] = {}  # name -> (size, mtime, sha256)
        self.artifacts: list[str] = []
        self.best_step: int | None = None
        self.status = "RUNNING"
        previous = self.run_dir / MANIFEST
        if previous.exists():
            try:
                data = json.loads(previous.read_text())
                self.artifacts = [a["file"] for a in data.get("files", []) if a.get("kind") == "artifact"]
                self.best_step = (data.get("best") or {}).get("step")
            except (OSError, ValueError):
                pass
        self.keep, self.every_steps, self.every_seconds = keep, every_steps, every_seconds
        self.log = log
        self.last_save = time.monotonic()
        self.best_metric: float | None = None

    def checkpoints(self) -> list[Path]:
        return sorted(self.run_dir.glob(CKPT_GLOB), key=_step_of, reverse=True)

    def due(self, step: int) -> bool:
        return step % self.every_steps == 0 or time.monotonic() - self.last_save >= self.every_seconds

    def save(self, state: dict, step: int, metric: float | None = None) -> Path:
        path = self.run_dir / f"ckpt-{step:08d}.pt"
        state = {**state, "step": step, "best_metric": self.best_metric, "saved_at": time.time()}
        atomic_torch_save(state, path)
        for old in self.checkpoints()[self.keep :]:
            old.unlink(missing_ok=True)
        is_best = metric is not None and (self.best_metric is None or metric < self.best_metric)
        if is_best:
            self.best_metric = metric
            self.best_step = step
            state["best_metric"] = metric
            atomic_torch_save(state, self.run_dir / "best.pt")
        self.last_save = time.monotonic()
        self.write_manifest(latest=path, step=step, latest_is_best=is_best)
        self.sync()
        return path

    # ------------------------------------------------------------ manifest

    def _entry(self, path: Path, kind: str, **extra) -> dict:
        stat = path.stat()
        cached = self._sha.get(path.name)
        if cached is None or cached[:2] != (stat.st_size, stat.st_mtime):
            cached = (stat.st_size, stat.st_mtime, sha256_file(path))
            self._sha[path.name] = cached
        return {"file": path.name, "kind": kind, "size": stat.st_size, "sha256": cached[2], **extra}

    def write_manifest(self, latest: Path | None = None, step: int | None = None, latest_is_best: bool = False) -> dict:
        """Atomically rewrite LATEST.json from the files that are complete right now."""
        checkpoints = self.checkpoints()
        latest = latest or (checkpoints[0] if checkpoints else None)
        files = [self._entry(p, "checkpoint", step=_step_of(p)) for p in checkpoints]
        best = self.run_dir / "best.pt"
        best_entry = None
        if best.exists():
            best_entry = self._entry(best, "best", step=self.best_step)
            files.append(best_entry)
        for name in self.artifacts:
            if (self.run_dir / name).exists():
                files.append(self._entry(self.run_dir / name, "artifact"))
        manifest = {
            "run": self.run_dir.name,
            "run_dir": str(self.run_dir.resolve()),
            "status": self.status,
            "step": step if step is not None else (_step_of(latest) if latest else 0),
            "checkpoint": (
                {**next(f for f in files if f["file"] == latest.name), "best": latest_is_best}
                if latest is not None else None
            ),
            "best": best_entry,
            "keep": [p.name for p in checkpoints],
            "files": files,
            "updated_at": time.time(),
        }
        atomic_write_text(self.run_dir / MANIFEST, json.dumps(manifest, indent=2))
        return manifest

    def publish(self, *names: str) -> None:
        """List final artifacts (adapters, heads, results) in the manifest with their sha256."""
        for name in names:
            if name not in self.artifacts:
                self.artifacts.append(name)
        self.write_manifest()
        self.sync()

    def mark(self, status: str, detail: str = "") -> None:
        """End of run: DONE or FAILED marker file plus the manifest status."""
        if status not in ("DONE", "FAILED"):
            raise ValueError(status)
        self.status = status
        atomic_write_text(self.run_dir / status, detail or status)
        self.write_manifest()
        self.sync()

    def sync(self) -> None:
        """Mirror checkpoints and small run files to the persistent location."""
        if self.sync_dir is None:
            return
        try:
            target = self.sync_dir
            target.mkdir(parents=True, exist_ok=True)
            names = [p for pattern in (CKPT_GLOB, "best.pt", *SYNC_FILES) for p in self.run_dir.glob(pattern)]
            for path in names:
                dst = target / path.name
                if not dst.exists() or dst.stat().st_size != path.stat().st_size or dst.stat().st_mtime < path.stat().st_mtime:
                    _copy_atomic(path, dst)
            local = {p.name for p in self.run_dir.glob(CKPT_GLOB)}
            for stale in target.glob(CKPT_GLOB):
                if stale.name not in local:
                    stale.unlink(missing_ok=True)
        except OSError as error:  # a flaky Drive mount must not kill training
            self.log(f"checkpoint sync to {self.sync_dir} failed: {error}")

    def _pull_from_sync(self) -> None:
        if self.sync_dir is None or self.checkpoints() or not self.sync_dir.exists():
            return
        pulled = [p for p in self.sync_dir.glob(CKPT_GLOB)] + list(self.sync_dir.glob("best.pt"))
        pulled += [p for pattern in SYNC_FILES for p in self.sync_dir.glob(pattern)]
        for path in pulled:
            _copy_atomic(path, self.run_dir / path.name)
        if pulled:
            self.log(f"restored {len(pulled)} files from {self.sync_dir}")

    def restore(self, expected_hash: str, force_new_run: bool = False, step: int | None = None) -> dict | None:
        """The newest loadable checkpoint (or the one at `step`, when given)."""
        self._pull_from_sync()
        for path in self.checkpoints():
            if step is not None and _step_of(path) != step:
                continue
            try:
                state = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as error:  # noqa: BLE001 - truncated/corrupt: try the previous one
                self.log(f"checkpoint {path.name} unreadable ({type(error).__name__}); trying the previous one")
                continue
            if state.get("config_hash") != expected_hash:
                if not force_new_run:
                    raise SystemExit(
                        f"{path} was written with config hash {state.get('config_hash')}, the current "
                        f"config hashes to {expected_hash}. Rerun with the original settings, or pass "
                        "--force-new-run to start over (old checkpoints are moved aside)."
                    )
                self._move_aside()
                return None
            self.best_metric = state.get("best_metric")
            self.log(f"resuming from {path.name} (step {state['step']})")
            return state
        return None

    def _move_aside(self) -> None:
        aside = self.run_dir / f"abandoned-{int(time.time())}"
        aside.mkdir()
        for path in [*self.run_dir.glob(CKPT_GLOB), *self.run_dir.glob("best.pt"), *self.run_dir.glob("*.csv")]:
            path.rename(aside / path.name)
        self.log(f"--force-new-run: previous checkpoints moved to {aside}")


class StopFlag:
    """SIGTERM/SIGINT -> `self.stop = True`; the loop saves a final checkpoint and exits."""

    def __init__(self, log: Callable[[str], None] = print):
        self.stop = False
        self.log = log
        self._previous = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._previous[sig] = signal.signal(sig, self._handle)

    def _handle(self, signum, _frame) -> None:
        self.log(f"received signal {signum}: checkpointing after this step")
        self.stop = True

    def restore(self) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
