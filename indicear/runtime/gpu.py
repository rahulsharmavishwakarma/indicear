"""GPU utilization sampling, device memory accounting and runtime description."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Self


def gil_enabled() -> bool:
    return getattr(sys, "_is_gil_enabled", lambda: True)()


def runtime_banner() -> str:
    return f"python {sys.version.split()[0]}, GIL {'on' if gil_enabled() else 'off'}"


_IOREG_UTIL = re.compile(r'"Device Utilization %"=(\d+)')
_IOREG_MEMORY = re.compile(r'"In use system memory"=(\d+)')


def nvidia_sample() -> tuple[int, int]:
    """(utilization %, memory used MiB) of the first CUDA GPU, from nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    first = result.stdout.strip().splitlines()[0]
    util, memory = (int(v) for v in first.split(", ")[:2])
    return util, memory


def apple_gpu_sample() -> tuple[int, int]:
    """(utilization %, GPU memory in use MiB) of the Apple GPU, from `ioreg` (IOAccelerator
    PerformanceStatistics: "Device Utilization %" and "In use system memory"). No sudo, no torch:
    safe in a background thread and from outside the training process."""
    listing = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                             capture_output=True, text=True, timeout=10, check=False).stdout
    util, memory = _IOREG_UTIL.search(listing), _IOREG_MEMORY.search(listing)
    if util is None or memory is None:
        raise ValueError("no IOAccelerator performance statistics")
    return int(util.group(1)), int(memory.group(1)) // 2**20


def gpu_sampler_backend() -> str | None:
    """"nvidia" (nvidia-smi on PATH), "apple" (macOS ioreg), or None."""
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if sys.platform == "darwin" and shutil.which("ioreg"):
        return "apple"
    return None


class GpuSampler:
    """Samples GPU utilization and memory to a CSV every `period` seconds (default 5 s; as in
    ../eg2-probe/run_langs.py): nvidia-smi on a CUDA VM, `ioreg` on Apple silicon (MPS), where
    memory is the GPU's in-use share of unified memory. Use as a context manager; `summary()`
    reports the result."""

    def __init__(self, path: str | Path, period: float = 5.0):
        self.path, self.period = Path(path), period
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.samples: list[tuple[int, int]] = []

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as out:
            out.write("time,util_percent,memory_mib\n")
            backend = gpu_sampler_backend()
            sample = {"nvidia": nvidia_sample, "apple": apple_gpu_sample}.get(backend)
            while not self._stop.is_set():
                try:
                    if sample is None:
                        raise OSError("no GPU sampler on this host")
                    util, memory = sample()
                    self.samples.append((util, memory))
                    out.write(f"{int(time.time())},{util},{memory}\n")
                    out.flush()
                except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                    pass  # no GPU, nvidia-smi or ioreg: keep running without samples
                self._stop.wait(self.period)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=15)

    def summary(self) -> str:
        if not self.samples:
            return "GPU utilization: no samples (nvidia-smi or ioreg)"
        utils = [u for u, _ in self.samples]
        busy = 100 * sum(u >= 80 for u in utils) / len(utils)
        peak = max(m for _, m in self.samples)
        return (
            f"GPU utilization: mean {sum(utils) / len(utils):.0f}%, at or above 80% for "
            f"{busy:.0f}% of {len(utils)} samples, peak memory {peak} MiB"
        )


# ---------------------------------------------------------------- device memory


# Peak tensor memory per padded encoder frame of a live v4+lid micro-batch (layers 1-12, heads,
# losses, backward) on Apple MPS, measured on an M5 Pro (64 GB, torch 2.14.1) with the real tower
# and the pilot's step-3788 weights on dev audio: 33-34 GiB at 16k padded frames in float32 eager,
# 20-28 GiB at 16k-22k frames under half-precision autocast eager, 12.9 GiB at 16.8k frames under
# fp16 autocast with torch.compile (docs/pilot.md). The Metal driver holds about 10-13 GiB more
# (the allocator's cached pool). Throughput was flat from 8k to 22k frames per micro-batch.
MPS_BYTES_PER_FRAME = {"fp32": 2.15 * 2**20, "half": 1.3 * 2**20, "half_compiled": 0.85 * 2**20}
MPS_MEMORY_SHARE = 0.5  # of the MPS memory limit; the rest: allocator pool, driver, OS, stream, CPU work
MPS_MEM_ENV = "INDICEAR_MPS_MEM_GB"  # the GiB cap the run was given (scripts/resume_local.sh: MPS_MEM_GB)


def mps_memory_limit() -> int:
    """Bytes this process may allocate on MPS: torch.mps.recommended_max_memory(), lowered to
    $INDICEAR_MPS_MEM_GB when set (the launcher also sets PYTORCH_MPS_HIGH_WATERMARK_RATIO to the same
    cap, so the allocator raises "MPS backend out of memory" there and OomBackoff adds micro-batches)."""
    import os

    import torch

    limit = torch.mps.recommended_max_memory()
    cap = os.environ.get(MPS_MEM_ENV)
    return min(limit, int(float(cap) * 2**30)) if cap else limit


def empty_device_cache() -> None:
    """Release cached allocator blocks on CUDA and MPS (after an out-of-memory error)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def auto_splits(device_type: str, frame_budget: int, precision: str = "fp32", compiled: bool = False,
                share: float = MPS_MEMORY_SHARE) -> int:
    """Micro-batches per step so a `frame_budget` step fits this device: 1 except on MPS, where
    ceil(budget * bytes per frame / (share * mps_memory_limit())). For the pilot's 67,045-frame
    budget on the 64 GB M5 Pro with a 42 GiB cap: 3 with fp16 + torch.compile (about 17 GiB of
    tensors per micro-batch), 4 with half precision eager, 7 in float32 eager."""
    if device_type != "mps":
        return 1
    if precision in ("fp16", "bf16"):
        per_frame = MPS_BYTES_PER_FRAME["half_compiled" if compiled else "half"]
    else:
        per_frame = MPS_BYTES_PER_FRAME["fp32"]
    return max(1, math.ceil(frame_budget * per_frame / (share * mps_memory_limit())))


# ---------------------------------------------------------------- host memory (unified memory on macOS)


def _sysctl(name: str) -> str:
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=False).stdout.strip()


def host_memory() -> dict:
    """macOS: swap used (GiB), free + inactive memory (GiB) and the kernel's memory pressure level
    ("normal", "warning", "critical"); {} elsewhere."""
    if sys.platform != "darwin":
        return {}
    out: dict = {}
    swap = re.search(r"used = ([0-9.]+)M", _sysctl("vm.swapusage"))
    if swap:
        out["swap_gib"] = float(swap.group(1)) / 1024
    level = _sysctl("kern.memorystatus_vm_pressure_level")
    out["pressure"] = {"1": "normal", "2": "warning", "4": "critical"}.get(level, level or "unknown")
    stats = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5, check=False).stdout
    page = re.search(r"page size of (\d+) bytes", stats)
    pages = {k: int(v) for k, v in re.findall(r"Pages (free|inactive|speculative):\s+(\d+)", stats)}
    if page and pages:
        out["free_gib"] = sum(pages.values()) * int(page.group(1)) / 2**30
    return out


class HostMemoryGuard:
    """Keeps a Mac usable while it trains: every `every` s it reads swap and the memory pressure
    level, logs them every `log_every` s, and `check()` returns True (at most once per `cooldown`
    s) when swap exceeds `swap_limit_gib` or pressure is critical; the trainer then adds a
    micro-batch split rather than let the machine thrash. A no-op off macOS."""

    def __init__(self, swap_limit_gib: float = 8.0, every: float = 60.0, log_every: float = 300.0,
                 cooldown: float = 600.0, log=print, probe=host_memory, clock=time.monotonic):
        self.swap_limit, self.every, self.log_every, self.cooldown = swap_limit_gib, every, log_every, cooldown
        self.log, self.probe, self.clock = log, probe, clock
        self.enabled = sys.platform == "darwin" or probe is not host_memory
        self.last_check = self.last_log = self.last_action = -float("inf")

    def check(self) -> bool:
        now = self.clock()
        if not self.enabled or now - self.last_check < self.every:
            return False
        self.last_check = now
        try:
            state = self.probe()
        except (OSError, subprocess.SubprocessError, ValueError):
            return False
        swap, pressure = state.get("swap_gib", 0.0), state.get("pressure", "unknown")
        if now - self.last_log >= self.log_every:
            self.last_log = now
            self.log(f"host memory: free {state.get('free_gib', float('nan')):.1f} GiB, swap {swap:.1f} GiB, "
                     f"pressure {pressure}")
        if (swap > self.swap_limit or pressure == "critical") and now - self.last_action >= self.cooldown:
            self.last_action = now
            return True
        return False


class DeviceMemory:
    """Peak tensor memory of a training device. CUDA: the allocator's own peak statistics. MPS has
    none, so `note()` samples torch.mps.current_allocated_memory() (the trainer calls it after
    each forward and backward) and the peak is the largest sample; `driver_bytes()` is the Metal
    driver's total (tensors plus the cached pool). CPU: 0."""

    def __init__(self, device):
        import torch

        self.type = torch.device(device).type
        self.peak = 0

    def note(self) -> None:
        if self.type == "mps":
            import torch

            self.peak = max(self.peak, torch.mps.current_allocated_memory())

    def reset_peak(self) -> None:
        import torch

        self.peak = 0
        if self.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def peak_bytes(self) -> int:
        import torch

        if self.type == "cuda":
            return torch.cuda.max_memory_allocated()
        return self.peak

    def driver_bytes(self) -> int:
        import torch

        if self.type == "mps":
            return torch.mps.driver_allocated_memory()
        if self.type == "cuda":
            return torch.cuda.memory_reserved()
        return 0
