"""On-the-fly augmentation on the GPU, on the padded waveform batch, before the GPU log-mel.

The CPU Corruptor path (indicear.augment) is CPU-bound on a 12-vCPU A100 VM (measured ~460 h of
audio per hour with 12-14 workers while the GPU idled), so the live pilot does everything except
codecs on the GPU in the main process:

    stream workers (CPU): decode -> codec on ~15% of clips (CodecAugment, ffmpeg -threads 1)
    main process (GPU):   GpuAugment(batch) -> GpuLogMel -> layers 1-6 -> ...

`GpuAugment` applies ONE kind per augmented row (rows the planner marked with view > 0), as the
CPU `AugmentPolicy` does:

| kind | operation | parameters |
| --- | --- | --- |
| noise | add a noise snippet at an SNR | U(5, 20) dB; white, pink or brown noise (or MUSAN files) from a bank |
| babble | add 3-6 other utterances of the same batch at an SNR | U(5, 20) dB; each talker power-normalized, random circular offset |
| reverb | FFT convolution with an RIR, power kept | synthetic RIR, RT60 U(0.3, 0.9) s, from a bank; with `rir_files`, a measured RIR with probability `real_rir_share` |
| reverb_pointnoise | reverb, then a reverberant noise snippet at an SNR | RIR as above; noise RIR synthetic, RT60 <= 0.8 s; U(5, 20) dB |
| speed | windowed-sinc resampling (Kaldi-style speed perturbation) | 0.9 or 1.1; length changes |
| tel8k | 300-3400 Hz band-limit | |

Banks: RIRs (`augment.synthetic_rir`) and coloured noise (`augment.colored_noise`) are generated
once at start-up from a fixed seed with the CPU generators, then kept in device memory: the
same signal models as the CPU Corruptor, identical on every device. Babble comes from the batch's
own clean waveforms (other rows with speech), so no extra audio is held.

Measured RIRs (`rir_files=`, e.g. the real RIRs of OpenSLR SLR28 that scripts/rirs.sh fetches;
the function `rir_files(directory)` lists them): `load_rirs` takes the first channel, resamples
to 16 kHz, moves the direct-path peak to 1 ms from the start, truncates to 1.2 s (the longest
synthetic RIR) and scales to unit energy. They are appended to the bank after the synthetic RIRs,
each labelled with its Schroeder RT60 (measured before truncation). A reverb row then takes a
measured RIR with probability `real_rir_share` (one extra numpy draw), else a synthetic one; the
point-noise RIR stays synthetic. Without `rir_files` the bank, every draw and `describe()` are
exactly those of the synthetic-only policy, so existing runs keep their config hash and resume
determinism.

Determinism: every choice for a row (kind, SNR, RT60 entry, noise entry and offset, babble
talkers and offsets, speed factor) comes from numpy `default_rng((seed, crc32(key), view))`, the
same seed the CPU path uses, and the banks are fixed; no torch RNG is consumed. A resumed run
replays the same batches (the stream state) and so the same corruptions. Augmentation runs on the
whole batch before any OOM split, so micro-batching never changes it.

Apple MPS: MPS has no float64, which the power, FFT-convolution and band-limit arithmetic use, so
on MPS `GpuAugment` runs on the CPU in the main process (`augment_device`), followed by the log-mel
on the CPU (indicear.encoder.feature_device); only the encoder runs on MPS. The plans (all the
randomness) are the same numpy draws on every device, and the arithmetic is the same torch code,
so the corruption of a row is the same up to floating-point rounding of the device's FFTs, and
deterministic on the CPU: a resumed run reproduces it exactly.

Differences from the CPU path, by design: the codec (when drawn) is applied in the worker
before the GPU corruptions, so added noise and reverb come after the codec; tel8k is a band-limit
on the 16 kHz signal (the CPU version also resamples through 8 kHz with soxr); FFT sizes differ, so
outputs agree statistically (SNR, RIR decay), not sample for sample (tests/test_gpu_augment.py).
"""

from __future__ import annotations

import hashlib
import math
import time
import zlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from indicear.augment import SR, colored_noise, synthetic_rir

GPU_KINDS = ("noise", "babble", "reverb", "reverb_pointnoise", "speed", "tel8k")
GPU_POLICY = {"noise": 0.25, "babble": 0.2, "reverb": 0.175, "reverb_pointnoise": 0.175, "speed": 0.15, "tel8k": 0.05}
COLOURS = ("white", "pink", "brown")
RIR_MAX_SECONDS = 1.2  # the longest synthetic RIR (augment.synthetic_rir caps it at 1.2 s)
RIR_PRE_PEAK = 16  # samples kept before the direct-path peak (1 ms at 16 kHz)
SLR28_RIR_LIST = Path("RIRS_NOISES") / "real_rirs_isotropic_noises" / "rir_list"
AUDIO_SUFFIXES = (".wav", ".flac")


def schroeder_rt60(rir: np.ndarray, rate: int = SR) -> float:
    """RT60 from a line fit to the Schroeder backward-integrated energy decay: the -5..-25 dB range
    (T20 x 3), else -5..-15 dB (T10 x 6) for short or noisy tails; clipped to [0.01, 5] s."""
    energy = np.cumsum(np.asarray(rir, np.float64)[::-1] ** 2)[::-1]
    if energy[0] <= 0:
        return 0.01
    decay = 10 * np.log10(energy / energy[0] + 1e-30)
    for low in (-25.0, -15.0):
        index = np.flatnonzero((decay <= -5.0) & (decay >= low))
        if len(index) >= 8:
            slope = np.polyfit(index / rate, decay[index], 1)[0]
            if slope < 0:
                return float(np.clip(-60.0 / slope, 0.01, 5.0))
    return 0.01


def rir_files(directory: str | Path, sets: Sequence[str] | None = None) -> list[str]:
    """Measured RIR files under `directory`, sorted by relative path (deterministic).

    SLR28 layout (RIRS_NOISES/real_rirs_isotropic_noises/rir_list under `directory`, or `directory`
    itself being that folder): only the files rir_list names, so the isotropic noise recordings
    beside them are left out; `sets` keeps names starting with one of the given database prefixes
    ("air", "RVB2014", "RWCP"). Any other directory: every .wav/.flac below it."""
    directory = Path(directory)
    listing = next((p for p in (directory / SLR28_RIR_LIST, directory / "rir_list") if p.is_file()), None)
    if listing is not None:
        base = directory if listing == directory / SLR28_RIR_LIST else directory.parent.parent
        files = []
        for line in listing.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            relative = Path(line.split()[-1])
            if sets and not any(relative.name.startswith(f"{prefix}_") for prefix in sets):
                continue
            path = base / relative
            if path.is_file():
                files.append(path)
        return [str(p) for p in sorted(set(files), key=lambda p: p.relative_to(base).as_posix())]
    found = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES]
    if sets:
        found = [p for p in found if any(p.name.startswith(f"{prefix}_") for prefix in sets)]
    return [str(p) for p in sorted(found, key=lambda p: p.relative_to(directory).as_posix())]


def load_rirs(paths: Sequence[str], max_seconds: float = RIR_MAX_SECONDS,
              skipped: Counter | None = None, rt60s: list[float] | None = None) -> tuple[list[np.ndarray], list[str]]:
    """Measured RIRs -> (rirs, kept paths): first channel, 16 kHz (soxr), the direct-path peak
    moved to RIR_PRE_PEAK samples from the start, truncated to `max_seconds`, unit energy (as
    `augment.synthetic_rir`). Unreadable, empty or near-silent files are left out and counted in
    `skipped` by reason; `rt60s`, when given, receives each kept RIR's Schroeder RT60, measured on
    the aligned RIR before truncation (a 1.2 s cut would bias long rooms)."""
    import soundfile as sf
    import soxr

    skipped = skipped if skipped is not None else Counter()
    rirs, kept = [], []
    for path in paths:
        try:
            audio, rate = sf.read(str(path), dtype="float64", always_2d=True)
        except Exception:  # noqa: BLE001 - one bad file must not stop the bank
            skipped["unreadable"] += 1
            continue
        if audio.size == 0:
            skipped["empty"] += 1
            continue
        rir = audio[:, 0]
        if rate != SR:
            rir = soxr.resample(rir, rate, SR)
        rir = np.asarray(rir, np.float64)
        if not np.isfinite(rir).all() or np.abs(rir).max() < 1e-6:
            skipped["silent"] += 1
            continue
        start = max(0, int(np.argmax(np.abs(rir))) - RIR_PRE_PEAK)
        if rt60s is not None:
            rt60s.append(schroeder_rt60(rir[start:]))
        rir = rir[start: start + int(max_seconds * SR)]
        rirs.append((rir / np.sqrt(np.sum(rir**2))).astype(np.float32))
        kept.append(str(path))
    return rirs, kept


def augment_device(model_device: str | torch.device) -> torch.device:
    """Where `GpuAugment` runs for a model on `model_device`: that device, except Apple MPS (no
    float64), where it runs on the CPU in the main process."""
    model_device = torch.device(model_device)
    return torch.device("cpu") if model_device.type == "mps" else model_device


def parse_gpu_policy(text: str | None) -> dict[str, float]:
    """"noise=0.25,babble=0.2,..." -> weights over GPU_KINDS (left out: 0); empty -> GPU_POLICY."""
    if not text:
        return dict(GPU_POLICY)
    weights = {kind: 0.0 for kind in GPU_KINDS}
    for part in text.split(","):
        name, _, value = part.partition("=")
        if name.strip() not in weights:
            raise ValueError(f"unknown GPU augmentation kind {name!r}; use {GPU_KINDS}")
        weights[name.strip()] = float(value)
    if sum(weights.values()) <= 0 or min(weights.values()) < 0:
        raise ValueError("augmentation weights must be >= 0 and not all 0")
    return weights


def _power(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean square over each row's valid samples (the CPU path's _power, + 1e-12)."""
    valid = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
    return (x.double().square() * valid).sum(-1) / lengths.clamp(min=1).double() + 1e-12


def _mask(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return x * (torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None])


def mix_at_snr(speech: torch.Tensor, noise: torch.Tensor, lengths: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    """speech + scale * noise with scale = sqrt(P_speech / (P_noise * 10^(snr/10))), powers over
    each row's valid samples (augment.mix_at_snr, batched)."""
    scale = torch.sqrt(_power(speech, lengths) / (_power(noise, lengths) * torch.pow(10.0, snr_db.double() / 10)))
    return speech + (scale[:, None] * noise.double()).to(speech.dtype)


def fft_convolve(x: torch.Tensor, lengths: torch.Tensor, rirs: torch.Tensor) -> torch.Tensor:
    """Each row convolved with its RIR, cropped to its length and rescaled to its input power
    (augment.convolve, batched)."""
    size = 1 << (x.shape[-1] + rirs.shape[-1] - 2).bit_length()
    out = torch.fft.irfft(torch.fft.rfft(x, size) * torch.fft.rfft(rirs, size), size)[:, : x.shape[-1]]
    out = _mask(out, lengths)
    gain = torch.sqrt(_power(x, lengths) / _power(out, lengths))
    return (out.double() * gain[:, None]).to(x.dtype)


def bandlimit(x: torch.Tensor, lengths: torch.Tensor, low: float = 300.0, high: float = 3400.0) -> torch.Tensor:
    # Linear-phase FIR band-pass by power-of-two FFT convolution. An exact-length FFT would make cuFFT
    # use Bluestein plans for awkward lengths, whose workspace can fail with CUFFT_INTERNAL_ERROR
    # when VRAM is nearly full.
    taps = 1023
    t = torch.arange(taps, device=x.device, dtype=torch.float64) - taps // 2
    fir = 2 * high / SR * torch.sinc(2 * high / SR * t) - 2 * low / SR * torch.sinc(2 * low / SR * t)
    fir = fir * torch.blackman_window(taps, periodic=False, dtype=torch.float64, device=x.device)
    size = 1 << (x.shape[-1] + taps - 2).bit_length()
    out = torch.fft.irfft(torch.fft.rfft(x.double(), size) * torch.fft.rfft(fir, size), size)
    return _mask(out[:, taps // 2: taps // 2 + x.shape[-1]].to(x.dtype), lengths)


def sinc_resample(x: torch.Tensor, lengths: torch.Tensor, factor: float, taps: int = 16,
                  chunk: int = 1 << 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Speed perturbation y(t) = x(factor * t): windowed-sinc interpolation (Hann, `taps` each side),
    low-passed at min(1, 1/factor) of the input Nyquist. Returns (y [R, M], new lengths), with
    new length round(n / factor) as `augment.speed` gives."""
    device = x.device
    new = torch.round(lengths.double() / factor).long()
    total = int(new.max()) if new.numel() else 0
    cutoff = min(1.0, 1.0 / factor)
    out = torch.zeros(x.shape[0], total, dtype=x.dtype, device=device)
    offsets = torch.arange(-taps + 1, taps + 1, device=device)
    for start in range(0, total, chunk):
        t = torch.arange(start, min(total, start + chunk), device=device, dtype=torch.float64)
        position = t * factor
        base = torch.floor(position).long()
        index = base[:, None] + offsets[None, :]  # [T, 2K]
        distance = (position[:, None] - index.double())
        kernel = cutoff * torch.sinc(cutoff * distance) * 0.5 * (1 + torch.cos(math.pi * distance / taps))
        kernel = (kernel * (distance.abs() < taps)).to(x.dtype)
        valid = (index[None] >= 0) & (index[None] < lengths[:, None, None])  # [R, T, 2K]
        gathered = x[:, index.clamp(0, x.shape[-1] - 1)]  # [R, T, 2K]
        out[:, start:start + t.numel()] = (gathered * kernel[None] * valid).sum(-1)
    return _mask(out, new), new


class GpuAugment:
    """See the module docstring. `__call__(waveforms [B, N], lengths [B] (CPU), keys, views,
    speech)` -> (waveforms [B, N'] on `device`, lengths [B] CPU int64, conditions per row or None)."""

    def __init__(self, device: str | torch.device = "cpu", weights: dict[str, float] | None = None, seed: int = 0,
                 snr_range: tuple[float, float] = (5.0, 20.0), rt60_range: tuple[float, float] = (0.3, 0.9),
                 speeds: Sequence[float] = (0.9, 1.1), rirs: int = 128, noise_clips: int = 8,
                 noise_seconds: float = 20.0, noise_files: Sequence[str] = (), bank_seed: int = 20261024,
                 rir_files: Sequence[str] = (), real_rir_share: float = 0.5):
        self.device = torch.device(device)
        self.weights = dict(weights or GPU_POLICY)
        self.kinds = [k for k in GPU_KINDS if self.weights.get(k, 0.0) > 0]
        total = sum(self.weights[k] for k in self.kinds)
        self.probs = np.array([self.weights[k] / total for k in self.kinds])
        self.seed, self.snr_range, self.rt60_range, self.speeds = seed, snr_range, rt60_range, tuple(speeds)
        rng = np.random.default_rng(bank_seed)
        self.rt60 = np.sort(rng.uniform(*rt60_range, size=rirs))
        bank = [synthetic_rir(float(rt), rng) for rt in self.rt60]
        self.synthetic_rirs = rirs
        self.point_rirs = np.flatnonzero(self.rt60 <= 0.8)  # synthetic only, also with measured RIRs
        # Measured RIRs come after the synthetic ones; the synthetic bank (and so `rng`) is unchanged.
        self.rir_skipped: Counter = Counter()
        real_rt60: list[float] = []
        real, self.real_rir_files = (load_rirs(rir_files, skipped=self.rir_skipped, rt60s=real_rt60)
                                     if rir_files else ([], []))
        if not 0.0 <= real_rir_share <= 1.0:
            raise ValueError("real_rir_share must lie in [0, 1]")
        self.real_rir_share = float(real_rir_share) if real else 0.0
        if real:
            self.rt60 = np.concatenate([self.rt60, real_rt60])
            digest = hashlib.sha256()
            for path in self.real_rir_files:
                digest.update(Path(path).name.encode() + b"\0" + Path(path).read_bytes())
            self.real_rir_sha256 = digest.hexdigest()
            bank += real
        width = max(len(r) for r in bank)
        self.rirs = torch.zeros(len(bank), width)
        for i, rir in enumerate(bank):
            self.rirs[i, : len(rir)] = torch.from_numpy(rir)
        self.rirs = self.rirs.to(self.device)
        length = int(noise_seconds * SR)
        clips = []
        for colour in COLOURS:
            clips += [colored_noise(length, rng, colour) for _ in range(noise_clips)]
        self.noise_kinds = [f"synthetic-{c}" for c in COLOURS for _ in range(noise_clips)]
        for path in list(noise_files)[:64]:
            from indicear.encoder import load_audio

            audio = load_audio(path)[:length]
            if len(audio) >= SR // 2:
                clips.append(np.pad(audio, (0, length - len(audio)), mode="wrap") if len(audio) < length else audio)
                self.noise_kinds.append("musan")
        self.noise = torch.from_numpy(np.stack(clips).astype(np.float32)).to(self.device)
        self.noise_from = len(COLOURS) * noise_clips if len(clips) > len(COLOURS) * noise_clips else 0

    def describe(self) -> dict:
        # "device": "gpu" names the main-process path (vs the stream workers' CPU recipe), not where
        # the arithmetic runs (the CPU on MPS hosts): it is part of the run's hashed config.
        out = {"device": "gpu", "weights": {k: self.weights.get(k, 0.0) for k in GPU_KINDS},
               "snr_db": list(self.snr_range), "rt60_s": list(self.rt60_range), "speeds": list(self.speeds),
               "rirs": int(self.synthetic_rirs), "noise_clips": int(self.noise.shape[0]),
               "musan": self.noise_from > 0}
        if self.real_rir_files:  # only then: synthetic-only runs keep their config hash
            out.update(real_rirs=len(self.real_rir_files), real_rir_share=self.real_rir_share,
                       real_rir_sha256=self.real_rir_sha256)
        return out

    # ------------------------------------------------------------ planning (CPU, per row)

    def _noise_choice(self, rng: np.random.Generator) -> tuple[int, int]:
        low = self.noise_from  # MUSAN files when given, else the coloured noise
        clip = int(rng.integers(low, self.noise.shape[0]))
        return clip, int(rng.integers(self.noise.shape[1]))

    def plan(self, keys: Sequence[str], views: Sequence[int], speech: Sequence[bool]) -> list[dict | None]:
        talkers = [i for i, ok in enumerate(speech) if ok]
        plans: list[dict | None] = []
        for row, (key, view) in enumerate(zip(keys, views, strict=True)):
            if not view or not speech[row]:
                plans.append(None)
                continue
            rng = np.random.default_rng((self.seed, zlib.crc32(key.encode()), int(view)))
            kind = self.kinds[int(rng.choice(len(self.kinds), p=self.probs))]
            snr = float(rng.uniform(*self.snr_range))
            others = [i for i in talkers if i != row]
            if kind == "babble" and len(others) < 2:
                kind = "noise"
            plan: dict = {"kind": kind, "snr": snr}
            if kind == "noise":
                plan["noise"] = self._noise_choice(rng)
                plan["condition"] = f"noise:{snr:.1f}"
            elif kind == "babble":
                count = int(rng.integers(3, 7))
                plan["talkers"] = [(others[int(rng.integers(len(others)))], int(rng.integers(1 << 30)))
                                   for _ in range(count)]
                plan["condition"] = f"babble:{snr:.1f}"
            elif kind in ("reverb", "reverb_pointnoise"):
                real = bool(self.real_rir_files) and bool(rng.random() < self.real_rir_share)
                if real:
                    plan["rir"] = self.synthetic_rirs + int(rng.integers(len(self.real_rir_files)))
                else:
                    plan["rir"] = int(rng.integers(self.synthetic_rirs))
                plan["condition"] = f"reverb{'-real' if real else ''}:{self.rt60[plan['rir']]:.2f}"
                if kind == "reverb_pointnoise":
                    plan["noise"] = self._noise_choice(rng)
                    plan["noise_rir"] = int(self.point_rirs[int(rng.integers(len(self.point_rirs)))])
                    plan["condition"] += f"+pointnoise:{snr:.1f}"
            elif kind == "speed":
                plan["factor"] = self.speeds[int(rng.integers(len(self.speeds)))]
                plan["condition"] = f"speed:{plan['factor']}"
            else:
                plan["condition"] = "tel8k"
            plans.append(plan)
        return plans

    # ------------------------------------------------------------ applying (device, batched)

    def _noise_rows(self, choices: Sequence[tuple[int, int]], width: int) -> torch.Tensor:
        clips = torch.tensor([c for c, _ in choices], device=self.device)
        starts = torch.tensor([o for _, o in choices], device=self.device)
        index = (starts[:, None] + torch.arange(width, device=self.device)[None, :]) % self.noise.shape[1]
        return self.noise[clips[:, None], index]

    @torch.no_grad()
    def __call__(self, waveforms: torch.Tensor, lengths: torch.Tensor, keys: Sequence[str], views: Sequence[int],
                 speech: Sequence[bool]) -> tuple[torch.Tensor, torch.Tensor, list[str | None]]:
        lengths_cpu = lengths.cpu().long()
        plans = self.plan(keys, views, speech)
        x = waveforms.to(self.device, torch.float32, non_blocking=True)
        if not any(plans):
            return x, lengths_cpu, [None] * len(plans)
        clean = x
        x = x.clone()
        dev_lengths = lengths_cpu.to(self.device)
        new_lengths = lengths_cpu.clone()
        by_kind: dict[str, list[int]] = {}
        for row, plan in enumerate(plans):
            if plan is not None:
                by_kind.setdefault(plan["kind"], []).append(row)
        # speed first: it changes lengths (and maybe the padded width)
        for factor in self.speeds:
            rows = [r for r in by_kind.get("speed", []) if plans[r]["factor"] == factor]
            if not rows:
                continue
            index = torch.tensor(rows, device=self.device)
            out, new = sinc_resample(clean[index], dev_lengths[index], factor)
            if out.shape[-1] > x.shape[-1]:
                x = torch.nn.functional.pad(x, (0, out.shape[-1] - x.shape[-1]))
            x[index] = 0
            x[index, : out.shape[-1]] = out
            new_lengths[rows] = new.cpu()
        for kind in ("reverb", "reverb_pointnoise"):
            rows = by_kind.get(kind, [])
            if not rows:
                continue
            index = torch.tensor(rows, device=self.device)
            rirs = self.rirs[torch.tensor([plans[r]["rir"] for r in rows], device=self.device)]
            x[index] = fft_convolve(x[index], dev_lengths[index], rirs)
            if kind == "reverb_pointnoise":
                noise = self._noise_rows([plans[r]["noise"] for r in rows], x.shape[-1])
                noise_rirs = self.rirs[torch.tensor([plans[r]["noise_rir"] for r in rows], device=self.device)]
                noise = fft_convolve(noise, dev_lengths[index], noise_rirs)
                snr = torch.tensor([plans[r]["snr"] for r in rows], device=self.device)
                x[index] = _mask(mix_at_snr(x[index], noise, dev_lengths[index], snr), dev_lengths[index])
        rows = by_kind.get("noise", [])
        if rows:
            index = torch.tensor(rows, device=self.device)
            noise = self._noise_rows([plans[r]["noise"] for r in rows], x.shape[-1])
            snr = torch.tensor([plans[r]["snr"] for r in rows], device=self.device)
            x[index] = _mask(mix_at_snr(x[index], noise, dev_lengths[index], snr), dev_lengths[index])
        rows = by_kind.get("babble", [])
        if rows:
            index = torch.tensor(rows, device=self.device)
            width = x.shape[-1]
            mix = torch.zeros(len(rows), width, device=self.device)
            positions = torch.arange(width, device=self.device)
            for slot in range(6):
                active = [(i, plans[r]["talkers"][slot]) for i, r in enumerate(rows) if slot < len(plans[r]["talkers"])]
                if not active:
                    continue
                at = torch.tensor([i for i, _ in active], device=self.device)
                talker = torch.tensor([t for _, (t, _) in active], device=self.device)
                offset = torch.tensor([o for _, (_, o) in active], device=self.device)
                length = dev_lengths[talker]
                source = (offset[:, None] + positions[None, :]) % length[:, None]
                clip = clean[talker[:, None], source]
                mix[at] += (clip / torch.sqrt(_power(clean[talker], length)).float()[:, None])
            snr = torch.tensor([plans[r]["snr"] for r in rows], device=self.device)
            x[index] = _mask(mix_at_snr(x[index], mix, dev_lengths[index], snr), dev_lengths[index])
        rows = by_kind.get("tel8k", [])
        if rows:
            index = torch.tensor(rows, device=self.device)
            x[index] = bandlimit(x[index], dev_lengths[index])
        conditions = [p["condition"] if p is not None else None for p in plans]
        return x, new_lengths, conditions


# ---------------------------------------------------------------- micro-benchmark


def benchmark(seconds_per_clip: float = 8.0, clips: int = 64, device: str | None = None) -> dict:
    """Augmentation throughput: the CPU Corruptor per core (one thread, per kind) and GpuAugment
    on `device` (default cuda if available, else cpu), in seconds of audio per second."""
    from indicear.augment import NoiseBank, apply_condition

    rng = np.random.default_rng(0)
    n = int(seconds_per_clip * SR)
    audios = [(0.1 * rng.standard_normal(n)).astype(np.float32) for _ in range(clips)]
    bank = NoiseBank(speech_files=())
    cpu = {}
    for condition in ("noise:10", "reverb:0.6", "reverb:0.6+pointnoise:10", "tel8k", "speed:1.1"):
        started = time.process_time()
        for i, audio in enumerate(audios[:16]):
            apply_condition(audio, condition, np.random.default_rng(i), bank)
        cpu[condition] = round(16 * seconds_per_clip / max(time.process_time() - started, 1e-9), 1)
    from indicear.augment import available_codecs, codec

    for name in sorted(available_codecs()):  # ffmpeg subprocesses (-threads 1): wall time, one at a time
        started = time.perf_counter()
        for audio in audios[:8]:
            codec(audio, name)
        cpu[name] = round(8 * seconds_per_clip / max(time.perf_counter() - started, 1e-9), 1)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    augment = GpuAugment(device)
    waveforms = torch.from_numpy(np.stack(audios))
    lengths = torch.full((clips,), n)
    out = {"cpu_seconds_of_audio_per_cpu_second": cpu, "device": device}
    for kind in GPU_KINDS:
        augment.kinds, augment.probs = [kind], np.array([1.0])
        keys = [f"bench/{i}" for i in range(clips)]
        augment(waveforms, lengths, keys, [1] * clips, [True] * clips)  # warm-up
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        repeats = 3
        for r in range(repeats):
            augment(waveforms, lengths, keys, [2 + r] * clips, [True] * clips)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        out.setdefault("gpu_seconds_of_audio_per_device_second", {})[kind] = round(
            repeats * clips * seconds_per_clip / max(elapsed, 1e-9), 1)
    return out


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(benchmark(device=sys.argv[1] if len(sys.argv) > 1 else None), indent=2))
