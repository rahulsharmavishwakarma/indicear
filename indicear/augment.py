"""Waveform corruptions for the robustness probe and the training augmentation recipe.

All functions take and return float32 mono 16 kHz numpy arrays and a numpy Generator.

- Noise at a target SNR: MUSAN-style noise files when `noise_dir` is given (e.g. MUSAN's
  `noise/` subset, OpenSLR 17, CC BY 4.0), else synthetic colored noise (recorded in reports).
- Babble: 3-6 other speech clips summed (FLEURS clips from other speakers/languages).
- Reverberation: a synthetic exponential-decay room impulse response (no pyroomacoustics: MIT,
  but compiled and not needed for a first probe), optionally with a reverberant point noise.
- Telephone: 300-3400 Hz band-limit through 8 kHz. Codecs AMR-NB, GSM 06.10 and Opus via the
  ffmpeg binary when its encoders exist (`available_codecs()`); otherwise reported as skipped.
- Speed: Kaldi-style speed perturbation by resampling (tempo and pitch both change).

Condition strings: clean, noise:<snr>, babble:<snr>, reverb:<rt60>, tel8k, amr, gsm, opus,
speed:<factor>. Combine with "+", e.g. "reverb:0.5+noise:10".

`AugmentPolicy` is the on-the-fly policy of the live (streamed) training path: an augmented row
draws ONE kind from weighted kinds and its parameters from the row's generator, which the stream
seeds from (seed, key, view) with view taken from the planner state, so a resumed run reproduces
every corruption. Babble there is built from the other utterances of the same batch.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

SR = 16_000
CODECS = {
    # name: (ffmpeg encoder, container, encode args)
    "amr": ("libopencore_amrnb", "amr", ["-ar", "8000", "-b:a", "12.2k"]),
    "gsm": ("libgsm", "gsm", ["-ar", "8000"]),
    "opus": ("libopus", "ogg", ["-b:a", "12k"]),
}



FFMPEG_ENV = {**os.environ, "OMP_NUM_THREADS": "1"}  # one OpenMP thread per ffmpeg process

def _power(x: np.ndarray) -> float:
    return float(np.mean(np.square(x, dtype=np.float64))) + 1e-12


def fit_length(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if len(noise) >= length:
        start = int(rng.integers(0, len(noise) - length + 1))
        return noise[start : start + length]
    reps = -(-length // max(1, len(noise)))
    return np.tile(noise, reps)[:length]


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = noise[: len(speech)]
    scale = np.sqrt(_power(speech) / (_power(noise) * 10 ** (snr_db / 10)))
    return (speech + scale * noise).astype(np.float32)


def colored_noise(length: int, rng: np.random.Generator, kind: str = "pink") -> np.ndarray:
    white = rng.standard_normal(length)
    if kind == "white":
        return white.astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length, 1 / SR)
    freqs[0] = freqs[1] if length > 1 else 1.0
    exponent = {"pink": 0.5, "brown": 1.0}[kind]
    noise = np.fft.irfft(spectrum / freqs**exponent, n=length)
    return (noise / (np.std(noise) + 1e-9)).astype(np.float32)


def synthetic_rir(rt60: float, rng: np.random.Generator, length_s: float | None = None) -> np.ndarray:
    """Direct path + exponentially decaying diffuse tail (60 dB down at rt60 seconds)."""
    length = int(SR * (length_s or min(1.5 * rt60, 1.2)))
    t = np.arange(length) / SR
    tail = rng.standard_normal(length) * np.exp(-6.9078 * t / rt60)
    pre_delay = int(0.003 * SR)
    tail[:pre_delay] = 0.0
    rir = tail * 0.3
    rir[0] = 1.0
    return (rir / np.sqrt(np.sum(rir**2))).astype(np.float32)


def convolve(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    n = len(audio) + len(rir) - 1
    size = 1 << (n - 1).bit_length()
    out = np.fft.irfft(np.fft.rfft(audio, size) * np.fft.rfft(rir, size), size)[: len(audio)]
    gain = np.sqrt(_power(audio) / _power(out))
    return (out * gain).astype(np.float32)


def bandlimit(audio: np.ndarray, low: float = 300.0, high: float = 3400.0) -> np.ndarray:
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1 / SR)
    spectrum[(freqs < low) | (freqs > high)] = 0
    return np.fft.irfft(spectrum, n=len(audio)).astype(np.float32)


def resample(audio: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    import soxr

    return soxr.resample(audio, rate_in, rate_out).astype(np.float32)


def telephone(audio: np.ndarray) -> np.ndarray:
    narrow = resample(bandlimit(audio), SR, 8000)
    return resample(narrow, 8000, SR)[: len(audio)]


def speed(audio: np.ndarray, factor: float) -> np.ndarray:
    """Play `factor` times faster (shorter) by resampling, as in Kaldi speed perturbation."""
    return resample(audio, round(SR * factor), SR)


@functools.cache
def available_codecs() -> frozenset[str]:
    if shutil.which("ffmpeg") is None:
        return frozenset()
    try:
        listing = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(name for name, (encoder, _, _) in CODECS.items() if f" {encoder} " in listing)


def codec(audio: np.ndarray, name: str) -> np.ndarray:
    """Encode and decode through ffmpeg (pipes, no temporary files). `-threads 1` on both sides:
    ffmpeg otherwise starts a thread per core for each call (about 27 per call measured on the
    12-vCPU VM), which oversubscribes the CPU when many workers run codecs at once."""
    encoder, container, args = CODECS[name]
    pcm = np.clip(audio, -1, 1).astype(np.float32).tobytes()
    encoded = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "1", "-f", "f32le", "-ar", str(SR), "-ac", "1",
         "-i", "pipe:0", "-c:a", encoder, *args, "-ac", "1", "-threads", "1", "-f", container, "pipe:1"],
        input=pcm, capture_output=True, check=True, env=FFMPEG_ENV,
    ).stdout
    decoded = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "1", "-f", container, "-i", "pipe:0",
         "-threads", "1", "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"],
        input=encoded, capture_output=True, check=True, env=FFMPEG_ENV,
    ).stdout
    out = np.frombuffer(decoded, dtype=np.float32)
    if len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)))
    return out[: len(audio)].copy()


class NoiseBank:
    """Lazily loaded noise and babble sources (picklable: only paths are stored)."""

    def __init__(self, noise_files: Sequence[str] = (), speech_files: Sequence[str] = ()):
        self.noise_files = list(noise_files)
        self.speech_files = list(speech_files)

    def noise(self, length: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        if self.noise_files:
            from indicear.encoder import load_audio

            path = self.noise_files[int(rng.integers(len(self.noise_files)))]
            return fit_length(load_audio(path), length, rng), "musan"
        kind = ("white", "pink", "brown")[int(rng.integers(3))]
        return colored_noise(length, rng, kind), f"synthetic-{kind}"

    def babble(self, length: int, rng: np.random.Generator, exclude: str = "") -> np.ndarray:
        from indicear.encoder import load_audio

        pool = [f for f in self.speech_files if f != exclude]
        if not pool:
            raise ValueError("babble needs speech files")
        talkers = int(rng.integers(3, 7))
        mix = np.zeros(length, dtype=np.float32)
        for _ in range(talkers):
            clip = load_audio(pool[int(rng.integers(len(pool)))])
            clip = fit_length(clip, length, rng)
            mix += clip / np.sqrt(_power(clip))
        return mix


def apply_condition(
    audio: np.ndarray, condition: str, rng: np.random.Generator, bank: NoiseBank, item: dict | None = None
) -> np.ndarray:
    for step in condition.split("+"):
        name, _, value = step.partition(":")
        if name == "clean":
            continue
        if name == "noise":
            noise, _ = bank.noise(len(audio), rng)
            audio = mix_at_snr(audio, noise, float(value))
        elif name == "babble":
            exclude = (item or {}).get("audio", "")
            audio = mix_at_snr(audio, bank.babble(len(audio), rng, exclude), float(value))
        elif name == "reverb":
            audio = convolve(audio, synthetic_rir(float(value), rng))
        elif name == "pointnoise":  # reverberant point noise source at an SNR
            noise, _ = bank.noise(len(audio), rng)
            noise = convolve(noise, synthetic_rir(float(rng.uniform(0.3, 0.8)), rng))
            audio = mix_at_snr(audio, noise, float(value))
        elif name == "tel8k":
            audio = telephone(audio)
        elif name in CODECS:
            if name not in available_codecs():
                raise RuntimeError(f"ffmpeg encoder for {name} is not available")
            audio = codec(audio, name)
        elif name == "speed":
            audio = speed(audio, float(value))
        else:
            raise ValueError(f"unknown condition {step!r}")
    return audio.astype(np.float32)


class Corruptor:
    """FeatureBatches transform applying one fixed condition (robustness probe)."""

    def __init__(self, condition: str, bank: NoiseBank):
        self.condition, self.bank = condition, bank

    def __call__(self, audio: np.ndarray, rng: np.random.Generator, item: dict) -> np.ndarray:
        return apply_condition(audio, self.condition, rng, self.bank, item)


class AugmentationRecipe:
    """Training views: item["view"] == 0 is clean; other views draw a random chain from the
    configurable recipe (speed, reverb + point noise, codec/telephone, babble, noise).
    Seeded by (key, view), so a rebuilt cache reproduces the same corruption."""

    def __init__(
        self,
        bank: NoiseBank,
        speeds: Sequence[float] = (0.9, 1.0, 1.1),
        p_reverb: float = 0.3,
        p_codec: float = 0.3,
        p_babble: float = 0.2,
        p_noise: float = 0.4,
        snr_range: tuple[float, float] = (5.0, 25.0),
        codecs: Sequence[str] = ("tel8k", "amr", "gsm", "opus"),
    ):
        self.bank = bank
        self.speeds = tuple(speeds)
        self.p_reverb, self.p_codec, self.p_babble, self.p_noise = p_reverb, p_codec, p_babble, p_noise
        self.snr_range = snr_range
        self.codecs = tuple(codecs)

    def chain(self, rng: np.random.Generator) -> str:
        steps = []
        factor = self.speeds[int(rng.integers(len(self.speeds)))]
        if factor != 1.0:
            steps.append(f"speed:{factor}")

        def snr() -> str:
            return f"{rng.uniform(*self.snr_range):.1f}"

        if rng.random() < self.p_reverb:
            steps.append(f"reverb:{rng.uniform(0.2, 0.9):.2f}")
            steps.append(f"pointnoise:{snr()}")
        if rng.random() < self.p_babble:
            steps.append(f"babble:{snr()}")
        elif rng.random() < self.p_noise:
            steps.append(f"noise:{snr()}")
        usable = [c for c in self.codecs if c == "tel8k" or c in available_codecs()]
        if usable and rng.random() < self.p_codec:
            steps.append(usable[int(rng.integers(len(usable)))])
        return "+".join(steps) or "clean"

    def __call__(self, audio: np.ndarray, rng: np.random.Generator, item: dict) -> np.ndarray:
        view = int(item.get("view", 0))
        if view == 0 or item.get("negative"):
            return audio
        local = np.random.default_rng((zlib.crc32(item["key"].encode()), view))
        return apply_condition(audio, self.chain(local), local, self.bank, item)


def make_negatives(
    out_dir: str | Path, count: int, bank: NoiseBank, seed: int = 0, seconds: tuple[float, float] = (1.0, 8.0)
) -> list[dict]:
    """Non-speech negatives with empty transcripts: noise, hum, silence-with-noise floor."""
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    items = []
    for index in range(count):
        length = int(SR * rng.uniform(*seconds))
        kind = index % 3
        if kind == 0:
            audio, source = bank.noise(length, rng)
        elif kind == 1:  # mains hum plus harmonics
            t = np.arange(length) / SR
            base = (50.0, 60.0)[int(rng.integers(2))]
            audio = sum(np.sin(2 * np.pi * base * h * t) / h for h in range(1, 6)).astype(np.float32)
            source = "hum"
        else:  # near-silence: low noise floor
            audio, source = colored_noise(length, rng, "pink") * 0.01, "floor"
        audio = (0.1 * audio / (np.max(np.abs(audio)) + 1e-9)).astype(np.float32)
        path = out_dir / f"neg-{index:05d}.flac"
        sf.write(path, audio, SR, format="FLAC", subtype="PCM_16")
        items.append({
            "key": f"negative/{index:05d}", "audio": str(path), "text": "", "lang": "none",
            "seconds": length / SR, "source": f"negative-{source}", "negative": True,
        })
    return items


# ---------------------------------------------------------------- on-the-fly policy (live training)

POLICY_KINDS = ("noise", "babble", "reverb", "reverb_pointnoise", "codec", "speed")
DEFAULT_POLICY = {"noise": 0.2, "babble": 0.15, "reverb": 0.15, "reverb_pointnoise": 0.15, "codec": 0.2, "speed": 0.15}
POLICY_CODECS = ("tel8k", "gsm", "opus")


def parse_policy(text: str | None) -> dict[str, float]:
    """"noise=0.2,babble=0.15,..." -> weights (kinds left out get 0); None or "" -> DEFAULT_POLICY."""
    if not text:
        return dict(DEFAULT_POLICY)
    weights = {kind: 0.0 for kind in POLICY_KINDS}
    for part in text.split(","):
        name, _, value = part.partition("=")
        if name.strip() not in weights:
            raise ValueError(f"unknown augmentation kind {name!r}; use {POLICY_KINDS}")
        weights[name.strip()] = float(value)
    if sum(weights.values()) <= 0 or min(weights.values()) < 0:
        raise ValueError("augmentation weights must be >= 0 and not all 0")
    return weights


def babble_from(clips: Sequence[np.ndarray], length: int, rng: np.random.Generator) -> np.ndarray:
    """3-6 talkers drawn (with replacement) from `clips`, each normalized, summed."""
    mix = np.zeros(length, dtype=np.float32)
    for _ in range(int(rng.integers(3, 7))):
        clip = clips[int(rng.integers(len(clips)))]
        clip = fit_length(clip, length, rng)
        mix += clip / np.sqrt(_power(clip))
    return mix


class AugmentPolicy:
    """Live-training augmentation: `StreamPlanner(augment_prob=1 - clean_prob)` marks a row as an
    augmented draw (view > 0); this transform then picks one kind by weight:

        noise              noise:U(snr)                 (MUSAN files if given, else colored noise)
        babble             babble:U(snr)                (3-6 other utterances of the same batch)
        reverb             reverb:U(rt60)
        reverb_pointnoise  reverb:U(rt60)+pointnoise:U(snr)
        codec              one of tel8k / gsm / opus, those ffmpeg can encode (tel8k always)
        speed              speed:0.9 or speed:1.1

    Every choice comes from the generator the stream passes in, so it is deterministic per
    (seed, key, view). Rows with an empty transcript (non-speech) are left clean. When a batch
    has fewer than two other utterances, babble falls back to noise. `last` records the condition
    of the most recent call per key (tests)."""

    uses_batch = True

    def __init__(self, bank: NoiseBank | None = None, weights: dict[str, float] | None = None,
                 snr_range: tuple[float, float] = (5.0, 20.0), rt60_range: tuple[float, float] = (0.3, 0.9),
                 speeds: Sequence[float] = (0.9, 1.1), codecs: Sequence[str] | None = None):
        self.bank = bank or NoiseBank()
        self.weights = dict(weights or DEFAULT_POLICY)
        self.kinds = [k for k in POLICY_KINDS if self.weights.get(k, 0.0) > 0]
        total = sum(self.weights[k] for k in self.kinds)
        self.probs = np.array([self.weights[k] / total for k in self.kinds])
        self.snr_range, self.rt60_range, self.speeds = snr_range, rt60_range, tuple(speeds)
        wanted = POLICY_CODECS if codecs is None else tuple(codecs)
        self.codecs = tuple(c for c in wanted if c == "tel8k" or c in available_codecs())

    def describe(self) -> dict:
        """The policy as data (hashed into the run config; available codecs are logged, not hashed)."""
        return {"weights": {k: self.weights.get(k, 0.0) for k in POLICY_KINDS}, "snr_db": list(self.snr_range),
                "rt60_s": list(self.rt60_range), "speeds": list(self.speeds), "codecs_wanted": list(POLICY_CODECS)}

    def condition(self, rng: np.random.Generator, babble_ok: bool = True) -> str:
        kind = self.kinds[int(rng.choice(len(self.kinds), p=self.probs))]
        snr = f"{rng.uniform(*self.snr_range):.1f}"
        rt60 = f"{rng.uniform(*self.rt60_range):.2f}"
        if kind == "noise" or (kind == "babble" and not babble_ok):
            return f"noise:{snr}"
        if kind == "babble":
            return f"babble:{snr}"
        if kind == "reverb":
            return f"reverb:{rt60}"
        if kind == "reverb_pointnoise":
            return f"reverb:{rt60}+pointnoise:{snr}"
        if kind == "codec":
            return self.codecs[int(rng.integers(len(self.codecs)))] if self.codecs else "tel8k"
        return f"speed:{self.speeds[int(rng.integers(len(self.speeds)))]}"

    def __call__(self, audio: np.ndarray, rng: np.random.Generator, item: dict,
                 batch: Sequence[np.ndarray] = (), index: int | None = None) -> np.ndarray:
        if not item.get("text"):
            return audio
        others = [clip for j, clip in enumerate(batch) if j != index and len(clip)]
        condition = self.condition(rng, babble_ok=len(others) >= 2)
        item["augment"] = condition
        if condition.startswith("babble:"):
            snr = float(condition.split(":", 1)[1])
            return mix_at_snr(audio, babble_from(others, len(audio), rng), snr)
        return apply_condition(audio, condition, rng, self.bank, item)


WORKER_CODECS = ("gsm", "opus", "amr")


class CodecAugment:
    """The CPU part of the GPU augmentation path (indicear.gpu_augment): in the stream workers, a
    fraction `prob` of speech clips (default 0.15) goes through one ffmpeg codec (gsm / opus / amr,
    those this ffmpeg can encode; none available -> no-op); everything else stays in the main
    process on the GPU. `all_rows`: the stream calls it for clean draws too (view 0), so the codec
    share is of all clips; `only_augmented=True` (the cache build) leaves view 0 clean. The draw
    uses the generator the caller passes (the stream: seeded from (seed, key, view, 1), separate
    from the GPU's (seed, key, view)), so it is deterministic under resume."""

    all_rows = True

    def __init__(self, prob: float = 0.15, codecs: Sequence[str] = WORKER_CODECS, only_augmented: bool = False):
        self.prob, self.only_augmented = prob, only_augmented
        self.codecs = tuple(c for c in codecs if c in available_codecs())

    def describe(self) -> dict:
        return {"codec_prob": self.prob, "codecs_wanted": list(WORKER_CODECS)}

    def __call__(self, audio: np.ndarray, rng: np.random.Generator, item: dict) -> np.ndarray:
        if not item.get("text") or not self.codecs or (self.only_augmented and not item.get("view")):
            return audio
        if rng.random() >= self.prob:
            return audio
        name = self.codecs[int(rng.integers(len(self.codecs)))]
        item["codec"] = name
        return codec(audio, name)
