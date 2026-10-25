"""GpuLogMel against the Hugging Face Gemma 4 extractor, and the "gpu" feature path end to end.

The extractor is built from the pinned EmbeddingGemma 2 preprocessor_config.json values (no
download); torch runs on the CPU here, so CUDA kernels (cuFFT, cuBLAS) are not exercised.
"""

import json

import numpy as np
import pytest
import torch

from indicear.augment import speed, telephone
from indicear.features import GpuLogMel, pad_waveforms

# preprocessor_config.json of EmbeddingGemma 2 at revision 914f7f89142e33e77833254d9c9b90c3cef7303b.
EG2_PREPROCESSOR = {
    "dither": 0.0, "feature_size": 128, "fft_length": 512, "fft_overdrive": False, "frame_length": 320,
    "hop_length": 160, "input_scale_factor": 1.0, "max_frequency": 8000.0, "mel_floor": 0.001,
    "min_frequency": 0.0, "padding_side": "right", "padding_value": 0.0, "per_bin_mean": None,
    "per_bin_stddev": None, "preemphasis": 0.0, "preemphasis_htk_flavor": True, "return_attention_mask": True,
    "sampling_rate": 16000,
}
TOLERANCE = 1e-4


def make_extractor(_path=None):
    from transformers.models.gemma4.feature_extraction_gemma4 import (
        Gemma4AudioFeatureExtractor,
    )

    return Gemma4AudioFeatureExtractor(**EG2_PREPROCESSOR)


@pytest.fixture(scope="module")
def extractor():
    return make_extractor()


@pytest.fixture(scope="module")
def logmel(extractor):
    return GpuLogMel.from_extractor(extractor)


def compare(extractor, logmel, audios, waveforms=None, lengths=None) -> float:
    reference = extractor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
    if waveforms is None:
        waveforms, lengths = pad_waveforms(audios)
    features, mask = logmel(waveforms, lengths)
    assert features.dtype == torch.float32 and mask.dtype == torch.bool
    assert features.shape == reference["input_features"].shape
    assert torch.equal(mask, reference["input_features_mask"])
    diff = float((features - reference["input_features"]).abs().max()) if features.numel() else 0.0
    assert diff <= TOLERANCE, diff
    return diff


@pytest.mark.parametrize(
    "lengths",
    [
        [16000],  # one utterance, a multiple of 128 samples
        [16001, 7, 160, 161, 320, 321, 481],  # frame and mask boundaries, a row with no valid frame
        [48037, 33333, 16127, 16128, 16129],  # pad_to_multiple_of boundaries
        [5 * 16000 + 17, 2 * 16000, 800],
    ],
)
def test_matches_extractor_on_random_padded_batches(extractor, logmel, lengths):
    rng = np.random.default_rng(len(lengths))
    loud = [rng.uniform(-1, 1, n).astype(np.float32) for n in lengths]
    quiet = [(0.001 * rng.standard_normal(n)).astype(np.float32) for n in lengths]
    compare(extractor, logmel, loud)
    compare(extractor, logmel, quiet)


def test_matches_extractor_on_band_limited_and_resampled_audio(extractor, logmel):
    """Near-empty high mel bins: where a float32 FFT drifts from numpy (see features.py)."""
    t = np.arange(4 * 16000) / 16000
    voiced = 0.3 * np.sin(2 * np.pi * 150 * t) * (np.sin(2 * np.pi * 2 * t) > 0)
    voiced = (voiced + sum(0.05 / h * np.sin(2 * np.pi * 150 * h * t) for h in range(2, 20))).astype(np.float32)
    compare(extractor, logmel, [telephone(voiced), speed(voiced, 1.1), voiced[:30000]])


def test_truncates_like_the_extractor_beyond_30_seconds(extractor, logmel):
    rng = np.random.default_rng(1)
    audios = [rng.uniform(-1, 1, 30 * 16000 + 4321).astype(np.float32), rng.uniform(-1, 1, 20000).astype(np.float32)]
    compare(extractor, logmel, audios)


def test_ignores_samples_past_each_length_and_extra_padding(extractor, logmel):
    rng = np.random.default_rng(2)
    audios = [rng.uniform(-1, 1, n).astype(np.float32) for n in (9000, 4000)]
    waveforms = torch.from_numpy(rng.uniform(-1, 1, (2, 12000)).astype(np.float32))  # junk everywhere
    for row, audio in enumerate(audios):
        waveforms[row, : len(audio)] = torch.from_numpy(audio)
    compare(extractor, logmel, audios, waveforms, torch.tensor([9000, 4000]))


def test_unsupported_extractor_settings_raise():
    from transformers.models.gemma4.feature_extraction_gemma4 import (
        Gemma4AudioFeatureExtractor,
    )

    with pytest.raises(NotImplementedError, match="dither"):
        GpuLogMel.from_extractor(Gemma4AudioFeatureExtractor(**{**EG2_PREPROCESSOR, "dither": 1e-4}))


# ---------------------------------------------------------------- loader path


def write_audio(directory, lengths) -> list[dict]:
    import soundfile as sf

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    items = []
    for index, n in enumerate(lengths):
        path = directory / f"u{index}.flac"
        sf.write(path, (0.3 * rng.standard_normal(n)).clip(-1, 1).astype(np.float32), 16000)
        items.append({"key": f"u{index}", "audio": str(path), "seconds": n / 16000})
    return items


@pytest.fixture
def pinned_extractor(monkeypatch):
    """`load_extractor()` returns the EG2-config extractor instead of downloading it."""
    import transformers

    monkeypatch.setattr(transformers.AutoFeatureExtractor, "from_pretrained", lambda *a, **k: make_extractor())


def test_gpu_backend_matches_cpu_backend_through_the_loader(tmp_path, pinned_extractor):
    from indicear.encoder import FeatureBatches, Featurizer, feature_loader

    items = write_audio(tmp_path, [16000, 23456, 8000, 31000])
    groups = [items[:3], items[3:]]
    logs = []
    cpu, gpu = Featurizer("cpu", "cpu", torch.float32), Featurizer("gpu", "cpu", torch.float32)
    cpu_batches = list(feature_loader(FeatureBatches(groups, backend="cpu"), 0))
    gpu_batches = list(feature_loader(FeatureBatches(groups, backend="gpu"), 0))
    assert set(gpu_batches[0][1]) == {"waveforms", "lengths"}
    for (i, cpu_batch), (j, gpu_batch) in zip(cpu_batches, gpu_batches, strict=True):
        assert i == j
        assert cpu.check_once(cpu_batch, logs.append) is None  # nothing to compare on "cpu"
        gpu.check_once(gpu_batch, logs.append)
        (a, mask_a), (b, mask_b) = cpu(cpu_batch), gpu(gpu_batch)
        assert torch.equal(mask_a, mask_b)
        assert float((a - b).abs().max()) <= TOLERANCE
    assert len(logs) == 1 and "max |diff|" in logs[0]  # checked once, on the first batch


def test_check_once_stops_on_a_feature_mismatch(tmp_path, pinned_extractor):
    from indicear.encoder import FeatureBatches, Featurizer, feature_loader

    items = write_audio(tmp_path, [16000, 8000])
    featurizer = Featurizer("gpu", "cpu", torch.float32)
    featurizer.logmel.mel_filters.mul_(1.01)
    (_, batch), = list(feature_loader(FeatureBatches([items], backend="gpu"), 0))
    with pytest.raises(SystemExit, match="differs from the extractor"):
        featurizer.check_once(batch, lambda _m: None)


def test_cache_does_not_depend_on_the_feature_backend(tmp_path, tiny_tower, pinned_extractor):
    from indicear.cache import Layer6Cache, build_cache
    from indicear.encoder import StagedTower

    items = write_audio(tmp_path / "audio", [16000, 23456, 8000, 31000, 12000])
    staged = StagedTower(tiny_tower)
    logs = []
    for backend in ("cpu", "gpu"):
        build_cache(items, tmp_path / backend, staged, batch_seconds=4.0, workers=0, features=backend,
                    log=logs.append)
    assert any("GPU log-mel vs Hugging Face extractor" in line for line in logs)
    meta = [json.loads((tmp_path / b / "cache.json").read_text()) for b in ("cpu", "gpu")]
    assert meta[0] == meta[1]
    index = [(tmp_path / b / "part-00000.jsonl").read_text() for b in ("cpu", "gpu")]
    assert index[0] == index[1]
    caches = [Layer6Cache.load(tmp_path / b, device="cpu", log=lambda _m: None) for b in ("cpu", "gpu")]
    keys = [item["key"] for item in items]
    (a, lengths_a), (b, lengths_b) = (c.batch(keys, "cpu", torch.float32) for c in caches)
    assert torch.equal(lengths_a, lengths_b)
    assert float((a - b).abs().max()) <= 0.02 * float(a.abs().max())  # bf16 rounding at most
