"""Live-encoder training (lower="live"): the live lower stack matches the layer-6 cache, and a
killed live run resumes at the right stream position with the same loss trajectory."""

import csv

import numpy as np
import pytest
import torch
from test_features import make_extractor

from indicear import phonetic
from indicear.encoder import Featurizer, StagedTower
from indicear.features import pad_waveforms
from indicear.lid import LidConfig
from indicear.model import DesignA, resolve_variant
from indicear.stream_train import (
    ListStream,
    LiveLower,
    StreamSource,
    frames_to_samples,
    samples_to_frames,
    synthetic_batch,
)
from indicear.trainer import TrainConfig, Trainer

RANGES = [(4, 20), (20, 40), (40, 50)]
VOCAB = 50


@pytest.fixture(scope="module")
def extractor():
    return make_extractor()


def tone_audio(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * 16_000)) / 16_000
    audio = 0.3 * np.sin(2 * np.pi * (180 + 40 * seed) * t) + 0.05 * rng.standard_normal(t.size)
    return audio.astype(np.float32)


def test_samples_to_frames_is_exact(tiny_tower, extractor):
    featurizer = Featurizer.from_extractor(extractor, "cpu", torch.float32)
    lower = LiveLower(StagedTower(tiny_tower, 6), featurizer)
    lengths = [481, 800, 1921, 2081, 8000, 12345, 16000, 16001, 16161, 40_000]
    waveforms, lengths_t = pad_waveforms([tone_audio(n / 16_000, i) for i, n in enumerate(lengths)])
    _, frames = lower({"waveforms": waveforms, "lengths": lengths_t})
    assert frames.tolist() == [samples_to_frames(n) for n in lengths] == samples_to_frames(lengths_t).tolist()
    assert all(samples_to_frames(frames_to_samples(f)) == f for f in (1, 2, 25, 750))
    assert samples_to_frames(frames_to_samples(25) - 1) == 24


def test_live_lower_matches_the_layer6_cache(tiny_tower, extractor, tmp_path, monkeypatch):
    """The cache path (HF extractor in the loader, layers 1-6, bf16 storage, RowCache.batch)
    against the live path (GpuLogMel, layers 1-6 on the padded batch) on the same audio."""
    import soundfile as sf

    import indicear.encoder
    from indicear.cache import Layer6Cache, build_cache

    monkeypatch.setattr(indicear.encoder, "load_extractor", lambda *_a: extractor)  # no Hub download
    staged = StagedTower(tiny_tower, 6)
    seconds = [1.3, 2.0, 0.7, 3.1]
    audios, items = [], []
    for i, s in enumerate(seconds):
        audio = tone_audio(s, i)
        path = tmp_path / f"a{i}.wav"
        sf.write(path, audio, 16_000, subtype="FLOAT")
        audios.append(audio)
        items.append({"key": f"k{i}", "audio": str(path), "seconds": s})
    build_cache(items, tmp_path / "cache", staged, storage="bf16", workers=0, features="cpu",
                batch_seconds=4.0, log=lambda _m: None)
    cache = Layer6Cache.load(tmp_path / "cache", device="cpu", log=lambda _m: None)
    keys = [i["key"] for i in items]
    cached, cached_lengths = cache.batch(keys, "cpu", torch.float32)

    waveforms, lengths = pad_waveforms(audios)
    lower = LiveLower(staged, Featurizer.from_extractor(extractor, "cpu", torch.float32))
    live, live_lengths = lower({"waveforms": waveforms, "lengths": lengths})
    assert torch.equal(live_lengths, cached_lengths)
    longest = int(cached_lengths.max())
    assert live.shape[:2] == (4, longest)
    scale = float(live.abs().max())
    # bf16 storage rounds to 8 mantissa bits: |x - bf16(x)| <= 2^-8 |x|.
    diff = (live - cached).abs()
    assert float(diff.max()) <= 2 ** -8 * scale + 1e-4, float(diff.max())
    assert torch.all(live[1, cached_lengths[1]:] == 0)  # padding zeroed, as RowCache.batch returns it
    # Before bf16 storage the two paths agree to the GpuLogMel tolerance (fp32 tower).
    with torch.no_grad():
        features = extractor(audios, sampling_rate=16_000, return_tensors="pt", padding=True)
        reference, valid = staged.lower(features["input_features"], features["input_features_mask"])
    reference = reference * valid[..., None]
    assert torch.allclose(live, reference, atol=1e-5 * max(1.0, scale))  # measured 2e-6


# ---------------------------------------------------------------- live trainer


def text_encoder(text: str):
    tokens = [4 + (ord(c) * 7) % (VOCAB - 4) for c in text.replace(" ", "")]
    units = [2 + ord(c) % (len(phonetic.UNITS) - 2) for c in text.replace(" ", "")]
    return tokens, units


def make_batches(count: int = 8):
    rng = np.random.default_rng(3)
    langs = ["hi", "en", "ta", "ur", "none"]
    batches = []
    for b in range(count):
        rows = 2 + b % 2
        audios = [tone_audio(float(rng.uniform(0.8, 1.6)), 10 * b + r) for r in range(rows)]
        waveforms, lengths = pad_waveforms(audios)
        text = ["".join(chr(97 + int(x)) for x in rng.integers(0, 26, 6)) for _ in range(rows)]
        lang = [langs[(b + r) % len(langs)] for r in range(rows)]
        text = ["" if x == "none" else t for t, x in zip(text, lang, strict=True)]
        word_langs = [["hi", "en", "hi"] if x == "hi" and b % 3 == 0 else None for x in lang]
        batches.append({"waveforms": waveforms, "lengths": lengths, "text": text, "lang": lang,
                        "word_langs": word_langs, "epoch": b // 4, "key": [f"b{b}r{r}" for r in range(rows)]})
    # One row whose transcript cannot fit its frames: dropped, never trained.
    batches[2]["text"][0] = "x" * 200
    return batches


def live_trainer(tmp_path, name, tower, extractor, stream, max_steps=8):
    staged = StagedTower(tower, 6)
    variant, lid = resolve_variant("v3+lid", LidConfig(utterance_weight=0.3, frame_weight=0.3))
    torch.manual_seed(0)
    model = DesignA(staged, variant, VOCAB, RANGES, len(phonetic.UNITS), lid=lid)
    config = TrainConfig(max_steps=max_steps, frame_budget=200, log_every=1, eval_every=0, warmup_steps=2,
                         ckpt_every_steps=1000, ckpt_every_seconds=1e9, compile_heads=False)
    featurizer = Featurizer.from_extractor(extractor, "cpu", torch.float32)
    trainer = Trainer(model, None, [], config, tmp_path / name, device="cpu", dtype=torch.float32,
                      log=lambda _m: None, lower="live", stream=stream, featurizer=featurizer,
                      text_encoder=text_encoder)
    return trainer, model


def run_live(tmp_path, name, tower, extractor, kill_at=None, monkeypatch=None):
    stream = ListStream(make_batches(), name="toy")
    trainer, model = live_trainer(tmp_path, name, tower, extractor, stream)
    try:
        if kill_at is not None:
            monkeypatch.setenv("INDICEAR_TEST_KILL_AT_STEP", str(kill_at))
        return trainer.run(), stream
    finally:
        if monkeypatch is not None:
            monkeypatch.delenv("INDICEAR_TEST_KILL_AT_STEP", raising=False)
        model.detach_adapters()


def losses(path):
    rows = list(csv.DictReader(path.open()))
    return [int(r["step"]) for r in rows], [[float(r[k]) for k in ("loss", "ctc", "lid_utt", "lid_frame")] for r in rows]


def test_live_kill_and_resume_restores_the_stream(tmp_path, tiny_tower, extractor, monkeypatch):
    full, full_stream = run_live(tmp_path, "full", tiny_tower, extractor)
    assert full["steps"] == 8 and full["lower"] == "live" and full["dropped_rows"] == 1
    assert full_stream.frame_budget == 200 and full_stream.yielded == list(range(8))
    steps_full, loss_full = losses(tmp_path / "full" / "train.csv")
    assert steps_full == list(range(1, 9))

    with pytest.raises(SystemExit) as stopped:
        run_live(tmp_path, "resumed", tiny_tower, extractor, kill_at=3, monkeypatch=monkeypatch)
    assert stopped.value.code == 143
    checkpoint = torch.load(next((tmp_path / "resumed").glob("ckpt-*.pt")), weights_only=False)
    # The prefetcher pulled ahead of step 3; the checkpoint holds the position after batch 3.
    assert checkpoint["step"] == 3
    assert checkpoint["sampler"] == {"lower": "live", "stream": {"position": 3}}

    resumed, resumed_stream = run_live(tmp_path, "resumed", tiny_tower, extractor)
    assert resumed_stream.yielded == [3, 4, 5, 6, 7]  # resumed at the stream position, nothing replayed
    assert resumed["steps"] == 8
    steps_resumed, loss_resumed = losses(tmp_path / "resumed" / "train.csv")
    assert steps_resumed == steps_full
    for a, b in zip(loss_resumed, loss_full, strict=True):
        assert a == pytest.approx(b, rel=1e-4, abs=1e-5)

    again, again_stream = run_live(tmp_path, "resumed", tiny_tower, extractor)
    assert again["steps"] == 8 and again_stream.yielded == []  # complete: the stream is not touched


def test_live_mode_validation_and_synthetic_batch(tmp_path, tiny_tower, extractor):
    stream = ListStream([], name="empty")
    assert isinstance(stream, StreamSource)
    with pytest.raises(ValueError, match="max_steps"):
        live_trainer(tmp_path, "a", tiny_tower, extractor, stream, max_steps=0)[1].detach_adapters()
    with pytest.raises(ValueError, match="lower"):
        Trainer(None, None, [], TrainConfig(), tmp_path / "b", lower="disk")
    trainer, model = live_trainer(tmp_path, "c", tiny_tower, extractor, stream)
    try:
        batch = synthetic_batch(model, rows=2, frames=40)
        assert batch["frames"].tolist() == [40, 40]
        logs: dict = {}
        model.train()
        trainer.micro_step(batch, 1.0, logs)  # the autotune step: lower + upper + heads + backward
        assert np.isfinite(logs["loss"]) and "lid_utt" in logs
    finally:
        model.detach_adapters()


class FakePlanner:
    def __init__(self, batches):
        self.batches, self.position, self.frame_budget = batches, 0, 0

    def state(self):
        return {"next": self.position}

    def load_state(self, state):
        self.position = state["next"]


class FakeLoader:
    """The shape of indicear.stream.StreamLoader: "items" rows and a per-batch "state"."""

    def __init__(self, batches):
        self.planner, self.state = FakePlanner(batches), None

    def __iter__(self):
        while self.planner.position < len(self.planner.batches):
            raw = self.planner.batches[self.planner.position]
            self.planner.position += 1
            items = [{"text": t, "lang": x, "word_langs": w}
                     for t, x, w in zip(raw["text"], raw["lang"], raw["word_langs"], strict=True)]
            self.state = self.planner.state()
            yield {"waveforms": raw["waveforms"], "lengths": raw["lengths"], "items": items, "state": self.state}


def test_loader_source_items_and_batch_state(tmp_path, tiny_tower, extractor):
    from indicear.stream_train import LoaderSource

    source = LoaderSource(FakeLoader(make_batches(5)), fingerprint={"mixture": "toy"})
    trainer, model = live_trainer(tmp_path, "loader", tiny_tower, extractor, source, max_steps=3)
    try:
        summary = trainer.run()
    finally:
        model.detach_adapters()
    assert summary["steps"] == 3 and source.loader.planner.frame_budget == 200
    checkpoint = torch.load(next((tmp_path / "loader").glob("ckpt-*.pt")), weights_only=False)
    assert checkpoint["sampler"]["stream"] == {"next": 3}
    source.load_state_dict(checkpoint["sampler"]["stream"])
    assert source.loader.planner.position == 3 and source.state_dict() == {"next": 3}
