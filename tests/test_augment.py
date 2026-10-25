import numpy as np
import pytest

from indicear import augment


def tone(seconds=2.0):
    t = np.arange(int(16000 * seconds)) / 16000
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_mix_at_snr_hits_target():
    rng = np.random.default_rng(0)
    speech = tone()
    noise = augment.colored_noise(len(speech), rng, "pink")
    mixed = augment.mix_at_snr(speech, noise, 10.0)
    snr = 10 * np.log10(np.mean(speech**2) / np.mean((mixed - speech) ** 2))
    assert snr == pytest.approx(10.0, abs=0.1)


def test_reverb_telephone_speed():
    rng = np.random.default_rng(0)
    audio = tone()
    rir = augment.synthetic_rir(0.6, rng)
    assert rir[0] == rir.max()
    assert augment.convolve(audio, rir).shape == audio.shape
    phone = augment.telephone(audio)
    assert phone.shape == audio.shape
    spectrum = np.abs(np.fft.rfft(augment.bandlimit(np.random.default_rng(1).standard_normal(16000).astype(np.float32))))
    freqs = np.fft.rfftfreq(16000, 1 / 16000)
    assert spectrum[freqs > 3500].max() < 1e-4 * spectrum.max()
    assert len(augment.speed(audio, 1.1)) == pytest.approx(len(audio) / 1.1, rel=0.01)
    assert len(augment.speed(audio, 0.9)) == pytest.approx(len(audio) / 0.9, rel=0.01)


def test_conditions_and_recipe():
    rng = np.random.default_rng(0)
    bank = augment.NoiseBank()
    out = augment.apply_condition(tone(), "reverb:0.5+noise:10+speed:0.9", rng, bank)
    assert out.dtype == np.float32 and len(out) > len(tone())
    with pytest.raises(ValueError):
        augment.apply_condition(tone(), "bogus:1", rng, bank)
    recipe = augment.AugmentationRecipe(bank, p_babble=0.0, codecs=("tel8k",))
    item = {"key": "fleurs/hi/train/x#v1", "view": 1}
    a = recipe(tone(), rng, item)
    b = recipe(tone(), np.random.default_rng(99), item)
    assert np.array_equal(a, b)  # seeded by (key, view), not by the worker RNG
    assert np.array_equal(recipe(tone(), rng, {"key": "k", "view": 0}), tone())


@pytest.mark.skipif(not augment.available_codecs(), reason="no ffmpeg codec encoders on this machine")
def test_codecs_round_trip():
    for name in augment.available_codecs():
        out = augment.codec(tone(), name)
        assert out.shape == tone().shape


def test_negatives(tmp_path):
    items = augment.make_negatives(tmp_path, 3, augment.NoiseBank(), seed=0)
    assert [i["text"] for i in items] == ["", "", ""] and all(i["negative"] for i in items)
