"""GPU augmentation (indicear.gpu_augment) against the CPU Corruptor: achieved SNR, RIR decay,
power, band limits, speed; determinism; and the codec worker transform."""

import numpy as np
import pytest
import torch

from indicear import augment
from indicear.gpu_augment import (
    GPU_KINDS,
    GpuAugment,
    bandlimit,
    fft_convolve,
    mix_at_snr,
    parse_gpu_policy,
    sinc_resample,
)

SR = 16_000


def tone(seconds, freq=440.0, seed=0):
    t = np.arange(int(seconds * SR)) / SR
    noise = np.random.default_rng(seed).normal(0, 0.02, t.size)
    return (0.3 * np.sin(2 * np.pi * freq * t) + noise).astype(np.float32)


def snr_db(clean, mixed):
    clean, mixed = np.asarray(clean, np.float64), np.asarray(mixed, np.float64)
    return 10 * np.log10(np.mean(clean ** 2) / np.mean((mixed - clean) ** 2))


def schroeder_rt60(rir):
    """RT60 from the -5..-25 dB slope of the Schroeder backward-integrated energy decay."""
    energy = np.cumsum(np.asarray(rir, np.float64)[::-1] ** 2)[::-1]
    decay = 10 * np.log10(energy / energy[0] + 1e-30)
    index = np.flatnonzero((decay <= -5) & (decay >= -25))
    slope = np.polyfit(index / SR, decay[index], 1)[0]
    return -60.0 / slope


def only(kind, **kwargs):
    return GpuAugment("cpu", {kind: 1.0}, rirs=32, noise_clips=2, noise_seconds=6.0, **kwargs)


def test_snr_is_achieved_like_the_cpu_corruptor():
    audios = [tone(2.0 + 0.5 * i, 300 + 50 * i, i) for i in range(6)]
    waveforms = torch.zeros(6, max(map(len, audios)))
    for i, a in enumerate(audios):
        waveforms[i, : len(a)] = torch.from_numpy(a)
    lengths = torch.tensor([len(a) for a in audios])
    gpu = only("noise")
    out, new, conditions = gpu(waveforms, lengths, [f"k{i}" for i in range(6)], [1] * 6, [True] * 6)
    assert torch.equal(new, lengths)
    for i, condition in enumerate(conditions):
        wanted = float(condition.split(":")[1])
        assert 5.0 <= wanted <= 20.0
        assert snr_db(audios[i], out[i, : len(audios[i])].numpy()) == pytest.approx(wanted, abs=0.5)
        assert float(out[i, len(audios[i]):].abs().max() if len(audios[i]) < out.shape[1] else 0) == 0.0
        noise, _ = augment.NoiseBank().noise(len(audios[i]), np.random.default_rng(i))
        cpu = augment.mix_at_snr(audios[i], noise, wanted)  # the CPU reference hits the same SNR
        assert snr_db(audios[i], cpu) == pytest.approx(wanted, abs=0.5)
    # babble at its SNR too, from the other rows only
    babble = only("babble")
    out, _, conditions = babble(waveforms, lengths, [f"k{i}" for i in range(6)], [1] * 6, [True] * 6)
    for i, condition in enumerate(conditions):
        assert condition.startswith("babble:")
        assert snr_db(audios[i], out[i, : len(audios[i])].numpy()) == pytest.approx(float(condition[7:]), abs=0.5)


def test_rir_bank_decays_like_the_cpu_rirs_and_reverb_keeps_power():
    gpu = only("reverb")
    rng = np.random.default_rng(3)
    for index in (0, len(gpu.rt60) // 2, len(gpu.rt60) - 1):
        rt60 = gpu.rt60[index]
        measured = schroeder_rt60(gpu.rirs[index].numpy())
        reference = schroeder_rt60(augment.synthetic_rir(float(rt60), rng))
        assert measured == pytest.approx(rt60, rel=0.15) and reference == pytest.approx(rt60, rel=0.15)
    assert 0.3 <= gpu.rt60.min() and gpu.rt60.max() <= 0.9
    audio = tone(3.0)
    x = torch.from_numpy(audio)[None]
    lengths = torch.tensor([len(audio)])
    out = fft_convolve(x, lengths, gpu.rirs[:1])
    cpu = augment.convolve(audio, gpu.rirs[0].numpy())
    assert np.mean(out[0].numpy() ** 2) == pytest.approx(np.mean(audio ** 2), rel=1e-3)
    assert np.allclose(out[0].numpy(), cpu, atol=1e-4)  # same operation as the CPU convolve


def test_bandlimit_and_speed():
    audio = tone(2.0, 200.0) + tone(2.0, 5000.0, 1)
    x = torch.from_numpy(audio)[None]
    lengths = torch.tensor([len(audio)])
    out = bandlimit(x, lengths)[0].numpy()[1024:-1024]  # steady state: skip the FIR edge transients
    spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    freqs = np.fft.rfftfreq(len(out), 1 / SR)
    assert spectrum[freqs > 3600].max() < 0.02 * spectrum.max() and spectrum[freqs < 250].max() < 0.02 * spectrum.max()
    sine = tone(1.0, 1000.0)
    for factor in (0.9, 1.1):
        y, new = sinc_resample(torch.from_numpy(sine)[None], torch.tensor([len(sine)]), factor)
        assert int(new[0]) == round(len(sine) / factor) == len(augment.speed(sine, factor))
        peak = np.fft.rfftfreq(int(new[0]), 1 / SR)[np.argmax(np.abs(np.fft.rfft(y[0].numpy())))]
        assert peak == pytest.approx(1000.0 * factor, abs=3.0)  # played `factor` times faster


def test_policy_is_deterministic_and_leaves_clean_rows_alone():
    audios = [tone(1.0 + 0.3 * i, 200 + 40 * i, i) for i in range(8)]
    waveforms = torch.zeros(8, max(map(len, audios)))
    for i, a in enumerate(audios):
        waveforms[i, : len(a)] = torch.from_numpy(a)
    lengths = torch.tensor([len(a) for a in audios])
    keys = [f"utt{i}" for i in range(8)]
    views = [0, 3, 4, 5, 6, 7, 8, 9]
    speech = [True] * 7 + [False]
    gpu = GpuAugment("cpu", rirs=16, noise_clips=2, noise_seconds=4.0)
    a, la, ca = gpu(waveforms, lengths, keys, views, speech)
    b, lb, cb = GpuAugment("cpu", rirs=16, noise_clips=2, noise_seconds=4.0)(waveforms, lengths, keys, views, speech)
    assert torch.equal(a, b) and torch.equal(la, lb) and ca == cb
    assert ca[0] is None and ca[7] is None  # clean draw; non-speech row
    assert torch.equal(a[0, : len(audios[0])], waveforms[0, : len(audios[0])])
    _, _, cc = gpu(waveforms, lengths, keys, [0, 13, 14, 15, 16, 17, 18, 19], speech)
    assert cc != ca  # another view draws another corruption
    counts = {k: 0 for k in GPU_KINDS}
    for plan in gpu.plan([f"k{i}" for i in range(4000)], [1] * 4000, [True] * 4000):
        counts[plan["kind"]] += 1
    for kind, weight in parse_gpu_policy("").items():
        assert counts[kind] / 4000 == pytest.approx(weight, abs=0.03)


def test_codec_worker_transform():
    codec = augment.CodecAugment(prob=1.0, codecs=("opus",))
    audio = tone(1.0)
    item = {"text": "x", "key": "a", "view": 0}
    out = codec(audio, np.random.default_rng(0), item)
    if "opus" in augment.available_codecs():
        assert item["codec"] == "opus" and out.shape == audio.shape and not np.array_equal(out, audio)
    else:
        assert out is audio
    assert augment.CodecAugment(prob=1.0, only_augmented=True)(audio, np.random.default_rng(0), {"text": "x"}) is audio
    assert augment.CodecAugment(prob=0.0)(audio, np.random.default_rng(0), item) is audio


def test_mix_at_snr_matches_numpy():
    rng = np.random.default_rng(0)
    speech, noise = rng.standard_normal(16000).astype(np.float32), rng.standard_normal(16000).astype(np.float32)
    torch_out = mix_at_snr(torch.from_numpy(speech)[None], torch.from_numpy(noise)[None], torch.tensor([16000]),
                           torch.tensor([7.5]))[0].numpy()
    assert np.allclose(torch_out, augment.mix_at_snr(speech, noise, 7.5), atol=1e-5)


# ---------------------------------------------------------------- measured RIRs (rir_files / load_rirs)


def write_rir(path, rate=16_000, channels=1, rt60=0.5, delay_s=0.05, seconds=1.5, seed=0):
    """A measured-looking RIR: silence, a direct-path peak at `delay_s`, an exponential tail."""
    import soundfile as sf

    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    tail = rng.standard_normal(n) * np.exp(-6.9078 * (t - delay_s) / rt60) * 0.2
    tail[t < delay_s + 0.002] = 0.0
    tail[int(delay_s * rate)] = 1.0
    data = np.stack([tail * (0.5 + 0.5 * c) for c in range(channels)], axis=1) * 0.5
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data.astype(np.float32), rate, subtype="PCM_16")
    return path


def slr28_tree(root):
    """An SLR28-like directory: rir_list names three RIRs; isotropic noises sit beside them."""
    folder = root / "RIRS_NOISES" / "real_rirs_isotropic_noises"
    names = ["RVB2014_type1_rir_largeroom1_far_angla.wav", "air_type1_air_binaural_aula_1.wav",
             "RWCP_type1_rir_circle_e1a_imp000.wav"]
    write_rir(folder / names[0], rate=16_000, channels=8, rt60=0.6, seed=1)
    write_rir(folder / names[1], rate=48_000, channels=2, rt60=0.9, delay_s=0.01, seconds=3.0, seed=2)
    write_rir(folder / names[2], rate=8_000, channels=1, rt60=0.3, seed=3)
    write_rir(folder / "RVB2014_type1_noise_largeroom1_1.wav", seed=4)  # isotropic noise: not an RIR
    (folder / "rir_list").write_text("".join(
        f"--rir-id {i:05d} --room-id room{i} RIRS_NOISES/real_rirs_isotropic_noises/{n}\n"
        for i, n in enumerate(names, 1)))
    return folder, names


def test_rir_files_follow_rir_list_and_sets(tmp_path):
    from indicear.gpu_augment import rir_files

    folder, names = slr28_tree(tmp_path)
    files = rir_files(tmp_path)
    assert [p.rsplit("/", 1)[1] for p in files] == sorted(names)  # sorted, noise file left out
    assert rir_files(folder) == files  # the folder itself works too
    assert [p.rsplit("/", 1)[1] for p in rir_files(tmp_path, sets=("air", "RVB2014"))] == sorted(names[:2])
    plain = tmp_path / "plain"
    for name in ("b.wav", "a.flac", "c.txt"):
        write_rir(plain / "x" / name, seed=5) if name != "c.txt" else (plain / "x" / name).write_text("x")
    assert [p.rsplit("/", 1)[1] for p in rir_files(plain)] == ["a.flac", "b.wav"]


def test_load_rirs_aligns_truncates_and_normalizes(tmp_path):
    from collections import Counter

    from indicear.gpu_augment import RIR_PRE_PEAK, load_rirs, rir_files

    slr28_tree(tmp_path)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    silent = write_rir(tmp_path / "silent.wav")
    import soundfile as sf

    sf.write(str(silent), np.zeros(1600, np.float32), 16_000)
    skipped, rt60s = Counter(), []
    rirs, kept = load_rirs([*rir_files(tmp_path), str(bad), str(silent)], skipped=skipped, rt60s=rt60s)
    assert len(rirs) == len(kept) == len(rt60s) == 3 and skipped == {"unreadable": 1, "silent": 1}
    for rir in rirs:
        assert rir.dtype == np.float32 and len(rir) <= int(1.2 * SR)
        assert int(np.argmax(np.abs(rir))) == RIR_PRE_PEAK  # direct path 1 ms in
        assert float(np.sum(rir.astype(np.float64) ** 2)) == pytest.approx(1.0, rel=1e-5)
    by_name = dict(zip([k.rsplit("/", 1)[1] for k in kept], zip(rirs, rt60s, strict=True), strict=True))
    assert len(by_name["air_type1_air_binaural_aula_1.wav"][0]) == int(1.2 * SR)  # 3 s at 48 kHz: cut
    assert by_name["air_type1_air_binaural_aula_1.wav"][1] == pytest.approx(0.9, rel=0.2)
    assert by_name["RVB2014_type1_rir_largeroom1_far_angla.wav"][1] == pytest.approx(0.6, rel=0.2)
    assert by_name["RWCP_type1_rir_circle_e1a_imp000.wav"][1] == pytest.approx(0.3, rel=0.25)  # from 8 kHz


def test_without_rir_files_plans_bank_and_describe_are_unchanged(tmp_path):
    keys, views, speech = [f"k{i}" for i in range(300)], [1] * 300, [True] * 300
    base = GpuAugment("cpu", rirs=16, noise_clips=2, noise_seconds=4.0)
    same = GpuAugment("cpu", rirs=16, noise_clips=2, noise_seconds=4.0, rir_files=(), real_rir_share=0.5)
    assert same.plan(keys, views, speech) == base.plan(keys, views, speech)
    assert torch.equal(same.rirs, base.rirs) and torch.equal(same.noise, base.noise)
    assert same.describe() == base.describe()
    assert set(base.describe()) == {"device", "weights", "snr_db", "rt60_s", "speeds", "rirs", "noise_clips", "musan"}
    assert base.describe()["rirs"] == 16


def test_real_rirs_share_determinism_and_power(tmp_path):
    from indicear.gpu_augment import rir_files

    slr28_tree(tmp_path)
    files = rir_files(tmp_path)
    kwargs = {"weights": {"reverb": 0.5, "reverb_pointnoise": 0.5}, "rirs": 16, "noise_clips": 2,
              "noise_seconds": 4.0, "rir_files": files, "real_rir_share": 0.3}
    gpu = GpuAugment("cpu", **kwargs)
    synthetic = GpuAugment("cpu", rirs=16, noise_clips=2, noise_seconds=4.0)
    assert gpu.rirs.shape[0] == 19 and torch.equal(gpu.rirs[:16, : synthetic.rirs.shape[1]], synthetic.rirs)
    assert torch.equal(gpu.noise, synthetic.noise)  # the noise bank still follows the same seed
    described = gpu.describe()
    assert described["real_rirs"] == 3 and described["real_rir_share"] == 0.3 and described["rirs"] == 16
    assert len(described["real_rir_sha256"]) == 64
    keys = [f"k{i}" for i in range(3000)]
    plans = gpu.plan(keys, [1] * 3000, [True] * 3000)
    real = [p for p in plans if p["rir"] >= 16]
    assert len(real) / 3000 == pytest.approx(0.3, abs=0.03)
    assert all(p["condition"].startswith("reverb-real:") for p in real)
    assert all(p["noise_rir"] < 16 for p in plans if p["kind"] == "reverb_pointnoise")  # point noise: synthetic
    assert {p["rir"] for p in real} == {16, 17, 18}
    assert GpuAugment("cpu", **kwargs).plan(keys, [1] * 3000, [True] * 3000) == plans  # deterministic
    assert GpuAugment("cpu", **{**kwargs, "rir_files": files[:2]}).describe()["real_rir_sha256"] != \
        described["real_rir_sha256"]  # the hash names the RIR set
    audios = [tone(1.5 + 0.25 * i, 250 + 30 * i, i) for i in range(6)]
    waveforms = torch.zeros(6, max(map(len, audios)))
    for i, a in enumerate(audios):
        waveforms[i, : len(a)] = torch.from_numpy(a)
    lengths = torch.tensor([len(a) for a in audios])
    only_real = GpuAugment("cpu", {"reverb": 1.0}, rirs=16, noise_clips=2, noise_seconds=4.0, rir_files=files,
                           real_rir_share=1.0)
    out, new, conditions = only_real(waveforms, lengths, [f"u{i}" for i in range(6)], [1] * 6, [True] * 6)
    assert torch.equal(new, lengths) and all(c.startswith("reverb-real:") for c in conditions)
    for i, audio in enumerate(audios):
        row = out[i].numpy()
        assert np.mean(row[: len(audio)].astype(np.float64) ** 2) == pytest.approx(np.mean(audio ** 2), rel=1e-3)
        assert not np.allclose(row[: len(audio)], audio, atol=1e-3)
        assert np.all(row[len(audio):] == 0)
    again, _, _ = GpuAugment("cpu", {"reverb": 1.0}, rirs=16, noise_clips=2, noise_seconds=4.0, rir_files=files,
                             real_rir_share=1.0)(waveforms, lengths, [f"u{i}" for i in range(6)], [1] * 6, [True] * 6)
    assert torch.equal(out, again)
