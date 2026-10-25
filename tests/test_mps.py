"""Apple MPS support for live training: device selection, the CPU log-mel and augmentation on MPS
hosts, exact micro-batch accumulation, cross-device resume (a CUDA-style checkpoint onto MPS),
fp16 autocast, and that none of it changes the config hash.

Tests marked `mps` run on the real MPS device and are skipped where it is not available."""

import copy
import csv
import os
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_features import make_extractor
from test_stream_train import make_batches, text_encoder, tone_audio

from indicear import encoder, phonetic
from indicear.encoder import (
    Featurizer,
    StagedTower,
    feature_device,
    half_safe_tower,
    runtime_device,
    select_device,
)
from indicear.features import pad_waveforms
from indicear.gpu_augment import GpuAugment, augment_device
from indicear.lid import NUM_LANGS, LidConfig, make_targets, utterance_lid_loss
from indicear.model import DesignA, resolve_variant
from indicear.runtime.batching import OomBackoff, is_oom, split_evenly
from indicear.runtime.checkpoint import derived_seed, rng_state, set_rng_state
from indicear.runtime.gpu import auto_splits
from indicear.stream_train import ListStream, LiveLower, pad_features, prepare, slice_live
from indicear.trainer import TrainConfig, Trainer, autocast_dtype, initial_splits

RANGES = [(4, 20), (20, 40), (40, 50)]
VOCAB = 50
# INDICEAR_TEST_SKIP_MPS: CI sets it on GitHub's macOS VMs, whose virtual GPU reports MPS but computes wrong values.
HAS_MPS = torch.backends.mps.is_available() and not os.environ.get("INDICEAR_TEST_SKIP_MPS")
mps = pytest.mark.skipif(not HAS_MPS, reason="Apple MPS not available (or INDICEAR_TEST_SKIP_MPS set)")
CLASS_WEIGHTS = tuple(0.6 + 0.1 * i for i in range(NUM_LANGS))


@pytest.fixture(scope="module")
def extractor():
    return make_extractor()


def tower_copy(tower, device="cpu"):
    """A private copy of the session tiny tower, without adapter hooks other tests left attached
    (deepcopy would keep hooks that call their CPU adapters)."""
    tower = copy.deepcopy(tower)
    for module in tower.modules():
        module._forward_hooks.clear()
    return tower.to(device)


# ---------------------------------------------------------------- device selection


def fake_host(monkeypatch, cuda: bool, mps_ok: bool):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: cuda)
    monkeypatch.setattr(encoder, "mps_available", lambda: mps_ok)


@pytest.mark.parametrize(("cuda", "mps_ok", "expected"), [
    (True, True, ("cuda", torch.bfloat16)), (False, True, ("mps", torch.float32)), (False, False, ("cpu", torch.float32)),
])
def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch, cuda, mps_ok, expected):
    monkeypatch.delenv("INDICEAR_DEVICE", raising=False)
    fake_host(monkeypatch, cuda, mps_ok)
    assert runtime_device() == expected
    assert runtime_device("auto") == expected


def test_device_override_by_argument_and_environment(monkeypatch):
    fake_host(monkeypatch, cuda=False, mps_ok=True)
    monkeypatch.setenv("INDICEAR_DEVICE", "cpu")
    assert select_device() == "cpu"
    assert select_device("mps") == "mps"  # --device wins over the environment
    monkeypatch.setenv("INDICEAR_DEVICE", "MPS")
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    assert runtime_device() == ("mps", torch.float32)  # fp32 on MPS: bf16 training crashes there
    assert encoder.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    with pytest.raises(SystemExit, match="CUDA is not available"):
        select_device("cuda")
    with pytest.raises(SystemExit, match="not in"):
        select_device("tpu")
    fake_host(monkeypatch, cuda=False, mps_ok=False)
    with pytest.raises(SystemExit, match="MPS is not available"):
        select_device("mps")


def test_mps_hosts_run_logmel_and_augmentation_on_the_cpu():
    assert feature_device("mps") == torch.device("cpu") == augment_device("mps")
    assert feature_device("cpu") == torch.device("cpu")
    assert feature_device("cuda") == torch.device("cuda") == augment_device("cuda")


def test_autocast_dtype_per_device_and_precision():
    assert autocast_dtype(torch.device("cuda"), torch.bfloat16) == torch.bfloat16
    assert autocast_dtype(torch.device("mps"), torch.float32) is None
    assert autocast_dtype(torch.device("mps"), torch.float32, "fp32") is None
    assert autocast_dtype(torch.device("mps"), torch.float32, "fp16") == torch.float16
    assert autocast_dtype(torch.device("mps"), torch.float32, "bf16") == torch.bfloat16
    assert autocast_dtype(torch.device("cpu"), torch.float32) is None
    with pytest.raises(ValueError, match="needs MPS"):
        autocast_dtype(torch.device("cpu"), torch.float32, "fp16")
    with pytest.raises(ValueError, match="for MPS"):
        autocast_dtype(torch.device("cuda"), torch.bfloat16, "fp16")


# ---------------------------------------------------------------- log-mel and augmentation


@mps
def test_logmel_for_mps_matches_the_cpu_featurizer_and_the_extractor(extractor):
    audios = [tone_audio(s, i) for i, s in enumerate((1.3, 0.4, 2.7))]
    waveforms, lengths = pad_waveforms(audios)
    batch = {"waveforms": waveforms, "lengths": lengths}
    on_mps = Featurizer.from_extractor(extractor, "mps", torch.float32)
    assert on_mps.logmel.window.device.type == "cpu"  # float64 log-mel on the CPU
    features, mask = on_mps(batch)
    assert features.device.type == "mps" and mask.device.type == "mps" and features.dtype == torch.float32
    reference, reference_mask = Featurizer.from_extractor(extractor, "cpu", torch.float32)(batch)
    assert torch.equal(features.cpu(), reference) and torch.equal(mask.cpu(), reference_mask)
    messages = []
    diff = on_mps.check_once(batch, messages.append)  # the first-batch check against the HF extractor
    assert diff is not None and diff < 1e-4 and "mask equal" in messages[0]


def test_cpu_augmentation_is_deterministic_and_matches_its_plans():
    """The MPS path: GpuAugment on the CPU. Same keys and views -> identical output, run after run
    (a resumed run replays the same corruption); the plans are device independent."""
    audios = [tone_audio(1.0 + 0.3 * i, i) for i in range(6)]
    waveforms, lengths = pad_waveforms(audios)
    keys, views, speech = [f"row{i}" for i in range(6)], [1, 2, 0, 1, 3, 1], [True] * 6
    first = GpuAugment(augment_device("mps"), rirs=16, noise_clips=2, noise_seconds=4.0)
    second = GpuAugment("cpu", rirs=16, noise_clips=2, noise_seconds=4.0)
    out_a, len_a, cond_a = first(waveforms, lengths, keys, views, speech)
    out_b, len_b, cond_b = second(waveforms.clone(), lengths.clone(), keys, views, speech)
    assert out_a.device.type == "cpu"
    assert torch.equal(out_a, out_b) and torch.equal(len_a, len_b) and cond_a == cond_b
    assert cond_a[2] is None and torch.equal(out_a[2, : lengths[2]], waveforms[2, : lengths[2]])
    assert sum(c is not None for c in cond_a) == 5
    assert first.describe()["device"] == "gpu"  # hashed: names the main-process path, not the hardware


# ---------------------------------------------------------------- exact micro-batches


def make_model(tower, specaugment: bool = False):
    staged = StagedTower(tower, 6)
    variant, lid = resolve_variant("v4+lid", LidConfig(utterance_weight=0.3, frame_weight=0.3,
                                                        class_weights=CLASS_WEIGHTS))
    torch.manual_seed(0)
    return DesignA(staged, replace(variant, specaugment=specaugment), VOCAB, RANGES, len(phonetic.UNITS), lid=lid)


def live_trainer(tmp_path, name, model, extractor, stream, device="cpu", max_steps=8, splits=0, precision="auto"):
    config = TrainConfig(max_steps=max_steps, frame_budget=200, log_every=1, eval_every=0, warmup_steps=2,
                         ckpt_every_steps=1000, ckpt_every_seconds=1e9, compile_heads=False, splits=splits)
    featurizer = Featurizer.from_extractor(extractor, device, torch.float32)
    return Trainer(model, None, [], config, tmp_path / name, device=device, dtype=torch.float32,
                   log=lambda _m: None, lower="live", stream=stream, featurizer=featurizer,
                   text_encoder=text_encoder, precision=precision)


def grads(model) -> torch.Tensor:
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).cpu()


def test_split_batches_accumulate_the_one_piece_gradient(tmp_path, tiny_tower, extractor):
    model = make_model(tiny_tower)
    try:
        trainer = live_trainer(tmp_path, "split", model, extractor, ListStream([]))
        raw = make_batches(12)
        rows = [r for b in raw[:4] for r in range(len(b["text"]))]
        audios = [tone_audio(0.8 + 0.07 * i, i) for i in range(len(rows))]
        waveforms, lengths = pad_waveforms(audios)
        text = [t for b in raw[:4] for t in b["text"]]
        text[0] = ""  # a non-speech row, as the stream carries
        lang = [x for b in raw[:4] for x in b["lang"]]
        word_langs = [w for b in raw[:4] for w in b["word_langs"]]
        batch = prepare({"waveforms": waveforms, "lengths": lengths, "text": text, "lang": lang,
                         "word_langs": word_langs}, text_encoder, pin=False)
        model.train()
        whole: dict = {}
        trainer.micro_step(batch, 1.0, whole)
        reference = grads(model)
        model.zero_grad(set_to_none=True)
        norms = trainer.loss_norms(batch)
        assert set(norms) == {"tokens", "units", "frames", "lid_utt", "lid_frame"}
        for parts in (2, 3, 5):
            split: dict = {}
            for chunk in split_evenly(list(range(len(text))), parts):
                trainer.micro_step(slice_live(batch, chunk), 1.0, split, norms)
            assert torch.allclose(grads(model), reference, rtol=1e-4, atol=1e-6), parts
            for key, value in whole.items():
                assert split[key] == pytest.approx(value, rel=1e-5, abs=1e-6), (parts, key)
            model.zero_grad(set_to_none=True)
        # Row-share weighting (the old OOM-split rule) is NOT the one-piece gradient: token counts differ.
        rowwise: dict = {}
        for chunk in split_evenly(list(range(len(text))), 3):
            trainer.micro_step(slice_live(batch, chunk), len(chunk) / len(text), rowwise)
        assert not torch.allclose(grads(model), reference, rtol=1e-4, atol=1e-6)
        model.zero_grad(set_to_none=True)
    finally:
        model.detach_adapters()


def test_utterance_lid_denominator_matches_torch_mean():
    logits = torch.randn(7, NUM_LANGS)
    targets = make_targets(["hi", "en", "none", "ta", "hi", "ur", "kn"]).utterance
    for weight in (None, torch.tensor(CLASS_WEIGHTS, dtype=torch.float32)):
        from indicear.lid import lid_denominators

        mean = utterance_lid_loss(logits, targets, 0.1, weight)
        denominator = lid_denominators(make_targets(["hi", "en", "none", "ta", "hi", "ur", "kn"]), weight)["lid_utt"]
        assert float(utterance_lid_loss(logits, targets, 0.1, weight, denominator=denominator)) == pytest.approx(
            float(mean), rel=1e-6)


def losses(path):
    rows = list(csv.DictReader(path.open()))
    return [int(r["step"]) for r in rows], [[float(r[k]) for k in ("loss", "ctc", "lid_utt", "lid_frame")] for r in rows]


def test_a_run_with_splits_matches_the_one_piece_run(tmp_path, tiny_tower, extractor):
    curves = {}
    for splits in (1, 3):
        model = make_model(tiny_tower)
        try:
            trainer = live_trainer(tmp_path, f"s{splits}", model, extractor, ListStream(make_batches(), name="toy"),
                                   splits=splits)
            summary = trainer.run()
        finally:
            model.detach_adapters()
        assert summary["splits"] == splits and summary["steps"] == 8
        curves[splits] = losses(tmp_path / f"s{splits}" / "train.csv")
        checkpoint = torch.load(next((tmp_path / f"s{splits}").glob("ckpt-*.pt")), weights_only=False)
        assert checkpoint["splits"] == splits and checkpoint["device"] == "cpu"
    assert curves[1][0] == curves[3][0]
    for a, b in zip(curves[1][1], curves[3][1], strict=True):
        assert a == pytest.approx(b, rel=1e-4, abs=1e-5)


# ---------------------------------------------------------------- splits, OOM, config hash


def test_initial_splits_follow_device_precision_and_request(monkeypatch):
    log = []
    cfg = TrainConfig()
    a100 = {"splits": 1, "frame_budget": 67045}  # an A100 checkpoint: no "device" key
    assert initial_splits(cfg, torch.device("cuda"), 67045, a100, log.append) == 1
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: int(51.84 * 2**30))
    monkeypatch.delenv("INDICEAR_MPS_MEM_GB", raising=False)
    eager = replace(cfg, compile_heads=False)
    assert initial_splits(eager, torch.device("mps"), 67045, a100, log.append, "fp32") == 6
    assert initial_splits(eager, torch.device("mps"), 67045, a100, log.append, "fp16") == 4
    monkeypatch.setenv("INDICEAR_MPS_MEM_GB", "42")  # a 42 GiB cap: 3 compiled fp16 micro-batches
    assert initial_splits(cfg, torch.device("mps"), 67045, a100, log.append, "fp16") == 3
    assert initial_splits(eager, torch.device("mps"), 67045, a100, log.append, "fp32") == 7
    assert "the checkpoint has 1 (cuda" in log[-1]
    mac = {"splits": 8, "device": "mps", "precision": "fp32"}
    assert initial_splits(cfg, torch.device("mps"), 67045, mac, log.append, "fp32") == 8  # kept (e.g. after OOM)
    assert initial_splits(replace(cfg, splits=2), torch.device("mps"), 67045, mac, log.append, "fp32") == 2
    assert initial_splits(cfg, torch.device("cuda"), 67045, mac, log.append) == 1  # back on CUDA: re-derived
    assert auto_splits("cpu", 10**6) == auto_splits("cuda", 10**6) == 1


def test_mps_out_of_memory_is_recognised_and_backed_off():
    error = RuntimeError("MPS backend out of memory (MPS allocated: 40.00 GB, other allocations: 2.00 GB, "
                         "max allowed: 51.84 GB). Tried to allocate 3.00 GB on private pool.")
    assert is_oom(error) and not is_oom(RuntimeError("shape mismatch"))
    calls = []

    def step(chunk, _weight):
        if len(chunk) > 4:
            raise RuntimeError(str(error))
        calls.append(tuple(chunk))

    backoff = OomBackoff(1, log=lambda _m: None)
    backoff.run(list(range(12)), step)
    assert backoff.splits == 4 and sorted(i for c in calls for i in c) == list(range(12))


def test_config_hash_ignores_device_splits_and_precision(tmp_path, tiny_tower, extractor, monkeypatch):
    assert TrainConfig(splits=4).semantic() == TrainConfig().semantic()
    hashes = set()
    for name, device, splits, precision in (("a", "cpu", 0, "auto"), ("b", "cpu", 4, "fp32"),
                                             ("c", "mps", 6, "fp32"), ("d", "mps", 4, "fp16")):
        if device == "mps" and not HAS_MPS:
            continue
        model = make_model(tiny_tower)
        try:
            trainer = live_trainer(tmp_path, name, model, extractor, ListStream(make_batches(), name="toy"),
                                   device=device, splits=splits, precision=precision)
            hashes.add(trainer.hash)
        finally:
            model.detach_adapters()
    assert len(hashes) == 1


# ---------------------------------------------------------------- RNG and cross-device resume


def test_rng_notes_for_a_cuda_state_on_a_host_without_cuda():
    state = rng_state()
    state["cuda"] = [torch.zeros(16, dtype=torch.uint8)]
    state.pop("mps", None)
    notes = set_rng_state(state)
    assert any("CUDA generator state (1 device(s)) cannot be restored" in n for n in notes)
    if HAS_MPS:
        assert any(f"seeded with {derived_seed(state)}" in n for n in notes)
        first = torch.rand(4, device="mps").cpu()
        set_rng_state(state)
        assert torch.equal(torch.rand(4, device="mps").cpu(), first)  # reproducible resume


@mps
def test_cuda_checkpoint_resumes_on_mps(tmp_path, tiny_tower, extractor, monkeypatch):
    """A tiny-tower live run killed at step 3 on the CPU, its checkpoint rewritten to look like an
    A100 one (a CUDA generator state, no MPS state, no device key, fused AdamW groups), resumed on
    MPS: same config hash, the stream continues at batch 3, every step matches the uninterrupted
    CPU run, and the new checkpoints carry MPS state."""
    cpu_tower = tower_copy(tiny_tower)  # no adapter hooks left over by other tests
    full_model = make_model(cpu_tower)
    try:
        live_trainer(tmp_path, "full", full_model, extractor, ListStream(make_batches(), name="toy")).run()
    finally:
        full_model.detach_adapters()
    _, loss_full = losses(tmp_path / "full" / "train.csv")

    model = make_model(cpu_tower)
    monkeypatch.setenv("INDICEAR_TEST_KILL_AT_STEP", "3")
    try:
        with pytest.raises(SystemExit):
            live_trainer(tmp_path, "moved", model, extractor, ListStream(make_batches(), name="toy")).run()
    finally:
        model.detach_adapters()
        monkeypatch.delenv("INDICEAR_TEST_KILL_AT_STEP")
    path = next((tmp_path / "moved").glob("ckpt-*.pt"))
    state = torch.load(path, weights_only=False)
    cpu_hash = state["config_hash"]
    state["rng"]["cuda"] = [torch.zeros(16, dtype=torch.uint8)]
    state["rng"].pop("mps", None)
    for key in ("device", "precision"):
        state.pop(key, None)
    for group in state["optimizer"]["param_groups"]:
        group["fused"] = True
    torch.save(state, path)

    tower = tower_copy(tiny_tower, "mps")
    model = make_model(tower).to("mps")
    messages = []
    stream = ListStream(make_batches(), name="toy")
    trainer = live_trainer(tmp_path, "moved", model, extractor, stream, device="mps")
    trainer.log = lambda m: messages.append(m)
    try:
        summary = trainer.run()
    finally:
        model.detach_adapters()
    assert trainer.hash == cpu_hash and summary["config_hash"] == cpu_hash
    assert summary["steps"] == 8 and stream.yielded == [3, 4, 5, 6, 7]
    assert any("CUDA generator state" in m for m in messages) and any("MPS generator seeded" in m for m in messages)
    _, loss_moved = losses(tmp_path / "moved" / "train.csv")
    for a, b in zip(loss_moved, loss_full, strict=True):
        assert a == pytest.approx(b, rel=2e-3, abs=2e-4)
    final = torch.load(max((tmp_path / "moved").glob("ckpt-*.pt")), weights_only=False)
    assert final["step"] == 8 and final["device"] == "mps" and "mps" in final["rng"]


# ---------------------------------------------------------------- fp16 on MPS


def test_half_safe_tower_caps_the_large_constants(tiny_tower):
    tower = tower_copy(tiny_tower)
    half_safe_tower(tower)
    limit = float(torch.finfo(torch.float16).max)
    clips = [m.gradient_clipping for m in tower.modules() if hasattr(m, "gradient_clipping")]
    assert clips and max(clips) <= limit
    assert tower.config.attention_invalid_logits_value >= -limit


@mps
def test_fp16_step_on_mps_is_finite_and_close_to_fp32(tmp_path, tiny_tower, extractor):
    tower = tower_copy(tiny_tower, "mps")
    batch = prepare(make_batches()[1], text_encoder, pin=False)
    results = {}
    for precision in ("fp32", "fp16"):
        model = make_model(tower).to("mps")
        try:
            trainer = live_trainer(tmp_path, precision, model, extractor, ListStream([]), device="mps",
                                   precision=precision)
            model.train()
            logs: dict = {}
            trainer.micro_step(batch, 1.0, logs)
            gradient = grads(model)
            if trainer.scaler is not None:
                gradient = gradient / trainer.scaler.get_scale()
            results[precision] = (logs, gradient)
        finally:
            model.detach_adapters()
    (logs32, g32), (logs16, g16) = results["fp32"], results["fp16"]
    assert all(np.isfinite(v) for v in logs16.values())
    assert logs16["loss"] == pytest.approx(logs32["loss"], rel=1e-2)
    assert float(torch.nn.functional.cosine_similarity(g16, g32, dim=0)) > 0.99


def test_frame_bucket_padding_is_exact_for_valid_frames(tiny_tower, extractor):
    """Padding the log-mel to a frame bucket (compiled MPS path) leaves every valid layer-6 frame
    unchanged: no right context, causal convolution."""
    tower = tower_copy(tiny_tower)
    staged = StagedTower(tower, 6)
    featurizer = Featurizer.from_extractor(extractor, "cpu", torch.float32)
    waveforms, lengths = pad_waveforms([tone_audio(s, i) for i, s in enumerate((1.1, 0.5, 1.7))])
    batch = {"waveforms": waveforms, "lengths": lengths}
    plain, plain_lengths = LiveLower(staged, featurizer)(batch)
    padded, padded_lengths = LiveLower(staged, featurizer, frame_bucket=48)(batch)
    assert padded.shape[1] % 48 == 0 and padded.shape[1] >= plain.shape[1]
    assert torch.equal(plain_lengths, padded_lengths)
    for row, n in enumerate(plain_lengths.tolist()):
        assert torch.allclose(padded[row, :n], plain[row, :n], atol=1e-5)
        assert torch.all(padded[row, n:] == 0)
    features, mask = pad_features(torch.ones(2, 5, 3), torch.ones(2, 5, dtype=torch.bool), 2)
    assert features.shape == (2, 8, 3) and mask[:, 5:].sum() == 0 and features[:, 5:].sum() == 0


@mps
def test_compiled_fp16_step_on_mps_matches_eager(tmp_path, tiny_tower, extractor):
    """The resume configuration on a tiny tower: fp16 autocast + torch.compile of the tower layers
    and heads, frame buckets, dynamic batch. Two batch shapes in one bucket compile once; losses
    and gradients agree with the eager float32 step."""
    import torch._dynamo

    torch._dynamo.reset()
    raw = make_batches()
    batches = [prepare(raw[1], text_encoder, pin=False), prepare(raw[4], text_encoder, pin=False)]
    results = {}
    for name, precision, compiled in (("eager", "fp32", False), ("compiled", "fp16", True)):
        tower = tower_copy(tiny_tower, "mps")
        model = make_model(tower).to("mps")
        try:
            trainer = live_trainer(tmp_path, name, model, extractor, ListStream([]), device="mps", precision=precision)
            trainer.cfg.compile_heads = compiled
            trainer.live_lower.frame_bucket = 48 if compiled else 0
            if trainer.scaler is not None:  # random tiny weights: a 2^14 scale overflows (training would skip and back off)
                trainer.scaler = torch.amp.GradScaler("mps", init_scale=1.0)
            trainer.compile_heads()
            model.eval()  # no SpecAugment: the same function in both modes
            out = []
            for batch in batches:
                logs: dict = {}
                trainer.micro_step(batch, 1.0, logs)
                gradient = grads(model) / (trainer.scaler.get_scale() if trainer.scaler is not None else 1.0)
                model.zero_grad(set_to_none=True)
                out.append((logs, gradient))
            results[name] = out
            assert model.staged.dynamic_batch is compiled
        finally:
            model.detach_adapters()
    torch._dynamo.reset()
    for (eager_logs, eager_grad), (logs, gradient) in zip(results["eager"], results["compiled"], strict=True):
        assert all(np.isfinite(v) for v in logs.values())
        assert logs["loss"] == pytest.approx(eager_logs["loss"], rel=1e-2)
        assert float(torch.nn.functional.cosine_similarity(gradient, eager_grad, dim=0)) > 0.99


def test_host_memory_guard_adds_pressure_signals_with_a_cooldown():
    from indicear.runtime.gpu import HostMemoryGuard

    now = [0.0]
    state = {"swap_gib": 5.0, "pressure": "normal", "free_gib": 20.0}
    messages = []
    guard = HostMemoryGuard(8.0, every=60, log_every=300, cooldown=600, log=messages.append,
                            probe=lambda: dict(state), clock=lambda: now[0])
    assert not guard.check() and "swap 5.0 GiB" in messages[0]
    state["swap_gib"] = 9.0
    now[0] = 30.0
    assert not guard.check()  # sampled once a minute
    now[0] = 61.0
    assert guard.check()  # swap past the limit: add a split
    now[0] = 200.0
    assert not guard.check()  # cooldown
    state.update(swap_gib=4.0, pressure="critical")
    now[0] = 700.0
    assert guard.check()
