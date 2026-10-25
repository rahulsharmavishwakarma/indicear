"""Model packages and transcription: export -> package -> transcribe on the tiny random tower.

No download: the tower is the tests' tiny Gemma4 tower and the feature extractor is built from the
pinned preprocessor values (indicear.smoke)."""

import itertools
import json
import pathlib
import math
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from safetensors.torch import load_file

from indicear import encoder, export, transcribe
from indicear.ablate import build_model
from indicear.common import FRAME_RATE, SAMPLE_RATE
from indicear.encoder import StagedTower
from indicear.lid import LID_LANGS
from indicear.smoke import make_extractor
from indicear.text import SCRIPTS
from indicear.tokenizer import AggregateTokenizer
from indicear.transcribe import SAMPLES_PER_FRAME, Transcriber

TEXTS = ["मैं घर जा रहा हूँ", "ನಾನು ಮನೆಗೆ ಹೋಗುತ್ತೇನೆ", "hello world", "मैं office जा", "आज मौसम अच्छा है"]
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
BASH = "/bin/bash" if Path("/bin/bash").exists() else "bash"


@pytest.fixture(scope="module")
def tok_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tok")
    AggregateTokenizer.train(TEXTS * 4, directory, {s: 60 for s in SCRIPTS})
    return directory


def make_run(root: Path, tiny_tower, tok_dir: Path, variant: str, seed: int = 0) -> tuple[Path, dict]:
    """A run directory as sync_runs.sh leaves it: published artifacts, result.json, command.sh,
    and a checkpoint holding the same weights. Returns (run dir, model state_dict)."""
    run = root / variant
    tokenizer = AggregateTokenizer.load(tok_dir)
    torch.manual_seed(seed)
    model = build_model(variant, StagedTower(tiny_tower, 6), tokenizer)
    try:
        with torch.no_grad():  # adapters start as an identity; make them do something
            for name, parameter in model.named_parameters():
                if "adapters" in name and ".up." in name:
                    parameter.normal_(0, 0.02)
            # Suppress blank so the random heads emit tokens (words, timings, spans to check).
            for head in (model.final, model.transducer.output if model.transducer is not None else None):
                if head is not None:
                    (head.script_head if hasattr(head, "script_head") else head.proj).bias[0] -= 8
        model.save(run)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        extra = {"variant": asdict(model.config), "encoder": "google/embeddinggemma-2@x"}
        if model.lid_config is not None:
            extra["lid"] = asdict(model.lid_config)
        torch.save({"model": state, "optimizer": {}, "config": {"lr": 1e-3, "extra": extra},
                    "config_hash": "feedc0de12345678", "git": "abc123", "step": 40, "best_metric": 9.5},
                   run / "best.pt")
    finally:
        model.detach_adapters()
    metrics = {"hi": {"cer": 12.5, "wer": 30.1, "utterances": 10}, "en": {"cer": 8.0, "wer": 20.0, "utterances": 9},
               "all": {"cer": 10.0, "wer": 25.0, "utterances": 19}, "negatives": {"clips": 5, "false_output_rate": 0.0}}
    result = {"variant": asdict(model.config), "train": {"steps": 40, "config_hash": "feedc0de12345678",
                                                         "mixture": "pilot", "audio_hours": 1.5},
              "metrics": {"ctc": metrics}, "notes": "test run", "git": "abc123",
              "lid": {"utterance": {"accuracy": 91.0, "macro_f1": 88.0, "utterances": 19}}}
    (run / "result.json").write_text(json.dumps(result))
    (run / "command.sh").write_text("#!/usr/bin/env bash\nindicear indicear.ablate --root R train\n")
    return run, state


def tone(seconds: float, freq: float = 220.0, seed: int = 0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    noise = np.random.default_rng(seed).normal(0, 0.02, len(t))
    return (0.3 * np.sin(2 * np.pi * freq * t * (1 + 0.1 * np.sin(t))) + noise).astype(np.float32)


def transcriber(package: Path, tiny_tower, **kwargs) -> Transcriber:
    return Transcriber(package, device="cpu", tower=tiny_tower, extractor=make_extractor(), log=lambda _m: None,
                       **kwargs)


# ---------------------------------------------------------------- export and verify


def test_export_verify_transcribe_round_trip_with_lid(tmp_path, tiny_tower, tok_dir):
    run, _ = make_run(tmp_path / "runs", tiny_tower, tok_dir, "v4+lid")
    (run / "ckpt-00000040.pt").write_bytes(b"checkpoint bytes")  # must never reach the package
    out = tmp_path / "models" / "v4+lid"
    manifest = export.export(run, tok_dir, out)
    assert manifest["format"] == export.FORMAT and manifest["variant"] == "v4+lid" and manifest["decoder"] == "ctc"
    assert manifest["encoder"]["repo"] == "google/embeddinggemma-2"
    assert manifest["encoder"]["revision"] == "914f7f89142e33e77833254d9c9b90c3cef7303b"
    assert manifest["encoder"]["component"] == "audio_tower" and manifest["encoder"]["license"] == "Apache-2.0"
    assert manifest["training"] == {**manifest["training"], "config_hash": "feedc0de12345678", "git": "abc123",
                                    "mixture": "pilot", "steps": 40}
    assert manifest["eval"]["metrics"]["ctc"]["hi"]["cer"] == 12.5
    assert manifest["lid"]["languages"] == list(LID_LANGS)
    assert manifest["languages"] == ["en", "hi", "kn", "ta"]  # the pilot mixture's languages
    files = set(manifest["files"])
    assert {"heads.safetensors", "adapters.safetensors", "variant.json", "lid.json", "README.md",
            "tokenizer/aggregate.json", "training/result.json", "training/command.sh"} <= files
    assert not any(name.endswith(".pt") for name in files)
    card = (out / "README.md").read_text()
    for needle in ("Reverberation", "Hindi and Urdu", "Apache-2.0", "data-licenses.md", "| hi | 12.5 | 30.1 |"):
        assert needle in card
    assert export.verify(out) == []
    export.main(["--verify", str(out)])

    with transcriber(out, tiny_tower) as model:
        result = model.transcribe(tone(3.0))
    assert result["decoder"] == "ctc" and result["duration"] == 3.0
    assert result["frames"] == pytest.approx(3.0 * FRAME_RATE, abs=2)
    assert math.isclose(sum(result["language"]["probabilities"].values()), 1.0, abs_tol=1e-3)
    assert result["language"]["top"] in LID_LANGS
    assert result["spans"], "a +lid model reports language spans"
    assert result["words"] and result["text"], "blank is suppressed: the random model emits words"
    for span in result["spans"]:
        assert span["lang"] in LID_LANGS and 0 <= span["start"] < span["end"] <= 3.0 + 1 / FRAME_RATE
    for word in result["words"]:
        assert 0 <= word["start"] < word["end"] <= 3.0 + 1 / FRAME_RATE
        assert word["lang"] in LID_LANGS and word["script"] in SCRIPTS
    assert result["text"] == " ".join(w["word"] for w in result["words"])

    # Tampering is caught, and a corrupt package does not load.
    (out / "heads.safetensors").write_bytes((out / "heads.safetensors").read_bytes()[:-1] + b"\0")
    (out / "stray.pt").write_bytes(b"x")
    problems = export.verify(out)
    assert any("heads.safetensors: sha256 mismatch" in p for p in problems)
    assert any("stray.pt: not listed" in p for p in problems) and any("checkpoint" in p for p in problems)
    with pytest.raises(SystemExit):
        export.main(["--verify", str(out)])
    with pytest.raises(SystemExit, match="verification failed"):
        transcriber(out, tiny_tower)
    with pytest.raises(SystemExit, match="exists"):
        export.export(run, tok_dir, out)
    export.export(run, tok_dir, out, force=True)
    assert export.verify(out) == []


def test_checkpoint_export_matches_published_weights(tmp_path, tiny_tower, tok_dir):
    run, _ = make_run(tmp_path / "runs", tiny_tower, tok_dir, "v3+lid", seed=3)
    published = export.export(run, tok_dir, tmp_path / "published")
    for name in ("heads.safetensors", "adapters.safetensors", "variant.json"):
        (run / name).unlink()  # a run that has not finished: checkpoints only
    with pytest.raises(SystemExit, match="--checkpoint"):
        export.export(run, tok_dir, tmp_path / "nope")
    from_ckpt = export.export(run, tok_dir, tmp_path / "from-ckpt", checkpoint="best")
    assert from_ckpt["weights"] == {"checkpoint": "best.pt", "step": 40, "best_metric": 9.5}
    assert published["weights"] == {"published": True}
    for name in ("heads.safetensors", "adapters.safetensors"):
        a, b = load_file(str(tmp_path / "published" / name)), load_file(str(tmp_path / "from-ckpt" / name))
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    audio = tone(2.0, seed=5)
    with transcriber(tmp_path / "published", tiny_tower) as one:
        first = one.transcribe(audio)
    with transcriber(tmp_path / "from-ckpt", tiny_tower) as two:
        second = two.transcribe(audio)
    assert first["text"] == second["text"] and first["language"] == second["language"]


def test_export_rejects_a_mismatched_tokenizer_or_partial_sync(tmp_path, tiny_tower, tok_dir):
    run, _ = make_run(tmp_path / "runs", tiny_tower, tok_dir, "v2")
    other = tmp_path / "other-tok"
    AggregateTokenizer.train(["hello world"] * 4, other, {s: 40 for s in SCRIPTS})
    with pytest.raises(SystemExit, match="do not match the tokenizer"):
        export.export(run, other, tmp_path / "bad")
    assert not (tmp_path / "bad").exists() and not (tmp_path / ".bad.partial").exists()
    (run / "LATEST.json").write_text(json.dumps({"files": [{"file": "heads.safetensors", "sha256": "0" * 64}]}))
    with pytest.raises(SystemExit, match="incomplete sync"):
        export.export(run, tok_dir, tmp_path / "bad")


# ---------------------------------------------------------------- windows and long audio


def test_windows_tile_the_audio_on_the_frame_grid():
    assert transcribe.windows(1000) == [(0, 1000)]
    spans = transcribe.windows(70 * SAMPLE_RATE)
    assert spans[0] == (0, 30 * SAMPLE_RATE) and spans[-1][1] == 70 * SAMPLE_RATE
    assert all(a % SAMPLES_PER_FRAME == 0 for a, _ in spans)
    assert all(b - a <= 30 * SAMPLE_RATE for a, b in spans)
    assert all(nxt[0] < cur[1] for cur, nxt in itertools.pairwise(spans))  # overlapping
    assert [a for a, _ in spans] == [0, 28 * SAMPLE_RATE, 56 * SAMPLE_RATE]
    with pytest.raises(ValueError):
        transcribe.windows(10, window_seconds=31)


def test_long_audio_chunks_cover_the_whole_file(tmp_path, tiny_tower, tok_dir):
    run, _ = make_run(tmp_path / "runs", tiny_tower, tok_dir, "v2+lid")
    out = tmp_path / "pkg"
    export.export(run, tok_dir, out)
    audio = tone(10.3, seed=1)
    with transcriber(out, tiny_tower) as model:
        merged = model.frame_outputs(audio, window_seconds=4.0, overlap_seconds=1.0, batch_windows=2)
        single = model.frame_outputs(audio[: 4 * SAMPLE_RATE], window_seconds=4.0)
        result = model.transcribe(audio, window_seconds=4.0, overlap_seconds=1.0)
    expected = len(transcribe.windows(len(audio), 4.0, 1.0))
    assert merged["windows"] == expected == 4
    # Kept frames tile [0, frames): every window continues exactly where the previous one ended.
    position = 0
    for offset, lo, hi in merged["kept"]:
        assert offset + lo == position and hi > lo
        position = offset + hi
    assert position == merged["frames"] >= len(audio) // SAMPLES_PER_FRAME - 2
    assert merged["ctc"]["top_ids"].shape[0] == merged["frames"] == merged["frame_lid"].shape[0]
    # The first window's frames equal a single pass over that window.
    first = merged["kept"][0][2]
    assert torch.allclose(merged["ctc"]["blank"][:first], single["ctc"]["blank"][:first], atol=1e-5)
    assert result["frames"] == merged["frames"] and result["windows"] == 4
    assert all(w["end"] <= 10.3 + 1 / FRAME_RATE for w in result["words"])


# ---------------------------------------------------------------- decoders


def test_ctc_prefix_beam_sums_paths_that_greedy_misses():
    # p(a) = 0.45 then 0.35, the rest blank: greedy says "", but P("a") = 0.6425 > P("") = 0.3575.
    # The most likely path into "a" is (a, blank), so "a" is timed at frame 0.
    blank = [math.log(0.55), math.log(0.65)]
    top_lp = [[math.log(0.55), math.log(0.45)], [math.log(0.65), math.log(0.35)]]
    top_ids = [[0, 7]] * 2
    assert transcribe.ctc_collapse([0, 0]) == ([], [])
    tokens, frames = transcribe.ctc_prefix_beam(blank, top_lp, top_ids, beam=4)
    assert tokens == [7] and frames == [0]
    # A peaky sequence: beam search equals greedy, repeats separated by a blank stay separate.
    ids = [0, 5, 5, 0, 5, 9, 0]
    probs = np.full((7, 10), 1e-4)
    probs[np.arange(7), ids] = 1
    lp = np.log(probs / probs.sum(1, keepdims=True))
    order = np.argsort(-lp, axis=1)[:, :3]
    beam = transcribe.ctc_prefix_beam(lp[:, 0].tolist(), np.take_along_axis(lp, order, 1).tolist(), order.tolist(), 4)
    assert beam == transcribe.ctc_collapse(ids) == ([5, 5, 9], [1, 4, 5])


def test_transducer_stream_matches_one_pass_greedy(tiny_tower, tok_dir):
    tokenizer = AggregateTokenizer.load(tok_dir)
    torch.manual_seed(0)
    model = build_model("v6", StagedTower(tiny_tower, 6), tokenizer)
    try:
        enc = torch.randn(1, 30, 64)
        reference = model.transducer.greedy(enc, torch.tensor([30]))[0]
        stream = transcribe.TransducerStream(model.transducer)
        stream.feed(enc[0, :11], 0)
        stream.feed(enc[0, 11:], 11)
        assert stream.tokens == reference
        assert stream.frames == sorted(stream.frames) and all(0 <= f < 30 for f in stream.frames)
    finally:
        model.detach_adapters()


def test_v6_package_uses_the_transducer_and_beam_on_ctc(tmp_path, tiny_tower, tok_dir):
    run, _ = make_run(tmp_path / "runs", tiny_tower, tok_dir, "v6")
    out = tmp_path / "v6"
    manifest = export.export(run, tok_dir, out)
    assert manifest["decoder"] == "rnnt" and manifest["decoders"] == ["ctc", "rnnt"]
    audio = tone(2.5, seed=2)
    with transcriber(out, tiny_tower) as model:
        rnnt = model.transcribe(audio)
        ctc = model.transcribe(audio, decoder="ctc", beam=4)
    assert rnnt["decoder"] == "rnnt" and rnnt["beam"] == 1 and "language" not in rnnt
    assert ctc["decoder"] == "ctc" and ctc["beam"] == 4
    for result in (rnnt, ctc):
        assert result["words"] and all(0 <= w["start"] < w["end"] <= 2.5 + 1 / FRAME_RATE for w in result["words"])


# ---------------------------------------------------------------- CLI, audio formats, devices


def test_cli_json_on_wav_flac_and_ffmpeg_formats(tmp_path, tiny_tower, tok_dir, monkeypatch, capsys):
    run, _ = make_run(tmp_path / "runs", tiny_tower, tok_dir, "v4+lid")
    out = tmp_path / "pkg"
    export.main(["--run", str(run), "--tok", str(tok_dir), "--out", str(out)])
    monkeypatch.setattr(encoder, "load_tower", lambda *_a, **_k: tiny_tower)
    monkeypatch.setattr(encoder, "load_extractor", make_extractor)
    wav, flac = tmp_path / "a.wav", tmp_path / "b.flac"
    stereo_8k = np.stack([tone(1.5)[::2], tone(1.5, seed=9)[::2]], axis=1)
    sf.write(wav, stereo_8k, 8000)  # resampled and down-mixed on load
    sf.write(flac, tone(2.0, seed=4), SAMPLE_RATE)
    audio = [wav, flac]
    if shutil.which("ffmpeg"):
        m4a = tmp_path / "c.m4a"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(flac), "-c:a", "aac", str(m4a)], check=True)
        audio.append(m4a)
    capsys.readouterr()
    transcribe.main(["--model", str(out), "--json", "--device", "cpu", *map(str, audio)])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["device"] == "cpu" and data["model"]["variant"] == "v4+lid"
    assert [r["file"] for r in data["results"]] == list(map(str, audio))
    assert data["results"][0]["duration"] == pytest.approx(1.5, abs=0.01)
    for result in data["results"]:
        assert "spans" in result and "probabilities" in result["language"] and result["timing"]["rtf"] > 0
    assert "RTF" in captured.err
    transcribe.main(["--model", str(out), "--device", "cpu", str(flac)])
    text = capsys.readouterr().out
    assert text.strip() == data["results"][1]["text"]


def test_a_bundled_encoder_is_loaded_from_the_package(tmp_path, tiny_tower, tok_dir, monkeypatch):
    """A package with encoder/ (export --with-encoder) loads the tower and extractor from it; the
    manifest lists its files, so verify still covers them."""
    from indicear.runtime.checkpoint import sha256_file

    run, _ = make_run(tmp_path / "runs", tiny_tower, tok_dir, "v4+lid")
    out = tmp_path / "pkg"
    export.export(run, tok_dir, out)
    (out / export.ENCODER_DIR).mkdir()
    config = out / export.ENCODER_DIR / "config.json"
    config.write_text("{}", encoding="utf-8")
    manifest = json.loads((out / export.MANIFEST).read_text(encoding="utf-8"))
    manifest["encoder"]["bundled"] = export.ENCODER_DIR
    manifest["files"]["encoder/config.json"] = {"sha256": sha256_file(config), "size": config.stat().st_size}
    (out / export.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    assert export.verify(out) == []
    seen = []
    monkeypatch.setattr(encoder, "load_tower", lambda _d, _t, path=None: seen.append(("tower", path)) or tiny_tower)
    monkeypatch.setattr(encoder, "load_extractor", lambda path=None: seen.append(("extractor", path)) or make_extractor())
    with Transcriber(out, device="cpu", log=lambda _m: None):
        pass
    assert seen == [("extractor", out / "encoder"), ("tower", out / "encoder")]
    config.write_text("{ }", encoding="utf-8")
    assert any("encoder/config.json" in problem for problem in export.verify(out))


def test_verify_ignores_hub_metadata_only(tmp_path, tiny_tower, tok_dir):
    """A package downloaded from the Hub has .gitattributes (and .cache/ with --local-dir); anything
    else unlisted is still a problem."""
    run, _ = make_run(tmp_path / "runs", tiny_tower, tok_dir, "v4+lid")
    out = tmp_path / "pkg"
    export.export(run, tok_dir, out)
    (out / ".gitattributes").write_text("*.safetensors filter=lfs\n", encoding="utf-8")
    (out / ".cache" / "huggingface").mkdir(parents=True)
    (out / ".cache" / "huggingface" / "x.metadata").write_text("x", encoding="utf-8")
    assert export.verify(out) == []
    (out / "extra.txt").write_text("x", encoding="utf-8")
    assert export.verify(out) == ["extra.txt: not listed in the manifest"]


def test_a_custom_card_and_its_images_are_packaged_and_hashed(tmp_path, tiny_tower, tok_dir):
    run, _ = make_run(tmp_path / "runs", tiny_tower, tok_dir, "v4+lid")
    card = tmp_path / "report.md"
    card.write_text("# Report\n\n![fig](figures/a.png)\n<audio controls src=\"figures/s.mp3\"></audio>\n", encoding="utf-8")
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "a.png").write_bytes(b"\x89PNG fake")
    (figures / "make.py").write_text("print()", encoding="utf-8")  # not an image: not copied
    (figures / "s.mp3").write_bytes(b"ID3 fake")  # an audio sample the card links to
    (figures / "b.png").write_bytes(b"\x89PNG fake")  # not linked from the card: not copied
    out = tmp_path / "pkg"
    manifest = export.export(run, tok_dir, out, model_name="my-model", card=card, card_dirs=[figures])
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\nlicense: apache-2.0") and "![fig](figures/a.png)" in readme
    assert manifest["name"] == "my-model" and {"figures/a.png", "figures/s.mp3"} <= set(manifest["files"])
    assert not (out / "figures" / "make.py").exists() and not (out / "figures" / "b.png").exists()
    assert export.verify(out) == []
    card.write_text("{{TABLE}}", encoding="utf-8")
    with pytest.raises(SystemExit, match="placeholders"):
        export.export(run, tok_dir, tmp_path / "pkg2", card=card)


def test_resolve_package_takes_a_directory_or_a_hub_repo_id(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo, revision=None: calls.append((repo, revision)) or str(tmp_path / "snap"))
    assert transcribe.resolve_package(tmp_path) == tmp_path
    assert transcribe.resolve_package("user/model") == tmp_path / "snap"
    assert transcribe.resolve_package("user/model@abc123") == tmp_path / "snap"
    assert transcribe.resolve_package("./not-there/pkg") == pathlib.Path("./not-there/pkg")
    assert calls == [("user/model", None), ("user/model", "abc123")]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_mps_matches_cpu_on_the_tiny_tower(tmp_path, tok_dir):
    from indicear.smoke import make_tiny_tower

    run, _ = make_run(tmp_path / "runs", make_tiny_tower(), tok_dir, "v4+lid")
    out = tmp_path / "pkg"
    export.export(run, tok_dir, out)
    audio = tone(6.0, seed=7)
    results = {}
    for device in ("cpu", "mps"):
        tower = make_tiny_tower()  # own copy: moving the shared fixture to MPS would affect other tests
        with Transcriber(out, device=device, tower=tower, extractor=make_extractor(), log=lambda _m: None) as model:
            assert model.device.type == device
            results[device] = model.frame_outputs(audio, window_seconds=4.0, overlap_seconds=1.0)
    cpu, mps = results["cpu"], results["mps"]
    assert cpu["kept"] == mps["kept"]
    assert torch.allclose(cpu["ctc"]["blank"], mps["ctc"]["blank"], atol=1e-3)
    assert torch.allclose(cpu["frame_lid"], mps["frame_lid"], atol=1e-3)
    assert torch.allclose(cpu["utt_probs"], mps["utt_probs"], atol=1e-3)


def test_fetch_tok_copies_the_roots_tokenizer(tmp_path, tok_dir):
    root = tmp_path / "vm-root"
    shutil.copytree(tok_dir, root / "tok")
    dest = tmp_path / "mac" / "tok-pilot"
    out = subprocess.run([BASH, str(SCRIPTS_DIR / "fetch_tok.sh"), "local", str(root), str(dest)],
                         capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 0, out.stderr + out.stdout
    assert f"{len(SCRIPTS)} scripts" in out.stdout and not (dest / ".incoming").exists()
    assert AggregateTokenizer.load(dest).vocab_size == AggregateTokenizer.load(tok_dir).vocab_size
    missing = subprocess.run([BASH, str(SCRIPTS_DIR / "fetch_tok.sh"), "local", str(tmp_path / "nowhere"),
                              str(tmp_path / "x")], capture_output=True, text=True, timeout=60, check=False)
    assert missing.returncode != 0 and "aggregate.json" in missing.stderr
