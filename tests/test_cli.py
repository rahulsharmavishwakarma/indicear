"""`ablate train` end to end on CPU: a tiny tower stands in for the downloaded encoder."""

import json

import torch

from indicear import ablate
from indicear.cache import to_storage, write_part
from indicear.common import write_json, write_jsonl
from indicear.text import SCRIPTS
from indicear.tokenizer import AggregateTokenizer


def build_root(root):
    hi = ["मैं घर जा रहा हूँ", "आज मौसम अच्छा है", "मैं office जा"]
    en = ["hello world", "good morning", "see you soon"]
    train = [{"key": f"tr{i}", "text": t, "lang": "hi" if t in hi else "en", "seconds": 2.0}
             for i, t in enumerate(hi + en)]
    dev = [{"key": "dv0", "text": "मैं बाज़ार जा रहा हूँ", "lang": "hi", "seconds": 2.0},
           {"key": "dv1", "text": en[0], "lang": "en", "seconds": 2.0}]
    cs = [{"key": "cs0", "text": f"{hi[1]} {en[1]}", "lang": "hi+en", "seconds": 4.0,
           "spans": [{"lang": "hi", "script": "deva", "text": hi[1]}, {"lang": "en", "script": "latn", "text": en[1]}]}]
    neg = [{"key": "ng0", "text": "", "lang": "none", "seconds": 1.0, "negative": True}]
    negd = [{"key": "nd0", "text": "", "lang": "none", "seconds": 1.0, "negative": True}]
    paths = ablate.Paths(root)
    import numpy as np
    import soundfile as sf

    (paths.root / "audio").mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(train + dev):
        path = paths.root / "audio" / f"{row['key']}.flac"
        t = np.arange(32000) / 16000
        sf.write(path, (0.2 * np.sin(2 * np.pi * (200 + 40 * index) * t)).astype(np.float32), 16000)
        row["audio"] = str(path)
    for name, rows in {"train": train, "dev": dev, "cs_dev": cs, "neg_train": neg, "neg_dev": negd}.items():
        write_jsonl(paths.manifest(name), rows)
    AggregateTokenizer.train([r["text"] for r in train], paths.tok, {s: 60 for s in SCRIPTS})
    keys = [r["key"] for rows in (train, dev, cs, neg, negd) for r in rows]
    generator = torch.Generator().manual_seed(0)
    rows = [(k, *to_storage(torch.randn(40 if k != "cs0" else 80, 64, generator=generator), "bf16")) for k in keys]
    paths.cache.mkdir(parents=True)
    write_part(paths.cache, 0, rows, "bf16")
    write_json(paths.cache / "cache.json", {"layer": 6, "dim": 64, "storage": "bf16", "frame_rate": 25,
                                            "encoder": "tiny", "parts": 1})
    return paths


def test_train_writes_results_resumes_and_skips(tmp_path, tiny_tower, monkeypatch, capsys):
    import indicear.encoder

    monkeypatch.setattr(indicear.encoder, "load_tower", lambda *a, **k: tiny_tower)
    paths = build_root(tmp_path / "root")
    argv = ["--root", str(paths.root), "train", "--variants", "v1,v4", "--epochs", "2", "--frame-budget", "120",
            "--eval-every", "2", "--no-compile", "--drive-copy", str(tmp_path / "drive")]
    ablate.main(argv)
    for variant in ("v1", "v4"):
        result = json.loads((paths.results / f"{variant}.json").read_text())
        assert result["train"]["steps"] == result["train"]["total_steps"] > 0
        assert {"hi", "en", "cs:hi+en", "negatives"} <= set(result["metrics"]["ctc"])
        assert (paths.runs / variant / "heads.safetensors").exists()
        run = paths.runs / variant
        manifest = json.loads((run / "LATEST.json").read_text())
        assert manifest["status"] == "DONE" and (run / "DONE").exists()
        listed = {f["file"] for f in manifest["files"]}
        assert {"result.json", "heads.safetensors", "best.pt"} <= listed and manifest["checkpoint"]["file"] in listed
        assert "indicear.ablate" in (run / "command.sh").read_text() and "--variants " + variant in (run / "command.sh").read_text()
        assert (tmp_path / "drive" / "runs" / variant / "result.json").exists()  # optional Drive copy
    assert (paths.runs / "v4" / "adapters.safetensors").exists()
    assert not (paths.runs / "v1" / "adapters.safetensors").exists()
    capsys.readouterr()
    ablate.main(argv)
    assert "v1: results/v1.json exists, skipping" in capsys.readouterr().out
    ablate.main(["--root", str(paths.root), "report"])
    assert "| v4 | ctc | cs:hi+en |" in (paths.results / "summary.md").read_text()


class FakeExtractor:
    """Stands in for the Gemma 4 log-mel extractor: 100 frames per second, 128 bins."""

    def __call__(self, audios, sampling_rate, return_tensors, padding):
        frames = [max(4, len(a) // 160) for a in audios]
        longest = max(frames)
        features = torch.zeros(len(audios), longest, 128)
        mask = torch.zeros(len(audios), longest, dtype=torch.bool)
        for row, (audio, n) in enumerate(zip(audios, frames, strict=True)):
            energy = torch.tensor(audio[: n * 160].reshape(n, 160)).pow(2).mean(-1, keepdim=True)
            features[row, :n] = torch.log(energy + 1e-6).expand(n, 128)
            mask[row, :n] = True
        return {"input_features": features, "input_features_mask": mask}


def test_robustness_probe_runs_skips_and_summarizes(tmp_path, tiny_tower, monkeypatch):
    import transformers

    import indicear.encoder
    from indicear import robustness_probe

    monkeypatch.setattr(indicear.encoder, "load_tower", lambda *a, **k: tiny_tower)
    monkeypatch.setattr(transformers.AutoFeatureExtractor, "from_pretrained", lambda *a, **k: FakeExtractor())
    paths = build_root(tmp_path / "root")
    ablate.main(["--root", str(paths.root), "train", "--variants", "v1", "--epochs", "1",
                 "--frame-budget", "120", "--eval-every", "0", "--no-compile"])
    argv = ["--root", str(paths.root), "--variant", "v1", "--workers", "0",
            "--conditions", "clean,noise:10,babble:5,reverb:0.6,tel8k,speed:1.1,amr"]
    robustness_probe.main(argv)
    out = paths.results / "robustness" / "v1"
    clean = json.loads((out / "clean.json").read_text())
    assert set(clean["per_lang"]) == {"hi", "en"} and clean["noise"] == "synthetic colored noise"
    assert (out / "speed_1.1.json").exists()
    before = (out / "clean.json").stat().st_mtime_ns
    robustness_probe.main(argv)  # rerun: every condition already has a JSON
    assert (out / "clean.json").stat().st_mtime_ns == before
    summary = robustness_probe.summarize(out)
    assert "clean" in summary["cer"] and "noise:10" in summary["cer"]


def test_robustness_probe_with_gpu_features_matches_cpu_features(tmp_path, tiny_tower, monkeypatch, capsys):
    """`--features gpu` (torch log-mel, here on the CPU) gives the same hypotheses as `--features cpu`."""
    import transformers

    import indicear.encoder
    from indicear import robustness_probe
    from tests.test_features import make_extractor

    monkeypatch.setattr(indicear.encoder, "load_tower", lambda *a, **k: tiny_tower)
    monkeypatch.setattr(transformers.AutoFeatureExtractor, "from_pretrained", lambda *a, **k: make_extractor())
    paths = build_root(tmp_path / "root")
    ablate.main(["--root", str(paths.root), "train", "--variants", "v1", "--epochs", "1",
                 "--frame-budget", "120", "--eval-every", "0", "--no-compile"])
    results = {}
    for backend in ("cpu", "gpu"):
        variant_dir = paths.results / "robustness" / "v1"
        robustness_probe.main(["--root", str(paths.root), "--variant", "v1", "--workers", "0",
                               "--conditions", "clean,tel8k", "--features", backend])
        results[backend] = {c: json.loads((variant_dir / f"{c}.json").read_text())["per_lang"] for c in ("clean", "tel8k")}
        for path in variant_dir.glob("*.json"):
            path.unlink()
    assert results["cpu"] == results["gpu"]
    assert "GPU log-mel vs Hugging Face extractor" in capsys.readouterr().out
