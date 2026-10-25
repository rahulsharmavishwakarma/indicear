"""Glue of the ablation runner with a tiny tower, a real tokenizer and an in-memory cache."""

import torch

from indicear import metrics, phonetic
from indicear.ablate import cmd_report, decode_items, make_examples, quick_eval, subset_per_lang
from indicear.cache import Layer6Cache
from indicear.common import write_json
from indicear.encoder import StagedTower
from indicear.model import VARIANTS, DesignA
from indicear.text import SCRIPTS
from indicear.tokenizer import AggregateTokenizer

TEXTS = ["मैं घर जा रहा हूँ", "ನಾನು ಮನೆಗೆ ಹೋಗುತ್ತೇನೆ", "hello world", "मैं office जा"]


def test_examples_decode_and_report(tmp_path, tiny_tower):
    tokenizer = AggregateTokenizer.train(TEXTS * 4, tmp_path / "tok", {s: 60 for s in SCRIPTS})
    items = [{"key": f"u{i}", "text": t, "lang": ["hi", "kn", "en", "hi"][i]} for i, t in enumerate(TEXTS)]
    items.append({"key": "short", "text": "मैं घर जा रहा हूँ", "lang": "hi"})
    frames = {"u0": 40, "u1": 40, "u2": 30, "u3": 35, "short": 2}
    rows, index, offset = [], {}, 0
    for key, n in frames.items():
        rows.append(torch.randn(n, 64))
        index[key] = (offset, n)
        offset += n
    cache = Layer6Cache(torch.cat(rows).to(torch.bfloat16), None, index)
    examples, dropped = make_examples(items, cache, tokenizer, views=1)
    assert len(examples) == 4 and dropped == {"ctc_infeasible:hi": 1}
    assert examples[0].units.tolist() == phonetic.encode(TEXTS[0])
    ranges = [tokenizer.script_range(s) for s in tokenizer.order]
    model = DesignA(StagedTower(tiny_tower, 6), VARIANTS["v4"], tokenizer.vocab_size, ranges, len(phonetic.UNITS))
    try:
        decoded = decode_items(model, cache, items[:4], tokenizer, 80, "cpu", torch.float32)
        assert len(decoded["ctc"]) == 4
        assert quick_eval(model, cache, items[:4], tokenizer, 80, "cpu", torch.float32) >= 0
        assert len(subset_per_lang(items, 1)) == 3
        report = metrics.report(items[:4], decoded["ctc"], metrics.train_vocabulary(TEXTS))
        assert set(report) == {"hi", "kn", "en"}
        results = tmp_path / "root" / "results"
        write_json(results / "v4.json", {"variant": {"name": "v4"}, "metrics": {"ctc": report}})

        class Args:
            root = str(tmp_path / "root")

        cmd_report(Args)
        assert "| v4 | ctc | hi |" in (results / "summary.md").read_text()
    finally:
        model.detach_adapters()
