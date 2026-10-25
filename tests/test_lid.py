"""Language identification heads: shapes, loss values, the run-CTC overfit, zero-weight
independence of the transcript decoder, the LID-only variant, persistence, spans and metrics."""

import csv
import math
import tempfile
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from indicear import lid, metrics, phonetic
from indicear.cache import Layer6Cache
from indicear.encoder import StagedTower
from indicear.lid import (
    FRAME_CLASSES,
    IGNORE,
    LID_LANGS,
    NON_SPEECH,
    NUM_LANGS,
    FrameLid,
    LidConfig,
    LidHeads,
    LidTargets,
    frame_lid_loss,
    make_targets,
)
from indicear.model import DesignA, load_lid_config, resolve_variant
from indicear.trainer import Example, TrainConfig, Trainer

RANGES = [(4, 20), (20, 40), (40, 50)]
VOCAB = 50
HI, EN, UR = LID_LANGS.index("hi") + 1, LID_LANGS.index("en") + 1, LID_LANGS.index("ur") + 1


def test_languages_and_classes():
    assert LID_LANGS == ("bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur")
    assert NUM_LANGS == 12 and FRAME_CLASSES == 13 and NON_SPEECH == 0


def test_head_shapes_and_distributions():
    torch.manual_seed(0)
    heads = LidHeads(32, LidConfig(layers=(6, 9, 12), hidden=16))
    states = {k: torch.randn(3, 20, 32) for k in (6, 9, 12)}
    out = heads(states, torch.tensor([20, 11, 1]))
    assert out["utt_logits"].shape == (3, NUM_LANGS) and out["utt_logits"].dtype == torch.float32
    assert out["frame_log_probs"].shape == (3, 20, FRAME_CLASSES)
    assert torch.allclose(out["frame_log_probs"].exp().sum(-1), torch.ones(3, 20), atol=1e-5)
    probs = lid.utterance_probs(out["utt_logits"])
    assert len(probs) == 3 and set(probs[0]) == set(LID_LANGS) and math.isclose(sum(probs[0].values()), 1, rel_tol=1e-5)
    # Padding never changes a row's utterance logits.
    padded = {k: torch.cat([v[1:2, :11], torch.randn(1, 9, 32)], 1) for k, v in states.items()}
    again = heads(padded, torch.tensor([11]))
    assert torch.allclose(again["utt_logits"][0], out["utt_logits"][1], atol=1e-5)
    assert sum(heads.utterance.mix.layer_weights().values()) == pytest.approx(1.0)


def test_make_targets():
    targets = make_targets(
        ["hi", "none", "hi", "en", "hi+en", "ta"],
        [None, None, ["hi", "hi", "en", "xx", "en", "hi"], ["en", "en"], None, []],
    )
    assert targets.utterance.tolist() == [HI - 1, IGNORE, HI - 1, EN - 1, IGNORE, LID_LANGS.index("ta")]
    assert targets.frame.tolist() == [HI, NON_SPEECH, IGNORE, EN, IGNORE, LID_LANGS.index("ta") + 1]
    assert targets.runs[2].tolist() == [HI, EN, HI]  # collapsed, unknown "xx" skipped
    assert [len(r) for r in targets.runs] == [0, 0, 3, 0, 0, 0]
    assert lid.collapse_runs(["hi", "hi", "en", "hi", "hi"]) == ["hi", "en", "hi"]


def test_balance_weights():
    weights = lid.balance_weights({"hi": 100, "en": 25, "ur": 4})
    by_lang = dict(zip(LID_LANGS, weights, strict=True))
    assert by_lang["ur"] > by_lang["en"] > by_lang["hi"]
    assert sum(by_lang[x] for x in ("hi", "en", "ur")) / 3 == pytest.approx(1.0)
    assert by_lang["bn"] == pytest.approx(by_lang["ur"])  # absent: the largest weight


def test_loss_values_on_toy_examples():
    uniform = torch.zeros(2, NUM_LANGS)
    targets = torch.tensor([0, 5])
    assert lid.utterance_lid_loss(uniform, targets, 0.1).item() == pytest.approx(math.log(12), rel=1e-5)
    sharp = torch.full((2, NUM_LANGS), -30.0)
    sharp[0, 0] = sharp[1, 5] = 30.0
    assert lid.utterance_lid_loss(sharp, targets, 0.0).item() == pytest.approx(0.0, abs=1e-6)
    eps = 0.1  # label smoothing: (1 - eps) * 0 + eps * mean over classes of the NLL
    assert lid.utterance_lid_loss(sharp, targets, eps).item() == pytest.approx(eps * 60 * 11 / 12, rel=1e-3)
    assert lid.utterance_lid_loss(uniform, torch.tensor([IGNORE, IGNORE]), 0.1).item() == 0.0
    weight = torch.ones(NUM_LANGS)
    weight[5] = 3.0
    wrong = torch.zeros(2, NUM_LANGS)
    wrong[1, 5] = 2.0  # row 1 is easy, row 0 is uniform-ish
    plain = F.cross_entropy(wrong, targets, reduction="none")
    assert lid.utterance_lid_loss(wrong, targets, 0.0, weight).item() == pytest.approx(
        ((plain[0] + 3 * plain[1]) / 4).item(), rel=1e-5)

    log_probs = torch.full((2, 6, FRAME_CLASSES), -math.log(FRAME_CLASSES))
    lengths = torch.tensor([6, 4])
    dense = LidTargets(torch.tensor([3, 3]), torch.tensor([HI, NON_SPEECH]), [torch.zeros(0, dtype=torch.long)] * 2)
    assert frame_lid_loss(log_probs, lengths, dense).item() == pytest.approx(math.log(FRAME_CLASSES), rel=1e-5)
    switched = LidTargets(torch.tensor([3, 3]), torch.tensor([IGNORE, HI]),
                          [torch.tensor([HI, EN, HI]), torch.zeros(0, dtype=torch.long)])
    expected_ctc = F.ctc_loss(log_probs[:1].transpose(0, 1), torch.tensor([HI, EN, HI]), torch.tensor([6]),
                              torch.tensor([3]), blank=0, reduction="sum") / 6
    expected = (math.log(FRAME_CLASSES) + expected_ctc.item()) / 2
    assert frame_lid_loss(log_probs, lengths, switched).item() == pytest.approx(expected, rel=1e-5)


def toy_frames(plan, seed=0, noise=0.1, dim=16):
    torch.manual_seed(seed)
    means = {HI: torch.randn(dim), EN: torch.randn(dim), UR: torch.randn(dim)}
    return torch.stack([torch.stack([means[c] + noise * torch.randn(dim) for c in row]) for row in plan])


def fit(features, lengths, targets, steps=150):
    head = FrameLid(features.shape[-1], (6,), hidden=32)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    history = []
    for _ in range(steps):
        loss = frame_lid_loss(head({6: features}), lengths, targets)
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    with torch.no_grad():
        return history, head({6: features})


SWITCHED = [[HI] * 10 + [EN] * 8 + [HI] * 12, [UR] * 9 + [EN] * 11 + [UR] * 10]


def test_run_ctc_loss_finite_and_decreases_on_a_toy_overfit():
    """Code-switched rows alone, no frame alignment: the run CTC is finite and overfits."""
    features = toy_frames(SWITCHED)
    targets = make_targets(["hi", "ur"], [["hi", "en", "en", "hi"], ["ur", "en", "ur"]])
    history, _ = fit(features, torch.tensor([30, 30]), targets)
    assert history[-1] < 0.05 * history[0]


def test_dense_and_run_targets_together_recover_the_spans():
    """With monolingual rows (dense CE) beside the code-switched ones (run CTC), the frame
    posteriors stay dense, so the greedy decode recovers each run and its frames exactly."""
    plan = SWITCHED + [[HI] * 30, [EN] * 30, [UR] * 30]
    features = toy_frames(plan, noise=0.3)
    targets = make_targets(["hi", "ur", "hi", "en", "ur"], [["hi", "en", "en", "hi"], ["ur", "en", "ur"], None, None, None])
    history, log_probs = fit(features, torch.full((5,), 30), targets)
    assert history[-1] < 0.01 * history[0]
    assert lid.frame_spans(log_probs[0], 30) == [("hi", 0, 10), ("en", 10, 18), ("hi", 18, 30)]
    assert lid.frame_spans(log_probs[1], 30) == [("ur", 0, 9), ("en", 9, 20), ("ur", 20, 30)]


def test_frame_spans_words_and_merge():
    path = [0, HI, HI, 0, HI, EN, EN, 0, 0, HI, 0, 0]
    log_probs = torch.full((len(path), FRAME_CLASSES), -20.0)
    for t, c in enumerate(path):
        log_probs[t, c] = 0.0
    assert lid.frame_spans(log_probs, 12) == [("hi", 1, 5), ("en", 5, 7), ("hi", 9, 10)]
    assert lid.frame_spans(log_probs, 12, min_frames=2) == [("hi", 1, 5), ("en", 5, 7)]
    assert lid.frame_spans(log_probs, 4) == [("hi", 1, 3)]
    # Words from CTC emissions: tokens at frames 1, 2 (same word), 5, 9; word starts 1, 5, 9.
    words = lid.word_frames([10, 11, 12, 13], [1, 2, 5, 9], [True, False, True, True], 12)
    assert words == [(1, 5), (5, 9), (9, 10)]
    word_langs, spans = lid.merge_with_words(log_probs, words)
    assert word_langs == ["hi", "en", "hi"]
    assert spans == [("hi", 1, 5), ("en", 5, 9), ("hi", 9, 10)]
    assert lid.utterance_spans(log_probs, 12, [10, 11, 12, 13], [1, 2, 5, 9], [True, False, True, True]) == spans
    assert lid.utterance_spans(log_probs, 12, [], [], []) == lid.frame_spans(log_probs, 12)


# ---------------------------------------------------------------- model integration


def batch_inputs(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    h6 = torch.randn(3, 40, 64, generator=generator)
    lengths = torch.tensor([40, 30, 12])
    tokens = [torch.tensor([5, 25, 41, 7]), torch.tensor([30, 31]), torch.tensor([], dtype=torch.long)]
    units = [torch.tensor([3, 4, 5, 1, 6]), torch.tensor([7, 8]), torch.tensor([], dtype=torch.long)]
    targets = make_targets(["hi", "en", "none"], [["hi", "en", "hi", "en"], None, None])
    return h6, lengths, tokens, units, targets


def build(tower, name: str, lid_config: LidConfig | None = None):
    staged = StagedTower(tower, 6)
    variant, config = resolve_variant(name, lid_config)
    torch.manual_seed(0)
    return DesignA(staged, variant, VOCAB, RANGES, len(phonetic.UNITS), lid=config)


def train_steps(model, steps: int = 3, lid_targets: bool = True):
    h6, lengths, tokens, units, targets = batch_inputs()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.train()
    torch.manual_seed(1)  # SpecAugment and CR-CTC draw from the global RNG
    for _ in range(steps):
        out = model.losses(h6, lengths, tokens, units, lid=targets if lid_targets else None)
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()
    model.eval()
    return model.decode(h6, lengths)


@pytest.mark.parametrize("name", ["v1", "v3", "v5", "v6"])
@pytest.mark.parametrize("detach", [True, False])
def test_zero_weight_lid_leaves_the_decoder_unchanged(tiny_tower, name, detach):
    plain = build(tiny_tower, name)
    try:
        reference = train_steps(plain)
    finally:
        plain.detach_adapters()
    zero = LidConfig(utterance_weight=0.0, frame_weight=0.0, detach=detach)
    with_lid = build(tiny_tower, f"{name}+lid", zero)
    try:
        decoded = train_steps(with_lid)
        assert "lid" in decoded and decoded["lid"]["utt_logits"].shape == (3, NUM_LANGS)
    finally:
        with_lid.detach_adapters()
    assert torch.equal(decoded["final_log_probs"], reference["final_log_probs"])
    assert decoded["ctc"] == reference["ctc"]
    for key in reference["inter"]:
        assert torch.equal(decoded["inter"][key], reference["inter"][key])
    if "rnnt" in reference:
        assert decoded["rnnt"] == reference["rnnt"]


def test_lid_losses_add_to_the_transcript_loss(tiny_tower):
    model = build(tiny_tower, "v3+lid", LidConfig(utterance_weight=0.5, frame_weight=0.25))
    try:
        model.eval()  # no SpecAugment: the two calls see identical inputs
        h6, lengths, tokens, units, targets = batch_inputs()
        with_lid = model.losses(h6, lengths, tokens, units, lid=targets)
        without = model.losses(h6, lengths, tokens, units)
        assert "lid_utt" not in without and torch.isfinite(with_lid["lid_frame"])
        expected = without["loss"] + 0.5 * with_lid["lid_utt"] + 0.25 * with_lid["lid_frame"]
        assert with_lid["loss"].item() == pytest.approx(expected.item(), rel=1e-6)
        model.train()
        model.zero_grad()
        model.losses(h6, lengths, tokens, units, lid=targets)["loss"].backward()
        assert all(p.grad is None for p in tiny_tower.parameters())
        assert model.lid.utterance.proj.weight.grad is not None and model.lid.frame.mlp[0].weight.grad is not None
    finally:
        model.detach_adapters()


def test_lid_only_variant(tiny_tower):
    model = build(tiny_tower, "lid", LidConfig(layers=(6, 9), utterance_weight=1.0, frame_weight=1.0))
    assert model.final is None and model.adapters is None and model.inter is None
    assert set(model.head_modules()) == {"lid"}
    h6, lengths, tokens, units, targets = batch_inputs()
    model.train()
    out = model.losses(h6, lengths, tokens, units, lid=targets)
    assert set(out) == {"lid_utt", "lid_frame", "loss"} and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert all(p.grad is None for p in tiny_tower.parameters())
    model.eval()
    decoded = model.decode(h6, lengths)
    assert "ctc" not in decoded and len(decoded["lid"]["spans"]) == 3
    with pytest.raises(ValueError, match="LID targets"):
        model.losses(h6, lengths, tokens, units)
    # Layer 6 only: the upper stack is never run.
    calls = []
    handle = tiny_tower.layers[6].register_forward_hook(lambda *_: calls.append(1))
    try:
        only6 = build(tiny_tower, "lid", LidConfig(layers=(6,)))
        only6.losses(h6, lengths, tokens, units, lid=targets)
    finally:
        handle.remove()
    assert not calls


def test_resolve_variant():
    variant, config = resolve_variant("v4+lid")
    assert variant.name == "v4" and config == LidConfig()
    assert resolve_variant("v2") == (resolve_variant("v2")[0], None)
    variant, config = resolve_variant("lid")
    assert variant.frozen_upper and config.only
    with pytest.raises(ValueError):
        resolve_variant("v9+lid")


def test_save_load_with_lid_heads(tiny_tower):
    config = LidConfig(class_weights=tuple(float(i + 1) for i in range(12)), layers=(6, 9, 12))
    model = build(tiny_tower, "v3+lid", config)
    try:
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            assert load_lid_config(directory) == config
            state = {k: v.clone() for k, v in model.state_dict().items()}
            for parameter in model.parameters():
                if parameter.requires_grad:
                    torch.nn.init.normal_(parameter)
            model.load(directory)
            for key, value in model.state_dict().items():
                assert torch.equal(value, state[key]), key
            assert any(k.startswith("lid.") for k in state)
    finally:
        model.detach_adapters()


# ---------------------------------------------------------------- trainer: kill and resume with LID heads


def make_cache_and_examples(dim: int):
    generator = torch.Generator().manual_seed(1)
    examples, index, rows = [], {}, []
    offset = 0
    langs = ["hi", "en", "ta", "ur", "none", "hi"]
    for i in range(14):
        frames = 10 + (i * 7) % 23
        key = f"utt{i}"
        rows.append(torch.randn(frames, dim, generator=generator))
        index[key] = (offset, frames)
        offset += frames
        tokens = torch.randint(4, VOCAB, (max(1, frames // 6),), generator=generator)
        units = torch.randint(2, len(phonetic.UNITS), (max(1, frames // 4),), generator=generator)
        lang = langs[i % len(langs)]
        word_langs = ("hi", "en", "hi") if i % 5 == 0 else None
        examples.append(Example((key,), tokens, units, frames, lang, word_langs=word_langs))
    cache = Layer6Cache(torch.cat(rows).to(torch.bfloat16), None, index)
    return cache, examples


def run(tmp_path, name, tower, kill_at=None, monkeypatch=None):
    model = build(tower, "v3+lid", LidConfig(utterance_weight=0.3, frame_weight=0.3, detach=False))
    cache, examples = make_cache_and_examples(model.staged.hidden_size)
    config = TrainConfig(epochs=2, frame_budget=70, log_every=1, eval_every=0, warmup_steps=2,
                         ckpt_every_steps=1000, ckpt_every_seconds=1e9, compile_heads=False, num_buckets=3)
    trainer = Trainer(model, cache, examples, config, tmp_path / name, device="cpu", dtype=torch.float32,
                      log=lambda _m: None)
    try:
        if kill_at is not None:
            monkeypatch.setenv("INDICEAR_TEST_KILL_AT_STEP", str(kill_at))
        return trainer.run(), model
    finally:
        if monkeypatch is not None:
            monkeypatch.delenv("INDICEAR_TEST_KILL_AT_STEP", raising=False)
        model.detach_adapters()


def rows(path):
    return list(csv.DictReader(path.open()))


def test_kill_and_resume_with_lid_heads(tmp_path, tiny_tower, monkeypatch):
    (full, full_model) = run(tmp_path, "full", tiny_tower)
    full_rows = rows(tmp_path / "full" / "train.csv")
    assert "lid_utt" in full_rows[0] and all(math.isfinite(float(r["lid_frame"])) for r in full_rows)
    with pytest.raises(SystemExit):
        run(tmp_path, "resumed", tiny_tower, kill_at=full["steps"] // 2 + 1, monkeypatch=monkeypatch)
    checkpoint = torch.load(next((tmp_path / "resumed").glob("ckpt-*.pt")), weights_only=False)
    assert any(k.startswith("lid.") for k in checkpoint["model"])
    (resumed, resumed_model) = run(tmp_path, "resumed", tiny_tower)
    resumed_rows = rows(tmp_path / "resumed" / "train.csv")
    assert resumed["steps"] == full["steps"]
    for key in ("loss", "lid_utt", "lid_frame"):
        assert [float(r[key]) for r in resumed_rows] == pytest.approx([float(r[key]) for r in full_rows], rel=1e-4, abs=1e-5)
    for (key, a), b in zip(full_model.lid.state_dict().items(), resumed_model.lid.state_dict().values(), strict=True):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), key


# ---------------------------------------------------------------- metrics


def test_lid_report():
    refs = ["hi", "hi", "ur", "ur", "mr", "pa", "pa", "en", "none"]
    hyps = ["hi", "ur", "hi", "ur", "hi", "hi", "pa", "en", "hi"]
    seconds = [1.0, 3.0, 3.0, 6.0, 12.0, 25.0, 1.5, 4.0, 2.0]
    report = metrics.lid_report(refs, hyps, seconds)
    assert report["utterances"] == 8 and report["accuracy"] == 50.0
    assert report["confusion"]["hi"] == {"hi": 1, "ur": 1}
    assert report["pairs"]["hi/ur"]["hi->ur"] == 1 and report["pairs"]["hi/ur"]["ur->hi"] == 1
    assert report["pairs"]["hi/mr"]["mr->hi"] == 1 and report["pairs"]["hi/mr"]["mr_as_hi_rate"] == 100.0
    assert report["pairs"]["pa/hi"]["pa->hi"] == 1 and report["pairs"]["pa/hi"]["pa_as_hi_rate"] == 50.0
    # per-language F1: hi p=1/4 r=1/2 -> 1/3; ur p=1/2 r=1/2; mr 0; pa p=1 r=1/2 -> 2/3; en 1
    assert report["macro_f1"] == pytest.approx(100 * (1 / 3 + 1 / 2 + 0 + 2 / 3 + 1) / 5, abs=0.02)
    assert report["by_duration"]["0-2s"] == {"utterances": 2, "accuracy": 100.0}
    assert report["by_duration"][">20s"] == {"utterances": 1, "accuracy": 0.0}
    assert metrics.lid_report(["none"], ["hi"])["accuracy"] is None


def test_lid_span_report():
    untimed = [[{"lang": "hi"}, {"lang": "en"}]]
    assert metrics.lid_span_report(untimed, [[("hi", 0, 10), ("en", 10, 20)]])["span_accuracy"] == 100.0
    swapped = metrics.lid_span_report(untimed, [[("en", 0, 10), ("hi", 10, 20)]])
    assert swapped["span_accuracy"] == 50.0 and swapped["sequence_accuracy"] == 0.0
    timed = [[{"lang": "hi", "start": 0.0, "end": 2.0}, {"lang": "en", "start": 2.0, "end": 4.0}]]
    good = metrics.lid_span_report(timed, [[("hi", 0, 45), ("en", 45, 100)]])
    assert good["span_accuracy"] == 100.0 and good["frame_accuracy"] == 95.0 and good["sequence_accuracy"] == 100.0
    late = metrics.lid_span_report(timed, [[("hi", 0, 80), ("en", 80, 100)]])
    assert late["span_accuracy"] == 50.0  # the English span is mostly labelled Hindi


def test_parameter_counts_at_full_width():
    counts = lid.parameter_counts(1024)
    assert counts["total"] == counts["utterance"] + counts["frame"]
    assert counts["utterance"] < 500_000 and counts["frame"] < 300_000
    lid_only = lid.parameter_counts(1024, replace(LidConfig(), layers=(6, 9, 12)))
    assert lid_only["total"] == counts["total"] + 2  # one more mixing weight per head


def test_lid_variants_decode_and_score(tiny_tower):
    """decode_items on the LID-only variant (no transcript) and on v3+lid (spans snapped to words);
    evaluate.score turns both into lid_report / lid_span_report numbers; quick_eval falls back to
    100 - LID accuracy for the LID-only variant."""
    from indicear.ablate import quick_eval
    from indicear.evaluate import decode_items, score
    from indicear.runtime.shards import RowCache

    class Tokenizer:
        def decode(self, ids):
            return " ".join(f"w{int(i)}" for i in ids)

        def word_starts(self, ids):
            return [True] * len(ids)

    staged = StagedTower(tiny_tower, 6)
    generator = torch.Generator().manual_seed(1)
    rows, index, offset = [], {}, 0
    items = [{"key": "a", "text": "मेरा order हो", "lang": "hi", "word_langs": ["hi", "en", "hi"], "seconds": 1.0},
             {"key": "b", "text": "hello there", "lang": "en", "seconds": 1.0},
             {"key": "n", "text": "", "lang": "none", "negative": True, "seconds": 1.0}]
    for item in items:
        rows.append(torch.randn(25, 64, generator=generator))
        index[item["key"]] = (offset, 25)
        offset += 25
    cache = RowCache(torch.cat(rows), None, index)
    for name in ("lid", "v3+lid"):
        variant, lid = resolve_variant(name)
        model = DesignA(staged, variant, 50, [(4, 20), (20, 40), (40, 50)], len(phonetic.UNITS), lid=lid)
        try:
            decoded = decode_items(model, cache, items, Tokenizer(), 200, "cpu", torch.float32)
        finally:
            model.detach_adapters()
        assert ("ctc" in decoded) == (name != "lid")
        assert all(lang in LID_LANGS for lang in decoded["lid"]["utt"])
        assert all(isinstance(spans, list) for spans in decoded["lid"]["spans"])
        scored = score(items, decoded)
        assert scored["lid"]["utterance"]["utterances"] == 2  # the negative is skipped
        assert scored["lid"]["spans"]["utterances"] == 1  # the one code-mixed row
        if name == "lid":
            assert scored["metrics"] == {}
            metric = quick_eval(model, cache, items[:2], Tokenizer(), 200, "cpu", torch.float32)
            assert 0.0 <= metric <= 100.0


def test_cache_train_with_lid_variants(tmp_path, tiny_tower, monkeypatch):
    """`ablate train` (cached lower) for "lid" and "v2+lid": LID config hashed only for LID runs,
    lid.json saved, LID metrics in the result; word_langs reach the examples."""
    import json

    import indicear.encoder
    from indicear import ablate
    from tests.test_cli import build_root

    monkeypatch.setattr(indicear.encoder, "load_tower", lambda *a, **k: tiny_tower)
    paths = build_root(tmp_path / "root")
    rows = [json.loads(line) for line in paths.manifest("train").read_text().splitlines()]
    rows[2]["word_langs"] = ["hi", "en", "hi"]
    paths.manifest("train").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    from indicear.cache import Layer6Cache
    from indicear.tokenizer import AggregateTokenizer

    tokenizer = AggregateTokenizer.load(paths.tok)
    cache = Layer6Cache.load(paths.cache, "cpu", log=lambda _m: None)
    examples, _ = ablate.make_examples(rows, cache, tokenizer, 1)
    assert examples[2].word_langs == ("hi", "en", "hi") and examples[0].word_langs is None
    ablate.main(["--root", str(paths.root), "train", "--variants", "lid,v2+lid", "--epochs", "1",
                 "--frame-budget", "120", "--eval-every", "2", "--no-compile"])
    for variant in ("lid", "v2+lid"):
        result = json.loads((paths.results / f"{variant}.json").read_text())
        assert result["lid"]["utterance"]["utterances"] > 0 and result["lid_config"] is not None
        assert (paths.runs / variant / "lid.json").exists()
    assert json.loads((paths.results / "lid.json").read_text())["metrics"] == {}
