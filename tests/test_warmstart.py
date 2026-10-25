"""Warm start (indicear.warmstart): tokenizer extension, head expansion with bit-identical outputs for
the kept scripts, LR multipliers for the fresh rows, and the LR schedule options."""

import json
import math
import shutil

import pytest
import torch

from indicear import phonetic, warmstart
from indicear.encoder import StagedTower
from indicear.lid import LidConfig
from indicear.model import VARIANTS, DesignA
from indicear.smoke import make_tiny_tower
from indicear.text import SCRIPTS, normalize
from indicear.tokenizer import AggregateTokenizer
from indicear.trainer import RowLr, TrainConfig, Trainer

KEEP = ["deva", "latn", "taml", "knda"]
PILOT_TEXTS = [normalize(t) for t in [
    "मैं आज office जा रहा हूँ", "ನಾನು ಇಂದು ಕೆಲಸಕ್ಕೆ ಹೋಗುತ್ತೇನೆ", "நான் இன்று வேலைக்கு போகிறேன்",
    "hello world it's fine", "ನಾನು meeting ಗೆ ಹೋಗ್ತೀನಿ", "நான் office போறேன்", "यह किताब नई है",
]]
FULL_TEXTS = PILOT_TEXTS + [normalize(t) for t in [
    "আমি ভাত খাই", "আমরা আজ বাজারে যাব", "ਮੈਂ ਘਰ ਜਾ ਰਿਹਾ ਹਾਂ", "నేను ఇంటికి వెళ్తున్నాను",
    "ഞാൻ വീട്ടിലേക്ക് പോകുന്നു", "હું ઘરે જાઉં છું", "ମୁଁ ଘରକୁ ଯାଉଛି", "یہ ایک کتاب ہے",
]]


@pytest.fixture(scope="module")
def toks(tmp_path_factory):
    base = tmp_path_factory.mktemp("pilot-tok")
    AggregateTokenizer.train(PILOT_TEXTS * 3, base, {s: 110 for s in SCRIPTS})
    new = tmp_path_factory.mktemp("full-tok")
    AggregateTokenizer.extend(base, KEEP, FULL_TEXTS * 3, new, {s: 140 for s in SCRIPTS})
    return base, new


def build(tokenizer, variant="v4", lid=True, seed=0):
    staged = StagedTower(make_tiny_tower(), 6)
    torch.manual_seed(seed)
    ranges = [tokenizer.script_range(s) for s in tokenizer.order]
    return DesignA(staged, VARIANTS[variant], tokenizer.vocab_size, ranges, len(phonetic.UNITS),
                   lid=LidConfig() if lid else None)


@pytest.fixture(scope="module")
def pilot_run(toks, tmp_path_factory):
    """A 'finished pilot run': random but non-trivial weights (adapters and back-projections are
    zero-initialized, so perturb every tensor), saved like ablate's finish_run."""
    base, _ = toks
    tokenizer = AggregateTokenizer.load(base)
    model = build(tokenizer, seed=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    run = tmp_path_factory.mktemp("runs") / "v4+lid"
    model.save(run)
    shutil.copytree(base, run / "tok")
    (run / "result.json").write_text(json.dumps({"train": {"config_hash": "pilothash"}}))
    return run, model


def test_extend_keeps_models_and_order(toks):
    base, new = toks
    old, ext = AggregateTokenizer.load(base), AggregateTokenizer.load(new)
    assert old.order == ext.order == tuple(SCRIPTS)
    for script in KEEP:
        assert (base / f"{script}.model").read_bytes() == (new / f"{script}.model").read_bytes()
        assert old.models[script].get_piece_size() == ext.models[script].get_piece_size()
    grown = [s for s in SCRIPTS if s not in KEEP and
             ext.models[s].get_piece_size() != old.models[s].get_piece_size()]
    assert grown, "the new scripts were retrained"
    meta = json.loads((new / "aggregate.json").read_text())
    assert set(meta["extended_from"]["kept"]) == set(KEEP)
    assert ext.decode(ext.encode(FULL_TEXTS[7])) == FULL_TEXTS[7]
    assert warmstart.kept_scripts(ext, base) == KEEP


def outputs(model, h6, lengths):
    model.eval()
    with torch.no_grad(), model.adapters_attached():
        states = {}
        hidden, inter = model.encode(h6, lengths, states=states)
        return model.final(hidden), inter, model.lid(states, lengths)


def kept_columns(tokenizer):
    columns = list(range(4))
    for script in KEEP:
        a, b = tokenizer.script_range(script)
        columns += list(range(a, b))
    return torch.tensor(columns)


def suppressed(run, tokenizer, tmp_path):
    """The run with its never-seen scripts' class logits pushed down, as the trained pilot has them
    (it saw only deva, latn, taml and knda): their probability is below float32 resolution."""
    from safetensors.torch import load_file, save_file

    copy = tmp_path / "suppressed"
    shutil.copytree(run, copy)
    heads = load_file(str(copy / "heads.safetensors"))
    for script in tokenizer.order:
        if script not in KEEP:
            heads["final.script_head.bias"][tokenizer.order.index(script) + 1] = -60.0
    save_file(heads, str(copy / "heads.safetensors"))
    return copy


@pytest.mark.parametrize(("mode", "trained"), [("copy", False), ("copy", True), ("fresh", True), ("fresh", False)])
def test_expansion_is_bit_identical_on_kept_scripts(toks, pilot_run, tmp_path, mode, trained):
    base, new = toks
    run, _ = pilot_run
    old_tok, new_tok = AggregateTokenizer.load(base), AggregateTokenizer.load(new)
    if trained:
        run = suppressed(run, old_tok, tmp_path)
    pilot = build(old_tok, seed=9)
    pilot.load(run)
    model = build(new_tok, seed=5)
    source = warmstart.load_source(run)
    # Bit-identity of kept scripts needs a new-script bias far below float32 resolution (-40); the
    # training default (-2) trades that for a bias Adam can move (see indicear.warmstart).
    warm = warmstart.apply(model, source, old_tok, new_tok, KEEP, new_script_logits=mode, new_script_bias=-40.0)
    pilot.detach_adapters()
    model.detach_adapters()
    torch.manual_seed(3)
    h6 = torch.randn(3, 48, 64) * 2
    lengths = torch.tensor([48, 40, 17])
    old_lp, old_inter, old_lid = outputs(pilot, h6, lengths)
    new_lp, new_inter, new_lid = outputs(model, h6, lengths)
    old_cols, new_cols = kept_columns(old_tok), kept_columns(new_tok)
    if mode == "copy" or trained:
        # identical before any training: always with copied class rows; with fresh ones when the source
        # gave its unseen scripts no probability (a trained pilot). Bit-identical on most CPUs; the
        # log-softmax over a wider vocabulary may round the last bit differently on others.
        torch.testing.assert_close(old_lp[..., old_cols], new_lp[..., new_cols], rtol=0, atol=1e-5)
    else:
        # an untrained source spread probability over every script: dropping it renormalizes, so the
        # kept outputs shift by one constant per frame and every within-script distribution is exact
        real = [0, *range(4, len(new_cols))]  # pad, bos and eos are a constant -1e4
        shift = new_lp[..., new_cols[real]] - old_lp[..., old_cols[real]]
        assert torch.allclose(shift, shift[..., :1].expand_as(shift), atol=1e-4)
        for script in KEEP:
            a, b = old_tok.script_range(script)
            c, d = new_tok.script_range(script)
            assert torch.allclose(old_lp[..., a:b].log_softmax(-1), new_lp[..., c:d].log_softmax(-1), atol=1e-5)
    assert all(torch.equal(old_inter[k], new_inter[k]) for k in old_inter)
    assert all(torch.equal(old_lid[k], new_lid[k]) for k in old_lid if torch.is_tensor(old_lid[k]))
    new_scripts = [s for s in SCRIPTS if s not in KEEP]
    assert warm.summary["new_scripts"] == new_scripts and warm.summary["kept_scripts"] == KEEP
    head = model.final
    for script in new_scripts:
        cls = new_tok.order.index(script) + 1
        if mode == "fresh":
            assert torch.all(head.script_head.weight[cls] == 0) and head.script_head.bias[cls] == -40.0
            a, b = new_tok.script_range(script)
            assert torch.all(new_lp[..., a:b] < -30)  # carries no probability at step 0
        assert [cls, cls + 1] in warm.new_rows["final.script_head.weight"]
    rows = {tuple(r) for r in warm.new_rows["final.token_head.weight"]}
    for script in new_scripts:
        assert tuple(new_tok.script_range(script)) in rows or any(
            a <= new_tok.script_range(script)[0] and new_tok.script_range(script)[1] <= b for a, b in rows)
    tokens = [torch.tensor(new_tok.encode(PILOT_TEXTS[0]))]
    units = [torch.tensor(phonetic.encode(PILOT_TEXTS[0]))]
    model.train()
    with model.adapters_attached():
        loss = model.losses(h6[:1], lengths[:1], tokens, units)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    if mode == "fresh":  # fresh rows still learn: nonzero gradient on a new script's class row
        target = [torch.tensor(new_tok.encode(FULL_TEXTS[7]))]
        model.zero_grad()
        with model.adapters_attached():
            model.losses(h6[:1], lengths[:1], target, [torch.tensor(phonetic.encode(FULL_TEXTS[7]))])["loss"].backward()
        cls = new_tok.order.index("beng") + 1
        assert head.script_head.weight.grad[cls].abs().sum() > 0


def test_checkpoint_source_and_structure_check(toks, pilot_run, tmp_path):
    base, new = toks
    run, pilot = pilot_run
    ckpt = tmp_path / "ckpt-00000010.pt"
    torch.save({"model": pilot.state_dict(), "config": {"extra": {"variant": json.loads(
        (run / "variant.json").read_text())}}, "config_hash": "abc"}, ckpt)
    shutil.copytree(base, tmp_path / "tok")
    source = warmstart.load_source(ckpt)
    assert source.config_hash == "abc" and set(source.tensors) == set(pilot.state_dict())
    old_tok, new_tok = AggregateTokenizer.load(base), AggregateTokenizer.load(new)
    model = build(new_tok)
    warmstart.apply(model, source, old_tok, new_tok, KEEP)
    with pytest.raises(SystemExit, match="variant field"):
        warmstart.apply(build(new_tok, variant="v2"), source, old_tok, new_tok, KEEP)


def test_prepare_records_and_resumes_without_the_source(toks, pilot_run, tmp_path):
    _, new = toks
    run, _ = pilot_run
    new_tok = AggregateTokenizer.load(new)
    run_dir = tmp_path / "runs" / "v4+lid"
    warm = warmstart.prepare(run_dir, run, build(new_tok), new_tok)
    assert (run_dir / warmstart.INIT_FILE).exists() and warm.summary["source_config_hash"] == "pilothash"
    again = warmstart.prepare(run_dir, tmp_path / "gone", build(new_tok), new_tok)  # VM without the source
    assert again.summary == warm.summary and again.new_rows == warm.new_rows
    with pytest.raises(SystemExit):
        warmstart.prepare(run_dir, tmp_path / "gone", build(new_tok), new_tok, force_new_run=True)


def test_row_lr_equals_a_separate_parameter_at_mult_lr():
    torch.manual_seed(0)
    weight = torch.randn(10, 4)
    rows = [[6, 10]]
    shared = torch.nn.Linear(4, 10, bias=False)
    shared.weight.data.copy_(weight)
    old = torch.nn.Parameter(weight[:6].clone())
    fresh = torch.nn.Parameter(weight[6:].clone())
    opt_a = torch.optim.AdamW(shared.parameters(), lr=1e-2, weight_decay=0.1)
    opt_b = torch.optim.AdamW([{"params": [old], "lr": 1e-2}, {"params": [fresh], "lr": 3e-2}], weight_decay=0.1)
    row_lr = RowLr(shared, {"weight": rows}, 3.0)
    assert row_lr
    for step in range(5):
        grad = torch.randn(10, 4)
        shared.weight.grad = grad.clone()
        old.grad, fresh.grad = grad[:6].clone(), grad[6:].clone()
        snapshot = row_lr.before()
        opt_a.step()
        row_lr.after(snapshot)
        opt_b.step()
        assert torch.allclose(shared.weight[:6], old, atol=1e-7), step
        assert torch.allclose(shared.weight[6:], fresh, atol=1e-6), step


def test_param_groups_and_schedule(tmp_path, toks, pilot_run):
    _, new = toks
    run, _ = pilot_run
    new_tok = AggregateTokenizer.load(new)
    model = build(new_tok)
    warm = warmstart.prepare(tmp_path / "run", run, model, new_tok)
    warm.fresh_params = ["final.norm.weight"]  # pretend one whole parameter had no source tensor
    cfg = TrainConfig(peak_lr=6e-4, final_lr=5e-5, new_lr_mult=3.0, warmup_steps=300, max_steps=3000)
    trainer = Trainer.__new__(Trainer)
    trainer.model, trainer.cfg, trainer.warm, trainer.device = model, cfg, warm, torch.device("cpu")
    assert cfg.peak(67_045) == 6e-4 and TrainConfig().peak(80_000) == pytest.approx(1e-3 * math.sqrt(4.0))
    optimizer, scheduler = trainer.optimizer(cfg.peak(67_045), 3000)
    lrs = sorted({g["initial_lr"] for g in optimizer.param_groups})
    assert lrs == pytest.approx([6e-4, 1.8e-3])
    assert trainer.row_lr and trainer.row_lr.mult == 3.0
    fresh_group = [g for g in optimizer.param_groups if g["initial_lr"] == pytest.approx(1.8e-3)]
    assert any(p is model.final.norm.weight for g in fresh_group for p in g["params"])
    seen = []
    for _ in range(3000):
        seen.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert seen[299] == pytest.approx(6e-4) and seen[0] == pytest.approx(6e-4 / 300)
    assert seen[-1] == pytest.approx(5e-5, rel=1e-3)
    assert TrainConfig().semantic() == {k: v for k, v in TrainConfig().semantic().items()
                                        if k not in ("peak_lr", "final_lr", "new_lr_mult")}
    assert {"peak_lr", "final_lr", "new_lr_mult"} <= set(cfg.semantic())
