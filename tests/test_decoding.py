"""Decode-time script constraints (indicear.decoding): synthetic log-prob cases for each rule, the
disabled flags reproducing the old decoders exactly, and round trips on the tiny tower."""

import json
import math

import pytest
import torch

from indicear import decoding, export, transcribe
from indicear.ablate import build_model
from indicear.decoding import (
    BARE,
    CONT,
    MARK,
    NEUTRAL,
    OFF,
    WORD,
    Constrainer,
    DecodeOptions,
    TokenTables,
    piece_kind,
)
from indicear.encoder import StagedTower
from indicear.lid import LID_LANGS
from indicear.model import ctc_greedy
from indicear.text import LANG_SCRIPT, SCRIPTS
from indicear.tokenizer import AggregateTokenizer

ORDER = ("deva", "taml", "knda", "latn")
PIECES = {
    "deva": ["▁स्थानीय", "ा", "न", "▁", "ाणियों", "ो"],
    "taml": ["▁ஆ", "▁சரி", "ி"],
    "knda": ["▁ಹಾಂ", "ಿಗೆ", "ಗಳು", "ಯ", "▁ಪ್ರ", "▁ಹೇಳಿ", "▁"],
    "latn": ["▁family", "▁company", "▁royal", "d", "1", "▁stande", "▁"],
}
VOCAB = ["<blank>", "<pad>", "<bos>", "<eos>"] + [p for s in ORDER for p in PIECES[s]]
ID = {(s, p): VOCAB.index(p, 4 + sum(len(PIECES[o]) for o in ORDER[: ORDER.index(s)])) for s in ORDER for p in PIECES[s]}
SCRIPT_OF = [0, -1, -1, -1] + [i + 1 for i, s in enumerate(ORDER) for _ in PIECES[s]]


def tables() -> TokenTables:
    return TokenTables(ORDER, SCRIPT_OF, [NEUTRAL] * 4 + [piece_kind(p) for s in ORDER for p in PIECES[s]])


def frames(*steps: dict) -> torch.Tensor:
    """Log-probs [T, V]: each step maps (script, piece) or "blank" to a probability; the remaining
    mass is spread over the other tokens (specials get none)."""
    rows = []
    for step in steps:
        probs = torch.full((len(VOCAB),), 0.0)
        for key, p in step.items():
            probs[0 if key == "blank" else ID[key]] = p
        rest = 1 - probs.sum()
        free = (probs == 0).nonzero().flatten()
        free = free[free > 3]
        probs[free] = rest / len(free)
        probs[1:4] = 0
        rows.append(probs.clamp_min(1e-12).log())
    return torch.stack(rows)


def decode(t: TokenTables, lp: torch.Tensor, options: DecodeOptions, utt: dict | None = None) -> list[int]:
    scripts = decoding.script_set(t.order, utt, options)
    blank, cat_lp, cat_id = t.reduce(lp)
    return decoding.constrained_greedy(t, blank, cat_lp, cat_id, options, scripts)[0]


def text(tokens: list[int]) -> str:
    return "".join(VOCAB[i] for i in tokens).replace("▁", " ").strip()


def valid(t: TokenTables, tokens: list[int]) -> bool:
    state = decoding.START
    for token in tokens:
        if not t.allows(state, token):
            return False
        state = t.next_state(state, token)
    return True


RULES = DecodeOptions(script_rules=True, lid_script_set=False)


def test_piece_kinds():
    assert [piece_kind(p) for p in ("▁है", "▁", "12", "ा", "ಿಗೆ", "ಗಳು", "d", "‍क")] == [
        WORD, BARE, NEUTRAL, MARK, MARK, CONT, CONT, MARK]
    assert piece_kind("<unk>", special=True) == NEUTRAL


@pytest.mark.parametrize("stem, tail, broken", [
    (("latn", "▁family"), ("knda", "ಿಗೆ"), "familyಿಗೆ"),
    (("latn", "▁royal"), ("deva", "ो"), "royalो"),
    (("knda", "▁ಪ್ರ"), ("deva", "ाणियों"), "ಪ್ರाणियों"),
    (("deva", "▁स्थानीय"), ("knda", "ಯ"), "स्थानीयಯ"),
])
def test_mid_word_script_switches_become_impossible(stem, tail, broken):
    t = tables()
    lp = frames({stem: 0.9}, {tail: 0.6, "blank": 0.3}, {"blank": 0.9})
    assert text(decode(t, lp, OFF)) == broken
    fixed = decode(t, lp, RULES)
    assert fixed == [ID[stem]] and valid(t, fixed)


def test_a_mark_after_its_own_script_and_latin_plus_full_akshara_stay_possible():
    t = tables()
    lp = frames({("latn", "▁company"): 0.9}, {"blank": 0.9}, {("knda", "ಗಳು"): 0.8}, {"blank": 0.9})
    assert decode(t, lp, RULES) == decode(t, lp, OFF) == [ID["latn", "▁company"], ID["knda", "ಗಳು"]]
    assert text(decode(t, lp, RULES)) == "companyಗಳು"
    # Same-script marks, digits and a bare "▁" before another script's akshara are untouched.
    lp = frames({("deva", "▁स्थानीय"): 0.9}, {("deva", "ा"): 0.8}, {("latn", "1"): 0.8}, {("deva", "▁"): 0.8},
                {("knda", "ಯ"): 0.8}, {("knda", "ಿಗೆ"): 0.8})
    assert decode(t, lp, RULES) == decode(t, lp, OFF)
    # The documented limitation: a Latin stem + a full Indic akshara cannot be told from a suffix.
    lp = frames({("latn", "▁stande"): 0.9}, {("deva", "न"): 0.8})
    assert text(decode(t, lp, RULES)) == "standeन"
    # Indic -> Latin mid-word needs a word boundary ("royalोd": "ो" and then "d" are both dropped).
    lp = frames({("latn", "▁royal"): 0.9}, {("deva", "ो"): 0.6, "blank": 0.3}, {("latn", "d"): 0.6, "blank": 0.3})
    assert text(decode(t, lp, OFF)) == "royalोd" and text(decode(t, lp, RULES)) == "royald"


def test_latin_split_turns_a_glued_latin_continuation_into_a_new_word():
    t = tables()
    # "स्थानीय" + "d": an English word glued to a Hindi one (a missed word boundary).
    lp = frames({("deva", "▁स्थानीय"): 0.9}, {("latn", "d"): 0.6, "blank": 0.3}, {"blank": 0.9})
    assert text(decode(t, lp, OFF)) == "स्थानीयd"
    split = decode(t, lp, RULES)
    assert split == [ID["deva", "▁स्थानीय"], ID["latn", "▁"], ID["latn", "d"]] and valid(t, split)
    assert text(split) == "स्थानीय d"
    assert text(decode(t, lp, DecodeOptions(script_rules=True, lid_script_set=False, latin_split=False))) == "स्थानीय"
    # Without rule (a) there is nothing to split; Indic -> Indic is still blocked.
    assert text(decode(t, lp, DecodeOptions(script_rules=False, lid_script_set=False))) == "स्थानीयd"
    lp = frames({("deva", "▁स्थानीय"): 0.9}, {("knda", "ಯ"): 0.6, "blank": 0.3})
    assert decode(t, lp, RULES) == [ID["deva", "▁स्थानीय"]]


def test_second_best_allowed_token_replaces_a_disallowed_one():
    t = tables()
    lp = frames({("latn", "▁family"): 0.9}, {("knda", "ಿಗೆ"): 0.5, ("knda", "ಗಳು"): 0.3, "blank": 0.15})
    assert text(decode(t, lp, RULES)) == "familyಗಳು"
    # The state is tracked across blanks and repeats.
    lp = frames({("knda", "▁ಪ್ರ"): 0.9}, {("knda", "▁ಪ್ರ"): 0.9}, {"blank": 0.9}, {"blank": 0.9},
                {("deva", "ाणियों"): 0.6, "blank": 0.3})
    assert decode(t, lp, RULES) == [ID["knda", "▁ಪ್ರ"]]


def test_utterance_script_set_masks_indic_bleed():
    t = tables()
    opts = DecodeOptions(script_rules=False, lid_script_set=True)
    # Kannada audio: the model prefers Tamil words, Kannada is second best.
    lp = frames({("taml", "▁ஆ"): 0.5, ("knda", "▁ಹಾಂ"): 0.4}, {"blank": 0.9},
                {("taml", "▁சரி"): 0.5, ("knda", "▁ಹೇಳಿ"): 0.4}, {"blank": 0.9})
    kn = dict.fromkeys(LID_LANGS, 0.01) | {"kn": 0.86, "ta": 0.04}
    assert text(decode(t, lp, OFF)) == "ஆ சரி"
    assert text(decode(t, lp, opts, kn)) == "ಹಾಂ ಹೇಳಿ"
    assert decoding.script_set(ORDER, kn, opts) == {ORDER.index("knda") + 1, ORDER.index("latn") + 1}
    # A second Indic language above the threshold stays allowed; top-k 1 drops it.
    both = kn | {"kn": 0.6, "ta": 0.3}
    assert text(decode(t, lp, opts, both)) == "ஆ சரி"
    assert text(decode(t, lp, DecodeOptions(False, True, lid_topk=1), both)) == "ಹಾಂ ಹೇಳಿ"
    # The top-1 language is always allowed, even below the threshold; Latin always is.
    flat = dict.fromkeys(LID_LANGS, 1 / 12) | {"ta": 0.10}
    assert ORDER.index("taml") + 1 in decoding.script_set(ORDER, flat, opts)
    # Languages sharing a script take one top-k slot: Hindi and Marathi (both deva) do not push
    # out Urdu when LID also finds it likely.
    hi_mr_ur = dict.fromkeys(LID_LANGS, 0.0) | {"hi": 0.40, "mr": 0.30, "ur": 0.25}
    assert decoding.script_set(SCRIPTS, hi_mr_ur, opts) == {SCRIPTS.index(s) + 1 for s in ("deva", "arab", "latn")}
    english = dict.fromkeys(LID_LANGS, 0.0) | {"en": 0.95, "kn": 0.05}
    assert decoding.script_set(ORDER, english, opts) == {ORDER.index("latn") + 1}
    # Hindi audio with a Kannada fragment: masked, while English inside Hindi is kept.
    lp = frames({("deva", "▁स्थानीय"): 0.9}, {("knda", "▁ಹಾಂ"): 0.5, "blank": 0.4}, {("latn", "▁company"): 0.8})
    hi = dict.fromkeys(LID_LANGS, 0.0) | {"hi": 0.9, "en": 0.1}
    assert text(decode(t, lp, opts, hi)) == "स्थानीय company"
    # Disabled rule or no LID: every script.
    assert decoding.script_set(ORDER, kn, OFF) is None and decoding.script_set(ORDER, None, opts) is None


def test_frame_lid_share_adds_scripts():
    opts = DecodeOptions(script_rules=False, lid_script_set=True, lid_frame_share=0.2)
    utt = dict.fromkeys(LID_LANGS, 0.0) | {"kn": 1.0}
    frame = torch.full((10, 13), -10.0)
    frame[:, 0] = 0.0
    frame[:4, 1 + LID_LANGS.index("kn")] = 1.0
    frame[4:7, 1 + LID_LANGS.index("ta")] = 1.0
    assert decoding.script_set(ORDER, utt, opts, frame) == {ORDER.index(s) + 1 for s in ("knda", "taml", "latn")}
    assert decoding.script_set(ORDER, utt, DecodeOptions(False, True), frame) == {3, 4}


def test_disabled_rules_reproduce_greedy_exactly_on_random_log_probs():
    t = tables()
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        logits = torch.randn(1, 40, len(VOCAB), generator=generator) * 3
        logits[..., 0] += 2
        lp = logits.log_softmax(-1)
        lp[..., 1:4] = -1e4
        old = ctc_greedy(lp, torch.tensor([40]))[0]
        assert decoding.constrained_greedy(t, *t.reduce(lp[0]), OFF, None) == old
        assert transcribe.ctc_collapse(lp[0].argmax(-1).tolist()) == old
        constrained = decoding.constrained_greedy(t, *t.reduce(lp[0]), RULES, None)
        assert valid(t, constrained[0])


def test_beam_search_respects_the_rules_and_is_unchanged_when_off():
    t = tables()
    lp = frames({("latn", "▁family"): 0.9}, {("knda", "ಿಗೆ"): 0.6, ("knda", "ಗಳು"): 0.2, "blank": 0.15},
                {"blank": 0.9}, {("latn", "▁company"): 0.9}, {("knda", "ಗಳು"): 0.8})
    blank, cat_lp, cat_id = t.reduce(lp)
    top_lp, top_ids = lp.topk(8, -1)
    old = transcribe.ctc_prefix_beam(blank.tolist(), top_lp.tolist(), top_ids.tolist(), 4)
    assert text(old[0]) == "familyಿಗೆ companyಗಳು"
    assert transcribe.ctc_prefix_beam(blank.tolist(), top_lp.tolist(), top_ids.tolist(), 4,
                                      decoding.prefix_rule(t, OFF)) == old
    c_lp, c_ids = decoding.beam_candidates(cat_lp, cat_id, top_lp, top_ids, t, None)
    new = transcribe.ctc_prefix_beam(blank.tolist(), c_lp, c_ids, 4, decoding.prefix_rule(t, RULES))
    assert text(new[0]) == "familyಗಳು companyಗಳು" and valid(t, new[0])
    generator = torch.Generator().manual_seed(1)
    for _ in range(5):
        lp = torch.randn(30, len(VOCAB), generator=generator).mul(3).log_softmax(-1)
        blank, cat_lp, cat_id = t.reduce(lp)
        top_lp, top_ids = lp.topk(6, -1)
        c_lp, c_ids = decoding.beam_candidates(cat_lp, cat_id, top_lp, top_ids, t, {3, 4})
        tokens, frames_ = transcribe.ctc_prefix_beam(blank.tolist(), c_lp, c_ids, 4, decoding.prefix_rule(t, RULES))
        tokens, _ = t.insert_breaks(tokens, frames_, RULES)  # as Transcriber.transcribe does
        assert valid(t, tokens) and all(t.token_script[x] in (3, 4) for x in tokens)


# ---------------------------------------------------------------- tiny tower round trips

TEXTS = ["मैं घर जा रहा हूँ", "ನಾನು ಮನೆಗೆ ಹೋಗುತ್ತೇನೆ", "hello world", "मैं office जा", "आज मौसम अच्छा है",
         "companyಗಳು ಬಂದವು", "நான் வீட்டுக்கு போகிறேன்"]


@pytest.fixture(scope="module")
def tok_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tok")
    AggregateTokenizer.train(TEXTS * 4, directory, {s: 60 for s in SCRIPTS})
    return directory


def test_tables_from_a_real_tokenizer(tok_dir):
    tokenizer = AggregateTokenizer.load(tok_dir)
    t = TokenTables.from_tokenizer(tokenizer)
    t.check(tokenizer.vocab_size, [tokenizer.script_range(s) for s in tokenizer.order])
    with pytest.raises(ValueError):
        t.check(tokenizer.vocab_size + 1, [tokenizer.script_range(s) for s in tokenizer.order])
    kinds = {tokenizer.piece(i): t.kinds[i] for i in range(4, tokenizer.vocab_size)}
    assert kinds["▁"] == BARE and any(k == MARK for k in kinds.values()) and any(k == WORD for k in kinds.values())
    # Every reference encoding is itself allowed by the rule (including "companyಗಳು").
    for sentence in TEXTS:
        assert valid(t, tokenizer.encode(sentence)), sentence


def test_design_a_decode_with_constraints(tiny_tower, tok_dir):
    tokenizer = AggregateTokenizer.load(tok_dir)
    torch.manual_seed(0)
    model = build_model("v4+lid", StagedTower(tiny_tower, 6), tokenizer)
    try:
        with torch.no_grad():
            model.final.script_head.bias[0] -= 8  # random heads: suppress blank so tokens come out
        model.eval()
        h6 = torch.randn(3, 60, 64)
        lengths = torch.tensor([60, 45, 20])
        plain = model.decode(h6, lengths)
        off = model.decode(h6, lengths, constraints=Constrainer.from_tokenizer(tokenizer, OFF))
        assert off["ctc"] == plain["ctc"] and "ctc_scripts" not in off
        on = model.decode(h6, lengths, constraints=Constrainer.from_tokenizer(tokenizer))
        t = TokenTables.from_tokenizer(tokenizer)
        assert torch.equal(on["lid"]["utt_logits"], plain["lid"]["utt_logits"])
        for row, (tokens, frames_) in enumerate(on["ctc"]):
            assert valid(t, tokens) and frames_ == sorted(frames_)
            allowed = on["ctc_scripts"][row]
            assert "latn" in allowed and {tokenizer.script_of(x) for x in tokens} <= set(allowed)
            utt = decoding.utt_probs_dict(plain["lid"]["utt_logits"][row])
            assert allowed == [s for s in tokenizer.order if s in
                               {tokenizer.order[i - 1] for i in decoding.script_set(tokenizer.order, utt, DecodeOptions())}]
        assert any(on["ctc"][r] != plain["ctc"][r] for r in range(3)), "the random head breaks the rules somewhere"
    finally:
        model.detach_adapters()


def test_transcriber_and_cli_round_trip(tmp_path, tiny_tower, tok_dir, monkeypatch, capsys):
    import soundfile as sf
    from test_export import make_run, tone
    from test_export import transcriber as open_transcriber

    from indicear import encoder
    from indicear.smoke import make_extractor

    run, _ = make_run(tmp_path / "runs", tiny_tower, tok_dir, "v4+lid")
    out = tmp_path / "models" / "v4+lid"
    export.export(run, tok_dir, out)
    audio = tone(4.0)
    with open_transcriber(out, tiny_tower) as model:
        old = model.transcribe(audio)
        assert model.transcribe(audio, constraints=OFF) == old
        outputs = model.frame_outputs(audio, tables=True)
        assert model.transcribe(audio, constraints=OFF, outputs=outputs)["text"] == old["text"]
        on = model.transcribe(audio, constraints=DecodeOptions())
        assert on == model.transcribe(audio, constraints=DecodeOptions(), outputs=outputs)
        allowed = on["constraints"]["allowed_scripts"]
        assert "latn" in allowed and all(w["script"] in allowed for w in on["words"])
        probs = on["language"]["probabilities"]
        assert LANG_SCRIPT[max(probs, key=probs.get)] in allowed
        hinted = model.transcribe(audio, constraints=DecodeOptions(), outputs=outputs, lang_hint="ta")
        assert set(hinted["constraints"]["allowed_scripts"]) == {"taml", "latn"}
        assert hinted["language"] == on["language"]  # the hint steers decoding only, never the reported LID
        old_beam = model.transcribe(audio, beam=4)
        assert model.transcribe(audio, beam=4, constraints=OFF) == old_beam
        beam_on = model.transcribe(audio, beam=4, constraints=DecodeOptions())
        assert all(w["script"] in beam_on["constraints"]["allowed_scripts"] for w in beam_on["words"])
    wav = tmp_path / "a.wav"
    sf.write(wav, audio, 16000)
    monkeypatch.setattr(encoder, "load_tower", lambda *_a, **_k: tiny_tower)
    monkeypatch.setattr(encoder, "load_extractor", make_extractor)
    common = ["--model", str(out), str(wav), "--json", "--device", "cpu"]
    capsys.readouterr()
    transcribe.main(common)
    default = json.loads(capsys.readouterr().out)["results"][0]
    transcribe.main([*common, "--script-rules", "off", "--lid-script-set", "off", "--exclusive-scripts", "off"])
    disabled = json.loads(capsys.readouterr().out)["results"][0]
    assert default["constraints"]["script_rules"] and default["constraints"]["lid_script_set"]
    assert default["text"] == on["text"]
    assert "constraints" not in disabled and disabled["text"] == old["text"]
    # decoding eval writes per-row transcripts for each decoder (the benchmark's rows)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    row = {"key": "tiny/hi/0", "audio": str(wav), "text": "नमस्ते", "lang": "hi", "seconds": len(audio) / 16000}
    (manifests / "eval_tiny_hi.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    decoding.main(["eval", "--model", str(out), "--manifests", str(manifests), "--out", str(tmp_path / "auto"),
                   "--baseline-out", str(tmp_path / "off"), "--hint-out", str(tmp_path / "hint"), "--device", "cpu"])
    for name, text in (("auto", default["text"]), ("off", old["text"])):
        written = [json.loads(line) for line in (tmp_path / name / "tiny_hi.rows.jsonl").read_text().splitlines()]
        assert written == [{"key": "tiny/hi/0", "lang": "hi", "hypothesis": text,
                            "detected_language": default["language"]["top"]}]
    assert (tmp_path / "hint" / "tiny_hi.rows.jsonl").is_file()


def test_exclusive_scripts_keep_one_of_devanagari_and_urdu():
    """Rule (c): Hindustani speech decoded partly in each script keeps only the script with more
    evidence (the sum of its best token's probability on frames where that beats blank)."""
    order = SCRIPTS
    deva, arab, latn = (order.index(s) + 1 for s in ("deva", "arab", "latn"))
    frames = 6
    blank = torch.full((frames,), math.log(0.2))
    cat = torch.full((frames, len(order), 3), -math.inf)
    cat[:4, deva - 1, 0] = math.log(0.7)  # four Devanagari frames
    cat[4:, arab - 1, 0] = math.log(0.7)  # two Urdu frames
    cat[2, latn - 1, 0] = math.log(0.6)
    kept = decoding.exclusive_scripts(order, {deva, arab, latn}, blank, cat, (("deva", "arab"),))
    assert kept == {deva, latn}
    assert decoding.exclusive_scripts(order, {arab, latn}, blank, cat, (("deva", "arab"),)) == {arab, latn}
    everything = decoding.exclusive_scripts(order, None, blank, cat, (("deva", "arab"),))
    assert everything == set(range(1, len(order) + 1)) - {arab}
    assert decoding.exclusive_scripts(order, None, blank, cat, ()) is None
    assert not decoding.OFF.active and decoding.DEFAULT.exclusive == (("deva", "arab"),)
