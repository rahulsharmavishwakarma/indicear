import pytest

from indicear.text import SCRIPTS, normalize
from indicear.tokenizer import BLANK, PAD, AggregateTokenizer, script_inventory

TEXTS = [normalize(t) for t in [
    "मैं आज office जा रहा हूँ", "ನಾನು ಇಂದು ಕೆಲಸಕ್ಕೆ ಹೋಗುತ್ತೇನೆ", "நான் இன்று வேலைக்கு போகிறேன்",
    "یہ ایک کتاب ہے", "hello world it's fine", "আমি ভাত খাই", "ਮੈਂ ਘਰ ਜਾ ਰਿਹਾ ਹਾਂ",
    "నేను ఇంటికి వెళ్తున్నాను", "ഞാൻ വീട്ടിലേക്ക് പോകുന്നു", "હું ઘરે જાઉં છું", "ମୁଁ ଘରକୁ ଯାଉଛି",
    "ನಾನು meeting ಗೆ ಹೋಗ್ತೀನಿ", "நான் office போறேன்",
]]


@pytest.fixture(scope="module")
def tok_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tok")
    AggregateTokenizer.train(TEXTS * 3, directory, {s: 120 for s in SCRIPTS})
    return directory


@pytest.fixture(scope="module")
def tokenizer(tok_dir):
    return AggregateTokenizer.load(tok_dir)


def test_disjoint_ranges_and_specials(tokenizer):
    ranges = [tokenizer.script_range(s) for s in tokenizer.order]
    assert ranges[0][0] == 4 and BLANK == 0 and PAD == 1
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:], strict=False))
    assert ranges[-1][1] == tokenizer.vocab_size
    scripts = tokenizer.token_scripts()
    assert scripts[:4] == [0, -1, -1, -1] and len(scripts) == tokenizer.vocab_size


@pytest.mark.parametrize("text", TEXTS)
def test_round_trip(tokenizer, text):
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_mixed_script_tokens_carry_their_script(tokenizer):
    ids = tokenizer.encode(normalize("ನಾನು meeting ಗೆ"))
    scripts = [tokenizer.script_of(i) for i in ids]
    assert scripts[0] == "knda" and "latn" in scripts and scripts[-1] == "knda"
    assert tokenizer.decode(ids) == "ನಾನು meeting ಗೆ"


def test_full_inventory_spells_unseen_characters(tokenizer):
    for script in ("mlym", "telu", "deva"):
        word = script_inventory(script)[-14:-10]  # characters absent from the corpus
        word = normalize(word)
        ids = tokenizer.encode(word)
        assert not any(tokenizer.is_unknown(i) for i in ids), script


def test_load_gives_identical_ids(tokenizer, tok_dir):
    again = AggregateTokenizer.load(tok_dir)
    assert again.vocab_size == tokenizer.vocab_size
    assert again.encode(TEXTS[-1]) == tokenizer.encode(TEXTS[-1])
