"""The loanword normaliser (indicear.loanwords): mining, rewriting and the packaged resources."""

import json

import pytest

from indicear import loanwords
from indicear.loanwords import LoanwordNormaliser, gloss_pairs, mine
from indicear.text import normalize

IV_ROWS = [
    "मेरा ऑर्डर [order] आ गया और डॉक्टर [doctor] ने कहा",
    "ऑर्डर [Order] कैंसल [cancel] करो <noise> कमरा साफ़ है",
    "ऑर्डर [order] फिर से <laughter> डॉक्टर [doctor] आए",
    "कमरा [camera] कमरा [camera] कमरा [camera] " + "कमरा बड़ा है " * 9,  # mostly unglossed: never rewritten
    "डॉक्टर [doctor] कल आएंगे [breathing] कैंसल [cancel] हुआ कैंसल [cancel]",
    "गवर्नमेंट स्कूल [government school] में",  # multi-word gloss: not a single-word pair
    "कारमें [car] गए कारमें [car] गए कारमें [car]",  # a suffix glued to the loanword: rejected
]


def write_resources(directory, lexicon, allow=None, block=None, params=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "lexicon.json").write_text(json.dumps({"langs": lexicon}, ensure_ascii=False))
    (directory / "allow.json").write_text(json.dumps(allow or {}, ensure_ascii=False))
    (directory / "block.json").write_text(json.dumps(block or {}, ensure_ascii=False))
    (directory / "params.json").write_text(json.dumps(params or {}))
    return directory


def test_gloss_pairs_skips_tags_and_multiword_glosses():
    pairs = gloss_pairs("ऑर्डर [order] <noise> कमरा [breathing] गवर्नमेंट स्कूल [government school]")
    assert pairs == [("ऑर्डर", "order"), ("कमरा", None), ("गवर्नमेंट", None), ("स्कूल", None)]


def test_mine_keeps_attested_loanwords_only():
    lexicon, stats = mine(IV_ROWS, "hi")
    assert {k: v["en"] for k, v in lexicon.items()} == {"ऑर्डर": "order", "डॉक्टर": "doctor", "कैंसल": "cancel"}
    assert stats["rejected"]["often_unglossed"] >= 1  # कमरा / camera
    assert stats["rejected"]["suffix_or_partial"] >= 1  # कारमें / car


@pytest.fixture
def normaliser(tmp_path):
    lexicon, _ = mine(IV_ROWS, "hi")
    directory = write_resources(tmp_path / "res", {"hi": lexicon}, allow={"hi": {"वाईफ़ाई": "wifi"}},
                                block={"hi": ["कैंसल"]})
    return LoanwordNormaliser.load(directory)


def test_rewrite_lexicon_allow_block(normaliser):
    text = normalize("डॉक्टर ने ऑर्डर कैंसल किया वाईफ़ाई बंद था", "hi")
    out, count = normaliser.rewrite(text, "hi")
    assert out == "doctor ने order कैंसल किया wifi बंद था"  # कैंसल is blocklisted
    assert count == 3
    assert out == normalize(out, "hi")  # still in normalized form


def test_native_lookalike_latin_and_other_langs_pass_through(normaliser):
    for text, lang in (("कमरा बड़ा है", "hi"), ("doctor ने order दिया", "hi"), ("the doctor", "en"),
                       ("ಡಾಕ್ಟರ್ ಬಂದರು", "kn"), ("डॉक्टर", "xx")):
        assert normaliser.rewrite(text, lang) == (text, 0)


def test_idempotent(normaliser):
    once, _ = normaliser.rewrite("डॉक्टर और ऑर्डर", "hi")
    assert normaliser.rewrite(once, "hi") == (once, 0)


def test_min_length(tmp_path):
    rows = ["टी [tea] टी [tea] टी [tea] टी [tea]"]  # a one-letter native word
    lexicon, stats = mine(rows, "hi")
    assert lexicon == {} and stats["rejected"]["too_short"] == 1


def test_fingerprint_stable_and_sensitive(tmp_path):
    lexicon, _ = mine(IV_ROWS, "hi")
    a = LoanwordNormaliser.load(write_resources(tmp_path / "a", {"hi": lexicon}))
    b = LoanwordNormaliser.load(write_resources(tmp_path / "b", {"hi": lexicon}))
    c = LoanwordNormaliser.load(write_resources(tmp_path / "c", {"hi": lexicon}, block={"hi": ["ऑर्डर"]}))
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()
    assert len(a.fingerprint()) == 64 and a.describe()["entries"] == {"hi": 3}


def test_word_list_path(tmp_path, monkeypatch):
    words = tmp_path / "words.tsv"
    words.write_text("# test list\ncomputer\t10\ncamera\t20\n")
    monkeypatch.setattr(loanwords, "ENGLISH_WORDS", words)
    directory = write_resources(tmp_path / "res", {}, params={"secondary": {"hi": True}})
    normaliser = LoanwordNormaliser.load(directory)
    assert normaliser.rewrite("कंप्यूटर चालू है", "hi") == ("computer चालू है", 1)
    assert normaliser.describe()["word_list"]


def test_packaged_resources():
    normaliser = LoanwordNormaliser.load()
    assert normaliser.rewrite("डॉक्टर ने कमरा देखा", "hi") == ("doctor ने कमरा देखा", 1)
    assert normaliser.rewrite("ಡಾಕ್ಟರ್ ಬಂದರು", "kn")[1] == 1
    assert normaliser.fingerprint() == LoanwordNormaliser.load().fingerprint()
    assert set(normaliser.lexicon) >= {"hi", "kn", "ta"}
