from indicear import phonetic
from indicear.phonetic import UNITS, decode, encode, romanize, text_units, word_units


def test_shared_units_across_brahmic_scripts():
    # The same word in Devanagari, Gujarati and Bengali spelling maps to the same units.
    assert word_units("कमल") == word_units("કમલ") == ["k", "a", "m", "a", "l"]
    assert word_units("ভারত") == ["bh", "ā", "r", "a", "t"]
    assert word_units("తెలుగు") == ["t", "e", "l", "u", "g", "u"]


def test_inherent_vowel_virama_nukta_and_schwa():
    assert romanize("न") == "na"  # monosyllable keeps its vowel
    assert romanize("कमल") == "kamal"  # final schwa deleted in Devanagari
    assert romanize("ಕನ್ನಡ") == "kannaḍa"  # Kannada keeps the final vowel
    assert romanize("क़लम") == "qalam"
    assert romanize("தமிழ்") == "tamiḻ"
    assert romanize("ਪੱਕਾ") == "pakkā"  # Gurmukhi addak doubles the consonant
    assert romanize("അവൻ") == "avan"  # Malayalam chillu


def test_urdu_and_english_rules():
    assert romanize("کھانا") == "khānā"
    assert romanize("پاکستان") == "pākstān"
    assert romanize("phone") == "fōn"
    assert romanize("office") == "ofis"
    assert romanize("the") == "the"
    assert word_units("data") == ["ḍ", "a", "ṭ", "a"]  # Indian-English retroflex t/d


def test_sentence_units_and_round_trip():
    units = text_units("मैं office जा")
    assert units.count("|") == 2
    ids = encode("मैं office 2024")
    assert all(0 < i < len(UNITS) for i in ids)
    assert decode(ids) == "maiṁ ofis 2024"


def test_every_unit_is_unique_and_small_inventory():
    assert len(set(UNITS)) == len(UNITS)
    assert 60 <= len(UNITS) <= 300
    assert UNITS[0] == phonetic.BLANK_UNIT


def test_unicode_digits_map_to_ascii_units():
    from indicear import phonetic

    assert phonetic.text_units("m²") == phonetic.text_units("m2")
    assert phonetic.text_units("२०२४") == phonetic.text_units("2024")
    assert phonetic.text_units("x² २") == phonetic.text_units("x2 2")
    assert phonetic.encode("10³ km²")
