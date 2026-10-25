from indicear.text import LANG_SCRIPT, normalize, prepare_transcript, rewrite_gloss, word_script


def test_normalize_keeps_combining_marks_and_drops_punctuation():
    assert normalize("किताब।") == "किताब"
    assert normalize("தமிழ், மொழி!") == "தமிழ் மொழி"  # pulli (virama) survives
    assert normalize("ಕನ್ನಡ.") == "ಕನ್ನಡ"
    assert normalize("Hello, World!") == "hello world"


def test_normalize_nfc_zwnj_apostrophe_and_script_split():
    assert normalize("क़") == normalize("क़")  # nukta forms compare equal after NFC
    assert normalize("क्‌ष") == "क्ष"  # ZWNJ deleted, never a word break
    assert normalize("don't 'quoted'") == "don't quoted"
    assert normalize("facebookपर") == "facebook पर"
    assert normalize("یہ كتاب") == "یہ کتاب"  # Arabic kaf folded to Urdu keheh


def test_word_script_for_all_pilot_languages():
    samples = {"hi": "हिंदी", "mr": "मराठी", "bn": "বাংলা", "pa": "ਪੰਜਾਬੀ", "gu": "ગુજરાતી", "or": "ଓଡ଼ିଆ",
               "ta": "தமிழ்", "te": "తెలుగు", "kn": "ಕನ್ನಡ", "ml": "മലയാളം", "ur": "اردو", "en": "english"}
    for lang, word in samples.items():
        assert word_script(word) == LANG_SCRIPT[lang], lang
    assert word_script("2024") is None


def test_gloss_rewrite():
    assert rewrite_gloss("मैं कंप्यूटर [computer] पर") == ("मैं computer पर", 1)
    assert rewrite_gloss("नमस्ते {hello} दोस्त") == ("hello दोस्त", 1)  # Vaani braces
    assert rewrite_gloss("[noise] मैं जा रहा") == ("मैं जा रहा", 0)
    assert rewrite_gloss("मैं [laughter] जा") == ("मैं जा", 0)  # Latin tag, not a gloss
    assert rewrite_gloss("मैं [शोर] जा") == ("मैं जा", 0)  # non-Latin bracket dropped
    assert rewrite_gloss("[computer] चालू") == ("चालू", 0)  # no native token before it
    assert rewrite_gloss("hello [world]") == ("hello", 0)  # preceding token already Latin
    assert rewrite_gloss("बात <unk> [background_noise]") == ("बात", 0)


def test_prepare_transcript():
    assert prepare_transcript("मैं कंप्यूटर [Computer] पर।", "hi", gloss=True) == ("मैं computer पर", 1)
    assert prepare_transcript("ನಾನು [noise] ಬಂದೆ.", "kn") == ("ನಾನು ಬಂದೆ", 0)
