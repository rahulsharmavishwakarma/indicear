from indicear import phonetic
from indicear.markup import INDICVOICES_TAGS, clean
from indicear.text import SCRIPTS
from indicear.tokenizer import AggregateTokenizer


def test_vaani_gloss_rewrite_multiword_glued_and_standard_spelling():
    # Patterns copied from public Vaani transcripts (Kannada, Hindi).
    c = clean("ನಮ್ ಮಕ್ಕಳೆಲ್ಲ ಗವರ್ನಮೆಂಟ್ ಸ್ಕೂಳ್ಳಲೇ {government school} ಓದಿರದು. ಚೆನಾಗ್--", "kn", "vaani")
    assert c.text == "ನಮ್ ಮಕ್ಕಳೆಲ್ಲ government school ಓದಿರದು ಚೆನಾಗ್" and c.glosses == 1 and c.code_mixed
    assert c.notes["cutoff"] == 1
    assert clean("ಕಾರ್ {car}ಆಲ್ಲಿ ಹೋಗಿಲ್ಲ", "kn", "vaani").text == "car ಆಲ್ಲಿ ಹೋಗಿಲ್ಲ"
    assert clean("ಸೈಕಲ್{cycle} ಓಡಿಸ್ತಿದ್ರು", "kn", "vaani").text == "cycle ಓಡಿಸ್ತಿದ್ರು"
    c = clean("ग्राउंड {ground} मे बोहोत {बहुत} सारे यहाँ पे (पर) बच्चे", "hi", "vaani")
    assert c.text == "ground मे बोहोत सारे यहाँ पे बच्चे"  # spoken form kept, standard spelling dropped
    assert c.notes["standard_spelling"] == 2 and c.glosses == 1


def test_vaani_tags_and_unusable_rows():
    c = clean("<noise> एक नीले कलर {color} की शर्ट {shirt} है [breathing] <PAUSE> </noise>", "hi", "vaani")
    assert c.text == "एक नीले color की shirt है" and c.glosses == 2 and c.drop is None
    assert clean("<static_noise> बिल्डिंग {building} </static_noise>", "hi", "vaani").text == "building"
    assert clean("[unintelligible] नीला बादल", "hi", "vaani").drop == "unintelligible"
    assert clean("<noise></noise> [breathing]", "hi", "vaani").drop == "empty"
    unattached = clean("{car} चालू", "hi", "vaani")
    assert unattached.text == "चालू" and unattached.notes["gloss_unattached"] == 1


def test_indicvoices_glosses_tags_and_typos():
    # Patterns copied from IndicVoices `unsanitized_normalized` rows (kn, hi, ta valid).
    assert len(INDICVOICES_TAGS) == 89
    c = clean("<Persistent-noise-start> ನಾನು <tapping> ಸಾಮಗ್ರಿಗಳ ಆರ್ಡರ್ [order] ಶಿಪ್‌ [ship] <cough> ಆಗಿದೆಯಾ "
              "<Persistent-noise-end>", "kn", "indicvoices")
    assert c.text == "ನಾನು ಸಾಮಗ್ರಿಗಳ order ship ಆಗಿದೆಯಾ" and c.glosses == 2 and c.code_mixed
    assert c.word_langs == ["kn", "kn", "en", "en", "kn"]
    c = clean("హలో [uhh] అవునండి [uh-huh][umm] [breathinhg] [Persistent-noise-end] సార్‌", "te", "indicvoices")
    assert c.text == "హలో అవునండి సార్" and c.glosses == 0 and c.notes["tag"] == 5  # a misspelt tag is a tag
    assert clean("<unintelligible> टूरिस्ट [Tourist] मोटल [Motel]", "hi", "indicvoices").drop == "unintelligible"
    off = clean("मैं कंप्यूटर [computer] पर", "hi", "indicvoices", indicvoices_gloss=False)
    assert off.text == "मैं कंप्यूटर पर" and off.notes["unknown_bracket_latin"] == 1


def test_paren_gloss_and_native_reference():
    from indicear.markup import native_reference

    raw = "प्रतिनिधितव और वारंटी (Warranty) विक्रेता द्वारा एस्क्रो (Escrow) समझौता"
    c = clean(raw, "hi", "paren_gloss")
    assert c.text == "प्रतिनिधितव और warranty विक्रेता द्वारा escrow समझौता" and c.glosses == 2
    assert native_reference(raw, "hi", "paren_gloss") == "प्रतिनिधितव और वारंटी विक्रेता द्वारा एस्क्रो समझौता"
    assert native_reference("ब्लैक {black} कलर", "hi", "vaani") == "ब्लैक कलर"
    assert native_reference("hello", "en", "plain") is None


def test_plain_scripts_code_mixing_and_validation():
    assert clean("CHAPTER SIXTEEN I MIGHT", "en", "plain").text == "chapter sixteen i might"
    c = clean("इस tutorial में हम impress window के", "hi", "plain")
    assert c.code_mixed and c.drop is None
    assert clean("தமிழ் हिंदी", "ta", "plain").drop == "foreign_script:deva"
    assert clean("hello दोस्त", "en", "plain").drop == "foreign_script:deva"
    assert clean("।।", "hi", "plain").drop == "empty"
    assert clean(None, "hi").drop == "missing_text"
    assert phonetic.encode(clean("ਪੋਸਟ x² ਦੀ", "pa", "plain").text)  # the "²" crash stays fixed


def test_every_cleaned_text_tokenizes(tmp_path):
    texts = [clean(raw, lang, markup).text for raw, lang, markup in (
        ("ನಮ್ ಕಾರ್ {car}ಆಲ್ಲಿ", "kn", "vaani"), ("यहाँ पे (पर) बुक्स {books}", "hi", "vaani"),
        ("హలో [uhh] సార్‌", "te", "indicvoices"), ("x² ਦੀ", "pa", "plain"), ("Don't STOP", "en", "plain"))]
    tokenizer = AggregateTokenizer.train(texts, tmp_path / "tok", {s: 60 for s in SCRIPTS})
    for text in texts:
        assert tokenizer.encode(text) and phonetic.encode(text)


def test_word_languages_for_lid():
    from indicear.markup import word_languages

    c = clean("ब्लैक {black} कलर {color} की कार 2 है", "hi", "vaani")
    assert c.text.split() == ["black", "color", "की", "कार", "2", "है"]
    assert c.word_langs == ["en", "en", "hi", "hi", "hi", "hi"]  # the digit inherits the previous tag
    assert len(c.word_langs) == len(c.text.split())
    assert clean("hello 2 world", "en").word_langs == ["en", "en", "en"]
    assert word_languages("یہ کتاب office", "ur") == ["ur", "ur", "en"]  # Urdu is told apart by script
    assert word_languages("2 ನಾನು", "kn") == ["kn", "kn"]
    # Native-script English (IndicVoices, MUCS) stays with the matrix language: a known limit.
    assert clean("स्लाइड इन्सर्ट करें", "hi").word_langs == ["hi", "hi", "hi"]


def test_gloss_applies_only_when_native_words_sound_like_the_english():
    from indicear.markup import sounds_alike

    for native, english in [("ಕಂಪ್ಯೂಟರ್", "computer"), ("सैकल", "cycle"), ("ಗವರ್ನಮೆಂಟ್ ಸ್ಕೂಳ್ಳಲೇ", "government school"),
                            ("ಕಾರ್", "car"), ("ఆఫీస్", "office"), ("फ़ोन", "phone"), ("जंगल", "jungle"),
                            ("சார்", "sir"), ("ரோடு", "road"), ("சைடுல", "side"), ("ரெட்", "red"), ("போர்டு", "board"),
                            ("ஆஃபீஸ்க்கு", "office"), ("ஒயிட்", "white"), ("பஸ்சுக்கு", "bus"), ("சுவிட்ச்", "switch"),
                            ("ஸ்கூல்", "school"), ("ಸಿ", "C"), ("ಕೆಜಿ", "kg"), ("ஜிபி", "GB"), ("யூ", "you")]:
        assert sounds_alike(native.split(), english.split()), (native, english)
    for native, english in [("காலு", "stone"), ("பண்ணுவாங்க", "will do"), ("पानी", "water"), ("घर", "house"),
                            ("काला", "black"), ("ಹಸಿರು", "green"),
                            ("दिन", "day"), ("ನೀರು", "water"), ("மரம்", "tree"), ("दुकान", "shop"), ("ರಸ್ತೆ", "road")]:
        assert not sounds_alike(native.split(), english.split()), (native, english)


def test_vaani_translation_gloss_keeps_native_text():
    from indicear.markup import clean

    spoken = clean("ಕಂಪ್ಯೂಟರ್ {computer} ಕೆಲಸ", "kn", "vaani")
    assert "computer" in spoken.text.split()
    translated = clean("ஒரு காலு{stone} கு உயிர் குடுத்தாங்க", "ta", "vaani")
    assert "stone" not in translated.text and "காலு" in translated.text
