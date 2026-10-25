from indicear import metrics


def test_wer_cer_and_script_error():
    refs = ["मैं घर जा रहा हूँ", "hello world"]
    hyps = ["मैं घर जा रहा हूँ", "hello word"]
    row = metrics.wer_cer(refs, hyps)
    assert row["wer"] == round(100 * 1 / 7, 2)
    assert metrics.script_error_rate(["मैं घर"], ["میں घर"]) == 50.0
    assert metrics.script_error_rate(["मैं office"], ["मैं office"]) == 0.0


def test_oov_recall():
    vocab = {"मैं", "घर"}
    out = metrics.oov_recall(["मैं नया घर देखा"], ["मैं नया घर देखो"], vocab)
    assert out == {"oov_recall": 50.0, "oov_words": 2}


def test_code_switch_spans():
    item = {"spans": [{"script": "deva", "text": "मैं घर जा रहा हूँ"}, {"script": "latn", "text": "see you soon"}]}
    good = metrics.code_switch_spans([item], ["मैं घर जा रहा हूँ see you soon"])
    assert good["span_accuracy"] == 100.0 and good["order_accuracy"] == 100.0
    swapped = metrics.code_switch_spans([item], ["see you soon मैं घर जा रहा हूँ"])
    assert swapped["span_accuracy"] == 0.0
    missing = metrics.code_switch_spans([item], ["मैं घर जा रहा हूँ"])
    assert missing["span_accuracy"] == 0.0


def test_report_groups_and_negatives():
    items = [
        {"text": "मैं घर", "lang": "hi"},
        {"text": "", "lang": "none", "negative": True},
        {"text": "मैं hi", "lang": "hi+en", "spans": [{"script": "deva", "text": "मैं"}, {"script": "latn", "text": "hi"}]},
    ]
    out = metrics.report(items, ["मैं घर", "uh", "मैं hi"], {"मैं"})
    assert out["hi"]["cer"] == 0.0 and out["negatives"]["false_output_rate"] == 100.0
    assert out["cs:hi+en"]["span_accuracy"] == 100.0


# ---------------------------------------------------------------- script-normalised WER and real code-switch spans


def test_script_normalised_wer_accepts_either_script_for_glossed_words():
    item = {"text": "मेरा order ship हो गया", "refs": ["मेरा ऑर्डर शिप हो गया"], "lang": "hi"}
    for hyp, strict, normalised in [
        ("मेरा order ship हो गया", 0.0, 0.0),  # Latin, as the primary reference
        ("मेरा ऑर्डर शिप हो गया", 40.0, 0.0),  # native script, as the stored native form
        ("मेरा ऑर्डर ship हो गया", 20.0, 0.0),  # mixed within one utterance
        ("मेरा ऑर्डर हो गया", 40.0, 20.0),  # a real deletion still counts
        ("मेरा ऑडर ship हो गया", 20.0, 20.0),  # a misspelt native form is an error
    ]:
        assert metrics.wer_cer([item["text"]], [hyp])["wer"] == strict
        assert metrics.script_normalised_wer([item], [hyp])["wer_script_normalised"] == normalised, hyp


def test_script_normalised_wer_rules():
    # no alternative reference: strict WER
    plain = {"text": "मैं घर जा रहा हूँ", "lang": "hi"}
    assert metrics.script_normalised_wer([plain], ["मैं घर जा रहा"])["wer_script_normalised"] == 20.0
    # a same-script variant (Vaani's second transcription) is not accepted
    variant = {"text": "मैं घर जा रहा हूँ", "refs": ["मैं घर जा रहा हूं"], "lang": "hi"}
    assert metrics.script_normalised_wer([variant], ["मैं घर जा रहा हूं"])["wer_script_normalised"] == 20.0
    # unequal block lengths: the whole block takes one form
    block = {"text": "मेरा order-ship हो", "refs": ["मेरा ऑर्डर शिप हो"], "lang": "hi"}
    assert metrics.script_normalised_wer([block], ["मेरा ऑर्डर शिप हो"])["wer_script_normalised"] == 0.0
    # corpus level: errors over primary reference words
    rows = [plain, {"text": "a b", "refs": [], "lang": "en"}]
    out = metrics.script_normalised_wer(rows, ["मैं घर जा रहा हूँ", "a"])
    assert out["wer_script_normalised"] == round(100 / 7, 2) and out["rows_with_alternative_ref"] == 0


def test_lattice_distance_is_edit_distance_without_alternatives():
    import jiwer

    ref, hyp = ["a", "b", "c", "d", "e"], ["a", "x", "c", "e", "f"]
    plain = metrics.lattice_distance([[(w,)] for w in ref], hyp)
    assert plain == round(jiwer.wer(" ".join(ref), " ".join(hyp)) * len(ref))


def test_code_mixed_spans_on_real_rows():
    item = {"text": "मेरा order ship हो गया", "lang": "hi", "word_langs": ["hi", "en", "en", "hi", "hi"]}
    good = metrics.code_mixed_spans([item], ["मेरा order ship हो गया"])
    assert good["cm_spans"] == 3 and good["cm_span_accuracy"] == 100.0
    native = metrics.code_mixed_spans([item], ["मेरा ऑर्डर शिप हो गया"])  # English span in the wrong script
    assert native["cm_span_accuracy_by_lang"] == {"en": 0.0, "hi": 100.0}
    assert metrics.code_mixed_spans([{"text": "मैं घर", "lang": "hi"}], ["मैं घर"])["cm_utterances"] == 0


def test_score_fold_equates_unicode_conventions_on_both_sides():
    from indicear.metrics import score_fold, wer_cer

    pairs = [
        ("അവന്‍", "അവൻ"),  # Malayalam old ZWJ chillu vs atomic chillu
        ("ত্‍", "ৎ"),  # Bengali khanda ta
        ("दुसर्‍या", "दुसऱ्या"),  # Marathi eyelash ra
        ("کتابَ", "کتاب"),  # Urdu short-vowel mark
        ("کـتاب", "کتاب"),  # tatweel
        ("ﻻ", "لا"),  # presentation form
        ("२०२४", "2024"),  # native digits
    ]
    for a, b in pairs:
        assert score_fold(a) == score_fold(b), (a, b)
        assert wer_cer([a], [b])["cer"] == 0.0
    assert score_fold("ആന") != score_fold("അന")  # real differences stay


def test_script_normalised_cer_accepts_either_reference():
    from indicear.metrics import script_normalised_wer

    items = [{"text": "मैं office जा रहा हूँ", "refs": ["मैं ऑफिस जा रहा हूँ"]}]
    out = script_normalised_wer(items, ["मैं ऑफिस जा रहा हूँ"])
    assert out["cer_script_normalised"] == 0.0 and out["wer_script_normalised"] == 0.0
    assert script_normalised_wer([{"text": "abc"}], ["abd"])["cer_script_normalised"] == round(100 / 3, 2)
