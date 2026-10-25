"""Transcript conventions: normalization, script detection and the gloss rewrite.

Convention: Indic words in native script, English words in Latin script. References and
hypotheses go through the same `normalize`, so WER/CER compare like with like.

Normalization extends ../eg2-probe/probe.py's `normalize` (NFC, lowercase, punctuation and
symbols dropped, combining marks kept) with three changes, all covered by tests:
- format characters (ZWJ/ZWNJ, category Cf) are deleted instead of becoming spaces, so they
  never split an Indic word;
- an apostrophe survives only between two Latin letters ("don't"), never as a quote;
- a word that mixes scripts is split at the script boundary ("facebookपर" -> "facebook पर"),
  so every word belongs to exactly one sub-tokenizer.
"""

from __future__ import annotations

import re
import unicodedata

# Unicode blocks per script. Letters, marks and native numerals get a script; ASCII digits and the
# apostrophe attach to the surrounding word. Unknown letters are "other".
# The nine Brahmic blocks share the ISCII-derived layout used by indicear.phonetic.
BRAHMIC_BASE = {
    "deva": 0x0900, "beng": 0x0980, "guru": 0x0A00, "gujr": 0x0A80, "orya": 0x0B00,
    "taml": 0x0B80, "telu": 0x0C00, "knda": 0x0C80, "mlym": 0x0D00,
}
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "latn": ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)),
    **{script: ((base, base + 0x7F),) for script, base in BRAHMIC_BASE.items()},
    "arab": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
}
SCRIPT_RANGES["deva"] += ((0xA8E0, 0xA8FF), (0x1CD0, 0x1CFF))
SCRIPT_RANGES["taml"] += ((0x11FC0, 0x11FFF),)
# The eleven output scripts of design A's two-level head, in a fixed order.
SCRIPTS = ("deva", "beng", "guru", "gujr", "orya", "taml", "telu", "knda", "mlym", "arab", "latn")
LANG_SCRIPT = {
    "hi": "deva", "mr": "deva", "bn": "beng", "pa": "guru", "gu": "gujr", "or": "orya",
    "ta": "taml", "te": "telu", "kn": "knda", "ml": "mlym", "ur": "arab", "en": "latn",
}
# Arabic-script letter variants folded to their Urdu forms (Arabic yeh/kaf/alef maksura).
_URDU_FOLD = str.maketrans({"\u064a": "\u06cc", "\u0649": "\u06cc", "\u0643": "\u06a9"})

_WS = re.compile(r"\s+")


def char_script(char: str) -> str | None:
    """Script of a letter, combining mark or native numeral; None for ASCII digits,
    punctuation and spaces (those attach to the surrounding word)."""
    category = unicodedata.category(char)[0]
    if category not in "LMN" or "0" <= char <= "9":
        return None
    code = ord(char)
    for script, ranges in SCRIPT_RANGES.items():
        if any(lo <= code <= hi for lo, hi in ranges):
            return script
    return "other" if category in "LM" else None


def word_script(word: str, default: str | None = None) -> str | None:
    """Majority script of a word's letters and marks (ties: first seen)."""
    counts: dict[str, int] = {}
    for char in word:
        script = char_script(char)
        if script is not None:
            counts[script] = counts.get(script, 0) + 1
    if not counts:
        return default
    return max(counts, key=counts.__getitem__)


def is_native(word: str) -> bool:
    """True for a word in a non-Latin script (the gloss rewrite's "native token")."""
    script = word_script(word)
    return script is not None and script != "latn"


def split_script_runs(text: str) -> str:
    """Insert a space wherever two adjacent scripted characters belong to different scripts."""
    out: list[str] = []
    current: str | None = None
    for char in text:
        if char.isspace():
            current = None
            out.append(char)
            continue
        script = char_script(char)
        if script is not None:
            if current is not None and script != current:
                out.append(" ")
            current = script
        out.append(char)
    return "".join(out)


def _keep_apostrophe(text: str, index: int) -> bool:
    return (
        0 < index < len(text) - 1
        and char_script(text[index - 1]) == "latn"
        and char_script(text[index + 1]) == "latn"
    )


def normalize(text: str, lang: str | None = None) -> str:
    """NFC, lowercase Latin, drop punctuation/symbols/controls, keep combining marks.

    `lang` is accepted for interface stability; every rule is script-driven, so a
    code-switched transcript normalizes the same whatever its language tag.
    """
    del lang
    text = unicodedata.normalize("NFC", text).replace("\u2019", "'").replace("\u2018", "'")
    text = text.translate(_URDU_FOLD)
    chars: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category == "Cf":
            continue  # ZWJ, ZWNJ, soft hyphen, BOM: deleted, never a word break
        if char == "'" or category[0] not in "PSZC":
            chars.append(char)
        else:
            chars.append(" ")
    text = "".join(chars).lower()
    text = "".join(
        " " if char == "'" and not _keep_apostrophe(text, i) else char
        for i, char in enumerate(text)
    )
    return _WS.sub(" ", split_script_runs(text)).strip()


# ---------------------------------------------------------------- gloss rewrite

# Bracketed spans: IndicVoices uses [english], Vaani uses {english}. Angle-bracket tags such as
# <noise> are always dropped.
_BRACKETS = re.compile(r"\[([^\[\]]*)\]|\{([^{}]*)\}|<[^<>]*>")
# Annotation tags that are Latin script but are not glosses. TODO: extend after running
# `python -m indicear.data inspect-indicvoices` on real data (PLAN.md open question).
# Only clearly non-lexical tags: a tag that is also an English word ("music") is ambiguous, and
# is listed only when annotation use is far more likely than a gloss of that word.
NOISE_TAGS = frozenset(
    """
    noise laughter laugh laughing music breath breathing cough coughing sneeze sigh silence sil
    unintelligible inaudible unclear overlap overlapping crosstalk applause hesitation filler um
    uh umm hmm hm mm static beep lipsmack babble vocalsound unk
    """.split()
) | frozenset({"lip smack", "no speech", "background noise", "vocal sound", "noise tag"})
_GLOSS_CHARS = re.compile(r"^[A-Za-z][A-Za-z' .-]*$")


def is_latin_gloss(content: str) -> bool:
    """True when a bracket holds a Latin-script English gloss rather than an annotation tag."""
    content = content.strip()
    if not content or not _GLOSS_CHARS.match(content):
        return False  # empty, non-Latin, digits or underscores (e.g. [background_noise])
    phrase = " ".join(content.lower().replace("-", " ").split())
    if phrase in NOISE_TAGS:
        return False
    return not all(word.strip(".'") in NOISE_TAGS for word in phrase.split())


def rewrite_gloss(text: str) -> tuple[str, int]:
    """Replace the native token before a Latin gloss with the gloss; drop every other bracket.

    "मैं कंप्यूटर [computer] पर" -> "मैं computer पर". A gloss fires only when its content is
    Latin script, is not an annotation tag, and directly follows a native-script token;
    otherwise the bracket is dropped. A multi-word gloss replaces exactly one native token
    (checked against real IndicVoices and Vaani markup during prep). Returns (text, glosses applied).
    """
    tokens: list[str] = []
    applied = 0
    position = 0
    for match in _BRACKETS.finditer(text):
        tokens.extend(text[position : match.start()].split())
        position = match.end()
        content = match.group(1) if match.group(1) is not None else match.group(2)
        if content is None or not is_latin_gloss(content):
            continue  # <tag>, [noise], [शोर], empty: dropped
        if tokens and is_native(tokens[-1]):
            tokens[-1] = content.strip()
            applied += 1
    tokens.extend(text[position:].split())
    return " ".join(tokens), applied


def strip_tags(text: str) -> str:
    """Drop every bracketed span without rewriting (sources without a gloss convention)."""
    return _WS.sub(" ", _BRACKETS.sub(" ", text)).strip()


def prepare_transcript(raw: str, lang: str, gloss: bool = False) -> tuple[str, int]:
    """Raw source transcript -> normalized reference. Returns (text, glosses applied)."""
    if gloss:
        text, applied = rewrite_gloss(raw)
    else:
        text, applied = strip_tags(raw), 0
    return normalize(text, lang), applied
