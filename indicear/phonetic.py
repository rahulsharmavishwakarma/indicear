"""Shared phonetic label set: the targets of design A's self-conditioned CTC at layers 6 and 9.

One unit inventory (~80 symbols) for every script, mapped by rule. No transliteration library is
used: `aksharamukha` is AGPL, `uroman` carries an attribution clause, and the rules below are
small enough to own. `indic_transliteration` (MIT) could be used later as a cross-check.

- **Brahmic scripts** (Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada,
  Malayalam): ISO 15919-style phonemes. The nine Unicode blocks share the ISCII-derived layout,
  so one offset table plus per-script overrides covers them, approximating the IIT Madras
  common label set. Consonants carry the inherent vowel "a" unless a vowel sign, virama or
  dead form follows. Word-final inherent "a" is dropped for Devanagari, Bengali, Gurmukhi and
  Gujarati (schwa deletion) when the word already has a vowel; medial schwa deletion is not
  modelled. Tamil's single stop series stays unvoiced (k, c, t, p): voicing is allophonic.
- **English**: an Indian-accent rule G2P from spelling (retroflex t/d, w->v, aspirated "th",
  magic-e lengthening, common digraphs). Rules, not a lexicon: irregular spellings map loosely.
- **Urdu** (Perso-Arabic): a consonant-skeleton romanization. Lossy by construction: short
  vowels are usually unwritten, so they are absent unless diacritics are present; و and ی are
  read as long vowels after consonants; ع and hamza are dropped.

Words are separated by the boundary unit "|". Digits map to shared digit units.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence

from indicear.text import BRAHMIC_BASE, word_script

BLANK_UNIT = "<blank>"
BOUNDARY = "|"
VOWELS = ("a", "ā", "i", "ī", "u", "ū", "r̥", "r̥̄", "l̥", "l̥̄", "e", "ē", "ai", "o", "ō", "au", "ê", "ô")
MODIFIERS = ("ṁ", "m̐", "ḥ", "ḵ")
CONSONANTS = (
    "k", "kh", "g", "gh", "ṅ", "c", "ch", "j", "jh", "ñ", "ṭ", "ṭh", "ḍ", "ḍh", "ṇ",
    "t", "th", "d", "dh", "n", "ṉ", "p", "ph", "b", "bh", "m", "y", "r", "ṟ", "l", "ḷ", "ḻ",
    "v", "ś", "ṣ", "s", "h", "q", "x", "ġ", "z", "ž", "f", "ṛ", "ṛh", "ẏ", "ts", "dz",
)
DIGITS = tuple("0123456789")
UNITS = (BLANK_UNIT, BOUNDARY, *VOWELS, *MODIFIERS, *CONSONANTS, *DIGITS)
UNIT_ID = {unit: index for index, unit in enumerate(UNITS)}
_VOWEL_SET = frozenset(VOWELS)

# ---------------------------------------------------------------- Brahmic table

# kind: C consonant (inherent a), V independent vowel, M vowel sign, X virama, K nukta,
# N modifier (anusvara...), D dead consonant (no inherent vowel), G gemination mark,
# R vowel carrier (Gurmukhi), S skip. Offsets follow the Devanagari layout.
_C = "kh g gh ṅ c ch j jh ñ ṭ ṭh ḍ ḍh ṇ t th d dh n ṉ p ph b bh m y r ṟ l ḷ ḻ v ś ṣ s h"
_BASE: dict[int, tuple[str, str]] = {
    0x00: ("N", "m̐"), 0x01: ("N", "m̐"), 0x02: ("N", "ṁ"), 0x03: ("N", "ḥ"),
    0x04: ("V", "a"), 0x05: ("V", "a"), 0x06: ("V", "ā"), 0x07: ("V", "i"), 0x08: ("V", "ī"),
    0x09: ("V", "u"), 0x0A: ("V", "ū"), 0x0B: ("V", "r̥"), 0x0C: ("V", "l̥"), 0x0D: ("V", "ê"),
    0x0E: ("V", "e"), 0x0F: ("V", "ē"), 0x10: ("V", "ai"), 0x11: ("V", "ô"), 0x12: ("V", "o"),
    0x13: ("V", "ō"), 0x14: ("V", "au"),
    0x15: ("C", "k"),
    **{0x16 + i: ("C", unit) for i, unit in enumerate(_C.split())},
    0x3A: ("M", "ô"), 0x3B: ("M", "ô"), 0x3C: ("K", ""), 0x3D: ("S", ""),
    0x3E: ("M", "ā"), 0x3F: ("M", "i"), 0x40: ("M", "ī"), 0x41: ("M", "u"), 0x42: ("M", "ū"),
    0x43: ("M", "r̥"), 0x44: ("M", "r̥̄"), 0x45: ("M", "ê"), 0x46: ("M", "e"), 0x47: ("M", "ē"),
    0x48: ("M", "ai"), 0x49: ("M", "ô"), 0x4A: ("M", "o"), 0x4B: ("M", "ō"), 0x4C: ("M", "au"),
    0x4D: ("X", ""), 0x4E: ("M", "e"), 0x4F: ("M", "au"),
    **{offset: ("S", "") for offset in range(0x50, 0x58)},
    0x58: ("C", "q"), 0x59: ("C", "x"), 0x5A: ("C", "ġ"), 0x5B: ("C", "z"), 0x5C: ("C", "ṛ"),
    0x5D: ("C", "ṛh"), 0x5E: ("C", "f"), 0x5F: ("C", "ẏ"),
    0x60: ("V", "r̥̄"), 0x61: ("V", "l̥̄"), 0x62: ("M", "l̥"), 0x63: ("M", "l̥̄"),
    0x64: ("S", ""), 0x65: ("S", ""),
}
assert _BASE[0x39] == ("C", "h") and _BASE[0x34] == ("C", "ḻ"), "consonant table misaligned"
_DIGIT_KIND = "#"  # digits are tagged separately from dead consonants below
for _d in range(10):
    _BASE[0x66 + _d] = (_DIGIT_KIND, str(_d))

_OVERRIDES: dict[str, dict[int, tuple[str, str]]] = {
    "deva": {0x50: ("O", "ō m"), 0x72: ("V", "ô")},
    "beng": {0x4E: ("D", "t"), 0x70: ("C", "r"), 0x71: ("C", "v")},
    "guru": {0x70: ("N", "ṁ"), 0x71: ("G", ""), 0x72: ("R", ""), 0x73: ("R", "")},
    "orya": {0x71: ("C", "v")},
    "taml": {0x03: ("N", "ḵ")},
    "telu": {0x58: ("C", "ts"), 0x59: ("C", "dz"), 0x5A: ("C", "ṟ"), 0x5D: ("D", "n")},
    "knda": {0x5D: ("D", "n"), 0x5E: ("C", "ḻ"), 0x71: ("N", "ḥ"), 0x72: ("N", "ḥ")},
    "mlym": {
        0x3A: ("C", "ṟ"), 0x3B: ("X", ""), 0x3C: ("X", ""), 0x4E: ("D", "r"),
        0x54: ("D", "m"), 0x55: ("D", "y"), 0x56: ("D", "ḻ"),
        **{offset: ("S", "") for offset in range(0x58, 0x5F)},
        0x5F: ("V", "ī"),
        0x7A: ("D", "ṇ"), 0x7B: ("D", "n"), 0x7C: ("D", "r"), 0x7D: ("D", "l"),
        0x7E: ("D", "ḷ"), 0x7F: ("D", "k"),
    },
}
_NUKTA = {"k": "q", "kh": "x", "g": "ġ", "j": "z", "ḍ": "ṛ", "ḍh": "ṛh", "ph": "f", "y": "ẏ"}
_SCHWA_DELETION = frozenset({"deva", "beng", "guru", "gujr"})
_OFFSET_SCRIPT = {base: script for script, base in BRAHMIC_BASE.items()}


def _brahmic_lookup(char: str) -> tuple[str, str]:
    code = ord(char)
    base = code & ~0x7F
    script = _OFFSET_SCRIPT.get(base)
    if script is None:
        return ("S", "")
    offset = code - base
    entry = _OVERRIDES.get(script, {}).get(offset) or _BASE.get(offset, ("S", ""))
    return entry


def brahmic_units(word: str, script: str) -> list[str]:
    units: list[str] = []
    pending = False  # a consonant whose vowel is still undecided
    geminate = False
    for char in unicodedata.normalize("NFC", word):
        if "0" <= char <= "9":
            kind, unit = _DIGIT_KIND, char
        else:
            kind, unit = _brahmic_lookup(char)
        if kind == "C":
            if pending:
                units.append("a")
            units.extend(unit.split() * (2 if geminate else 1))
            pending, geminate = True, False
        elif kind == "K":
            if pending and units:
                units[-1] = _NUKTA.get(units[-1], units[-1])
        elif kind == "M":
            units.append(unit)
            pending = False
        elif kind == "X":
            pending = False
        elif kind == "R":  # vowel carrier: its vowel sign (if any) supplies the vowel
            if pending:
                units.append("a")
            pending = False
        elif kind == "G":
            if pending:
                units.append("a")
            pending, geminate = False, True
        elif kind in ("V", "D", "N", "O", _DIGIT_KIND):
            if pending:
                units.append("a")
            units.extend(unit.split())
            pending = False
    if pending:
        has_vowel = any(u in _VOWEL_SET for u in units)
        if not (script in _SCHWA_DELETION and has_vowel):
            units.append("a")
    return units


# ---------------------------------------------------------------- English (Indian accent)

# Ordered (pattern, units) rules applied left to right, longest first. "^" anchors to word start.
_EN_RULES: tuple[tuple[str, str], ...] = (
    ("tion", "ś a n"), ("sion", "ś a n"), ("ture", "c a r"), ("^kn", "n"), ("^wr", "r"),
    ("^wh", "v"), ("tch", "c"), ("dge", "j"), ("ch", "c"), ("sh", "ś"), ("th", "th"),
    ("ph", "f"), ("ck", "k"), ("qu", "k v"), ("ng", "ṅ"), ("gh", ""),
    ("ee", "ī"), ("ea", "ī"), ("oo", "ū"), ("ou", "au"), ("ow", "au"), ("ai", "ē"),
    ("ay", "ē"), ("ei", "ē"), ("ey", "ē"), ("oa", "ō"), ("oi", "o y"), ("oy", "o y"),
    ("au", "ô"), ("aw", "ô"), ("ie", "ī"), ("ue", "ū"), ("ui", "ū"),
    ("ar", "ā r"), ("or", "ô r"), ("er", "a r"), ("ir", "a r"), ("ur", "a r"),
)
_EN_SINGLE = {
    "a": "a", "b": "b", "d": "ḍ", "e": "e", "f": "f", "h": "h", "i": "i", "j": "j", "k": "k",
    "l": "l", "m": "m", "n": "n", "o": "o", "p": "p", "q": "k", "r": "r", "s": "s", "t": "ṭ",
    "u": "a", "v": "v", "w": "v", "x": "k s", "y": "i", "z": "z",
}
_MAGIC_E = {"a": "ē", "e": "ī", "i": "ai", "o": "ō", "u": "ū"}
_EN_VOWELS = set("aeiou")


def english_units(word: str) -> list[str]:
    word = "".join(c for c in word.lower() if c.isalnum())
    if not word:
        return []
    if len(word) <= 2 and word.endswith("e") and word[0] not in _EN_VOWELS:
        return [*_EN_SINGLE.get(word[0], word[0]).split(), "ī"]  # he, be, me, we
    # Final e after a consonant is silent when the word has another vowel letter ("phone",
    # "office"; "the" keeps it). Magic e: in a single-vowel stem, V C e lengthens the vowel.
    magic = None
    limit = len(word)
    silent_e = (
        len(word) >= 3
        and word[-1] == "e"
        and word[-2] not in _EN_VOWELS
        and any(c in _EN_VOWELS for c in word[:-1])
    )
    if silent_e:
        limit = len(word) - 1  # the e still serves as lookahead for c/g softening
        stem_vowels = [c for c in word[:-1] if c in _EN_VOWELS]
        if word[-3] in _MAGIC_E and len(stem_vowels) == 1:
            magic = len(word) - 3
    units: list[str] = []
    i = 0
    while i < limit:
        if i == magic:
            units.append(_MAGIC_E[word[i]])
            i += 1
            continue
        for pattern, out in _EN_RULES:
            anchored = pattern.startswith("^")
            text = pattern[1:] if anchored else pattern
            if (not anchored or i == 0) and word.startswith(text, i) and i + len(text) <= limit:
                units.extend(out.split())
                i += len(text)
                break
        else:
            char = word[i]
            if char.isdigit():
                units.append(ascii_digit(char))
            elif char == "c":
                units.append("s" if word[i + 1 : i + 2] in ("e", "i", "y") else "k")
            elif char == "g":
                units.append("j" if word[i + 1 : i + 2] in ("e", "i", "y") and i > 0 else "g")
            elif char == "y" and i == 0:
                units.append("y")
            elif char in _EN_SINGLE:
                units.extend(_EN_SINGLE[char].split())
            i += 1
    # Collapse doubled consonants from doubled letters (ll, tt, ss...).
    collapsed: list[str] = []
    for unit in units:
        if collapsed and unit == collapsed[-1] and unit not in _VOWEL_SET:
            continue
        collapsed.append(unit)
    return collapsed


# ---------------------------------------------------------------- Urdu (Perso-Arabic)

_UR_CONSONANTS = {
    "ب": "b", "پ": "p", "ت": "t", "ٹ": "ṭ", "ث": "s", "ج": "j", "چ": "c", "ح": "h", "خ": "x",
    "د": "d", "ڈ": "ḍ", "ذ": "z", "ر": "r", "ڑ": "ṛ", "ز": "z", "ژ": "ž", "س": "s", "ش": "ś",
    "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "غ": "ġ", "ف": "f", "ق": "q", "ک": "k", "ك": "k",
    "گ": "g", "ل": "l", "م": "m", "ن": "n", "ۃ": "t", "ة": "t",
}
_UR_ASPIRATE = {"k": "kh", "g": "gh", "c": "ch", "j": "jh", "ṭ": "ṭh", "ḍ": "ḍh", "t": "th",
                "d": "dh", "p": "ph", "b": "bh", "ṛ": "ṛh"}
_UR_DIACRITICS = {"َ": "a", "ِ": "i", "ُ": "u", "ٰ": "ā", "ً": "a n"}
_UR_DIGITS = {chr(0x0660 + d): str(d) for d in range(10)} | {chr(0x06F0 + d): str(d) for d in range(10)}


def urdu_units(word: str) -> list[str]:
    units: list[str] = []
    chars = list(unicodedata.normalize("NFC", word))
    for i, char in enumerate(chars):
        initial = not units
        final = i == len(chars) - 1
        after_consonant = bool(units) and units[-1] not in _VOWEL_SET
        if char in _UR_CONSONANTS:
            units.append(_UR_CONSONANTS[char])
        elif char == "ھ":  # do-chashmi he: aspiration of the previous consonant
            if units and units[-1] in _UR_ASPIRATE:
                units[-1] = _UR_ASPIRATE[units[-1]]
            else:
                units.append("h")
        elif char in ("ہ", "ه"):
            units.append("a" if final and after_consonant else "h")
        elif char == "ا":
            units.append("a" if initial else "ā")
        elif char == "آ":
            units.append("ā")
        elif char == "و":
            units.append("v" if initial or not after_consonant else "ū")
        elif char in ("ی", "ي", "ى"):
            units.append("y" if initial or not after_consonant else "ī")
        elif char in ("ے", "ۓ"):
            units.append("ē")
        elif char == "ں":
            units.append("m̐")
        elif char == "ّ":  # shadda: geminate
            if units and units[-1] not in _VOWEL_SET:
                units.append(units[-1])
        elif char in _UR_DIACRITICS:
            units.extend(_UR_DIACRITICS[char].split())
        elif char in _UR_DIGITS or "0" <= char <= "9":
            units.append(_UR_DIGITS.get(char, char))
        # ع, ء, ئ, sukun and anything else: dropped (documented loss)
    return units


# ---------------------------------------------------------------- dispatch and vocabulary


def word_units(word: str) -> list[str]:
    script = word_script(word, default="digits")
    if script == "latn":
        return english_units(word)
    if script == "arab":
        return urdu_units(word)
    if script in BRAHMIC_BASE:
        return brahmic_units(word, script)
    if script == "digits":
        return [ascii_digit(c) for c in word if c.isdigit()]
    return []  # other scripts: no units


def ascii_digit(char: str) -> str:
    """Any Unicode digit (Devanagari, Arabic-Indic, superscript "²", ...) -> its ASCII digit unit."""
    return str(unicodedata.digit(char))


def text_units(text: str) -> list[str]:
    """Normalized text -> phonetic units with "|" between words (no leading/trailing "|")."""
    out: list[str] = []
    for word in text.split():
        units = word_units(word)
        if not units:
            continue
        if out:
            out.append(BOUNDARY)
        out.extend(units)
    return out


def encode(text: str) -> list[int]:
    # Units outside the inventory are dropped rather than failing a whole training run.
    return [UNIT_ID[unit] for unit in text_units(text) if unit in UNIT_ID]


def decode(ids: Iterable[int]) -> str:
    """IDs -> readable romanization: units concatenated, words separated by spaces."""
    words, current = [], []
    for index in ids:
        unit = UNITS[int(index)]
        if unit == BLANK_UNIT:
            continue
        if unit == BOUNDARY:
            if current:
                words.append("".join(current))
            current = []
        else:
            current.append(unit)
    if current:
        words.append("".join(current))
    return " ".join(words)


def romanize(text: str) -> str:
    return decode(encode(text))


def units_per_second(texts: Sequence[str], seconds: Sequence[float]) -> list[float]:
    return [len(text_units(t)) / max(s, 1e-3) for t, s in zip(texts, seconds, strict=True)]

