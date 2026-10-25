"""Per-source transcript markup -> the IndicEAR transcript convention.

Convention (docs/PLAN.md): Indic words in native script, English words in Latin script; every
reference goes through `indicear.text.normalize`.

Conventions found on 2026-10-24 (sources in `indicear.sources` and docs/data-licenses.md):

- **Vaani** (`markup="vaani"`), from public Vaani transcripts and the Vaani-Benchmark card:
  - `{english}` follows the native-script rendering of an English word: "सैकल{cycle}",
    "ब्लैक {black} कलर {color}". A gloss can cover several words ("ಗವರ್ನಮೆಂಟ್ ಸ್ಕೂಳ್ಳಲೇ
    {government school}") and can be glued to the next suffix ("ಕಾರ್ {car}ಆಲ್ಲಿ").
    Rewrite: the gloss replaces as many preceding native-script tokens as it has words.
  - `{native}` and `(native)` give the standard spelling of a dialect form ("पे {पर}",
    "बोहोत {बहुत}"): dropped, so the reference keeps what was said.
  - `<noise> ... </noise>`, `<static_noise>`, `<PAUSE>`, `[breathing]`, `[lip smacks]`: tags
    dropped, enclosed speech kept. `--` marks a cut-off word: dropped, the fragment kept.
  - `[unintelligible]`: the row is dropped.
- **IndicVoices** (`markup="indicvoices"`, read from `unsanitized_normalized`; verified on the
  gated rows): English loanwords are in native script followed by a Latin gloss in square
  brackets ("ಆರ್ಡರ್ [order] ಶಿಪ್‌ [ship]"; 50-71% of valid rows in hi/kn/ta). Tags are the 89
  Table-13 names (arXiv 2403.01926) written as `<tag>` or `[tag]`, sometimes capitalised or
  misspelt; close matches to a tag name are tags, never glosses. `<unintelligible>` drops the row.
- **IndicContextEval** (`markup="paren_gloss"`): "वारंटी (Warranty)": a Latin parenthesis glosses
  the preceding word.
- **Plain** sources: any bracketed tag is dropped (`text.strip_tags`).

After cleaning, a row is dropped when it is empty, when a word is in a script other than the
language's own or Latin, or when it yields no phonetic units. A row of an Indic language with at
least one Latin and one native word is flagged `code_mixed`.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from indicear import phonetic
from indicear.text import (
    LANG_SCRIPT,
    char_script,
    is_native,
    normalize,
    strip_tags,
    word_script,
)

# IndicVoices paper, Table 13 (verbatim list of the 89 transcription tags).
INDICVOICES_TAGS = frozenset("""
TV animal baby baby_crying tsk barking beep bell sniffle child child_crying child_laughing child_talking
dishes child_yelling children children_talking tones click clicking clink clinking cough child_whining door
footsteps gasp horn hum inhaling laughter meow motorcycle music nose_blowing noise phone_vibrating popping
pounding screeching rattling ringing rustling scratching printer smack sneezing sniffing bird_squawk snorting
squawking squeak stammers static throat_clearing thumping tone children_yelling trill baby_talking typewriter
ugh uhh wheezing whispering whistling yawning yelling buzzer clanking phone_ringing hmm unintelligible buzz
clanging persistent-noise-start tapping singing talking umm hiss breathing chiming groan persistent-noise-end
sigh swallowing uh-huh siren
""".lower().split())
UNUSABLE = ("unintelligible", "inaudible", "unclear")

_MARKUP = re.compile(r"\{([^{}]*)\}|\[([^\[\]]*)\]|<([^<>]*)>|\(([^()]*)\)")
_CUTOFF = re.compile(r"-{2,}")


SCRIPT_LANG_FALLBACK = {"latn": "en"}


def word_languages(text: str, lang: str) -> list[str]:
    """Per-word language tags for LID training, derived from script: Latin words are "en", words in
    the row language's own script are `lang`, and script-less words (digits) inherit the previous
    word's tag (else `lang`). One tag per whitespace-separated word of the normalized text.

    Limits: an English word written in native script (common in IndicVoices, some MUCS and every
    unglossed Vaani loanword) is tagged with the matrix language; Hindi and Urdu are told apart
    by script only."""
    own = LANG_SCRIPT[lang]
    tags: list[str] = []
    for word in text.split():
        script = word_script(word)
        if script is None:
            tags.append(tags[-1] if tags else lang)
        elif script == own:
            tags.append(lang)
        else:
            tags.append(SCRIPT_LANG_FALLBACK.get(script, lang))
    return tags


def unusable(raw: str | None) -> bool:
    """True when any bracketed tag (<...>, [...], {...}, (...)) marks unintelligible speech."""
    return bool(raw) and any(any(word in _tag_key(m.group(0)) for word in UNUSABLE) for m in _MARKUP.finditer(raw))


@dataclass
class Cleaned:
    text: str
    drop: str | None = None
    glosses: int = 0
    code_mixed: bool = False
    notes: Counter = field(default_factory=Counter)
    word_langs: list[str] = field(default_factory=list)
    extra_refs: list[str] = field(default_factory=list)  # alternative references (loanword normalisation)


def is_indicvoices_tag(content: str) -> bool:
    """A Table-13 tag, including annotator typos ('[breathinhg]') and the capitalised forms
    ('<Persistent-noise-start>'): close matches count as tags, never as English glosses."""
    import difflib

    key = _tag_key(content)
    return key in INDICVOICES_TAGS or bool(difflib.get_close_matches(key, INDICVOICES_TAGS, n=1, cutoff=0.85))


def _paren_gloss(gloss: bool):
    """IndicContextEval: 'वारंटी (Warranty)': a Latin parenthesis is the English gloss of the
    preceding native word; other brackets are tags."""
    def handle(kind: str, content: str, tokens: list[str], notes: Counter) -> int:
        if kind == "(" and _content_scripts(content) == {"latn"}:
            if not gloss:
                notes["gloss_kept_native"] += 1
                return 0
            return int(_apply_gloss(tokens, content.strip(), notes))
        notes["tag"] += 1
        return 0

    return handle


def _tag_key(content: str) -> str:
    return "_".join(content.strip().lower().split())


def _content_scripts(content: str) -> set[str]:
    return {s for s in (char_script(c) for c in content) if s is not None}


# Broad consonant classes: voicing, aspiration, place within a class and nasal place are ignored,
# because scripts such as Tamil write p/b, t/d and k/g with one letter.
_CONSONANT_CLASS = {c: cls for cls, members in {
    "P": "pbf", "T": "td", "K": "kgq", "S": "cjsz", "N": "mn", "L": "l", "R": "r", "V": "vw"}.items() for c in members}


def consonant_skeleton(units: list[str]) -> list[str]:
    """Phonetic units -> broad consonant classes (no vowels, h or y), repeats collapsed. Native and
    English renderings of the same word share a skeleton."""
    out: list[str] = []
    for unit in units:
        if unit == phonetic.BOUNDARY:
            continue
        base = "".join(c for c in unicodedata.normalize("NFD", unit) if not unicodedata.combining(c)).lower()
        for char in base:
            if char in "aeiouhy":
                continue
            for cls in (["K", "S"] if char == "x" else [_CONSONANT_CLASS.get(char, char)]):
                if not out or out[-1] != cls:
                    out.append(cls)
    return out


def _edit_distance(a: list[str], b: list[str]) -> int:
    row = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        previous, row[0] = row[:], i
        for j in range(1, len(b) + 1):
            row[j] = min(previous[j] + 1, row[j - 1] + 1, previous[j - 1] + (a[i - 1] != b[j - 1]))
    return row[len(b)]


def sounds_alike(native: list[str], english: list[str]) -> bool:
    """True when the native-script words are a rendering of the English words (ಕಂಪ್ಯೂಟರ್ / computer),
    false for a translation gloss (காலு / stone, पानी / water). The native words may carry a case
    suffix (சைடுல / side); short skeletons must match exactly."""
    if all(w.isalpha() and len(w) <= 4 and (w.isupper() or w in ("kg", "km", "cm")) for w in english):
        return True  # spelled-out letters: "ಸಿ" (C), "ಕೆಜಿ" (kg), "ஜிபி" (GB)
    a = consonant_skeleton([u for w in native for u in phonetic.word_units(w.replace("ஃ", ""))])
    b = consonant_skeleton([u for w in english for u in phonetic.english_units(w.lower().replace("sch", "sk"))])
    if not a and not b:
        return True  # vowel-only words: "यू" (you), "आई" (I)
    if not a or not b:
        return False
    candidates = [b, b[1:]] if len(b) > 1 and b[0] == "V" else [b]  # "white" is said "oyit"
    for target in candidates:
        if len(target) <= 1:
            distance = _edit_distance(a, target)
        else:  # best native prefix, allowing a trailing suffix
            distance = min(_edit_distance(a[:k], target) for k in range(len(target) - 1, len(a) + 1)) \
                if len(a) >= len(target) - 1 else _edit_distance(a, target)
        if (distance == 0) if len(target) <= 2 else (distance / len(target) <= 0.34):
            return True
    return False


def _apply_gloss(tokens: list[str], content: str, notes: Counter) -> bool:
    """Replace the trailing native-script tokens with the Latin gloss when they sound alike (the
    speaker said the English word). A translation gloss keeps the native words. Returns True if applied."""
    words = content.split()
    trailing = 0
    while trailing < len(tokens) and is_native(tokens[-1 - trailing]):
        trailing += 1
    take = min(len(words), trailing)
    if take == 0:
        notes["gloss_unattached"] += 1
        return False
    if not sounds_alike(tokens[len(tokens) - take:], words):
        notes["gloss_translation_kept_native"] += 1
        return False
    if take < len(words):
        notes["gloss_span_short"] += 1
    del tokens[len(tokens) - take:]
    tokens.extend(words)
    return True


def _walk(raw: str, handle) -> tuple[str, Counter, int, str | None]:
    """Tokenize around markup; `handle(kind, content, tokens, notes)` returns glosses applied
    (or raises _Drop)."""
    tokens: list[str] = []
    notes: Counter = Counter()
    glosses = 0
    position = 0
    for match in _MARKUP.finditer(raw):
        tokens.extend(raw[position:match.start()].split())
        position = match.end()
        kind = "{}[]<>()"[2 * next(i for i in range(4) if match.group(i + 1) is not None)]
        content = next(g for g in match.groups() if g is not None)
        if any(word in _tag_key(content) for word in UNUSABLE):
            return "", notes, 0, "unintelligible"
        glosses += handle(kind, content, tokens, notes)
    tokens.extend(raw[position:].split())
    text = " ".join(tokens)
    if _CUTOFF.search(text):
        notes["cutoff"] += 1
        text = _CUTOFF.sub(" ", text)
    return text, notes, glosses, None


def _vaani(gloss: bool):
    def handle(kind: str, content: str, tokens: list[str], notes: Counter) -> int:
        return _vaani_one(kind, content, tokens, notes, gloss)

    return handle


def _vaani_one(kind: str, content: str, tokens: list[str], notes: Counter, gloss: bool) -> int:
    if kind == "{":
        scripts = _content_scripts(content)
        if scripts == {"latn"}:
            if not gloss:
                notes["gloss_kept_native"] += 1
                return 0
            return int(_apply_gloss(tokens, content.strip(), notes))
        notes["standard_spelling" if scripts else "empty_brace"] += 1
        return 0
    if kind == "(":
        notes["standard_spelling"] += 1
        return 0
    notes["tag"] += 1
    return 0


def _indicvoices(gloss: bool):
    def handle(kind: str, content: str, tokens: list[str], notes: Counter) -> int:
        if kind == "<":
            notes["tag"] += 1
            return 0
        if kind == "(":  # not part of the documented convention: keep the words
            notes["parenthesis"] += 1
            tokens.extend(content.split())
            return 0
        if kind == "[" and is_indicvoices_tag(content):
            notes["tag"] += 1
            return 0
        scripts = _content_scripts(content)
        if scripts == {"latn"}:
            if gloss:
                return int(_apply_gloss(tokens, content.strip(), notes))
            notes["unknown_bracket_latin"] += 1
        else:
            notes["unknown_bracket_other"] += 1
        return 0

    return handle


def finalize(text: str, lang: str, notes: Counter | None = None, glosses: int = 0) -> Cleaned:
    """Normalize and apply the row-level checks shared by every source."""
    notes = notes if notes is not None else Counter()
    text = normalize(text, lang)
    if not text:
        return Cleaned("", "empty", glosses, False, notes)
    own = LANG_SCRIPT[lang]
    latin = native = False
    for word in text.split():
        script = word_script(word)
        if script is None:
            continue  # digits
        if script == "latn":
            latin = True
        elif script == own:
            native = True
        else:
            return Cleaned(text, f"foreign_script:{script}", glosses, False, notes)
    try:
        units = phonetic.encode(text)
    except Exception:  # noqa: BLE001 - any failure here is a data problem, counted and dropped
        return Cleaned(text, "phonetic_error", glosses, False, notes)
    if not units:
        return Cleaned(text, "no_units", glosses, False, notes)
    return Cleaned(text, None, glosses, lang != "en" and latin and native, notes, word_languages(text, lang))


def clean(raw: str, lang: str, markup: str = "plain", indicvoices_gloss: bool = True, gloss: bool = True) -> Cleaned:
    """Raw source transcript -> normalized reference, drop reason, gloss count and flags.
    `gloss=False` keeps English words in the native-script rendering the annotators wrote (the
    alternative reference for script-normalised scoring)."""
    if raw is None:
        return Cleaned("", "missing_text")
    if markup == "vaani":
        text, notes, glosses, drop = _walk(raw, _vaani(gloss))
    elif markup == "indicvoices":
        text, notes, glosses, drop = _walk(raw, _indicvoices(gloss and indicvoices_gloss))
    elif markup == "paren_gloss":
        text, notes, glosses, drop = _walk(raw, _paren_gloss(gloss))
    elif markup == "plain":
        notes = Counter()
        if any(word in raw.lower() for word in ("[unintelligible]", "<unintelligible>")):
            return Cleaned("", "unintelligible")
        text, glosses, drop = strip_tags(raw), 0, None
        if text != " ".join(raw.split()):
            notes["tag"] += 1
    else:
        raise ValueError(f"unknown markup convention {markup!r}")
    if drop:
        return Cleaned("", drop, 0, False, notes)
    return finalize(text, lang, notes, glosses)


def native_reference(raw: str, lang: str, markup: str) -> str | None:
    """The same transcript with English left in native script (no gloss rewrite), or None when the
    convention has no glosses. Scoring against both references gives a script-normalised WER."""
    if markup not in ("vaani", "indicvoices", "paren_gloss"):
        return None
    alt = clean(raw, lang, markup, gloss=False)
    return alt.text if not alt.drop else None


def apply_loanwords(cleaned: Cleaned, lang: str, normaliser, primary: bool = True) -> Cleaned:
    """Loanword normalisation for native-only sources (Kathbath, Shrutilipi): they write English
    loanwords in native script, while IndicVoices and Vaani (after the gloss rewrite) write them in
    Latin, so the same spoken word gets conflicting targets. `normaliser` (indicear.loanwords)
    rewrites the native renderings of common English words to Latin.

    primary=True (training and dev rows): the rewritten text becomes the reference and the
    original native-script text an alternative in `extra_refs` (scored by the script-normalised
    WER). primary=False (eval rows): the published reference stays primary, so strict WER and CER
    remain comparable with other systems, and the rewrite is the alternative."""
    if cleaned.drop or normaliser is None or lang == "en" or not cleaned.text:
        return cleaned
    rewritten, count = normaliser.rewrite(cleaned.text, lang)
    if not count or rewritten == cleaned.text:
        return cleaned
    if not primary:
        cleaned.extra_refs = [*cleaned.extra_refs, rewritten]
        cleaned.notes["loanword_alternative"] += 1
        return cleaned
    new = finalize(rewritten, lang, cleaned.notes, cleaned.glosses)
    if new.drop:  # never lose a row to the rewrite
        cleaned.notes["loanword_rewrite_rejected"] += 1
        return cleaned
    new.notes["loanword_rows"] += 1
    new.notes["loanword_words"] += count
    new.extra_refs = [*cleaned.extra_refs, cleaned.text]
    return new
