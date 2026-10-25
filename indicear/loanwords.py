"""English loanwords written in native script -> Latin, for native-only training sources.

Kathbath, Shrutilipi and other native-only sources write English loanwords in the native script
("ಆರ್ಡರ್"), while IndicVoices and Vaani after IndicEAR's gloss handling write them in Latin
("order"). `LoanwordNormaliser.rewrite` turns the first convention into the second for words
whose identity is well attested, so the two kinds of source stop teaching conflicting spellings.
It is deliberately conservative: a word that is not clearly a loanword stays native.

    python -m indicear.loanwords build --iv DIR --out indicear/resources/loanwords   # mine the lexicon
    python -m indicear.loanwords review --text DIR --out review/ --per-lang 100   # sample rewrites

Where the mappings come from (docs/loanwords.md):

1. **Mined lexicon** (the main path). IndicVoices `unsanitized_normalized` transcripts follow an
   English loanword with its Latin gloss: "ಆರ್ಡರ್ [order]". A single native word -> single
   English word pair enters the lexicon when
   - `markup.sounds_alike` accepts it and the *whole* native word matches the English consonant
     skeleton (no case suffix: "ಕಾರ್ನಲ್ಲಿ [car]" is not "car"),
   - it was seen at least `min_count` times and is the native word's dominant gloss,
   - the native word is glossed in at least `min_gloss_share` of its IndicVoices occurrences,
     so a native word that merely sounds like an English one (कमरा "room" vs camera) is never
     rewritten,
   - the native word has at least `min_letters` letters and the English word at least
     `min_english_letters` (initialisms such as "MM" are skipped).
2. **Word-list path** (secondary; per language, only where its reviewed precision met the bar):
   a native word absent from the lexicon and never seen unglossed in IndicVoices is matched by
   consonant skeleton and `sounds_alike` against common English words (SCOWL sizes <= 50, see
   `indicear/resources/english_words.tsv` and its licence), with a vowel-aware check and a unique
   best candidate. Disabled for every language in the shipped parameters (see docs/loanwords.md).
3. **Allowlist / blocklist** per language (`allow.json`, `block.json`): forced mappings and
   native words that are never rewritten.

Input to `rewrite` is text already normalized by `indicear.text.normalize`; the output keeps that
form (Latin words lowercase, single spaces) and the same word count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from indicear import phonetic
from indicear.markup import (
    _edit_distance,
    consonant_skeleton,
    is_indicvoices_tag,
    sounds_alike,
)
from indicear.text import LANG_SCRIPT, normalize, word_script

RESOURCES = Path(__file__).resolve().parent / "resources"
DEFAULT_DIR = RESOURCES / "loanwords"
ENGLISH_WORDS = RESOURCES / "english_words.tsv"
VERSION = 1
INDIC_LANGS = ("hi", "mr", "bn", "pa", "gu", "or", "ur", "ta", "te", "kn", "ml")

DEFAULT_PARAMS = {
    "min_count": 3,  # times the pair was glossed in IndicVoices
    "min_gloss_share": 0.8,  # glossed occurrences / all occurrences of the native word
    "min_dominance": 0.9,  # this English word / all glosses of the native word
    "min_letters": 3,  # letters (base characters, not marks) in the native word
    "min_english_letters": 3,
    "max_skeleton_ratio": 0.34,  # whole-word consonant-skeleton edit distance / English skeleton length
    "max_word_level": 50,  # the English side must be a common English word (SCOWL size <= this) ...
    "english_extra": [],  # ... or one of these (brands and app names people type in Latin)
    "english_block": [],  # never a target: common-word homographs of country names (japan, china, turkey)
    "secondary": {},  # lang -> true: enable the word-list path (none by default)
    "secondary_max_level": 35,  # SCOWL size cut-off for the word-list path
    "secondary_min_skeleton": 3,
    "secondary_max_vowel_ratio": 0.25,
}

_GLOSS = re.compile(r"\[([^\[\]]*)\]")
_MARKUP = re.compile(r"\{[^{}]*\}|<[^<>]*>|[()]")
_LATIN_GLOSS = re.compile(r"^[A-Za-z][A-Za-z'.-]*$")


def letters(word: str) -> int:
    return sum(1 for c in word if unicodedata.category(c)[0] == "L")


def skeleton(word: str) -> list[str]:
    return consonant_skeleton(phonetic.word_units(word))


def english_skeleton(word: str) -> list[str]:
    return consonant_skeleton(phonetic.english_units(word.lower().replace("sch", "sk")))


def whole_word_match(native: str, english: str, max_ratio: float) -> bool:
    """The whole native word (no trailing case suffix) has the English word's consonant skeleton,
    up to `max_ratio` edits per skeleton class (exact for skeletons of 1-2 classes)."""
    a, b = skeleton(native), english_skeleton(english)
    if not a or not b:
        return not a and not b
    candidates = [b, b[1:]] if len(b) > 1 and b[0] == "V" else [b]
    for target in candidates:
        distance = _edit_distance(a, target)
        if (distance == 0) if len(target) <= 2 else distance / len(target) <= max_ratio:
            return True
    return False


# Broad vowel classes for the word-list path: length and diphthong marks ignored.
_VOWEL_CLASS = {"a": "a", "ā": "a", "i": "i", "ī": "i", "u": "u", "ū": "u", "e": "e", "ē": "e", "ai": "e",
                "o": "o", "ō": "o", "au": "o", "ê": "e", "ô": "o"}


def _broad(units: list[str]) -> list[str]:
    out = []
    for unit in units:
        if unit == phonetic.BOUNDARY:
            continue
        if unit in _VOWEL_CLASS:
            out.append(_VOWEL_CLASS[unit])
        else:
            out.extend(consonant_skeleton([unit]) or [])
    return out


# ---------------------------------------------------------------- the normaliser


class LoanwordNormaliser:
    def __init__(self, lexicon: dict[str, dict[str, str]], allow: dict[str, dict[str, str]],
                 block: dict[str, list[str]], params: dict, english: dict[str, int] | None = None,
                 files: dict[str, bytes] | None = None):
        self.lexicon = {lang: dict(entries) for lang, entries in lexicon.items()}
        self.allow = {lang: dict(entries) for lang, entries in allow.items()}
        self.block = {lang: frozenset(words) for lang, words in block.items()}
        self.params = {**DEFAULT_PARAMS, **params}
        self.english = english or {}
        self._files = files or {}
        self._index: dict[tuple[str, ...], list[str]] | None = None

    @classmethod
    def load(cls, directory: Path | None = None) -> LoanwordNormaliser:
        directory = Path(directory) if directory is not None else DEFAULT_DIR
        files: dict[str, bytes] = {}

        def read(name: str, default):
            path = directory / name
            if not path.exists():
                return default
            files[name] = path.read_bytes()
            return json.loads(files[name].decode("utf-8"))

        lexicon = read("lexicon.json", {})
        lexicon = {lang: {native: entry[0] if isinstance(entry, list) else entry.get("en") if isinstance(entry, dict)
                          else entry for native, entry in entries.items()}
                   for lang, entries in lexicon.get("langs", lexicon).items()}
        allow = read("allow.json", {})
        block = read("block.json", {})
        params = read("params.json", {})
        english = None
        if any((params.get("secondary") or {}).values()):
            english = load_english_words(ENGLISH_WORDS)
            files["english_words.tsv"] = ENGLISH_WORDS.read_bytes()
        return cls(lexicon, allow, block, params, english, files)

    # ------------------------------------------------------------ identity

    def fingerprint(self) -> str:
        """sha256 over the resource files' bytes and the effective parameters (file names sorted,
        so it is the same on every machine)."""
        digest = hashlib.sha256()
        for name in sorted(self._files):
            digest.update(name.encode() + b"\0" + hashlib.sha256(self._files[name]).digest())
        digest.update(json.dumps(self.params, sort_keys=True).encode())
        digest.update(f"v{VERSION}".encode())
        return digest.hexdigest()

    def describe(self) -> dict:
        return {
            "version": VERSION, "fingerprint": self.fingerprint(), "params": self.params,
            "entries": {lang: len(entries) for lang, entries in sorted(self.lexicon.items())},
            "allow": {lang: len(v) for lang, v in sorted(self.allow.items()) if v},
            "block": {lang: len(v) for lang, v in sorted(self.block.items()) if v},
            "word_list": "SCOWL 2020.12.07 sizes <= 50 (permissive SCOWL licence; indicear/resources/SCOWL-LICENSE.txt): "
                         "filters the mined lexicon at build time" + ("; also used at runtime by the word-list path"
                                                                      if self.english else ""),
        }

    # ------------------------------------------------------------ rewriting

    def lookup(self, word: str, lang: str) -> str | None:
        """The Latin form of one native-script word, or None to keep it."""
        if word in self.block.get(lang, ()):
            return None
        forced = self.allow.get(lang, {}).get(word)
        if forced:
            return forced
        found = self.lexicon.get(lang, {}).get(word)
        if found:
            return found
        if self.params["secondary"].get(lang) and self.english:
            return self._secondary(word)
        return None

    def rewrite(self, text: str, lang: str) -> tuple[str, int]:
        if lang == "en" or lang not in LANG_SCRIPT or not (
                self.lexicon.get(lang) or self.allow.get(lang) or self.params["secondary"].get(lang)):
            return text, 0
        own = LANG_SCRIPT[lang]
        words = text.split(" ")
        count = 0
        for index, word in enumerate(words):
            if not word or word_script(word) != own:
                continue
            latin = self.lookup(word, lang)
            if latin:
                words[index] = latin
                count += 1
        return (" ".join(words), count) if count else (text, 0)

    # ------------------------------------------------------------ word-list path

    def _build_index(self) -> dict[tuple[str, ...], list[str]]:
        index: dict[tuple[str, ...], list[str]] = defaultdict(list)
        cut = int(self.params["secondary_max_level"])
        for word, level in self.english.items():
            if level <= cut and len(word) >= self.params["min_english_letters"]:
                index[tuple(english_skeleton(word))].append(word)
        return index

    def _secondary(self, word: str) -> str | None:
        return _secondary_cached(self, word)

    def secondary_candidate(self, word: str) -> str | None:
        if letters(word) < self.params["min_letters"]:
            return None
        if self._index is None:
            self._index = self._build_index()
        key = tuple(skeleton(word))
        if len(key) < self.params["secondary_min_skeleton"]:
            return None
        native = _broad(phonetic.word_units(word))
        scored = []
        for english in self._index.get(key, ()):
            if not sounds_alike([word], [english]):
                continue
            target = _broad(phonetic.english_units(english))
            scored.append((_edit_distance(native, target) / max(len(target), 1), self.english[english], english))
        if not scored:
            return None
        scored.sort()
        best = scored[0]
        if best[0] > self.params["secondary_max_vowel_ratio"]:
            return None
        if len(scored) > 1 and scored[1][0] - best[0] < 0.1:  # no unique best candidate
            return None
        return best[2]


@lru_cache(maxsize=200_000)
def _secondary_cached(normaliser: LoanwordNormaliser, word: str) -> str | None:
    return normaliser.secondary_candidate(word)


def english_level(word: str, english_words: dict[str, int]) -> int:
    """SCOWL size of a word (999 when absent)."""
    return english_words.get(word, 999)


def load_english_words(path: Path | None = None) -> dict[str, int]:
    """word -> SCOWL size (10 = most common); lines 'word<TAB>size', '#' comments."""
    out = {}
    for line in Path(path or ENGLISH_WORDS).read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            word, _, level = line.partition("\t")
            out[word] = int(level)
    return out


# ---------------------------------------------------------------- mining (build)


def gloss_pairs(raw: str) -> Iterable[tuple[str, str | None]]:
    """IndicVoices `unsanitized_normalized` -> (native-or-Latin word, single-word gloss or None)
    for every word outside brackets. Tags ([noise], <laughter>, ...) are dropped; a word followed
    by a one-word Latin gloss yields that gloss; multi-word glosses yield None for every word."""
    text = _MARKUP.sub(" ", raw)
    position = 0
    out: list[list] = []
    for match in _GLOSS.finditer(text):
        words = text[position:match.start()].split()
        out.extend([w, None] for w in words)
        position = match.end()
        content = match.group(1).strip()
        if not content or is_indicvoices_tag(content):
            continue
        parts = content.split()
        if len(parts) == 1 and _LATIN_GLOSS.match(parts[0]) and out and out[-1][1] is None and words:
            out[-1][1] = parts[0]
    out.extend([w, None] for w in text[position:].split())
    return [(w, g) for w, g in out]


def mine(texts: Iterable[str], lang: str, params: dict | None = None,
         english: dict[str, int] | None = None) -> tuple[dict, dict]:
    """Mine (lexicon entries, statistics) for one language from IndicVoices raw transcripts.
    `english` (word -> SCOWL size; default the packaged list) restricts the English side to common
    English words plus `params["english_extra"]`: proper names (countries, people, Indian places)
    are left in the native script."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    vocab = load_english_words() if english is None else english
    extra = set(params["english_extra"])
    blocked = set(params["english_block"])
    own = LANG_SCRIPT[lang]
    seen, glossed = Counter(), Counter()
    pairs: dict[str, Counter] = defaultdict(Counter)
    for raw in texts:
        for word, gloss in gloss_pairs(raw):
            norm = normalize(word, lang)
            if " " in norm or not norm or word_script(norm) != own:
                continue
            seen[norm] += 1
            if gloss:
                glossed[norm] += 1
                pairs[norm][normalize(gloss, "en")] += 1
    lexicon, rejected = {}, Counter()
    for native, glosses in pairs.items():
        english, count = glosses.most_common(1)[0]
        if count < params["min_count"]:
            rejected["min_count"] += 1
            continue
        if count / sum(glosses.values()) < params["min_dominance"]:
            rejected["not_dominant"] += 1
            continue
        if glossed[native] / seen[native] < params["min_gloss_share"]:
            rejected["often_unglossed"] += 1
            continue
        if letters(native) < params["min_letters"] or letters(english) < params["min_english_letters"]:
            rejected["too_short"] += 1
            continue
        if not english.isascii() or not english.replace("'", "").isalpha():
            rejected["not_a_word"] += 1
            continue
        if english in blocked:
            rejected["blocked_english"] += 1
            continue
        if english not in extra and english_level(english, vocab) > params["max_word_level"]:
            rejected["not_common_english"] += 1
            continue
        if not sounds_alike([native], [english]):
            rejected["sounds_unlike"] += 1
            continue
        if not whole_word_match(native, english, params["max_skeleton_ratio"]):
            rejected["suffix_or_partial"] += 1
            continue
        lexicon[native] = {"en": english, "count": count, "share": round(glossed[native] / seen[native], 3)}
    stats = {"native_types": len(seen), "glossed_types": len(pairs), "entries": len(lexicon),
             "rejected": dict(rejected), "tokens": sum(seen.values()), "glossed_tokens": sum(glossed.values())}
    return dict(sorted(lexicon.items())), stats


def read_texts(path: Path) -> list[str]:
    """One JSON string per line ("\\n" only: str.splitlines would also split on U+2028 inside a string)."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def dump_lexicon(record: dict) -> str:
    """lexicon.json with one line per entry ("native": ["english", count, glossed share]), so diffs stay
    readable and the file small."""
    head = {k: v for k, v in record.items() if k != "langs"}
    lines = ["{", *(f" {json.dumps(k)}: {json.dumps(v, ensure_ascii=False, sort_keys=True)}," for k, v in head.items()),
             ' "langs": {']
    langs = list(record["langs"].items())
    for index, (lang, entries) in enumerate(langs):
        lines.append(f"  {json.dumps(lang)}: {{")
        items = list(entries.items())
        for position, (native, entry) in enumerate(items):
            value = [entry["en"], entry["count"], entry["share"]]
            comma = "," if position < len(items) - 1 else ""
            lines.append(f"   {json.dumps(native, ensure_ascii=False)}: {json.dumps(value)}{comma}")
        lines.append("  }" + ("," if index < len(langs) - 1 else ""))
    lines += [" }", "}"]
    return "\n".join(lines) + "\n"


def cmd_build(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params_path = out / "params.json"
    params = {**DEFAULT_PARAMS, **(json.loads(params_path.read_text()) if params_path.exists() else {})}
    langs, stats = {}, {}
    for lang in args.langs.split(","):
        sources = sorted(Path(args.iv).glob(f"iv_{lang}.jsonl")) + sorted(Path(args.iv).glob(f"iv_{lang}_*.jsonl"))
        if not sources:
            print(f"{lang}: no iv_{lang}*.jsonl in {args.iv}, skipped")
            continue
        texts = [t for source in sources for t in read_texts(source)]
        langs[lang], stats[lang] = mine(texts, lang, params)
        stats[lang]["iv_rows"] = len(texts)
        print(f"{lang}: {stats[lang]}")
    record = {"version": VERSION, "source": "IndicVoices unsanitized_normalized glosses (ai4bharat/IndicVoices "
              "@c96f9088, CC BY 4.0), train split, text column only", "params": params,
              "stats": stats, "langs": langs}
    (out / "lexicon.json").write_text(dump_lexicon(record), encoding="utf-8")
    if not params_path.exists():
        params_path.write_text(json.dumps(DEFAULT_PARAMS, indent=1) + "\n", encoding="utf-8")


def cmd_review(args) -> None:
    """Apply the normaliser to native-only text (one JSON string per line, `<prefix>_<lang>.jsonl`)
    and sample rewrites (with context) for manual review."""
    normaliser = LoanwordNormaliser.load(Path(args.resources) if args.resources else None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for lang in args.langs.split(","):
        directory = Path(args.text)
        sources = sorted(directory.glob(f"{args.prefix}_{lang}.jsonl")) + sorted(
            directory.glob(f"{args.prefix}_{lang}_*.jsonl"))
        if not sources:
            continue
        rewrites, words = [], 0
        for row, raw in enumerate(t for source in sources for t in read_texts(source)):
            text = normalize(raw, lang)
            tokens = text.split()
            words += len(tokens)
            for index, token in enumerate(tokens):
                if word_script(token) != LANG_SCRIPT[lang]:
                    continue
                latin = normaliser.lookup(token, lang)
                if latin:
                    path = "lexicon" if token in normaliser.lexicon.get(lang, {}) else (
                        "allow" if token in normaliser.allow.get(lang, {}) else "word_list")
                    rewrites.append({"native": token, "en": latin, "path": path, "row": row,
                                     "context": " ".join(tokens[max(0, index - 4): index + 5])})
        sample = random.Random(f"review:{lang}").sample(rewrites, min(args.per_lang, len(rewrites)))
        summary[lang] = {"words": words, "rewrites": len(rewrites),
                         "per_1000_words": round(1000 * len(rewrites) / max(words, 1), 2),
                         "types": len({r["native"] for r in rewrites}), "sampled": len(sample)}
        (out / f"{lang}.json").write_text(json.dumps({"lang": lang, **summary[lang], "sample": sample},
                                                     ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(lang, summary[lang])
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="mine the lexicon from IndicVoices transcripts (iv_<lang>.jsonl)")
    build.add_argument("--iv", required=True, help="directory of iv_<lang>[_*].jsonl (one raw transcript per line)")
    build.add_argument("--out", default=str(DEFAULT_DIR))
    build.add_argument("--langs", default=",".join(INDIC_LANGS))
    review = sub.add_parser("review", help="sample rewrites of native-only text for manual review")
    review.add_argument("--text", required=True, help="directory of <prefix>_<lang>[_*].jsonl")
    review.add_argument("--prefix", default="kb")
    review.add_argument("--out", required=True)
    review.add_argument("--resources", default=None)
    review.add_argument("--langs", default=",".join(INDIC_LANGS))
    review.add_argument("--per-lang", type=int, default=100)
    args = parser.parse_args(argv)
    {"build": cmd_build, "review": cmd_review}[args.command](args)


if __name__ == "__main__":
    main()
