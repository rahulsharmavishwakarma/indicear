"""Metrics for the ablation and robustness reports. References and hypotheses are normalized
with `indicear.text.normalize` and then `score_fold` (scoring-only spelling folds: Malayalam
chillus, Bengali khanda ta, Marathi eyelash ra, Urdu heh/tatweel/vowel marks/presentation forms,
native digits) before scoring.

- WER / CER: corpus-level (jiwer 4.0.0; CER counts spaces as characters).
- Script-error rate: share of hypothesis words whose script does not occur in the reference
  utterance (e.g. Urdu output for Hindi speech, Latin output for Kannada speech).
- OOV-word recall: of reference words never seen in the training transcripts, the share
  recognized exactly (aligned as a hit).
- Code-switch span accuracy (synthetic concatenated set): a span is correct when the
  hypothesis words in that span's script, in order, have CER <= `span_cer` against the span
  reference, and the two spans' scripts appear in the right order. Also reports per-span CER.
- Script-normalised WER (`script_normalised_wer`): WER where an English word written in native
  script and the same word in Latin both count as correct. Rows carry `refs`, alternative
  references (the loader stores the native-script form of glossed rows there). Each alternative
  is word-aligned to the primary reference (difflib, no autojunk); a differing block counts as an
  equivalent spelling only when one side is in Latin and the other only in a native script, so
  same-script spelling variants are not accepted. Blocks of equal length pair word by word (each
  word may take either form); otherwise the whole block takes one form. The reference becomes a lattice of segments
  (fixed words, or {Latin form, native form}); the distance is the minimum word edit distance
  of the hypothesis to any path, so a hypothesis may mix both forms within one utterance. The
  denominator is the primary reference's word count. Rows without `refs` score as strict WER:
  with no stored alternative (Kathbath, FLEURS, MUCS), no transliteration is attempted.
- Real code-switch span accuracy (`code_mixed_spans`, rows with `word_langs` holding more than one
  language run): each reference run of one language is a span; the hypothesis words aligned to
  it (jiwer word alignment; insertions strictly inside the span included) must be in the span
  language's script and have CER <= `span_cer` against the span.
- Negatives: share of non-speech clips with any output, and words emitted per clip.
- Language identification (`lid_report`, `lid_span_report`): utterance accuracy, macro-F1, the
  confusion matrix with the Hindi/Urdu, Hindi/Marathi and Punjabi/Hindi pairs, accuracy by
  duration bucket, and language-span accuracy on code-switched sets.
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher

import jiwer

from indicear.text import LANG_SCRIPT, normalize, word_script

# Scoring-only folds, applied to references and hypotheses alike (never to training targets):
# spellings that write the same sound in two Unicode conventions, which `normalize` keeps apart.
_SCORE_FOLD = str.maketrans({
    # Malayalam atomic chillus vs consonant + virama (the old ZWJ form, whose ZWJ normalize drops)
    "\u0d7a": "\u0d23\u0d4d", "\u0d7b": "\u0d28\u0d4d", "\u0d7c": "\u0d30\u0d4d", "\u0d7d": "\u0d32\u0d4d",
    "\u0d7e": "\u0d33\u0d4d", "\u0d7f": "\u0d15\u0d4d",
    "\u09ce": "\u09a4\u09cd",  # Bengali khanda ta vs ta + hasanta (the ZWJ form)
    "\u0931": "\u0930",  # Marathi eyelash ra vs ra (+ virama + ZWJ, whose ZWJ normalize drops)
    "\u0647": "\u06c1",  # Urdu: Arabic heh written for heh goal
    "\u0640": None,  # tatweel
    **{chr(c): None for c in (*range(0x064B, 0x0653), 0x0670)},  # Arabic short-vowel marks, superscript alef
})


def score_fold(text: str) -> str:
    """`normalize`, then fold scoring-only variants: Malayalam chillus, Bengali khanda ta, Marathi
    eyelash ra, Urdu presentation forms (NFKC), heh, tatweel and vowel marks, and native digits to
    ASCII. Applied to both sides of every metric here."""
    text = normalize(text)
    if any("\ufb50" <= c <= "\ufdff" or "\ufe70" <= c <= "\ufeff" for c in text):
        text = normalize(unicodedata.normalize("NFKC", text))
    text = text.translate(_SCORE_FOLD)
    return "".join(str(unicodedata.digit(c)) if c > "\x7f" and unicodedata.category(c) == "Nd" else c for c in text)


_norm = score_fold


def _clean(texts: Iterable[str]) -> list[str]:
    return [_norm(t) for t in texts]


def wer_cer(refs: Sequence[str], hyps: Sequence[str]) -> dict:
    refs, hyps = _clean(refs), _clean(hyps)
    pairs = [(r, h) for r, h in zip(refs, hyps, strict=True) if r]
    if not pairs:
        return {"wer": None, "cer": None, "utterances": 0}
    r, h = zip(*pairs, strict=True)
    return {
        "wer": round(100 * jiwer.wer(list(r), list(h)), 2),
        "cer": round(100 * jiwer.cer(list(r), list(h)), 2),
        "utterances": len(pairs),
        "words": sum(len(x.split()) for x in r),
    }


def script_error_rate(refs: Sequence[str], hyps: Sequence[str]) -> float | None:
    wrong = total = 0
    for ref, hyp in zip(_clean(refs), _clean(hyps), strict=True):
        allowed = {word_script(w) for w in ref.split()} - {None}
        for word in hyp.split():
            script = word_script(word)
            if script is None:
                continue  # digits carry no script
            total += 1
            wrong += script not in allowed
    return round(100 * wrong / total, 2) if total else None


def oov_recall(refs: Sequence[str], hyps: Sequence[str], train_vocab: set[str]) -> dict:
    refs, hyps = _clean(refs), _clean(hyps)
    pairs = [(r, h) for r, h in zip(refs, hyps, strict=True) if r]
    if not pairs:
        return {"oov_recall": None, "oov_words": 0}
    output = jiwer.process_words([p[0] for p in pairs], [p[1] for p in pairs])
    oov = hit = 0
    for ref_words, chunks in zip(output.references, output.alignments, strict=True):
        hits = set()
        for chunk in chunks:
            if chunk.type == "equal":
                hits.update(range(chunk.ref_start_idx, chunk.ref_end_idx))
        for index, word in enumerate(ref_words):
            if word not in train_vocab:
                oov += 1
                hit += index in hits
    return {"oov_recall": round(100 * hit / oov, 2) if oov else None, "oov_words": oov}


def code_switch_spans(items: Sequence[dict], hyps: Sequence[str], span_cer: float = 0.5) -> dict:
    """items carry `spans` (two dicts with script and text) in spoken order."""
    correct = total = order_ok = 0
    span_cers: list[float] = []
    for item, hyp in zip(items, _clean(hyps), strict=True):
        words = hyp.split()
        positions: dict[str, list[int]] = defaultdict(list)
        for index, word in enumerate(words):
            positions[word_script(word) or ""].append(index)
        spans = item["spans"]
        first, second = spans[0]["script"], spans[1]["script"]
        ordered = bool(positions[first]) and bool(positions[second]) and (
            sorted(positions[first])[len(positions[first]) // 2]
            < sorted(positions[second])[len(positions[second]) // 2]
        )
        order_ok += ordered
        for span in spans:
            total += 1
            got = " ".join(words[i] for i in positions[span["script"]])
            ref = _norm(span["text"])
            error = jiwer.cer(ref, got) if got else 1.0
            span_cers.append(error)
            correct += bool(got) and error <= span_cer and ordered
    return {
        "span_accuracy": round(100 * correct / total, 2) if total else None,
        "span_cer_mean": round(100 * sum(span_cers) / len(span_cers), 2) if span_cers else None,
        "order_accuracy": round(100 * order_ok / len(items), 2) if items else None,
        "span_cer_threshold": span_cer,
    }


# ---------------------------------------------------------------- script-normalised WER


def _latin(words: Sequence[str]) -> bool:
    return any(word_script(w) == "latn" for w in words)


def _native_only(words: Sequence[str]) -> bool:
    scripts = {word_script(w) for w in words} - {None}
    return bool(scripts) and "latn" not in scripts


def script_swap(a: Sequence[str], b: Sequence[str]) -> bool:
    """One side is (partly) Latin and the other only in a native script."""
    return bool(a) and bool(b) and ((_latin(a) and _native_only(b)) or (_latin(b) and _native_only(a)))


def reference_lattice(primary: Sequence[str], alternate: Sequence[str]) -> list[list[tuple[str, ...]]]:
    """Segments of alternatives: fixed words, or {primary block, alternate block} where the two
    references differ only by writing the same words in Latin vs a native script."""
    segments: list[list[tuple[str, ...]]] = []
    matcher = SequenceMatcher(None, list(primary), list(alternate), autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        block, other = tuple(primary[i1:i2]), tuple(alternate[j1:j2])
        if tag != "equal" and len(block) == len(other) and all(
                script_swap((a,), (b,)) for a, b in zip(block, other, strict=True)):
            segments.extend([(a,), (b,)] for a, b in zip(block, other, strict=True))  # word by word
        elif tag != "equal" and script_swap(block, other):
            segments.append([block, other])
        else:
            segments.extend([(word,)] for word in block)
    return segments


def lattice_distance(segments: Sequence[Sequence[tuple[str, ...]]], hyp: Sequence[str]) -> int:
    """Minimum word edit distance from `hyp` to any path through the lattice."""
    row = list(range(len(hyp) + 1))
    for alternatives in segments:
        best: list[int] | None = None
        for words in alternatives:
            current = row
            for word in words:
                new = [current[0] + 1] + [0] * len(hyp)
                for j in range(1, len(hyp) + 1):
                    new[j] = min(current[j] + 1, new[j - 1] + 1, current[j - 1] + (word != hyp[j - 1]))
                current = new
            best = current if best is None else [min(x, y) for x, y in zip(best, current, strict=True)]
        row = best if best is not None else row
    return row[-1]


def script_normalised_wer(items: Sequence[dict], hyps: Sequence[str]) -> dict:
    """WER counting Latin and native-script spellings of the same words as equal (see the
    module docstring), and the matching CER: character edit distance to the closest whole
    reference (primary or an alternative; no mixing within a row), over the primary's characters.
    Returns both and how many rows had an alternative reference."""
    from rapidfuzz.distance import Levenshtein

    errors = words = with_alt = char_errors = chars = 0
    for item, hyp in zip(items, hyps, strict=True):
        primary = _norm(item["text"]).split()
        if not primary:
            continue
        hyp_words = _norm(hyp).split()
        alternates = [_norm(r).split() for r in item.get("refs") or ()]
        alternates = [a for a in alternates if a and a != primary]
        best = lattice_distance([[(w,)] for w in primary], hyp_words)
        for alternate in alternates:
            segments = reference_lattice(primary, alternate)
            if any(len(s) > 1 for s in segments):
                best = min(best, lattice_distance(segments, hyp_words))
        with_alt += bool(alternates)
        errors += best
        words += len(primary)
        reference = " ".join(primary)
        char_errors += min(Levenshtein.distance(" ".join(r), " ".join(hyp_words)) for r in [primary, *alternates])
        chars += len(reference)
    return {"wer_script_normalised": round(100 * errors / words, 2) if words else None,
            "cer_script_normalised": round(100 * char_errors / chars, 2) if chars else None,
            "rows_with_alternative_ref": with_alt}


# ---------------------------------------------------------------- real code-switched rows


def _runs(word_langs: Sequence[str]) -> list[tuple[str, int, int]]:
    runs: list[tuple[str, int, int]] = []
    for index, lang in enumerate(word_langs):
        if runs and runs[-1][0] == lang:
            runs[-1] = (lang, runs[-1][1], index + 1)
        else:
            runs.append((lang, index, index + 1))
    return runs


def row_word_langs(item: dict, words: Sequence[str]) -> list[str] | None:
    """The row's per-word languages for its normalized words (recomputed from script when the
    stored tags do not line up with the normalized words)."""
    from indicear.markup import word_languages

    tags = item.get("word_langs")
    if tags is not None and len(tags) == len(words):
        return list(tags)
    lang = item.get("lang")
    return word_languages(" ".join(words), lang) if lang in LANG_SCRIPT and words else None


def code_mixed_spans(items: Sequence[dict], hyps: Sequence[str], span_cer: float = 0.5) -> dict:
    """Span accuracy on real code-mixed rows (more than one language run in `word_langs`)."""
    total = correct = utterances = 0
    cers: list[float] = []
    by_lang: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for item, hyp in zip(items, hyps, strict=True):
        ref_words = _norm(item["text"]).split()
        tags = row_word_langs(item, ref_words)
        if not tags or len({t for t in tags}) < 2:
            continue
        hyp_words = _norm(hyp).split()
        utterances += 1
        matched: dict[int, int] = {}  # ref index -> its hyp index (equal or substitute)
        inserted: dict[int, list[int]] = defaultdict(list)  # ref index -> hyp words inserted after it
        if hyp_words:
            output = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
            for chunk in output.alignments[0]:
                if chunk.type in ("equal", "substitute"):
                    for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                        matched[chunk.ref_start_idx + k] = chunk.hyp_start_idx + k
                elif chunk.type == "insert" and chunk.ref_start_idx > 0:
                    inserted[chunk.ref_start_idx - 1].extend(range(chunk.hyp_start_idx, chunk.hyp_end_idx))
        for lang, start, end in _runs(tags):
            positions = [matched[i] for i in range(start, end) if i in matched]
            positions += [j for i in range(start, end - 1) for j in inserted.get(i, ())]  # inside the span only
            positions.sort()
            got = " ".join(hyp_words[j] for j in positions)
            ref = " ".join(ref_words[start:end])
            error = jiwer.cer(ref, got) if got else 1.0
            script = word_script(got) if got else None
            ok = bool(got) and error <= span_cer and script == LANG_SCRIPT.get(lang)
            total += 1
            correct += ok
            cers.append(error)
            by_lang[lang][0] += ok
            by_lang[lang][1] += 1
    return {
        "cm_utterances": utterances, "cm_spans": total,
        "cm_span_accuracy": round(100 * correct / total, 2) if total else None,
        "cm_span_cer_mean": round(100 * sum(cers) / len(cers), 2) if cers else None,
        "cm_span_accuracy_by_lang": {k: round(100 * a / n, 2) for k, (a, n) in sorted(by_lang.items())},
        "span_cer_threshold": span_cer,
    }


def negatives(hyps: Sequence[str]) -> dict:
    hyps = _clean(hyps)
    if not hyps:
        return {"clips": 0}
    return {
        "clips": len(hyps),
        "false_output_rate": round(100 * sum(bool(h) for h in hyps) / len(hyps), 2),
        "words_per_clip": round(sum(len(h.split()) for h in hyps) / len(hyps), 3),
    }


def train_vocabulary(texts: Iterable[str]) -> set[str]:
    return {word for text in texts for word in _norm(text).split()}


def report(items: Sequence[dict], hyps: Sequence[str], train_vocab: set[str]) -> dict:
    """Per-language metrics for monolingual items, span metrics for code-switch items."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        if item.get("negative"):
            groups["negatives"].append(index)
        elif "spans" in item:
            groups[f"cs:{item['lang']}"].append(index)
        else:
            groups[item["lang"]].append(index)
    out: dict[str, dict] = {}
    for name, indexes in sorted(groups.items()):
        sub_items = [items[i] for i in indexes]
        refs = [i["text"] for i in sub_items]
        sub_hyps = [hyps[i] for i in indexes]
        if name == "negatives":
            out[name] = negatives(sub_hyps)
            continue
        row = wer_cer(refs, sub_hyps)
        row["script_error_rate"] = script_error_rate(refs, sub_hyps)
        if name.startswith("cs:"):
            row.update(code_switch_spans(sub_items, sub_hyps))
        else:
            row.update(oov_recall(refs, sub_hyps, train_vocab))
        out[name] = row
    return out


# ---------------------------------------------------------------- language identification
#
# The LID heads (indicear.lid) are outputs, never routing; these score them. Utterance LID:
# accuracy, macro-F1 over the reference languages, the confusion matrix with the confusable
# pairs called out, and accuracy by utterance duration. Spans: code-switched sets.

LID_PAIRS = (("hi", "ur"), ("hi", "mr"), ("pa", "hi"))  # same sound (hi/ur), same script (hi/mr), neighbours (pa/hi)
DURATION_BUCKETS = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, float("inf")))


def _bucket_name(low: float, high: float) -> str:
    return f"{low:g}-{high:g}s" if high != float("inf") else f">{low:g}s"


def lid_report(refs: Sequence[str], hyps: Sequence[str], seconds: Sequence[float] | None = None) -> dict:
    """Utterance LID: refs/hyps are language codes per utterance (rows with an empty or "none"
    reference are skipped). Macro-F1 averages per-language F1 over the reference languages."""
    pairs = [(r, h) for r, h in zip(refs, hyps, strict=True) if r and r != "none"]
    durations = None
    if seconds is not None:
        durations = [s for r, s in zip(refs, seconds, strict=True) if r and r != "none"]
    if not pairs:
        return {"utterances": 0, "accuracy": None, "macro_f1": None}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ref, hyp in pairs:
        confusion[ref][hyp] += 1
    labels = sorted({r for r, _ in pairs})
    per_lang = {}
    for lang in labels:
        tp = confusion[lang][lang]
        support = sum(confusion[lang].values())
        predicted = sum(row.get(lang, 0) for row in confusion.values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_lang[lang] = {"precision": round(100 * precision, 2), "recall": round(100 * recall, 2),
                          "f1": round(100 * f1, 2), "support": support}
    correct = sum(r == h for r, h in pairs)
    out = {
        "utterances": len(pairs),
        "accuracy": round(100 * correct / len(pairs), 2),
        "macro_f1": round(sum(v["f1"] for v in per_lang.values()) / len(per_lang), 2),
        "per_lang": per_lang,
        "confusion": {ref: dict(sorted(row.items())) for ref, row in sorted(confusion.items())},
        "pairs": {},
    }
    for a, b in LID_PAIRS:
        a_total, b_total = sum(confusion[a].values()), sum(confusion[b].values())
        out["pairs"][f"{a}/{b}"] = {
            f"{a}->{b}": confusion[a].get(b, 0), f"{b}->{a}": confusion[b].get(a, 0),
            f"{a}_utterances": a_total, f"{b}_utterances": b_total,
            f"{a}_as_{b}_rate": round(100 * confusion[a].get(b, 0) / a_total, 2) if a_total else None,
            f"{b}_as_{a}_rate": round(100 * confusion[b].get(a, 0) / b_total, 2) if b_total else None,
        }
    if durations is not None:
        buckets = {}
        for low, high in DURATION_BUCKETS:
            rows = [(r, h) for (r, h), s in zip(pairs, durations, strict=True) if low <= s < high]
            if rows:
                buckets[_bucket_name(low, high)] = {
                    "utterances": len(rows), "accuracy": round(100 * sum(r == h for r, h in rows) / len(rows), 2)}
        out["by_duration"] = buckets
    return out


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            table[i + 1][j + 1] = table[i][j] + 1 if x == y else max(table[i][j + 1], table[i + 1][j])
    return table[-1][-1]


def _collapse(langs: Iterable[str]) -> list[str]:
    out: list[str] = []
    for lang in langs:
        if not out or out[-1] != lang:
            out.append(lang)
    return out


def lid_span_report(
    references: Sequence[Sequence[dict]], hypotheses: Sequence[Sequence[tuple[str, int, int]]], frame_rate: float = 25.0
) -> dict:
    """Language-span accuracy on code-switched sets.

    references: per utterance, its spans in spoken order as dicts with "lang" and optionally
    "start"/"end" in seconds. hypotheses: per utterance, the ordered (lang, start_frame,
    end_frame) spans of indicear.lid (end exclusive).

    - sequence_accuracy: the hypothesis's collapsed language sequence equals the reference's.
    - span_accuracy: share of reference spans recovered. With times, a span is recovered when
      one hypothesis language covers more than half of its frames and that language is the
      span's; without times, it counts the reference spans matched in order (longest common
      subsequence of the two language sequences).
    - frame_accuracy (timed references only): share of reference-span frames whose hypothesis
      language is right.
    """
    total = recovered = sequence_ok = timed_frames = right_frames = 0
    for ref_spans, hyp_spans in zip(references, hypotheses, strict=True):
        ref_seq = _collapse(s["lang"] for s in ref_spans)
        hyp_seq = _collapse(lang for lang, _, _ in hyp_spans)
        sequence_ok += ref_seq == hyp_seq
        timed = all("start" in s and "end" in s for s in ref_spans)
        if not timed:
            total += len(ref_spans)
            recovered += _lcs([s["lang"] for s in ref_spans], [lang for lang, _, _ in hyp_spans])
            continue
        for span in ref_spans:
            start, end = round(span["start"] * frame_rate), round(span["end"] * frame_rate)
            if end <= start:
                continue
            cover: dict[str, int] = defaultdict(int)
            for lang, h_start, h_end in hyp_spans:
                overlap = min(end, h_end) - max(start, h_start)
                if overlap > 0:
                    cover[lang] += overlap
            total += 1
            timed_frames += end - start
            right_frames += cover.get(span["lang"], 0)
            best = max(cover, key=cover.__getitem__) if cover else None
            recovered += best == span["lang"] and 2 * cover[best] > end - start
    return {
        "utterances": len(references),
        "spans": total,
        "span_accuracy": round(100 * recovered / total, 2) if total else None,
        "sequence_accuracy": round(100 * sequence_ok / len(references), 2) if references else None,
        "frame_accuracy": round(100 * right_frames / timed_frames, 2) if timed_frames else None,
    }
