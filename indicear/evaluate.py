"""Evaluation on the fixed eval sets: decoding (cached or live lower), scoring and the result files.

Shared by `ablate train` (the dev report), `ablate eval` (every eval set, every variant) and
`indicear.baselines` (other systems on the same sets, same metrics, same files).

Live lower for evaluation: utterances are read from their local FLAC files (the eval sets were
decoded by `ablate prep --stream`), loaded and optionally corrupted in a thread pool, batched by
padded frames, featurized and run through layers 1-6 ONCE per chunk; the layer-6 states go into an
in-memory `RowCache`, and every model then decodes from it with `decode_items`, exactly as the
cached path does. A chunk holds about `chunk_seconds` of audio (default one hour: ~180 MB of bf16
states), so any eval set fits. Several models share one tower: each model's adapters are attached
only while it decodes.

Result files: `results/<system>/<eval set>.json` with "kind": "eval", per decoder and per language
(plus "all"): CER, WER, the script-normalised WER, the script-error rate, real code-switch span
accuracy on code-mixed rows, synthetic span accuracy on cs_dev, the non-speech false-output rate,
and, for models with LID heads, `metrics.lid_report` / `lid_span_report`. `summarize` writes
results/eval_summary.{md,json} across every system.

Eval sets (`eval_sets`):
- every `manifests/eval_<name>.jsonl` written by prep (IndicVoices valid, Kathbath test known /
  unknown clean / noisy, Svarah, IndicContextEval, MUCS test / blind, Vaani test, Lahaja,
  LibriSpeech, VoxPopuli, and FLEURS dev as `eval_fleurs_dev.jsonl`);
- `<name>_code_mixed` for IndicContextEval sets: the rows whose `code_mixing` flag is set;
- `cs_dev` (synthetic inter-sentential code-switch) and `neg_dev` (non-speech);
- `robust_<condition>`: a fixed subset (`robust_per_lang` utterances per language from the
  Kathbath clean test-unknown sets and Svarah, else the dev set) corrupted at load time with the
  robustness probe's conditions (`indicear.robustness_probe`), seeded per utterance.
"""

from __future__ import annotations

import fnmatch
import json
import time
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from indicear import metrics
from indicear.common import FRAME_RATE, read_jsonl, write_json
from indicear.text import normalize

ROBUST_CONDITIONS = ("noise:10", "noise:5", "babble:10", "reverb:0.6", "reverb:0.9", "reverb:0.6+pointnoise:10")
ROBUST_BASES = ("kathbath_clean_test_unknown_*", "svarah")


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------- decoding


@torch.no_grad()
def decode_items(
    model, cache, items: Sequence[dict], tokenizer, budget: int, device, dtype, keep: Callable | None = None,
    log: Callable[[str], None] = _log, constraints=None,
) -> dict:
    """Greedy decode cached items in frame-budget batches (OOM: split and retry).

    Returns {"ctc": [text], "rnnt": [text] (transducer variants)} and, for models with LID heads,
    "lid": {"utt": [language or None], "spans": [[(language, start_frame, end_frame), ...]]}.
    Spans are snapped to the CTC words when the model has a transcript head (indicear.lid
    .utterance_spans), else they are the frame head's own greedy spans. The LID-only variant
    has no transcript keys.

    `constraints` (a `indicear.decoding.Constrainer`) adds "ctc_constrained": the same CTC
    log-probabilities decoded under the decode-time script rules (no second forward pass)."""
    from indicear.lid import LID_LANGS, utterance_spans
    from indicear.runtime.batching import BucketFrameSampler, OomBackoff

    model.eval()
    lengths = [cache.frames(i["key"]) for i in items]
    sampler = BucketFrameSampler(lengths, budget, num_buckets=10, shuffle=False)
    outputs: dict = {}
    if model.final is not None:
        outputs["ctc"] = [""] * len(items)
        if constraints is not None and getattr(model, "script_ranges", None):
            constraints.tables.check(model.vocab_size, model.script_ranges)
            outputs["ctc_constrained"] = [""] * len(items)
    if model.transducer is not None:
        outputs["rnnt"] = [""] * len(items)
    lid = {"utt": [None] * len(items), "spans": [None] * len(items)} if model.lid is not None else None
    backoff = OomBackoff(log=log)
    for batch in sampler.batches_for_epoch(0):

        def run(chunk, _weight, batch=batch):
            rows = [batch[i] for i in chunk]
            hidden, lens = cache.batch([items[r]["key"] for r in rows], device, dtype)
            with torch.autocast(device_type=torch.device(device).type, dtype=dtype,
                                enabled=torch.device(device).type == "cuda"):
                result = model.decode(hidden, lens)
            constrained = None
            if "ctc_constrained" in outputs:
                heads = result.get("lid") or {}
                constrained, _ = constraints.greedy(result["final_log_probs"].float(), lens, heads.get("utt_logits"),
                                                    heads.get("frame_log_probs"))
            for j, row in enumerate(rows):
                if constrained is not None:
                    outputs["ctc_constrained"][row] = tokenizer.decode(constrained[j][0])
                tokens, frames = result["ctc"][j] if "ctc" in result else ([], [])
                if "ctc" in outputs:
                    outputs["ctc"][row] = tokenizer.decode(tokens)
                if "rnnt" in result:
                    outputs["rnnt"][row] = tokenizer.decode(result["rnnt"][j])
                if lid is not None:
                    out = result["lid"]
                    if "utt_logits" in out:
                        lid["utt"][row] = LID_LANGS[int(out["utt_logits"][j].argmax())]
                    if "frame_log_probs" in out:
                        length = int(lens[j])
                        if tokens:
                            spans = utterance_spans(out["frame_log_probs"][j], length, tokens, frames,
                                                    tokenizer.word_starts(tokens))
                        else:
                            spans = out["spans"][j]
                        lid["spans"][row] = [(lang, int(a), int(b)) for lang, a, b in spans]
                if keep is not None:
                    keep(row, j, result, int(lens[j]))

        backoff.run(list(range(len(batch))), run)
    if lid is not None:
        outputs["lid"] = lid
    return outputs


def load_audios(items: Sequence[dict], pool: ThreadPoolExecutor, transform: Callable | None = None,
                seed: int = 0) -> list[np.ndarray]:
    """16 kHz float32 audio of each item, corrupted by `transform(audio, rng, item)` when given
    (seeded per utterance, so every system sees the same corruption)."""
    from indicear.encoder import load_audio

    def one(item: dict) -> np.ndarray:
        audio = load_audio(item["audio"])
        if transform is not None:
            audio = transform(audio, np.random.default_rng((seed, zlib.crc32(item["key"].encode()))), item)
        return np.asarray(audio, dtype=np.float32)

    return list(pool.map(one, items))


@torch.no_grad()
def lower_items(items: Sequence[dict], staged, featurizer, frame_budget: int, pool: ThreadPoolExecutor,
                transform: Callable | None = None, seed: int = 0, log: Callable[[str], None] = _log):
    """Layer-6 states of `items` in an in-memory RowCache (the tower's dtype and device; padding
    excluded), computed as the live training path computes them."""
    from indicear.cache import group_by_padded_seconds
    from indicear.features import pad_waveforms
    from indicear.runtime.shards import RowCache

    groups = group_by_padded_seconds(items, max(frame_budget / FRAME_RATE, 1.0))
    rows, index, offset = [], {}, 0
    ahead = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-ahead")  # never waits on itself
    pending = ahead.submit(load_audios, groups[0], pool, transform, seed) if groups else None
    for number, group in enumerate(groups):
        audios = pending.result()
        if number + 1 < len(groups):  # load the next group while this one runs
            pending = ahead.submit(load_audios, groups[number + 1], pool, transform, seed)
        waveforms, lengths = pad_waveforms(audios)
        batch = {"waveforms": waveforms, "lengths": lengths}
        featurizer.check_once(batch, log)
        features, mask = featurizer(batch)
        hidden, valid = staged.lower(features, mask)
        counts = valid.sum(-1).tolist()
        for j, item in enumerate(group):
            n = max(1, int(counts[j]))
            rows.append(hidden[j, :n])
            index[item["key"]] = (offset, n)
            offset += n
    ahead.shutdown(wait=True)
    data = torch.cat(rows) if rows else torch.zeros(0, staged.hidden_size)
    return RowCache(data, None, index)


def chunks(items: Sequence[dict], seconds: float) -> Iterable[list[int]]:
    current, total = [], 0.0
    for index, item in enumerate(items):
        current.append(index)
        total += float(item.get("seconds") or 0.0)
        if total >= seconds:
            yield current
            current, total = [], 0.0
    if current:
        yield current


@torch.no_grad()
def decode_models(models: dict, items: Sequence[dict], staged, featurizer, tokenizer, frame_budget: int, device,
                  dtype, workers: int = 8, transform: Callable | None = None, seed: int = 0,
                  chunk_seconds: float = 3600.0, log: Callable[[str], None] = _log, constraints=None) -> dict[str, dict]:
    """Decode `items` with every model in `models` ({name: DesignA}); layers 1-6 run once per
    chunk for all of them. Returns {name: decode_items output over all items}."""
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="eval-audio") as pool:
        for indexes in chunks(items, chunk_seconds):
            chunk = [items[i] for i in indexes]
            cache = lower_items(chunk, staged, featurizer, frame_budget, pool, transform, seed, log)
            for name, model in models.items():
                with model.adapters_attached():
                    decoded = decode_items(model, cache, chunk, tokenizer, 2 * frame_budget, device, dtype, log=log,
                                           constraints=constraints)
                merged = out.setdefault(name, {})
                for key, values in decoded.items():
                    if key == "lid":
                        target = merged.setdefault("lid", {"utt": [], "spans": []})
                        target["utt"] += values["utt"]
                        target["spans"] += values["spans"]
                    else:
                        merged.setdefault(key, []).extend(values)
            del cache
    return out


# ---------------------------------------------------------------- scoring


def lid_metrics(items: Sequence[dict], lid: dict) -> dict:
    """Utterance LID over speech rows; language-span accuracy over code-mixed rows."""
    rows = [i for i, item in enumerate(items) if not item.get("negative") and lid["utt"][i] is not None]
    out = {"utterance": metrics.lid_report([items[i]["lang"] for i in rows], [lid["utt"][i] for i in rows],
                                           [float(items[i].get("seconds") or 0.0) for i in rows])}
    references, hypotheses = [], []
    for i, item in enumerate(items):
        if lid["spans"][i] is None:
            continue
        if "spans" in item:  # synthetic code-switch rows: one span per source utterance
            references.append([{"lang": s["lang"]} for s in item["spans"]])
        else:
            words = normalize(item["text"]).split()
            tags = metrics.row_word_langs(item, words)
            if not tags or len(set(tags)) < 2:
                continue
            references.append([{"lang": lang} for lang, _, _ in metrics._runs(tags)])
        hypotheses.append(lid["spans"][i])
    if references:
        out["spans"] = metrics.lid_span_report(references, hypotheses)
    return out


def score_transcripts(items: Sequence[dict], hyps: Sequence[str], train_vocab: set[str] | None = None) -> dict:
    """Per language (and "all" when there are several): CER/WER, script-normalised WER, script
    error rate, real code-switch spans; synthetic spans for cs rows; false outputs for negatives."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        if item.get("negative"):
            groups["negatives"].append(index)
        elif "spans" in item:
            groups[f"cs:{item['lang']}"].append(index)
        else:
            groups[item["lang"]].append(index)
    speech = [i for i, item in enumerate(items) if not item.get("negative") and "spans" not in item]
    langs = {items[i]["lang"] for i in speech}
    if len(langs) > 1:
        groups["all"] = speech
    out: dict[str, dict] = {}
    for name, indexes in sorted(groups.items()):
        sub_items = [items[i] for i in indexes]
        sub_hyps = [hyps[i] for i in indexes]
        if name == "negatives":
            out[name] = metrics.negatives(sub_hyps)
            continue
        refs = [i["text"] for i in sub_items]
        row = metrics.wer_cer(refs, sub_hyps)
        row["script_error_rate"] = metrics.script_error_rate(refs, sub_hyps)
        if name.startswith("cs:"):
            row.update(metrics.code_switch_spans(sub_items, sub_hyps))
        else:
            row.update(metrics.script_normalised_wer(sub_items, sub_hyps))
            row.update(metrics.code_mixed_spans(sub_items, sub_hyps))
            if train_vocab is not None:
                row.update(metrics.oov_recall(refs, sub_hyps, train_vocab))
        row["hours"] = round(sum(float(i.get("seconds") or 0.0) for i in sub_items) / 3600, 3)
        out[name] = row
    return out


def score(items: Sequence[dict], decoded: dict, train_vocab: set[str] | None = None) -> dict:
    """{"metrics": {decoder: per-group rows}, "lid": ...} from a decode_items-shaped output (any
    key other than "lid" is a decoder's hypotheses: "ctc", "rnnt", or a baseline's own name)."""
    result: dict = {"metrics": {key: score_transcripts(items, hyps, train_vocab)
                                for key, hyps in decoded.items() if key != "lid"}}
    if decoded.get("lid") is not None:
        result["lid"] = lid_metrics(items, decoded["lid"])
    return result


# ---------------------------------------------------------------- eval sets


@dataclass
class EvalSet:
    name: str
    items: list[dict]
    condition: str | None = None  # robustness sets: the corruption applied at load time
    notes: str = ""
    transform: Callable | None = field(default=None, repr=False)


def subset_per_lang(items: Sequence[dict], per_lang: int) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    out = []
    for item in items:
        if counts[item["lang"]] < per_lang:
            out.append(item)
            counts[item["lang"]] += 1
    return out


def eval_sets(manifests: Path, patterns: Sequence[str] = ("*",), max_per_lang: int = 0,
              robust_conditions: Sequence[str] = ROBUST_CONDITIONS, robust_per_lang: int = 100,
              noise_files: Sequence[str] = ()) -> list[EvalSet]:
    """Every eval set available under `manifests`, filtered by glob `patterns` on the set name."""
    from indicear.augment import Corruptor, NoiseBank
    from indicear.robustness_probe import condition_name

    sets: list[EvalSet] = []
    for path in sorted(Path(manifests).glob("eval_*.jsonl")):
        name = path.stem[len("eval_"):]
        rows = read_jsonl(path)
        sets.append(EvalSet(name, rows))
        if any(r.get("code_mixing") is not None for r in rows):
            sets.append(EvalSet(f"{name}_code_mixed", [r for r in rows if r.get("code_mixing")],
                                notes="rows with the source's code_mixing flag"))
    for name in ("cs_dev", "neg_dev"):
        path = Path(manifests) / f"{name}.jsonl"
        if path.exists():
            sets.append(EvalSet(name, read_jsonl(path)))
    if robust_conditions and robust_per_lang:
        bases = [s for s in sets if any(fnmatch.fnmatch(s.name, p) for p in ROBUST_BASES)]
        base = [r for s in bases for r in s.items]
        if not base and (Path(manifests) / "dev.jsonl").exists():
            base = read_jsonl(Path(manifests) / "dev.jsonl")
        base = subset_per_lang(base, robust_per_lang)
        if base:
            bank = NoiseBank(list(noise_files), [r["audio"] for r in base])
            for condition in robust_conditions:
                sets.append(EvalSet(f"robust_{condition_name(condition)}", base, condition,
                                    "Kathbath clean test-unknown + Svarah subset, corrupted at load time",
                                    Corruptor(condition, bank)))
    picked = [s for s in sets if any(fnmatch.fnmatch(s.name, p) for p in patterns)]
    if max_per_lang:
        for s in picked:
            s.items = subset_per_lang(s.items, max_per_lang)
    return [s for s in picked if s.items]


def result_path(results: Path, system: str, set_name: str) -> Path:
    return Path(results) / system / f"{set_name}.json"


def write_result(results: Path, system: str, eval_set: EvalSet, scored: dict, extra: dict | None = None) -> Path:
    path = result_path(results, system, eval_set.name)
    write_json(path, {"kind": "eval", "system": system, "set": eval_set.name, "condition": eval_set.condition,
                      "notes": eval_set.notes, "utterances": len(eval_set.items),
                      "hours": round(sum(float(i.get("seconds") or 0.0) for i in eval_set.items) / 3600, 3),
                      **scored, **(extra or {})})
    return path


def write_rows(path: Path, items: Sequence[dict], hyps: Sequence[str], languages: Sequence[str | None] | None = None) -> Path:
    """One JSON line per row next to a result file (`<set>.rows.jsonl`): the row key, its language and
    the system's transcript (and detected language), so a benchmark can publish per-clip outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for index, (item, hyp) in enumerate(zip(items, hyps, strict=True)):
            row = {"key": item["key"], "lang": item.get("lang"), "hypothesis": hyp}
            if languages is not None:
                row["detected_language"] = languages[index]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------- summary

SUMMARY_COLUMNS = ("cer", "wer", "cer_script_normalised", "wer_script_normalised", "script_error_rate", "cm_span_accuracy", "span_accuracy",
                   "false_output_rate")


def summarize(results: Path) -> dict:
    """results/eval_summary.{json,md}: one row per (system, set, decoder, group)."""
    results = Path(results)
    rows = []
    for path in sorted(results.glob("*/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("kind") != "eval":
            continue
        lid = (data.get("lid") or {}).get("utterance") or {}
        spans = (data.get("lid") or {}).get("spans") or {}
        for decoder, groups in (data.get("metrics") or {"-": {}}).items():
            for group, values in (groups or {"-": {}}).items():
                rows.append({"system": data["system"], "set": data["set"], "decoder": decoder, "group": group,
                             **{k: values.get(k) for k in SUMMARY_COLUMNS},
                             "lid_accuracy": lid.get("accuracy"), "lid_span_accuracy": spans.get("span_accuracy")})
        if not data.get("metrics") and lid:
            rows.append({"system": data["system"], "set": data["set"], "decoder": "lid", "group": "all",
                         **{k: None for k in SUMMARY_COLUMNS}, "lid_accuracy": lid.get("accuracy"),
                         "lid_span_accuracy": spans.get("span_accuracy")})
    header = ["system", "set", "decoder", "group", *SUMMARY_COLUMNS, "lid_accuracy", "lid_span_accuracy"]
    titles = {"cer_script_normalised": "sn-CER", "wer_script_normalised": "sn-WER", "script_error_rate": "script err", "cm_span_accuracy": "cm span acc",
              "span_accuracy": "synth span acc", "false_output_rate": "false output", "lid_accuracy": "LID acc",
              "lid_span_accuracy": "LID span acc"}
    lines = ["| " + " | ".join(titles.get(h, h) for h in header) + " |",
             "| " + " | ".join("---" if i < 4 else "---:" for i in range(len(header))) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("" if row[h] is None else str(row[h]) for h in header) + " |")
    table = "\n".join(lines) + "\n"
    (results / "eval_summary.md").write_text(table, encoding="utf-8")
    write_json(results / "eval_summary.json", {"rows": rows})
    return {"rows": rows, "table": table}
