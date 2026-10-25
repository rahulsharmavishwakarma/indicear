"""Score Vesper-IndicEAR and every Vesper speech model on the same clips of Vesper's public ASR benchmark.

    uv run --locked python benchmark/score_vesper.py --slice ROOT --ours ROOT/results/auto --out DIR

Inputs come from benchmark/vesper_slice.py (ROOT/manifests, ROOT/vesper_rows.jsonl) and from
`indicear.decoding eval` on those manifests (per-clip `<set>.rows.jsonl`). One scorer for every system:
`indicear.metrics.wer_cer` after `score_fold` normalisation (lowercase, no punctuation, the same Indic
spelling folds), strict script: an English word written in Devanagari does not match one in Latin.

A Vesper model is scored in the configuration (runtime and precision) that covers the most clips of a
set, using its first transcript per clip; every system is scored on the clips all chosen configurations
cover, so the numbers in one table are on identical clips. Vesper's own recorded per-row scores
are averaged alongside as a cross-check of the scorer. Writes DIR/scores.json and DIR/scores.md.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from indicear import metrics

LABELS = {
    "Qwen/Qwen3-ASR-0.6B": "Qwen3-ASR 0.6B",
    "nvidia/nemotron-3.5-asr-streaming-0.6b": "Nemotron 3.5 ASR 0.6B",
    "CohereLabs/cohere-transcribe-03-2026": "Cohere Transcribe",
    "nvidia/parakeet-tdt-0.6b-v3": "Parakeet TDT 0.6B v3",
    "openai/whisper-large-v3-turbo": "Whisper large-v3-turbo",
    "openai/whisper-small": "Whisper small",
    "moondream/parakeet-redux": "Parakeet Redux",
    "FermionResearch/Phonon-2": "Phonon-2",
    "LiquidAI/LFM2.5-Audio-1.5B": "LFM2.5-Audio 1.5B",
    "ibm-granite/granite-speech-5.0-470m-turboctc": "Granite Speech 5.0 470M",
    "moonshine-ai/moonshine-base": "Moonshine base",
}
SETS = {"fleurs_hi": "FLEURS Hindi test", "fleurs_en": "FLEURS US English test", "svarah": "Svarah (Indian English)",
        "mucs_hi_en": "MUCS 2021 Hindi-English test"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(refs: list[str], hyps: list[str]) -> dict:
    result = metrics.wer_cer(refs, hyps)
    return {"wer": round(result["wer"], 2), "cer": round(result["cer"], 2), "clips": len(refs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--ours", type=Path, nargs="+", required=True, help="decoding eval output directories")
    parser.add_argument("--ours-label", nargs="+", default=["Vesper-IndicEAR"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.slice / "vesper_rows.jsonl")
    out, table = {}, []
    for name, title in SETS.items():
        manifest = args.slice / "manifests" / f"eval_vesper_{name}.jsonl"
        if not manifest.exists():
            continue
        clips = {m["key"]: m["text"] for m in read_jsonl(manifest)}
        keys = sorted(clips)
        refs = [clips[k] for k in keys]
        scores = {}
        configs = defaultdict(dict)
        for r in rows:
            if r["set"] != name or r.get("error"):
                continue
            sid = r["sample_id"].removesuffix("--clean")
            config = (r["asr_model"], r["asr_runtime"], r["precision"])
            configs[config].setdefault(sid, r)
        best = {}  # per model, the configuration covering the most clips
        for (model, runtime, precision), by_clip in configs.items():
            covered = {k for k in keys if k in by_clip}
            if model not in best or len(covered) > len(best[model][3]):
                best[model] = (runtime, precision, by_clip, covered)
        # every system is scored on the same clips: those every chosen configuration covers. A configuration
        # covering under 60% of the best one's clips is left out rather than shrinking the common set.
        top = max((len(c) for *_, c in best.values()), default=0)
        best = {m: v for m, v in best.items() if len(v[3]) >= 0.6 * top}
        common = sorted(set(keys).intersection(*(v[3] for v in best.values()))) if best else keys
        refs = [clips[k] for k in common]
        for directory, label in zip(args.ours, args.ours_label, strict=True):
            path = directory / f"vesper_{name}.rows.jsonl"
            if path.exists():
                hyp = {r["key"]: r["hypothesis"] for r in read_jsonl(path)}
                scores[label] = score(refs, [hyp[k] for k in common])
        for model, (runtime, precision, by_clip, _) in best.items():
            hyps = [by_clip[k]["hypothesis_raw"] or "" for k in common]
            reported = [((by_clip[k].get("raw_score") or {}).get("wordErrorRate") or {}).get("rate") for k in common]
            reported = [v for v in reported if v is not None]
            entry = score(refs, hyps) | {"runtime": runtime, "precision": precision}
            if reported:
                entry["vesper_reported_wer_mean"] = round(100 * sum(reported) / len(reported), 2)
            scores[LABELS.get(model, model)] = entry
        keys = common
        out[name] = scores
        table += [f"### {title} ({len(keys)} clips)", "", "| System | CER | WER | Vesper-reported WER (row mean) |",
                  "|---|---:|---:|---:|"]
        for label, s in sorted(scores.items(), key=lambda kv: kv[1]["cer"]):
            table.append(f"| {label} | {s['cer']:.1f} | {s['wer']:.1f} | {s.get('vesper_reported_wer_mean', '–')} |")
        table.append("")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scores.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / "scores.md").write_text("\n".join(table), encoding="utf-8")
    print("\n".join(table))


if __name__ == "__main__":
    main()
