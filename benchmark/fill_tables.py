"""Fill the result tables of benchmark/dataset-card.md (the Hugging Face dataset card), and optionally of a report, from the evaluation outputs.

    uv run --locked python benchmark/fill_tables.py --results ROOT/results --vesper results/vesper-benchmark \
        --bench-dir DIR [--card benchmark/dataset-card.md] [--report REPORT.md] --out DIR

Rewrites the blocks between `<!-- BEGIN KEY -->` and `<!-- END KEY -->` in place (HELDOUT_TABLE and
VESPER_SECTION in a --report, RESULTS in the dataset card), so a re-run refills them; writes the dataset
card with its {{SETS}}, {{CLIPS}}, {{HOURS}}, {{TRANSCRIPTS}} and {{SLICE}} counts filled to --out.
"""

import argparse
import json
from pathlib import Path

FAMILIES = [
    ("kathbath_clean_test_unknown", "Kathbath clean, unseen speakers"),
    ("kathbath_clean_test_known", "Kathbath clean, test-known speakers"),
    ("kathbath_noisy_test_unknown", "Kathbath noisy, unseen speakers"),
    ("indicvoices_valid", "IndicVoices valid"),
    ("vaani_test", "Vaani test"),
    ("indiccontext", "IndicContext"),
    ("mucs_test", "MUCS 2021 (hi, bn)"),
]
LANGS = ["hi", "bn", "te", "ml", "mr", "gu", "pa", "or", "ur", "ta", "kn"]
SYSTEMS = [("vesper-indicear-auto", "Vesper-IndicEAR (auto)"), ("vesper-indicear-hinted", "Vesper-IndicEAR (lang. given)"),
           ("indicconformer-ctc", "IndicConformer 600M"), ("sravaani", "SraVaani-1.0")]
METRIC = "cer_script_normalised"


def value(results: Path, system: str, set_name: str, lang: str):
    path = results / system / f"{set_name}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    per_lang = next(iter(data["metrics"].values()), {}) if "metrics" in data else data
    return (per_lang.get(lang) or {}).get(METRIC)


def fmt(x) -> str:
    return "–" if x is None else f"{x:.1f}"


def family_table(results: Path) -> list[str]:
    head = "| Eval family | " + " | ".join(label for _, label in SYSTEMS) + " | languages |"
    lines = [head, "|---|" + "---:|" * (len(SYSTEMS) + 1)]
    for family, title in FAMILIES:
        per = {s: [] for s, _ in SYSTEMS}
        for lang in LANGS:
            vals = {s: value(results, s, f"{family}_{lang}", lang) for s, _ in SYSTEMS}
            if None in vals.values():
                continue
            for s, v in vals.items():
                per[s].append(v)
        if per[SYSTEMS[0][0]]:
            means = [sum(v) / len(v) for v in per.values()]
            best = min(means)
            cells = [f"**{m:.1f}**" if m == best else f"{m:.1f}" for m in means]
            lines.append(f"| {title} | " + " | ".join(cells) + f" | {len(per[SYSTEMS[0][0]])} |")
    return lines


def language_tables(results: Path) -> list[str]:
    out = []
    for family, title in FAMILIES:
        rows = []
        for lang in LANGS:
            vals = [value(results, s, f"{family}_{lang}", lang) for s, _ in SYSTEMS]
            if vals[0] is None:
                continue
            rows.append(f"| {lang} | " + " | ".join(fmt(v) for v in vals) + " |")
        if rows:
            out += [f"<details><summary>{title}: per language</summary>", "",
                    "| lang | " + " | ".join(label for _, label in SYSTEMS) + " |", "|---|" + "---:|" * len(SYSTEMS),
                    *rows, "", "</details>", ""]
    return out


def wins(results: Path) -> tuple[int, int, int]:
    total = auto = hinted = 0
    for family, _ in FAMILIES:
        for lang in LANGS:
            vals = [value(results, s, f"{family}_{lang}", lang) for s, _ in SYSTEMS]
            if None in vals[:3]:
                continue
            baseline = min(v for v in vals[2:] if v is not None)
            total += 1
            auto += vals[0] < baseline
            hinted += vals[1] < baseline
    return total, auto, hinted


def vesper_section(vesper: Path) -> list[str]:
    scores = json.loads((vesper / "scores.json").read_text(encoding="utf-8"))
    titles = {"fleurs_hi": "FLEURS Hindi test", "mucs_hi_en": "MUCS 2021 Hindi-English test",
              "svarah": "Svarah (Indian English)", "fleurs_en": "FLEURS US English test"}
    lines = ["Vesper's public [ASR benchmark dataset](https://huggingface.co/datasets/rahulsharmavishwakarma/vesper-asr-benchmark) "
             "records the transcripts of the speech models Vesper has evaluated on public clips. We decoded the clips it shares with "
             "Vesper-IndicEAR's languages and scored every system with one scorer (the same normalisation, strict script). "
             "Each Vesper model is taken in the configuration that covers the most clips of a set, and every system is "
             "scored on the clips all of them cover, so each table is on identical clips. Vesper's own recorded WER is "
             "shown as a check on the scorer.", "",
             '<p align="center"><img src="figures/fig10_vesper_benchmark.png" width="100%"></p>', "",
             "**Figure 10.** CER on Vesper's benchmark clips. Vesper-IndicEAR in Ember; the other models in Vesper's benchmark in grey.", ""]
    for name in ("fleurs_hi", "mucs_hi_en", "svarah", "fleurs_en"):
        if name not in scores:
            continue
        rows = sorted(scores[name].items(), key=lambda kv: kv[1]["cer"])
        clips = rows[0][1]["clips"]
        lines += [f"**{titles[name]}** ({clips} clips)", "", "| System | Runtime, precision | CER | WER | Vesper-recorded WER |",
                  "|---|---|---:|---:|---:|"]
        for label, s in rows:
            precision = s.get("precision", "")
            if precision.startswith("each model's pinned catalog artifacts"):
                precision = "pinned catalog artifacts"
            config = f"{s['runtime']}, {precision}" if "runtime" in s else "IndicEAR, Apple MPS, fp32"
            lines.append(f"| {label} | {config} | {s['cer']:.1f} | {s['wer']:.1f} | {s.get('vesper_reported_wer_mean', '–')} |")
        lines.append("")
    lines.append("On Hindi, Nemotron 3.5 and Qwen3-ASR are 2–3× more accurate; on English Vesper-IndicEAR is far "
                 "behind every model in the benchmark. On the 16 MUCS Hindi-English code-switched clips it has the lowest CER, "
                 "consistent with Section 7.1, but 16 clips are too few to rank systems.")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--vesper", type=Path, required=True)
    parser.add_argument("--bench-dir", type=Path, required=True, help="the built dataset directory (build_bench.py --out)")
    parser.add_argument("--report", type=Path, help="a Markdown report with HELDOUT_TABLE and VESPER_SECTION markers")
    parser.add_argument("--card", type=Path, default=Path("benchmark/dataset-card.md"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    total, auto, hinted = wins(args.results)
    table = family_table(args.results)
    heldout = ["Mean script-normalised CER over the languages every system supports (Urdu excluded, as SraVaani has no "
               "Urdu); lower is better, best in bold. 300 clips per language per set; per-clip transcripts are in "
               "Vesper Indic Bench.", "", *table, "",
               f"Over {total} (set, language) pairs, Vesper-IndicEAR is lower than the better of the two baselines in "
               f"{auto} automatically and {hinted} with the language given.", "", *language_tables(args.results)]
    import re

    def put(text: str, key: str, body: str) -> str:
        pattern = re.compile(rf"<!-- BEGIN {key} -->.*?<!-- END {key} -->", re.S)
        if not pattern.search(text):
            raise SystemExit(f"marker {key} not found")
        return pattern.sub(lambda _m: f"<!-- BEGIN {key} -->\n{body}\n<!-- END {key} -->", text)

    if args.report:
        report = args.report.read_text(encoding="utf-8")
        report = put(report, "HELDOUT_TABLE", "\n".join(heldout))
        report = put(report, "VESPER_SECTION", "\n".join(vesper_section(args.vesper)))
        args.report.write_text(report, encoding="utf-8")
    clips = [json.loads(line) for line in (args.bench_dir / "data" / "clips.jsonl").read_text(encoding="utf-8").splitlines()]
    transcripts = sum(1 for _ in (args.bench_dir / "data" / "transcripts.jsonl").open(encoding="utf-8"))
    slice_path = args.bench_dir / "data" / "vesper_asr_benchmark_slice.jsonl"
    card = put(args.card.read_text(encoding="utf-8"), "RESULTS", "\n".join(heldout[:len(heldout) - len(language_tables(args.results))]))
    args.card.write_text(card, encoding="utf-8")
    counts = {"{{SETS}}": str(len({c["set"] for c in clips})), "{{CLIPS}}": f"{len(clips):,}",
              "{{HOURS}}": f"{sum(c['seconds'] for c in clips) / 3600:.1f}", "{{TRANSCRIPTS}}": f"{transcripts:,}",
              "{{SLICE}}": f"{sum(1 for _ in slice_path.open()) if slice_path.exists() else 0:,}"}
    for key, val in counts.items():
        card = card.replace(key, val)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "README.md").write_text(card, encoding="utf-8")  # the dataset card, counts filled in
    if "{{" in card:
        raise SystemExit("card: unfilled placeholders remain")
    print("\n".join(table))
    print(f"wins: {auto} auto / {hinted} hinted of {total}")


if __name__ == "__main__":
    main()
