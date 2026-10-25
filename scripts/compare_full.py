"""IndicEAR (automatic LID / language hint) vs other systems on the same eval subset, per eval family
and language. Reads `indicear.decoding eval` outputs (one JSON per set, keyed by language) and
`indicear.baselines` outputs (metrics[decoder][language]). Writes a Markdown table.

    python scripts/compare_full.py INDICEAR_AUTO INDICEAR_HINTED OUT.md \
        --baseline IndicConformer=DIR [--baseline SraVaani=DIR ...] [--metric cer_script_normalised]
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
    ("mucs_test", "MUCS"),
]
LANGS = ["hi", "bn", "te", "ml", "mr", "gu", "pa", "or", "ur", "ta", "kn"]


def indicear(path: Path, lang: str, metric: str):
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return (data.get(lang) or {}).get(metric)


def baseline(path: Path, lang: str, metric: str):
    """The system's only decoder (ctc for IndicConformer, tdt for SraVaani)."""
    if not path.exists():
        return None
    metrics = json.loads(path.read_text(encoding="utf-8")).get("metrics") or {}
    decoder = next(iter(metrics.values()), {})
    return decoder.get(lang, {}).get(metric)


def fmt(x) -> str:
    return "–" if x is None else f"{x:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("auto", type=Path)
    parser.add_argument("hinted", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--baseline", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--metric", default="cer_script_normalised")
    args = parser.parse_args()
    baselines = [(name, Path(path)) for name, path in (b.split("=", 1) for b in args.baseline)]
    names = [name for name, _ in baselines]
    lines = [f"Metric: `{args.metric}` (lower is better). Same rows for every system: the first 300 per language of "
             "each eval set. IndicEAR auto = its own utterance LID picks the script; hinted = the row's language is "
             "given. IndicConformer is given the language; SraVaani picks it itself (no Urdu). – = not supported.",
             ""]
    wins = {name: {"auto": 0, "hinted": 0, "rows": 0} for name in names}
    for family, title in FAMILIES:
        rows = []
        for lang in LANGS:
            name = f"{family}_{lang}.json"
            a, h = indicear(args.auto / name, lang, args.metric), indicear(args.hinted / name, lang, args.metric)
            others = [baseline(path / name, lang, args.metric) for _, path in baselines]
            if a is None and all(b is None for b in others):
                continue
            for system, b in zip(names, others):
                if None not in (a, h, b):
                    wins[system]["rows"] += 1
                    wins[system]["auto"] += a < b
                    wins[system]["hinted"] += h < b
            rows.append("| " + " | ".join([lang, fmt(a), fmt(h), *map(fmt, others)]) + " |")
        if rows:
            lines += [f"### {title}", "", "| lang | IndicEAR auto | IndicEAR hinted | " + " | ".join(names) + " |",
                      "|---|" + "---:|" * (2 + len(names)), *rows, ""]
    summary = [f"IndicEAR is lower than {system} in {w['auto']} of {w['rows']} (set, language) rows automatically "
               f"and in {w['hinted']} with the language hint." for system, w in wins.items()]
    lines[1:1] = ["", *summary]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:2 + len(summary)]))


if __name__ == "__main__":
    main()
