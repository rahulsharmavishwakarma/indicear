"""Build the Vesper Indic Bench dataset: the clips every system decoded, their references where the
source licence allows publishing them, and every system's per-clip transcript and scores.

    uv run --locked python benchmark/build_bench.py --manifests ROOT/manifests --results ROOT/results --out DIR

The clips are exactly those `indicear.decoding eval --per-lang 300` and `indicear.baselines
--max-per-lang 300` select (the first 300 rows per language of each eval set, `evaluate.eval_sets`).
Writes DIR/data/clips.jsonl, DIR/data/transcripts.jsonl and DIR/summary.json. No audio: clips are
referenced by their source dataset, revision and id, with the audio file's sha1.

Reference text is published only for sources whose licence allows it; MUCS (CC BY-SA 4.0) and Svarah
(gated; CC BY 4.0 only in its card text, no licence tag) keep ids, transcripts and scores without reference text, as in Vesper's
ASR benchmark dataset.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from indicear import evaluate, metrics, sources

SYSTEMS = {  # results directory -> published system name
    "vesper-indicear-auto": "vesper-indicear (automatic language)",
    "vesper-indicear-hinted": "vesper-indicear (language given)",
    "vesper-indicear-unconstrained": "vesper-indicear (automatic language, no decode-time script rules)",
    "indicconformer-ctc": "ai4bharat/indic-conformer-600m-multilingual (CTC, language given)",
    "sravaani": "ARTPARK-IISc/SraVaani-1.0 (TDT, automatic language)",
}
WITHHELD = {"mucs": "CC BY-SA 4.0 (share-alike)", "svarah": "gated; CC BY 4.0 only in the card text, no licence tag"}
SOURCE_OF_PREFIX = {"kathbath_test": "kathbath_test", "indicvoices": "indicvoices", "vaani": "vaani",
                    "indiccontext_eval": "indiccontext_eval", "mucs": "mucs", "svarah": "svarah",
                    "librispeech": "librispeech", "voxpopuli": "voxpopuli", "fleurs": "fleurs"}


def read_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["key"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--per-lang", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vesper-slice", type=Path, help="decoding eval outputs on benchmark/vesper_slice.py's manifests "
                        "(directories auto/ and hinted/): written as data/vesper_asr_benchmark_slice.jsonl")
    args = parser.parse_args()
    sets = evaluate.eval_sets(args.manifests, ["*"], max_per_lang=args.per_lang, robust_conditions=())
    (args.out / "data").mkdir(parents=True, exist_ok=True)
    clips, transcripts = [], []
    summary: dict = defaultdict(dict)
    for eval_set in sets:
        for item in eval_set.items:
            prefix = item["key"].split("/", 1)[0]
            source = sources.SOURCES[SOURCE_OF_PREFIX[prefix]]
            withheld = WITHHELD.get(SOURCE_OF_PREFIX[prefix])
            clips.append({
                "set": eval_set.name, "key": item["key"], "lang": item["lang"], "source": source.name,
                "source_repo": source.repo, "source_revision": source.revision, "source_licence": source.spdx,
                "split": item.get("split"), "seconds": float(item.get("seconds") or 0.0),
                "audio_sha1": item.get("audio_sha1"), "code_mixed": str(item.get("code_mixed")) == "True",
                "reference": None if withheld else item["text"],
                "reference_alternatives": None if withheld else (item.get("refs") or None),
                "reference_withheld": withheld,
            })
        for directory, system in SYSTEMS.items():
            rows = read_rows(args.results / directory / f"{eval_set.name}.rows.jsonl")
            if not rows:
                continue
            for item in eval_set.items:
                row = rows.get(item["key"])
                if row is None:
                    continue  # a language the system does not support
                strict = metrics.wer_cer([item["text"]], [row["hypothesis"]])
                normalised = metrics.script_normalised_wer([item], [row["hypothesis"]])
                transcripts.append({
                    "set": eval_set.name, "key": item["key"], "lang": item["lang"], "system": system,
                    "hypothesis": row["hypothesis"], "detected_language": row.get("detected_language"),
                    "cer": strict["cer"], "wer": strict["wer"],
                    "cer_script_normalised": round(normalised["cer_script_normalised"], 2)
                    if normalised.get("cer_script_normalised") is not None else None,
                })
            result = args.results / directory / f"{eval_set.name}.json"
            if result.exists():
                data = json.loads(result.read_text(encoding="utf-8"))
                per_lang = next(iter(data["metrics"].values())) if "metrics" in data else data
                summary[system][eval_set.name] = {
                    lang: {k: per_lang[lang].get(k) for k in ("cer", "wer", "cer_script_normalised", "wer_script_normalised")}
                    for lang in {i["lang"] for i in eval_set.items} if isinstance(per_lang.get(lang), dict)}
    for name, rows in (("clips", clips), ("transcripts", transcripts)):
        (args.out / "data" / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    if args.vesper_slice:
        slice_rows = []
        for mode, system in (("auto", SYSTEMS["vesper-indicear-auto"]), ("hinted", SYSTEMS["vesper-indicear-hinted"])):
            for path in sorted((args.vesper_slice / mode).glob("*.rows.jsonl")):
                for row in read_rows(path).values():
                    slice_rows.append({"set": path.name.removesuffix(".rows.jsonl"), "sample_id": row["key"],
                                       "lang": row["lang"], "system": system, "hypothesis": row["hypothesis"],
                                       "detected_language": row.get("detected_language")})
        (args.out / "data" / "vesper_asr_benchmark_slice.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in slice_rows), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    text = (args.out / "data" / "clips.jsonl").read_text(encoding="utf-8")
    if "/Users/" in text or "/content/" in text:
        raise SystemExit("a local path leaked into clips.jsonl")
    print(f"{len(sets)} sets, {len(clips)} clips, {len(transcripts)} transcripts, "
          f"{sum(c['reference'] is None for c in clips)} references withheld")


if __name__ == "__main__":
    main()
