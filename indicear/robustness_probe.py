"""Robustness probe: CER per condition and language, before any robustness training.

Run it with the v1 linear-CTC heads (frozen encoder, no adapters) to measure the frozen
encoder itself; any later variant can be probed the same way. Each condition corrupts FLEURS dev
audio inside the feature-extraction workers, then runs layers 1-12 and the heads.

    python -m indicear.robustness_probe --root R --variant v1 [--conditions clean,noise:10,...]
    python -m indicear.robustness_probe --root R --variant v1 --summarize

One JSON per condition under results/robustness/<variant>/; existing ones are skipped, so a
rerun continues where it stopped. Several processes can split the condition list on one GPU
(see scripts/robustness.sh). A condition "collapses" when its CER exceeds twice the clean CER
or the clean CER + 20 points.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from indicear import metrics
from indicear.ablate import Paths, build_model, log, noise_files, subset_per_lang
from indicear.augment import CODECS, Corruptor, NoiseBank, available_codecs
from indicear.common import GpuSampler, read_jsonl, runtime_banner, write_json

DEFAULT_CONDITIONS = (
    "clean", "noise:20", "noise:10", "noise:5", "babble:20", "babble:10", "babble:5",
    "reverb:0.3", "reverb:0.6", "reverb:0.9", "reverb:0.6+pointnoise:10",
    "tel8k", "amr", "gsm", "opus", "speed:0.9", "speed:1.1",
)


def condition_name(condition: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]", "_", condition)


@torch.no_grad()
def probe_condition(
    model, staged, items, tokenizer, condition, bank, workers, batch_seconds, featurizer=None
) -> list[str]:
    """`featurizer` (an `encoder.Featurizer`, default: the default backend on the tower's
    device) is shared across conditions, so its GPU-vs-CPU feature check runs once."""
    from indicear.cache import group_by_padded_seconds
    from indicear.encoder import (
        FeatureBatches,
        Featurizer,
        default_feature_backend,
        feature_loader,
    )
    from indicear.runtime.batching import OomBackoff

    order = {item["key"]: index for index, item in enumerate(items)}
    groups = group_by_padded_seconds(items, batch_seconds)
    device = next(staged.tower.parameters()).device
    dtype = next(staged.tower.parameters()).dtype
    featurizer = featurizer or Featurizer(default_feature_backend(), device, dtype)
    dataset = FeatureBatches(groups, transform=Corruptor(condition, bank), seed=17, backend=featurizer.backend)
    hyps = [""] * len(items)
    backoff = OomBackoff(log=log)
    model.eval()
    for index, batch in feature_loader(dataset, workers):
        featurizer.check_once(batch, log)
        inputs, mask = featurizer(batch)

        def run(rows, _weight, inputs=inputs, mask=mask, index=index):
            sub_mask = mask[rows]
            frames = int(sub_mask.sum(-1).max())
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden, valid = staged.lower(inputs[rows, :frames], sub_mask[:, :frames])
                result = model.decode(hidden, valid.sum(-1))
            for j, row in enumerate(rows):
                hyps[order[groups[index][row]["key"]]] = tokenizer.decode(result["ctc"][j][0])

        backoff.run(list(range(inputs.shape[0])), run)
    return hyps


def summarize(out_dir: Path) -> dict:
    table: dict[str, dict[str, float | None]] = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        result = json.loads(path.read_text())
        if result.get("skipped"):
            table[result["condition"]] = {"skipped": result["skipped"]}
            continue
        table[result["condition"]] = {lang: row["cer"] for lang, row in result["per_lang"].items()}
    clean = table.get("clean", {})
    collapses = []
    for condition, row in table.items():
        for lang, cer in row.items():
            base = clean.get(lang)
            if isinstance(cer, (int, float)) and isinstance(base, (int, float)) and (cer > 2 * base or cer > base + 20):
                collapses.append({"condition": condition, "lang": lang, "cer": cer, "clean_cer": base})
    summary = {"cer": table, "collapses": collapses}
    write_json(out_dir / "summary.json", summary)
    langs = sorted({lang for row in table.values() for lang in row if lang != "skipped"})
    lines = ["| condition | " + " | ".join(langs) + " |", "| --- |" + " ---: |" * len(langs)]
    for condition, row in table.items():
        lines.append(f"| {condition} | " + " | ".join(str(row.get(lang, row.get("skipped", ""))) for lang in langs) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"collapses: {len(collapses)}")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="/content/indicear")
    parser.add_argument("--variant", default="v1")
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--max-per-lang", type=int, default=200)
    parser.add_argument("--noise-dir", default=None, help="MUSAN-style noise; synthetic colored noise otherwise")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--batch-seconds", type=float, default=600.0)
    parser.add_argument("--features", choices=("cpu", "gpu"), default=None,
                        help="where the log-mel is computed (default: gpu when CUDA is available)")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    paths = Paths(args.root)
    out_dir = paths.results / "robustness" / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(out_dir)
        return
    conditions = [c for c in args.conditions.split(",") if c]
    todo = [c for c in conditions if not (out_dir / f"{condition_name(c)}.json").exists()]
    if not todo:
        log("all conditions done")
        return
    from indicear.encoder import (
        Featurizer,
        StagedTower,
        default_feature_backend,
        load_tower,
        runtime_device,
    )
    from indicear.tokenizer import AggregateTokenizer

    log(runtime_banner())
    tokenizer = AggregateTokenizer.load(paths.tok)
    device, dtype = runtime_device()
    staged = StagedTower(load_tower(device, dtype))
    model = build_model(args.variant, staged, tokenizer).to(device)
    model.load(paths.runs / args.variant)
    parameter = next(staged.tower.parameters())
    featurizer = Featurizer(args.features or default_feature_backend(), parameter.device, parameter.dtype)
    log(f"features on {featurizer.backend}")
    items = subset_per_lang(read_jsonl(paths.manifest("dev")), args.max_per_lang)
    bank = NoiseBank(noise_files(args.noise_dir), [i["audio"] for i in read_jsonl(paths.manifest("dev"))])
    codecs = available_codecs()
    with GpuSampler(out_dir / f"gpu-{os.getpid()}.csv") as gpu:
        for condition in todo:
            path = out_dir / f"{condition_name(condition)}.json"
            missing = [s.split(":")[0] for s in condition.split("+") if s.split(":")[0] in CODECS and s.split(":")[0] not in codecs]
            if missing:
                write_json(path, {"condition": condition, "skipped": f"ffmpeg encoder missing for {missing}"})
                log(f"{condition}: skipped (no ffmpeg encoder for {missing})")
                continue
            started = time.time()
            hyps = probe_condition(model, staged, items, tokenizer, condition, bank, args.workers,
                                   args.batch_seconds, featurizer)
            by_lang = defaultdict(list)
            for item, hyp in zip(items, hyps, strict=True):
                by_lang[item["lang"]].append((item["text"], hyp))
            per_lang = {lang: metrics.wer_cer([r for r, _ in v], [h for _, h in v]) for lang, v in sorted(by_lang.items())}
            write_json(path, {
                "condition": condition, "variant": args.variant, "per_lang": per_lang,
                "noise": "musan" if bank.noise_files else "synthetic colored noise",
                "utterances": len(items), "seconds": round(time.time() - started),
            })
            log(f"{condition}: " + " ".join(f"{lang} {row['cer']}" for lang, row in per_lang.items()))
        log(gpu.summary())


if __name__ == "__main__":
    sys.exit(main())
