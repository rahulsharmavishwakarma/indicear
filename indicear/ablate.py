"""Design-A ablation: FLEURS on the layer-6 cache, or a mixture streamed live (FLEURS test is never read).

    python -m indicear.ablate prep   --root R                      # manifests, tokenizer, CS dev, negatives, label rates
    python -m indicear.ablate cache  --root R [--aug-views 2]      # layers 1-6 once (resumable)
    python -m indicear.ablate train  --root R --variants v1,v2,v3,v4 [--drive-copy DRIVE]
    python -m indicear.ablate report --root R

Streamed mixture (the pilot):

    python -m indicear.ablate --root R prep --mixture pilot --stream
    python -m indicear.ablate --root R train --lower live --variants v2,v3,v4,v4+lid --train-hours 6500
    python -m indicear.ablate --root R eval --variants v2,v3,v4,v4+lid     # results/<v>/<eval set>.json
    python -m indicear.ablate --root R report                              # results/summary.md, eval_summary.md

Live training (`--lower live`): the StreamLoader of the prepared mixture feeds the trainer, the
log-mel and layers 1-6 run every step, and rows are augmented on the fly (`--clean-prob`,
`--augment`; indicear.augment.AugmentPolicy). By default every variant trains in ONE process from
ONE stream (`trainer.LiveGroup`: audio decoded and layers 1-6 run once per batch for all of
them); `--no-share-stream` trains them one after another, each with its own stream. The run
length is `--train-hours` of audio (default: --epochs x the mixture's planned hours), converted
to optimizer steps once the frame budget is fixed (`trainer.live_steps`, an estimate logged at
start; `--max-steps` fixes it instead). Variants: v1-v6, "<v>+lid" (LID heads on top) and "lid"
(LID heads alone on the frozen encoder).

Devices (`--device` or $INDICEAR_DEVICE; default auto: CUDA, then Apple MPS, then the CPU): on MPS
training runs in float32 (`--precision fp16`: float16 autocast with a GradScaler), the log-mel
and the main-process augmentation run on the CPU, and a frame budget too large for unified
memory runs as `--splits` micro-batches whose losses are normalized by the whole batch, so the
optimization is the one-piece one. Device, precision and splits never enter the config hash: a
CUDA run resumes on MPS (scripts/resume_local.sh) and back.

Every stage is safe to rerun: finished outputs are skipped, a variant with results/<v>.json is
skipped, and a variant with checkpoints resumes. Run several cached `train` processes on one GPU
with --vram-share 1/N --cache-placement mmap.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

# Before torch is imported: on Apple MPS, ops without an MPS kernel run on the CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from indicear import metrics, phonetic
from indicear.common import (
    CODE_SWITCH_PAIRS,
    EG2,
    FLEURS,
    FRAME_RATE,
    LANGS,
    read_jsonl,
    runtime_banner,
    write_json,
    write_jsonl,
)
from indicear.encoder import DEVICE_CHOICES
from indicear.evaluate import decode_items
from indicear.runtime.gpu import empty_device_cache

ROBUST_DEFAULT = ("noise:10", "noise:5", "babble:10", "reverb:0.6", "reverb:0.9", "reverb:0.6+pointnoise:10")

# ---------------------------------------------------------------- paths


class Paths:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifests = self.root / "manifests"
        self.tok = self.root / "tok"
        self.cache = self.root / "cache" / "layer6"
        self.runs = self.root / "runs"
        self.results = self.root / "results"

    def manifest(self, name: str) -> Path:
        return self.manifests / f"{name}.jsonl"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------- prep


def label_rates(items: Sequence[dict], tokenizer) -> dict:
    """Output labels per second vs the 25 fps frame rate, per language."""
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_lang[item["lang"]].append(item)
    out = {}
    for lang, rows in sorted(by_lang.items()):
        tokens = [len(tokenizer.encode(r["text"])) / r["seconds"] for r in rows]
        units = [len(phonetic.text_units(r["text"])) / r["seconds"] for r in rows]
        frames = [r["seconds"] * FRAME_RATE for r in rows]
        unit_infeasible = sum(len(phonetic.text_units(r["text"])) * 1.0 > f for r, f in zip(rows, frames, strict=True))

        def summary(values):
            values = sorted(values)
            return {"mean": round(statistics.fmean(values), 2), "p95": round(values[int(0.95 * (len(values) - 1))], 2),
                    "max": round(values[-1], 2)}

        out[lang] = {"tokens_per_s": summary(tokens), "units_per_s": summary(units),
                     "unit_frames_infeasible": unit_infeasible, "utterances": len(rows),
                     "frame_rate": FRAME_RATE}
    return out


def check_mixture(paths: Paths, mixture: str, source: str | None = None) -> None:
    """One mixture per root: refuse to mix FLEURS and multi-source manifests in one directory.
    The marker also records the mixture file (path and sha256) for `train --lower live`."""
    marker = paths.manifests / "mixture.json"
    if marker.exists():
        found = json.loads(marker.read_text(encoding="utf-8"))["mixture"]
    elif paths.manifest("dev").exists():
        found = "fleurs"  # roots prepared before --mixture existed
    else:
        record = {"mixture": mixture}
        if source is not None:
            path = mixture_path(source)
            record.update(file=str(path), sha256=file_sha256(path))
        write_json(marker, record)
        return
    if found != mixture:
        raise SystemExit(f"{paths.root} was prepared with --mixture {found}; use another --root for {mixture}")


def mixture_path(name: str | Path) -> Path:
    """The JSON file behind a mixture name (indicear/mixtures/<name>.json) or path."""
    from indicear.mixture import MIXTURES

    path = Path(name)
    return (MIXTURES / f"{name}.json").resolve() if not path.suffix else path.resolve()


def file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prep_fleurs(args, paths: Paths) -> tuple[list[str], dict]:
    from indicear.data import fleurs_items

    langs = args.langs.split(",")
    if not paths.manifest("dev").exists():
        train, dev = [], []
        for lang in langs:
            lang_train = fleurs_items(paths.root, lang, "train")
            if args.max_train_hours:
                kept, seconds = [], 0.0
                for item in lang_train:
                    if seconds >= args.max_train_hours * 3600:
                        break
                    kept.append(item)
                    seconds += item["seconds"]
                lang_train = kept
            train += lang_train
            dev += fleurs_items(paths.root, lang, "dev")
            log(f"{lang}: train {len(lang_train)} ({sum(i['seconds'] for i in lang_train) / 3600:.1f} h)")
        write_jsonl(paths.manifest("train"), train)
        write_jsonl(paths.manifest("dev"), dev)
    return langs, {"fleurs": f"{FLEURS[0]}@{FLEURS[1]}"}


def prep_mixture(args, paths: Paths) -> tuple[list[str], dict]:
    """Multi-source mixture (indicear.mixture): eval sets, speaker-disjoint train/dev, licences."""
    from indicear.mixture import build, load_mixture
    from indicear.sources import write_license_manifest

    mixture = load_mixture(args.mixture)
    allow = {s for s in args.allow_review.split(",") if s}
    if not paths.manifest("dev").exists():
        built = build(paths.root, mixture, allow, workers=args.download_workers, decode_workers=args.decode_workers,
                      disk_budget_gb=args.disk_budget_gb, keep_downloads=args.keep_downloads, log=log)
        write_jsonl(paths.manifest("train"), built["train"])
        for name, rows in sorted(built["eval"].items()):
            write_jsonl(paths.manifest(f"eval_{name}"), rows)
        write_json(paths.manifests / "data_report.json", built["report"])
        write_license_manifest(paths.manifests / "sources.json", [x["source"] for x in built["report"]["licenses"]])
        write_jsonl(paths.manifest("dev"), built["dev"])  # written last: the stage's completion marker
    report = json.loads((paths.manifests / "data_report.json").read_text(encoding="utf-8"))
    langs = sorted({r["lang"] for r in read_jsonl(paths.manifest("train"))})
    return langs, {"mixture": mixture["name"], "data": report}


def prep_stream(args, paths: Paths) -> tuple[list[str], dict]:
    """Streaming mode: download only the fixed sets (eval, dev, tar-only sources), plan the
    streams, and collect the training transcripts from the text columns alone (no audio)."""
    from indicear.mixture import load_mixture
    from indicear.sources import write_license_manifest
    from indicear.stream import build_planner, prepare_fixed, text_pass

    mixture = load_mixture(args.mixture)
    allow = {s for s in args.allow_review.split(",") if s}
    fixed = prepare_fixed(paths.root, mixture, allow, workers=args.download_workers,
                          decode_workers=args.decode_workers or None, disk_budget_gb=args.disk_budget_gb, log=log)
    texts_path = paths.manifest("train_texts")
    plan_path = paths.manifests / "stream_plan.json"
    if not texts_path.exists():
        setup = build_planner(paths.root, mixture, frame_budget=1, allow_review=allow, log=log)
        try:
            texts = text_pass(setup, paths.root, log=log)
            write_json(plan_path, {"mixture": mixture["name"], "allow_review": sorted(allow), "shares": setup.shares,
                                   "probabilities": setup.planner.probabilities(),
                                   "pools": {name: [s["path"] for s in pool] for name, pool in setup.pools.items()}})
        finally:
            setup.cache.close()
        write_jsonl(texts_path, texts)
    fleurs_langs = fleurs_dev_langs(args.fleurs_dev_langs, mixture)
    if fleurs_langs and not paths.manifest("eval_fleurs_dev").exists():
        from indicear.data import fleurs_items

        rows = []
        for lang in fleurs_langs:  # FLEURS dev for continuity with the FLEURS ablation (test is never read)
            rows += [{**r, "eval_set": "fleurs_dev"} for r in fleurs_items(paths.root, lang, "dev")]
        write_jsonl(paths.manifest("eval_fleurs_dev"), rows)
    sources = sorted({s["source"] for s in fixed["streams"]} | {
        r["source"] for name in ("dev", "local_train") for r in read_jsonl(paths.manifest(name))})
    write_license_manifest(paths.manifests / "sources.json", sources)
    langs = sorted({r["lang"] for r in read_jsonl(texts_path)})
    return langs, {"mixture": mixture["name"], "stream": True, "fixed": fixed,
                   "plan": json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None}


def fleurs_dev_langs(option: str, mixture: dict) -> list[str]:
    """--fleurs-dev-langs: "mixture" (the mixture's training languages), "none", or a list."""
    if option == "none":
        return []
    if option == "mixture":
        return sorted({entry["lang"] for entry in mixture.get("train", []) if entry["lang"] in LANGS})
    return [lang for lang in option.split(",") if lang]


def validate_texts(paths: Paths, tokenizer, names: Sequence[str]) -> dict:
    """Every transcript must encode through the tokenizer and the phonetic units; drop and count failures."""
    report = {}
    for name in names:
        path = paths.manifest(name)
        if not path.exists():
            continue
        rows, kept, counts = read_jsonl(path), [], defaultdict(int)
        for row in rows:
            try:
                ids = tokenizer.encode(row["text"])
                units = phonetic.encode(row["text"])
            except Exception:  # noqa: BLE001 - counted and dropped, never fatal
                counts["encode_error"] += 1
                continue
            if not ids or not units:
                counts["empty_encoding"] += 1
                continue
            if any(tokenizer.is_unknown(i) for i in ids):
                counts["unk_pieces"] += 1  # kept: an unseen character, spelled as <unk>
            kept.append(row)
        if len(kept) != len(rows):
            write_jsonl(path, kept)
        report[name] = {"rows": len(rows), **counts}
    return report


def cmd_prep(args) -> None:
    from indicear.augment import NoiseBank, make_negatives
    from indicear.data import synth_code_switch
    from indicear.tokenizer import AggregateTokenizer

    paths = Paths(args.root)
    if args.mixture == "fleurs":
        check_mixture(paths, "fleurs")
    else:
        from indicear.mixture import load_mixture

        check_mixture(paths, load_mixture(args.mixture)["name"], args.mixture)
    if args.mixture == "fleurs":
        langs, extra = prep_fleurs(args, paths)
    elif args.stream:
        langs, extra = prep_stream(args, paths)
    else:
        langs, extra = prep_mixture(args, paths)
    train_name = "train_texts" if args.stream else "train"
    train, dev = read_jsonl(paths.manifest(train_name)), read_jsonl(paths.manifest("dev"))
    if not (paths.tok / "aggregate.json").exists():
        if args.tok_from:  # warm start: keep the source run's models for these scripts, train the rest
            keep = [s for s in args.tok_keep.split(",") if s]
            AggregateTokenizer.extend(args.tok_from, keep, [i["text"] for i in train], paths.tok)
            log(f"tokenizer: kept {', '.join(keep)} from {args.tok_from}; trained the other scripts on {train_name}")
        else:
            AggregateTokenizer.train([i["text"] for i in train], paths.tok)
    tokenizer = AggregateTokenizer.load(paths.tok)
    if args.mixture != "fleurs":
        names = [train_name, "dev", *sorted(p.stem for p in paths.manifests.glob("eval_*.jsonl"))]
        extra["text_check"] = validate_texts(paths, tokenizer, names)
        train, dev = read_jsonl(paths.manifest(train_name)), read_jsonl(paths.manifest("dev"))
    if not paths.manifest("cs_dev").exists():
        by_lang = defaultdict(list)
        for item in dev:
            by_lang[item["lang"]].append(item)
        pairs = [p for p in CODE_SWITCH_PAIRS if p[0] in langs and p[1] in langs]
        cs = synth_code_switch(by_lang, paths.root / "audio" / "cs_dev", pairs, args.cs_per_pair)
        write_jsonl(paths.manifest("cs_dev"), cs)
    bank = NoiseBank(noise_files(args.noise_dir))
    if not paths.manifest("neg_train").exists():
        write_jsonl(paths.manifest("neg_train"), make_negatives(paths.root / "audio" / "neg_train", args.negatives, bank, seed=1))
    if not paths.manifest("neg_dev").exists():
        negs = make_negatives(paths.root / "audio" / "neg_dev", args.negatives_dev, bank, seed=2)
        write_jsonl(paths.manifest("neg_dev"), [{**n, "key": n["key"].replace("negative/", "negative-dev/")} for n in negs])
    timed = [i for i in train if i.get("seconds")]
    rates = label_rates(timed, tokenizer)
    tok_meta = json.loads((paths.tok / "aggregate.json").read_text(encoding="utf-8"))
    write_json(paths.root / "prep_report.json", {
        **extra, "langs": langs, "tokenizer": {**tokenizer.describe(),
                                               **({"extended_from": tok_meta["extended_from"]}
                                                  if "extended_from" in tok_meta else {})},
        "label_rates": rates, "train_hours": round(sum(i["seconds"] for i in timed) / 3600, 2),
        "dev_hours": round(sum(i["seconds"] for i in dev) / 3600, 2),
    })
    if args.mixture != "fleurs" and not args.stream:
        print_data_report(extra["data"])
    log(f"{'lang':5} {'tok/s p95':>9} {'tok/s max':>9} {'unit/s p95':>10} {'unit/s max':>10} (frames: {FRAME_RATE}/s)")
    for lang, row in rates.items():
        log(f"{lang:5} {row['tokens_per_s']['p95']:9.1f} {row['tokens_per_s']['max']:9.1f} "
            f"{row['units_per_s']['p95']:10.1f} {row['units_per_s']['max']:10.1f}")
        if row["units_per_s"]["p95"] > 0.6 * FRAME_RATE:
            log(f"WARNING {lang}: phonetic units at p95 exceed 60% of the frame rate")


def print_data_report(report: dict) -> None:
    log(f"mixture {report['mixture']}: train {report['train']['total']} h, dev {report['dev']['total']} h, "
        f"code-switched {report['code_switched_hours']} h (target {report.get('code_switched_target_hours')})")
    for lang, hours in report["train"]["by_lang"].items():
        log(f"  {lang}: {hours} h")
    for key, hours in report["train"]["by_source"].items():
        log(f"  {key}: {hours} h")
    for name, table in report["eval"].items():
        log(f"  eval {name}: {table['total']} h")
    for source, drops in report["drops"].items():
        log(f"  drops {source}: {drops}")
    for message in report["skipped"] + report["unavailable"]:
        log(f"  NOT LOADED {message}")
    for label, hours in report["shortfall_hours"].items():
        log(f"  SHORT {label}: {hours} h less than requested")
    log("  licences: " + "; ".join(f"{x['source']} {x['spdx']} ({x['status']})" for x in report["licenses"]))


def noise_files(directory: str | None) -> list[str]:
    if not directory:
        return []
    return sorted(str(p) for p in Path(directory).rglob("*") if p.suffix.lower() in (".wav", ".flac"))


# ---------------------------------------------------------------- cache


def cache_items(paths: Paths, views: int) -> list[dict]:
    from indicear.data import with_views

    train = read_jsonl(paths.manifest("train"))
    others = [read_jsonl(paths.manifest(n)) for n in ("dev", "cs_dev", "neg_train", "neg_dev")]
    items = with_views(train, views) + [{**i, "view": 0} for rows in others for i in rows]
    return sorted(items, key=lambda i: i["key"])


def cmd_cache(args) -> None:
    from indicear.augment import (
        AugmentationRecipe,
        CodecAugment,
        NoiseBank,
        available_codecs,
    )
    from indicear.cache import build_cache
    from indicear.encoder import (
        StagedTower,
        default_feature_backend,
        load_tower,
        runtime_device,
    )

    paths = Paths(args.root)
    views = 1 + args.aug_views
    items = cache_items(paths, views)
    transform = gpu_augment = None
    device, dtype = runtime_device()
    features = args.features or default_feature_backend()
    augment_device = args.augment_device or ("gpu" if features == "gpu" else "cpu")
    if args.aug_views and augment_device == "gpu":
        from indicear.gpu_augment import GpuAugment, parse_gpu_policy

        if features != "gpu":
            raise SystemExit("--augment-device gpu needs --features gpu")
        transform = CodecAugment(args.codec_prob, only_augmented=True)  # workers: decode + codecs only
        gpu_augment = GpuAugment(device, parse_gpu_policy(args.augment), noise_files=noise_files(args.noise_dir))
        log(f"augmentation on the GPU: {args.aug_views} extra views per train utterance, weights "
            f"{gpu_augment.weights}; codecs {list(transform.codecs)} on {args.codec_prob:.0%} of views in the workers")
    elif args.aug_views:
        speech = [i["audio"] for i in read_jsonl(paths.manifest("train"))]
        transform = AugmentationRecipe(NoiseBank(noise_files(args.noise_dir), speech))
        log(f"augmentation: {args.aug_views} extra views per train utterance; ffmpeg codecs {sorted(available_codecs())}")
    tower = load_tower(device, dtype)
    staged = StagedTower(tower)
    log(f"{runtime_banner()}; {len(items)} cache items, {sum(i['seconds'] for i in items) / 3600:.1f} h")
    build_cache(items, paths.cache, staged, storage=args.storage, batch_seconds=args.batch_seconds,
                workers=args.workers, transform=transform, features=features, gpu_augment=gpu_augment, log=log)


# ---------------------------------------------------------------- model, examples, decoding


def build_model(variant: str, staged, tokenizer, overrides: dict | None = None, lid=None):
    """A DesignA for "v1".."v6", "<v>+lid" or "lid" (indicear.model.resolve_variant); `lid`
    overrides the default LidConfig of a LID variant."""
    from dataclasses import replace

    from indicear.model import DesignA, resolve_variant

    base, lid_config = resolve_variant(variant, lid)
    config = replace(base, **(overrides or {}))
    ranges = [tokenizer.script_range(s) for s in tokenizer.order]
    return DesignA(staged, config, tokenizer.vocab_size, ranges, len(phonetic.UNITS), lid=lid_config)


def lid_config_for(variant: str, hours_by_lang: dict[str, float] | None, args=None):
    """The LidConfig of a LID variant, class-balanced by the training hours per language
    (lid.balance_weights); None for variants without LID heads. `args.lid_utt_weight` /
    `args.lid_frame_weight`, when set, replace the loss weights (default: LidConfig's)."""
    from dataclasses import replace

    from indicear.lid import LidConfig, balance_weights
    from indicear.model import resolve_variant

    if resolve_variant(variant)[1] is None:
        return None
    weights = balance_weights(hours_by_lang) if hours_by_lang else None
    config = resolve_variant(variant, LidConfig(class_weights=weights))[1]
    overrides = {name: value for name, value in (("utterance_weight", getattr(args, "lid_utt_weight", None)),
                                                 ("frame_weight", getattr(args, "lid_frame_weight", None)))
                 if value is not None}
    return replace(config, **overrides) if overrides else config


def variant_overrides(args) -> dict:
    """VariantConfig fields set on the command line (`--inter-weight`); absent ones keep the
    variant's defaults, and so the config hashes of earlier runs."""
    overrides = {"checkpoint": args.checkpoint_layers}
    if getattr(args, "inter_weight", None) is not None:
        overrides["inter_weight"] = args.inter_weight
    return overrides


def make_examples(items: Sequence[dict], cache, tokenizer, views: int) -> tuple[list, dict]:
    from indicear.trainer import Example

    examples, dropped = [], defaultdict(int)
    for item in items:
        if item["key"] not in cache.index:
            dropped["not_cached"] += 1
            continue
        tokens = tokenizer.encode(item["text"])
        frames = cache.frames(item["key"])
        repeats = sum(a == b for a, b in itertools.pairwise(tokens))
        if len(tokens) + repeats > frames:
            dropped[f"ctc_infeasible:{item['lang']}"] += 1
            continue
        keys = [item["key"]] + [f"{item['key']}#v{v}" for v in range(1, views) if f"{item['key']}#v{v}" in cache.index]
        word_langs = tuple(item["word_langs"]) if item.get("word_langs") else None
        examples.append(Example(tuple(keys), torch.tensor(tokens, dtype=torch.long),
                                torch.tensor(phonetic.encode(item["text"]), dtype=torch.long), frames, item["lang"],
                                float(item.get("sample_weight", 1.0)), word_langs))
    return examples, dict(dropped)


def quick_eval(model, cache, items, tokenizer, budget, device, dtype, log: Callable[[str], None] | None = None) -> float:
    """Macro CER over languages on a dev subset (lower is better): the best-checkpoint metric.
    The LID-only variant uses 100 - utterance LID accuracy instead. `log` gets each language's CER."""
    decoded = decode_items(model, cache, items, tokenizer, budget, device, dtype)
    if "ctc" not in decoded:
        report = metrics.lid_report([i["lang"] for i in items], decoded["lid"]["utt"])
        return 100.0 - (report["accuracy"] or 0.0)
    hyps = decoded["ctc"]
    by_lang = defaultdict(list)
    for item, hyp in zip(items, hyps, strict=True):
        by_lang[item["lang"]].append((item["text"], hyp))
    cers = {lang: metrics.wer_cer([r for r, _ in v], [h for _, h in v])["cer"] for lang, v in sorted(by_lang.items())}
    if log is not None:
        log("dev CER by language: " + ", ".join(f"{lang} {c:.1f}" if c is not None else f"{lang} -"
                                                for lang, c in cers.items()))
    return float(statistics.fmean(c for c in cers.values() if c is not None))


def subset_per_lang(items: Sequence[dict], per_lang: int) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    out = []
    for item in items:
        if counts[item["lang"]] < per_lang:
            out.append(item)
            counts[item["lang"]] += 1
    return out


# ---------------------------------------------------------------- train


def data_notes(paths: Paths) -> str:
    marker = paths.manifests / "mixture.json"
    mixture = json.loads(marker.read_text(encoding="utf-8"))["mixture"] if marker.exists() else "fleurs"
    if mixture == "fleurs":
        return "FLEURS dev; FLEURS test never read. CS set is synthetic inter-sentential concatenation."
    return (f"mixture {mixture}: speaker-disjoint dev carved from the training sources (manifests/data_report.json); "
            "CS set is synthetic inter-sentential concatenation of dev utterances.")


RUNTIME_ARGS = ("--device", "--splits", "--precision", "--stream-workers", "--readahead")  # host-specific, never hashed
RUNTIME_FLAGS = ("--lenient-stream-restore", "--prefetch-pool")  # one-off: never repeated by command.sh


def strip_runtime_args(argv: list[str]) -> list[str]:
    """argv without the host-specific options (device, micro-batches, precision, stream
    workers) and one-off flags, so command.sh reruns on any host with that host's defaults."""
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
            continue
        name = arg.split("=", 1)[0]
        if name in RUNTIME_FLAGS:
            continue
        if name in RUNTIME_ARGS:
            skip = "=" not in arg
            continue
        out.append(arg)
    return out


def scaled_workers(minimum: int = 4, maximum: int = 16, per: int = 4) -> int:
    """Network/IO threads scaled with the host: cores // per, within [minimum, maximum]
    (a 12-vCPU A100 VM keeps 4; a 48-vCPU G4 VM gets 12)."""
    return max(minimum, min(maximum, (os.cpu_count() or 4) // per))


def default_stream_workers(device: str) -> int:
    """Decode/codec threads for the stream: CPU count - 1 on a GPU VM (the GPU does the rest);
    on MPS the main process also needs CPU cores for the log-mel and the augmentation, so
    STREAM_WORKERS_MPS of them."""
    cores = os.cpu_count() or 4
    return min(STREAM_WORKERS_MPS, cores - 1) if device == "mps" else max(1, cores - 1)


STREAM_WORKERS_MPS = 8


def write_command(run_dir: Path, variant: str, args, group: Sequence[str] | None = None) -> None:
    """runs/<run>/command.sh: the exact command, re-runnable on a fresh VM. It rebuilds what a
    lost disk held (environment, data, layer-6 cache: each step skips finished work), then
    relaunches this training command, which resumes from the uploaded checkpoint. A live run
    that shares its stream with other variants (`group`) reruns the whole group, so every
    member's run directory must be restored first; a live run needs no layer-6 cache."""
    import shlex

    from indicear.runtime.checkpoint import atomic_write_text

    run_dir.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parent.parent
    argv = strip_runtime_args(list(args.argv))
    # One variant per command so a resumed VM re-runs exactly this run.
    if "--variants" in argv:
        argv[argv.index("--variants") + 1] = ",".join(group) if group else variant
    env = {k: os.environ[k] for k in ("ROOT", "VENV", "AUG_VIEWS", "STORAGE", "NOISE_DIR", "PREP_ARGS", "SYNC", "SMOKE")
           if os.environ.get(k)}
    live = getattr(args, "lower", "cache") == "live"
    env["ROOT"] = str(Path(args.root).resolve())
    exports = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in sorted(env.items()))
    script = f"""#!/usr/bin/env bash
# Original command for run {variant}; written by indicear.ablate. Safe to rerun: every step resumes.
set -euo pipefail
export CODE={shlex.quote(str(code))}
{exports}
bash "$CODE/scripts/setup_env.sh"
bash "$CODE/scripts/prep.sh"
{"" if live else 'bash "$CODE/scripts/cache.sh"'}
source "$CODE/scripts/env.sh"
indicear indicear.ablate {shlex.join(argv)}
"""
    atomic_write_text(run_dir / "command.sh", script)
    os.chmod(run_dir / "command.sh", 0o755)


def device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        import platform
        import subprocess

        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  timeout=5, check=False).stdout.strip()
        except OSError:
            chip = ""
        return f"{chip or platform.machine()} (MPS)"
    return "cpu"


def environment() -> dict:
    return {"runtime": runtime_banner(), "torch": torch.__version__,
            "gpu": device_name(),
            "encoder": f"{EG2[0]}@{EG2[1]}", "fleurs": f"{FLEURS[0]}@{FLEURS[1]}"}


def finish_run(paths: Paths, variant: str, trainer, model, train: dict, scored: dict) -> None:
    """Save the model, write result.json (run dir) and results/<variant>.json, publish, DONE."""
    run_dir = paths.runs / variant
    from indicear.runtime.checkpoint import git_revision

    result = {"variant": asdict(model.config),
              "lid_config": asdict(model.lid_config) if model.lid_config is not None else None,
              "train": train, **scored, "environment": environment(), "notes": data_notes(paths),
              # provenance for indicear.export (older runs: read from a checkpoint instead)
              "git": git_revision(), "train_config": json.loads(json.dumps(asdict(trainer.cfg), default=str))}
    write_json(run_dir / "result.json", result)
    artifacts = ["result.json", "heads.safetensors", "variant.json"]
    if model.adapters is not None:
        artifacts.append("adapters.safetensors")
    if model.lid_config is not None:
        artifacts.append("lid.json")
    trainer.manager.publish(*artifacts)
    write_json(paths.results / f"{variant}.json", result)
    trainer.manager.mark("DONE")
    log(f"{variant}: wrote results/{variant}.json")


def mark_failed(trainers, error: BaseException) -> None:
    if isinstance(error, SystemExit) and error.code in (0, None, 143):
        return  # a clean stop (SIGTERM) leaves the run resumable, not failed
    detail = f"exit {error.code}" if isinstance(error, SystemExit) else f"{type(error).__name__}: {error}"
    for trainer in trainers:
        trainer.manager.mark("FAILED", detail)


def todo_variants(paths: Paths, variants: Sequence[str]) -> list[str]:
    todo = [v for v in variants if not (paths.results / f"{v}.json").exists()]
    for v in sorted(set(variants) - set(todo)):
        log(f"{v}: results/{v}.json exists, skipping")
    return todo


def cmd_train(args) -> None:
    if args.lower == "live":
        return cmd_train_live(args)
    from indicear.cache import Layer6Cache
    from indicear.encoder import StagedTower, load_tower, runtime_device
    from indicear.evaluate import score
    from indicear.tokenizer import AggregateTokenizer
    from indicear.trainer import TrainConfig, Trainer

    if args.init_from:
        raise SystemExit("--init-from is implemented for live training (--lower live)")
    paths = Paths(args.root)
    paths.results.mkdir(parents=True, exist_ok=True)
    todo = todo_variants(paths, [v for v in args.variants.split(",") if v])
    if not todo:
        return
    device, dtype = runtime_device(args.device)
    tokenizer = AggregateTokenizer.load(paths.tok)
    train_items = read_jsonl(paths.manifest("train")) + read_jsonl(paths.manifest("neg_train"))
    dev = read_jsonl(paths.manifest("dev"))
    eval_items = dev + read_jsonl(paths.manifest("cs_dev")) + read_jsonl(paths.manifest("neg_dev"))
    views = 1 + args.aug_views
    wanted = {i["key"] for i in eval_items} | {
        f"{i['key']}#v{v}" if v else i["key"] for i in train_items for v in range(views)
    }
    cache = Layer6Cache.load(paths.cache, device, vram_fraction=0.45 * args.vram_share, keys=wanted,
                             log=log, placement=args.cache_placement)
    examples, dropped = make_examples(train_items, cache, tokenizer, views)
    tower = load_tower(device, dtype)
    staged = StagedTower(tower)
    train_vocab = metrics.train_vocabulary(i["text"] for i in read_jsonl(paths.manifest("train")))
    quick_items = subset_per_lang(dev, args.eval_items_per_lang)
    hours_by_lang: dict[str, float] = defaultdict(float)
    for item in train_items:
        hours_by_lang[item["lang"]] += float(item.get("seconds") or 0.0) / 3600
    log(f"{runtime_banner()}; {len(examples)} training examples; dropped {dropped}")
    for variant in todo:
        torch.manual_seed(args.seed)
        lid = lid_config_for(variant, hours_by_lang, args)
        model = build_model(variant, staged, tokenizer, variant_overrides(args), lid=lid).to(device)
        extra = {"variant": asdict(model.config), "aug_views": args.aug_views, "encoder": f"{EG2[0]}@{EG2[1]}"}
        if lid is not None:  # only LID runs carry it, so the existing variants keep their config hashes
            extra["lid"] = asdict(lid)
        config = TrainConfig(
            epochs=args.epochs, lr=args.lr, seed=args.seed, frame_budget=args.frame_budget,
            vram_share=args.vram_share, eval_every=args.eval_every, ckpt_every_steps=args.ckpt_every_steps,
            ckpt_every_seconds=args.ckpt_every_seconds, compile_heads=not args.no_compile, splits=args.splits,
            warmup_steps=args.warmup_steps, peak_lr=args.peak_lr, final_lr=args.final_lr, extra=extra,
        )
        run_dir = paths.runs / variant
        copy_dir = Path(args.drive_copy) / "runs" / variant if args.drive_copy else None
        write_command(run_dir, variant, args)
        trainer = Trainer(model, cache, examples, config, run_dir, copy_dir=copy_dir,
                          force_new_run=args.force_new_run, device=device, dtype=dtype, log=log,
                          precision=args.precision)

        def evaluate(m, trainer=trainer):  # eval has no backward: twice the training budget
            return quick_eval(m, cache, quick_items, tokenizer, 2 * trainer.frame_budget, device, dtype)

        trainer.evaluate = evaluate
        try:
            summary = trainer.run()
            model.save(run_dir)
            decoded = decode_items(model, cache, eval_items, tokenizer, 2 * summary["frame_budget"], device, dtype)
            train = {**summary, "examples": len(examples), "dropped": dropped,
                     "train_hours": round(sum(e.frames for e in examples) / FRAME_RATE / 3600, 2)}
            finish_run(paths, variant, trainer, model, train, score(eval_items, decoded, train_vocab))
        except BaseException as error:
            mark_failed([trainer], error)
            raise
        model.detach_adapters()
        del model, trainer
        empty_device_cache()


# ---------------------------------------------------------------- live (streamed) training


def text_encoder_for(tokenizer):
    """text -> (tokenizer ids, phonetic unit ids); None rejects a row whose transcript does not
    encode (the live trainer drops and counts it). Empty text (non-speech) -> ((), ())."""

    def encode(text: str):
        if not text:
            return (), ()
        try:
            ids, units = tokenizer.encode(text), phonetic.encode(text)
        except Exception:  # noqa: BLE001 - dropped and counted, never fatal
            return None
        return (ids, units) if ids and units else None

    return encode


def live_mixture(paths: Paths, args) -> tuple[dict, Path, set[str]]:
    """The prepared mixture (file recorded by prep, unless --mixture) and its allow-review set."""
    from indicear.mixture import load_mixture

    marker = paths.manifests / "mixture.json"
    record = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
    source = args.mixture or record.get("file") or record.get("mixture")
    if not source or source == "fleurs":
        raise SystemExit("train --lower live needs a streamed mixture: run `prep --mixture pilot --stream` first")
    path = mixture_path(source)
    plan_path = paths.manifests / "stream_plan.json"
    if not plan_path.exists():
        raise SystemExit(f"{plan_path} missing: run `prep --mixture {source} --stream` first")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    allow = set(plan.get("allow_review", [])) if args.allow_review is None else {
        s for s in args.allow_review.split(",") if s}
    return load_mixture(path), path, allow


def parse_stream_scale(text: str) -> dict[str, float]:
    """"indicvoices/hi=2,vaani/hi=1.5" -> {"indicvoices/hi": 2.0, "vaani/hi": 1.5}."""
    out = {}
    for part in (p.strip() for p in (text or "").split(",")):
        if not part:
            continue
        name, sep, factor = part.rpartition("=")
        if not sep or not name or float(factor) < 0:
            raise SystemExit(f"--stream-scale: expected NAME=FACTOR (factor >= 0), got {part!r}")
        out[name] = float(factor)
    return out


class LiveData:
    """One stream for one group of variants: planner, shard cache, loader, LoaderSource."""

    def __init__(self, paths: Paths, args, mixture: dict, mixture_file: Path, allow: set[str], policy, log_fn=log):
        from indicear.stream import StreamLoader, build_planner
        from indicear.stream_train import LoaderSource

        negatives = read_jsonl(paths.manifest("neg_train")) if args.negatives_share > 0 else []
        self.setup = build_planner(
            paths.root, mixture, frame_budget=args.frame_budget or 1, allow_review=allow,
            buffer_seconds=args.buffer_seconds, cache_gb=args.cache_gb, fetch_workers=args.fetch_workers,
            augment_prob=0.0 if policy is None else 1.0 - args.clean_prob, extra_rows=negatives,
            fixed_shares={"none": args.negatives_share} if negatives else None, log=log_fn,
            keep_shards=getattr(args, "keep_shards", False), readahead=getattr(args, "readahead", 0),
            zero_streams=[n for n in (getattr(args, "zero_streams", "") or "").split(",") if n],
            stream_scale=parse_stream_scale(getattr(args, "stream_scale", "")))
        self.loader = StreamLoader(self.setup.planner, transform=policy, workers=args.stream_workers or None,
                                   eval_audio=set(self.setup.exclude.get("audio", ())), seed=int(mixture.get("seed", 0)),
                                   cache=self.setup.cache, log=log_fn)
        self.fingerprint = {
            "mixture": mixture["name"], "mixture_sha256": file_sha256(mixture_file), "seed": int(mixture.get("seed", 0)),
            "allow_review": sorted(allow), "buffer_seconds": args.buffer_seconds,
            "augment_prob": round(self.setup.planner.augment_prob, 6),
            "negatives_share": args.negatives_share if negatives else 0.0,
        }
        if mixture.get("loanwords"):  # the normaliser's resources change the targets: hash them
            from indicear.ingest import context_options

            self.fingerprint["loanwords"] = context_options(mixture)["loanwords"].fingerprint()
        self.source = LoaderSource(self.loader, fingerprint=self.fingerprint)
        self.setup.planner.strict_restore = not getattr(args, "lenient_stream_restore", False)
        self.hours = sum(s.spec.hours for s in self.setup.planner.streams if s.spec.lang != "none")
        self.hours_by_lang: dict[str, float] = defaultdict(float)
        for stream in self.setup.planner.streams:
            if stream.spec.lang != "none":
                self.hours_by_lang[stream.spec.lang] += stream.spec.hours

    def close(self) -> None:
        self.setup.close()


def cmd_train_live(args) -> None:
    """Live-lower training of one or more variants on the streamed mixture (see the module doc)."""
    from concurrent.futures import ThreadPoolExecutor

    from indicear.augment import AugmentPolicy, CodecAugment, NoiseBank, parse_policy
    from indicear.encoder import (
        Featurizer,
        StagedTower,
        load_tower,
        runtime_device,
    )
    from indicear.evaluate import decode_models, lower_items, score
    from indicear.tokenizer import AggregateTokenizer
    from indicear.trainer import LiveGroup, TrainConfig, Trainer, eager

    paths = Paths(args.root)
    paths.results.mkdir(parents=True, exist_ok=True)
    variants = [v for v in args.variants.split(",") if v]
    todo = todo_variants(paths, variants)
    if not todo:
        return
    mixture, mixture_file, allow = live_mixture(paths, args)
    if args.prefetch_pool:  # download the whole stream pool for --keep-shards, then stop
        from indicear.stream import download_pool

        args.keep_shards = True
        data = LiveData(paths, args, mixture, mixture_file, allow, None)
        try:
            log(f"prefetch-pool: {download_pool(data.setup, workers=args.fetch_workers, log=log)} in {paths.root / 'stream-cache'}")
        finally:
            data.close()
        return
    device, dtype = runtime_device(args.device)
    if not args.stream_workers:
        args.stream_workers = default_stream_workers(device)
    if args.init_from and not args.share_stream and len(todo) > 1:
        log("warm start: each variant starts from the same source")
    tokenizer = AggregateTokenizer.load(paths.tok)
    tower = load_tower(device, dtype)
    staged = StagedTower(tower)
    tower_dtype = next(tower.parameters()).dtype
    featurizer = Featurizer("gpu", device, tower_dtype)  # torch log-mel on the model device (also on a CPU)
    policy = gpu_augment = None  # policy: the stream workers' transform; gpu_augment: the main process's
    notes: list[str] = []  # host-specific setup, logged (and written to every run's train.log) by each trainer
    augment = {"clean_prob": args.clean_prob, "device": args.augment_device}
    if args.clean_prob < 1.0 and args.augment_device == "gpu":
        from indicear.gpu_augment import GpuAugment, augment_device, parse_gpu_policy

        policy = CodecAugment(args.codec_prob)
        rir_options = {}
        if args.rir_dir:  # real room impulse responses (scripts/rirs.sh) beside the synthetic ones
            from indicear.gpu_augment import rir_files

            rir_options = {"rir_files": rir_files(args.rir_dir), "real_rir_share": args.real_rir_share}
            if not rir_options["rir_files"]:
                raise SystemExit(f"--rir-dir {args.rir_dir}: no RIR files found")
        gpu_augment = GpuAugment(augment_device(device), parse_gpu_policy(args.augment),
                                 seed=int(mixture.get("seed", 0)), noise_files=noise_files(args.noise_dir),
                                 **rir_options)
        augment.update(**policy.describe(), gpu=gpu_augment.describe())
        notes.append(f"augmentation in the main process on {gpu_augment.device.type} (model on {device}): "
                     f"{100 * (1 - args.clean_prob):.0f}% of rows, weights {gpu_augment.weights}; codecs "
                     f"{list(policy.codecs)} of {list(augment['codecs_wanted'])} available on this host, on "
                     f"{args.codec_prob:.0%} of clips in {args.stream_workers} stream workers")
        missing = sorted(set(augment["codecs_wanted"]) - set(policy.codecs))
        if missing:
            notes.append(f"WARNING: ffmpeg here cannot encode {missing}; codec draws go to {list(policy.codecs)} "
                         "instead, so codec augmentation differs from a host that has them")
    elif args.clean_prob < 1.0:
        policy = AugmentPolicy(NoiseBank(noise_files(args.noise_dir)), parse_policy(args.augment))
        augment.update(policy.describe())
        notes.append(f"augmentation on the CPU: {100 * (1 - args.clean_prob):.0f}% of rows, weights {policy.weights}, "
                     f"codecs available {list(policy.codecs)}")
    dev = read_jsonl(paths.manifest("dev"))
    eval_items = dev + read_jsonl(paths.manifest("cs_dev")) + read_jsonl(paths.manifest("neg_dev"))
    quick_items = subset_per_lang(dev, args.eval_items_per_lang)
    texts = paths.manifest("train_texts")
    train_vocab = metrics.train_vocabulary(i["text"] for i in read_jsonl(texts)) if texts.exists() else None
    encode = text_encoder_for(tokenizer)
    groups = [todo] if args.share_stream else [[v] for v in todo]
    pool = ThreadPoolExecutor(max_workers=max(1, args.stream_workers or 4), thread_name_prefix="dev-audio")
    try:
        for group in groups:
            data = LiveData(paths, args, mixture, mixture_file, allow, policy)
            trainers, models = [], {}
            try:
                train_hours = args.train_hours or args.epochs * data.hours
                quick: dict = {}
                for variant in group:
                    torch.manual_seed(args.seed)
                    lid = lid_config_for(variant, data.hours_by_lang, args)
                    model = build_model(variant, staged, tokenizer, variant_overrides(args), lid=lid).to(device)
                    extra = {"variant": asdict(model.config), "encoder": f"{EG2[0]}@{EG2[1]}", "augment": augment}
                    if lid is not None:
                        extra["lid"] = asdict(lid)
                    run_dir = paths.runs / variant
                    warm = None
                    if args.init_from:  # a new run initialized from a finished one (indicear.warmstart)
                        from indicear import warmstart

                        keep = [s for s in args.init_keep_scripts.split(",") if s] if args.init_keep_scripts else None
                        warm = warmstart.prepare(run_dir, args.init_from, model, tokenizer, keep, args.init_tok,
                                                 args.new_script_logits, args.new_script_bias, args.force_new_run, log)
                        extra["init_from"] = warm.summary
                    config = TrainConfig(
                        epochs=args.epochs, lr=args.lr, seed=args.seed, frame_budget=args.frame_budget,
                        vram_share=args.vram_share, eval_every=args.eval_every, ckpt_every_steps=args.ckpt_every_steps,
                        ckpt_every_seconds=args.ckpt_every_seconds, compile_heads=not args.no_compile,
                        max_steps=args.max_steps, train_hours=0.0 if args.max_steps else train_hours,
                        packing=args.packing, splits=args.splits, warmup_steps=args.warmup_steps,
                        peak_lr=args.peak_lr, final_lr=args.final_lr, new_lr_mult=args.new_lr_mult, extra=extra)
                    copy_dir = Path(args.drive_copy) / "runs" / variant if args.drive_copy else None
                    write_command(run_dir, variant, args, group if len(group) > 1 else None)
                    trainer = Trainer(model, None, [], config, run_dir, copy_dir=copy_dir,
                                      force_new_run=args.force_new_run, device=device, dtype=dtype, log=log,
                                      lower="live", stream=data.source, featurizer=featurizer, text_encoder=encode,
                                      augment=gpu_augment, precision=args.precision, warm=warm)
                    trainer.log(f"host: {environment()['gpu']}, device {device}, precision {args.precision}, "
                                f"torch {torch.__version__}")
                    for note in notes:
                        trainer.log(note)

                    def evaluate(m, trainer=trainer, quick=quick):  # the dev subset's layer-6 states, once
                        if "cache" not in quick:
                            quick["cache"] = lower_items(quick_items, staged, featurizer, trainer.frame_budget, pool,
                                                         log=log)
                        return quick_eval(m, quick["cache"], quick_items, tokenizer, 2 * trainer.frame_budget,
                                          device, dtype, log=trainer.log)

                    trainer.evaluate = evaluate if quick_items else None
                    trainers.append(trainer)
                    models[variant] = model
                log(f"{runtime_banner()}; live training of {', '.join(group)} on {mixture['name']} "
                    f"({data.hours:.0f} h planned, target {train_hours:.0f} h of audio)")
                summaries = LiveGroup(trainers).run()
                quick.clear()
                for variant, model in models.items():
                    with model.adapters_attached():
                        model.save(paths.runs / variant)
                budget = summaries[0]["frame_budget"]
                with eager():  # arbitrary eval shapes: never compile them
                    decoded = decode_models(models, eval_items, staged, featurizer, tokenizer, budget, device, dtype,
                                            workers=args.stream_workers or 4, log=log)
                stream_report = data.setup.planner.report()
                for (variant, model), trainer, summary in zip(models.items(), trainers, summaries, strict=True):
                    train = {**summary, "mixture": mixture["name"], "train_hours_target": round(train_hours, 2),
                             "stream": {k: stream_report[k] for k in ("probabilities", "epochs")},
                             "stream_drops": stream_report["drops"]}
                    finish_run(paths, variant, trainer, model, train, score(eval_items, decoded[variant], train_vocab))
            except BaseException as error:
                mark_failed(trainers, error)
                if isinstance(error, SystemExit) and error.code == 143 and os.environ.get("INDICEAR_FAST_EXIT"):
                    # Stopped after its checkpoint: do not wait for in-flight shard downloads (on a slow link
                    # closing the shard cache can block for many minutes); scripts/resume_local.sh sets it.
                    log("stopped after the checkpoint; exiting without waiting for in-flight downloads")
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(143)
                raise
            finally:
                data.close()
                for model in models.values():
                    model.detach_adapters()
                del trainers, models
                empty_device_cache()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------- eval


def cmd_eval(args) -> None:
    """Every eval set x every trained variant -> results/<variant>/<eval set>.json, then the summary."""
    from indicear import evaluate
    from indicear.common import GpuSampler
    from indicear.decoding import Constrainer
    from indicear.encoder import (
        Featurizer,
        StagedTower,
        load_tower,
        runtime_device,
    )
    from indicear.model import load_lid_config
    from indicear.tokenizer import AggregateTokenizer

    paths = Paths(args.root)
    variants = [v for v in args.variants.split(",") if v]
    ready = [v for v in variants if (paths.runs / v / "heads.safetensors").exists() and (paths.results / f"{v}.json").exists()]
    for v in sorted(set(variants) - set(ready)):
        log(f"{v}: not trained yet (no results/{v}.json and heads), skipping")
    sets = evaluate.eval_sets(paths.manifests, args.sets.split(","), args.max_per_lang,
                              [c for c in args.robust_conditions.split(",") if c], args.robust_per_lang,
                              noise_files(args.noise_dir))
    log(f"eval sets: {', '.join(f'{s.name} ({len(s.items)})' for s in sets)}")
    pending = {s.name: [v for v in ready if not evaluate.result_path(paths.results, v, s.name).exists()] for s in sets}
    if any(pending.values()):
        device, dtype = runtime_device(args.device)
        tokenizer = AggregateTokenizer.load(paths.tok)
        # Scored as a second decoder, "ctc_constrained" (indicear.decoding defaults), beside plain "ctc".
        constraints = None if args.no_constraints else Constrainer.from_tokenizer(tokenizer)
        tower = load_tower(device, dtype)
        staged = StagedTower(tower)
        featurizer = Featurizer("gpu", device, next(tower.parameters()).dtype)  # torch log-mel, as in training
        models = {}
        for variant in ready:
            if not any(variant in todo for todo in pending.values()):
                continue
            model = build_model(variant, staged, tokenizer, lid=load_lid_config(paths.runs / variant)).to(device)
            model.load(paths.runs / variant)
            model.detach_adapters()
            models[variant] = model
        budgets = [json.loads((paths.results / f"{v}.json").read_text())["train"].get("frame_budget") for v in models]
        budget = args.frame_budget or min([b for b in budgets if b] or [2000])
        texts = paths.manifest("train_texts") if paths.manifest("train_texts").exists() else paths.manifest("train")
        vocab = metrics.train_vocabulary(i["text"] for i in read_jsonl(texts)) if texts.exists() else None
        paths.results.mkdir(parents=True, exist_ok=True)
        with GpuSampler(paths.results / "eval-gpu.csv") as gpu:
            for eval_set in sets:
                todo = {v: models[v] for v in pending[eval_set.name]}
                if not todo:
                    continue
                started = time.time()
                decoded = evaluate.decode_models(todo, eval_set.items, staged, featurizer, tokenizer, budget, device,
                                                 dtype, workers=args.workers, transform=eval_set.transform,
                                                 seed=17, chunk_seconds=args.chunk_seconds, log=log,
                                                 constraints=constraints)
                for variant in todo:
                    evaluate.write_result(paths.results, variant, eval_set,
                                          evaluate.score(eval_set.items, decoded[variant], vocab),
                                          {"environment": environment(), "frame_budget": budget})
                log(f"{eval_set.name}: {', '.join(todo)} in {time.time() - started:.0f} s")
            log(gpu.summary())
        for model in models.values():
            model.detach_adapters()
    summary = evaluate.summarize(paths.results)
    log(f"results/eval_summary.md: {len(summary['rows'])} rows")


# ---------------------------------------------------------------- report


def cmd_report(args) -> None:
    paths = Paths(args.root)
    rows = []
    for path in sorted(paths.results.glob("*.json")):  # results/<variant>.json: the training runs' dev report
        result = json.loads(path.read_text())
        if not isinstance(result, dict) or "variant" not in result:
            continue
        for decoder, groups in result["metrics"].items():
            for group, values in groups.items():
                rows.append((path.stem, decoder, group, values))
        lid = (result.get("lid") or {}).get("utterance")
        if lid:
            rows.append((path.stem, "lid", "utterance LID", {"lid_accuracy": lid.get("accuracy")}))
    lines = ["| variant | decoder | set | CER | WER | sn-WER | script err | OOV recall | span acc | false output | LID acc |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant, decoder, group, v in rows:
        lines.append(
            f"| {variant} | {decoder} | {group} | {v.get('cer', '')} | {v.get('wer', '')} | "
            f"{v.get('wer_script_normalised', '')} | {v.get('script_error_rate', '')} | {v.get('oov_recall', '')} | "
            f"{v.get('span_accuracy', '')} | {v.get('false_output_rate', '')} | {v.get('lid_accuracy', '')} |"
        )
    table = "\n".join(lines)
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    from indicear.evaluate import summarize

    evaluation = summarize(paths.results)
    if evaluation["rows"]:
        print(evaluation["table"])


# ---------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="/content/indicear")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prep")
    prep.add_argument("--mixture", default="fleurs",
                      help="fleurs (default, the FLEURS ablation), pilot, full, or a mixture JSON path")
    prep.add_argument("--allow-review", default="",
                      help="comma-separated sources under licence review to use for training (e.g. mucs)")
    prep.add_argument("--disk-budget-gb", type=float, default=None,
                      help="stop cleanly before the downloaded audio would exceed this many GiB")
    prep.add_argument("--download-workers", type=int, default=0,
                      help="sources/shards fetched at once; 0 = CPU count / 4, between 4 and 16")
    prep.add_argument("--decode-workers", type=int, default=0, help="audio decode threads; 0 = CPU count")
    prep.add_argument("--keep-downloads", action="store_true", help="keep parquet/tar files after decoding")
    prep.add_argument("--stream", action="store_true",
                      help="streaming mode: download only eval/dev/tar-only sets; training audio streams from the Hub "
                           "(indicear.stream); the tokenizer is trained from the text columns alone")
    prep.add_argument("--langs", default=",".join(LANGS), help="FLEURS languages (the mixture sets its own)")
    prep.add_argument("--fleurs-dev-langs", default="mixture",
                      help="--stream: FLEURS dev as eval set fleurs_dev, for continuity: 'mixture' (its training "
                           "languages), 'none', or a comma list")
    prep.add_argument("--max-train-hours", type=float, default=0.0, help="per language; 0 = all")
    prep.add_argument("--cs-per-pair", type=int, default=100)
    prep.add_argument("--negatives", type=int, default=400)
    prep.add_argument("--negatives-dev", type=int, default=100)
    prep.add_argument("--noise-dir", default=None, help="MUSAN-style noise files (optional)")
    prep.add_argument("--tok-from", default=None,
                      help="warm start: the source run's tokenizer directory; the --tok-keep scripts reuse its "
                           "SentencePiece models unchanged and the others are trained on this mixture's text")
    prep.add_argument("--tok-keep", default="deva,latn,taml,knda", help="scripts kept from --tok-from")
    cache = sub.add_parser("cache")
    cache.add_argument("--storage", choices=("bf16", "int8"), default="bf16")
    cache.add_argument("--aug-views", type=int, default=0, help="extra augmented copies per train utterance")
    cache.add_argument("--noise-dir", default=None)
    cache.add_argument("--batch-seconds", type=float, default=600.0)
    cache.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    cache.add_argument("--features", choices=("cpu", "gpu"), default=None,
                       help="where the log-mel is computed (default: gpu when CUDA is available); "
                            "not part of the cache identity")
    cache.add_argument("--augment-device", choices=("cpu", "gpu"), default=None,
                       help="--aug-views: where views are corrupted (default: gpu with --features gpu; "
                            "codecs stay in the workers)")
    cache.add_argument("--augment", default="", help="GPU augmentation kind weights (indicear.gpu_augment)")
    cache.add_argument("--codec-prob", type=float, default=0.15, help="share of augmented views through a codec")
    train = sub.add_parser("train")
    train.add_argument("--variants", default="v1,v2,v3,v4")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--lr", type=float, default=1e-3, help="peak LR before the sqrt(budget / 20k frames) scaling")
    train.add_argument("--peak-lr", type=float, default=0.0,
                       help="the peak LR itself, without budget scaling (0: --lr scaled; warm start: 6e-4)")
    train.add_argument("--final-lr", type=float, default=0.0,
                       help="the cosine floor as an absolute LR (0: 5%% of the peak; warm start: 5e-5)")
    train.add_argument("--warmup-steps", type=int, default=300, help="linear warmup (at most 10%% of the steps)")
    train.add_argument("--inter-weight", type=float, default=None,
                       help="phonetic intermediate CTC weight (default: the variant's, 0.3)")
    train.add_argument("--lid-frame-weight", type=float, default=None, help="frame LID loss weight (default 0.1)")
    train.add_argument("--lid-utt-weight", type=float, default=None, help="utterance LID loss weight (default 0.1)")
    warm = train.add_argument_group("warm start (indicear.warmstart: a new run initialized from a finished one)")
    warm.add_argument("--init-from", default=None,
                      help="run directory (heads/adapters safetensors, variant.json, lid.json, tok/) or checkpoint")
    warm.add_argument("--init-tok", default=None, help="the source's tokenizer (default: <run>/tok)")
    warm.add_argument("--init-keep-scripts", default=None,
                      help="scripts whose output rows are copied (default: those prep --tok-from kept)")
    warm.add_argument("--new-script-logits", choices=("fresh", "copy"), default="fresh",
                      help="a new script's class row: fresh (zero weights, --new-script-bias) or the source's")
    warm.add_argument("--new-script-bias", type=float, default=-2.0,
                      help="bias of a fresh script class: low enough that the kept outputs stay bit-identical")
    warm.add_argument("--new-lr-mult", type=float, default=1.0,
                      help="LR multiplier of the rows and parameters the warm start left fresh (warm start: 3)")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--aug-views", type=int, default=0, help="must match the cache build")
    train.add_argument("--frame-budget", type=int, default=0, help="0 = size from free VRAM")
    train.add_argument("--vram-share", type=float, default=1.0, help="1/N when N jobs share the GPU")
    train.add_argument("--cache-placement", choices=("auto", "vram", "pinned", "mmap"), default="auto")
    train.add_argument("--checkpoint-layers", action="store_true", help="activation checkpointing in layers 7-12")
    train.add_argument("--eval-every", type=int, default=1000)
    train.add_argument("--eval-items-per-lang", type=int, default=60)
    train.add_argument("--ckpt-every-steps", type=int, default=500)
    train.add_argument("--ckpt-every-seconds", type=float, default=600.0)
    train.add_argument("--drive-copy", default=None,
                       help="optional second copy of runs/ (e.g. a Drive mount); the primary copy is pulled to the workstation")
    train.add_argument("--force-new-run", action="store_true")
    train.add_argument("--no-compile", action="store_true")
    train.add_argument("--device", choices=DEVICE_CHOICES, default=None,
                       help="default: $INDICEAR_DEVICE, else auto (CUDA, then MPS, then CPU); not hashed")
    train.add_argument("--splits", type=int, default=0,
                       help="micro-batches per step (gradient accumulation, exact); 0 = the checkpoint's count on "
                            "the same device type, else derived from device memory; not hashed")
    train.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto",
                       help="auto: bf16 autocast on CUDA, fp32 on MPS/CPU; bf16/fp16: half-precision autocast on MPS "
                            "(fp16 with a GradScaler); not hashed")
    live = train.add_argument_group("live lower (--lower live: the streamed mixture prepared with prep --stream)")
    live.add_argument("--lower", choices=("cache", "live"), default="cache",
                      help="cache: layer-6 cache (FLEURS ablation); live: stream the prepared mixture, layers 1-6 every step")
    live.add_argument("--mixture", default=None, help="default: the mixture recorded by prep in manifests/mixture.json")
    live.add_argument("--allow-review", default=None, help="default: what prep recorded in manifests/stream_plan.json")
    live.add_argument("--train-hours", type=float, default=0.0,
                      help="audio hours to train on (default: --epochs x the mixture's planned hours; pilot 650 x 10)")
    live.add_argument("--max-steps", type=int, default=0, help="fix the step count instead of --train-hours")
    live.add_argument("--packing", type=float, default=0.85,
                      help="assumed real frames / frame budget per streamed batch, for the hours -> steps estimate")
    live.add_argument("--share-stream", action=argparse.BooleanOptionalAction, default=True,
                      help="train every variant in one process from one stream (default); --no-share-stream: one by one")
    live.add_argument("--clean-prob", type=float, default=0.5, help="share of rows left clean (1 disables augmentation)")
    live.add_argument("--augment-device", choices=("gpu", "cpu"), default="gpu",
                      help="gpu (default): noise/babble/reverb/speed/tel8k on the device in the main process, codecs on "
                           "--codec-prob of clips in the stream workers (indicear.gpu_augment); cpu: everything in the "
                           "workers (augment.AugmentPolicy)")
    live.add_argument("--augment", default="",
                      help="kind weights; gpu default noise=0.25,babble=0.2,reverb=0.175,reverb_pointnoise=0.175,"
                           "speed=0.15,tel8k=0.05; cpu default noise=0.2,babble=0.15,reverb=0.15,"
                           "reverb_pointnoise=0.15,codec=0.2,speed=0.15")
    live.add_argument("--codec-prob", type=float, default=0.15,
                      help="--augment-device gpu: share of clips through an ffmpeg codec in the workers")
    live.add_argument("--noise-dir", default=None, help="MUSAN-style noise files (else synthetic colored noise)")
    live.add_argument("--negatives-share", type=float, default=0.02,
                      help="sampling share of the non-speech negatives (manifests/neg_train.jsonl); 0 = none")
    live.add_argument("--buffer-seconds", type=float, default=1800.0, help="shuffle buffer, seconds of audio")
    live.add_argument("--cache-gb", type=float, default=40.0, help="rolling shard cache bound")
    live.add_argument("--zero-streams", default="",
                      help="comma-separated stream names never drawn from (weight 0), e.g. streams whose filters "
                           "leave too few rows; not hashed, so a run resumes with it (it is logged and kept in "
                           "command.sh)")
    live.add_argument("--stream-scale", default="",
                      help="comma-separated NAME=FACTOR: multiply streams' sampling weights (e.g. rebalance two "
                           "languages mid-run); not hashed, logged and kept in command.sh")
    live.add_argument("--readahead", type=int, default=64,
                      help="rows each streamed source reads ahead in its own thread (0: read on demand)")
    live.add_argument("--fetch-workers", type=int, default=0,
                      help="shard downloads in flight; 0 = CPU count / 4, between 4 and 16")
    live.add_argument("--rir-dir", default=None,
                      help="real room impulse responses (scripts/rirs.sh: OpenSLR SLR28) used beside the synthetic ones")
    live.add_argument("--real-rir-share", type=float, default=0.5,
                      help="--rir-dir: share of reverberated rows that use a real RIR")
    live.add_argument("--keep-shards", action="store_true",
                      help="keep downloaded shards in ROOT/stream-cache across epochs and restarts (the whole pool, "
                           "68 GiB for the pilot) instead of deleting each after use: for hosts with disk but little "
                           "bandwidth; --prefetch-pool downloads it first; not hashed")
    live.add_argument("--prefetch-pool", action="store_true",
                      help="download every pool shard into ROOT/stream-cache, then exit (no training)")
    live.add_argument("--lenient-stream-restore", action="store_true",
                      help="on resume, leave out buffered rows this host's filters reject (a prep root rebuilt "
                           "elsewhere excludes other dev speakers) instead of stopping; logged; not hashed")
    live.add_argument("--stream-workers", type=int, default=0,
                      help="decode/codec threads; 0 = CPU count - 1 (CUDA) or STREAM_WORKERS_MPS (MPS)")
    evaluation = sub.add_parser("eval", help="every eval set x every trained variant -> results/<variant>/<set>.json")
    evaluation.add_argument("--variants", default="v2,v3,v4,v4+lid")
    evaluation.add_argument("--sets", default="*", help="comma-separated globs on eval set names")
    evaluation.add_argument("--max-per-lang", type=int, default=0, help="cap per set and language (0 = all)")
    evaluation.add_argument("--robust-conditions", default=",".join(ROBUST_DEFAULT),
                            help="robustness subset conditions (indicear.augment); '' disables")
    evaluation.add_argument("--robust-per-lang", type=int, default=100)
    evaluation.add_argument("--noise-dir", default=None)
    evaluation.add_argument("--frame-budget", type=int, default=0, help="0 = the smallest training budget")
    evaluation.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    evaluation.add_argument("--chunk-seconds", type=float, default=3600.0, help="audio lowered per chunk")
    evaluation.add_argument("--no-constraints", action="store_true",
                            help="skip the decode-time script-constrained decoder (ctc_constrained)")
    evaluation.add_argument("--device", choices=DEVICE_CHOICES, default=None,
                            help="default: $INDICEAR_DEVICE, else auto (CUDA, then MPS, then CPU)")
    sub.add_parser("report")
    args = parser.parse_args(argv)
    args.argv = list(argv if argv is not None else sys.argv[1:])
    for name in ("download_workers", "fetch_workers"):  # 0: scale with the host's cores
        if getattr(args, name, None) == 0:
            setattr(args, name, scaled_workers())
    if os.environ.get("INDICEAR_TORCH_THREADS"):  # CPU cap (scripts/resume_local.sh: CPU_THREADS)
        torch.set_num_threads(int(os.environ["INDICEAR_TORCH_THREADS"]))
    log(f"{runtime_banner()}, torch threads {torch.get_num_threads()}")
    {"prep": cmd_prep, "cache": cmd_cache, "train": cmd_train, "eval": cmd_eval,
     "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
