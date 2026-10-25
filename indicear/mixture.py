"""Training mixtures: hours per language per source, speaker-disjoint dev, eval sets, leak checks and
temperature sampling.

    python -m indicear.mixture plan --mixture pilot [--allow-review mucs]   # offline size estimate
    python -m indicear.ablate --root R prep --mixture pilot                 # build (indicear.ablate)

A mixture file (`indicear/mixtures/<name>.json`) lists training tasks `{source, lang, split, hours,
code_mixed_only?}` and eval sets `{name, source, lang, split, hours?}`. Building one:

1. runs the eval tasks first, then the training tasks (`indicear.ingest`, resumable, disk-checked);
2. drops duplicate keys and duplicate audio (SHA-1 of the 16 kHz PCM);
3. removes from training every row whose speaker, speaker group or audio hash occurs in any eval
   set (counted per reason), then asserts that no eval utterance or speaker is left in training;
4. carves a dev set per language from whole speaker groups (speaker id, else a recording/file-name
   group, else the audio hash), so train and dev are speaker-disjoint, and asserts it;
5. computes temperature-sampling shares over languages plus a code-switched group
   (p ~ hours^alpha, alpha 0.5, with a Hindi cap) and writes each row's `sample_weight`, which the
   trainer's sampler uses to draw each epoch.

Licence gate (indicear.sources): `review` sources are used for eval sets but are skipped for training
unless named in `--allow-review`; `excluded` and `blocked` sources are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from indicear.ingest import FLAC_MB_PER_HOUR, Task, TaskResult
from indicear.sources import SOURCES, allowed

MIXTURES = Path(__file__).resolve().parent / "mixtures"
CODE_SWITCHED = "cs"


class LeakError(AssertionError):
    """An evaluation utterance or speaker reached the training or dev data."""


def load_mixture(name: str | Path) -> dict:
    path = Path(name)
    if not path.suffix:
        path = MIXTURES / f"{name}.json"
    mixture = json.loads(path.read_text(encoding="utf-8"))
    mixture.setdefault("name", path.stem)
    for entry in mixture.get("train", []):
        source = SOURCES[entry["source"]]
        if entry["lang"] not in source.langs:
            raise ValueError(f"{path.name}: {entry['source']} has no language {entry['lang']}")
        if entry["split"] not in source.splits:
            raise ValueError(f"{path.name}: {entry['source']} has no split {entry['split']}")
    for entry in mixture.get("eval", []):
        source = SOURCES[entry["source"]]
        if entry["lang"] not in source.langs or entry["split"] not in source.splits:
            raise ValueError(f"{path.name}: bad eval entry {entry}")
    return mixture


def plan_tasks(mixture: dict, allow_review: set[str] = frozenset()) -> tuple[list[Task], list[Task], list[str]]:
    """(eval tasks, train tasks, skipped messages) after the licence gate."""
    skipped: list[str] = []
    evals, trains = [], []
    for entry in mixture.get("eval", []):
        source = SOURCES[entry["source"]]
        ok, why = allowed(source, allow_review | ({source.name} if source.license_status == "review" else set()))
        if not ok:
            skipped.append(f"eval {entry['name']}: {why}")
            continue
        evals.append(Task(entry["source"], entry["lang"], entry["split"], float(entry.get("hours", math.inf)),
                          role="eval", eval_name=entry["name"]))
    for entry in mixture.get("train", []):
        ok, why = allowed(SOURCES[entry["source"]], allow_review)
        if not ok:
            skipped.append(f"train {entry['source']}/{entry['lang']}: {why} ({entry['hours']} h not loaded)")
            continue
        trains.append(Task(entry["source"], entry["lang"], entry["split"], float(entry["hours"]),
                           code_mixed_only=bool(entry.get("code_mixed_only")),
                           reverse=bool(entry.get("code_mixed_only")), weight=float(entry.get("weight", 1.0))))
    return evals, trains, skipped


# ---------------------------------------------------------------- estimates (offline)


CODE_MIXED_SHARE = 0.3  # assumed share of Vaani rows with an English gloss (public samples: higher)
KEEP_RATIO = 0.85  # assumed share of rows surviving markup and duration filters


def estimate(mixture: dict, allow_review: set[str] = frozenset(), workers: int = 4) -> dict:
    """Download volume and disk use from each source's MB-per-hour figure (planning only)."""
    evals, trains, skipped = plan_tasks(mixture, allow_review)
    per_source: dict[str, dict[str, float]] = defaultdict(lambda: {"hours": 0.0, "download_gb": 0.0, "flac_gb": 0.0})
    largest_shard_gb = {"indicvoices": 0.85, "vaani": 0.66, "kathbath": 0.6, "shrutilipi": 0.5, "spring_inx": 0.92,
                        "librispeech": 0.6, "peoples_speech": 0.5, "voxpopuli": 3.3, "svarah": 0.37, "lahaja": 0.37,
                        "vaani_benchmark": 0.42}
    tar_gb = 0.0
    for task in trains + evals:
        source = SOURCES[task.source]
        hours = task.hours if task.hours != math.inf else _full_hours(task)
        scan = hours / KEEP_RATIO / (CODE_MIXED_SHARE if task.code_mixed_only else 1.0)
        row = per_source[task.source]
        row["hours"] += hours
        row["flac_gb"] += hours * FLAC_MB_PER_HOUR / 1000
        if source.kind == "url-tar":
            continue
        row["download_gb"] += scan * source.mb_per_hour / 1000
    for name, source in SOURCES.items():
        if source.kind != "url-tar" or name not in per_source:
            continue
        needed = {t.split for t in trains + evals if t.source == name}
        files = _tar_files(name, needed, {t.lang for t in trains + evals if t.source == name})
        size = sum((source.files[f]["size"] or 1.0e9) for f in files) / 1e9
        per_source[name]["download_gb"] += size
        tar_gb += size if name == "kathbath_test" else 0.0
    download = sum(r["download_gb"] for r in per_source.values())
    flac = sum(r["flac_gb"] for r in per_source.values())
    used = {t.source for t in trains + evals}
    transient = workers * max([largest_shard_gb.get(s, 0.5) for s in used] or [0.0]) + tar_gb + (
        7.4 if any(t.source == "mucs" and t.split == "train" for t in trains) else 0.0)
    eval_hours = sum(t.hours if t.hours != math.inf else _full_hours(t) for t in evals)
    langs = {t.lang for t in trains}
    dev_hours = len(langs) * float(mixture.get("dev", {}).get("max_hours_per_lang", 2.0))
    tar_train_hours = sum(t.hours for t in trains if SOURCES[t.source].kind == "url-tar")
    stream_network = sum(t.hours / KEEP_RATIO / (CODE_MIXED_SHARE if t.code_mixed_only else 1.0)
                         * SOURCES[t.source].mb_per_hour / 1000 for t in trains if SOURCES[t.source].kind == "hf-parquet")
    fixed_flac = (eval_hours + dev_hours + tar_train_hours) * FLAC_MB_PER_HOUR / 1000
    streaming = {"fixed_flac_gb": round(fixed_flac, 1), "cache_gb": 40.0,
                 "fixed_downloads_gb": round(tar_gb + sum(
                     (t.hours if t.hours != math.inf else _full_hours(t)) * SOURCES[t.source].mb_per_hour / 1000
                     for t in evals if SOURCES[t.source].kind == "hf-parquet") + dev_hours * 4 * 0.1, 1),
                 "disk_gb": round(fixed_flac + 40.0 + tar_gb, 1),
                 "network_gb_per_pass": round(stream_network, 1)}
    return {
        "mixture": mixture["name"], "train_hours": round(sum(t.hours for t in trains), 1),
        "streaming": streaming,
        "eval_hours_estimate": round(sum(t.hours if t.hours != math.inf else _full_hours(t) for t in evals), 1),
        "download_gb": round(download, 1), "flac_gb": round(flac, 1),
        "peak_transient_gb": round(transient, 1), "disk_gb": round(flac + transient, 1),
        "per_source": {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in sorted(per_source.items())},
        "skipped": skipped,
        "assumptions": (f"MB per hour from card bytes / card hours; FLAC {FLAC_MB_PER_HOUR:.0f} MB/h; "
                        f"{KEEP_RATIO:.0%} of rows kept; code-mixed mining scans 1/{CODE_MIXED_SHARE:.0%}; "
                        f"{workers} shards in flight; downloads deleted after decoding"),
    }


# Planning sizes of whole eval splits (hours): card figures where stated, else bytes / MB-per-hour.
EVAL_HOURS = {
    ("svarah", "test"): 9.6, ("lahaja", "test"): 12.5, ("vaani_benchmark", "test"): 10.9,
    ("librispeech", "test.clean"): 5.4, ("librispeech", "test.other"): 5.3, ("voxpopuli", "test"): 5.1,
    ("voxpopuli", "accented_test"): 27.0, ("mucs", "test"): 6.0, ("mucs", "blind"): 12.0,
    ("indicvoices", "valid"): 7.5, ("kathbath_test", None): 2.5,
}


def _full_hours(task: Task) -> float:
    return EVAL_HOURS.get((task.source, task.split), EVAL_HOURS.get((task.source, None), 3.0))


def _tar_files(name: str, splits: set[str], langs: set[str]) -> list[str]:
    source = SOURCES[name]
    if name == "mucs":
        files = []
        for lang in langs:
            for split in splits:
                files.append("blindtest" if split == "blind" else f"{source.langs[lang]}_{split}")
        return sorted(set(files))
    files = set()
    for split in splits:
        noise = split.split("_", 1)[0]
        kind = "testkn" if split.endswith("known") and not split.endswith("unknown") else "testunk"
        files |= {f"{noise}_transcripts", f"{noise}_{kind}"}
    return sorted(files)


# ---------------------------------------------------------------- speaker groups, dev, leaks


def group_of(row: dict) -> str:
    return row.get("speaker") or row.get("group") or f"audio:{row['audio_sha1']}"


def _stable(value: str, seed: int) -> int:
    return int(hashlib.sha1(f"{seed}:{value}".encode()).hexdigest()[:12], 16)


def dedupe(rows: Iterable[dict], drops: Counter) -> list[dict]:
    keys, hashes, out = set(), set(), []
    for row in rows:
        if row["key"] in keys:
            drops["duplicate_key"] += 1
            continue
        if row["audio_sha1"] in hashes:
            drops["duplicate_audio"] += 1
            continue
        keys.add(row["key"])
        hashes.add(row["audio_sha1"])
        out.append(row)
    return out


def eval_index(eval_rows: Iterable[dict]) -> dict[str, set[str]]:
    index = {"speaker": set(), "group": set(), "audio": set()}
    for row in eval_rows:
        if row.get("speaker"):
            index["speaker"].add(row["speaker"])
        if row.get("group"):
            index["group"].add(row["group"])
        index["audio"].add(row["audio_sha1"])
    return index


def remove_eval_overlap(rows: Iterable[dict], eval_rows: Sequence[dict], drops: Counter) -> list[dict]:
    """Drop training rows that share a speaker, a speaker group or the audio with any eval row."""
    index = eval_index(eval_rows)
    kept = []
    for row in rows:
        if row.get("speaker") and row["speaker"] in index["speaker"]:
            drops[f"eval_speaker:{row['source']}"] += 1
        elif row.get("group") and row["group"] in index["group"]:
            drops[f"eval_group:{row['source']}"] += 1
        elif row["audio_sha1"] in index["audio"]:
            drops[f"eval_audio:{row['source']}"] += 1
        else:
            kept.append(row)
    return kept


def assert_no_leak(train_rows: Iterable[dict], eval_rows: Sequence[dict]) -> None:
    """Raise LeakError when an eval utterance (audio hash) or speaker/group occurs in training."""
    index = eval_index(eval_rows)
    for row in train_rows:
        for kind, value in (("speaker", row.get("speaker")), ("group", row.get("group")),
                            ("audio", row["audio_sha1"])):
            if value and value in index[kind]:
                raise LeakError(f"eval {kind} {value} occurs in training row {row['key']}")


def split_dev(rows: Sequence[dict], fraction: float, max_hours: float, min_hours: float = 0.0,
              seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Speaker-disjoint dev per language: whole groups in a seeded order until the dev share
    (fraction of the language, clamped to [min_hours, max_hours]) is reached."""
    by_lang: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_lang[row["lang"]][group_of(row)].append(row)
    train, dev = [], []
    for lang in sorted(by_lang):
        groups = by_lang[lang]
        total = sum(r["seconds"] for g in groups.values() for r in g)
        target = min(max(fraction * total, min_hours * 3600), max_hours * 3600, 0.5 * total)
        taken = 0.0
        for group in sorted(groups, key=lambda g: _stable(g, seed)):
            members = groups[group]
            if taken < target and len(groups) > 1:
                dev.extend(members)
                taken += sum(r["seconds"] for r in members)
            else:
                train.extend(members)
    train_groups = {group_of(r) for r in train}
    clash = [r["key"] for r in dev if group_of(r) in train_groups]
    if clash:
        raise LeakError(f"dev speakers also in train: {clash[:3]}")
    return train, dev


# ---------------------------------------------------------------- temperature sampling


def sampling_group(row: dict) -> str:
    return CODE_SWITCHED if row.get("code_mixed") else row["lang"]


def temperature_shares(hours: dict[str, float], alpha: float = 0.5, max_share: dict[str, float] | None = None,
                       min_share: dict[str, float] | None = None) -> dict[str, float]:
    """p_g ~ hours_g ** alpha, then clamp to the caps/floors and spread the remainder over the rest
    in proportion to their tempered weight."""
    hours = {g: h for g, h in hours.items() if h > 0}
    if not hours:
        return {}
    weight = {g: h ** alpha for g, h in hours.items()}
    max_share, min_share = dict(max_share or {}), dict(min_share or {})
    fixed: dict[str, float] = {}
    for _ in range(len(weight) + 1):
        free = {g: w for g, w in weight.items() if g not in fixed}
        rest = 1.0 - sum(fixed.values())
        total = sum(free.values())
        shares = {g: rest * w / total for g, w in free.items()} if total else {}
        changed = False
        for g, share in shares.items():
            if g in max_share and share > max_share[g]:
                fixed[g], changed = max_share[g], True
            elif g in min_share and share < min_share[g]:
                fixed[g], changed = min_share[g], True
        if not changed:
            return {**fixed, **shares}
    return {**fixed, **shares}


def apply_sample_weights(rows: Sequence[dict], shares: dict[str, float], epoch_hours: float | None = None) -> dict:
    """row['sample_weight'] = expected draws per epoch, so each group gets its share of the epoch.
    Returns the plan {group: {pool_hours, share, epoch_hours, weight}}."""
    pool: dict[str, float] = defaultdict(float)
    for row in rows:
        pool[sampling_group(row)] += row["seconds"] / 3600
    epoch = epoch_hours or sum(pool.values())
    plan = {}
    for group, hours in pool.items():
        share = shares.get(group, 0.0)
        weight = share * epoch / hours if hours else 0.0
        plan[group] = {"pool_hours": round(hours, 2), "share": round(share, 4),
                       "epoch_hours": round(share * epoch, 2), "weight": round(weight, 4)}
    for row in rows:
        row["sample_weight"] = plan[sampling_group(row)]["weight"]
    return plan


# ---------------------------------------------------------------- build


def hours_table(rows: Iterable[dict]) -> dict:
    by_lang, by_source, by_cm = Counter(), Counter(), Counter()
    for row in rows:
        h = row["seconds"] / 3600
        by_lang[row["lang"]] += h
        by_source[f"{row['source']}/{row['lang']}"] += h
        by_cm["code_mixed" if row.get("code_mixed") else "monolingual"] += h
    rnd = lambda c: {k: round(v, 2) for k, v in sorted(c.items())}
    return {"total": round(sum(by_lang.values()), 2), "by_lang": rnd(by_lang), "by_source": rnd(by_source),
            "by_code_mixed": rnd(by_cm)}


def build(root: Path, mixture: dict, allow_review: set[str] = frozenset(), workers: int = 4,
          decode_workers: int | None = None, disk_budget_gb: float | None = None, keep_downloads: bool = False,
          log: Callable[[str], None] = print, runner: Callable | None = None) -> dict:
    """Run the mixture's tasks and return {"train", "dev", "eval": {name: rows}, "report"}."""
    from indicear.ingest import Context, DiskGuard, run_tasks, used_bytes

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    evals, trains, skipped = plan_tasks(mixture, allow_review)
    for message in skipped:
        log(f"skip {message}")
    guard = DiskGuard(root, int(disk_budget_gb * 2**30) if disk_budget_gb else None, used_bytes=used_bytes(root),
                      log=log)
    from indicear.ingest import context_options

    ctx = Context(root, guard, seed=int(mixture.get("seed", 0)), min_seconds=float(mixture.get("min_seconds", 0.5)),
                  max_seconds=float(mixture.get("max_seconds", 20.0)), keep_downloads=keep_downloads, log=log,
                  **context_options(mixture))
    if decode_workers:
        ctx.decode_workers = decode_workers
    runner = runner or run_tasks
    try:
        eval_results = runner(evals, ctx, workers)
        train_results = runner(trains, ctx, workers)
    finally:
        ctx.close()
    return assemble(mixture, eval_results, train_results, skipped, allow_review)


def assemble(mixture: dict, eval_results: Sequence[TaskResult], train_results: Sequence[TaskResult],
             skipped: Sequence[str] = (), allow_review: set[str] = frozenset()) -> dict:
    drops: dict[str, Counter] = defaultdict(Counter)
    notes: dict[str, Counter] = defaultdict(Counter)
    unavailable, shortfall = [], {}
    evals: dict[str, list[dict]] = defaultdict(list)
    for result in eval_results:
        if result.error:
            unavailable.append(f"eval {result.task.eval_name}: {result.error}")
            continue
        drops[result.task.source].update(result.drops)
        notes[result.task.source].update(result.notes)
        evals[result.task.eval_name].extend({**row, "eval_set": result.task.eval_name} for row in result.rows)
    pool: list[dict] = []
    for result in train_results:
        if result.error:
            unavailable.append(f"train {result.task.label}: {result.error}")
            continue
        drops[result.task.source].update(result.drops)
        notes[result.task.source].update(result.notes)
        pool.extend(result.rows)
        if result.shortfall_hours > 0.01:
            shortfall[result.task.label] = round(result.shortfall_hours, 2)
    eval_rows = [row for rows in evals.values() for row in rows]
    mixing = Counter()
    pool = dedupe(pool, mixing)
    pool = remove_eval_overlap(pool, eval_rows, mixing)
    dev_cfg = mixture.get("dev", {})
    train, dev = split_dev(pool, float(dev_cfg.get("fraction", 0.02)), float(dev_cfg.get("max_hours_per_lang", 2.0)),
                           float(dev_cfg.get("min_hours_per_lang", 0.0)), int(mixture.get("seed", 0)))
    assert_no_leak(train, eval_rows)
    assert_no_leak(dev, eval_rows)
    sampling = mixture.get("sampling", {})
    pool_hours = Counter()
    for row in train:
        pool_hours[sampling_group(row)] += row["seconds"] / 3600
    shares = temperature_shares(pool_hours, float(sampling.get("alpha", 0.5)), sampling.get("max_share"),
                                sampling.get("min_share"))
    plan = apply_sample_weights(train, shares, sampling.get("epoch_hours"))
    for row in dev:
        row["split"] = "dev"
    used_sources = sorted({r["source"] for r in train + dev + eval_rows})
    report = {
        "mixture": mixture["name"],
        "allow_review": sorted(allow_review),
        "train": hours_table(train), "dev": hours_table(dev),
        "eval": {name: hours_table(rows) for name, rows in sorted(evals.items())},
        "code_switched_hours": round(sum(r["seconds"] for r in train if r.get("code_mixed")) / 3600, 2),
        "code_switched_target_hours": mixture.get("code_switched_target_hours"),
        "drops": {source: dict(counter) for source, counter in sorted(drops.items())},
        "markup_notes": {source: dict(counter) for source, counter in sorted(notes.items())},
        "mixing_drops": dict(mixing),
        "sampling": {"alpha": sampling.get("alpha", 0.5), "max_share": sampling.get("max_share"),
                     "min_share": sampling.get("min_share"), "plan": plan},
        "shortfall_hours": shortfall, "skipped": list(skipped), "unavailable": unavailable,
        "leak_check": "passed: no eval speaker, speaker group or audio hash in train or dev",
        "licenses": [{"source": s, "license": SOURCES[s].license, "spdx": SOURCES[s].spdx,
                      "status": SOURCES[s].license_status, "attribution": SOURCES[s].attribution,
                      "revision": SOURCES[s].revision} for s in used_sources],
    }
    return {"train": train, "dev": dev, "eval": dict(evals), "report": report}


# Mean row durations (seconds) for the processed-hours estimate: the pilot's dev set by source and
# language (measured 24 October 2026), else by source; languages the pilot did not have use the
# source mean (an assumption until the full run's prep report gives the real values).
MEAN_SECONDS = {"indicvoices": 6.8, "kathbath": 6.1, "vaani": 5.0, "librispeech": 11.8, "peoples_speech": 13.4,
                "voxpopuli": 8.85, "mucs": 8.0, "spring_inx": 6.0, "shrutilipi": 8.0}
MEAN_SECONDS_BY_LANG = {("indicvoices", "hi"): 8.82, ("indicvoices", "kn"): 5.5, ("indicvoices", "ta"): 6.05,
                        ("kathbath", "hi"): 5.03, ("kathbath", "kn"): 7.44, ("kathbath", "ta"): 5.87,
                        ("vaani", "hi"): 8.44, ("vaani", "kn"): 4.01, ("vaani", "ta"): 3.57, ("vaani", "en"): 4.42}


def processed_hours(mixture: dict, train_hours: float, allow_review: set[str] = frozenset(),
                    negatives_share: float = 0.02, negative_seconds: float = 4.0,
                    mean_seconds: dict | None = None) -> dict:
    """Expected hours of audio each stream contributes to a live run of `train_hours`: the planner
    draws ROWS (stream weight = group share x hours x weight / the group's), so a stream's share of
    the audio is its row share x its mean row duration. Returns per-stream rows plus totals by
    language, sampling group and source. An estimate: the durations are means (MEAN_SECONDS)."""
    from indicear.stream import stream_specs

    specs, _, _ = stream_specs(mixture, allow_review)
    sampling = mixture.get("sampling", {})
    pool: dict[str, float] = defaultdict(float)
    draw: dict[str, float] = defaultdict(float)
    for spec in specs:
        pool[spec.group] += spec.hours
        draw[spec.group] += spec.draw_hours
    fixed = {"none": negatives_share} if negatives_share else {}
    shares = temperature_shares(pool, float(sampling.get("alpha", 0.5)), {**(sampling.get("max_share") or {}), **fixed},
                                {**(sampling.get("min_share") or {}), **fixed})
    means = {**MEAN_SECONDS_BY_LANG, **(mean_seconds or {})}
    rows = []
    for spec in specs:
        weight = shares.get(spec.group, 0.0) * spec.draw_hours / draw[spec.group] if draw[spec.group] else 0.0
        seconds = means.get((spec.source, spec.lang), MEAN_SECONDS.get(spec.source, 6.0))
        rows.append({"stream": spec.name, "source": spec.source, "lang": spec.lang, "group": spec.group,
                     "pool_hours": spec.hours, "row_share": weight, "mean_seconds": seconds})
    if negatives_share:
        rows.append({"stream": "negatives", "source": "negatives", "lang": "none", "group": "none", "pool_hours": 0.0,
                     "row_share": shares.get("none", negatives_share), "mean_seconds": negative_seconds})
    mass = sum(r["row_share"] * r["mean_seconds"] for r in rows)
    totals: dict[str, dict[str, float]] = {"lang": defaultdict(float), "group": defaultdict(float),
                                           "source": defaultdict(float)}
    for row in rows:
        row["hours"] = train_hours * row["row_share"] * row["mean_seconds"] / mass if mass else 0.0
        row["passes"] = row["hours"] / row["pool_hours"] if row["pool_hours"] else None
        totals["lang"][row["lang"]] += row["hours"]
        totals["group"][row["group"]] += row["hours"]
        totals["source"][row["source"]] += row["hours"]
    return {"train_hours": train_hours, "streams": rows,
            **{f"by_{k}": {kk: round(vv, 1) for kk, vv in sorted(v.items(), key=lambda x: -x[1])}
               for k, v in totals.items()}}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="offline download/disk estimate")
    plan.add_argument("--mixture", default="pilot")
    plan.add_argument("--allow-review", default="")
    plan.add_argument("--workers", type=int, default=4)
    hours = sub.add_parser("hours", help="expected processed hours per stream, language and source of a live run")
    hours.add_argument("--mixture", default="full_warm")
    hours.add_argument("--train-hours", type=float, default=8000.0)
    hours.add_argument("--allow-review", default="")
    args = parser.parse_args(argv)
    allow = {s for s in args.allow_review.split(",") if s}
    if args.command == "hours":
        result = processed_hours(load_mixture(args.mixture), args.train_hours, allow)
        print("| stream | pool h | row share | processed h | passes |\n| --- | ---: | ---: | ---: | ---: |")
        for row in result["streams"]:
            passes = f"{row['passes']:.2f}" if row["passes"] is not None else ""
            print(f"| {row['stream']} | {row['pool_hours']:.0f} | {row['row_share']:.3f} | {row['hours']:.0f} | {passes} |")
        print(json.dumps({k: v for k, v in result.items() if k != "streams"}, indent=1))
        return
    print(json.dumps(estimate(load_mixture(args.mixture), allow, args.workers), indent=2))


if __name__ == "__main__":
    main()
