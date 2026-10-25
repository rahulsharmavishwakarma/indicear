"""FLEURS manifests and the synthetic code-switch dev set for the ablation.

FLEURS (google/fleurs @ 70bb2e84, CC BY 4.0): train and dev only; test is never read here.
`fetch_fleurs` is probe.py's proven download/extract path. FLEURS has no speaker IDs, so the
official train/dev split is used as is.

Synthetic code-switch dev: pairs of FLEURS dev utterances from two languages, concatenated with a
short linear crossfade, references joined in each language's native script. This is
inter-sentential switching only; it does not test intra-sentential (word-level) switching.
"""

from __future__ import annotations

import csv
import tarfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from indicear.common import CODE_SWITCH_PAIRS, FLEURS, FLEURS_LOCALES, SAMPLE_RATE
from indicear.text import LANG_SCRIPT, normalize

MIN_SECONDS, MAX_SECONDS = 0.5, 20.0


def fetch_fleurs(root: Path, locale: str, split: str) -> list[dict]:
    """Download and extract one FLEURS split; return rows with audio path and raw text."""
    if split == "test":
        raise ValueError("FLEURS test is held out and never read by this code")
    from huggingface_hub import hf_hub_download

    base = root / "fleurs" / locale / split
    if not (base / "meta.tsv").exists():
        tsv = hf_hub_download(
            FLEURS[0], f"data/{locale}/{split}.tsv", repo_type="dataset", revision=FLEURS[1]
        )
        tgz = hf_hub_download(
            FLEURS[0], f"data/{locale}/audio/{split}.tar.gz", repo_type="dataset", revision=FLEURS[1]
        )
        base.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz) as tar:
            tar.extractall(base, filter="data")
        (base / "meta.tsv").write_text(Path(tsv).read_text(encoding="utf-8"), encoding="utf-8")
    wavs = {path.name: path for path in base.rglob("*.wav")}
    rows = []
    with (base / "meta.tsv").open(encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) < 3 or not row[1].endswith(".wav"):
                raise ValueError(f"unexpected FLEURS tsv row in {base}: {row[:3]}")
            if row[1] in wavs:
                rows.append({"sentence_id": row[0], "file": row[1], "audio": str(wavs[row[1]]), "raw": row[2]})
    return rows


def fleurs_items(root: Path, lang: str, split: str) -> list[dict]:
    import soundfile as sf

    items = []
    for row in fetch_fleurs(root, FLEURS_LOCALES[lang], split):
        text = normalize(row["raw"], lang)
        seconds = sf.info(row["audio"]).duration
        if not text or not MIN_SECONDS <= seconds <= MAX_SECONDS:
            continue
        items.append({
            "key": f"fleurs/{lang}/{split}/{Path(row['file']).stem}",
            "audio": row["audio"],
            "text": text,
            "raw": row["raw"],
            "lang": lang,
            "script": LANG_SCRIPT[lang],
            "seconds": seconds,
            "source": "fleurs",
            "split": split,
        })
    return items


def crossfade_concat(first: np.ndarray, second: np.ndarray, crossfade_ms: float = 50.0) -> np.ndarray:
    fade = min(int(SAMPLE_RATE * crossfade_ms / 1000), len(first), len(second))
    if fade == 0:
        return np.concatenate([first, second]).astype(np.float32)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    middle = first[-fade:] * (1 - ramp) + second[:fade] * ramp
    return np.concatenate([first[:-fade], middle, second[fade:]]).astype(np.float32)


def synth_code_switch(
    dev_by_lang: dict[str, list[dict]],
    out_dir: Path,
    pairs: Sequence[tuple[str, str]] = CODE_SWITCH_PAIRS,
    per_pair: int = 100,
    crossfade_ms: float = 50.0,
    seed: int = 0,
    max_seconds: float = 30.0,
) -> list[dict]:
    import soundfile as sf

    from indicear.encoder import load_audio

    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for lang_a, lang_b in pairs:
        pool_a, pool_b = dev_by_lang.get(lang_a, []), dev_by_lang.get(lang_b, [])
        if not pool_a or not pool_b:
            continue
        made = attempts = 0
        while made < per_pair and attempts < 20 * per_pair:
            attempts += 1
            a = pool_a[int(rng.integers(len(pool_a)))]
            b = pool_b[int(rng.integers(len(pool_b)))]
            if a["seconds"] + b["seconds"] > max_seconds:
                continue
            first, second = (a, b) if rng.random() < 0.5 else (b, a)
            audio = crossfade_concat(load_audio(first["audio"]), load_audio(second["audio"]), crossfade_ms)
            key = f"cs/{lang_a}-{lang_b}/{made:04d}"
            path = out_dir / f"{lang_a}-{lang_b}-{made:04d}.flac"
            sf.write(path, audio, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
            items.append({
                "key": key,
                "audio": str(path),
                "text": f"{first['text']} {second['text']}",
                "lang": f"{lang_a}+{lang_b}",
                "seconds": len(audio) / SAMPLE_RATE,
                "source": "synthetic-code-switch",
                "split": "dev",
                "spans": [
                    {"lang": first["lang"], "script": first["script"], "text": first["text"]},
                    {"lang": second["lang"], "script": second["script"], "text": second["text"]},
                ],
            })
            made += 1
    return items


def with_views(items: Sequence[dict], views: int) -> list[dict]:
    """Expand training items into `views` cached copies: view 0 clean, views >= 1 augmented."""
    out = []
    for item in items:
        for view in range(views):
            out.append({**item, "key": item["key"] if view == 0 else f"{item['key']}#v{view}", "view": view})
    return out
