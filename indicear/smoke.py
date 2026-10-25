"""CPU smoke mode: run the real entry points with no download and no GPU.

    python -m indicear.smoke ablate --root R prep --mixture R/smoke-mixture.json --stream ...
    python -m indicear.smoke baselines --root R --systems smoke-echo
    python -m indicear.smoke mixture R/smoke-mixture.json        # write the smoke mixture

(`indicear.ablate` / `indicear.baselines` are accepted for the module name too: scripts/env.sh's
`indicear` function passes the module name it would give `python -m`.)

`install(root)` replaces, in this process only:
- the EmbeddingGemma 2 audio tower with a tiny random 12-layer Gemma4 tower (the tests' tower:
  pinned attention geometry, width 64; seeded, so every process builds the same weights);
- the Gemma 4 feature extractor with one built from the pinned preprocessor_config.json values;
- the Hugging Face Hub (shard listing, shard download, ranged text reads) with synthetic parquet
  shards in the verified source schemas, written once under R/smoke-hub;
- FLEURS dev with a few synthetic clips per language;
- the baselines registry gains "smoke-echo", which returns each row's reference.

`write_mixture(path)` writes the small mixture the pilot smoke run uses: Kathbath Hindi and
LibriSpeech English streams, and eval sets named like the pilot's (Kathbath clean test-unknown,
IndicContextEval with code-mixed rows, LibriSpeech test). scripts/pilot.sh runs the whole chain
this way with SMOKE=1.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np

EG2_PREPROCESSOR = {  # preprocessor_config.json of google/embeddinggemma-2 @ 914f7f89
    "dither": 0.0, "feature_size": 128, "fft_length": 512, "fft_overdrive": False, "frame_length": 320,
    "hop_length": 160, "input_scale_factor": 1.0, "max_frequency": 8000.0, "mel_floor": 0.001,
    "min_frequency": 0.0, "padding_side": "right", "padding_value": 0.0, "per_bin_mean": None,
    "per_bin_stddev": None, "preemphasis": 0.0, "preemphasis_htk_flavor": True, "return_attention_mask": True,
    "sampling_rate": 16000,
}
TEXTS_HI = ["मैं घर जा रहा हूँ", "आज मौसम अच्छा है", "हम office जा रहे हैं", "यह किताब नई है", "मेरा order आ गया"]
TEXTS_EN = ["HELLO WORLD", "GOOD MORNING", "SEE YOU SOON", "THANK YOU", "THE BOOK IS NEW"]
CONTEXT_HI = ["मेरा वारंटी (Warranty) खत्म हो गया", "आज बाजार बंद है", "यह फ़ोन (phone) नया है", "हम घर पर हैं"]


def make_tiny_tower(layers: int = 12, hidden: int = 64):
    import torch
    from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel

    config = Gemma4AudioConfig(
        hidden_size=hidden, num_attention_heads=4, num_hidden_layers=layers, attention_chunk_size=12,
        attention_context_left=13, attention_context_right=0, conv_kernel_size=5,
        subsampling_conv_channels=[128, 8], output_proj_dims=96, use_clipped_linears=True,
        residual_weight=0.5, gradient_clipping=1e10,
    )
    torch.manual_seed(0)
    tower = Gemma4AudioModel(config).eval()
    with torch.no_grad():
        for name, parameter in tower.named_parameters():
            if "norm" not in name:
                parameter.normal_(0, 0.05)
    for parameter in tower.parameters():
        parameter.requires_grad_(False)
    return tower


def make_extractor(_path=None):
    from transformers.models.gemma4.feature_extraction_gemma4 import (
        Gemma4AudioFeatureExtractor,
    )

    return Gemma4AudioFeatureExtractor(**EG2_PREPROCESSOR)


def wav_bytes(seconds: float, rate: int = 16000, freq: float = 220.0, seed: int = 0) -> bytes:
    import soundfile as sf

    t = np.arange(int(seconds * rate)) / rate
    audio = (0.2 * np.sin(2 * np.pi * freq * t) + np.random.default_rng(seed).normal(0, 0.01, len(t)))
    buffer = io.BytesIO()
    sf.write(buffer, audio.astype(np.float32), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def write_parquet(path: Path, rows: list[dict], audio_column: str = "audio") -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    base = sum(path.name.encode()) * 7
    records = []
    for index, row in enumerate(rows):
        row = dict(row)
        seconds = row.pop("seconds")
        row[audio_column] = {"bytes": wav_bytes(seconds, freq=200 + 10 * index + base % 97, seed=base + index),
                             "path": row.pop("_path", f"utt{index}.wav")}
        records.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), path, row_group_size=4)
    return path


def kathbath_shard(path: Path, start: int, speakers: list[str], n: int = 12) -> Path:
    rows = [{"seconds": 1.0 + (i % 4) * 0.4, "fname": f"8444{start + i:010d}-{speakers[i % len(speakers)]}-f.m4a",
             "text": TEXTS_HI[i % len(TEXTS_HI)], "lang": "hindi", "duration": 1.0 + (i % 4) * 0.4,
             "gender": "female", "speaker_id": int(speakers[i % len(speakers)])} for i in range(n)]
    return write_parquet(path, rows, "audio_filepath")


def libri_shard(path: Path, start: int, n: int = 12, speaker_base: int = 10) -> Path:
    rows = [{"seconds": 1.2 + (i % 3) * 0.4, "file": "x", "text": TEXTS_EN[i % len(TEXTS_EN)],
             "speaker_id": speaker_base + (start + i) % 5, "chapter_id": 1,
             "id": f"{speaker_base + (start + i) % 5}-1-{start + i:04d}"} for i in range(n)]
    return write_parquet(path, rows)


def context_shard(path: Path, n: int = 8) -> Path:
    rows = [{"seconds": 1.5, "reference_text": CONTEXT_HI[i % len(CONTEXT_HI)], "speaker_id": f"ctx{i % 3}",
             "duration": 1.5, "code_mixing": "(" in CONTEXT_HI[i % len(CONTEXT_HI)], "domain_name": "retail",
             "speech_style": "read"} for i in range(n)]
    return write_parquet(path, rows)


def build_hub(directory: Path) -> dict[tuple[str, str, str], list[str]]:
    """Synthetic shards in the verified schemas, written once (the marker lists them)."""
    directory = Path(directory)
    marker = directory / "shards.json"
    if marker.exists():
        return {tuple(k.split("|")): v for k, v in json.loads(marker.read_text()).items()}
    shards = {
        ("kathbath", "hi", "train"): [kathbath_shard(directory / f"kb{i}.parquet", 100 * i, [str(100 + i), str(200 + i)])
                                      for i in range(3)],
        ("kathbath", "hi", "valid"): [kathbath_shard(directory / "kbv.parquet", 900, ["300", "301"], n=6)],
        ("librispeech", "en", "train.clean.360"): [libri_shard(directory / f"ls{i}.parquet", 100 * i) for i in range(2)],
        ("librispeech", "en", "test.clean"): [libri_shard(directory / "lst.parquet", 900, n=6, speaker_base=50)],
        ("indiccontext_eval", "hi", "test"): [context_shard(directory / "ctx.parquet")],
    }
    table = {"|".join(k): [str(p) for p in v] for k, v in shards.items()}
    marker.write_text(json.dumps(table, indent=1))
    return {k: [str(p) for p in v] for k, v in shards.items()}


SMOKE_MIXTURE = {
    "name": "smoke", "description": "CPU smoke run of the pilot chain (synthetic shards)", "seed": 0,
    "min_seconds": 0.5, "max_seconds": 20.0, "dev": {"fraction": 0.5, "max_hours_per_lang": 0.002},
    "sampling": {"alpha": 0.5, "max_share": {"hi": 0.7}},
    "train": [{"source": "kathbath", "lang": "hi", "split": "train", "hours": 1},
              {"source": "librispeech", "lang": "en", "split": "train.clean.360", "hours": 1}],
    "eval": [{"name": "kathbath_clean_test_unknown_hi", "source": "kathbath", "lang": "hi", "split": "valid"},
             {"name": "indiccontext_hi", "source": "indiccontext_eval", "lang": "hi", "split": "test"},
             {"name": "librispeech_test_clean", "source": "librispeech", "lang": "en", "split": "test.clean"}],
}


def write_mixture(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(SMOKE_MIXTURE, indent=1))
    return path


def install(root: Path, patch=setattr) -> None:
    """Patch this process: tiny tower, local extractor, synthetic hub, synthetic FLEURS, echo baseline.
    `patch(obj, name, value)` defaults to setattr; tests pass pytest's monkeypatch.setattr."""
    import transformers

    import indicear.data
    import indicear.encoder
    import indicear.ingest
    import indicear.stream
    from indicear import baselines

    hub = build_hub(Path(root) / "smoke-hub")
    tower = {}

    def load_tower(device="cpu", *_args, **_kwargs):
        if "tower" not in tower:
            tower["tower"] = make_tiny_tower()
        tower["tower"] = tower["tower"].to(device)  # INDICEAR_DEVICE=mps runs the smoke on MPS
        return tower["tower"]

    def lister(source, lang, split, _root):
        paths = hub.get((source.name, lang, split), [])
        return [{"repo": f"smoke/{source.name}", "revision": "0" * 40, "path": f"{lang}/{split}-{i:05d}.parquet",
                 "size": Path(p).stat().st_size, "_local": p} for i, p in enumerate(paths)]

    def download_hf(shard, _source, root_dir):
        target = Path(root_dir) / "downloads" / Path(shard["path"]).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shard["_local"], target)
        return target

    def hub_fetch(shard, directory):
        target = Path(directory) / shard["path"].replace("/", "_")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shard["_local"], target)
        return target

    def fetch_fleurs(root_dir, locale, split):
        if split == "test":
            raise ValueError("FLEURS test is never read")
        base = Path(root_dir) / "fleurs-smoke" / locale / split
        base.mkdir(parents=True, exist_ok=True)
        texts = TEXTS_EN if locale.startswith("en") else TEXTS_HI
        rows = []
        for i in range(3):
            path = base / f"{locale}-{i}.wav"
            if not path.exists():
                path.write_bytes(wav_bytes(1.4, freq=300 + 20 * i, seed=7 + i))
            rows.append({"sentence_id": str(i), "file": path.name, "audio": str(path), "raw": texts[i]})
        return rows

    patch(transformers.AutoFeatureExtractor, "from_pretrained", staticmethod(lambda *a, **k: make_extractor()))
    patch(indicear.encoder, "load_tower", load_tower)
    for module in (indicear.ingest, indicear.stream):
        patch(module, "list_hf_shards", lister)
    patch(indicear.ingest, "download_hf", download_hf)
    patch(indicear.stream, "hub_fetch", hub_fetch)
    patch(indicear.stream, "open_remote", lambda shard: open(shard["_local"], "rb"))  # noqa: SIM115
    patch(indicear.data, "fetch_fleurs", fetch_fleurs)
    echo = baselines.System(
        "smoke-echo", "smoke/echo", "0" * 40, frozenset({"hi", "en"}), "echo",
        lambda _args: (lambda audios, items: [i["text"] for i in items]), "returns the reference (smoke test)")
    patch(baselines, "SYSTEMS", {**baselines.SYSTEMS, "smoke-echo": echo})


def root_of(argv: list[str]) -> Path:
    return Path(argv[argv.index("--root") + 1]) if "--root" in argv else Path("/content/indicear")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        argv[0] = argv[0].removeprefix("indicear.")
    if not argv or argv[0] not in ("ablate", "baselines", "mixture"):
        raise SystemExit("usage: python -m indicear.smoke {ablate|baselines|mixture} ARGS...")
    command, rest = argv[0], argv[1:]
    if command == "mixture":
        print(write_mixture(Path(rest[0])))
        return
    install(root_of(rest))
    if command == "ablate":
        from indicear import ablate

        ablate.main(rest)
    else:
        from indicear import baselines

        baselines.main(rest)


if __name__ == "__main__":
    main()
