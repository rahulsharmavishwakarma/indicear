"""Package a trained run as one self-contained model directory, and verify packages.

    python -m indicear.export --run RUN --tok TOK --out PKG
    python -m indicear.export --run RUN --tok TOK --out OUT --checkpoint best   # run without published heads
    python -m indicear.export --verify PKG
    python -m indicear.export --run RUN --tok TOK --out OUT --with-encoder --held-out RESULTS.md   # for the Hub

A run directory (as scripts/sync_runs.sh copies it) holds the published artifacts
(`heads.safetensors`, `adapters.safetensors` when the variant has adapters, `variant.json`,
`lid.json` for +lid variants, `result.json`) plus checkpoints and logs. The tokenizer is not in
the run directory: it lives in the training ROOT (`<ROOT>/tok/`, fetched once per ROOT with
scripts/fetch_tok.sh) and must be the one the run trained with.

The package (`FORMAT`):

    model.json             manifest: sha256 + size of every other file, encoder pin, variant,
                           decoder, tokenizer, LID, training provenance, the dev-set eval summary
    README.md              model card
    heads.safetensors      final head, self-conditioning heads, transducer, LID heads ("lid.")
    adapters.safetensors   residual adapters on encoder layers 7-12 (variants v2-v6)
    variant.json, lid.json the model configuration (lid.json only with LID heads)
    tokenizer/             aggregate.json + one SentencePiece model per script
    training/              result.json, config.json (trainer config, hash, git) and command.sh
                           (with a historical-record header) when the run has them
    LICENSE, NOTICE        Apache-2.0 and the attribution notice, from the IndicEAR distribution

    encoder/               optional (`--with-encoder`): the pinned EmbeddingGemma 2 audio tower
                           (config.json, model.safetensors, preprocessor_config.json; Apache-2.0):
                           every audio_tower tensor with values unchanged, keys without the
                           "audio_tower." prefix; nothing else from the checkpoint

No checkpoint or optimizer state is copied. Without `encoder/`, only the audio tower of the full
pinned EmbeddingGemma 2 checkpoint is used, downloaded at load time and never modified.

Export loads the heads and adapters into a model built from the variant and the tokenizer on a
stand-in encoder (same layer count and width, no weights), so a tokenizer from another ROOT or
an incompletely synced file fails here, not at transcription time. Published artifacts are also
checked against the run's LATEST.json sha256 when it lists them. `--checkpoint best|latest|PATH`
builds the heads and adapters from a checkpoint's model state instead (a run that has not
finished, or the best dev checkpoint rather than the final weights).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from indicear.common import (
    CACHE_LAYER,
    EG2,
    ENCODER_LAYERS,
    FRAME_RATE,
    LANGS,
    SAMPLE_RATE,
)
from indicear.runtime.checkpoint import CKPT_GLOB, sha256_file

FORMAT = "indicear-model/1"
ENCODER_DIR = "encoder"
HUB_METADATA = (".gitattributes",)
CARD_MEDIA = (".png", ".jpg", ".jpeg", ".mp3", ".wav", ".flac")  # images and audio samples a card may link to
MANIFEST = "model.json"
TOKENIZER_DIR = "tokenizer"
TRAINING_DIR = "training"
ENCODER_LICENSE = "Apache-2.0"
LICENSE_ID = "apache-2.0"
WINDOW_SECONDS = 30.0  # the Gemma 4 feature extractor's max_length (480 000 samples)
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".ckpt", ".bin")
LICENSE_FILES = ("LICENSE", "NOTICE")  # Apache-2.0 4(a)/(d): the licence and notices travel with the weights
HISTORICAL_HEADER = (
    "# Historical record: the command this run was launched with, kept unchanged below. The project was\n"
    "# then called Sangam (module `sangam`, now `indicear`) and the paths are the original training VM's.\n"
    "# To reproduce the run, use the commands in docs/training-recipe.md of the IndicEAR repository.\n"
)


# ---------------------------------------------------------------- model construction


class _StandInLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.feed_forward2 = nn.Identity()  # where AdapterSet hooks attach


class StandInTower(nn.Module):
    """Shape-only stand-in for the audio tower: lets a DesignA be built, loaded and saved with
    no encoder weights (export, verification)."""

    def __init__(self, hidden_size: int, layers: int = ENCODER_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList(_StandInLayer() for _ in range(layers))
        self.config = SimpleNamespace(num_hidden_layers=layers, hidden_size=hidden_size)


def _clean_key(key: str) -> str:
    return key.replace("._orig_mod.", ".").removeprefix("_orig_mod.")


def infer_hidden_size(tensors: dict[str, torch.Tensor]) -> int:
    """Encoder width from saved head/adapter tensors."""
    for key, value in tensors.items():
        if key.endswith("norm.weight") and value.dim() == 1 and not key.startswith("transducer."):
            return int(value.shape[0])
    for key in ("lid.utterance.proj.weight", "lid.frame.mlp.0.weight", "transducer.enc_proj.weight"):
        if key in tensors:
            return int(tensors[key].shape[1])
    raise ValueError("cannot infer the encoder width from the saved tensors")


def infer_num_units(tensors: dict[str, torch.Tensor]) -> int | None:
    for key, value in tensors.items():
        if key.startswith("inter.") and key.endswith("head.proj.weight"):
            return int(value.shape[0])
    return None


def build_design(staged, variant, lid_config, tokenizer, num_units: int | None = None):
    """A DesignA for `variant` (a VariantConfig) on `staged`, sized by the tokenizer."""
    from indicear import phonetic
    from indicear.model import DesignA

    ranges = [tokenizer.script_range(s) for s in tokenizer.order]
    return DesignA(staged, variant, tokenizer.vocab_size, ranges, num_units or len(phonetic.UNITS), lid=lid_config)


def load_design(staged, directory: str | Path, tokenizer):
    """The model saved in `directory` (variant.json, lid.json, heads, adapters) on `staged`.
    Loading is strict: a tokenizer or variant that does not match the tensors raises."""
    from safetensors.torch import load_file

    from indicear.model import load_lid_config, load_variant_config

    directory = Path(directory)
    variant = load_variant_config(directory / "variant.json")
    lid_config = load_lid_config(directory)
    heads = load_file(str(directory / "heads.safetensors"))
    model = build_design(staged, variant, lid_config, tokenizer, infer_num_units(heads))
    try:
        model.load(directory)
    except BaseException:
        model.detach_adapters()
        raise
    return model


def standin_staged(hidden_size: int):
    from indicear.encoder import StagedTower

    return StagedTower(StandInTower(hidden_size), CACHE_LAYER)


# ---------------------------------------------------------------- run metadata


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def checkpoint_path(run: Path, which: str) -> Path:
    if which == "best":
        path = run / "best.pt"
    elif which == "latest":
        checkpoints = sorted(run.glob(CKPT_GLOB), key=lambda p: int(p.stem.split("-")[1]))
        if not checkpoints:
            raise SystemExit(f"{run}: no {CKPT_GLOB} checkpoint")
        path = checkpoints[-1]
    else:
        path = Path(which)
    if not path.is_file():
        raise SystemExit(f"checkpoint {path} not found")
    return path


def load_checkpoint(path: Path) -> dict:
    # The run's own checkpoint (it holds numpy RNG state, so weights_only=False); mmap avoids
    # reading the optimizer state when only metadata is needed.
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def newest_checkpoint(run: Path) -> Path | None:
    candidates = sorted(run.glob(CKPT_GLOB), key=lambda p: int(p.stem.split("-")[1]))
    if candidates:
        return candidates[-1]
    return run / "best.pt" if (run / "best.pt").is_file() else None


def training_provenance(run: Path, result: dict | None, state: dict | None) -> dict:
    """Config hash, git revision and trainer config: from result.json, else a checkpoint."""
    train = (result or {}).get("train") or {}
    out = {"config_hash": train.get("config_hash") or (result or {}).get("config_hash"),
           "git": (result or {}).get("git"), "config": (result or {}).get("train_config")}
    if state is None and (out["git"] is None or out["config"] is None or out["config_hash"] is None):
        path = newest_checkpoint(run)
        if path is not None:
            try:
                state = load_checkpoint(path)
                out["checkpoint_read"] = path.name
            except Exception as error:  # noqa: BLE001 - provenance is best effort
                out["checkpoint_error"] = f"{path.name}: {type(error).__name__}: {error}"
    if state is not None:
        out["config_hash"] = out["config_hash"] or state.get("config_hash")
        out["git"] = out["git"] or state.get("git")
        out["config"] = out["config"] or state.get("config")
    return out


def mixture_info(result: dict | None) -> dict:
    """The training mixture's name and, when it is a packaged mixture, its sources and licences."""
    train = (result or {}).get("train") or {}
    name = train.get("mixture")
    if name is None and "FLEURS" in str((result or {}).get("notes") or ""):
        name = "fleurs"
    info: dict = {"name": name or "unknown"}
    if name == "fleurs":
        info.update(description="FLEURS train (12 design-A languages); FLEURS test never read.",
                    sources=[_source_row("fleurs")])
        return info
    path = Path(__file__).resolve().parent / "mixtures" / f"{name}.json"
    mixture = read_json(path) if name else None
    if mixture:
        info["description"] = mixture.get("description", "")
        hours: dict[str, float] = {}
        for entry in mixture.get("train", []):
            hours[entry["source"]] = hours.get(entry["source"], 0) + float(entry.get("hours", 0))
        info["sources"] = [{**_source_row(name), "planned_hours": round(h, 1)} for name, h in hours.items()]
        info["languages"] = sorted({e["lang"] for e in mixture.get("train", []) if e.get("lang") in LANGS})
    return info


def _source_row(name: str) -> dict:
    from indicear.sources import SOURCES

    source = SOURCES.get(name)
    if source is None:
        return {"source": name}
    return {"source": name, "title": source.title, "license": source.license, "spdx": source.spdx,
            "license_status": source.license_status}


def model_languages(result: dict | None, mixture: dict) -> list[str]:
    if mixture.get("languages"):
        return list(mixture["languages"])
    groups = set()
    for per_group in ((result or {}).get("metrics") or {}).values():
        groups |= {g for g in per_group if g in LANGS}
    return sorted(groups) or sorted(LANGS)


def allow_review_note(run: Path) -> str | None:
    command = run / "command.sh"
    if command.is_file() and "--allow-review" in command.read_text(encoding="utf-8", errors="replace"):
        return ("The run's command.sh passes --allow-review: a review-status source (e.g. MUCS, CC BY-SA 4.0) "
                "may be in the training data; see docs/data-licenses.md.")
    return None


# ---------------------------------------------------------------- export


def _check_against_latest(run: Path, names: list[str]) -> None:
    """Published artifacts must match the sha256 the run's LATEST.json lists for them."""
    latest = read_json(run / "LATEST.json")
    if not latest:
        return
    listed = {f["file"]: f["sha256"] for f in latest.get("files", [])}
    for name in names:
        if name in listed and sha256_file(run / name) != listed[name]:
            raise SystemExit(f"{run / name}: sha256 differs from LATEST.json (incomplete sync?); rerun sync_runs.sh")


def _stage_from_run(run: Path, staging: Path) -> list[str]:
    from indicear.model import load_variant_config

    names = ["heads.safetensors", "variant.json"]
    missing = [n for n in names if not (run / n).is_file()]
    if missing:
        raise SystemExit(f"{run}: no {', '.join(missing)} (the run has not published its model yet); "
                         "pass --checkpoint best or --checkpoint latest to export from a checkpoint")
    if load_variant_config(run / "variant.json").adapters:
        names.append("adapters.safetensors")
    if (run / "lid.json").is_file():
        names.append("lid.json")
    for name in names:
        if not (run / name).is_file():
            raise SystemExit(f"{run}: {name} is missing")
    _check_against_latest(run, names)
    for name in names:
        shutil.copyfile(run / name, staging / name)
    return names


def _stage_from_checkpoint(path: Path, staging: Path, tokenizer, state: dict) -> list[str]:
    from indicear.lid import lid_config_from_dict
    from indicear.model import VariantConfig

    extra = (state.get("config") or {}).get("extra") or {}
    if "variant" not in extra:
        raise SystemExit(f"{path}: the checkpoint config has no extra.variant")
    variant = dict(extra["variant"])
    variant["inter_layers"] = tuple(variant.get("inter_layers", (6, 9)))
    lid_config = lid_config_from_dict(extra["lid"]) if extra.get("lid") else None
    weights = {_clean_key(k): v for k, v in state["model"].items()}
    staged = standin_staged(infer_hidden_size(weights))
    model = build_design(staged, VariantConfig(**variant), lid_config, tokenizer, infer_num_units(weights))
    try:
        model.load_state_dict(weights, strict=True)
        model.save(staging)
    finally:
        model.detach_adapters()
    return [p.name for p in sorted(staging.iterdir()) if p.is_file()]


def _copy_tokenizer(tok: Path, target: Path):
    from indicear.tokenizer import AggregateTokenizer

    tokenizer = AggregateTokenizer.load(tok)
    target.mkdir(parents=True)
    shutil.copyfile(tok / "aggregate.json", target / "aggregate.json")
    for script in tokenizer.order:
        shutil.copyfile(tok / f"{script}.model", target / f"{script}.model")
    return tokenizer


def _eval_summary(result: dict | None) -> dict | None:
    if not result or not result.get("metrics"):
        return None
    return {"source": "result.json (dev-set report written at the end of training)", "notes": result.get("notes"),
            "metrics": result["metrics"], "lid": result.get("lid")}


def export(run: str | Path, tok: str | Path, out: str | Path, checkpoint: str | None = None,
           force: bool = False, with_encoder: bool = False, held_out: str | Path | None = None,
           model_name: str | None = None, card: str | Path | None = None, card_dirs: list | None = None) -> dict:
    """Write the package for `run` to `out`; returns the manifest. `with_encoder` adds the audio tower
    as `encoder/`; `held_out` is a Markdown file of held-out results for the generated model card;
    `model_name` names the model; `card` replaces the generated card's body with a Markdown file (the front
    matter is still generated), and the images and audio samples (`CARD_MEDIA`) of `card_dirs` that the card links to are copied next to it."""
    from indicear.model import load_lid_config, load_variant_config

    run, tok, out = Path(run).expanduser(), Path(tok).expanduser(), Path(out).expanduser()
    if not run.is_dir():
        raise SystemExit(f"run directory {run} not found")
    if not (tok / "aggregate.json").is_file():
        raise SystemExit(f"{tok} is not a tokenizer directory (no aggregate.json); fetch it with scripts/fetch_tok.sh")
    if out.exists() and any(out.iterdir()):
        if not force:
            raise SystemExit(f"{out} exists and is not empty (pass --force to replace a previous package)")
        if not (out / MANIFEST).is_file():
            raise SystemExit(f"refusing to replace {out}: it is not a model package (no {MANIFEST})")
    staging = out.parent / f".{out.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        tokenizer = _copy_tokenizer(tok, staging / TOKENIZER_DIR)
        state = None
        if checkpoint:
            ckpt = checkpoint_path(run, checkpoint)
            state = load_checkpoint(ckpt)
            _stage_from_checkpoint(ckpt, staging, tokenizer, state)
            weights_from = {"checkpoint": ckpt.name, "step": state.get("step"), "best_metric": state.get("best_metric")}
        else:
            _stage_from_run(run, staging)
            weights_from = {"published": True}
        # Strict structural check on a stand-in encoder: tokenizer, variant and tensors agree.
        from safetensors.torch import load_file

        heads = load_file(str(staging / "heads.safetensors"))
        hidden = infer_hidden_size(heads)
        try:
            model = load_design(standin_staged(hidden), staging, tokenizer)
        except RuntimeError as error:
            raise SystemExit(f"the weights do not match the tokenizer in {tok} (vocab {tokenizer.vocab_size}): use the "
                             f"tok/ of the ROOT this run trained in (scripts/fetch_tok.sh).\n{error}") from None
        model.detach_adapters()
        variant = load_variant_config(staging / "variant.json")
        lid_config = load_lid_config(staging)
        result = read_json(run / "result.json")
        provenance = training_provenance(run, result, state)
        training_dir = staging / TRAINING_DIR
        training_dir.mkdir()
        if result is not None:
            shutil.copyfile(run / "result.json", training_dir / "result.json")
        if provenance.get("config") is not None:
            (training_dir / "config.json").write_text(json.dumps(
                {"config": provenance["config"], "config_hash": provenance["config_hash"], "git": provenance["git"]},
                indent=2, default=str), encoding="utf-8")
        if (run / "command.sh").is_file():
            (training_dir / "command.sh").write_text(_historical_command(run / "command.sh"), encoding="utf-8")
        for name in LICENSE_FILES:
            shutil.copyfile(_license_file(name), staging / name)
        mixture = mixture_info(result)
        name = run.name
        variant_name = variant.name + ("+lid" if lid_config is not None and not lid_config.only else "")
        if model.final is None:
            decoder = None
        else:
            decoder = "rnnt" if variant.transducer else "ctc"
        manifest = {
            "format": FORMAT,
            "name": name,
            "variant": variant_name,
            "decoder": decoder,
            "decoders": [d for d in ("ctc" if model.final is not None else None,
                                     "rnnt" if model.transducer is not None else None) if d],
            "languages": model_languages(result, mixture),
            "encoder": {
                "repo": EG2[0], "revision": EG2[1], "component": "audio_tower", "frozen": True,
                "hidden_size": hidden, "layers": ENCODER_LAYERS, "cached_layers": CACHE_LAYER,
                "license": ENCODER_LICENSE,
                "note": "Only the audio tower of the pinned checkpoint is used; it is downloaded from the "
                        "Hugging Face Hub at load time and never modified. Not included in this package.",
            },
            "audio": {"sample_rate": SAMPLE_RATE, "frame_rate": FRAME_RATE, "window_seconds": WINDOW_SECONDS},
            "tokenizer": {"dir": TOKENIZER_DIR, "vocab_size": tokenizer.vocab_size, "order": list(tokenizer.order)},
            "lid": None if lid_config is None else {
                "languages": list(_lid_langs()), "utterance": lid_config.utterance, "frame": lid_config.frame,
                "only": lid_config.only},
            "weights": weights_from,
            "training": {
                "run": name,
                "config_hash": provenance.get("config_hash"),
                "git": provenance.get("git"),
                "mixture": mixture.get("name"),
                "steps": ((result or {}).get("train") or {}).get("steps") or (state or {}).get("step"),
                "audio_hours": ((result or {}).get("train") or {}).get("audio_hours")
                or ((result or {}).get("train") or {}).get("train_hours"),
                "environment": (result or {}).get("environment"),
            },
            "eval": _eval_summary(result),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if with_encoder:
            _stage_encoder(staging / ENCODER_DIR)
            manifest["encoder"]["bundled"] = ENCODER_DIR
            manifest["encoder"]["note"] = ("Only the audio tower of the pinned checkpoint is used, never modified. "
                                           f"{ENCODER_DIR}/ holds every audio_tower tensor of the checkpoint with its "
                                           "values unchanged and the 'audio_tower.' key prefix removed; the rest of "
                                           "the checkpoint is not included.")
        if model_name:
            manifest["name"] = model_name
        held_out_text = Path(held_out).expanduser().read_text(encoding="utf-8") if held_out else None
        if card:
            body = Path(card).expanduser().read_text(encoding="utf-8")
            if "{{" in body:
                raise SystemExit(f"{card} still has unfilled {{{{...}}}} placeholders")
            (staging / "README.md").write_text(front_matter(manifest) + body, encoding="utf-8")
            for directory in card_dirs or []:
                directory = Path(directory).expanduser()
                for image in sorted(directory.glob("*")):
                    if image.suffix.lower() in CARD_MEDIA and f"{directory.name}/{image.name}" in body:
                        (staging / directory.name).mkdir(exist_ok=True)
                        shutil.copy2(image, staging / directory.name / image.name)
        else:
            (staging / "README.md").write_text(model_card(manifest, mixture, allow_review_note(run), held_out_text),
                                               encoding="utf-8")
        manifest["files"] = {
            path.relative_to(staging).as_posix(): {"sha256": sha256_file(path), "size": path.stat().st_size}
            for path in sorted(staging.rglob("*")) if path.is_file()
        }
        (staging / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
                                        encoding="utf-8")
        problems = verify(staging)
        if problems:
            raise SystemExit("package failed verification:\n" + "\n".join(problems))
        if out.exists():
            shutil.rmtree(out)
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _historical_command(path: Path) -> str:
    """A run's command.sh with a header: it records the run as it was launched, under the project's
    former name (Sangam) and the original VM's paths, so it is not meant to be rerun as is."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    shebang = lines.pop(0) if lines and lines[0].startswith("#!") else ""
    return shebang + HISTORICAL_HEADER + "".join(lines)


def _license_file(name: str) -> Path:
    """LICENSE or NOTICE: from the source checkout, or from the installed wheel's metadata."""
    source = Path(__file__).resolve().parents[1] / name
    if source.is_file():
        return source
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        for file in distribution("indicear").files or []:
            if file.name == name and "licenses" in file.parts:
                return Path(file.locate())
    except PackageNotFoundError:
        pass
    raise SystemExit(f"{name} not found next to the indicear sources or in its installed metadata")


def _lid_langs() -> tuple[str, ...]:
    from indicear.lid import LID_LANGS

    return LID_LANGS


# ---------------------------------------------------------------- model card


LANG_NAMES = {"hi": "Hindi", "mr": "Marathi", "bn": "Bengali", "pa": "Punjabi", "gu": "Gujarati", "or": "Odia",
              "ur": "Urdu", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "en": "English"}


def _metrics_table(evaluation: dict | None, decoder: str | None) -> str:
    if not evaluation:
        return "No evaluation summary: the run directory had no result.json.\n"
    metrics = evaluation["metrics"]
    key = decoder if decoder in metrics else next(iter(metrics))
    rows = metrics[key]
    order = sorted(rows, key=lambda g: (g != "all", g.startswith("cs:"), g == "negatives", g))
    lines = [f"Decoder `{key}`, from {evaluation['source']}. {evaluation.get('notes') or ''}".rstrip(), "",
             "| Group | CER | WER | Utterances |", "| --- | --- | --- | --- |"]
    for group in order:
        row = rows[group]
        if group == "negatives":
            lines.append(f"| non-speech: false-output rate {row.get('false_output_rate')}% | | | {row.get('clips', '')} |")
            continue
        lines.append(f"| {group} | {row.get('cer')} | {row.get('wer')} | {row.get('utterances', '')} |")
    lid = (evaluation.get("lid") or {}).get("utterance")
    if lid and lid.get("accuracy") is not None:
        lines += ["", (f"Utterance language ID: accuracy {lid['accuracy']}%, macro-F1 {lid.get('macro_f1')} "
                       f"({lid.get('utterances')} utterances).")]
    return "\n".join(lines) + "\n"


HUB_DATASETS = ("ai4bharat/IndicVoices", "ai4bharat/Kathbath", "ARTPARK-IISc/Vaani-transcription-part",
                "openslr/librispeech_asr", "MLCommons/peoples_speech", "facebook/voxpopuli")


def front_matter(manifest: dict) -> str:
    """Hugging Face model-card metadata (YAML front matter)."""
    return "\n".join([
        "---", f"license: {LICENSE_ID}", "language:", *[f"- {code}" for code in manifest["languages"]],
        "pipeline_tag: automatic-speech-recognition", f"base_model: {manifest['encoder']['repo']}",
        "datasets:", *[f"- {d}" for d in HUB_DATASETS],
        "tags:", *[f"- {tag}" for tag in ("asr", "ctc", "indic", "code-switching", "language-identification",
                                          "frozen-encoder", "adapters", "vesper")],
        "---", ""])


def model_card(manifest: dict, mixture: dict, review_note: str | None, held_out: str | None = None) -> str:
    training = manifest["training"]
    languages = ", ".join(f"{LANG_NAMES.get(code, code)} (`{code}`)" for code in manifest["languages"])
    decoder = {"ctc": "CTC, greedy (or prefix beam search with `--beam N`)",
               "rnnt": "factored RNN-T, greedy (the CTC branch is also available)",
               None: "none (language ID only)"}[manifest["decoder"]]
    sources = mixture.get("sources") or []
    source_lines = "\n".join(
        f"- {s.get('title', s['source'])}: {s.get('license', 'licence: see docs/data-licenses.md')}"
        + (f", {s['license_status']}" if s.get("license_status") else "")
        + (" (evaluation only unless the run passed --allow-review)" if s.get("license_status") == "review" else "")
        + (f", {s['planned_hours']} h planned" if s.get("planned_hours") else "")
        for s in sources) or "- see docs/data-licenses.md"
    lid = manifest["lid"]
    lid_text = ("Utterance and code-switch span language ID over "
                + ", ".join(lid["languages"]) + " (outputs only: never used to route or gate the transcript)."
                if lid else "No language-ID heads.")
    title = manifest["name"] if manifest["name"] == manifest["variant"] else f"{manifest['name']} ({manifest['variant']})"
    encoder = manifest["encoder"]
    bundled = encoder.get("bundled")
    front = front_matter(manifest)
    loading = (f"The audio tower is included (`{bundled}/`, every tensor byte-identical to the pinned checkpoint), "
               "so nothing else is downloaded." if bundled else
               "The first run downloads the pinned EmbeddingGemma 2 checkpoint from the Hugging Face Hub and keeps "
               "only its audio tower.")
    encoder_licence = (f"`{encoder['repo']}` audio tower weights, {encoder['license']}, included unchanged in "
                       f"`{bundled}/` (the text and vision weights are not)." if bundled else
                       f"`{encoder['repo']}` weights under {encoder['license']} (downloaded from the Hub, not "
                       "redistributed here).")
    description = (mixture.get("description") or "").replace(" (docs/PLAN.md)", "")
    return front + f"""# IndicEAR: {title}

Speech recognition for 11 Indian languages and English, including code-switched speech, built on the
**frozen** EmbeddingGemma 2 audio encoder (`{encoder['repo']}@{encoder['revision']}`, audio tower only).
Layers 1-6 run unchanged; layers 7-12 run with small residual adapters, followed by the output heads in
this package. About 45 MB of weights are trained; the encoder is never modified.

**Research release, not a production model.** It shows that a frozen embedding model's audio encoder
can be turned into a multilingual recogniser with built-in language ID; dedicated ASR models such as
IndicConformer and SraVaani have several times lower error rates (see "Held-out evaluation").

- **Variant:** `{manifest['variant']}` (docs/framework.md, "Design A as implemented")
- **Decoder:** {decoder}
- **Output:** normalized text: lowercase, no punctuation, each word in its own script (native script for
  Indic words, Latin for English)
- **Languages:** {languages}
- **Language ID:** {lid_text}

## Use

```bash
pip install indicear
indicear-transcribe --model <this repo id or a local copy> audio.wav [more.flac ...]
indicear-transcribe --model <repo id> --json audio.wav      # words, timings, language, code-switch spans
indicear-export --verify <local copy>                       # check every file's sha256
```

```python
from indicear.transcribe import Transcriber

with Transcriber("<repo id>") as model:
    result = model.transcribe_file("audio.wav")
    print(result["text"], result["language"]["top"])
```

{loading} Audio is decoded to 16 kHz mono and transcribed in {manifest['audio']['window_seconds']:.0f} s
windows with a short overlap. Device `auto` picks CUDA, then Apple MPS, then the CPU.

## Held-out evaluation

{held_out.strip() if held_out else "Not included in this card; see the repository's docs/full-run.md."}

## Training

Full recipe, hyperparameters, cost and the log of mistakes: `docs/training-recipe.md` in the code
repository.

- Mixture: `{mixture.get('name')}`. {description}
- Run `{training['run']}`: {training.get('steps')} steps, {training.get('audio_hours')} h of audio, config hash
  `{training.get('config_hash')}`.

Training data sources:

{source_lines}

## Dev-set metrics (end of training)

{_metrics_table(manifest['eval'], manifest['decoder'])}
## Known limits

- **Accuracy.** Several times the error rate of dedicated Indic ASR models on every held-out family
  except MUCS. Not suitable for production transcription.
- **Urdu.** The utterance language ID labels most Kathbath and Vaani Urdu speech as Hindi, so it comes
  out in Devanagari; passing the language (`lang_hint="ur"` in `Transcriber.transcribe`) fixes much of it.
- **Hindi and Urdu script.** Spoken Hindi and Urdu overlap heavily. The model picks Devanagari or
  Perso-Arabic script from the audio and its training data, and the language-ID labels separate the
  two by script only, so the same speech can come out in either script.
- **Reverberation.** The frozen encoder is weak under reverberation: at RT60 0.9 s the error rate
  roughly doubles. Far-field and speakerphone audio degrade sharply.
- English written in native script in the training transcripts, and words the model never saw, may
  come out in the matrix language's script.
- Word timings come from emission frames (40 ms): approximate word starts, not forced alignment. No
  language model, punctuation or casing.
- Audio longer than {manifest['audio']['window_seconds']:.0f} s is split into overlapping windows; utterance language
  probabilities of long audio are the duration-weighted mean over windows.
- Numbers are for the data and conditions above only; they do not carry over to other domains.

## Licence

- This package's adapters, heads and tokenizer: {LICENSE_ID}.
- Encoder: {encoder_licence}
- Training data: per source, as listed above and in docs/data-licenses.md (CC BY / CC0 sources for
  training; share-alike and unconfirmed sources are evaluation-only).
{('- ' + review_note) if review_note else ''}
"""


# ---------------------------------------------------------------- verification


def verify(package: str | Path) -> list[str]:
    """Problems with a package (empty when every listed file is present with its sha256 and size,
    nothing else is present apart from Hugging Face Hub metadata, and no checkpoint or optimizer state
    is included)."""
    package = Path(package).expanduser()
    manifest = read_json(package / MANIFEST)
    if manifest is None:
        return [f"{package / MANIFEST}: missing or not JSON"]
    problems = []
    if manifest.get("format") != FORMAT:
        problems.append(f"format {manifest.get('format')!r} is not {FORMAT!r}")
    files = manifest.get("files") or {}
    if not files:
        problems.append("the manifest lists no files")
    for name, entry in sorted(files.items()):
        path = package / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        if path.stat().st_size != entry.get("size"):
            problems.append(f"{name}: size {path.stat().st_size} != {entry.get('size')}")
        elif sha256_file(path) != entry.get("sha256"):
            problems.append(f"{name}: sha256 mismatch")
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(package).as_posix()
        if name in HUB_METADATA or name.startswith(".cache/"):  # added by the Hub or hf download --local-dir
            continue
        if name != MANIFEST and name not in files:
            problems.append(f"{name}: not listed in the manifest")
        if path.suffix in FORBIDDEN_SUFFIXES:
            problems.append(f"{name}: checkpoint or optimizer state must not be in a package")
    return problems


def _stage_encoder(target: Path) -> None:
    """Save the pinned EG2 audio tower and its feature-extractor config to `target`, in the
    checkpoint's own dtype, and check every tensor byte-identical to the checkpoint's audio_tower."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoFeatureExtractor, AutoModel

    model = AutoModel.from_pretrained(EG2[0], revision=EG2[1], dtype=torch.bfloat16)
    model.audio_tower.save_pretrained(target, safe_serialization=True)
    del model
    AutoFeatureExtractor.from_pretrained(EG2[0], revision=EG2[1]).save_pretrained(target)
    original = hf_hub_download(EG2[0], "model.safetensors", revision=EG2[1])
    with safe_open(original, "pt") as a, safe_open(str(target / "model.safetensors"), "pt") as b:
        keys = set(b.keys())
        expected = {k.removeprefix("audio_tower.") for k in a.keys() if k.startswith("audio_tower.")}
        if keys != expected:
            raise SystemExit(f"extracted tower keys differ from the checkpoint's audio_tower: {sorted(keys ^ expected)[:5]}")
        for key in keys:
            if not torch.equal(a.get_tensor("audio_tower." + key), b.get_tensor(key)):
                raise SystemExit(f"extracted tensor {key} differs from the checkpoint")


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, help="run directory (as scripts/sync_runs.sh copied it)")
    parser.add_argument("--tok", type=Path, help="the training ROOT's tok/ directory (scripts/fetch_tok.sh)")
    parser.add_argument("--out", type=Path, help="package directory to write")
    parser.add_argument("--checkpoint", help="best | latest | PATH: take the weights from a checkpoint")
    parser.add_argument("--force", action="store_true", help="replace an existing package at --out")
    parser.add_argument("--verify", type=Path, metavar="PACKAGE", help="verify a package's file hashes and exit")
    parser.add_argument("--with-encoder", action="store_true",
                        help="include the pinned audio tower as encoder/ (about 582 MiB, tensor values unchanged)")
    parser.add_argument("--held-out", type=Path, metavar="MD", help="Markdown of held-out results for the model card")
    parser.add_argument("--name", help="model name in the manifest (default: the variant)")
    parser.add_argument("--card", type=Path, metavar="MD", help="Markdown body of the model card (front matter is generated)")
    parser.add_argument("--card-dir", type=Path, action="append", default=[], metavar="DIR",
                        help="directory of images or audio samples the card links to, copied as <package>/<DIR name>/ (repeatable)")
    args = parser.parse_args(argv)
    if args.verify:
        problems = verify(args.verify)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            raise SystemExit(f"{args.verify}: {len(problems)} problem(s)")
        files = read_json(args.verify / MANIFEST)["files"]
        print(f"{args.verify}: OK ({len(files)} files verified)")
        return
    if not (args.run and args.tok and args.out):
        parser.error("--run, --tok and --out are required (or --verify PACKAGE)")
    manifest = export(args.run, args.tok, args.out, args.checkpoint, args.force, args.with_encoder, args.held_out,
                      args.name, args.card, args.card_dir)
    size = sum(f["size"] for f in manifest["files"].values())
    print(f"{args.out}: {manifest['variant']} ({manifest['decoder'] or 'LID only'}), {len(manifest['files'])} files, "
          f"{size / 2**20:.1f} MiB; weights {json.dumps(manifest['weights'])}")


if __name__ == "__main__":
    main()
