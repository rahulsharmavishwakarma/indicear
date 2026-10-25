"""Baselines on the same eval sets, with the same metrics and result files as `ablate eval`.

    python -m indicear.baselines --root R [--systems indicconformer-ctc,...] [--sets 'kathbath_*']

Each system writes results/<system>/<eval set>.json ("kind": "eval", scored by
`indicear.evaluate.score` on the rows in the languages the system supports) and the run writes
results/baselines.json: per system, what ran, what was skipped and why. Finished (system, set)
pairs are skipped on rerun. `ablate report` includes every system in results/eval_summary.md.

Systems (cards and code read on 2026-10-24; revisions pinned; weights are downloaded at run time
with `snapshot_download(revision=...)`, gated repos with HF_TOKEN from the environment):

| system | repo @ revision | languages | status |
| --- | --- | --- | --- |
| indicconformer-ctc / -rnnt | ai4bharat/indic-conformer-600m-multilingual @ 2a77d305 | 22 Indic incl. hi ta kn (no en) | runnable: repo code (`model_onnx.py`, onnxruntime), batch 1 |
| ep-hindi-asr-mixed / -native | epoch-discovery/ep-hindi-asr @ d7822345 | hi (mixed head: English in Latin) | card API used as documented; code not readable without the gate: unverified |
| dulu-malayalam | serendipitylabs/dulu-malayalam-asr @ 0d90991e | ml | runnable (repo code `dulu_asr.py`); runs only when an eval set has ml rows |
| kothagemma | nehsan/KothaGemma @ c0e92637 | bn | runnable (repo code `bangla_asr.py`, greedy, no KenLM); runs only with bn rows |
| sravaani | ARTPARK-IISc/SraVaani-1.0 @ f5dd5358 | 20 Indic + en, no ur (automatic language; no language input) | runnable: repo code (`modeling_sravaani.py`, TorchScript FastConformer + TDT greedy), card read 2026-10-25 |
| dulu-malayalam-ctc-778h | whiletruelearn/dulu-malayalam-ctc-778h @ d6a6b905 | ml | skipped: the card gives no inference path and no licence tag |

Each system runs the repo's own inference code, loaded from the pinned snapshot (remote code is
executed: review it before running). They need packages the training environment does not have
(onnxruntime, scipy) and Python <= 3.13 (no free-threaded onnxruntime wheels), so they run in
a separate CPython 3.13 environment built from the `baselines` dependency group (see scripts/baselines.sh).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from indicear import evaluate
from indicear.common import write_json


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


Transcribe = Callable[[Sequence[np.ndarray], Sequence[dict]], list[str]]  # (16 kHz audios, items) -> texts


@dataclass
class System:
    name: str
    repo: str
    revision: str
    langs: frozenset[str]
    decoder: str  # the decoder key in the result file
    load: Callable[[argparse.Namespace], Transcribe] | None  # None: not runnable
    why: str = ""  # why it is skipped, or caveats when it runs
    license: str = ""
    gated: bool = False
    extra: dict = field(default_factory=dict)


def snapshot(repo: str, revision: str, allow: Sequence[str] | None = None) -> Path:
    from huggingface_hub import snapshot_download

    from indicear.ingest import hf_token

    # A real directory, not the hub cache: onnxruntime >= 1.31 rejects external-data paths that
    # resolve (through the cache's symlinks) outside the model directory.
    root = Path(os.environ.get("INDICEAR_MODELS_DIR", Path.home() / ".cache" / "indicear-baselines"))
    local = root / f"{repo.replace('/', '__')}@{revision[:12]}"
    return Path(snapshot_download(repo, revision=revision, token=hf_token(), allow_patterns=allow, local_dir=local))


def import_file(path: Path, name: str):
    """Import a repo's own Python file (its directory goes on sys.path for sibling imports)."""
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def device_name(args) -> str:
    import torch

    return args.device or ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- adapters

INDICCONFORMER = ("ai4bharat/indic-conformer-600m-multilingual", "2a77d305038dfc502fd9464e7d028b90381976cb")
INDICCONFORMER_LANGS = frozenset(["as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"])


def indicconformer(decoding: str) -> Callable[[argparse.Namespace], Transcribe]:
    """The repo's `model_onnx.IndicASRModel`, built directly on the pinned snapshot (its
    `from_pretrained` downloads the head revision and ignores `revision`). Batch size 1:
    `model(wav[1, T] float32 16 kHz, lang, "ctc" | "rnnt") -> str`."""

    def load(_args) -> Transcribe:
        import torch

        path = snapshot(*INDICCONFORMER)
        module = import_file(path / "model_onnx.py", "indicconformer_model_onnx")
        model = module.IndicASRModel(module.IndicASRConfig(ts_folder=str(path)))

        def transcribe(audios, items):
            out = []
            with torch.inference_mode():
                for audio, item in zip(audios, items, strict=True):
                    text = model(torch.from_numpy(np.ascontiguousarray(audio))[None], item["lang"], decoding)
                    out.append(text[0] if isinstance(text, tuple) else str(text))
            return out

        return transcribe

    return load


EP_HINDI = ("epoch-discovery/ep-hindi-asr", "d78223456d1936ea3bec0feb00c36c1038d07475")


def ep_hindi(head: str) -> Callable[[argparse.Namespace], Transcribe]:
    """The card's Python API: `load(ckpt, device) -> (model, units, ck)`, `log_probs(model, wav,
    device, head=...)`, `greedy(lp, units)`; the mixed head writes English words in Latin
    (IndicEAR's convention), the native head writes Devanagari. Greedy, no KenLM."""

    def load(args) -> Transcribe:
        import torch

        path = snapshot(*EP_HINDI, allow=["*.py", "*.json", "*.safetensors", "*.txt", "*.md"])
        module = import_file(path / "transcribe.py", "ep_hindi_transcribe")
        device = torch.device(device_name(args))
        model, units, ck = module.load(str(path / "model.safetensors"), device)
        table = ck["units_mixed"] if head == "mixed" else units

        def transcribe(audios, _items):
            out = []
            with torch.inference_mode():
                for audio in audios:
                    lp = module.log_probs(model, np.asarray(audio, dtype=np.float32), device,
                                          **({"head": "mixed"} if head == "mixed" else {}))
                    out.append(module.greedy(lp, table))
            return out

        return transcribe

    return load


DULU = ("serendipitylabs/dulu-malayalam-asr", "0d90991e2d6878cda31ab01754c8ba8cb1a56d68")


def dulu(args) -> Transcribe:
    """`DuluASR.from_pretrained(dir, device=...)`, `.transcribe(array, sample_rate=16000) -> str`."""
    path = snapshot(*DULU)
    module = import_file(path / "dulu_asr.py", "dulu_asr")
    model = module.DuluASR.from_pretrained(str(path), device=device_name(args))
    return lambda audios, _items: [model.transcribe(a, sample_rate=16_000) for a in audios]


KOTHAGEMMA = ("nehsan/KothaGemma", "c0e92637ce8ee78bf831610482bf4cebe500f5dc")


def kothagemma(args) -> Transcribe:
    """`BanglaASR.from_pretrained(dir, variant="a").transcribe([arrays], use_lm=False)` (greedy)."""
    path = snapshot(*KOTHAGEMMA, allow=["*.py", "arm-a/*", "tokenizer/*", "*.txt", "*.md"])
    module = import_file(path / "bangla_asr.py", "kothagemma_bangla_asr")
    model = module.BanglaASR.from_pretrained(str(path), variant="a", device=device_name(args))
    return lambda audios, _items: list(model.transcribe(list(audios), use_lm=False))


SRAVAANI = ("ARTPARK-IISc/SraVaani-1.0", "f5dd5358325a5208775b91dad98918e079ea2b27")
SRAVAANI_LANGS = frozenset(["as", "bn", "brx", "doi", "en", "gu", "hi", "kn", "kok", "mai", "ml", "mni", "mr", "ne", "or",
                            "pa", "sa", "sat", "sd", "ta", "te"])


def sravaani(args) -> Transcribe:
    """`AutoModel.from_pretrained(dir, trust_remote_code=True).transcribe([arrays], batch_size=16)`: the repo's
    TorchScript FastConformer encoder with greedy TDT decoding. It takes no language (it picks it itself,
    like IndicEAR's automatic mode) and, per its card, does not support Urdu. The graphs are loaded on the
    CPU (upcast to fp32 by the repo code) and then moved; the float64 feature buffers are cast to fp32
    first, which MPS requires (about 6x faster than the CPU on an M5 Pro, same transcripts)."""
    from transformers import AutoModel

    path = snapshot(*SRAVAANI, allow=["*.ts", "*.pt", "*.model", "*.json", "*.py"])
    model = AutoModel.from_pretrained(str(path), trust_remote_code=True).eval()
    model._ensure_loaded()
    model._window, model._fb = model._window.float(), model._fb.float()
    model = model.to(device_name(args))
    return lambda audios, _items: list(model.transcribe([np.asarray(a, dtype=np.float32) for a in audios],
                                                        batch_size=16))


SYSTEMS: dict[str, System] = {s.name: s for s in (
    System("indicconformer-ctc", *INDICCONFORMER, INDICCONFORMER_LANGS, "ctc", indicconformer("ctc"),
           "ONNX via the repo's model_onnx.py; English unsupported", "MIT", gated=True),
    System("indicconformer-rnnt", *INDICCONFORMER, INDICCONFORMER_LANGS, "rnnt", indicconformer("rnnt"),
           "ONNX via the repo's model_onnx.py; English unsupported", "MIT", gated=True),
    System("ep-hindi-asr-mixed", *EP_HINDI, frozenset({"hi"}), "ctc", ep_hindi("mixed"),
           "card API; eg2_model.py/transcribe.py not readable without the gate (unverified)", "Apache-2.0",
           gated=True),
    System("ep-hindi-asr-native", *EP_HINDI, frozenset({"hi"}), "ctc", ep_hindi("native"),
           "card API; eg2_model.py/transcribe.py not readable without the gate (unverified)", "Apache-2.0",
           gated=True),
    System("dulu-malayalam", *DULU, frozenset({"ml"}), "ctc", dulu, "repo code dulu_asr.py", "Apache-2.0"),
    System("kothagemma", *KOTHAGEMMA, frozenset({"bn"}), "ctc", kothagemma,
           "repo code bangla_asr.py, greedy (no KenLM)", "CC-BY-SA-4.0 (weights), Apache-2.0 (code)"),
    System("sravaani", *SRAVAANI, SRAVAANI_LANGS, "tdt", sravaani,
           "repo code modeling_sravaani.py (TorchScript + greedy TDT); automatic language; no Urdu", "MIT",
           gated=True),
    System("dulu-malayalam-ctc-778h", "whiletruelearn/dulu-malayalam-ctc-778h",
           "d6a6b905c24696ec82f187ba89142f15355c6522", frozenset({"ml"}), "ctc", None,
           "the card has no inference instructions and the repo no licence tag; its inference.py imports the "
           "training scripts and a home-directory checkpoint path"),
)}
DEFAULT_SYSTEMS = ("indicconformer-ctc", "indicconformer-rnnt", "ep-hindi-asr-mixed", "ep-hindi-asr-native",
                   "dulu-malayalam", "kothagemma", "sravaani", "dulu-malayalam-ctc-778h")


# ---------------------------------------------------------------- runner


def run_system(system: System, sets: Sequence[evaluate.EvalSet], results: Path, args) -> dict:
    record = {"repo": system.repo, "revision": system.revision, "languages": sorted(system.langs),
              "license": system.license, "gated": system.gated, "notes": system.why, "sets": {}, "skipped": {}}
    if system.load is None:
        record["status"] = f"not runnable: {system.why}"
        return record
    work = []
    for eval_set in sets:
        items = [i for i in eval_set.items if i.get("lang") in system.langs or i.get("negative")]
        if not [i for i in items if not i.get("negative")]:
            record["skipped"][eval_set.name] = "no rows in the system's languages"
            continue
        subset = evaluate.EvalSet(eval_set.name, items, eval_set.condition, eval_set.notes, eval_set.transform)
        if evaluate.result_path(results, system.name, subset.name).exists():
            record["sets"][subset.name] = "done earlier"
            continue
        work.append(subset)
    if not work:
        record["status"] = "nothing to run" if not record["sets"] else "complete"
        return record
    try:
        transcribe = system.load(args)
    except Exception as error:  # noqa: BLE001 - recorded; the other systems still run
        record["status"] = f"load failed: {type(error).__name__}: {error}"
        return record
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for eval_set in work:
            started = time.time()
            hyps: list[str] = []
            for start in range(0, len(eval_set.items), args.batch):
                chunk = eval_set.items[start:start + args.batch]
                audios = evaluate.load_audios(chunk, pool, eval_set.transform, seed=17)
                hyps += transcribe(audios, chunk)
            scored = evaluate.score(eval_set.items, {system.decoder: hyps})
            evaluate.write_result(results, system.name, eval_set, scored, {"baseline": {
                "repo": system.repo, "revision": system.revision, "decoder": system.decoder, "notes": system.why,
                "languages_scored": sorted({i["lang"] for i in eval_set.items if not i.get("negative")})}})
            evaluate.write_rows(evaluate.result_path(results, system.name, eval_set.name).with_suffix(".rows.jsonl"),
                                eval_set.items, hyps)
            seconds = time.time() - started
            record["sets"][eval_set.name] = f"{len(eval_set.items)} rows in {seconds:.0f} s"
            log(f"{system.name} {eval_set.name}: {len(eval_set.items)} rows in {seconds:.0f} s")
    record["status"] = "ran"
    return record


def main(argv: Sequence[str] | None = None) -> None:
    from indicear.ablate import ROBUST_DEFAULT, Paths, noise_files

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="/content/indicear")
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--sets", default="*", help="comma-separated globs on eval set names")
    parser.add_argument("--max-per-lang", type=int, default=0)
    parser.add_argument("--robust-conditions", default=",".join(ROBUST_DEFAULT))
    parser.add_argument("--robust-per-lang", type=int, default=100)
    parser.add_argument("--noise-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8, help="audio loading threads")
    parser.add_argument("--batch", type=int, default=64, help="utterances loaded per transcribe call")
    args = parser.parse_args(argv)
    paths = Paths(args.root)
    sets = evaluate.eval_sets(paths.manifests, args.sets.split(","), args.max_per_lang,
                              [c for c in args.robust_conditions.split(",") if c], args.robust_per_lang,
                              noise_files(args.noise_dir))
    report_path = paths.results / "baselines.json"
    report = {}
    for name in [n for n in args.systems.split(",") if n]:
        if name not in SYSTEMS:
            raise SystemExit(f"unknown system {name!r}: {sorted(SYSTEMS)}")
        report[name] = run_system(SYSTEMS[name], sets, paths.results, args)
        log(f"{name}: {report[name]['status']}")
    write_json(report_path, report)
    evaluate.summarize(paths.results)


if __name__ == "__main__":
    sys.exit(main())
