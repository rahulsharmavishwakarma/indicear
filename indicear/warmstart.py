"""Warm start: initialize a new run from a finished run (or one checkpoint) of the same variant.

    python -m indicear.ablate --root R train --lower live --variants v4+lid --init-from RUN_OR_CKPT ...

Unlike a resume, a warm start begins a NEW run (its own config hash, step 0, fresh AdamW and
scheduler state). It copies the source's trainable weights into the new model:

- adapters (layers 7-12), the self-conditioned phonetic CTC heads (82 shared units), the final
  LayerNorm, the LID heads and, for v6, the prediction network and joint: shapes must match;
- the vocabulary-indexed rows of the output layers, mapped by script. The new run's tokenizer
  (`AggregateTokenizer.extend`) keeps the source's SentencePiece models byte for byte for the
  `kept` scripts (pilot: deva, latn, taml, knda) and trains new ones for the others, in the same
  script order. A kept script's rows move from its old id range to its new range (same size,
  same pieces); the specials (blank, pad, bos, eos) keep ids 0-3. A new script's token rows keep
  the new model's random initialization ("fresh").
- the two-level output's script classes (blank + one per script, same order): blank and the kept
  scripts are copied. A new script's class row is initialized fresh as zero weights with bias
  `new_script_bias` (default -2, about where the pilot's own unseen scripts settled, -1.8). A very
  negative bias such as -40 makes the kept scripts' outputs bit-identical at step 0 but does not
  train: Adam moves a bias by about the learning rate per step, so -40 needs ~10^5 steps and the
  new languages' CTC loss stays near 30 (measured in the first full run, 24 Oct). `new_script_logits="copy"` copies the source's rows instead (the pilot
  learned to suppress scripts it never saw), which is bit-identical by construction.

The rows and parameters that start fresh are returned as `WarmStart.new_rows` / `fresh_params`,
for the trainer's higher learning rate on new rows (`trainer.LrMultipliers`).

`init_from.json` in the run directory records the source and the row map; a resumed warm-started
run (VM lost) reads its weights from its own checkpoint and only needs that file, not the source.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

INIT_FILE = "init_from.json"
SCRIPT_LOGIT_MODES = ("fresh", "copy")
# Output tensors indexed by token id (dim 0 = vocabulary) and by script class (dim 0 = 1 + scripts).
VOCAB_KEYS = ("final.token_head.weight", "final.token_head.bias", "final.proj.weight", "final.proj.bias",
              "transducer.embed.weight", "transducer.output.token_head.weight", "transducer.output.token_head.bias")
SCRIPT_KEYS = ("final.script_head.weight", "final.script_head.bias", "transducer.output.script_head.weight",
               "transducer.output.script_head.bias")
# Variant fields that change the parameter structure (the loss weights may differ from the source).
STRUCTURE = ("adapters", "phonetic", "factored", "cr_ctc", "transducer", "adapter_rank", "inter_layers")


@dataclass
class Source:
    """The weights and metadata of the run a warm start begins from."""

    path: Path
    tensors: dict[str, torch.Tensor]  # in the model's state_dict key space
    tok: Path
    variant: dict
    lid: dict | None
    config_hash: str | None
    weights_sha256: str


@dataclass
class WarmStart:
    summary: dict  # hashed into the new run's config (TrainConfig.extra["init_from"])
    new_rows: dict[str, list[list[int]]] = field(default_factory=dict)  # parameter -> [[start, end), ...]
    fresh_params: list[str] = field(default_factory=list)  # parameters with no source tensor
    report: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"summary": self.summary, "new_rows": self.new_rows, "fresh_params": self.fresh_params,
                "report": self.report}

    @classmethod
    def from_json(cls, data: dict) -> WarmStart:
        return cls(data["summary"], {k: [list(r) for r in v] for k, v in data["new_rows"].items()},
                   list(data["fresh_params"]), data.get("report", {}))


def _sha256_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def load_source(path: str | Path, tok: str | Path | None = None) -> Source:
    """A finished run directory (heads.safetensors, adapters.safetensors, variant.json, lid.json,
    tok/) or one checkpoint file (ckpt-*.pt / best.pt: its model state and config). `tok`
    overrides the tokenizer directory (default: <run>/tok, or the checkpoint's directory's tok/)."""
    from safetensors.torch import load_file

    path = Path(path).expanduser().resolve()
    if path.is_dir():
        heads = path / "heads.safetensors"
        if not heads.exists():
            raise SystemExit(f"--init-from {path}: no heads.safetensors (pass a finished run or a checkpoint file)")
        files = [heads]
        tensors = dict(load_file(str(heads)))
        adapters = path / "adapters.safetensors"
        if adapters.exists():
            files.append(adapters)
            tensors.update({f"adapters.{k}": v for k, v in load_file(str(adapters)).items()})
        variant = json.loads((path / "variant.json").read_text(encoding="utf-8"))
        lid = json.loads((path / "lid.json").read_text(encoding="utf-8")) if (path / "lid.json").exists() else None
        config_hash = None
        result = path / "result.json"
        if result.exists():
            config_hash = json.loads(result.read_text(encoding="utf-8")).get("train", {}).get("config_hash")
        tok_dir = Path(tok) if tok else path / "tok"
        sha = _sha256_files(files)
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
        tensors = dict(state["model"])
        extra = state.get("config", {}).get("extra", {})
        variant, lid = extra.get("variant"), extra.get("lid")
        if variant is None:
            raise SystemExit(f"--init-from {path}: the checkpoint has no variant config")
        config_hash = state.get("config_hash")
        tok_dir = Path(tok) if tok else path.parent / "tok"
        sha = _sha256_files([path])
    if not (tok_dir / "aggregate.json").exists():
        raise SystemExit(f"--init-from: tokenizer {tok_dir} not found (pass --init-tok DIR)")
    return Source(path, {k: v.float() if v.is_floating_point() else v for k, v in tensors.items()},
                  tok_dir, variant, lid, config_hash, sha)


def check_variant(source: Source, config) -> None:
    from dataclasses import asdict

    target = asdict(config)
    for key in STRUCTURE:
        want, have = target[key], source.variant.get(key)
        if key == "inter_layers":
            want, have = list(want), list(have or [])
        if want != have:
            raise SystemExit(f"--init-from: variant field {key} differs (source {have}, new run {want}); "
                             "a warm start needs the same model structure")


def row_map(old_tok, new_tok, keep: Sequence[str]) -> dict:
    """{"tokens": [(old_start, old_end, new_start, new_end) per kept script + the specials],
    "new_scripts": [scripts], "classes": [(old_class, new_class)]} with exact piece checks."""
    from indicear.tokenizer import SPECIALS

    if tuple(old_tok.order) != tuple(new_tok.order):
        raise SystemExit(f"--init-from: script order differs (source {old_tok.order}, new {new_tok.order}); "
                         "build the new tokenizer with AggregateTokenizer.extend (prep --tok-from)")
    spans = [(0, len(SPECIALS), 0, len(SPECIALS))]
    classes = [(0, 0)]
    for index, script in enumerate(new_tok.order):
        if script not in keep:
            continue
        old_model, new_model = old_tok.models[script], new_tok.models[script]
        size = old_model.get_piece_size()
        if new_model.get_piece_size() != size or any(
                old_model.id_to_piece(i) != new_model.id_to_piece(i) for i in range(size)):
            raise SystemExit(f"--init-from: the new tokenizer's {script} model differs from the source's; "
                             "kept scripts must reuse the source's SentencePiece models unchanged")
        (a, b), (c, d) = old_tok.script_range(script), new_tok.script_range(script)
        spans.append((a, b, c, d))
        classes.append((index + 1, index + 1))
    return {"tokens": spans, "new_scripts": [s for s in new_tok.order if s not in keep], "classes": classes}


@torch.no_grad()
def apply(model, source: Source, old_tok, new_tok, keep: Sequence[str], new_script_logits: str = "fresh",
          new_script_bias: float = -2.0, strict: bool = True) -> WarmStart:
    """Copy `source` into `model` (a freshly built DesignA on `new_tok`); see the module docstring."""
    if new_script_logits not in SCRIPT_LOGIT_MODES:
        raise ValueError(f"new_script_logits must be one of {SCRIPT_LOGIT_MODES}")
    check_variant(source, model.config)
    mapping = row_map(old_tok, new_tok, keep)
    new_classes = [new_tok.order.index(s) + 1 for s in mapping["new_scripts"]]
    target = model.state_dict()
    params = dict(model.named_parameters())
    new_rows: dict[str, list[list[int]]] = {}
    copied, mapped, fresh, unused = [], [], [], sorted(set(source.tensors) - set(target))
    for key, tensor in target.items():
        if key not in source.tensors:
            fresh.append(key)
            continue
        value = source.tensors[key]
        if key in VOCAB_KEYS:
            if value.shape[0] != old_tok.vocab_size or tensor.shape[0] != new_tok.vocab_size:
                raise SystemExit(f"--init-from: {key} has {value.shape[0]} rows, the source tokenizer {old_tok.vocab_size}")
            for a, b, c, d in mapping["tokens"]:
                tensor[c:d] = value[a:b].to(tensor)
            rows = sorted(_complement([(c, d) for _, _, c, d in mapping["tokens"]], new_tok.vocab_size))
            if rows and key in params:
                new_rows[key] = [list(r) for r in rows]
            mapped.append(key)
        elif key in SCRIPT_KEYS:
            if value.shape != tensor.shape:
                raise SystemExit(f"--init-from: {key} has shape {tuple(value.shape)}, the new model {tuple(tensor.shape)}")
            for old, new in mapping["classes"]:
                tensor[new] = value[old].to(tensor)
            for cls in new_classes:
                if new_script_logits == "copy":
                    tensor[cls] = value[cls].to(tensor)
                elif key.endswith("bias"):
                    tensor[cls] = new_script_bias
                else:
                    tensor[cls] = 0.0
            if new_classes and key in params:
                new_rows[key] = [[c, c + 1] for c in new_classes]
            mapped.append(key)
        else:
            if value.shape != tensor.shape:
                if strict:
                    raise SystemExit(f"--init-from: {key} has shape {tuple(value.shape)} in the source and "
                                     f"{tuple(tensor.shape)} in the new model")
                fresh.append(key)
                continue
            tensor.copy_(value.to(tensor))
            copied.append(key)
    missing_params = [k for k in fresh if k in params]
    if strict and missing_params:
        raise SystemExit(f"--init-from: the source has no weights for {missing_params}")
    model.load_state_dict(target)
    summary = {
        "source_config_hash": source.config_hash, "weights_sha256": source.weights_sha256,
        "tokenizer_sha256": hashlib.sha256((source.tok / "aggregate.json").read_bytes()).hexdigest(),
        "kept_scripts": list(keep), "new_scripts": mapping["new_scripts"],
        "new_script_logits": new_script_logits,
        **({"new_script_bias": new_script_bias} if new_script_logits == "fresh" else {}),
    }
    report = {"source": str(source.path), "copied": len(copied), "mapped": mapped, "fresh": fresh, "unused": unused,
              "token_spans": mapping["tokens"], "old_vocab": old_tok.vocab_size, "new_vocab": new_tok.vocab_size}
    return WarmStart(summary, new_rows, missing_params, report)


def _complement(spans: Sequence[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    out, position = [], 0
    for start, end in sorted(spans):
        if start > position:
            out.append((position, start))
        position = max(position, end)
    if position < size:
        out.append((position, size))
    return out


def prepare(run_dir: Path, init_from: str | Path, model, new_tok, keep: Sequence[str] | None = None,
            tok: str | Path | None = None, new_script_logits: str = "fresh", new_script_bias: float = -2.0,
            force_new_run: bool = False, log=print) -> WarmStart:
    """The warm start of one run. When the source exists it is always applied (deterministic; a
    resumed run then loads its checkpoint over these weights) and init_from.json is written. When
    the source is gone (a fresh VM resuming the run from its synced checkpoints), the record is
    read from init_from.json or the newest checkpoint ("warm_start"), and the run must resume."""
    from indicear.runtime.checkpoint import atomic_write_text
    from indicear.tokenizer import AggregateTokenizer

    run_dir = Path(run_dir)
    record = run_dir / INIT_FILE
    if not Path(init_from).expanduser().exists():
        stored = stored_record(run_dir)
        if stored is None:
            raise SystemExit(f"--init-from {init_from} does not exist and {run_dir} has no warm-start record")
        if force_new_run:
            raise SystemExit(f"--force-new-run needs the warm-start source {init_from}, which is missing")
        log(f"warm start: source {init_from} is not on this host; resuming from the run's checkpoint "
            f"(record of {stored.report.get('source')})")
        return stored
    source = load_source(init_from, tok)
    old_tok = AggregateTokenizer.load(source.tok)
    if keep is None:
        keep = kept_scripts(new_tok, source.tok)
    warm = apply(model, source, old_tok, new_tok, keep, new_script_logits, new_script_bias)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(record, json.dumps(warm.to_json(), indent=1))
    rows = sum(e - s for spans in warm.new_rows.values() for s, e in spans)
    log(f"warm start from {source.path} (config {source.config_hash}): {warm.report['copied']} tensors copied, "
        f"{len(warm.report['mapped'])} output tensors mapped by script (kept {', '.join(keep)}; new "
        f"{', '.join(warm.summary['new_scripts']) or 'none'}), {rows} fresh output rows; vocabulary "
        f"{old_tok.vocab_size} -> {new_tok.vocab_size}")
    return warm


def stored_record(run_dir: Path) -> WarmStart | None:
    """init_from.json, else the "warm_start" entry of the newest checkpoint in the run directory."""
    record = Path(run_dir) / INIT_FILE
    if record.exists():
        return WarmStart.from_json(json.loads(record.read_text(encoding="utf-8")))
    for path in sorted(Path(run_dir).glob("ckpt-*.pt"), reverse=True):
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001 - a truncated checkpoint: try the previous one
            state = {}
        if state.get("warm_start"):
            return WarmStart.from_json(state["warm_start"])
    return None


def kept_scripts(new_tok, source_tok: Path) -> list[str]:
    """The scripts whose models the new tokenizer took from the source tokenizer unchanged
    (aggregate.json "extended_from", checked by sha256)."""
    directory = getattr(new_tok, "directory", None)
    meta = json.loads((Path(directory) / "aggregate.json").read_text(encoding="utf-8")) if directory else {}
    extended = meta.get("extended_from")
    if not extended:
        raise SystemExit("--init-from: the run's tokenizer was not built with --tok-from (aggregate.json has no "
                         "extended_from); pass --init-keep-scripts to name the kept scripts")
    base_sha = hashlib.sha256((Path(source_tok) / "aggregate.json").read_bytes()).hexdigest()
    if extended["aggregate_sha256"] != base_sha:
        raise SystemExit("--init-from: the run's tokenizer was extended from another tokenizer than the source run's")
    for script, sha in extended["kept"].items():
        if hashlib.sha256((Path(source_tok) / f"{script}.model").read_bytes()).hexdigest() != sha:
            raise SystemExit(f"--init-from: {script}.model differs from the source run's")
    return list(extended["kept"])
