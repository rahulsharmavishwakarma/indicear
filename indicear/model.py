"""Design A on cached layer-6 states, and the ablation variants.

    v1  linear CTC on frozen layer 12 (probe baseline; layers 7-12 frozen, no adapters)
    v2  adapters on layers 7-12 + final CTC
    v3  v2 + self-conditioned phonetic CTC at layers 6 and 9
    v4  v3 + two-level script-factored output (CTC form)
    v5  v4 + CR-CTC consistency regularization (two augmented views, symmetric KL)
    v6  v5 + factored RNN-T final head, CTC branch at weight 0.3 (design A with RNN-T for TDT)

Loss: main = CTC (v1-v5) or (1 - w_ctc) * RNN-T + w_ctc * CTC (v6); with phonetic CTC,
total = (1 - w_inter) * main + w_inter * mean(phonetic CTC at 6, 9).

Language identification (indicear.lid) is an optional *output* on top of any variant: `DesignA(...,
lid=LidConfig())` adds w_utt * utterance-LID + w_frame * frame-LID to the loss. It never routes,
gates or masks the transcript. `resolve_variant("v3+lid")` names such a run; "lid" alone is the
cheap LID-only run on the frozen encoder (no transcript heads). The LID config is kept out of
`VariantConfig`, so the existing variants' config hashes (and their checkpoints) are unchanged.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import nn

from indicear.encoder import AdapterSet, StagedTower, lengths_to_valid
from indicear.heads import (
    CTCOutput,
    FactoredTransducer,
    HiddenSpecAugment,
    ScriptFactoredOutput,
    SelfConditionedCTC,
)
from indicear.lid import (
    LidConfig,
    LidHeads,
    LidTargets,
    frame_spans,
    lid_config_from_dict,
)
from indicear.losses import consistency_loss, ctc_loss
from indicear.tokenizer import PAD


@dataclass(frozen=True)
class VariantConfig:
    name: str
    adapters: bool = False
    phonetic: bool = False
    factored: bool = False
    cr_ctc: bool = False
    transducer: bool = False
    adapter_rank: int = 128
    inter_layers: tuple[int, ...] = (6, 9)
    inter_weight: float = 0.3
    ctc_weight: float = 0.3  # CTC branch weight when the transducer is the main head
    cr_weight: float = 0.2
    cr_time_scale: float = 2.5  # CR-CTC uses heavier time masking (arXiv 2410.05101)
    specaugment: bool = True
    checkpoint: bool = False  # activation checkpointing of layers 7-12

    @property
    def frozen_upper(self) -> bool:
        return not self.adapters and not self.phonetic


_V1 = VariantConfig("v1")
_V2 = replace(_V1, name="v2", adapters=True)
_V3 = replace(_V2, name="v3", phonetic=True)
_V4 = replace(_V3, name="v4", factored=True)
_V5 = replace(_V4, name="v5", cr_ctc=True)
_V6 = replace(_V5, name="v6", transducer=True)
VARIANTS = {v.name: v for v in (_V1, _V2, _V3, _V4, _V5, _V6)}
LID_ONLY = replace(_V1, name="lid")


def resolve_variant(name: str, lid: LidConfig | None = None) -> tuple[VariantConfig, LidConfig | None]:
    """"v3" -> (v3, None); "v3+lid" -> (v3, LidConfig()); "lid" -> the LID-only run on the frozen
    encoder (v1's frozen stack, no transcript heads). `lid` overrides the default LidConfig."""
    if name == "lid":
        return LID_ONLY, replace(lid or LidConfig(), only=True)
    base, _, suffix = name.partition("+")
    if suffix not in ("", "lid") or base not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}: use one of {sorted(VARIANTS)}, '<v>+lid' or 'lid'")
    return VARIANTS[base], (replace(lid or LidConfig(), only=False) if suffix else None)


def ctc_greedy(log_probs: torch.Tensor, lengths: torch.Tensor) -> list[tuple[list[int], list[int]]]:
    """Best-path decode. Returns (tokens, emission frames) per utterance."""
    best = log_probs.argmax(-1).cpu()
    out = []
    for row, length in zip(best, lengths.cpu().tolist(), strict=True):
        tokens, frames, previous = [], [], 0
        for frame, token in enumerate(row[:length].tolist()):
            if token != previous and token != 0:
                tokens.append(token)
                frames.append(frame)
            previous = token
        out.append((tokens, frames))
    return out


class DesignA(nn.Module):
    def __init__(
        self,
        staged: StagedTower,
        config: VariantConfig,
        vocab_size: int,
        script_ranges: list[tuple[int, int]],
        num_units: int,
        lid: LidConfig | None = None,
    ):
        super().__init__()
        object.__setattr__(self, "staged", staged)  # not a submodule: never saved or trained
        self.config = config
        self.lid_config = lid
        self.vocab_size, self.script_ranges, self.num_units = vocab_size, [tuple(r) for r in script_ranges], num_units
        lid_only = lid is not None and lid.only
        if lid_only and not config.frozen_upper:
            raise ValueError("the LID-only variant runs on the frozen encoder: use resolve_variant('lid')")
        if lid is not None and not all(staged.split <= k <= staged.num_layers for k in lid.layers):
            raise ValueError(f"LID layers {lid.layers} must lie in [{staged.split}, {staged.num_layers}]")
        dim = staged.hidden_size
        self.adapters = AdapterSet(dim, rank=config.adapter_rank) if config.adapters else None
        if self.adapters is not None:
            self.adapters.attach(staged.tower)
        self.inter = (
            nn.ModuleDict({str(k): SelfConditionedCTC(dim, num_units) for k in config.inter_layers})
            if config.phonetic
            else None
        )
        self.final = None if lid_only else (
            ScriptFactoredOutput(dim, script_ranges, vocab_size) if config.factored else CTCOutput(dim, vocab_size)
        )
        self.transducer = (
            FactoredTransducer(dim, script_ranges, vocab_size) if config.transducer else None
        )
        self.specaug = HiddenSpecAugment()
        # Built last, so enabling LID leaves every transcript parameter's initialization unchanged.
        self.lid = LidHeads(dim, lid) if lid is not None else None

    def detach_adapters(self) -> None:
        if self.adapters is not None:
            self.adapters.detach()

    @contextlib.contextmanager
    def adapters_attached(self):
        """Attach this model's adapters to the shared tower for the block (when they are not
        attached already), detaching them afterwards: several models can share one tower."""
        if self.adapters is None or self.adapters.attached:
            yield
            return
        self.adapters.attach(self.staged.tower)
        try:
            yield
        finally:
            self.adapters.detach()

    # ------------------------------------------------------------ forward

    def encode(
        self,
        h6: torch.Tensor,
        lengths: torch.Tensor,
        augment: bool = False,
        time_scale: float = 1.0,
        states: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """`states`, when given, receives the output of each LID layer (layer k's output before
        the self-conditioning at k, as the encoder computed it; stop-gradient if lid.detach).
        The LID-only variant stops the stack at its highest LID layer."""
        if augment and self.config.specaugment:
            h6 = self.specaug(h6, lengths, scale=time_scale)
        valid = lengths_to_valid(lengths, h6.shape[1])
        inter: dict[str, torch.Tensor] = {}
        callbacks = {}
        if self.inter is not None:
            for key, module in self.inter.items():

                def callback(hidden, key=key, module=module):
                    hidden, log_probs = module(hidden)
                    inter[key] = log_probs
                    return hidden

                callbacks[int(key)] = callback
        stop_after = None
        if states is not None and self.lid is not None:
            detach = self.lid.config.detach


            def capture(layer, inner):
                def callback(hidden):
                    states[layer] = hidden.detach() if detach else hidden
                    return inner(hidden) if inner is not None else hidden

                return callback

            for layer in self.lid.layers:
                callbacks[layer] = capture(layer, callbacks.get(layer))
            if self.final is None:
                stop_after = max(self.lid.layers)
        if stop_after is not None and stop_after <= self.staged.split:
            callbacks[self.staged.split](h6)
            return h6, inter
        frozen = self.config.frozen_upper
        context = torch.no_grad() if frozen else contextlib.nullcontext()
        with context:
            hidden = self.staged.upper(h6, valid, callbacks, checkpoint=self.config.checkpoint and not frozen,
                                       stop_after=stop_after)
        return hidden, inter

    def lid_outputs(self, h6: torch.Tensor, lengths: torch.Tensor, augment: bool) -> dict[str, torch.Tensor]:
        """LID heads alone (the LID-only variant): no transcript head runs."""
        states: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            self.encode(h6, lengths, augment=augment, states=states)
        return self.lid(states, lengths)

    def losses(
        self,
        h6: torch.Tensor,
        lengths: torch.Tensor,
        tokens: list[torch.Tensor],
        units: list[torch.Tensor],
        lid: LidTargets | None = None,
        norms: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Transcript losses plus, when the model has LID heads and `lid` targets are given,
        w_utt * lid_utt + w_frame * lid_frame (indicear.lid). The transcript terms never depend on
        the LID heads.

        `norms` (from `loss_norms` on a whole batch): every term is normalized by the whole
        batch's denominator instead of this batch's, so the losses (and gradients) of the
        micro-batches of a split batch add up exactly to those of the batch in one piece."""
        cfg = self.config
        norms = norms or {}
        if self.final is None:  # LID-only
            if lid is None:
                raise ValueError("the LID-only variant needs LID targets")
            out = self.lid.losses(self.lid_outputs(h6, lengths, self.training), lengths, lid, norms=norms)
            out["loss"] = self.lid.weighted(out)
            if out["loss"] is None:
                raise ValueError("the LID-only variant needs a non-zero LID weight")
            return out
        scale = cfg.cr_time_scale if cfg.cr_ctc else 1.0
        states: dict[int, torch.Tensor] | None = {} if self.lid is not None and lid is not None else None
        hidden, inter = self.encode(h6, lengths, augment=self.training, time_scale=scale, states=states)
        log_probs = self.final(hidden)
        out: dict[str, torch.Tensor] = {"ctc": ctc_loss(log_probs, lengths, tokens, denominator=norms.get("tokens"))}
        ctc_term = out["ctc"]
        inter_views = [inter]
        if cfg.cr_ctc and self.training:
            hidden_b, inter_b = self.encode(h6, lengths, augment=True, time_scale=scale)
            log_probs_b = self.final(hidden_b)
            valid = lengths_to_valid(lengths, h6.shape[1]).float()
            out["ctc_b"] = ctc_loss(log_probs_b, lengths, tokens, denominator=norms.get("tokens"))
            out["consistency"] = consistency_loss(log_probs, log_probs_b, valid, denominator=norms.get("frames"))
            ctc_term = 0.5 * (out["ctc"] + out["ctc_b"]) + cfg.cr_weight * out["consistency"]
            inter_views.append(inter_b)
        main = ctc_term
        if self.transducer is not None:
            out["rnnt"] = self.transducer.loss(hidden, lengths, tokens, PAD, denominator=norms.get("tokens"))
            main = (1 - cfg.ctc_weight) * out["rnnt"] + cfg.ctc_weight * ctc_term
        if self.inter is not None:
            inter_losses = [
                ctc_loss(view[key], lengths, units, denominator=norms.get("units")) for view in inter_views for key in view
            ]
            out["phonetic"] = torch.stack(inter_losses).mean()
            main = (1 - cfg.inter_weight) * main + cfg.inter_weight * out["phonetic"]
        if states is not None:
            out.update(self.lid.losses(self.lid(states, lengths), lengths, lid, norms=norms))
            extra = self.lid.weighted(out)
            if extra is not None:
                main = main + extra
        out["loss"] = main
        return out

    def loss_norms(self, tokens: list[torch.Tensor], units: list[torch.Tensor], lengths: torch.Tensor,
                   lid: LidTargets | None = None) -> dict[str, float]:
        """The whole batch's loss denominators for `losses(norms=...)`: target tokens (CTC,
        RNN-T), phonetic units (intermediate CTC), valid frames (CR-CTC consistency) and, with
        LID heads and targets, the LID normalizers (indicear.lid.lid_denominators); each >= 1."""
        norms = {"tokens": float(max(1, sum(len(t) for t in tokens))),
                 "units": float(max(1, sum(len(u) for u in units))),
                 "frames": float(max(1, int(torch.as_tensor(lengths).sum())))}
        if self.lid is not None and lid is not None:
            norms.update(self.lid.loss_norms(lid))
        return norms

    @torch.no_grad()
    def decode(self, h6: torch.Tensor, lengths: torch.Tensor, constraints=None) -> dict:
        """Transcript decodes (absent for the LID-only variant) and, with LID heads,
        result["lid"] = {"utt_logits" [B, 12], "frame_log_probs" [B, T, 13], "spans": per row an
        ordered list of (language, start_frame, end_frame) from the frame head alone}. Spans
        snapped to transcript words: indicear.lid.utterance_spans with the tokenizer's word_starts.

        `constraints` (a `indicear.decoding.Constrainer`, built from the tokenizer): the CTC greedy
        decode becomes the constrained best path under its decode-time script rules, the LID
        heads (when present) choosing the allowed scripts; result["ctc_scripts"] lists them per
        row (None: all). None, or both rules off, keeps the unconstrained decode. The LID output
        itself is never changed by it, and the RNN-T decode is not constrained."""
        states: dict[int, torch.Tensor] | None = {} if self.lid is not None else None
        hidden, inter = self.encode(h6, lengths, states=states)
        result: dict = {"inter": inter}
        lid_out = self.lid(states, lengths) if self.lid is not None else None
        if self.final is not None:
            log_probs = self.final(hidden)
            if constraints is not None and constraints.options.active:
                constraints.tables.check(self.vocab_size, self.script_ranges)
                heads = lid_out or {}
                decoded, scripts = constraints.greedy(log_probs, lengths, heads.get("utt_logits"),
                                                      heads.get("frame_log_probs"))
                result.update({"ctc": decoded, "ctc_scripts": scripts, "final_log_probs": log_probs})
            else:
                result.update({"ctc": ctc_greedy(log_probs, lengths), "final_log_probs": log_probs})
        if self.transducer is not None:
            result["rnnt"] = self.transducer.greedy(hidden, lengths)
        if self.lid is not None:
            if "frame_log_probs" in lid_out:
                lid_out["spans"] = [
                    frame_spans(lid_out["frame_log_probs"][row], int(n), self.lid.config.min_span_frames)
                    for row, n in enumerate(lengths.tolist())
                ]
            result["lid"] = lid_out
        return result

    # ------------------------------------------------------------ persistence

    def head_modules(self) -> dict[str, nn.Module]:
        modules: dict[str, nn.Module] = {}
        if self.final is not None:
            modules["final"] = self.final
        if self.inter is not None:
            modules["inter"] = self.inter
        if self.transducer is not None:
            modules["transducer"] = self.transducer
        if self.lid is not None:
            modules["lid"] = self.lid
        return modules

    def save(self, directory: str | Path) -> None:
        """adapters.safetensors (own file) + heads.safetensors (LID heads under "lid.") +
        variant.json (+ lid.json with LID heads)."""
        from safetensors.torch import save_file

        from indicear.common import write_json

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.adapters is not None:
            self.adapters.save(directory / "adapters.safetensors")
        tensors = {
            f"{name}.{key}": value.detach().cpu().contiguous()
            for name, module in self.head_modules().items()
            for key, value in module.state_dict().items()
        }
        save_file(tensors, str(directory / "heads.safetensors"))
        write_json(directory / "variant.json", asdict(self.config))
        if self.lid_config is not None:
            write_json(directory / "lid.json", asdict(self.lid_config))

    def load(self, directory: str | Path) -> None:
        from safetensors.torch import load_file

        directory = Path(directory)
        if self.adapters is not None:
            self.adapters.load(directory / "adapters.safetensors")
        tensors = load_file(str(directory / "heads.safetensors"))
        for name, module in self.head_modules().items():
            prefix = f"{name}."
            module.load_state_dict({k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)})


def load_variant_config(path: str | Path) -> VariantConfig:
    import json

    data = json.loads(Path(path).read_text())
    data["inter_layers"] = tuple(data.get("inter_layers", (6, 9)))
    return VariantConfig(**data)


def load_lid_config(directory: str | Path) -> LidConfig | None:
    """The LID config saved beside a model (lid.json), or None when it has no LID heads."""
    import json

    path = Path(directory) / "lid.json"
    return lid_config_from_dict(json.loads(path.read_text())) if path.exists() else None
