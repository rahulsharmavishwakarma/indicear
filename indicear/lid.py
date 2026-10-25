"""Automatic language identification as *outputs*, never as routing.

Design A deliberately has no language-ID routing (docs/architecture.md): the transcript decoder
never sees these heads. They read encoder states, add their own losses, and report what was
spoken; in training and in the default decode, nothing gates, masks or re-weights the
transcript's scripts by a predicted language. The one exception is opt-in and decode-time only:
`indicear.decoding`'s LID script set (`--lid-script-set on`, the transcribe CLI's default) masks
the scripts outside Latin plus the likely languages' scripts after the heads have run. It never
changes a loss, a weight or the LID outputs, and Latin plus every language above the threshold
stays allowed, so code-switching into English is never routed away.

Two optional heads (`LidConfig`), trainable on top of any variant v1-v6 or alone on the frozen
encoder (the "lid" variant):

- **Utterance head** (`UtteranceLid`): a learned softmax-weighted sum of encoder layer states
  (default layers 6 and 12: layer 6 is the cached/live lower output, layer 12 is the top of the
  upper stack, adapted when the variant has adapters; both are computed anyway, so the head adds
  no encoder work), each layer normalized by a parameter-free LayerNorm, then Linear -> SiLU,
  attentive statistics pooling (attention-weighted mean and standard deviation over valid
  frames), and a two-layer MLP to 12 language logits. Loss: cross-entropy with label smoothing
  and an optional per-language class-balance weight (`balance_weights`). Target: the row's
  `lang` (for code-mixed rows that is the matrix language); rows whose `lang` is not one of the
  twelve (non-speech negatives, "hi+en" evaluation rows) are ignored.
- **Frame head** (`FrameLid`): its own layer mix, then Linear -> SiLU -> Linear to 13 classes per
  25 fps frame: class 0 is non-speech (also the CTC blank), classes 1..12 are `LID_LANGS`.
  Targets, without forced alignment:
  * monolingual rows: every valid frame gets the utterance language (dense cross-entropy). There
    is no VAD, so pauses inside an utterance count as that language; the non-speech class is
    learned from the non-speech negatives (every frame -> class 0) and from the blank of the
    code-switch CTC below.
  * code-mixed rows with `word_langs`: the word languages are collapsed into language runs
    (["hi", "hi", "en", "hi"] -> [hi, en, hi]) and the head is trained with CTC over that run
    sequence with blank = class 0 (`frame_lid_loss`). CTC sums over every alignment, so the head
    learns where the switches are from the audio alone; a dense labelling ("hi hi hi en en hi")
    is itself a valid CTC path, so the dense monolingual targets and the run CTC agree. On
    code-mixed rows *alone* CTC goes peaky (blank takes many speech frames); it is the dense
    monolingual rows in the same mixture that keep the posteriors dense (tests/test_lid.py
    shows both on a toy). Word languages that are not one of the twelve are skipped; rows whose
    runs collapse to one language are treated as monolingual.
  Per row the loss is the mean per-frame CE, or CTC / frames; the batch loss is the mean over the
  rows that have a target, so both kinds of row are on a per-frame scale.

Inference (`frame_spans`, `word_frames`, `merge_with_words`): greedy decode of the frame head
(argmax per frame, drop class 0, collapse repeats; since run targets never repeat a language, two
runs of one language separated only by non-speech merge into one span) gives an ordered list of
(language, start_frame, end_frame) with an exclusive end. When the transcript's CTC word timings
are available, each word takes the language with the most frame-head probability mass over its
frames (non-speech excluded, so this also holds up when the posteriors are peaky), and spans are
rebuilt from consecutive words of one language, so span boundaries fall on word boundaries.

By default the heads read *detached* states (`detach=True`): their gradients never reach the
adapters or the self-conditioning heads, so adding them cannot change transcription training,
and with weight 0 the transcript output is bit-identical. `detach=False` makes it multi-task
learning through the adapters (still no routing).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from indicear.common import LANGS

LID_LANGS: tuple[str, ...] = tuple(sorted(LANGS))  # bn en gu hi kn ml mr or pa ta te ur
NUM_LANGS = len(LID_LANGS)
NON_SPEECH = 0  # frame class 0: non-speech and CTC blank; language i is frame class i + 1
FRAME_CLASSES = NUM_LANGS + 1
IGNORE = -100
NON_SPEECH_LANGS = ("none",)  # `lang` of the non-speech negatives (indicear.augment.make_negatives)
_INDEX = {lang: i for i, lang in enumerate(LID_LANGS)}


@dataclass(frozen=True)
class LidConfig:
    utterance: bool = True
    frame: bool = True
    utterance_weight: float = 0.1
    frame_weight: float = 0.1
    layers: tuple[int, ...] = (6, 12)  # encoder layers mixed (1-based, >= the cached layer)
    hidden: int = 256
    label_smoothing: float = 0.1
    class_weights: tuple[float, ...] | None = None  # per LID_LANGS entry; see balance_weights
    detach: bool = True  # heads read stop-gradient states: transcription training is untouched
    only: bool = False  # the "lid" variant: no transcript heads, the encoder stays frozen
    min_span_frames: int = 1  # inference: spans shorter than this are dropped, then re-merged

    def __post_init__(self):
        if not (self.utterance or self.frame):
            raise ValueError("LidConfig needs the utterance head, the frame head, or both")
        if self.class_weights is not None and len(self.class_weights) != NUM_LANGS:
            raise ValueError(f"class_weights needs {NUM_LANGS} values, in LID_LANGS order")


def lid_config_from_dict(data: Mapping) -> LidConfig:
    data = dict(data)
    data["layers"] = tuple(data.get("layers", (6, 12)))
    if data.get("class_weights") is not None:
        data["class_weights"] = tuple(data["class_weights"])
    return LidConfig(**data)


def lang_index(lang: str | None) -> int:
    """Index in LID_LANGS, or IGNORE for anything else."""
    return _INDEX.get(lang, IGNORE) if lang is not None else IGNORE


def balance_weights(counts: Mapping[str, float], power: float = 0.5) -> tuple[float, ...]:
    """Class-balance weights (n_c ** -power), normalized to mean 1 over the languages present;
    absent languages get the largest weight. `counts` may be rows or hours per language."""
    present = {lang: float(counts[lang]) for lang in LID_LANGS if counts.get(lang, 0) > 0}
    if not present:
        return tuple(1.0 for _ in LID_LANGS)
    raw = {lang: n ** -power for lang, n in present.items()}
    mean = sum(raw.values()) / len(raw)
    top = max(raw.values()) / mean
    return tuple(round(raw[lang] / mean, 6) if lang in raw else round(top, 6) for lang in LID_LANGS)


# ---------------------------------------------------------------- targets


def collapse_runs(word_langs: Sequence[str]) -> list[str]:
    """Language runs of a word sequence: unknown languages skipped, consecutive repeats merged."""
    runs: list[str] = []
    for lang in word_langs:
        if lang in _INDEX and (not runs or runs[-1] != lang):
            runs.append(lang)
    return runs


@dataclass
class LidTargets:
    utterance: torch.Tensor  # [B] long: LID_LANGS index or IGNORE
    frame: torch.Tensor  # [B] long: dense frame class (lang + 1, NON_SPEECH) or IGNORE
    runs: list[torch.Tensor] = field(default_factory=list)  # [B] frame classes of the runs; empty: none

    def to(self, device) -> LidTargets:
        return LidTargets(self.utterance.to(device), self.frame.to(device), [r.to(device) for r in self.runs])

    def select(self, rows: Sequence[int]) -> LidTargets:
        index = torch.tensor(list(rows), dtype=torch.long, device=self.utterance.device)
        return LidTargets(self.utterance[index], self.frame[index], [self.runs[r] for r in rows])


def make_targets(langs: Sequence[str | None], word_langs: Sequence[Sequence[str] | None] | None = None) -> LidTargets:
    """Targets for both heads from each row's `lang` and optional `word_langs`."""
    word_langs = word_langs if word_langs is not None else [None] * len(langs)
    if len(word_langs) != len(langs):
        raise ValueError("word_langs and langs differ in length")
    utterance, frame, runs = [], [], []
    for lang, words in zip(langs, word_langs, strict=True):
        utterance.append(lang_index(lang))
        sequence = collapse_runs(words) if words else []
        if len(sequence) > 1:
            frame.append(IGNORE)
            runs.append(torch.tensor([_INDEX[x] + 1 for x in sequence], dtype=torch.long))
            continue
        runs.append(torch.zeros(0, dtype=torch.long))
        if sequence:
            frame.append(_INDEX[sequence[0]] + 1)
        elif lang in NON_SPEECH_LANGS:
            frame.append(NON_SPEECH)
        else:
            index = lang_index(lang)
            frame.append(index + 1 if index != IGNORE else IGNORE)
    return LidTargets(torch.tensor(utterance, dtype=torch.long), torch.tensor(frame, dtype=torch.long), runs)


# ---------------------------------------------------------------- heads


def _valid(lengths: torch.Tensor, frames: int) -> torch.Tensor:
    return torch.arange(frames, device=lengths.device)[None, :] < lengths[:, None]


class LayerMix(nn.Module):
    """softmax(w)-weighted sum of parameter-free LayerNorm'd layer states; w starts uniform."""

    def __init__(self, layers: Sequence[int], dim: int):
        super().__init__()
        self.layers = tuple(layers)
        self.logits = nn.Parameter(torch.zeros(len(self.layers)))
        self.dim = dim

    def forward(self, states: Mapping[int, torch.Tensor]) -> torch.Tensor:
        weights = self.logits.float().softmax(0)
        mixed = None
        for weight, layer in zip(weights, self.layers, strict=True):
            normed = F.layer_norm(states[layer].float(), (self.dim,))
            mixed = weight * normed if mixed is None else mixed + weight * normed
        return mixed

    def layer_weights(self) -> dict[int, float]:
        return dict(zip(self.layers, self.logits.detach().float().softmax(0).tolist(), strict=True))


class UtteranceLid(nn.Module):
    """Layer mix -> Linear+SiLU -> attentive mean+std pooling -> MLP -> 12 language logits."""

    def __init__(self, dim: int, layers: Sequence[int], hidden: int = 256):
        super().__init__()
        self.mix = LayerMix(layers, dim)
        self.proj = nn.Linear(dim, hidden)
        self.attention = nn.Linear(hidden, 1)
        self.classifier = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, NUM_LANGS))

    def forward(self, states: Mapping[int, torch.Tensor], valid: torch.Tensor) -> torch.Tensor:
        z = F.silu(self.proj(self.mix(states))).float()  # [B, T, H]
        scores = self.attention(z).squeeze(-1).float().masked_fill(~valid, -1e4)
        alpha = scores.softmax(-1) * valid  # an all-padding row pools to zeros, never NaN
        alpha = alpha / alpha.sum(-1, keepdim=True).clamp_min(1e-6)
        mean = (alpha[..., None] * z).sum(1)
        var = (alpha[..., None] * z.pow(2)).sum(1) - mean.pow(2)
        std = var.clamp_min(1e-5).sqrt()
        return self.classifier(torch.cat([mean, std], dim=-1)).float()


class FrameLid(nn.Module):
    """Layer mix -> Linear+SiLU -> Linear: per-frame log-probabilities over non-speech + 12."""

    def __init__(self, dim: int, layers: Sequence[int], hidden: int = 256):
        super().__init__()
        self.mix = LayerMix(layers, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, FRAME_CLASSES))

    def forward(self, states: Mapping[int, torch.Tensor]) -> torch.Tensor:
        return self.mlp(self.mix(states)).float().log_softmax(-1)


def utterance_lid_loss(
    logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.1, weight: torch.Tensor | None = None,
    denominator: float | None = None,
) -> torch.Tensor:
    """Class-weighted cross entropy, the weighted mean over rows with a target (torch's "mean":
    the sum / the summed class weights of the targets). `denominator` replaces that weight sum
    (a micro-batch normalized by its whole batch's, see `lid_denominators`)."""
    targets = targets.to(logits.device)
    if not bool((targets != IGNORE).any()):
        return logits.sum() * 0.0
    if denominator is not None:
        return F.cross_entropy(logits.float(), targets, weight=weight, ignore_index=IGNORE,
                               label_smoothing=label_smoothing, reduction="sum") / denominator
    return F.cross_entropy(logits.float(), targets, weight=weight, ignore_index=IGNORE, label_smoothing=label_smoothing)


def lid_denominators(targets: LidTargets, weight: torch.Tensor | None = None) -> dict[str, float]:
    """The normalizers of the two LID losses over a whole batch: "lid_utt" the summed class
    weights of the rows with an utterance target, "lid_frame" the number of rows with a frame
    target (dense or code-mixed runs); each at least 1."""
    utterance = targets.utterance[targets.utterance != IGNORE]
    utt = float(weight.cpu().float()[utterance.cpu()].sum()) if weight is not None else float(utterance.numel())
    frame = int((targets.frame != IGNORE).sum()) + sum(1 for run in targets.runs if len(run) > 0)
    return {"lid_utt": utt if utt > 0 else 1.0, "lid_frame": float(max(1, frame))}


def frame_lid_loss(log_probs: torch.Tensor, lengths: torch.Tensor, targets: LidTargets,
                   denominator: float | None = None) -> torch.Tensor:
    """Mean over rows with a target of: per-frame CE (dense rows) or run CTC / frames (code-mixed).
    `denominator` replaces the number of such rows (see `lid_denominators`)."""
    device = log_probs.device
    lengths = lengths.to(device)
    frame_target = targets.frame.to(device)
    per_row: list[torch.Tensor] = []
    dense = torch.nonzero(frame_target != IGNORE).flatten()
    if dense.numel():
        valid = _valid(lengths[dense], log_probs.shape[1]).float()
        chosen = log_probs[dense].gather(-1, frame_target[dense][:, None, None].expand(-1, log_probs.shape[1], 1))
        nll = -(chosen.squeeze(-1) * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        per_row.append(nll)
    switched = [r for r, run in enumerate(targets.runs) if len(run) > 0]
    if switched:
        rows = torch.tensor(switched, device=device)
        run_targets = [targets.runs[r].to(device) for r in switched]
        target_lengths = torch.tensor([len(r) for r in run_targets], device=device)
        ctc = F.ctc_loss(
            log_probs[rows].float().transpose(0, 1), torch.cat(run_targets), lengths[rows], target_lengths,
            blank=NON_SPEECH, reduction="none", zero_infinity=True,
        )
        per_row.append(ctc / lengths[rows].clamp_min(1))
    if not per_row:
        return log_probs.sum() * 0.0
    if denominator is not None:
        return torch.cat(per_row).sum() / denominator
    return torch.cat(per_row).mean()


class LidHeads(nn.Module):
    """The optional utterance and frame heads plus their weighted loss."""

    def __init__(self, dim: int, config: LidConfig):
        super().__init__()
        self.config = config
        self.utterance = UtteranceLid(dim, config.layers, config.hidden) if config.utterance else None
        self.frame = FrameLid(dim, config.layers, config.hidden) if config.frame else None
        weights = torch.tensor(config.class_weights, dtype=torch.float32) if config.class_weights else None
        self.register_buffer("class_weights", weights, persistent=False)

    @property
    def layers(self) -> tuple[int, ...]:
        return self.config.layers

    def forward(self, states: Mapping[int, torch.Tensor], lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        any_state = next(iter(states.values()))
        valid = _valid(lengths.to(any_state.device), any_state.shape[1])
        out: dict[str, torch.Tensor] = {}
        if self.utterance is not None:
            out["utt_logits"] = self.utterance(states, valid)
        if self.frame is not None:
            out["frame_log_probs"] = self.frame(states)
        return out

    def losses(self, out: Mapping[str, torch.Tensor], lengths: torch.Tensor, targets: LidTargets,
               norms: Mapping[str, float] | None = None) -> dict[str, torch.Tensor]:
        """`norms` ("lid_utt", "lid_frame" from `loss_norms` of the whole batch): normalize a
        micro-batch by its whole batch, so micro-batch losses add up to the batch loss."""
        norms = norms or {}
        losses: dict[str, torch.Tensor] = {}
        if "utt_logits" in out:
            losses["lid_utt"] = utterance_lid_loss(
                out["utt_logits"], targets.utterance, self.config.label_smoothing, self.class_weights,
                denominator=norms.get("lid_utt"),
            )
        if "frame_log_probs" in out:
            losses["lid_frame"] = frame_lid_loss(out["frame_log_probs"], lengths, targets,
                                                 denominator=norms.get("lid_frame"))
        return losses

    def loss_norms(self, targets: LidTargets) -> dict[str, float]:
        return lid_denominators(targets, self.class_weights)

    def weighted(self, losses: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
        """w_utt * lid_utt + w_frame * lid_frame; terms with weight 0 are left out entirely (so a
        non-finite LID loss can never leak into the transcript loss through 0 * nan)."""
        terms = []
        if "lid_utt" in losses and self.config.utterance_weight:
            terms.append(self.config.utterance_weight * losses["lid_utt"])
        if "lid_frame" in losses and self.config.frame_weight:
            terms.append(self.config.frame_weight * losses["lid_frame"])
        return sum(terms[1:], terms[0]) if terms else None


def parameter_counts(dim: int, config: LidConfig | None = None) -> dict[str, int]:
    heads = LidHeads(dim, config or LidConfig())
    count = lambda m: sum(p.numel() for p in m.parameters()) if m is not None else 0
    return {"utterance": count(heads.utterance), "frame": count(heads.frame), "total": count(heads)}


# ---------------------------------------------------------------- inference


def utterance_probs(logits: torch.Tensor) -> list[dict[str, float]]:
    """Per row: {language: probability} over LID_LANGS."""
    probs = logits.float().softmax(-1).cpu().tolist()
    return [dict(zip(LID_LANGS, row, strict=True)) for row in probs]


def predicted_langs(logits: torch.Tensor) -> list[str]:
    return [LID_LANGS[i] for i in logits.argmax(-1).cpu().tolist()]


Span = tuple[str, int, int]  # (language, start_frame, end_frame), end exclusive, 25 fps


def _merge_adjacent(spans: list[Span]) -> list[Span]:
    merged: list[Span] = []
    for lang, start, end in spans:
        if merged and merged[-1][0] == lang:
            merged[-1] = (lang, merged[-1][1], end)
        else:
            merged.append((lang, start, end))
    return merged


def frame_spans(log_probs: torch.Tensor, length: int | None = None, min_frames: int = 1) -> list[Span]:
    """Greedy CTC-style decode of one row's frame head [T, 13]: argmax per frame, drop non-speech,
    one span per language run (a run continues across non-speech frames until another language
    appears). Spans shorter than `min_frames` are dropped and neighbours of one language merge."""
    path = log_probs[: length if length is not None else log_probs.shape[0]].argmax(-1).cpu().tolist()
    spans: list[Span] = []
    for frame, label in enumerate(path):
        if label == NON_SPEECH:
            continue
        lang = LID_LANGS[label - 1]
        if spans and spans[-1][0] == lang:
            spans[-1] = (lang, spans[-1][1], frame + 1)
        else:
            spans.append((lang, frame, frame + 1))
    if min_frames > 1:
        spans = _merge_adjacent([s for s in spans if s[2] - s[1] >= min_frames])
    return spans


def word_frames(tokens: Sequence[int], frames: Sequence[int], word_starts: Sequence[bool], length: int) -> list[tuple[int, int]]:
    """Word timings from a CTC greedy decode (`model.ctc_greedy`: token ids and emission frames)
    and the tokenizer's `word_starts(tokens)`: a word runs from its first token's frame to the next
    word's first frame (the last word ends one frame after its last token)."""
    starts = [i for i, start in enumerate(word_starts) if start or i == 0]
    out = []
    for n, first in enumerate(starts):
        last = (starts[n + 1] if n + 1 < len(starts) else len(tokens)) - 1
        begin = frames[first]
        end = frames[starts[n + 1]] if n + 1 < len(starts) else min(length, frames[last] + 1)
        out.append((begin, max(end, begin + 1)))
    return out


def merge_with_words(log_probs: torch.Tensor, words: Sequence[tuple[int, int]]) -> tuple[list[str], list[Span]]:
    """Each word takes the language with the most frame-head probability over its frames
    (non-speech excluded); spans are rebuilt from consecutive words of one language."""
    probs = log_probs.float().exp()[:, 1:].cpu()  # [T, 12]
    word_langs: list[str] = []
    spans: list[Span] = []
    for start, end in words:
        mass = probs[start:end].sum(0) if end > start else probs[min(start, probs.shape[0] - 1)]
        lang = LID_LANGS[int(mass.argmax())]
        word_langs.append(lang)
        spans.append((lang, start, end))
    return word_langs, _merge_adjacent(spans)


def utterance_spans(
    log_probs: torch.Tensor,
    length: int,
    tokens: Sequence[int] | None = None,
    frames: Sequence[int] | None = None,
    word_starts: Sequence[bool] | None = None,
    min_frames: int = 1,
) -> list[Span]:
    """Language spans for one utterance: from the transcript's word timings when given (and the
    transcript is non-empty), else from the frame head's greedy decode alone."""
    if tokens and frames is not None and word_starts is not None:
        return merge_with_words(log_probs[:length], word_frames(tokens, frames, word_starts, length))[1]
    return frame_spans(log_probs, length, min_frames)

