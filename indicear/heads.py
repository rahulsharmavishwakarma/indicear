"""Output heads for design A.

- `SelfConditionedCTC`: intermediate CTC (InterCTC, arXiv 2102.03216) with self-conditioning (arXiv 2104.02724;
  LUPET-style phonetic units, arXiv 2401.03689): logits = W LN(x), and x' = x + B softmax(logits)
  feeds the next encoder layer. B is zero-initialized so training starts from the frozen encoder.
- `ScriptFactoredOutput`: the two-level output, log p(token) = log p(script(token) | h) +
  log p(token | script, h), with blank as script class 0. Tokens follow the aggregate
  tokenizer's contiguous per-script ranges; pad/bos/eos get a constant -1e4.
- `FactoredTransducer`: RNN-T (prediction LSTM + joint) whose joint output is script-factored;
  its loss gathers only the target script's segment per label position.
- `HiddenSpecAugment`: time and feature masking on cached hidden states.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from indicear.losses import make_transducer_loss

NEG = -1e4  # log-prob for tokens that never occur (pad/bos/eos); finite to keep CTC stable


class CTCOutput(nn.Module):
    """LayerNorm + Linear + log_softmax over the full vocabulary."""

    def __init__(self, dim: int, vocab: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, vocab)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(hidden)).float().log_softmax(-1)


class SelfConditionedCTC(nn.Module):
    def __init__(self, dim: int, vocab: int, condition: bool = True):
        super().__init__()
        self.head = CTCOutput(dim, vocab)
        self.back = nn.Linear(vocab, dim) if condition else None
        if self.back is not None:
            nn.init.zeros_(self.back.weight)
            nn.init.zeros_(self.back.bias)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = self.head(hidden)
        if self.back is not None:
            hidden = hidden + self.back(log_probs.exp().to(hidden.dtype)).to(hidden.dtype)
        return hidden, log_probs


class ScriptFactoredOutput(nn.Module):
    def __init__(self, dim: int, ranges: Sequence[tuple[int, int]], vocab: int, norm: bool = True):
        super().__init__()
        self.ranges = [tuple(r) for r in ranges]
        self.vocab = vocab
        self.first = self.ranges[0][0]  # ids below are blank (0) and specials
        if any(a != b for (_, a), (b, _) in zip(self.ranges, self.ranges[1:], strict=False)):
            raise ValueError("script ranges must be contiguous")
        if self.ranges[-1][1] != vocab:
            raise ValueError("script ranges must end at the vocabulary size")
        self.norm = nn.LayerNorm(dim) if norm else nn.Identity()
        self.script_head = nn.Linear(dim, len(self.ranges) + 1)
        self.token_head = nn.Linear(dim, vocab)
        token_script = torch.full((vocab,), -1, dtype=torch.long)
        token_script[0] = 0
        for index, (start, end) in enumerate(self.ranges):
            token_script[start:end] = index + 1
        self.register_buffer("token_script", token_script, persistent=False)

    def script_log_probs(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.script_head(hidden).float().log_softmax(-1)

    def forward(self, hidden: torch.Tensor, normed: bool = False) -> torch.Tensor:
        """Full log-probabilities [..., vocab] (sums to 1 over blank + all script tokens)."""
        hidden = hidden if normed else self.norm(hidden)
        script_lp = self.script_log_probs(hidden)
        logits = self.token_head(hidden).float()
        pieces = [script_lp[..., :1], logits.new_full((*logits.shape[:-1], self.first - 1), NEG)]
        for index, (start, end) in enumerate(self.ranges):
            segment = logits[..., start:end]
            pieces.append(
                segment - segment.logsumexp(-1, keepdim=True) + script_lp[..., index + 1 : index + 2]
            )
        return torch.cat(pieces, dim=-1)

    def gathered(
        self, hidden: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transducer log-probs without the full vocabulary tensor.

        hidden [B, T, U+1, J], targets [B, U] (padding: any id with no script) ->
        blank_lp [B, T, U+1], label_lp [B, T, U] where label_lp[b, t, u] = log p(targets[b, u]).
        """
        hidden = self.norm(hidden)
        script_lp = self.script_log_probs(hidden)
        blank_lp = script_lp[..., 0]
        batch, frames, _, _ = hidden.shape
        labels = targets.shape[1]
        label_lp = blank_lp.new_zeros(batch, frames, labels)
        target_scripts = self.token_script[targets.long()]
        for index, (start, end) in enumerate(self.ranges):
            b_idx, u_idx = torch.nonzero(target_scripts == index + 1, as_tuple=True)
            if b_idx.numel() == 0:
                continue
            selected = hidden[b_idx, :, u_idx, :]  # [N, T, J]
            weight = self.token_head.weight[start:end]
            bias = self.token_head.bias[start:end]
            logits = torch.nn.functional.linear(selected, weight, bias).float()  # [N, T, L]
            local = (targets[b_idx, u_idx].long() - start)[:, None, None].expand(-1, frames, 1)
            chosen = logits.gather(-1, local).squeeze(-1) - logits.logsumexp(-1)
            chosen = chosen + script_lp[b_idx, :, u_idx, index + 1]
            label_lp[b_idx, :, u_idx] = chosen  # advanced indices around a slice: [N, T]
        return blank_lp, label_lp


class FactoredTransducer(nn.Module):
    """RNN-T head with a script-factored joint output. Blank (id 0) is the start symbol."""

    def __init__(
        self,
        enc_dim: int,
        ranges: Sequence[tuple[int, int]],
        vocab: int,
        pred_dim: int = 640,
        joint_dim: int = 640,
        dropout: float = 0.1,
        loss_kind: str = "rnnt",
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab, pred_dim)
        self.lstm = nn.LSTM(pred_dim, pred_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.enc_proj = nn.Linear(enc_dim, joint_dim)
        self.pred_proj = nn.Linear(pred_dim, joint_dim)
        self.output = ScriptFactoredOutput(joint_dim, ranges, vocab)
        self.loss_fn = make_transducer_loss(loss_kind)

    def predict(self, tokens: torch.Tensor, state=None):
        """Prediction network in float32 (cuDNN LSTM under bf16 autocast is avoided)."""
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            out, state = self.lstm(self.dropout(self.embed(tokens)).float(), state)
            return self.pred_proj(out), state

    def joint_hidden(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.enc_proj(enc)[:, :, None, :] + pred[:, None, :, :].to(enc.dtype))

    def loss(
        self, enc: torch.Tensor, enc_lengths: torch.Tensor, targets: list[torch.Tensor], pad: int,
        denominator: float | None = None,
    ) -> torch.Tensor:
        """Token-normalized RNN-T loss (`denominator` replaces the token count, as in ctc_loss)."""
        device = enc.device
        lengths = torch.tensor([len(t) for t in targets], device=device)
        padded = torch.full((len(targets), max(1, int(lengths.max()))), pad, device=device)
        for row, target in enumerate(targets):
            padded[row, : len(target)] = target.to(device)
        start = torch.zeros(len(targets), 1, dtype=torch.long, device=device)
        pred, _ = self.predict(torch.cat([start, padded], dim=1))
        hidden = self.joint_hidden(enc, pred)  # [B, T, U+1, J]
        blank_lp, label_lp = self.output.gathered(hidden, padded)
        per_utt = self.loss_fn(blank_lp, label_lp, enc_lengths, lengths)
        return per_utt.sum() / (lengths.sum().clamp_min(1) if denominator is None else denominator)

    @torch.no_grad()
    def greedy(self, enc: torch.Tensor, lengths: torch.Tensor, max_symbols: int = 5) -> list[list[int]]:
        batch, frames, _ = enc.shape
        projected = self.enc_proj(enc)
        hyps: list[list[int]] = [[] for _ in range(batch)]
        token = torch.zeros(batch, 1, dtype=torch.long, device=enc.device)
        pred, state = self.predict(token)
        pred = pred[:, 0]
        for t in range(frames):
            active = t < lengths
            for _ in range(max_symbols):
                hidden = torch.tanh(projected[:, t] + pred.to(projected.dtype))
                best = self.output(hidden).argmax(-1)
                emit = active & (best != 0)
                if not bool(emit.any()):
                    break
                new_pred, new_state = self.predict(best[:, None], state)
                pred = torch.where(emit[:, None], new_pred[:, 0], pred)
                state = tuple(
                    torch.where(emit[None, :, None], new, old)
                    for new, old in zip(new_state, state, strict=True)
                )
                for row in torch.nonzero(emit).flatten().tolist():
                    hyps[row].append(int(best[row]))
                active = emit
        return hyps


class HiddenSpecAugment(nn.Module):
    """Time and feature masks on hidden states [B, T, D]; masked values take the utterance mean
    (the hidden-state analogue of SpecAugment's zero on normalized features)."""

    def __init__(
        self,
        time_masks_per_100: float = 2.0,
        time_mask_max: int = 10,
        feature_masks: int = 2,
        feature_mask_max: int = 64,
    ):
        super().__init__()
        self.time_masks_per_100 = time_masks_per_100
        self.time_mask_max = time_mask_max
        self.feature_masks = feature_masks
        self.feature_mask_max = feature_mask_max

    def forward(self, hidden: torch.Tensor, lengths: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Vectorized on the tensor's device (GPU-side augmentation; uses the device RNG)."""
        batch, frames, dim = hidden.shape
        device = hidden.device
        lengths = lengths.to(device)
        rate = self.time_masks_per_100 * scale / 100
        masks = max(1, round(rate * frames))
        counts = torch.round(rate * lengths.float()).long()  # masks per row, by its own length
        widths = torch.randint(0, self.time_mask_max + 1, (batch, masks), device=device)
        widths = widths * (torch.arange(masks, device=device)[None, :] < counts[:, None])
        starts = (torch.rand(batch, masks, device=device) * (lengths[:, None] - widths).clamp_min(1)).long()
        positions = torch.arange(frames, device=device)
        time_masked = ((positions[None, None, :] >= starts[..., None])
                       & (positions[None, None, :] < (starts + widths)[..., None])).any(1)  # [B, T]
        f_widths = torch.randint(0, self.feature_mask_max + 1, (batch, self.feature_masks), device=device)
        f_starts = (torch.rand(batch, self.feature_masks, device=device) * (dim - f_widths).clamp_min(1)).long()
        dims = torch.arange(dim, device=device)
        feature_masked = ((dims[None, None, :] >= f_starts[..., None])
                          & (dims[None, None, :] < (f_starts + f_widths)[..., None])).any(1)  # [B, D]
        masked = time_masked[:, :, None] | feature_masked[:, None, :]
        valid = (positions[None, :] < lengths[:, None]).unsqueeze(-1)
        mean = (hidden * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)
        return torch.where(masked, mean.to(hidden.dtype), hidden)
