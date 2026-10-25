"""CTC, CR-CTC and transducer losses.

Transducer: `rnnt_loss_from_logprobs` is an exact RNN-T loss over *gathered* log-probabilities
(blank and next-label per (t, u)), computed by a differentiable forward recursion that is
vectorized over the label axis with `torch.logcumsumexp` (T sequential steps). Gathering first is
what makes the two-level output cheap: only the target token's script segment needs a softmax,
so memory is B*T*(U+1)*max_segment instead of B*T*(U+1)*vocab. Tested against
`torchaudio.functional.rnnt_loss` (torchaudio 2.11.0, which still ships it in maintenance mode).

TDT (token-and-duration transducer, arXiv 2304.06795) is not implemented: NeMo's TDT kernels are
importable only through `nemo.collections.asr`, whose package init pulls in the full ASR stack.
`TransducerLoss` is the seam where a TDT loss (duration logits + its own recursion) plugs in.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F


def ctc_loss(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    targets: list[torch.Tensor],
    blank: int = 0,
    denominator: float | None = None,
) -> torch.Tensor:
    """Token-normalized CTC: sum over the batch / total target tokens. log_probs [B, T, V].
    `denominator` replaces the batch's token count (a micro-batch normalized by its whole batch's
    count, so the micro-batch losses add up to the whole batch's loss)."""
    target_lengths = torch.tensor([len(t) for t in targets], device=log_probs.device)
    flat = (
        torch.cat(targets).to(log_probs.device)
        if targets and sum(len(t) for t in targets)
        else torch.zeros(0, dtype=torch.long, device=log_probs.device)
    )
    loss = F.ctc_loss(
        log_probs.float().transpose(0, 1),
        flat,
        lengths,
        target_lengths,
        blank=blank,
        reduction="sum",
        zero_infinity=True,
    )
    return loss / (target_lengths.sum().clamp_min(1) if denominator is None else denominator)


def consistency_loss(
    log_probs_a: torch.Tensor, log_probs_b: torch.Tensor, valid: torch.Tensor, denominator: float | None = None
) -> torch.Tensor:
    """CR-CTC (arXiv 2410.05101): symmetric KL between the frame posteriors of two augmented
    views, each direction against a stop-gradient target. Mean over valid frames (or the sum over
    / `denominator`, as in ctc_loss)."""
    a, b = log_probs_a.float(), log_probs_b.float()
    kl_ab = F.kl_div(b, a.detach(), log_target=True, reduction="none").sum(-1)  # KL(sg(a) || b)
    kl_ba = F.kl_div(a, b.detach(), log_target=True, reduction="none").sum(-1)
    per_frame = 0.5 * (kl_ab + kl_ba)
    return (per_frame * valid).sum() / (valid.sum().clamp_min(1) if denominator is None else denominator)


def rnnt_loss_from_logprobs(
    blank_lp: torch.Tensor,
    label_lp: torch.Tensor,
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    """Per-utterance RNN-T negative log-likelihood.

    blank_lp [B, T, U+1]: log p(blank | t, u); label_lp [B, T, U]: log p(y_{u+1} | t, u).
    alpha[t, u] = logaddexp(alpha[t-1, u] + blank[t-1, u], alpha[t, u-1] + label[t, u-1]).
    Along u this is a linear recursion: alpha[t, u] = C[u] + logcumsumexp_k(a[k] - C[k]) with
    a[k] = alpha[t-1, k] + blank[t-1, k] and C = cumulative label log-probs at row t.
    """
    batch, frames, u1 = blank_lp.shape
    blank_lp, label_lp = blank_lp.float(), label_lp.float()
    zero = blank_lp.new_zeros(batch, 1)
    neg_inf = blank_lp.new_full((batch, u1 - 1), float("-inf"))
    rows = []
    previous = None
    for t in range(frames):
        cumulative = torch.cat([zero, torch.cumsum(label_lp[:, t], dim=-1)], dim=-1)
        if previous is None:
            a = torch.cat([zero, neg_inf], dim=-1)
        else:
            a = previous + blank_lp[:, t - 1]
        alpha = cumulative + torch.logcumsumexp(a - cumulative, dim=-1)
        rows.append(alpha)
        previous = alpha
    alphas = torch.stack(rows, dim=1)  # [B, T, U+1]
    index_b = torch.arange(batch, device=blank_lp.device)
    t_last = (logit_lengths - 1).long()
    u_last = target_lengths.long()
    final = alphas[index_b, t_last, u_last] + blank_lp[index_b, t_last, u_last]
    return -final


def gather_rnnt_logprobs(
    log_probs: torch.Tensor, targets: torch.Tensor, blank: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full log-probs [B, T, U+1, V] + padded targets [B, U] -> (blank_lp, label_lp)."""
    blank_lp = log_probs[..., blank]
    index = targets.long()[:, None, :, None].expand(-1, log_probs.shape[1], -1, 1)
    label_lp = log_probs[:, :, :-1, :].gather(-1, index).squeeze(-1)
    return blank_lp, label_lp


class TransducerLoss(Protocol):
    """Seam for RNN-T now and TDT later: per-utterance NLL from gathered log-probabilities."""

    def __call__(
        self,
        blank_lp: torch.Tensor,
        label_lp: torch.Tensor,
        logit_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor: ...


def make_transducer_loss(kind: str = "rnnt") -> TransducerLoss:
    if kind == "rnnt":
        return rnnt_loss_from_logprobs
    if kind == "tdt":
        raise NotImplementedError(
            "TDT needs duration logits and NeMo's TDT recursion (nemo.collections.asr); "
            "not installed in this environment. Use kind='rnnt'."
        )
    raise ValueError(kind)
