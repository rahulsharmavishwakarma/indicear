import pytest
import torch

from indicear.losses import consistency_loss, ctc_loss, gather_rnnt_logprobs, make_transducer_loss, rnnt_loss_from_logprobs


def test_rnnt_matches_torchaudio_values_and_gradients():
    torchaudio = pytest.importorskip("torchaudio")
    torch.manual_seed(0)
    logits = torch.randn(3, 9, 5, 11, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(1, 11, (3, 4), dtype=torch.int32)
    t_len = torch.tensor([9, 6, 3], dtype=torch.int32)
    u_len = torch.tensor([4, 2, 0], dtype=torch.int32)
    reference = torchaudio.functional.rnnt_loss(logits.float(), targets, t_len, u_len, blank=0, reduction="none")
    blank, label = gather_rnnt_logprobs(logits.log_softmax(-1), targets)
    ours = rnnt_loss_from_logprobs(blank, label, t_len, u_len)
    assert torch.allclose(ours.detach().float(), reference.detach(), rtol=1e-5)
    (grad,) = torch.autograd.grad(ours.sum(), logits)
    check = logits.detach().float().requires_grad_()
    torchaudio.functional.rnnt_loss(check, targets, t_len, u_len, blank=0, reduction="none").sum().backward()
    assert torch.allclose(grad.float(), check.grad, atol=1e-5)


def test_rnnt_trivial_case():
    log_probs = torch.log(torch.tensor([[[[0.25, 0.75]]]]))  # T=1, U=0
    blank, label = gather_rnnt_logprobs(log_probs, torch.zeros(1, 0, dtype=torch.long))
    loss = rnnt_loss_from_logprobs(blank, label, torch.tensor([1]), torch.tensor([0]))
    assert loss.item() == pytest.approx(-torch.log(torch.tensor(0.25)).item())


def test_ctc_and_consistency():
    log_probs = torch.randn(2, 12, 7).log_softmax(-1)
    loss = ctc_loss(log_probs, torch.tensor([12, 8]), [torch.tensor([1, 2, 3]), torch.tensor([], dtype=torch.long)])
    assert torch.isfinite(loss)
    valid = torch.ones(2, 12)
    assert consistency_loss(log_probs, log_probs, valid).item() == pytest.approx(0.0, abs=1e-6)
    assert consistency_loss(log_probs, torch.randn(2, 12, 7).log_softmax(-1), valid).item() > 0


def test_tdt_is_behind_the_interface():
    assert make_transducer_loss("rnnt") is rnnt_loss_from_logprobs
    with pytest.raises(NotImplementedError):
        make_transducer_loss("tdt")
