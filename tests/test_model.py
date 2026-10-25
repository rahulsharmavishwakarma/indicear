import tempfile

import pytest
import torch

from indicear import phonetic
from indicear.encoder import StagedTower
from indicear.heads import FactoredTransducer, ScriptFactoredOutput
from indicear.losses import gather_rnnt_logprobs
from indicear.model import VARIANTS, DesignA

RANGES = [(4, 20), (20, 40), (40, 50)]


def test_factored_output_is_a_distribution_and_gather_matches():
    head = ScriptFactoredOutput(8, RANGES, 50)
    hidden = torch.randn(2, 5, 3, 8)
    full = head(hidden)
    assert torch.allclose(full.exp().sum(-1), torch.ones(2, 5, 3), atol=1e-5)
    targets = torch.tensor([[5, 45], [30, 1]])  # 1 = pad
    blank, label = head.gathered(hidden, targets)
    ref_blank, ref_label = gather_rnnt_logprobs(full, targets)
    assert torch.allclose(blank, ref_blank) and torch.allclose(label[0], ref_label[0], atol=1e-5)
    assert torch.allclose(label[1, :, 0], ref_label[1, :, 0], atol=1e-5)


def test_factored_transducer_loss_and_greedy():
    head = FactoredTransducer(16, RANGES, 50, pred_dim=8, joint_dim=8)
    enc = torch.randn(2, 7, 16)
    loss = head.loss(enc, torch.tensor([7, 5]), [torch.tensor([5, 45, 9]), torch.tensor([30])], pad=1)
    loss.backward()
    assert torch.isfinite(loss)
    hyps = head.greedy(enc, torch.tensor([7, 5]))
    assert len(hyps) == 2 and all(0 < t < 50 for h in hyps for t in h)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_variant_forward_backward_decode_save(tiny_tower, name):
    staged = StagedTower(tiny_tower, 6)
    torch.manual_seed(0)
    model = DesignA(staged, VARIANTS[name], 50, RANGES, len(phonetic.UNITS))
    try:
        model.train()
        h6 = torch.randn(3, 40, 64)
        lengths = torch.tensor([40, 30, 12])
        tokens = [torch.tensor([5, 25, 41, 7]), torch.tensor([30, 31]), torch.tensor([], dtype=torch.long)]
        units = [torch.tensor([3, 4, 5, 1, 6]), torch.tensor([7, 8]), torch.tensor([], dtype=torch.long)]
        out = model.losses(h6, lengths, tokens, units)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        assert all(p.grad is None for p in tiny_tower.parameters())
        assert ("phonetic" in out) == VARIANTS[name].phonetic
        assert ("consistency" in out) == VARIANTS[name].cr_ctc
        assert ("rnnt" in out) == VARIANTS[name].transducer
        model.eval()
        decoded = model.decode(h6, lengths)
        assert len(decoded["ctc"]) == 3
        if VARIANTS[name].phonetic:
            assert set(decoded["inter"]) == {"6", "9"}
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            state = {k: v.clone() for k, v in model.state_dict().items()}
            for p in model.parameters():
                torch.nn.init.zeros_(p) if p.requires_grad else None
            model.load(directory)
            for key, value in model.state_dict().items():
                assert torch.equal(value, state[key]), key
    finally:
        model.detach_adapters()


def test_hidden_specaugment_respects_lengths_and_budget():
    from indicear.heads import HiddenSpecAugment

    torch.manual_seed(0)
    hidden = torch.randn(3, 50, 32)
    lengths = torch.tensor([50, 20, 5])
    time_only = HiddenSpecAugment(time_masks_per_100=10, time_mask_max=4, feature_masks=0)
    out = time_only(hidden, lengths)
    masked_frames = (out != hidden).any(-1)  # [B, T]
    assert masked_frames.any()
    for row, length in enumerate(lengths.tolist()):
        assert not masked_frames[row, length:].any()  # never inside padding
        assert masked_frames[row].sum() <= round(0.10 * length) * 4
    nothing = HiddenSpecAugment(time_masks_per_100=0, feature_masks=0)
    assert torch.equal(nothing(hidden, lengths), hidden)
    features = HiddenSpecAugment(time_masks_per_100=0, feature_masks=2, feature_mask_max=8)(hidden, lengths)
    masked_dims = (features != hidden).any(1)  # [B, D]
    assert masked_dims.any() and masked_dims.sum(-1).max() <= 16
