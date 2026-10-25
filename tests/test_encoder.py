"""The staged call path reproduces Gemma4AudioModel.forward; adapters never touch base weights."""

import torch

from indicear.encoder import AdapterSet, StagedTower, lengths_to_valid


def full_forward_layers(tower, feats, mask):
    captured = {}
    handles = [layer.register_forward_hook(lambda _m, _i, o, k=i: captured.__setitem__(k, o))
               for i, layer in enumerate(tower.layers)]
    with torch.no_grad():
        output = tower(feats, mask)
    for handle in handles:
        handle.remove()
    return captured, output


def test_staged_matches_full_forward(tiny_tower):
    torch.manual_seed(1)
    feats = torch.randn(3, 400, 128)
    mask = torch.arange(400)[None] < torch.tensor([400, 260, 97])[:, None]
    captured, output = full_forward_layers(tiny_tower, feats, mask)
    staged = StagedTower(tiny_tower, 6)
    with torch.no_grad():
        h6, valid = staged.lower(feats, mask)
        h12 = staged.upper(h6, valid)
    assert torch.equal(valid, output.attention_mask)
    assert torch.equal(h6, captured[5])
    assert torch.equal(h12, captured[11])
    assert torch.equal(tiny_tower.output_proj(h12), output.last_hidden_state)


def test_per_utterance_cache_is_padding_independent(tiny_tower):
    torch.manual_seed(2)
    feats = torch.randn(2, 400, 128)
    mask = torch.arange(400)[None] < torch.tensor([400, 97])[:, None]
    staged = StagedTower(tiny_tower, 6)
    with torch.no_grad():
        h6, valid = staged.lower(feats, mask)
        h12 = staged.upper(h6, valid)
        n = int(valid[1].sum())
        alone, alone_valid = staged.lower(feats[1:2, :97], mask[1:2, :97])
        assert int(alone_valid.sum()) == n
        assert torch.allclose(alone[0, :n], h6[1, :n], atol=1e-5)
        repadded = torch.zeros(1, 60, h6.shape[-1])
        repadded[0, :n] = alone[0, :n]
        upper = staged.upper(repadded, lengths_to_valid(torch.tensor([n]), 60))
        assert torch.allclose(upper[0, :n], h12[1, :n], atol=1e-5)


def test_adapters_start_as_identity_train_separately_and_detach(tiny_tower, tmp_path):
    staged = StagedTower(tiny_tower, 6)
    before = {k: v.clone() for k, v in tiny_tower.state_dict().items()}
    h6 = torch.randn(2, 30, 64)
    valid = lengths_to_valid(torch.tensor([30, 20]), 30)
    with torch.no_grad():
        base = staged.upper(h6, valid)
    adapters = AdapterSet(64, rank=8).attach(tiny_tower)
    with torch.no_grad():
        assert torch.equal(staged.upper(h6, valid), base)
    optimizer = torch.optim.SGD(adapters.parameters(), lr=0.5)
    for _ in range(2):
        staged.upper(h6, valid).pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad()
    with torch.no_grad():
        assert not torch.allclose(staged.upper(h6, valid), base)
    for key, value in tiny_tower.state_dict().items():
        assert torch.equal(value, before[key]), key  # base weights byte-identical
    adapters.save(tmp_path / "adapters.safetensors")
    other = AdapterSet(64, rank=8)
    other.load(tmp_path / "adapters.safetensors")
    adapters.detach()
    with torch.no_grad():
        assert torch.equal(staged.upper(h6, valid), base)
