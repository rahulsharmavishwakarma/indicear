import json

import numpy as np
import pytest
import torch

from indicear.cache import Layer6Cache, dequantize, plan_parts, quantize_int8, to_storage, write_part


def test_int8_roundtrip_error_small():
    rows = torch.randn(50, 1024) * 3
    q, scale = quantize_int8(rows)
    assert q.dtype == torch.int8
    error = (dequantize(q, scale, torch.float32) - rows).abs().max(-1).values
    assert torch.all(error <= rows.abs().max(-1).values / 127 * 0.51 + 1e-6)


def write_cache(directory, storage, utterances):
    meta = {"layer": 6, "dim": 16, "storage": storage, "frame_rate": 25, "encoder": "test", "parts": 2}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cache.json").write_text(json.dumps(meta))
    for part, keys in enumerate((["a", "b"], ["c"])):
        rows = []
        for key in keys:
            data, scale = to_storage(utterances[key], storage)
            rows.append((key, data, scale))
        write_part(directory, part, rows, storage)


@pytest.mark.parametrize("storage", ["bf16", "int8"])
@pytest.mark.parametrize("placement", ["auto", "mmap"])
def test_write_load_batch(tmp_path, storage, placement):
    torch.manual_seed(0)
    utterances = {"a": torch.randn(7, 16), "b": torch.randn(3, 16), "c": torch.randn(5, 16)}
    write_cache(tmp_path, storage, utterances)
    cache = Layer6Cache.load(tmp_path, device="cpu", log=lambda _m: None, placement=placement)
    hidden, lengths = cache.batch(["c", "a", "b"], "cpu", torch.float32)
    assert hidden.shape == (3, 7, 16) and lengths.tolist() == [5, 7, 3]
    tolerance = 0.05 if storage == "int8" else 0.02
    for row, key in enumerate(["c", "a", "b"]):
        n = utterances[key].shape[0]
        assert torch.allclose(hidden[row, :n], utterances[key], atol=tolerance * utterances[key].abs().max())
        assert torch.all(hidden[row, n:] == 0)


def test_incomplete_part_detected_and_partial_selection(tmp_path):
    utterances = {"a": torch.randn(7, 16), "b": torch.randn(3, 16), "c": torch.randn(5, 16)}
    write_cache(tmp_path, "bf16", utterances)
    subset = Layer6Cache.load(tmp_path, device="cpu", keys={"b"}, log=lambda _m: None)
    assert set(subset.index) == {"b"} and subset.data.shape[0] == 3
    (tmp_path / "part-00001.jsonl").unlink()  # the index is the completion marker
    with pytest.raises(FileNotFoundError):
        Layer6Cache.load(tmp_path, device="cpu", log=lambda _m: None)


def test_plan_parts_is_deterministic():
    items = [{"key": str(i), "seconds": 4.0} for i in range(10)]
    parts = plan_parts(items, 300, lambda i: 4 * 25 + 1)
    assert [len(p) for p in parts] == [2] * 5
    assert plan_parts(items, 300, lambda i: 4 * 25 + 1) == parts
    assert np.isclose(sum(len(p) for p in parts), 10)
