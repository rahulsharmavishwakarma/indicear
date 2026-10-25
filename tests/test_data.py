import numpy as np
import pytest

from indicear.data import crossfade_concat, fetch_fleurs, synth_code_switch, with_views


def test_crossfade_length_and_continuity():
    a = np.ones(16000, dtype=np.float32)
    b = -np.ones(8000, dtype=np.float32)
    out = crossfade_concat(a, b, crossfade_ms=50)
    assert len(out) == len(a) + len(b) - 800
    assert out[0] == 1 and out[-1] == -1 and np.all(np.abs(np.diff(out)) < 0.01)


def test_synthetic_code_switch(tmp_path):
    import soundfile as sf

    def item(lang, script, text, name):
        path = tmp_path / f"{name}.flac"
        sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
        return {"lang": lang, "script": script, "text": text, "audio": str(path), "seconds": 1.0}

    pools = {"hi": [item("hi", "deva", "नमस्ते", "h")], "en": [item("en", "latn", "hello there", "e")]}
    items = synth_code_switch(pools, tmp_path / "cs", pairs=[("hi", "en")], per_pair=3)
    assert len(items) == 3
    for row in items:
        assert {s["script"] for s in row["spans"]} == {"deva", "latn"}
        assert row["text"] == f"{row['spans'][0]['text']} {row['spans'][1]['text']}"
        assert row["seconds"] == pytest.approx(2.0 - 0.05)


def test_views_and_test_split_guard(tmp_path):
    rows = with_views([{"key": "k", "seconds": 1.0}], 3)
    assert [r["key"] for r in rows] == ["k", "k#v1", "k#v2"] and [r["view"] for r in rows] == [0, 1, 2]
    with pytest.raises(ValueError):
        fetch_fleurs(tmp_path, "hi_in", "test")
