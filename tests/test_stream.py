"""Streaming path: interleave, shuffle buffer, bounded shard cache, exact resume, decoded batches."""

import json
import math
import shutil
from collections import Counter
from pathlib import Path

import pytest
import torch
from data_fixtures import FakeHub, mucs_tarball, write_parquet

from indicear.common import write_json
from indicear.mixture import LeakError
from indicear.stream import StreamLoader, build_planner, prepare_fixed

TEXTS_HI = ["मैं घर जा रहा हूँ", "आज मौसम अच्छा है", "हम office जा रहे हैं", "यह किताब नई है"]
TEXTS_EN = ["HELLO WORLD", "GOOD MORNING", "SEE YOU SOON", "THANK YOU"]


def kathbath_shard(path, start, speakers, n=10):
    rows = [{"seconds": 1.0 + (i % 4) * 0.5, "fname": f"8444{start + i:010d}-{speakers[i % len(speakers)]}-f.m4a",
             "text": TEXTS_HI[i % 4], "lang": "hindi", "duration": 1.0 + (i % 4) * 0.5, "gender": "female",
             "speaker_id": int(speakers[i % len(speakers)])} for i in range(n)]
    return write_parquet(path, rows, "audio_filepath")


def libri_shard(path, start, n=10):
    rows = [{"seconds": 1.5 + (i % 3) * 0.5, "file": "x", "text": TEXTS_EN[i % 4], "speaker_id": 10 + (start + i) % 5,
             "chapter_id": 1, "id": f"{10 + (start + i) % 5}-1-{start + i:04d}"} for i in range(n)]
    return write_parquet(path, rows)


@pytest.fixture
def hub(tmp_path):
    shards = {
        ("kathbath", "hi", "train"): [kathbath_shard(tmp_path / f"kb{i}.parquet", 100 * i, [str(100 + i), str(200 + i)])
                                      for i in range(3)],
        ("librispeech", "en", "train.clean.360"): [libri_shard(tmp_path / f"ls{i}.parquet", 100 * i) for i in range(2)],
    }
    return FakeHub(shards)


def fetcher(hub):
    def fetch(shard, directory):
        hub.downloads.append(shard["path"])
        target = Path(directory) / shard["path"].replace("/", "_")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shard["_local"], target)
        return target

    return fetch


MIXTURE = {
    "name": "tiny-stream", "seed": 0, "min_seconds": 0.5, "max_seconds": 20.0,
    "sampling": {"alpha": 0.5, "max_share": {"hi": 0.7}},
    "train": [{"source": "kathbath", "lang": "hi", "split": "train", "hours": 1},
              {"source": "librispeech", "lang": "en", "split": "train.clean.360", "hours": 1}],
    "eval": [],
}


def setup(root, hub, exclude=None, **kwargs):
    write_json(Path(root) / "manifests" / "stream_exclude.json",
               exclude or {"speaker": [], "group": [], "key": [], "audio": []})
    options = {"buffer_seconds": 12.0, "cache_gb": 1.0, "log": lambda m: None, **kwargs}
    return build_planner(root, MIXTURE, frame_budget=200, lister=hub.lister, fetch=fetcher(hub), **options)


def keys(rows):
    return [r["key"] for r in rows]


def test_resume_mid_stream_continues_with_the_same_keys(tmp_path, hub):
    first = setup(tmp_path / "a", hub)
    batches, states = [], []
    for _ in range(16):
        rows, state = first.planner.next_batch()
        batches.append(keys(rows))
        states.append(json.loads(json.dumps(state)))  # the checkpoint is plain JSON
    first.cache.close()
    for cut in (3, 9):  # kill after batch `cut`, resume on a fresh "VM" (new cache directory)
        second = setup(tmp_path / f"b{cut}", hub)
        second.planner.load_state(states[cut])
        resumed = [keys(second.planner.next_batch()[0]) for _ in range(16 - cut - 1)]
        second.cache.close()
        assert resumed == batches[cut + 1:]
    seen = Counter(k for batch in batches for k in batch)
    assert max(seen.values()) <= 2  # 16 batches cover about two passes of the 50-row pools
    assert {k.split("/")[0] for k in seen} == {"kathbath", "librispeech"}


def test_readahead_gives_the_same_batches_and_states(tmp_path, hub):
    def run(root, readahead, count=16, state=None):
        plan = setup(root, hub, readahead=readahead)
        if state is not None:
            plan.planner.load_state(state)
        out = []
        for _ in range(count):
            rows, st = plan.planner.next_batch()
            out.append((keys(rows), json.loads(json.dumps(st))))
        plan.close()
        return out

    plain = run(tmp_path / "plain", 0)
    for depth in (1, 3, 64):
        assert run(tmp_path / f"ra{depth}", depth) == plain
    # A checkpoint written with read-ahead resumes the same with or without it (and vice versa).
    ahead = run(tmp_path / "ahead", 3)
    for cut in (3, 9):
        assert run(tmp_path / f"r{cut}", 0, 16 - cut - 1, ahead[cut][1]) == plain[cut + 1:]
        assert run(tmp_path / f"p{cut}", 3, 16 - cut - 1, plain[cut][1]) == plain[cut + 1:]


def test_readahead_reports_reader_errors_and_stops_its_threads(tmp_path, hub):
    import threading

    plan = setup(tmp_path, hub, readahead=4)
    stream = plan.planner.by_name["kathbath/hi"]
    original = stream._sync_next
    calls = Counter()

    def failing():
        calls["n"] += 1
        if calls["n"] > 2:
            raise OSError("shard read failed")
        return original()

    stream._sync_next = failing
    stream.next()
    stream.next()
    with pytest.raises(OSError, match="shard read failed"):
        stream.next()
    assert stream.state() == {"epoch": 0, "pos": 0, "row": 2, "dry": 0}  # after the two rows handed out
    assert plan.planner.by_name["librispeech/en"].next() is not None  # the other streams still work
    plan.close()
    assert not [t for t in threading.enumerate() if t.name.startswith("readahead-")]


@pytest.mark.parametrize("readahead", [0, 4])
def test_a_stream_whose_filters_reject_every_row_is_stopped(tmp_path, hub, readahead):
    """Every Kathbath speaker excluded: after one full pass of its pool the stream is marked dead
    (instead of looping and re-downloading its shards forever) and the other streams continue."""
    speakers = [f"kathbath:{100 + i}" for i in range(3)] + [f"kathbath:{200 + i}" for i in range(3)]
    logs = []
    plan = setup(tmp_path, hub, exclude={"speaker": speakers, "group": [], "key": [], "audio": []},
                 readahead=readahead, log=logs.append)
    rows = [r for _ in range(12) for r in plan.planner.next_batch()[0]]
    assert rows and {r["stream"] for r in rows} == {"librispeech/en"}
    assert plan.planner.dead == {0}
    assert any("kathbath/hi: no row accepted in a full pass" in m for m in logs)
    plan.close()


def test_zero_streams_are_never_drawn_and_old_states_still_load(tmp_path, hub):
    first = setup(tmp_path / "a", hub)
    _, state = first.planner.next_batch()
    first.close()
    for s in state["streams"].values():  # a state written before the dry-shard counter existed
        s.pop("dry")
    plan = setup(tmp_path / "b", hub, zero_streams=["kathbath/hi"])
    assert plan.planner.probabilities()["kathbath/hi"] == 0.0
    plan.planner.load_state(state)
    rows = [r for _ in range(8) for r in plan.planner.next_batch()[0]]
    buffered = {tuple(ref[1:]) for ref in state["buffer"] if ref[0] == "kathbath/hi"}
    drawn = {tuple(r["ref"]) for r in rows if r["stream"] == "kathbath/hi"}
    assert drawn <= buffered  # only rows already in the restored buffer, never new draws
    plan.close()
    with pytest.raises(SystemExit, match="no such stream"):
        setup(tmp_path / "c", hub, zero_streams=["kathbath/xx"])
    scaled = setup(tmp_path / "d", hub, stream_scale={"kathbath/hi": 3.0}).planner.probabilities()
    base = setup(tmp_path / "e", hub).planner.probabilities()
    ratio = lambda p: p["kathbath/hi"] / p["librispeech/en"]
    assert ratio(scaled) == pytest.approx(3 * ratio(base))


def test_frame_budget_and_duration_buckets(tmp_path, hub):
    plan = setup(tmp_path, hub).planner
    for _ in range(10):
        rows, _ = plan.next_batch()
        longest = max(math.ceil(r["seconds"] * 25) for r in rows)
        assert len(rows) == 1 or len(rows) * longest <= 200
    assert plan.report()["probabilities"]["kathbath/hi"] == pytest.approx(0.5, abs=0.01)  # alpha 0.5, equal hours


def test_temperature_interleave_follows_the_shares(tmp_path, hub):
    mixture = {**MIXTURE, "train": [{**MIXTURE["train"][0], "hours": 4}, MIXTURE["train"][1]]}
    write_json(tmp_path / "manifests" / "stream_exclude.json", {"speaker": [], "group": [], "key": [], "audio": []})
    setup_ = build_planner(tmp_path, mixture, frame_budget=200, buffer_seconds=400.0, lister=hub.lister,
                           fetch=fetcher(hub), log=lambda m: None)
    probs = setup_.planner.probabilities()
    assert probs["kathbath/hi"] == pytest.approx(2 / 3)  # sqrt(4) : sqrt(1)
    setup_.planner._fill()
    drawn = Counter(r["stream"] for r in setup_.planner.buffer)
    assert drawn["kathbath/hi"] / sum(drawn.values()) == pytest.approx(2 / 3, abs=0.12)
    setup_.cache.close()


def test_shard_cache_is_bounded_and_deletes_consumed_shards(tmp_path, hub):
    size = max(Path(p).stat().st_size for paths in hub.shards.values() for p in paths)
    plan = setup(tmp_path, hub, cache_gb=2.5 * size / 2**30)
    peak_files = 0
    for _ in range(30):
        plan.planner.next_batch()
        peak_files = max(peak_files, len(list((tmp_path / "stream-cache").glob("*.parquet"))))
    stats = plan.cache.stats()
    assert stats["peak_gb"] * 2**30 <= 3.5 * size  # the bound, overshot by at most one shard
    assert peak_files <= 3
    plan.cache.close()
    assert not list((tmp_path / "stream-cache").glob("*.parquet"))


def test_exclusions_and_leak_guard(tmp_path, hub):
    exclude = {"speaker": ["kathbath:100"], "group": [], "key": ["librispeech/en/train.clean.360/10-1-0000"],
               "audio": []}
    plan = setup(tmp_path, hub, exclude=exclude).planner
    rows = [r for _ in range(20) for r in plan.next_batch()[0]]
    assert "kathbath:100" not in {r["speaker"] for r in rows}
    assert "librispeech/en/train.clean.360/10-1-0000" not in keys(rows)
    plan.eval_index["speaker"].add(rows[0]["speaker"])
    with pytest.raises(LeakError):
        plan._check_leak(rows[0])


def test_loader_yields_gpu_feature_batches_with_lid_labels(tmp_path, hub):
    plan = setup(tmp_path, hub, augment_prob=1.0)
    calls = []

    def transform(audio, rng, item):
        calls.append(item["view"])
        return audio * 0.5

    loader = StreamLoader(plan.planner, transform=transform, workers=2, prefetch=2, cache=plan.cache,
                          log=lambda m: None)
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) == 4:
            break
    plan.cache.close()
    for batch in batches:
        assert batch["waveforms"].dtype == torch.float32 and batch["lengths"].dtype == torch.int64
        assert batch["waveforms"].shape[0] == len(batch["items"]) == len(batch["lengths"])
        for item, length in zip(batch["items"], batch["lengths"].tolist(), strict=True):
            assert abs(length / 16000 - item["seconds"]) < 0.01  # resampled from 8 kHz to 16 kHz
            assert len(item["word_langs"]) == len(item["text"].split())
            assert item["lang"] in ("hi", "en") and "audio_bytes" not in item
    assert calls and all(v > 0 for v in calls)
    assert loader.state == batches[-1]["state"]
    office = [i for b in batches for i in b["items"] if "office" in i["text"]]
    assert all(i["word_langs"] == ["hi", "en", "hi", "hi", "hi"] and i["code_mixed"] for i in office)


def test_eval_audio_duplicates_are_dropped_at_decode(tmp_path, hub):
    from indicear.stream import decode_batch

    plan = setup(tmp_path, hub)
    rows, state = plan.planner.next_batch()
    plan.cache.close()
    clean = decode_batch(rows, state, None, set())
    from indicear.ingest import write_flac

    digest = write_flac(tmp_path / "x.flac", clean["waveforms"][0, : clean["lengths"][0]].numpy())[0]
    again = decode_batch(rows, state, None, {digest})
    assert again["dropped"] == 1 and len(again["items"]) == len(rows) - 1


def test_parallel_row_decode_matches_serial(tmp_path, hub):
    from concurrent.futures import ThreadPoolExecutor

    from indicear.augment import AugmentPolicy
    from indicear.stream import decode_batch

    plan = setup(tmp_path, hub, augment_prob=0.7)
    batches = [plan.planner.next_batch() for _ in range(3)]
    plan.close()
    policy = AugmentPolicy(codecs=("tel8k",))
    with ThreadPoolExecutor(max_workers=6) as pool:
        for rows, state in batches:
            serial = decode_batch(rows, state, policy, set(), seed=3)
            parallel = decode_batch(rows, state, policy, set(), seed=3, pool=pool)
            assert keys(serial["items"]) == keys(parallel["items"])
            assert [i.get("augment") for i in serial["items"]] == [i.get("augment") for i in parallel["items"]]
            assert torch.equal(serial["waveforms"], parallel["waveforms"])
            assert torch.equal(serial["lengths"], parallel["lengths"])


def test_prepare_fixed_dev_eval_and_local_streams(tmp_path, hub, monkeypatch):
    import indicear.ingest

    valid = kathbath_shard(tmp_path / "kbv.parquet", 900, ["100", "999"], n=4)
    hub.shards[("kathbath", "hi", "valid")] = [valid]
    archive = mucs_tarball(tmp_path / "Hindi-English_train.tar.gz", split="train")
    monkeypatch.setattr(indicear.ingest, "list_hf_shards", hub.lister)
    monkeypatch.setattr(indicear.ingest, "download_hf", hub.downloader)
    monkeypatch.setattr(indicear.ingest, "download_url", lambda spec, dest, guard=None, label="": archive)
    mixture = {**MIXTURE, "dev": {"fraction": 0.5, "max_hours_per_lang": 0.002},
               "sampling": {"alpha": 0.5, "min_share": {"cs": 0.3}},
               "train": MIXTURE["train"] + [{"source": "mucs", "lang": "hi", "split": "train", "hours": 1}],
               "eval": [{"name": "kb_valid", "source": "kathbath", "lang": "hi", "split": "valid"}]}
    root = tmp_path / "root"
    report = prepare_fixed(root, mixture, {"mucs"}, decode_workers=2, log=lambda m: None)
    local = (root / "manifests" / "local_train.jsonl").read_text().splitlines()
    assert report["dev_hours"] > 0 and local and not report["unavailable"]
    exclude = json.loads((root / "manifests" / "stream_exclude.json").read_text())
    assert "kathbath:100" in exclude["speaker"]  # the eval speaker
    dev = [json.loads(line) for line in (root / "manifests" / "dev.jsonl").read_text().splitlines()]
    assert {r["key"] for r in dev} <= set(exclude["key"])
    plan = build_planner(root, mixture, frame_budget=200, allow_review={"mucs"}, buffer_seconds=12.0,
                         lister=hub.lister, fetch=fetcher(hub), log=lambda m: None)
    assert "mucs/hi/local" in plan.planner.probabilities()
    rows = [r for _ in range(25) for r in plan.planner.next_batch()[0]]
    plan.cache.close()
    dev_speakers = {r["speaker"] for r in dev if r.get("speaker")}
    assert not ({r["speaker"] for r in rows if r.get("speaker")} & (dev_speakers | {"kathbath:100"}))
    assert not ({r["key"] for r in rows} & {r["key"] for r in dev})
    assert any(r["source"] == "mucs" for r in rows)
    assert prepare_fixed(root, mixture, {"mucs"}, log=lambda m: None) == report  # finished: not redone


def test_ablate_prep_stream_end_to_end(tmp_path, hub, monkeypatch):
    import indicear.ingest
    import indicear.stream
    from indicear import ablate

    hub.shards[("kathbath", "hi", "valid")] = [kathbath_shard(tmp_path / "kbv.parquet", 900, ["100", "999"], n=4)]
    for module in (indicear.ingest, indicear.stream):
        monkeypatch.setattr(module, "list_hf_shards", hub.lister)
    monkeypatch.setattr(indicear.ingest, "download_hf", hub.downloader)
    monkeypatch.setattr(indicear.stream, "hub_fetch", fetcher(hub))
    monkeypatch.setattr(indicear.stream, "open_remote", lambda shard: open(shard["_local"], "rb"))  # noqa: SIM115
    import soundfile as sf

    import indicear.data

    def fake_fleurs(root, locale, split):  # one synthetic FLEURS dev clip per language, no download
        wav = tmp_path / f"fleurs_{locale}.wav"
        sf.write(wav, torch.zeros(16000).numpy(), 16000)
        raw = {"hi_in": "नमस्ते दुनिया", "en_us": "hello world"}[locale]
        return [{"sentence_id": "1", "file": wav.name, "audio": str(wav), "raw": raw}]

    monkeypatch.setattr(indicear.data, "fetch_fleurs", fake_fleurs)
    mixture = {**MIXTURE, "dev": {"fraction": 0.5, "max_hours_per_lang": 0.002},
               "eval": [{"name": "kb_valid", "source": "kathbath", "lang": "hi", "split": "valid"}]}
    path = tmp_path / "stream.json"
    path.write_text(json.dumps(mixture))
    root = tmp_path / "root"
    ablate.main(["--root", str(root), "prep", "--mixture", str(path), "--stream", "--negatives", "2",
                 "--negatives-dev", "1", "--cs-per-pair", "1", "--decode-workers", "2"])
    report = json.loads((root / "prep_report.json").read_text())
    assert report["stream"] and report["plan"]["probabilities"] and report["label_rates"]
    texts = (root / "manifests" / "train_texts.jsonl").read_text().splitlines()
    assert len(texts) == 50  # every pool transcript, read without audio
    assert not (root / "manifests" / "train.jsonl").exists()  # no training audio was materialized
    dev = (root / "manifests" / "dev.jsonl").read_text().splitlines()
    evals = (root / "manifests" / "eval_kb_valid.jsonl").read_text().splitlines()
    flacs = [p for d in ("kathbath", "librispeech") for p in (root / "audio" / d).rglob("*.flac")]
    assert len(flacs) == len(dev) + len(evals)  # only the fixed sets are on disk
    assert not list((root / "downloads").rglob("*.parquet")) and not list((root / "stream-cache").rglob("*.parquet"))
    assert (root / "tok" / "aggregate.json").exists()
    fleurs = [json.loads(line) for line in (root / "manifests" / "eval_fleurs_dev.jsonl").read_text().splitlines()]
    assert sorted(r["lang"] for r in fleurs) == ["en", "hi"]


@pytest.mark.parametrize("which", ["policy", "codec"])
def test_worker_augmentation_is_deterministic_under_resume(tmp_path, hub, which):
    """Rows augmented in the stream workers (the CPU AugmentPolicy, or the GPU path's CodecAugment)
    come out identical after the planner is restored from a batch's state."""
    from indicear.augment import AugmentPolicy, CodecAugment

    def transform():
        return AugmentPolicy(codecs=("tel8k",)) if which == "policy" else CodecAugment(prob=0.5, codecs=("opus",))

    plan = setup(tmp_path / "a", hub, augment_prob=0.5)
    stream = iter(StreamLoader(plan.planner, transform=transform(), workers=2, log=lambda m: None))
    first = [next(stream) for _ in range(6)]
    stream.close()
    plan.cache.close()
    again_plan = setup(tmp_path / "b", hub, augment_prob=0.5)
    again_plan.planner.load_state(first[2]["state"])
    again = iter(StreamLoader(again_plan.planner, transform=transform(), workers=3, log=lambda m: None))
    replay = [next(again) for _ in range(3)]
    again.close()
    again_plan.cache.close()
    for x, y in zip(first[3:], replay, strict=True):
        assert keys(x["items"]) == keys(y["items"])
        assert torch.equal(x["waveforms"], y["waveforms"])
        mark = "augment" if which == "policy" else "codec"
        assert [i.get(mark) for i in x["items"]] == [i.get(mark) for i in y["items"]]
    rows = [i for b in first for i in b["items"]]
    if which == "policy":
        assert any(i.get("augment") for i in rows) and any(not i.get("augment") for i in rows)


def test_restore_on_a_host_whose_filters_reject_buffered_rows(tmp_path, hub):
    """A prep root rebuilt on another host may exclude other dev speakers: strict restore stops,
    lenient restore leaves those buffered rows out, logs them and continues from the same stream
    positions (the Mac resume of the pilot)."""
    first = setup(tmp_path / "a", hub)
    for _ in range(4):
        _, state = first.planner.next_batch()
    state = json.loads(json.dumps(state))
    first.cache.close()
    buffered = [ref for ref in state["buffer"] if ref[0] == "kathbath/hi"]
    assert buffered
    rows = first.planner.buffer
    speaker = next(r["speaker"] for r in rows if r["stream"] == "kathbath/hi")
    exclude = {"speaker": [speaker], "group": [], "key": [], "audio": []}
    strict = setup(tmp_path / "b", hub, exclude=exclude)
    with pytest.raises(RuntimeError, match="no longer reads back"):
        strict.planner.load_state(state)
    strict.cache.close()
    messages = []
    lenient = setup(tmp_path / "c", hub, exclude=exclude, log=messages.append)
    lenient.planner.strict_restore = False
    lenient.planner.load_state(state)
    dropped = sum(1 for r in rows if r.get("speaker") == speaker)
    assert len(lenient.planner.buffer) == len(state["buffer"]) - dropped > 0
    assert all(r.get("speaker") != speaker for r in lenient.planner.buffer)
    assert any(f"{dropped} of {len(state['buffer'])} buffered rows" in m and "excluded_speaker" in m for m in messages)
    for _ in range(3):  # and it keeps streaming, never yielding the newly excluded speaker
        batch, _ = lenient.planner.next_batch()
        assert all(r.get("speaker") != speaker for r in batch)
    lenient.cache.close()


def test_keep_shards_downloads_the_pool_once(tmp_path, hub):
    """keep_shards (the Mac): consumed shards stay on disk across epochs and restarts, and
    download_pool fetches the whole pool ahead of training."""
    from indicear.stream import download_pool, hub_fetch, local_copy

    plan = setup(tmp_path, hub, keep_shards=True)
    pool = [shard for stream in plan.planner.streams for shard in getattr(stream, "pool", ())]
    messages = []
    assert download_pool(plan, workers=2, log=messages.append)["shards"] == len(pool) == 5
    assert sorted(hub.downloads) == sorted(s["path"] for s in pool)
    for _ in range(30):  # several stream epochs: every open finds its local file
        plan.planner.next_batch()
    plan.cache.close()
    files = list((tmp_path / "stream-cache").glob("*.parquet"))
    assert len(files) == 5  # nothing deleted, on consumption or on close
    real = dict(pool[0], repo="org/data", path="x/shard.parquet", size=4)
    target = local_copy(real, tmp_path / "hubcache")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"done")
    assert hub_fetch(real, tmp_path / "hubcache") == target  # complete copy: no network
