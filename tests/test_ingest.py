import pytest
import soundfile as sf
from data_fixtures import FakeHub, kathbath_tars, mucs_tarball, write_parquet

from indicear.ingest import (
    BudgetExceeded,
    Context,
    DiskGuard,
    Task,
    run_parquet_task,
    run_tar_task,
    speaker_and_group,
)
from indicear.runtime.shards import read_marker
from indicear.sources import SOURCES

GIB = 2**30


def vaani_shard(path, n=6, seconds=3.0, offset=0):
    rows = [{"seconds": seconds, "transcript": f"यह ब्लैक {{black}} कलर {{color}} की कार {i + offset} है",
             "gender": "male", "state": "UP", "district": "Gorakhpur",
             "_path": f"IISc_VaaniProject_S_Uttarpradesh_Gorakhpur_{9000 + (i + offset) // 2}_x_{i + offset}.wav"}
            for i in range(n)]
    rows.append({"seconds": seconds, "transcript": "[unintelligible] नीला", "gender": "f", "state": "UP",
                 "district": "Gorakhpur", "_path": f"IISc_VaaniProject_S_U_G_77_x_{offset}u.wav"})
    return write_parquet(path, rows)


def context(root, free=100 * GIB, budget=None):
    return Context(root, DiskGuard(root, budget, reserve_bytes=0, free_bytes=lambda: free), decode_workers=2,
                   log=lambda m: None)


def test_parquet_task_caps_hours_resamples_and_resumes(tmp_path):
    hub = FakeHub({("vaani", "hi", "train"): [vaani_shard(tmp_path / f"s{i}.parquet", offset=10 * i) for i in range(3)]})
    task = Task("vaani", "hi", "train", hours=20.0 / 3600)  # 20 s: two shards of 18 s
    ctx = context(tmp_path / "root")
    result = run_parquet_task(task, ctx, hub.downloader, hub.lister)
    ctx.close()
    assert result.shards == 2 and len(hub.downloads) == 2
    assert 20.0 <= result.seconds < 24.0
    assert result.drops["unintelligible"] == 2
    row = result.rows[0]
    assert row["text"] == "यह black color की कार {} है".format(row["text"].split()[-2])
    assert row["code_mixed"] and row["glosses"] == 2 and row["group"].startswith("vaani:fn:")
    info = sf.info(row["audio"])
    assert info.samplerate == 16000 and info.channels == 1 and info.format == "FLAC"
    # Downloads are deleted after decoding; markers and rows stay.
    assert not list((tmp_path / "root" / "downloads").glob("*.parquet"))
    # Rerun: nothing is downloaded again and the result is identical.
    again = run_parquet_task(task, context(tmp_path / "root"), hub.downloader, hub.lister)
    assert len(hub.downloads) == 2 and [r["key"] for r in again.rows] == [r["key"] for r in result.rows]
    # A larger cap redoes only the shard that was cut, then continues.
    bigger = run_parquet_task(Task("vaani", "hi", "train", hours=1.0), context(tmp_path / "root"), hub.downloader,
                              hub.lister)
    assert bigger.shards == 3 and len(bigger.rows) == 18 and len(hub.downloads) == 4


def test_resume_after_interruption(tmp_path):
    shards = [vaani_shard(tmp_path / f"s{i}.parquet", offset=10 * i) for i in range(3)]
    hub = FakeHub({("vaani", "hi", "train"): shards})
    task = Task("vaani", "hi", "train", hours=1.0)
    order = [s["path"] for s in __import__("indicear.ingest", fromlist=["order_shards"]).order_shards(
        hub.lister(SOURCES["vaani"], "hi", "train", tmp_path), task, 0)]
    hub.fail_on = {order[1]}
    with pytest.raises(KeyboardInterrupt):
        run_parquet_task(task, context(tmp_path / "root"), hub.downloader, hub.lister)
    stems = list((tmp_path / "root" / "sources" / "vaani" / "hi" / "train").glob("*.done.json"))
    assert len(stems) == 1  # the first shard finished and kept its marker
    result = run_parquet_task(task, context(tmp_path / "root"), hub.downloader, hub.lister)
    assert result.shards == 3 and len(result.rows) == 18
    assert hub.downloads.count(order[0]) == 1  # the finished shard was not fetched again


def test_disk_budget_stops_cleanly_before_a_shard(tmp_path):
    hub = FakeHub({("vaani", "hi", "train"): [vaani_shard(tmp_path / f"s{i}.parquet", offset=10 * i) for i in range(2)]})
    root = tmp_path / "root"
    with pytest.raises(BudgetExceeded, match="disk budget"):
        run_parquet_task(Task("vaani", "hi", "train", hours=1.0), context(root, budget=1000), hub.downloader,
                         hub.lister)
    assert hub.downloads == []  # refused before downloading
    with pytest.raises(BudgetExceeded, match="free"):
        run_parquet_task(Task("vaani", "hi", "train", hours=1.0), context(root, free=10), hub.downloader, hub.lister)
    guard = DiskGuard(root, budget_bytes=100, reserve_bytes=0, free_bytes=lambda: 10**9)
    held = guard.acquire(60, "a")
    with pytest.raises(BudgetExceeded):
        guard.acquire(60, "b")  # reservations count against the budget
    guard.release(held, 50)
    assert guard.used == 50 and guard.acquire(40, "c") == 40


def test_peoples_speech_and_voxpopuli_schemas(tmp_path):
    ps = write_parquet(tmp_path / "ps.parquet", [
        {"seconds": 2.0, "id": f"forum_SLASH_rec_DOT_mp3_{i:05d}.flac", "duration_ms": 2000,
         "text": "we are starting late"} for i in range(3)])
    vp = write_parquet(tmp_path / "vp.parquet", [
        {"seconds": 2.0, "audio_id": f"2018-en_{i}", "raw_text": "These are not easy issues.",
         "normalized_text": "x", "speaker_id": "124", "is_gold_transcript": i != 1, "gender": "f", "accent": "None"}
        for i in range(3)])
    hub = FakeHub({("peoples_speech", "en", "train"): [ps], ("voxpopuli", "en", "train"): [vp]})
    ctx = context(tmp_path / "root")
    people = run_parquet_task(Task("peoples_speech", "en", "train", 1.0), ctx, hub.downloader, hub.lister)
    vox = run_parquet_task(Task("voxpopuli", "en", "train", 1.0), ctx, hub.downloader, hub.lister)
    ctx.close()
    assert len(people.rows) == 3 and people.rows[0]["group"] == "peoples_speech:rec:forum"
    assert people.rows[0]["speaker"] is None
    assert len(vox.rows) == 2 and vox.drops["not_gold"] == 1
    assert vox.rows[0]["speaker"] == "voxpopuli:124" and vox.rows[0]["text"] == "these are not easy issues"


def test_mucs_tarball_two_pass_segments(tmp_path):
    archive = mucs_tarball(tmp_path / "Hindi-English_test.tar.gz")
    fetched = []

    def fetch(spec, dest, guard, label):
        fetched.append(dest.name)
        return archive

    ctx = context(tmp_path / "root")
    result = run_tar_task(Task("mucs", "hi", "test", float("inf"), role="eval"), ctx, fetch)
    ctx.close()
    assert len(result.rows) == 6 and fetched == ["Hindi-English_test.tar.gz"]
    assert all(r["code_mixed"] for r in result.rows)
    assert {r["speaker"] for r in result.rows} == {"mucs:spk1", "mucs:spk2"}
    assert all(abs(r["seconds"] - 2.5) < 0.01 for r in result.rows)
    assert read_marker(tmp_path / "root" / "sources" / "mucs" / "hi" / "test" / "archive")["rows"] == 6


def test_kathbath_test_tars(tmp_path):
    transcripts, audio = kathbath_tars(tmp_path / "kb")

    def fetch(spec, dest, guard, label):
        return transcripts if "transcripts" in dest.name else audio

    ctx = context(tmp_path / "root")
    result = run_tar_task(Task("kathbath_test", "hi", "clean_test_known", float("inf"), role="eval"), ctx, fetch)
    ctx.close()
    assert len(result.rows) == 2
    assert {r["speaker"] for r in result.rows} == {"kathbath:930", "kathbath:931"}


def test_speaker_namespaces():
    kb = SOURCES["kathbath"]
    assert speaker_and_group(kb, {"speaker_id": 12, "fname": "844424930443097-930-f.m4a"}, "") == (
        "kathbath:930", "kathbath:930")
    sv = SOURCES["svarah"]
    assert speaker_and_group(sv, {}, "281474976884635_f3269_chunk_0.wav") == (
        None, "svarah:rec:281474976884635_f3269")


def test_schema_change_is_reported_not_silently_dropped(tmp_path, monkeypatch):
    import indicear.ingest

    bad = write_parquet(tmp_path / "bad.parquet", [{"seconds": 2.0, "text": "wrong column"}])
    hub = FakeHub({("vaani", "hi", "train"): [bad]})
    monkeypatch.setattr(indicear.ingest, "list_hf_shards", hub.lister)
    monkeypatch.setattr(indicear.ingest, "download_hf", hub.downloader)
    ctx = context(tmp_path / "root")
    [result] = indicear.ingest.run_tasks([Task("vaani", "hi", "train", 1.0)], ctx, workers=1)
    ctx.close()
    assert result.error and "transcript" in result.error and not result.rows
