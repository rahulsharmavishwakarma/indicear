import json
from pathlib import Path

import pytest
from data_fixtures import FakeHub, write_parquet

from indicear import ablate
from indicear.mixture import (
    LeakError,
    apply_sample_weights,
    assemble,
    assert_no_leak,
    estimate,
    load_mixture,
    plan_tasks,
    remove_eval_overlap,
    split_dev,
    temperature_shares,
)
from indicear.runtime.batching import BucketFrameSampler


def row(key, lang="hi", speaker=None, group=None, sha=None, seconds=10.0, source="src", code_mixed=False):
    return {"key": key, "lang": lang, "speaker": speaker, "group": group, "audio_sha1": sha or key,
            "seconds": seconds, "source": source, "code_mixed": code_mixed}


def test_dev_split_is_speaker_disjoint_and_bounded():
    rows = [row(f"u{i}", speaker=f"s:{i % 20}") for i in range(200)] + [row(f"e{i}", "en", group=f"g:{i % 5}")
                                                                        for i in range(50)]
    train, dev = split_dev(rows, fraction=0.1, max_hours=0.1, seed=3)
    assert len(train) + len(dev) == 250
    assert not ({r["speaker"] or r["group"] for r in train} & {r["speaker"] or r["group"] for r in dev})
    dev_hi = sum(r["seconds"] for r in dev if r["lang"] == "hi")
    assert 0 < dev_hi <= 0.1 * 3600 + 100  # whole groups: at most one group over the target
    assert split_dev(rows, 0.1, 0.1, seed=3) == (train, dev)  # deterministic


def test_eval_overlap_is_removed_and_leaks_are_caught():
    evals = [row("ev1", speaker="kathbath:930"), row("ev2", group="vaani:fn:77"), row("ev3", sha="dup")]
    pool = [row("a", speaker="kathbath:930"), row("b", group="vaani:fn:77"), row("c", sha="dup"),
            row("d", speaker="kathbath:1"), row("e", group="vaani:fn:78")]
    drops = __import__("collections").Counter()
    kept = remove_eval_overlap(pool, evals, drops)
    assert [r["key"] for r in kept] == ["d", "e"]
    assert drops == {"eval_speaker:src": 1, "eval_group:src": 1, "eval_audio:src": 1}
    assert_no_leak(kept, evals)
    with pytest.raises(LeakError, match="speaker kathbath:930"):
        assert_no_leak(pool, evals)
    with pytest.raises(LeakError, match="audio dup"):
        assert_no_leak([row("x", sha="dup")], evals)


def test_temperature_shares_caps_floors_and_weights():
    hours = {"hi": 200, "ta": 150, "kn": 150, "en": 100, "cs": 10}
    plain = temperature_shares(hours, alpha=0.5)
    assert abs(sum(plain.values()) - 1) < 1e-9
    assert plain["hi"] / plain["en"] == pytest.approx((200 / 100) ** 0.5)
    capped = temperature_shares(hours, 0.5, max_share={"hi": 0.2}, min_share={"cs": 0.1})
    assert capped["hi"] == pytest.approx(0.2) and capped["cs"] == pytest.approx(0.1)
    assert abs(sum(capped.values()) - 1) < 1e-9
    assert capped["ta"] / capped["en"] == pytest.approx((150 / 100) ** 0.5)
    rows = [row(f"h{i}", "hi") for i in range(20)] + [row(f"c{i}", "hi", code_mixed=True) for i in range(2)]
    plan = apply_sample_weights(rows, {"hi": 0.5, "cs": 0.5})
    expected = {g: sum(r["seconds"] * r["sample_weight"] for r in rows if (r["code_mixed"] and g == "cs") or
                       (not r["code_mixed"] and g == "hi")) / 3600 for g in ("hi", "cs")}
    assert expected["hi"] == pytest.approx(expected["cs"]) == pytest.approx(plan["cs"]["epoch_hours"], abs=0.01)


def test_sampler_weights_draw_expected_counts_deterministically():
    lengths = [10] * 100
    weights = [0.25] * 50 + [2.5] * 50
    sampler = BucketFrameSampler(lengths, 40, num_buckets=4, seed=1, weights=weights)
    counts = [0] * 100
    for epoch in range(40):
        for batch in sampler.batches_for_epoch(epoch):
            for i in batch:
                counts[i] += 1
    assert sum(counts[:50]) / (50 * 40) == pytest.approx(0.25, abs=0.05)
    assert sum(counts[50:]) / (50 * 40) == pytest.approx(2.5, abs=0.1)
    again = BucketFrameSampler(lengths, 40, num_buckets=4, seed=1, weights=weights)
    assert again.batches_for_epoch(3) == sampler.batches_for_epoch(3)
    plain = BucketFrameSampler(lengths, 40, num_buckets=4, seed=1)
    assert sorted(i for b in plain.batches_for_epoch(0) for i in b) == list(range(100))


def test_shipped_mixtures_and_licence_gate():
    pilot, full = load_mixture("pilot"), load_mixture("full")
    assert sum(t["hours"] for t in pilot["train"]) == 650
    evals, trains, skipped = plan_tasks(pilot)
    assert not any(t.source == "mucs" for t in trains)  # review source: not trained on by default
    assert any(t.source == "mucs" for t in evals)  # but usable as an eval set
    assert any("mucs" in s for s in skipped)
    _, trains, _ = plan_tasks(pilot, {"mucs"})
    assert sum(t.hours for t in trains) == 650
    langs = {t["lang"] for t in full["train"]}
    assert langs == {"hi", "mr", "bn", "pa", "gu", "or", "ur", "kn", "ta", "te", "ml", "en"}
    plan = estimate(pilot, {"mucs"})
    assert plan["train_hours"] == 650 and 0 < plan["flac_gb"] < plan["download_gb"]
    for mixture in (pilot, full):  # no excluded or blocked source is ever named
        names = {e["source"] for e in mixture["train"] + mixture["eval"]}
        assert not names & {"common_voice", "indictts"}


def tiny_mixture(path: Path) -> Path:
    mixture = {
        "name": "tiny", "seed": 0, "min_seconds": 0.5, "max_seconds": 20.0,
        "dev": {"fraction": 0.2, "max_hours_per_lang": 1.0},
        "sampling": {"alpha": 0.5, "max_share": {"hi": 0.6}},
        "train": [{"source": "kathbath", "lang": "hi", "split": "train", "hours": 1},
                  {"source": "librispeech", "lang": "en", "split": "train.clean.360", "hours": 1}],
        "eval": [{"name": "kathbath_valid_hi", "source": "kathbath", "lang": "hi", "split": "valid"}],
    }
    path.write_text(json.dumps(mixture))
    return path


def kathbath_rows(n, speakers, start=0):
    texts = ["मैं घर जा रहा हूँ", "आज मौसम अच्छा है", "हम office जा रहे हैं"]
    return [{"seconds": 2.0, "fname": f"8444249{start + i:08d}-{speakers[i % len(speakers)]}-f.m4a",
             "text": texts[i % 3], "lang": "hindi", "duration": 2.0, "gender": "female",
             "speaker_id": int(speakers[i % len(speakers)])} for i in range(n)]


def test_prep_with_a_mixture_end_to_end(tmp_path, monkeypatch):
    import indicear.ingest

    train_hi = write_parquet(tmp_path / "kb_train.parquet",
                             kathbath_rows(24, ["101", "102", "103", "104", "105", "106"]), "audio_filepath")
    valid_hi = write_parquet(tmp_path / "kb_valid.parquet", kathbath_rows(4, ["101", "900"], start=500),
                             "audio_filepath")
    english = write_parquet(tmp_path / "ls.parquet", [
        {"seconds": 2.0, "file": "x", "text": t, "speaker_id": 10 + i % 4, "chapter_id": 1, "id": f"10-1-{i:04d}"}
        for i, t in enumerate(["HELLO WORLD", "GOOD MORNING", "SEE YOU SOON", "THANK YOU"] * 3)])
    hub = FakeHub({("kathbath", "hi", "train"): [train_hi], ("kathbath", "hi", "valid"): [valid_hi],
                   ("librispeech", "en", "train.clean.360"): [english]})
    monkeypatch.setattr(indicear.ingest, "list_hf_shards", hub.lister)
    monkeypatch.setattr(indicear.ingest, "download_hf", hub.downloader)
    root = tmp_path / "root"
    argv = ["--root", str(root), "prep", "--mixture", str(tiny_mixture(tmp_path / "tiny.json")),
            "--negatives", "2", "--negatives-dev", "1", "--cs-per-pair", "2", "--decode-workers", "2"]
    ablate.main(argv)
    paths = ablate.Paths(root)
    report = json.loads((root / "prep_report.json").read_text())
    data = report["data"]
    assert report["mixture"] == "tiny" and data["leak_check"].startswith("passed")
    assert data["mixing_drops"] == {"eval_speaker:kathbath": 4}  # speaker 101 is in the eval set
    train = [json.loads(line) for line in paths.manifest("train").read_text().splitlines()]
    dev = [json.loads(line) for line in paths.manifest("dev").read_text().splitlines()]
    assert {r["speaker"] for r in train}.isdisjoint({r["speaker"] for r in dev})
    assert "kathbath:101" not in {r["speaker"] for r in train + dev}
    assert all("sample_weight" in r for r in train) and set(data["train"]["by_lang"]) == {"hi", "en"}
    assert any(r["code_mixed"] for r in train) and "code_mixed" in data["train"]["by_code_mixed"]  # "हम office ..."
    assert {x["source"] for x in data["licenses"]} == {"kathbath", "librispeech"}
    assert json.loads((paths.manifests / "sources.json").read_text())["sources"]
    assert paths.manifest("eval_kathbath_valid_hi").exists() and report["text_check"]["train"]["rows"] == len(train)
    downloads = len(hub.downloads)
    ablate.main(argv)  # rerun: every stage is finished, nothing is fetched
    assert len(hub.downloads) == downloads
    with pytest.raises(SystemExit, match="prepared with --mixture tiny"):
        ablate.main(["--root", str(root), "prep"])  # FLEURS must not reuse a mixture root


def test_assemble_reports_unavailable_and_shortfall():
    from indicear.ingest import Task, TaskResult

    ok = TaskResult(Task("librispeech", "en", "train.clean.360", 1.0),
                    rows=[{**row("a", "en", speaker="librispeech:1"), "source": "librispeech"}], seconds=10.0)
    gated = TaskResult(Task("indicvoices", "hi", "train", 5.0), error="gated and not accessible")
    built = assemble({"name": "t", "dev": {"fraction": 0.0, "max_hours_per_lang": 0.0}}, [], [ok, gated])
    report = built["report"]
    assert report["unavailable"] == ["train indicvoices/hi/train: gated and not accessible"]
    assert report["shortfall_hours"] == {"librispeech/en/train.clean.360": 1.0}
