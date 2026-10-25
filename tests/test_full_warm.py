"""The continued (full) run: the full_warm mixture and its sampling, stream weights, disjoint
code-mixed pools, loanword handling in the loaders, prep packing and restore, the worker defaults,
push_init.sh and scripts/full_warm.sh in smoke mode."""

import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from data_fixtures import FakeHub
from test_stream import fetcher, kathbath_shard

from indicear import ablate, prep_pack, smoke
from indicear.common import write_json
from indicear.ingest import Context, DiskGuard, Task, clean_row, context_options
from indicear.markup import Cleaned, apply_loanwords, clean
from indicear.mixture import load_mixture, plan_tasks, processed_hours
from indicear.sources import SOURCES
from indicear.stream import StreamPlanner, build_planner, stream_specs

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
INDIC = {"hi", "mr", "bn", "pa", "gu", "or", "ur", "ta", "te", "kn", "ml"}
NEW = {"bn", "te", "ml", "mr", "gu", "pa", "or", "ur"}


# ---------------------------------------------------------------- the mixture


def test_full_warm_mixture_sources_evals_and_shares():
    mixture = load_mixture("full_warm")
    evals, trains, skipped = plan_tasks(mixture)
    assert not skipped
    assert {t.lang for t in trains} == INDIC | {"en"}
    for task in trains:  # licence-verified sources only; MUCS stays eval-only
        assert SOURCES[task.source].license_status == "allowed", task.source
        assert task.source not in ("shrutilipi", "mucs", "spring_inx")
    names = {t.eval_name for t in evals}
    for lang in INDIC:
        assert f"indicvoices_valid_{lang}" in names and f"vaani_test_{lang}" in names
        assert {f"kathbath_{s}_{lang}" for s in ("clean_test_known", "clean_test_unknown", "noisy_test_known",
                                                   "noisy_test_unknown")} <= names
    assert ablate.fleurs_dev_langs("mixture", mixture) == sorted(INDIC | {"en"})
    shares = mixture["sampling"]["max_share"]
    assert shares == mixture["sampling"]["min_share"] and sum(shares.values()) == pytest.approx(1.0)
    assert sum(shares[lang] for lang in NEW) == pytest.approx(0.65)
    assert shares["cs"] == 0.15 and shares["kn"] > max(shares["hi"], shares["ta"])
    assert mixture["loanwords"]["sources"] == ["kathbath", "kathbath_test"]


def test_full_warm_processed_hours_meet_the_targets():
    result = processed_hours(load_mixture("full_warm"), 8000)
    total = sum(r["hours"] for r in result["streams"])
    assert total == pytest.approx(8000)
    by_group, by_source = result["by_group"], result["by_source"]
    refresher = sum(by_group[g] for g in ("hi", "ta", "kn", "en")) / total
    assert 0.20 <= refresher <= 0.25
    assert 0.13 <= by_group["cs"] / total <= 0.17
    assert by_group["kn"] > max(by_group["hi"], by_group["ta"])  # Kannada slightly upweighted
    assert by_source["kathbath"] / total > 0.25  # more read speech than the pilot (about 13%)
    assert all(result["by_lang"][lang] > 550 for lang in NEW)


def stub_streams(mixture):
    specs, _, _ = stream_specs(mixture)

    class Stub:
        def __init__(self, spec):
            self.spec, self.count = spec, 0

        def state(self):
            return {}

        def next(self):
            self.count += 1
            return {"seconds": 5.0, "key": f"{self.spec.name}/{self.count}", "stream": self.spec.name,
                    "ref": [0, 0, self.count]}

    return [Stub(spec) for spec in specs]


def test_planner_draws_follow_the_pinned_shares_and_stream_weights():
    mixture = load_mixture("full_warm")
    streams = stub_streams(mixture)
    shares = mixture["sampling"]["max_share"]
    planner = StreamPlanner(streams, shares, frame_budget=5000, buffer_seconds=5.0 * 30_000, seed=0)
    planner._fill()
    counts = Counter(r["stream"] for r in planner.buffer)
    total = sum(counts.values())
    group = Counter()
    for stream in streams:
        group[stream.spec.group] += counts[stream.spec.name]
    for name, share in shares.items():
        assert group[name] / total == pytest.approx(share, abs=0.006), name
    kathbath, voices = counts["kathbath/bn"], counts["indicvoices/bn"]  # weight 2 on 90 h vs 1 on 300 h
    assert kathbath / voices == pytest.approx(2 * 90 / 300, rel=0.1)


def test_stream_weights_leave_the_pilot_unchanged_and_must_agree():
    for spec in stream_specs(load_mixture("pilot"))[0]:
        assert spec.weight == 1.0 and spec.draw_hours == spec.hours
    bad = {"name": "bad", "train": [{"source": "kathbath", "lang": "hi", "split": "train", "hours": 1, "weight": 2},
                                    {"source": "kathbath", "lang": "hi", "split": "valid", "hours": 1}], "eval": []}
    with pytest.raises(ValueError, match="same weight"):
        stream_specs(bad)


def test_disjoint_code_mixed_pools(tmp_path):
    shards = {("kathbath", "hi", "train"): [kathbath_shard(tmp_path / f"kb{i}.parquet", 100 * i, [str(100 + i)])
                                            for i in range(8)]}
    hub = FakeHub(shards)

    def lister(source, lang, split, root):  # big shards (0.5 h each), so a pool is a subset
        return [{**s, "size": 50_000_000} for s in hub.lister(source, lang, split, root)]

    write_json(tmp_path / "manifests" / "stream_exclude.json", {"speaker": [], "group": [], "key": [], "audio": []})
    base = {"name": "cm", "seed": 0, "sampling": {"alpha": 0.5},
            "train": [{"source": "kathbath", "lang": "hi", "split": "train", "hours": 1.0},
                      {"source": "kathbath", "lang": "hi", "split": "train", "hours": 0.15, "code_mixed_only": True}],
            "eval": []}
    pools = {}
    for flag in (False, True):
        setup = build_planner(tmp_path, {**base, "disjoint_code_mixed_pools": flag}, frame_budget=200,
                              lister=lister, fetch=fetcher(hub), log=lambda m: None)
        pools[flag] = {name: [s["path"] for s in pool] for name, pool in setup.pools.items()}
        setup.cache.close()
    main, mined = pools[True]["kathbath/hi"], pools[True]["kathbath/hi/cm"]
    assert main == pools[False]["kathbath/hi"] and not set(main) & set(mined)
    assert list(pools[True]) == ["kathbath/hi", "kathbath/hi/cm"]  # the mixture's order


# ---------------------------------------------------------------- loanwords in the loaders


class FakeNormaliser:
    def rewrite(self, text, lang):
        words = ["computer" if w == "कंप्यूटर" else w for w in text.split()]
        return " ".join(words), sum(w == "कंप्यूटर" for w in text.split())

    def fingerprint(self):
        return "fake"


def test_apply_loanwords_primary_and_eval():
    raw = "मेरा कंप्यूटर खराब है"
    cleaned = clean(raw, "hi")
    train = apply_loanwords(clean(raw, "hi"), "hi", FakeNormaliser())
    assert train.text == "मेरा computer खराब है" and train.extra_refs == [cleaned.text]
    assert train.code_mixed and train.word_langs == ["hi", "en", "hi", "hi"] and train.notes["loanword_words"] == 1
    evaluation = apply_loanwords(clean(raw, "hi"), "hi", FakeNormaliser(), primary=False)
    assert evaluation.text == cleaned.text and evaluation.extra_refs == ["मेरा computer खराब है"]
    untouched = apply_loanwords(clean("आज मौसम अच्छा है", "hi"), "hi", FakeNormaliser())
    assert untouched.extra_refs == [] and not untouched.code_mixed
    assert apply_loanwords(Cleaned("", "empty"), "hi", FakeNormaliser()).drop == "empty"


def test_clean_row_rewrites_only_the_named_sources(tmp_path):
    ctx = Context(tmp_path, DiskGuard(tmp_path, None, reserve_bytes=0), loanwords=FakeNormaliser(),
                  loanword_sources=frozenset({"kathbath"}))
    drops, notes = Counter(), Counter()
    raw = "मेरा कंप्यूटर खराब है"
    train = clean_row(SOURCES["kathbath"], Task("kathbath", "hi", "train", 1.0), raw, ctx, drops, notes)
    assert train.text.split()[1] == "computer" and notes["loanword_rows"] == 1
    test = clean_row(SOURCES["kathbath"], Task("kathbath", "hi", "train", 1.0, role="eval"), raw, ctx, drops, notes)
    assert test.text.split()[1] == "कंप्यूटर" and test.extra_refs
    other = clean_row(SOURCES["indicvoices"], Task("indicvoices", "hi", "train", 1.0), raw, ctx, drops, notes)
    assert other.text.split()[1] == "कंप्यूटर"
    options = context_options(load_mixture("full_warm"))
    assert options["loanword_sources"] == {"kathbath", "kathbath_test"}
    assert options["loanwords"].fingerprint() == context_options(load_mixture("full_warm"))["loanwords"].fingerprint()
    assert "loanwords" not in context_options(load_mixture("pilot"))


# ---------------------------------------------------------------- workers, prep pack, scripts


def test_scaled_workers(monkeypatch):
    for cores, expected in ((12, 4), (18, 4), (48, 12), (96, 16), (None, 4)):
        monkeypatch.setattr(os, "cpu_count", lambda c=cores: c)
        assert ablate.scaled_workers() == expected


@pytest.fixture
def prepared_root(tmp_path, monkeypatch):
    smoke.install(tmp_path, monkeypatch.setattr)
    root = tmp_path / "root"
    mixture = smoke.write_mixture(tmp_path / "smoke-mixture.json")
    args = ["--root", str(root), "prep", "--mixture", str(mixture), "--stream", "--negatives", "4",
            "--negatives-dev", "2", "--cs-per-pair", "2", "--decode-workers", "2"]
    ablate.main(args)
    return root, args


def no_text_pass(*_args, **_kwargs):
    raise AssertionError("the text pass ran after a restore")


def test_prep_pack_with_audio_restores_a_root_that_skips_prep(prepared_root, tmp_path, monkeypatch):
    import indicear.ingest
    import indicear.stream

    root, args = prepared_root
    archive = tmp_path / "prep.tar"
    report = prep_pack.pack(root, archive, with_audio=True, log=lambda m: None)
    assert report["audio"] > 0
    new = tmp_path / "vm2"
    prep_pack.restore(archive, new, log=lambda m: None)
    for name in ("dev", "eval_kathbath_clean_test_unknown_hi", "neg_train", "cs_dev"):
        for row in map(json.loads, (new / "manifests" / f"{name}.jsonl").read_text().splitlines()):
            assert row["audio"].startswith(str(new.resolve())) and Path(row["audio"]).exists()
    assert (new / "tok" / "aggregate.json").read_bytes() == (root / "tok" / "aggregate.json").read_bytes()
    monkeypatch.setattr(indicear.stream, "text_pass", no_text_pass)
    monkeypatch.setattr(indicear.ingest, "run_tasks", no_text_pass)  # no download or decode either
    ablate.main(["--root", str(new), *args[2:]])  # every stage is found done
    with pytest.raises(SystemExit, match="already has a tokenizer"):
        prep_pack.restore(archive, new, log=lambda m: None)
    broken = tmp_path / "broken.tar"
    import tarfile

    with tarfile.open(archive) as tar, tarfile.open(broken, "w") as out:
        for member in tar:
            data = tar.extractfile(member).read() if member.isfile() else None
            if member.name == "tok/deva.model":
                data = b"corrupt"
                member.size = len(data)
            out.addfile(member, __import__("io").BytesIO(data) if data is not None else None)
    with pytest.raises(SystemExit, match="sha256"):
        prep_pack.restore(broken, tmp_path / "vm3", log=lambda m: None)


def test_prep_pack_text_only_skips_the_text_pass_and_tokenizer(prepared_root, tmp_path, monkeypatch):
    import indicear.stream
    import indicear.tokenizer

    root, args = prepared_root
    archive = tmp_path / "prep.tar.gz"
    report = prep_pack.pack(root, archive, log=lambda m: None)
    assert report["audio"] == 0
    new = tmp_path / "vm2"
    prep_pack.restore(archive, new, log=lambda m: None)
    assert not (new / "manifests" / "dev.jsonl").exists() and (new / "manifests" / "train_texts.jsonl").exists()
    monkeypatch.setattr(indicear.stream, "text_pass", no_text_pass)
    monkeypatch.setattr(indicear.tokenizer.AggregateTokenizer, "train", no_text_pass)
    ablate.main(["--root", str(new), *args[2:]])  # rebuilds only the fixed sets
    assert (new / "manifests" / "dev.jsonl").exists()
    dev_old = sorted(json.loads(r)["key"] for r in (root / "manifests" / "dev.jsonl").read_text().splitlines())
    dev_new = sorted(json.loads(r)["key"] for r in (new / "manifests" / "dev.jsonl").read_text().splitlines())
    assert dev_old == dev_new  # the seeded carve-out is reproduced


def test_push_init_uploads_and_verifies(tmp_path):
    run = tmp_path / "pilot" / "v4+lid"
    (run / "tok").mkdir(parents=True)
    for name in ("heads.safetensors", "adapters.safetensors", "variant.json", "lid.json", "tok/aggregate.json",
                 "tok/deva.model"):
        (run / name).write_bytes(random.Random(name).randbytes(64))
    remote = tmp_path / "vm" / "init" / "v4+lid"
    out = subprocess.run(["bash", str(SCRIPTS / "push_init.sh"), "local", str(run), str(remote)],
                         capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 0, out.stderr + out.stdout
    assert (remote / "tok" / "deva.model").read_bytes() == (run / "tok" / "deva.model").read_bytes()
    assert "sha256 verified" in out.stdout


def test_full_warm_sh_smoke_chain(tmp_path):
    root = tmp_path / "full"
    env = {**os.environ, "SMOKE": "1", "ROOT": str(root), "PY": sys.executable, "WAIT_POLL": "1"}
    env.pop("PYTHON_GIL", None)

    def run(*args):
        return subprocess.run(["bash", str(SCRIPTS / "full_warm.sh"), *args], env=env, capture_output=True,
                              text=True, timeout=600, check=False)

    done = run("--foreground")
    assert done.returncode == 0, done.stdout + done.stderr
    stages = ("setup", "prep", "train", "eval", "report")
    assert all((root / "full_warm" / f"{s}.DONE").exists() for s in stages)
    result = json.loads((root / "results" / "v4+lid.json").read_text())
    init = result["train_config"]["extra"]["init_from"]
    assert init["kept_scripts"] == ["deva", "latn", "taml", "knda"] and init["new_script_logits"] == "fresh"
    config = result["train_config"]
    assert (config["peak_lr"], config["final_lr"], config["new_lr_mult"]) == (6e-4, 5e-5, 3.0)
    assert result["variant"]["inter_weight"] == 0.15 and result["lid_config"]["frame_weight"] == 0.2
    assert config["extra"]["augment"]["gpu"]["weights"]["reverb"] == 0.3
    assert json.loads((root / "tok" / "aggregate.json").read_text())["extended_from"]["kept"]
    assert (root / "runs" / "v4+lid" / "init_from.json").exists()
    assert (root / "results" / "v4+lid" / "kathbath_clean_test_unknown_hi.json").exists()
    command = (root / "runs" / "v4+lid" / "command.sh").read_text()
    assert "--init-from" in command and "--tok-from" in command
    again = run("--foreground")
    assert again.returncode == 0 and again.stdout.count("already done") == len(stages)
