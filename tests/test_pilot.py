"""The pilot path on CPU: `ablate train --lower live` end to end with kill/resume, the shared-stream
group (LiveGroup), `ablate eval` result files, and scripts/pilot.sh in smoke mode."""

import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_stream_train import RANGES, VOCAB, make_batches, text_encoder

from indicear import ablate, phonetic, smoke
from indicear.encoder import Featurizer, StagedTower
from indicear.model import VARIANTS, DesignA
from indicear.stream_train import ListStream
from indicear.trainer import LiveGroup, TrainConfig, Trainer

TRAIN = ["train", "--lower", "live", "--variants", "v2,v4+lid", "--max-steps", "6", "--frame-budget", "300",
         "--eval-every", "3", "--no-compile", "--stream-workers", "2", "--buffer-seconds", "20"]
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """One root prepared with `prep --mixture smoke --stream`; copies of it share its audio files."""
    smoke.install(tmp_path, monkeypatch.setattr)
    base = tmp_path / "base"
    mixture = smoke.write_mixture(tmp_path / "smoke-mixture.json")
    ablate.main(["--root", str(base), "prep", "--mixture", str(mixture), "--stream", "--negatives", "4",
                 "--negatives-dev", "2", "--cs-per-pair", "2", "--decode-workers", "2"])

    def copy(name: str) -> Path:
        root = tmp_path / name
        shutil.copytree(base, root)
        return root

    return copy


def csv_losses(path: Path):
    rows = list(csv.DictReader(path.open()))
    return [int(r["step"]) for r in rows], [float(r["loss"]) for r in rows]


def test_live_train_kill_resume_and_eval(prepared, monkeypatch, capsys):
    full = prepared("full")
    ablate.main(["--root", str(full), *TRAIN])
    cut = prepared("cut")
    monkeypatch.setenv("INDICEAR_TEST_KILL_AT_STEP", "3")
    with pytest.raises(SystemExit) as stopped:
        ablate.main(["--root", str(cut), *TRAIN])
    assert stopped.value.code == 143
    monkeypatch.delenv("INDICEAR_TEST_KILL_AT_STEP")
    for variant in ("v2", "v4+lid"):
        run = cut / "runs" / variant
        assert [p.name for p in run.glob("ckpt-*.pt")] == ["ckpt-00000003.pt"]
        assert not (run / "DONE").exists() and not (run / "FAILED").exists()  # a clean stop is resumable
        command = (run / "command.sh").read_text()
        assert "--variants v2,v4+lid" in command and "cache.sh" not in command  # the group reruns as one
    ablate.main(["--root", str(cut), *TRAIN])
    for variant in ("v2", "v4+lid"):
        steps_full, loss_full = csv_losses(full / "runs" / variant / "train.csv")
        steps_cut, loss_cut = csv_losses(cut / "runs" / variant / "train.csv")
        assert steps_full == steps_cut == [6]  # logged every 25 steps and at the end
        assert loss_cut == pytest.approx(loss_full, rel=1e-4, abs=1e-5)  # same stream, augmentation and RNG
        result = json.loads((cut / "results" / f"{variant}.json").read_text())
        train = result["train"]
        assert train["steps"] == 6 and train["lower"] == "live" and train["group"] == ["v2", "v4+lid"]
        assert train["audio_hours"] > 0 and 0 < train["packing_measured"] <= 1.0
        assert {"hi", "en", "cs:hi+en", "negatives"} <= set(result["metrics"]["ctc"])
        assert "wer_script_normalised" in result["metrics"]["ctc"]["hi"]
        assert (cut / "runs" / variant / "DONE").exists()
        assert "dev CER by language: en " in (cut / "runs" / variant / "train.log").read_text()
    lid_result = json.loads((cut / "results" / "v4+lid.json").read_text())
    assert lid_result["lid"]["utterance"]["utterances"] > 0 and lid_result["lid_config"]["class_weights"]
    assert (cut / "runs" / "v4+lid" / "lid.json").exists()
    capsys.readouterr()
    ablate.main(["--root", str(cut), *TRAIN])
    assert "v2: results/v2.json exists, skipping" in capsys.readouterr().out

    ablate.main(["--root", str(cut), "eval", "--variants", "v2,v4+lid,v6", "--robust-per-lang", "2",
                 "--robust-conditions", "reverb:0.6,noise:10", "--workers", "2"])
    names = {p.stem for p in (cut / "results" / "v2").glob("*.json")}
    assert {"kathbath_clean_test_unknown_hi", "indiccontext_hi", "indiccontext_hi_code_mixed", "fleurs_dev",
            "librispeech_test_clean", "cs_dev", "neg_dev", "robust_reverb_0.6", "robust_noise_10"} <= names
    context = json.loads((cut / "results" / "v4+lid" / "indiccontext_hi_code_mixed.json").read_text())
    assert context["kind"] == "eval" and context["metrics"]["ctc"]["hi"]["rows_with_alternative_ref"] > 0
    assert context["metrics"]["ctc"]["hi"]["cm_spans"] > 0 and context["lid"]["spans"]["spans"] > 0
    robust = json.loads((cut / "results" / "v2" / "robust_reverb_0.6.json").read_text())
    assert robust["condition"] == "reverb:0.6"
    assert not (cut / "results" / "v6").exists()  # untrained variants are skipped
    before = (cut / "results" / "v2" / "cs_dev.json").stat().st_mtime_ns
    ablate.main(["--root", str(cut), "eval", "--variants", "v2", "--robust-conditions", ""])
    assert (cut / "results" / "v2" / "cs_dev.json").stat().st_mtime_ns == before  # finished sets are skipped
    ablate.main(["--root", str(cut), "report"])
    summary = (cut / "results" / "eval_summary.md").read_text()
    assert "| v4+lid | indiccontext_hi_code_mixed | ctc | hi |" in summary
    # Every variant is also scored under the decode-time script rules (its vocabulary is per script).
    assert "| v4+lid | indiccontext_hi_code_mixed | ctc_constrained | hi |" in summary
    assert "| v4+lid | lid | utterance LID |" in (cut / "results" / "summary.md").read_text()


def test_baselines_runner_records_skips_and_scores(prepared):
    from indicear import baselines

    root = prepared("baselines")
    baselines.main(["--root", str(root), "--systems", "smoke-echo,kothagemma,dulu-malayalam-ctc-778h",
                    "--robust-conditions", "", "--workers", "2"])
    report = json.loads((root / "results" / "baselines.json").read_text())
    assert report["smoke-echo"]["status"] == "ran"
    assert report["kothagemma"]["status"] == "nothing to run"  # no Bengali eval rows
    assert report["dulu-malayalam-ctc-778h"]["status"].startswith("not runnable")
    echo = json.loads((root / "results" / "smoke-echo" / "indiccontext_hi.json").read_text())
    assert echo["metrics"]["echo"]["hi"]["wer"] == 0.0 and echo["baseline"]["revision"] == "0" * 40
    rows = [json.loads(line) for line in
            (root / "results" / "smoke-echo" / "indiccontext_hi.rows.jsonl").read_text().splitlines()]
    assert len(rows) == echo["utterances"] and {"key", "lang", "hypothesis"} <= set(rows[0])


# ---------------------------------------------------------------- the shared-stream group


def build_trainer(tmp_path, name, staged, stream, featurizer, max_steps=4, log=None):
    torch.manual_seed(0)
    variant = replace(VARIANTS["v2"], specaugment=False)  # no randomness: members must match a solo run
    model = DesignA(staged, variant, VOCAB, RANGES, len(phonetic.UNITS))
    config = TrainConfig(max_steps=max_steps, frame_budget=200, log_every=1, eval_every=0, warmup_steps=2,
                         ckpt_every_steps=1000, ckpt_every_seconds=1e9, compile_heads=False, extra={"name": name})
    return Trainer(model, None, [], config, tmp_path / name, device="cpu", dtype=torch.float32,
                   log=log or (lambda _m: None), lower="live", stream=stream, featurizer=featurizer,
                   text_encoder=text_encoder)


def weights(trainer):
    return {k: v.clone() for k, v in trainer.model.state_dict().items()}


def test_group_members_match_a_solo_run_and_never_see_each_other(tmp_path, tiny_tower):
    extractor = smoke.make_extractor()
    featurizer = Featurizer.from_extractor(extractor, "cpu", torch.float32)
    staged = StagedTower(tiny_tower, 6)
    solo = build_trainer(tmp_path, "solo", staged, ListStream(make_batches(), "toy"), featurizer)
    try:
        solo_summary = solo.run()
    finally:
        solo.model.detach_adapters()
    stream = ListStream(make_batches(), "toy")
    a = build_trainer(tmp_path, "a", staged, stream, featurizer)
    b = build_trainer(tmp_path, "b", staged, stream, featurizer)
    try:
        summaries = LiveGroup([a, b]).run()
        assert not a.model.adapters.attached and not b.model.adapters.attached
    finally:
        for t in (a, b):
            t.model.detach_adapters()
    assert [s["steps"] for s in summaries] == [4, 4] and summaries[0]["group"] == ["a", "b"]
    assert stream.yielded == list(range(4))  # one stream pass feeds both members
    reference = weights(solo)
    for member in (a, b):  # had a member seen the other's adapters, its weights would differ
        for key, value in weights(member).items():
            assert torch.allclose(value, reference[key], atol=1e-6), key
    assert solo_summary["steps"] == 4
    with pytest.raises(ValueError, match="one stream"):
        other = build_trainer(tmp_path, "c", staged, ListStream(make_batches(), "toy"), featurizer)
        try:
            LiveGroup([a, other])
        finally:
            other.model.detach_adapters()


def test_group_resumes_members_saved_at_different_steps(tmp_path, tiny_tower, monkeypatch):
    extractor = smoke.make_extractor()
    featurizer = Featurizer.from_extractor(extractor, "cpu", torch.float32)
    staged = StagedTower(tiny_tower, 6)
    messages: list[str] = []

    def group(names, kill=None):
        stream = ListStream(make_batches(), "toy")
        members = [build_trainer(tmp_path, n, staged, stream, featurizer, max_steps=5, log=messages.append)
                   for n in names]
        if kill is not None:
            monkeypatch.setenv("INDICEAR_TEST_KILL_AT_STEP", str(kill))
        try:
            return LiveGroup(members).run(), stream, members
        finally:
            monkeypatch.delenv("INDICEAR_TEST_KILL_AT_STEP", raising=False)
            for m in members:
                m.model.detach_adapters()

    _, _, reference = group(["ra", "rb"])
    with pytest.raises(SystemExit):
        group(["a", "b"], kill=2)
    # a VM lost between the members' saves: b got one checkpoint further than a
    state = torch.load(tmp_path / "b" / "ckpt-00000002.pt", weights_only=False)
    state["step"] = 3
    torch.save(state, tmp_path / "b" / "ckpt-00000003.pt")
    summaries, stream, members = group(["a", "b"])
    assert any("all resume at step 2" in m for m in messages)
    assert stream.yielded == [2, 3, 4] and [s["steps"] for s in summaries] == [5, 5]
    for member, ref in zip(members, reference, strict=True):
        for key, value in weights(member).items():
            assert torch.allclose(value, weights(ref)[key], atol=1e-6), key


def test_hours_are_converted_to_steps_once_the_budget_is_known():
    from indicear.trainer import live_steps

    assert live_steps(6500, 100_000, 0.85) == 6883  # 6500 h x 90,000 frames/h / (100k x 0.85)
    assert live_steps(1, 90_000, 1.0) == 1
    with pytest.raises(ValueError):
        live_steps(0, 1000, 0.85)


# ---------------------------------------------------------------- scripts/pilot.sh


def run_pilot(root: Path, *args, extra_env=None, timeout=300):
    env = {**os.environ, "SMOKE": "1", "ROOT": str(root), "PY": sys.executable, "WAIT_POLL": "1",
           **(extra_env or {})}
    env.pop("PYTHON_GIL", None)
    return subprocess.run(["bash", str(SCRIPTS / "pilot.sh"), *args], env=env, capture_output=True, text=True,
                          timeout=timeout, check=False)


STAGES = ("setup", "prep", "train", "eval", "baselines", "report")


def test_pilot_sh_smoke_chain_foreground_and_queued(tmp_path):
    import time

    root = tmp_path / "fg"
    done = run_pilot(root, "--foreground")
    assert done.returncode == 0, done.stdout + done.stderr
    assert all((root / "pilot" / f"{s}.DONE").exists() for s in STAGES)
    for variant in ("v2", "v3", "v4", "v4+lid"):
        assert (root / "results" / f"{variant}.json").exists()
        assert (root / "results" / variant / "indiccontext_hi.json").exists()
        assert (root / "runs" / variant / "DONE").exists() and (root / "runs" / variant / "gpu.csv").exists()
    assert (root / "results" / "smoke-echo" / "fleurs_dev.json").exists()
    assert (root / "results" / "eval_summary.md").exists()
    again = run_pilot(root, "--foreground")
    assert again.returncode == 0 and again.stdout.count("already done") == len(STAGES)

    queued = tmp_path / "queued"
    started = run_pilot(queued)
    assert started.returncode == 0 and started.stdout.count("started pilot-") == len(STAGES)
    deadline = time.time() + 240
    while time.time() < deadline and not (queued / "pilot" / "report.DONE").exists():
        assert not list((queued / "pilot").glob("*.FAILED")), (queued / "logs").glob("*.log")
        time.sleep(1)
    assert (queued / "pilot" / "report.DONE").exists()
    assert "ready: " in (queued / "logs" / "pilot-report.log").read_text()


def test_launch_wait_gives_up_when_the_awaited_step_fails(tmp_path):
    env = {**os.environ, "ROOT": str(tmp_path), "WAIT_POLL": "1", "PY": sys.executable,
           "WAIT_FOR": str(tmp_path / "step.DONE")}
    env.pop("PYTHON_GIL", None)
    out = subprocess.run(["bash", str(SCRIPTS / "launch.sh"), "after", "true"], env=env, capture_output=True,
                         text=True, timeout=30, check=True)
    assert "waiting for" in out.stdout
    (tmp_path / "step.FAILED").write_text("exit 1")
    import time

    pid = int((tmp_path / "logs" / "after.pid").read_text())
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.5)
    assert "abort: " in (tmp_path / "logs" / "after.log").read_text()
