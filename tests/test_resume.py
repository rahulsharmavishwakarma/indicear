"""Kill a tiny smoke run mid-epoch (SIGTERM), rerun the same command, and check it ends with the
same step count and loss trajectory as an uninterrupted run."""

import csv

import pytest
import torch

from indicear import phonetic
from indicear.cache import Layer6Cache
from indicear.encoder import StagedTower
from indicear.model import VARIANTS, DesignA
from indicear.trainer import Example, TrainConfig, Trainer

RANGES = [(4, 20), (20, 40), (40, 50)]
VOCAB = 50


def make_cache_and_examples(dim: int):
    generator = torch.Generator().manual_seed(1)
    examples, index, rows = [], {}, []
    offset = 0
    for i in range(14):
        frames = 10 + (i * 7) % 23
        key = f"utt{i}"
        rows.append(torch.randn(frames, dim, generator=generator))
        index[key] = (offset, frames)
        offset += frames
        tokens = torch.randint(4, VOCAB, (max(1, frames // 6),), generator=generator)
        units = torch.randint(2, len(phonetic.UNITS), (max(1, frames // 4),), generator=generator)
        examples.append(Example((key,), tokens, units, frames, "xx"))
    cache = Layer6Cache(torch.cat(rows).to(torch.bfloat16), None, index)
    return cache, examples


def run(tmp_path, name, tower, kill_at=None, monkeypatch=None, sync_dir=None):
    staged = StagedTower(tower, 6)
    cache, examples = make_cache_and_examples(staged.hidden_size)
    torch.manual_seed(0)
    model = DesignA(staged, VARIANTS["v3"], VOCAB, RANGES, len(phonetic.UNITS))
    config = TrainConfig(
        epochs=2, frame_budget=70, log_every=1, eval_every=0, warmup_steps=2,
        ckpt_every_steps=1000, ckpt_every_seconds=1e9, compile_heads=False, num_buckets=3,
    )
    trainer = Trainer(model, cache, examples, config, tmp_path / name, copy_dir=sync_dir,
                      device="cpu", dtype=torch.float32, log=lambda _m: None)
    try:
        if kill_at is not None:
            monkeypatch.setenv("INDICEAR_TEST_KILL_AT_STEP", str(kill_at))
        return trainer.run()
    finally:
        if monkeypatch is not None:
            monkeypatch.delenv("INDICEAR_TEST_KILL_AT_STEP", raising=False)
        model.detach_adapters()


def losses(path):
    rows = list(csv.DictReader(path.open()))
    return [int(r["step"]) for r in rows], [float(r["loss"]) for r in rows]


def test_kill_and_resume_matches_uninterrupted(tmp_path, tiny_tower, monkeypatch):
    full = run(tmp_path, "full", tiny_tower)
    steps_full, loss_full = losses(tmp_path / "full" / "train.csv")
    assert full["steps"] == full["total_steps"] and full["steps"] >= 6

    kill_at = full["steps"] // 2 + 1  # mid-epoch for this schedule
    sync = tmp_path / "drive"
    with pytest.raises(SystemExit) as stopped:
        run(tmp_path, "resumed", tiny_tower, kill_at=kill_at, monkeypatch=monkeypatch, sync_dir=sync)
    assert stopped.value.code == 143
    assert list((tmp_path / "resumed").glob("ckpt-*.pt")), "final checkpoint written on SIGTERM"
    assert list(sync.glob("ckpt-*.pt")), "checkpoint synced off the VM"

    resumed = run(tmp_path, "resumed", tiny_tower, sync_dir=sync)
    steps_resumed, loss_resumed = losses(tmp_path / "resumed" / "train.csv")
    assert resumed["steps"] == full["steps"]
    assert steps_resumed == steps_full
    assert loss_resumed == pytest.approx(loss_full, rel=1e-4, abs=1e-5)


def test_fresh_vm_restores_from_sync_dir(tmp_path, tiny_tower, monkeypatch):
    sync = tmp_path / "drive"
    with pytest.raises(SystemExit):
        run(tmp_path, "vm1", tiny_tower, kill_at=3, monkeypatch=monkeypatch, sync_dir=sync)
    # A new VM: the local run directory is gone, only the synced copy remains.
    result = run(tmp_path, "vm2", tiny_tower, sync_dir=sync)
    steps, _ = losses(tmp_path / "vm2" / "train.csv")
    assert steps[0] == 1 and steps[-1] == result["steps"] == result["total_steps"]


def test_corrupt_latest_falls_back_and_config_change_refused(tmp_path, tiny_tower, monkeypatch):
    with pytest.raises(SystemExit):
        run(tmp_path, "r", tiny_tower, kill_at=2, monkeypatch=monkeypatch)
    with pytest.raises(SystemExit):
        run(tmp_path, "r", tiny_tower, kill_at=4, monkeypatch=monkeypatch)
    checkpoints = sorted((tmp_path / "r").glob("ckpt-*.pt"))
    assert [p.name for p in checkpoints] == ["ckpt-00000002.pt", "ckpt-00000004.pt"]
    checkpoints[-1].write_bytes(checkpoints[-1].read_bytes()[:100])  # truncate the latest
    result = run(tmp_path, "r", tiny_tower)
    assert result["steps"] == result["total_steps"]

    from indicear.runtime.checkpoint import CheckpointManager

    manager = CheckpointManager(tmp_path / "r", log=lambda _m: None)
    with pytest.raises(SystemExit, match="config hash"):
        manager.restore("different-hash")
    assert manager.restore("different-hash", force_new_run=True) is None
    assert not list((tmp_path / "r").glob("ckpt-*.pt"))


def test_resume_from_synced_local_copy_into_fresh_run_dir(tmp_path, tiny_tower, monkeypatch):
    """VM lost: the run continues from the copy pulled to the workstation, uploaded to a fresh run dir."""
    import subprocess
    from pathlib import Path

    scripts = Path(__file__).resolve().parent.parent / "scripts"
    bash = "/bin/bash" if Path("/bin/bash").exists() else "bash"
    full = run(tmp_path, "full", tiny_tower)
    _, loss_full = losses(tmp_path / "full" / "train.csv")
    with pytest.raises(SystemExit):
        run(tmp_path, "vm", tiny_tower, kill_at=full["steps"] // 2, monkeypatch=monkeypatch)
    mac = tmp_path / "mac" / "vm"
    subprocess.run([bash, str(scripts / "sync_runs.sh"), "--once", "local", str(tmp_path / "vm"), str(mac)],
                   check=True, capture_output=True, text=True)
    assert (mac / "LATEST.json").exists() and list(mac.glob("ckpt-*.pt"))
    fresh = tmp_path / "fresh"  # a new VM: nothing local but what resume_on uploads
    subprocess.run([bash, str(scripts / "resume_on.sh"), "--local", str(mac), "--remote-dir", str(fresh),
                    "--no-code", "--no-launch", "local", "vm"], check=True, capture_output=True, text=True)
    result = run(tmp_path, "fresh", tiny_tower)
    steps, loss_resumed = losses(fresh / "train.csv")
    assert result["steps"] == full["steps"] and steps[-1] == full["steps"]
    assert loss_resumed == pytest.approx(loss_full, rel=1e-4, abs=1e-5)


def test_non_finite_gradients_skip_the_optimizer_step_without_a_scaler():
    """bf16/fp32 (no GradScaler): an inf/nan gradient must not reach the weights; the schedule
    still advances, so the run's LR curve is unchanged."""
    from types import SimpleNamespace

    import torch

    from indicear.trainer import Trainer

    model = torch.nn.Linear(2, 1)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    logs = []
    fake = SimpleNamespace(model=model, scaler=None, cfg=SimpleNamespace(grad_clip=1.0), row_lr=None,
                           skipped_steps=0, log=logs.append)
    model(torch.tensor([[float("nan"), 1.0]])).sum().backward()
    Trainer.optimizer_step(fake, optimizer, scheduler)
    assert fake.skipped_steps == 1 and "non-finite" in logs[0]
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    assert scheduler.last_epoch == 1 and all(p.grad is None for p in model.parameters())
    model(torch.tensor([[1.0, 1.0]])).sum().backward()
    Trainer.optimizer_step(fake, optimizer, scheduler)
    assert fake.skipped_steps == 1
    assert not all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
