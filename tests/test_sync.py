"""scripts/sync_runs.sh against a simulated remote (SESSION=local: the remote is a local dir).

Runs under /bin/bash (bash 3.2 on macOS) to catch portability problems."""

import json
import subprocess
from pathlib import Path

import torch

from indicear.runtime.checkpoint import CheckpointManager, sha256_file

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
BASH = "/bin/bash" if Path("/bin/bash").exists() else "bash"


def sync(remote: Path, dest: Path, *options: str, timeout: float = 60) -> str:
    out = subprocess.run(
        [BASH, str(SCRIPTS / "sync_runs.sh"), *options, "local", str(remote), str(dest)],
        capture_output=True, text=True, timeout=timeout,
    )
    assert out.returncode == 0, out.stderr + out.stdout
    return out.stdout


def save(manager: CheckpointManager, step: int, metric=None) -> Path:
    return manager.save({"weights": torch.full((256,), float(step)), "config_hash": "h"}, step, metric)


def local_checkpoints(dest: Path) -> list[str]:
    return sorted(p.name for p in dest.glob("ckpt-*.pt"))


def test_manifest_gated_verified_pruned(tmp_path):
    remote, dest = tmp_path / "vm" / "runs" / "v2", tmp_path / "mac" / "v2"
    manager = CheckpointManager(remote, keep=2, log=lambda _m: None)
    save(manager, 1, metric=5.0)
    save(manager, 2)
    manifest = json.loads((remote / "LATEST.json").read_text())
    assert manifest["checkpoint"]["file"] == "ckpt-00000002.pt" and manifest["checkpoint"]["best"] is False
    assert manifest["best"]["step"] == 1 and manifest["keep"] == ["ckpt-00000002.pt", "ckpt-00000001.pt"]

    # A save in progress (not in the manifest) must never be pulled.
    (remote / "ckpt-00000003.pt.tmp").write_bytes(b"partial")
    (remote / "ckpt-00000099.pt").write_bytes(b"unlisted")
    sync(remote, dest, "--once")
    assert local_checkpoints(dest) == ["ckpt-00000001.pt", "ckpt-00000002.pt"]
    assert (dest / "best.pt").exists() and not (dest / "ckpt-00000003.pt.tmp").exists()
    for entry in json.loads((dest / "LATEST.json").read_text())["files"]:
        assert sha256_file(dest / entry["file"]) == entry["sha256"]
    (remote / "ckpt-00000099.pt").unlink()

    # A file whose bytes do not match the manifest is rejected; the local manifest stays put.
    save(manager, 3)
    (remote / "ckpt-00000003.pt").write_bytes(b"corrupted after the manifest was written")
    out = sync(remote, dest, "--once")
    assert "REJECTED ckpt-00000003.pt" in out
    assert "ckpt-00000003.pt" not in local_checkpoints(dest)
    assert json.loads((dest / "LATEST.json").read_text())["step"] == 2

    # Next good save: pulled, and the local copy keeps the last 2 plus best.pt.
    save(manager, 4, metric=1.0)
    sync(remote, dest, "--once")
    # (The manifest describes the bytes on disk, so the rewritten ckpt-3 is re-hashed and listed.)
    local = json.loads((dest / "LATEST.json").read_text())
    assert local["step"] == 4 and local["best"]["step"] == 4
    assert local_checkpoints(dest) == ["ckpt-00000003.pt", "ckpt-00000004.pt"]  # 1 and 2 pruned
    assert sha256_file(dest / "best.pt") == local["best"]["sha256"]


def test_exits_on_done_after_final_pull(tmp_path):
    remote, dest = tmp_path / "vm" / "v3", tmp_path / "mac" / "v3"
    manager = CheckpointManager(remote, log=lambda _m: None)
    save(manager, 1)
    (remote / "result.json").write_text('{"cer": 1}')
    manager.publish("result.json")
    (remote / "train.csv").write_text("step,loss\n1,2.0\n")
    manager.mark("DONE")
    out = sync(remote, dest, "--interval", "1", timeout=30)  # no --once: must stop by itself
    assert "run finished: DONE" in out
    assert (dest / "DONE").exists() and (dest / "result.json").exists() and (dest / "train.csv").exists()
    assert json.loads((dest / "LATEST.json").read_text())["status"] == "DONE"


def test_ssh_config_uses_one_multiplexed_connection(tmp_path):
    script = (
        f'source "{SCRIPTS / "lib" / "transport.sh"}"; INDICEAR_SSH_DIR="{tmp_path}"; '
        'transport_init my-session; cat "$SSH_CONF"'
    )
    out = subprocess.run([BASH, "-c", script], capture_output=True, text=True, check=True).stdout
    assert "ProxyCommand colab ssh --proxy-mode -s my-session" in out
    assert "ControlMaster auto" in out and "ControlPersist 10m" in out and f"ControlPath {tmp_path}/cm-%C" in out


def test_pulls_the_tokenizer_beside_the_run(tmp_path):
    root = tmp_path / "vm"
    remote, dest = root / "runs" / "v4+lid", tmp_path / "mac" / "v4+lid"
    (root / "tok").mkdir(parents=True)
    (root / "tok" / "aggregate.json").write_text('{"order": ["deva"]}')
    (root / "tok" / "deva.model").write_bytes(b"model")
    (root / "tok" / "deva.vocab").write_text("vocab")
    manager = CheckpointManager(remote, log=lambda _m: None)
    save(manager, 1)
    manager.mark("DONE")
    out = sync(remote, dest, "--interval", "1", timeout=30)
    assert "pulled the tokenizer" in out
    assert sorted(p.name for p in (dest / "tok").iterdir()) == ["aggregate.json", "deva.model", "deva.vocab"]
