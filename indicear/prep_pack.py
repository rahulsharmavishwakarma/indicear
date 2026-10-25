"""Package a prepared ROOT so a new VM can skip `ablate prep`.

    python -m indicear.prep_pack pack    --root R --out prep.tar.gz                # text products only
    python -m indicear.prep_pack pack    --root R --out prep.tar --with-audio     # + every fixed set
    python -m indicear.prep_pack restore --root R2 --archive prep.tar[.gz]

`prep --stream` on a fresh VM took about 30 minutes for the pilot (downloads and decoding of the
eval and dev sets, the text pass over every pool shard, the tokenizer). A pack holds its products:

- always: tok/ (the tokenizer, which every checkpoint depends on), manifests/train_texts.jsonl
  and stream_plan.json (the text pass), manifests/mixture.json, the shard listings
  (sources/*/listing-*.json, so no Hub listing is needed) and prep_report.json;
- `--with-audio`: also every other manifest (eval sets, dev, cs_dev, negatives, FLEURS dev, the
  stream exclusions and stream_fixed.json) and every audio file they reference under ROOT
  (decoded 16 kHz FLAC). After such a restore, `prep` finds every stage done and only rechecks
  the transcripts.

Without audio, `prep` rebuilds the fixed sets on the new VM (seeded, so the same dev carve-out);
the text pass and the tokenizer are skipped. Audio paths in manifests are absolute, so they are
stored relative to ROOT and rewritten for the new ROOT on restore. The mixture file recorded in
manifests/mixture.json is re-pointed at this checkout's copy when the original path is missing and
the sha256 matches. `prep_pack.json` in the archive lists every file with its sha256; restore
verifies them before anything is moved into place.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path

TEXT_MANIFESTS = ("train_texts.jsonl", "stream_plan.json", "mixture.json")
INDEX = "prep_pack.json"
PATH_FIELDS = ("audio",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str, root: Path) -> str | None:
    try:
        return Path(value).resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return None


def _rewrite_rows(path: Path, transform: Callable[[str], str]) -> bytes:
    """A JSONL manifest with every path field passed through `transform`."""
    out = io.StringIO()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in PATH_FIELDS:
                if isinstance(row.get(key), str):
                    row[key] = transform(row[key])
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out.getvalue().encode("utf-8")


def plan(root: Path, with_audio: bool) -> tuple[list[Path], list[Path], list[Path]]:
    """(manifests to rewrite, other files, audio files) to pack, all under `root`."""
    root = Path(root).resolve()
    manifests = root / "manifests"
    if not (root / "tok" / "aggregate.json").exists() or not (manifests / "train_texts.jsonl").exists():
        raise SystemExit(f"{root} is not a prepared streaming root (no tok/aggregate.json or train_texts.jsonl)")
    files = sorted(p for p in (root / "tok").rglob("*") if p.is_file())
    files += sorted((root / "sources").glob("*/listing-*.json")) if (root / "sources").exists() else []
    files += [p for p in (root / "prep_report.json",) if p.exists()]
    rewrite: list[Path] = []
    audio: list[Path] = []
    for path in sorted(manifests.iterdir()):
        if not path.is_file():
            continue
        if path.name in TEXT_MANIFESTS:
            files.append(path)
        elif with_audio:
            (rewrite if path.suffix == ".jsonl" else files).append(path)
    if with_audio:
        seen: set[Path] = set()
        for manifest in rewrite:
            for line in manifest.open(encoding="utf-8"):
                if not line.strip():
                    continue
                value = json.loads(line).get("audio")
                if isinstance(value, str) and _relative(value, root) is not None:
                    file = Path(value).resolve()
                    if file not in seen:
                        if not file.exists():
                            raise SystemExit(f"{manifest.name} references a missing audio file {file}")
                        seen.add(file)
                        audio.append(file)
    return rewrite, files, audio


def pack(root: str | Path, out: str | Path, with_audio: bool = False, log: Callable[[str], None] = print) -> dict:
    root, out = Path(root).resolve(), Path(out)
    rewrite, files, audio = plan(root, with_audio)
    entries: dict[str, dict] = {}
    mode = "w:gz" if out.name.endswith((".tar.gz", ".tgz")) else "w"
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    started = time.time()
    with tarfile.open(partial, mode) as tar:
        for manifest in rewrite:
            data = _rewrite_rows(manifest, lambda v: _relative(v, root) or v)
            name = manifest.relative_to(root).as_posix()
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), int(time.time())
            tar.addfile(info, io.BytesIO(data))
            entries[name] = {"sha256": hashlib.sha256(data).hexdigest(), "kind": "manifest"}
        for path in files + audio:
            name = path.relative_to(root).as_posix()
            tar.add(path, arcname=name, recursive=False)
            entries[name] = {"sha256": _sha256(path), "kind": "audio" if path in audio else "file"}
        index = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "root": str(root), "with_audio": with_audio,
                 "files": entries}
        data = json.dumps(index, indent=1).encode("utf-8")
        info = tarfile.TarInfo(INDEX)
        info.size, info.mtime = len(data), int(time.time())
        tar.addfile(info, io.BytesIO(data))
    os.replace(partial, out)
    size = out.stat().st_size
    log(f"prep pack: {len(entries)} files ({len(audio)} audio) from {root} -> {out} "
        f"({size / 2**30:.2f} GiB, {time.time() - started:.0f} s)")
    return {"files": len(entries), "audio": len(audio), "bytes": size}


def _members(tar: tarfile.TarFile) -> Iterator[tarfile.TarInfo]:
    for member in tar:
        name = Path(member.name)
        if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
            raise SystemExit(f"refusing archive member {member.name!r}")
        yield member


def restore(archive: str | Path, root: str | Path, force: bool = False, log: Callable[[str], None] = print) -> dict:
    """Unpack into a staging directory beside ROOT, verify every sha256, rewrite the manifests'
    audio paths for this ROOT, then move the files into place (existing files are kept unless
    `force`). Refuses a ROOT that already holds a prepared tokenizer, unless `force`."""
    from indicear.common import write_json

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "tok" / "aggregate.json").exists() and not force:
        raise SystemExit(f"{root} already has a tokenizer: restore would mix two preps (use --force to overwrite)")
    staging = Path(tempfile.mkdtemp(prefix=".prep-restore-", dir=root))
    started = time.time()
    try:
        with tarfile.open(archive, "r:*") as tar:
            tar.extractall(staging, members=_members(tar), filter="data")
        index = json.loads((staging / INDEX).read_text(encoding="utf-8"))
        for name, entry in index["files"].items():
            if _sha256(staging / name) != entry["sha256"]:
                raise SystemExit(f"prep pack: {name} fails its sha256 check")
        moved = 0
        for name, entry in index["files"].items():
            source, target = staging / name, root / name
            if target.exists() and not force:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if entry["kind"] == "manifest":
                target.write_bytes(_rewrite_rows(source, lambda v: str(root / v) if not Path(v).is_absolute() else v))
            else:
                shutil.move(source, target)
            moved += 1
        repoint_mixture(root, log)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    report = {"archive": str(archive), "root": str(root), "files": moved, "with_audio": index["with_audio"],
              "seconds": round(time.time() - started)}
    write_json(root / "prep_restore.json", report)
    log(f"prep restore: {moved} files into {root} in {report['seconds']} s (with audio: {index['with_audio']})")
    return report


def repoint_mixture(root: Path, log: Callable[[str], None] = print) -> None:
    """manifests/mixture.json names the mixture file by absolute path (for `train --lower live`);
    on another host, point it at this checkout's copy when the sha256 matches."""
    from indicear.common import write_json
    from indicear.mixture import MIXTURES

    marker = root / "manifests" / "mixture.json"
    if not marker.exists():
        return
    record = json.loads(marker.read_text(encoding="utf-8"))
    if not record.get("file") or Path(record["file"]).exists():
        return
    local = MIXTURES / Path(record["file"]).name
    if local.exists() and _sha256(local) == record.get("sha256"):
        write_json(marker, {**record, "file": str(local)})
        log(f"prep restore: mixture file re-pointed to {local}")
    else:
        raise SystemExit(f"the prepared mixture {record['file']} is not on this host and {local} differs from it "
                         "(sha256); use the code revision the root was prepared with")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("pack")
    one.add_argument("--root", required=True)
    one.add_argument("--out", required=True, help=".tar.gz compresses (text only); .tar stores (with audio)")
    one.add_argument("--with-audio", action="store_true", help="also the eval/dev/negative manifests and their audio")
    two = sub.add_parser("restore")
    two.add_argument("--root", required=True)
    two.add_argument("--archive", required=True)
    two.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "pack":
        pack(args.root, args.out, args.with_audio)
    else:
        restore(args.archive, args.root, args.force)


if __name__ == "__main__":
    main()
