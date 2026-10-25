"""Tiny synthetic stand-ins for the verified source layouts (no network).

- HF parquet shards: an `Audio` column stored as struct {bytes, path} (WAV bytes, any sample rate)
  plus the source's text/speaker/duration columns, as read from the real parquet footers.
- MUCS: a .tar.gz with <split>/<recording>.wav first and <split>/transcripts/{text,segments,utt2spk}
  last (the order seen in the real test tarball's head).
- Kathbath tests: a transcripts tar (`kb_data_clean_m4a/<language>/<split>/transcription_n2w.txt`)
  and an audio tar of '<utt>-<speaker>-<gender>.m4a' members (WAV bytes inside, which soundfile reads).
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf


def wav_bytes(seconds: float, rate: int = 8000, freq: float = 220.0, seed: int = 0) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    noise = np.random.default_rng(seed).normal(0, 0.01, len(t))
    audio = (0.2 * np.sin(2 * np.pi * freq * t) + noise).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def write_parquet(path: Path, rows: list[dict], audio_column: str = "audio") -> Path:
    """rows: dicts with 'seconds' (audio length) and any other columns; audio is synthesized."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    base = sum(path.name.encode()) * 7  # distinct audio per file, so audio hashes never collide by accident
    for index, row in enumerate(rows):
        row = dict(row)
        seconds = row.pop("seconds")
        name = row.pop("_path", f"utt{index}.wav")
        row[audio_column] = {"bytes": wav_bytes(seconds, freq=200 + 10 * index + base % 97, seed=base + index),
                             "path": name}
        records.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), path, row_group_size=4)
    return path


class FakeHub:
    """Stands in for list_hf_shards / download_hf: {(source, lang, split): [parquet paths]}."""

    def __init__(self, shards: dict[tuple[str, str, str], list[Path]]):
        self.shards = shards
        self.downloads: list[str] = []
        self.fail_on: set[str] = set()

    def lister(self, source, lang, split, root):
        return [{"repo": f"fake/{source.name}", "revision": "0" * 40, "path": f"{lang}/{split}-{i:05d}.parquet",
                 "size": Path(p).stat().st_size, "_local": str(p)}
                for i, p in enumerate(self.shards.get((source.name, lang, split), []))]

    def downloader(self, shard, source, root):
        if shard["path"] in self.fail_on:
            self.fail_on.discard(shard["path"])
            raise KeyboardInterrupt("simulated interruption")
        self.downloads.append(shard["path"])
        target = Path(root) / "downloads" / Path(shard["path"]).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(shard["_local"]).read_bytes())
        return target


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def mucs_tarball(path: Path, split: str = "test") -> Path:
    """Two recordings, three segments each; transcripts after the audio."""
    texts, segments, utt2spk = [], [], []
    with tarfile.open(path, "w:gz") as tar:
        for rec, speaker in (("recA", "spk1"), ("recB", "spk2")):
            _add(tar, f"{split}/{rec}.wav", wav_bytes(9.0, rate=16000, seed=hash(rec) % 100))
            for i in range(3):
                utt = f"{rec}_{i:04d}"
                segments.append(f"{utt} {rec} {3 * i:.2f} {3 * i + 2.5:.2f}")
                utt2spk.append(f"{utt} {speaker}")
                texts.append(f"{utt} इस tutorial में हम window {i} देखेंगे")
        _add(tar, f"{split}/transcripts/text", "\n".join(texts).encode())
        _add(tar, f"{split}/transcripts/segments", "\n".join(segments).encode())
        _add(tar, f"{split}/transcripts/utt2spk", "\n".join(utt2spk).encode())
    return path


def kathbath_tars(directory: Path, speakers=("930", "931")) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    transcripts, audio = directory / "transcripts.tar", directory / "audio.tar"
    lines = []
    with tarfile.open(audio, "w") as tar:
        for i, speaker in enumerate(speakers):
            name = f"84442493{i:07d}-{speaker}-f.m4a"
            lines.append(f"{name}\tमैं घर जा रहा हूँ {i}")
            _add(tar, f"kb_data_clean_m4a/hindi/test_known/{name}", wav_bytes(2.0, rate=16000, seed=i))
    with tarfile.open(transcripts, "w") as tar:
        _add(tar, "kb_data_clean_m4a/hindi/test_known/transcription_n2w.txt", "\n".join(lines).encode())
    return transcripts, audio
