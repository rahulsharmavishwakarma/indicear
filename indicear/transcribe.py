"""Transcribe audio files with an exported model package (indicear.export).

    python -m indicear.transcribe --model rahulsharmavishwakarma/vesper-indicEAR talk.wav call.flac
    python -m indicear.transcribe --model PKG --json --device auto --beam 8 talk.m4a

Pipeline, per file:

1. Decode to 16 kHz mono float32: soundfile (WAV, FLAC, OGG, MP3, ...), resampled with soxr; any
   other format through `ffmpeg -f f32le -ac 1 -ar 16000`.
2. Split into windows of `--window` seconds (at most 30 s, the Gemma 4 feature extractor's
   max_length) that overlap by `--overlap` seconds. Window starts fall on the encoder's 40 ms frame
   grid. The encoder has no look-ahead (right context 0), so a window's *end* is as good as in a
   full-length pass and only its *start* lacks left context: each window keeps the frames after the
   previous window's last frame, and the overlap only warms its start up. The kept frames tile the
   file exactly (no gap, no duplicate) and the decoders run over the merged frame sequence, so a
   token across a window boundary is decoded once.
3. Torch log-mel (`GpuLogMel`, as in training; on the CPU for MPS, which has no float64 FFT), the
   frozen audio tower's layers 1-6, layers 7-12 with the package's adapters, then the heads.
4. Decode with the model's best path: CTC greedy (or CTC prefix beam search with `--beam N`) for
   v1-v5, the factored RNN-T greedy decode for v6 (carrying the prediction network state across
   windows). The text is the tokenizer's output: each word in its own script. CTC decoding applies
   the decode-time script constraints by default (`indicear.decoding`): `--script-rules on|off` (no
   mid-word script switch, no combining mark after another script's piece) and `--lid-script-set
   on|off` (only Latin plus the scripts of the likely languages from the utterance LID head;
   `--lid-topk`, `--lid-threshold`, `--lid-frame-share`). Both off gives the unconstrained decode.
5. `--json`: per-word start/end times from the emission frames (approximate), each word's script and,
   with a frame LID head, its language; the utterance language probabilities (duration-weighted mean
   over windows) and the code-switch language spans snapped to words.

The frozen tower is loaded once (`indicear.encoder.load_tower`: the package's `encoder/` when it has
one, else the audio tower of the full pinned EmbeddingGemma 2 checkpoint) and kept for every file.
`--model` takes a package directory or a Hugging Face repo id (`owner/name[@revision]`). Device `auto` prefers CUDA, then Apple MPS, then the CPU.
On MPS, one second of silence runs at load: if it fails, half precision falls back to float32 on
MPS, and float32 on MPS falls back to the CPU. dtype `auto` is bfloat16 on CUDA
GPUs that support it and float32 elsewhere (MPS half precision is opt-in with `--dtype`). The
real-time factor (wall time / audio duration) is printed per file on stderr.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch: unsupported MPS ops run on the CPU

import argparse
import contextlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Self

import numpy as np
import torch

from indicear import decoding, encoder
from indicear.common import EG2, FRAME_RATE, SAMPLE_RATE

MAX_WINDOW_SECONDS = 30.0
DEFAULT_OVERLAP_SECONDS = 2.0
SAMPLES_PER_FRAME = SAMPLE_RATE // FRAME_RATE  # 640: one encoder frame (40 ms)
BLANK_SKIP = math.log(0.999)  # beam search: frames this sure of blank only move mass to blank
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# ---------------------------------------------------------------- audio


def read_audio(path: str | Path) -> np.ndarray:
    """16 kHz mono float32: soundfile first, ffmpeg for anything it cannot read."""
    path = str(path)
    try:
        return encoder.load_audio(path)
    except Exception as soundfile_error:  # noqa: BLE001 - any libsndfile failure: try ffmpeg
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise SystemExit(f"{path}: soundfile cannot read it ({soundfile_error}) and ffmpeg is not installed") from None
        out = subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le",
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
            capture_output=True, check=False,
        )
        if out.returncode != 0:
            raise SystemExit(f"{path}: ffmpeg could not decode it: {out.stderr.decode(errors='replace').strip()}") from None
        return np.frombuffer(out.stdout, dtype=np.float32).copy()


def windows(samples: int, window_seconds: float = MAX_WINDOW_SECONDS,
            overlap_seconds: float = DEFAULT_OVERLAP_SECONDS) -> list[tuple[int, int]]:
    """[start, end) sample ranges covering `samples`: `window_seconds` long, starts on the 40 ms
    frame grid, consecutive windows overlapping by about `overlap_seconds`."""
    if not 0 < window_seconds <= MAX_WINDOW_SECONDS:
        raise ValueError(f"window must be in (0, {MAX_WINDOW_SECONDS:g}] seconds")
    window = round(window_seconds * FRAME_RATE) * SAMPLES_PER_FRAME
    overlap = round(overlap_seconds * FRAME_RATE) * SAMPLES_PER_FRAME
    if not 0 <= overlap < window:
        raise ValueError("overlap must be >= 0 and shorter than the window")
    step = window - overlap
    out = [(0, min(window, samples))]
    while out[-1][0] + window < samples:
        start = out[-1][0] + step
        out.append((start, min(start + window, samples)))
    return out


# ---------------------------------------------------------------- devices


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda: CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps: MPS is not available")
    return torch.device(name)


def pick_dtype(device: torch.device, name: str = "auto") -> torch.dtype:
    if name != "auto":
        dtype = DTYPES[name]
        if device.type == "cpu" and dtype == torch.float16:
            raise SystemExit("--dtype fp16 is not supported on the CPU; use fp32 or bf16")
        return dtype
    if device.type == "cuda":
        return encoder.default_dtype()  # bf16 where supported (not on a T4)
    return torch.float32  # MPS half precision is unverified for this encoder: opt in with --dtype


# ---------------------------------------------------------------- decoding helpers


def ctc_collapse(ids: Sequence[int]) -> tuple[list[int], list[int]]:
    """Greedy CTC path -> (tokens, emission frames): drop blanks (0), collapse repeats."""
    tokens, frames, previous = [], [], 0
    for frame, token in enumerate(ids):
        if token != previous and token != 0:
            tokens.append(int(token))
            frames.append(frame)
        previous = token
    return tokens, frames


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def _add_path(beams: dict, prefix: tuple[int, ...], slot: int, value: float, frames: tuple[int, ...]) -> None:
    """Add a path's probability to beams[prefix] = [p_blank, p_nonblank, frames, best path score];
    the emission frames follow the most likely path into the prefix."""
    entry = beams.setdefault(prefix, [-math.inf, -math.inf, frames, -math.inf])
    entry[slot] = _logaddexp(entry[slot], value)
    if value > entry[3]:
        entry[2], entry[3] = frames, value


def ctc_prefix_beam(blank_lp: Sequence[float], top_lp: Sequence[Sequence[float]], top_ids: Sequence[Sequence[int]],
                    beam: int, allow: Callable[[tuple[int, ...], int], bool] | None = None) -> tuple[list[int], list[int]]:
    """CTC prefix beam search (no language model) over per-frame candidate lists.

    blank_lp[t]: log p(blank); top_lp[t] / top_ids[t]: the frame's most likely tokens (blank may be
    among them and is skipped there). Frames with p(blank) > 0.999 only move probability to the
    blank state. Returns the best prefix and the frame where each of its tokens was emitted.
    `allow(prefix, token)`, when given, prunes extensions of a prefix by a new token (the
    script rules, `indicear.decoding.prefix_rule`); a repeat of the prefix's last token is never
    pruned."""
    neg = -math.inf
    beams: dict[tuple[int, ...], tuple[float, float, tuple[int, ...]]] = {(): (0.0, neg, ())}
    for t, blank in enumerate(blank_lp):
        if blank > BLANK_SKIP:
            beams = {p: (_logaddexp(pb, pnb) + blank, neg, f) for p, (pb, pnb, f) in beams.items()}
            continue
        nxt: dict[tuple[int, ...], list] = {}
        for prefix, (pb, pnb, frames) in beams.items():
            total = _logaddexp(pb, pnb)
            _add_path(nxt, prefix, 0, total + blank, frames)
            last = prefix[-1] if prefix else None
            for lp, token in zip(top_lp[t], top_ids[t], strict=True):
                if token == 0:
                    continue
                if token == last:
                    _add_path(nxt, prefix, 1, pnb + lp, frames)  # repeat without blank: same prefix
                    _add_path(nxt, (*prefix, token), 1, pb + lp, (*frames, t))  # blank in between: new token
                elif allow is not None and not allow(prefix, token):
                    continue
                else:
                    _add_path(nxt, (*prefix, token), 1, total + lp, (*frames, t))
        ranked = sorted(nxt.items(), key=lambda kv: _logaddexp(kv[1][0], kv[1][1]), reverse=True)[:beam]
        beams = {p: (v[0], v[1], v[2]) for p, v in ranked}
    best_prefix, (_, _, best_frames) = max(beams.items(), key=lambda kv: _logaddexp(kv[1][0], kv[1][1]))
    return list(best_prefix), list(best_frames)


class TransducerStream:
    """Greedy factored RNN-T decode over consecutive chunks of encoder frames of ONE utterance,
    carrying the prediction network state across chunks (same result as decoding the merged
    sequence in one call). Runs on the CPU: one frame at a time, so a GPU would only add syncs."""

    def __init__(self, transducer, max_symbols: int = 5):
        import copy

        self.head = copy.deepcopy(transducer).float().cpu().eval()
        self.max_symbols = max_symbols
        self.tokens: list[int] = []
        self.frames: list[int] = []
        with torch.no_grad():
            pred, self.state = self.head.predict(torch.zeros(1, 1, dtype=torch.long))
        self.pred = pred[0, 0]

    @torch.no_grad()
    def feed(self, enc: torch.Tensor, first_frame: int) -> None:
        projected = self.head.enc_proj(enc.float().cpu())  # [T, J]
        for t in range(projected.shape[0]):
            for _ in range(self.max_symbols):
                best = int(self.head.output(torch.tanh(projected[t] + self.pred)[None]).argmax(-1))
                if best == 0:
                    break
                self.tokens.append(best)
                self.frames.append(first_frame + t)
                pred, self.state = self.head.predict(torch.tensor([[best]]), self.state)
                self.pred = pred[0, 0]


# ---------------------------------------------------------------- the transcriber


def resolve_package(package: str | Path) -> Path:
    """A local package directory, or a Hugging Face model repo id ("owner/name" or "owner/name@revision")
    downloaded to the Hub cache."""
    path = Path(package).expanduser()
    if path.exists() or str(package).count("/") != 1 or str(package).startswith((".", "~", "/")):
        return path
    from huggingface_hub import snapshot_download

    repo, _, revision = str(package).partition("@")
    return Path(snapshot_download(repo, revision=revision or None))


class Transcriber:
    """Loads a package once; `transcribe(audio)` / `transcribe_file(path)` per input."""

    def __init__(self, package: str | Path, device: str = "auto", dtype: str = "auto", verify: bool = True,
                 tower=None, extractor=None, log=None):
        from indicear import export
        from indicear.tokenizer import AggregateTokenizer

        self.log = log or (lambda message: print(message, file=sys.stderr, flush=True))
        self.package = resolve_package(package)
        if verify:
            problems = export.verify(self.package)
            if problems:
                raise SystemExit(f"{self.package}: package verification failed:\n" + "\n".join(problems)
                                 + "\n(re-export it, or pass --no-verify to load it anyway)")
        manifest = export.read_json(self.package / export.MANIFEST)
        if manifest is None:
            raise SystemExit(f"{self.package} is not a model package (no {export.MANIFEST})")
        self.manifest = manifest
        pinned = f"{EG2[0]}@{EG2[1]}"
        trained = f"{manifest['encoder']['repo']}@{manifest['encoder']['revision']}"
        if trained != pinned:
            raise SystemExit(f"{self.package} was trained on {trained}, but this code loads {pinned}")
        started = time.time()
        self.tokenizer = AggregateTokenizer.load(self.package / manifest["tokenizer"]["dir"])
        self.device = pick_device(device)
        self.dtype = pick_dtype(self.device, dtype)
        bundled = manifest["encoder"].get("bundled")
        bundled = self.package / bundled if bundled else None
        self.extractor = extractor if extractor is not None else encoder.load_extractor(bundled)
        if tower is None:
            tower = encoder.load_tower(self.device, self.dtype, bundled)
        self.tower = tower.to(self.device, self.dtype).eval()
        self.staged = encoder.StagedTower(self.tower)
        self.model = export.load_design(self.staged, self.package, self.tokenizer).to(self.device).eval()
        self.tables = decoding.TokenTables.from_tokenizer(self.tokenizer)
        self._featurizer()
        if self.device.type == "mps":
            self._mps_fallback_check()
        self.load_seconds = time.time() - started

    # ------------------------------------------------------------ setup

    def _featurizer(self) -> None:
        # GpuLogMel computes in float64, which MPS lacks: on MPS the features are made on the CPU.
        feature_device = "cpu" if self.device.type == "mps" else self.device
        self.featurizer = encoder.Featurizer.from_extractor(self.extractor, feature_device, self.dtype)

    def _mps_fallback_check(self) -> None:
        """One second of silence through the whole path. On failure: half precision on MPS falls
        back to float32 on MPS, float32 on MPS falls back to the CPU."""
        for fallback in (torch.device("mps"), torch.device("cpu")):
            try:
                self.frame_outputs(np.zeros(SAMPLE_RATE, dtype=np.float32), max_window_seconds=1.0)
                return
            except Exception as error:  # noqa: BLE001 - any MPS failure: try the next fallback
                if fallback.type == "mps" and self.dtype == torch.float32:
                    fallback = torch.device("cpu")
                self.log(f"{self.device.type}/{str(self.dtype).removeprefix('torch.')} forward failed "
                         f"({type(error).__name__}: {error}); falling back to {fallback.type}/float32")
                self.device, self.dtype = fallback, torch.float32
                self.tower.to(self.device, self.dtype)
                self.model.to(self.device)
                self._featurizer()
                if fallback.type == "cpu":
                    return

    def close(self) -> None:
        self.model.detach_adapters()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @property
    def decoder(self) -> str | None:
        return self.manifest.get("decoder")

    def _autocast(self):
        if self.dtype == torch.float32:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    # ------------------------------------------------------------ encoder pass

    @torch.inference_mode()
    def frame_outputs(self, audio: np.ndarray, window_seconds: float = MAX_WINDOW_SECONDS,
                      overlap_seconds: float = DEFAULT_OVERLAP_SECONDS, batch_windows: int = 4, top_k: int = 1,
                      decoder: str | None = None, max_window_seconds: float | None = None,
                      tables: bool = False) -> dict:
        """Run the encoder and heads over `audio` in windows and merge the kept frames.

        Returns {"frames": T, "kept": [(window start frame, first kept, end) ...],
        "ctc": {"blank" [T], "top_lp" [T, k], "top_ids" [T, k]} (CTC models; with `tables`, also
        "cat_lp" / "cat_id" [T, scripts, 3], the per-(script, kind group) reduction that the
        script constraints decode from: `indicear.decoding.TokenTables.reduce`),
        "rnnt": (tokens, frames) (with decoder "rnnt"), "frame_lid" [T, 13], "utt_probs" [12]
        (LID heads)}; every tensor on the CPU."""
        from indicear.features import pad_waveforms

        model = self.model
        if max_window_seconds is not None:
            window_seconds = min(window_seconds, max_window_seconds)
            overlap_seconds = min(overlap_seconds, window_seconds / 2)
        spans = windows(len(audio), window_seconds, overlap_seconds) if len(audio) else []
        decoder = decoder or self.decoder
        ctc_parts: dict[str, list[torch.Tensor]] = {"blank": [], "top_lp": [], "top_ids": []}
        if tables:
            ctc_parts.update({"cat_lp": [], "cat_id": []})
        lid_frames: list[torch.Tensor] = []
        utt_sum, utt_weight = None, 0
        stream = TransducerStream(model.transducer) if decoder == "rnnt" and model.transducer is not None else None
        kept: list[tuple[int, int, int]] = []
        end = 0  # global frames emitted so far
        for first in range(0, len(spans), max(1, batch_windows)):
            group = spans[first:first + max(1, batch_windows)]
            waveforms, lengths = pad_waveforms([audio[a:b] for a, b in group])
            features, mask = self.featurizer({"waveforms": waveforms, "lengths": lengths})
            features, mask = features.to(self.device), mask.to(self.device)
            with self._autocast(), model.adapters_attached():
                h6, valid = self.staged.lower(features, mask)
                counts = valid.sum(-1)
                states: dict[int, torch.Tensor] | None = {} if model.lid is not None else None
                hidden, _ = model.encode(h6, counts, states=states)
                final = model.final(hidden) if model.final is not None and stream is None else None
                lid_out = model.lid(states, counts) if model.lid is not None else None
            counts = counts.cpu().tolist()
            for j, (start_sample, _end_sample) in enumerate(group):
                offset = start_sample // SAMPLES_PER_FRAME
                lo, hi = max(0, end - offset), int(counts[j])
                if hi <= lo:
                    continue
                kept.append((offset, lo, hi))
                if final is not None:
                    lp = final[j, lo:hi].float()
                    top_lp, top_ids = lp.topk(min(top_k, lp.shape[-1]), dim=-1)
                    ctc_parts["blank"].append(lp[:, 0].cpu())
                    ctc_parts["top_lp"].append(top_lp.cpu())
                    ctc_parts["top_ids"].append(top_ids.cpu())
                    if tables:
                        _, cat_lp, cat_id = self.tables.reduce(lp)
                        ctc_parts["cat_lp"].append(cat_lp.cpu())
                        ctc_parts["cat_id"].append(cat_id.cpu())
                if stream is not None:
                    stream.feed(hidden[j, lo:hi], offset + lo)
                if lid_out is not None:
                    if "frame_log_probs" in lid_out:
                        lid_frames.append(lid_out["frame_log_probs"][j, lo:hi].float().cpu())
                    if "utt_logits" in lid_out:
                        probs = lid_out["utt_logits"][j].float().softmax(-1).cpu() * (hi - lo)
                        utt_sum = probs if utt_sum is None else utt_sum + probs
                        utt_weight += hi - lo
                end = offset + hi
        out: dict = {"frames": end, "kept": kept, "windows": len(spans)}
        if model.final is not None and stream is None:
            out["ctc"] = {k: torch.cat(v) if v else torch.zeros(0) for k, v in ctc_parts.items()}
        if stream is not None:
            out["rnnt"] = (stream.tokens, stream.frames)
        if model.lid is not None:
            if lid_frames:
                out["frame_lid"] = torch.cat(lid_frames)
            if utt_sum is not None:
                out["utt_probs"] = utt_sum / max(utt_weight, 1)
        return out

    # ------------------------------------------------------------ transcription

    def transcribe(self, audio: np.ndarray, beam: int = 1, decoder: str | None = None,
                   window_seconds: float = MAX_WINDOW_SECONDS, overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
                   batch_windows: int = 4, constraints: decoding.DecodeOptions | None = None,
                   outputs: dict | None = None, lang_hint: str | None = None) -> dict:
        """`constraints`: the decode-time script rules for CTC decoding (indicear.decoding; None or
        both rules off: the unconstrained decode). `outputs`: a `frame_outputs` result of this
        audio to decode again (computed with `tables=True` for constraints and the matching
        top_k for a beam), so one encoder pass can be decoded several ways. `lang_hint` (a known
        language, as baselines such as IndicConformer are given): the constraints use it in place of
        the utterance LID, so only its script and Latin are allowed; the reported LID is unchanged."""
        from indicear import lid

        decoder = decoder or self.decoder
        if decoder == "rnnt" and self.model.transducer is None:
            raise SystemExit("this model has no transducer head; use --decoder ctc")
        if decoder == "ctc" and self.model.final is None:
            raise SystemExit("this model has no transcript head (LID only)")
        constrained = constraints is not None and constraints.active and decoder != "rnnt"
        top_k = 1 if beam <= 1 else min(max(beam, 8), 64)  # candidates per frame for the beam search
        if outputs is None:
            outputs = self.frame_outputs(audio, window_seconds, overlap_seconds, batch_windows, top_k=top_k,
                                         decoder=decoder, tables=constrained)
        total = outputs["frames"]
        tokens: list[int] = []
        frames: list[int] = []
        scripts: set[int] | None = None
        if "rnnt" in outputs:
            tokens, frames = outputs["rnnt"]
        elif "ctc" in outputs and total and constrained:
            ctc = outputs["ctc"]
            probs = None
            if lang_hint in lid.LID_LANGS:
                probs = {lang: float(lang == lang_hint) for lang in lid.LID_LANGS}
            elif "utt_probs" in outputs:
                probs = decoding.utt_probs_dict(outputs["utt_probs"], logits=False)
            scripts = decoding.allowed_scripts(self.tables.order, probs, constraints, outputs.get("frame_lid"),
                                               ctc["blank"], ctc["cat_lp"])
            if beam > 1:
                top_lp, top_ids = decoding.beam_candidates(ctc["cat_lp"], ctc["cat_id"], ctc["top_lp"],
                                                           ctc["top_ids"], self.tables, scripts)
                tokens, frames = ctc_prefix_beam(ctc["blank"].tolist(), top_lp, top_ids, beam,
                                                 decoding.prefix_rule(self.tables, constraints))
                tokens, frames = self.tables.insert_breaks(tokens, frames, constraints)
            else:
                tokens, frames = decoding.constrained_greedy(self.tables, ctc["blank"], ctc["cat_lp"],
                                                             ctc["cat_id"], constraints, scripts)
        elif "ctc" in outputs and total:
            ctc = outputs["ctc"]
            if beam > 1:
                tokens, frames = ctc_prefix_beam(ctc["blank"].tolist(), ctc["top_lp"].tolist(),
                                                 ctc["top_ids"].tolist(), beam)
            else:
                tokens, frames = ctc_collapse(ctc["top_ids"][:, 0].tolist())
        words = self._words(tokens, frames, total)
        result: dict = {
            "text": " ".join(w["word"] for w in words if w["word"]),
            "decoder": decoder if self.model.final is not None else None,
            "beam": beam if decoder == "ctc" else 1,
            "duration": round(len(audio) / SAMPLE_RATE, 3),
            "frames": total,
            "windows": outputs["windows"],
            "words": words,
            "scripts": _counts(w["script"] for w in words if w["script"]),
        }
        if constrained:
            result["constraints"] = {**asdict(constraints), "allowed_scripts": None if scripts is None else
                                     [self.tables.order[i - 1] for i in sorted(scripts)]}
        if "utt_probs" in outputs:
            probs = dict(zip(lid.LID_LANGS, (round(float(p), 4) for p in outputs["utt_probs"]), strict=True))
            result["language"] = {"top": max(probs, key=probs.get),
                                  "probabilities": dict(sorted(probs.items(), key=lambda kv: -kv[1]))}
        if "frame_lid" in outputs:
            frame_lid = outputs["frame_lid"]
            if tokens:
                word_frames = [(w["_start"], w["_end"]) for w in words]
                langs, spans = lid.merge_with_words(frame_lid, word_frames)
                for word, lang in zip(words, langs, strict=True):
                    word["lang"] = lang
            else:
                spans = lid.frame_spans(frame_lid, total, self.model.lid.config.min_span_frames)
            result["spans"] = [
                {"lang": lang, "start": round(a / FRAME_RATE, 2), "end": round(b / FRAME_RATE, 2),
                 "text": " ".join(w["word"] for w in words if a <= w["_start"] < b and w["word"])}
                for lang, a, b in spans
            ]
        for word in words:
            word.pop("_start")
            word.pop("_end")
        result["words"] = [w for w in words if w["word"]]
        return result

    def _words(self, tokens: list[int], frames: list[int], total: int) -> list[dict]:
        from indicear.lid import word_frames

        if not tokens:
            return []
        starts = self.tokenizer.word_starts(tokens)
        timings = word_frames(tokens, frames, starts, max(total, frames[-1] + 1))
        firsts = [i for i, start in enumerate(starts) if start or i == 0]
        words = []
        for n, first in enumerate(firsts):
            stop = firsts[n + 1] if n + 1 < len(firsts) else len(tokens)
            begin, end = timings[n]
            words.append({"word": self.tokenizer.decode(tokens[first:stop]),
                          "start": round(begin / FRAME_RATE, 2), "end": round(end / FRAME_RATE, 2),
                          "script": self.tokenizer.script_of(tokens[first]), "_start": begin, "_end": end})
        return words

    def transcribe_file(self, path: str | Path, **options) -> dict:
        started = time.time()
        audio = read_audio(path)
        result = self.transcribe(audio, **options)
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - started
        duration = len(audio) / SAMPLE_RATE
        result = {"file": str(path), **result,
                  "timing": {"wall_seconds": round(wall, 3), "rtf": round(wall / duration, 4) if duration else None}}
        return result


def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="model package directory (indicear.export) or Hugging Face repo id owner/name[@revision]")
    parser.add_argument("audio", nargs="+", type=Path, help="audio files (any format soundfile or ffmpeg reads)")
    parser.add_argument("--json", action="store_true", help="JSON with words, timings, language and spans")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--dtype", default="auto", choices=("auto", *DTYPES))
    parser.add_argument("--beam", type=int, default=1, help="CTC prefix beam width (1: greedy; RNN-T is greedy)")
    parser.add_argument("--decoder", choices=("ctc", "rnnt"), help="override the package's decoder")
    parser.add_argument("--window", type=float, default=MAX_WINDOW_SECONDS, help="window seconds (<= 30)")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_SECONDS, help="window overlap seconds")
    parser.add_argument("--batch-windows", type=int, default=4, help="windows per encoder batch")
    parser.add_argument("--no-verify", action="store_true", help="skip the package's sha256 check")
    decoding.add_flags(parser)
    args = parser.parse_args(argv)
    missing = [str(p) for p in args.audio if not p.is_file()]
    if missing:
        raise SystemExit(f"not found: {', '.join(missing)}")
    if args.beam < 1:
        parser.error("--beam must be >= 1")
    with Transcriber(args.model, args.device, args.dtype, verify=not args.no_verify) as transcriber:
        decoder = args.decoder or transcriber.decoder
        if args.beam > 1 and decoder == "rnnt":
            transcriber.log("--beam applies to CTC decoding; the RNN-T decode is greedy")
        transcriber.log(f"model {transcriber.manifest['name']} ({transcriber.manifest['variant']}, "
                        f"{decoder or 'LID only'}) on {transcriber.device.type}/{str(transcriber.dtype).removeprefix('torch.')}, "
                        f"loaded in {transcriber.load_seconds:.1f} s")
        results = []
        total_audio = total_wall = 0.0
        for path in args.audio:
            result = transcriber.transcribe_file(path, beam=args.beam, decoder=args.decoder, window_seconds=args.window,
                                                 overlap_seconds=args.overlap, batch_windows=args.batch_windows,
                                                 constraints=decoding.options_from_args(args))
            results.append(result)
            total_audio += result["duration"]
            total_wall += result["timing"]["wall_seconds"]
            language = f", language {result['language']['top']}" if "language" in result else ""
            transcriber.log(f"{path}: {result['duration']:.1f} s of audio in {result['timing']['wall_seconds']:.2f} s "
                            f"(RTF {result['timing']['rtf']}){language}")
            if not args.json:
                if len(args.audio) > 1:
                    print(f"==> {path} <==")
                print(result["text"] if result["decoder"] else f"[language: {result.get('language', {}).get('top')}]",
                      flush=True)
        if len(args.audio) > 1 and total_audio:
            transcriber.log(f"total: {total_audio:.1f} s of audio in {total_wall:.2f} s (RTF {total_wall / total_audio:.4f})")
        if args.json:
            print(json.dumps({
                "model": {k: transcriber.manifest.get(k) for k in ("name", "variant", "decoder", "languages")}
                | {"package": str(transcriber.package)},
                "device": transcriber.device.type, "dtype": str(transcriber.dtype).removeprefix("torch."),
                "load_seconds": round(transcriber.load_seconds, 3), "results": results,
            }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
