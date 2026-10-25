"""Gemma 4 log-mel features in torch, batched on the model's device.

A drop-in for transformers' `Gemma4AudioFeatureExtractor.__call__(audios, padding=True)` (5.20.2)
on padded waveforms, so DataLoader workers only decode and corrupt audio and the GPU computes the
features. The numpy reference, step by step:

1. `pad`: truncate each waveform to `max_length` (480 000 samples = 30 s), pad with zeros to the
   longest truncated length rounded up to a multiple of `pad_to_multiple_of` (128 samples).
2. Semicausal padding: prepend `frame_length // 2` (160) zeros to waveform and sample mask.
3. Frames: unfold with size `frame_length + 1` (321) and step `hop_length` (160), keep the first
   320 samples (the extra sample only feeds preemphasis, which is 0 here).
4. Periodic Hann window (float32), `rfft(n=512)` in float32 (numpy 2 keeps float32 input in
   complex64), magnitude, matmul with the float64 HTK mel filterbank (128 filters, 0-8000 Hz,
   no norm), `log(mel + 1e-3)` in float64, cast to float32.
5. Mask: frame i is valid when the (left-padded) sample mask is 1 at i*160 + 320, i.e. original
   sample i*160 + 160 is real audio, so `valid = (length - 1) // 160` frames. Invalid frames are
   zeroed.

The window and filterbank are copied from an extractor instance, not rebuilt. The window product
is float32 as in the reference; the FFT, magnitude, mel matmul and log then run in float64
(`compute_dtype`). That matches numpy to one float32 ulp (4.8e-7 on random, quiet, band-limited,
codec-like and resampled test audio), whereas a float32 torch FFT (measured on CPU) differs by up
to ~3e-3 in near-empty mel bins of band-limited or resampled audio. float64 also keeps TF32, which
the trainer enables, out of the matmul. Dither, preemphasis, input scaling and per-bin
normalization are not implemented; the pinned EmbeddingGemma 2 config uses none of them and
construction fails if one is set.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class GpuLogMel(nn.Module):
    """`forward(waveforms [B, N] float, lengths [B] int)` -> (input_features [B, T, 128] float32,
    input_features_mask [B, T] bool), equal to the Gemma 4 extractor on `waveforms[b, :lengths[b]]`."""

    def __init__(
        self,
        window: np.ndarray,
        mel_filters: np.ndarray,
        *,
        frame_length: int = 320,
        hop_length: int = 160,
        fft_length: int = 512,
        mel_floor: float = 1e-3,
        max_length: int | None = 480_000,
        pad_to_multiple_of: int | None = 128,
        compute_dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.frame_length, self.hop_length, self.fft_length = frame_length, hop_length, fft_length
        self.mel_floor = float(mel_floor)
        self.max_length, self.pad_to_multiple_of = max_length, pad_to_multiple_of
        self.register_buffer("window", torch.from_numpy(np.asarray(window, dtype=np.float32).copy()), persistent=False)
        self.register_buffer("mel_filters", torch.from_numpy(np.asarray(mel_filters, dtype=np.float64).copy()).to(compute_dtype), persistent=False)

    @classmethod
    def from_extractor(cls, extractor) -> GpuLogMel:
        """Copy the window, filterbank and framing from a `Gemma4AudioFeatureExtractor`; the
        defaults of `__call__` (max_length 480 000, pad_to_multiple_of 128) are the ones used."""
        unsupported = {
            "dither": float(extractor.dither) != 0.0,
            "preemphasis": float(extractor.preemphasis) != 0.0,
            "input_scale_factor": float(extractor.input_scale_factor) != 1.0,
            "per_bin_mean": extractor.per_bin_mean is not None,
            "per_bin_stddev": extractor.per_bin_stddev is not None,
            "padding_value": float(extractor.padding_value) != 0.0,
            "padding_side": getattr(extractor, "padding_side", "right") != "right",
        }
        bad = [name for name, set_ in unsupported.items() if set_]
        if bad:
            raise NotImplementedError(f"GpuLogMel does not implement extractor settings {bad}")
        return cls(
            extractor.window,
            extractor.mel_filters,
            frame_length=extractor.frame_length,
            hop_length=extractor.hop_length,
            fft_length=extractor.fft_length,
            mel_floor=float(extractor.mel_floor),
        )

    def padded_length(self, lengths: torch.Tensor) -> int:
        longest = int(lengths.max()) if lengths.numel() else 0
        multiple = self.pad_to_multiple_of
        if multiple and longest % multiple:
            longest = (longest // multiple + 1) * multiple
        return longest

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.window.device
        lengths = lengths.long()
        if self.max_length is not None:
            lengths = lengths.clamp(max=self.max_length)
        padded = self.padded_length(lengths)  # shapes from (usually CPU) lengths: no device sync
        waveforms = waveforms.to(device, torch.float32, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        batch = waveforms.shape[0]
        # Zero everything past each row's (truncated) length, then right-pad to `padded`.
        waveforms = waveforms[:, :padded]
        samples = torch.arange(waveforms.shape[1], device=device)
        waveforms = waveforms * (samples[None, :] < lengths[:, None])
        pad_left = self.frame_length // 2
        waveforms = nn.functional.pad(waveforms, (pad_left, padded - waveforms.shape[1]))
        span = self.frame_length + 1  # the reference unfolds frame_length + 1 samples
        frames_count = max(0, (padded + pad_left - span) // self.hop_length + 1)
        bins = self.mel_filters.shape[1]
        if frames_count == 0:
            return (torch.zeros(batch, 0, bins, device=device),
                    torch.zeros(batch, 0, dtype=torch.bool, device=device))
        frames = waveforms.unfold(-1, self.frame_length, self.hop_length)[:, :frames_count]
        windowed = (frames * self.window).to(self.compute_dtype)  # float32 product, as in the reference
        spectrum = torch.fft.rfft(windowed, n=self.fft_length, dim=-1).abs()
        mel = torch.matmul(spectrum, self.mel_filters)
        features = torch.log(mel + self.mel_floor).to(torch.float32)
        valid = ((lengths - 1).clamp(min=0) // self.hop_length).clamp(max=frames_count)
        mask = torch.arange(frames_count, device=device)[None, :] < valid[:, None]
        return features * mask[..., None], mask


def pad_waveforms(audios: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad float32 mono waveforms with zeros: ([B, longest] float32, [B] int64 lengths)."""
    lengths = torch.tensor([len(a) for a in audios], dtype=torch.long)
    waveforms = torch.zeros(len(audios), int(lengths.max()) if len(audios) else 0, dtype=torch.float32)
    for row, audio in enumerate(audios):
        waveforms[row, : len(audio)] = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    return waveforms, lengths
