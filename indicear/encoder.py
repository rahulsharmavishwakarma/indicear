"""The frozen EmbeddingGemma 2 audio tower, run in two stages, with residual adapters.

Call path (transformers 5.20.2, `Gemma4AudioModel.forward`), reproduced with the model's own
modules and mask helpers so nothing is reimplemented:

    hidden, valid = tower.subsample_conv_projection(input_features, input_features_mask)
    position_embeddings = tower.rel_pos_enc(hidden)          # depends only on dtype/device
    mask = create_bidirectional_mask(config, hidden, valid,
                                     and_mask_function=sliding_window_mask_function(
                                         (attention_context_left - 1, attention_context_right)))
    mask = tower._convert_4d_mask_to_blocked_5d(mask)       # if not None
    for layer in tower.layers: hidden = layer(hidden, attention_mask=mask,
                                              position_embeddings=position_embeddings)
    last_hidden_state = tower.output_proj(hidden)

`lower()` runs layers 1-6 (cached once); `upper()` rebuilds the same mask and position
embeddings from the valid-frame mask and runs layers 7-12. With attention_context_right = 0 and
a causal depthwise conv, valid frames never depend on padding, so per-utterance caching is exact.

Adapters are forward hooks on `layers[i].feed_forward2` (after the block's second FFN, before its
clamp and `norm_out`); the base modules and weights are untouched and the adapters save to their
own safetensors file.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset

from indicear.common import CACHE_LAYER, EG2, SAMPLE_RATE

LayerCallback = Callable[[torch.Tensor], torch.Tensor]


DEVICE_ENV = "INDICEAR_DEVICE"  # auto | cuda | mps | cpu; `--device` overrides it
DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def default_dtype() -> torch.dtype:
    # L4 and A100 run bfloat16; a T4 has no bfloat16 kernels (as in probe.py). MPS and the CPU run
    # float32 (bfloat16 training on MPS hits an MPS matmul assertion: "Destination NDArray and
    # Accumulator NDArray cannot have different datatype").
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def select_device(name: str | None = None) -> str:
    """"cuda", "mps" or "cpu": `name` if given, else $INDICEAR_DEVICE, else auto (CUDA, then Apple
    MPS, then the CPU). An explicit device that is not available stops the run. Choosing MPS also
    sets PYTORCH_ENABLE_MPS_FALLBACK=1 (unsupported ops run on the CPU); torch reads it when it is
    imported, so entry points and scripts set it before importing torch as well."""
    choice = (name or os.environ.get(DEVICE_ENV) or "auto").strip().lower()
    if choice not in DEVICE_CHOICES:
        raise SystemExit(f"device {choice!r} not in {DEVICE_CHOICES} (--device or ${DEVICE_ENV})")
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "mps" if mps_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("device cuda requested but CUDA is not available")
    if choice == "mps":
        if not mps_available():
            raise SystemExit("device mps requested but Apple MPS is not available")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return choice


def runtime_device(name: str | None = None) -> tuple[str, torch.dtype]:
    """(device, dtype) for training and evaluation: ("cuda", bf16 or fp32) on a GPU VM, ("mps",
    fp32) on Apple silicon, ("cpu", fp32) for local smoke tests. See `select_device`."""
    device = select_device(name)
    return device, default_dtype() if device == "cuda" else torch.float32


def load_tower(device: str | torch.device, dtype: torch.dtype | None = None, path: str | Path | None = None) -> nn.Module:
    """The pinned EG2 audio tower, frozen, in eval mode: from `path` when given (a model package's
    `encoder/`, the tower extracted unchanged by `export --bundle-encoder`), else from the full
    pinned checkpoint (probe.py's proven loading path)."""
    from transformers import AutoModel

    dtype = dtype or default_dtype()
    if path is not None:
        tower = AutoModel.from_pretrained(str(path), dtype=dtype).to(device).eval()
    else:
        model = AutoModel.from_pretrained(EG2[0], revision=EG2[1], dtype=dtype)
        tower = model.audio_tower.to(device).eval()
        del model
    for parameter in tower.parameters():
        parameter.requires_grad_(False)
    return tower


def half_safe_tower(tower: nn.Module, dtype: torch.dtype = torch.float16) -> None:
    """Make the frozen tower's two large constants representable in `dtype` (bfloat16 or
    float16), for half-precision autocast on MPS over float32 weights (transformers 5.20.2):

    - the conformer blocks' activation clamp, config.gradient_clipping = 1e10, which
      Gemma4AudioLayer caps with finfo(its *weights'* dtype).max = 3.4e38, so torch.clamp(half
      tensor, +-3.4e38) raises "value cannot be converted to type c10::Half without overflow":
      capped at finfo(dtype).max (65504 for float16, 3.39e38 for bfloat16);
    - the attention's masked-logit fill, config.attention_invalid_logits_value = -1e9, applied to
      autocast half-precision logits: raised to -finfo(dtype).max when it is out of range.

    Neither changes float32 results: activations stay far below the cap, and a masked logit next
    to soft-capped logits (|x| <= 50) has exactly zero softmax weight either way."""
    limit = float(torch.finfo(dtype).max)
    for module in tower.modules():
        clip = getattr(module, "gradient_clipping", None)
        if isinstance(clip, (int, float)) and clip > limit:
            module.gradient_clipping = limit
        config = getattr(module, "config", None)
        fill = getattr(config, "attention_invalid_logits_value", None)
        if isinstance(fill, (int, float)) and fill < -limit:
            config.attention_invalid_logits_value = -limit


MPS_RECOMPILE_LIMIT = 256  # graphs per layer: frame buckets x {no_grad lower, grad upper} x eval shapes


def compile_tower_layers(tower: nn.Module, modules: tuple[nn.Module, ...] = ()) -> None:
    """torch.compile every conformer layer of the tower (plus `modules`, e.g. the heads) in place,
    for MPS. Shapes: the batch dimension is marked dynamic by `StagedTower` and automatic dynamic
    shapes are off, so each frame length is its own static graph (static graphs measured ~2x faster
    than dynamic-T ones on MPS); the trainer pads frames to a few buckets (`LiveLower.frame_bucket`)
    to bound the recompiles. The state_dict keys are unchanged."""
    import torch._dynamo

    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, MPS_RECOMPILE_LIMIT)
    torch._dynamo.config.accumulated_recompile_limit = max(
        torch._dynamo.config.accumulated_recompile_limit, 4 * MPS_RECOMPILE_LIMIT)
    for module in (*tower.layers, *modules):
        module.compile()


class StagedTower(nn.Module):
    """Runs a frozen Gemma4AudioModel as layers [0, split) and [split, num_layers)."""

    def __init__(self, tower: nn.Module, split: int = CACHE_LAYER):
        super().__init__()
        self.tower = tower
        self.split = split
        self.num_layers = tower.config.num_hidden_layers
        self.hidden_size = tower.config.hidden_size
        self.dynamic_batch = False  # compiled layers: mark the batch dimension dynamic

    def _mark(self, *tensors) -> None:
        if self.dynamic_batch:
            for tensor in tensors:
                if isinstance(tensor, torch.Tensor) and tensor.ndim and tensor.shape[0] > 1:
                    torch._dynamo.maybe_mark_dynamic(tensor, 0)

    def masks(self, hidden: torch.Tensor, valid: torch.Tensor):
        from transformers.models.gemma4 import modeling_gemma4 as g4

        config = self.tower.config
        position_embeddings = self.tower.rel_pos_enc(hidden)
        mask = g4.create_bidirectional_mask(
            config=config,
            inputs_embeds=hidden,
            attention_mask=valid,
            and_mask_function=g4.sliding_window_mask_function(
                (config.attention_context_left - 1, config.attention_context_right)
            ),
        )
        if mask is not None:
            mask = self.tower._convert_4d_mask_to_blocked_5d(mask)
        return mask, position_embeddings

    @torch.no_grad()
    def lower(
        self, input_features: torch.Tensor, input_features_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Layers 1..split. Returns (hidden [B, T, 1024], valid [B, T] bool)."""
        hidden, valid = self.tower.subsample_conv_projection(input_features, input_features_mask)
        mask, position_embeddings = self.masks(hidden, valid)
        self._mark(hidden, mask)
        for layer in self.tower.layers[: self.split]:
            hidden = layer(hidden, attention_mask=mask, position_embeddings=position_embeddings)
        return hidden, valid

    def upper(
        self,
        hidden: torch.Tensor,
        valid: torch.Tensor,
        callbacks: dict[int, LayerCallback] | None = None,
        checkpoint: bool = False,
        stop_after: int | None = None,
    ) -> torch.Tensor:
        """Layers split+1..num_layers (1-based). `callbacks[k]` maps the output of layer k
        (k >= split) before the next layer: used for self-conditioned CTC at layers 6 and 9."""
        callbacks = callbacks or {}
        mask, position_embeddings = self.masks(hidden, valid)
        self._mark(hidden, mask)
        if self.split in callbacks:
            hidden = callbacks[self.split](hidden)
        last = stop_after or self.num_layers
        for index in range(self.split, last):
            layer = self.tower.layers[index]
            if checkpoint and torch.is_grad_enabled():
                hidden = torch.utils.checkpoint.checkpoint(
                    layer, hidden, mask, position_embeddings, use_reentrant=False
                )
            else:
                hidden = layer(hidden, attention_mask=mask, position_embeddings=position_embeddings)
            if index + 1 in callbacks:
                hidden = callbacks[index + 1](hidden)
        return hidden


def lengths_to_valid(lengths: torch.Tensor, frames: int) -> torch.Tensor:
    return torch.arange(frames, device=lengths.device)[None, :] < lengths[:, None]


# ---------------------------------------------------------------- adapters


class BottleneckAdapter(nn.Module):
    """x -> up(act(down(norm(x)))): zero-initialized `up`, so it starts as an exact identity."""

    def __init__(self, dim: int, rank: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        self.act = nn.SiLU()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(self.norm(hidden))))


class AdapterSet(nn.Module):
    """Residual adapters after `feed_forward2` of encoder layers `layers` (1-based)."""

    def __init__(self, dim: int, layers: tuple[int, ...] = (7, 8, 9, 10, 11, 12), rank: int = 128):
        super().__init__()
        self.layer_numbers = tuple(layers)
        self.rank = rank
        self.adapters = nn.ModuleDict({str(n): BottleneckAdapter(dim, rank) for n in layers})
        self._handles: list = []

    def attach(self, tower: nn.Module) -> AdapterSet:
        self.detach()
        for number in self.layer_numbers:
            adapter = self.adapters[str(number)]

            def hook(_module, _inputs, output, adapter=adapter):
                return output + adapter(output).to(output.dtype)

            ffn = tower.layers[number - 1].feed_forward2
            self._handles.append(ffn.register_forward_hook(hook))
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    def save(self, path: str | Path) -> None:
        from safetensors.torch import save_file

        tensors = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(tensors, str(path), metadata={"layers": ",".join(map(str, self.layer_numbers)),
                                                "rank": str(self.rank), "encoder": f"{EG2[0]}@{EG2[1]}"})

    def load(self, path: str | Path) -> None:
        from safetensors.torch import load_file

        self.load_state_dict(load_file(str(path)))


# ---------------------------------------------------------------- feature extraction


def load_audio(path: str) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        import soxr

        audio = soxr.resample(audio, rate, SAMPLE_RATE).astype(np.float32)
    return audio


FEATURE_BACKENDS = ("cpu", "gpu")
FEATURE_TOLERANCE = 1e-4  # max |GPU - CPU| log-mel on the first batch (float64 FFT: ~5e-7 measured)


def default_feature_backend() -> str:
    """"gpu" (torch log-mel on the model device) when CUDA is available, else "cpu"."""
    return "gpu" if torch.cuda.is_available() else "cpu"


class FeatureBatches(Dataset):
    """Each item is one batch, built in a worker process (probe.py's pattern, so the GIL never
    starves the GPU). `transform(audio, rng, item)` lets the robustness probe and the augmentation
    recipe corrupt audio inside the worker.

    backend "cpu": the worker also computes Gemma 4 log-mel features with the Hugging Face
    extractor (numpy) and the item is its output. backend "gpu": the item is
    {"waveforms": [B, N] float32 zero-padded, "lengths": [B] int64} and `Featurizer` computes the
    same features on the model's device in the main process."""

    def __init__(
        self,
        groups: list[list[dict]],
        transform: Callable[[np.ndarray, np.random.Generator, dict], np.ndarray] | None = None,
        seed: int = 0,
        backend: str = "cpu",
    ):
        if backend not in FEATURE_BACKENDS:
            raise ValueError(f"feature backend {backend!r} not in {FEATURE_BACKENDS}")
        self.groups, self.transform, self.seed, self.backend, self.extractor = groups, transform, seed, backend, None

    def __len__(self) -> int:
        return len(self.groups)

    def audio(self, item: dict, rng: np.random.Generator) -> np.ndarray:
        audio = load_audio(item["audio"])
        if self.transform is not None:
            audio = self.transform(audio, rng, item)
        return audio

    def __getitem__(self, index: int):
        rng = np.random.default_rng((self.seed, index))
        audios = [self.audio(item, rng) for item in self.groups[index]]
        if self.backend == "gpu":
            from indicear.features import pad_waveforms

            waveforms, lengths = pad_waveforms(audios)
            return index, {"waveforms": waveforms, "lengths": lengths}
        if self.extractor is None:  # loaded once per worker process
            self.extractor = load_extractor()
        # Gemma 4's extractor always returns its own `input_features_mask`.
        features = self.extractor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        return index, dict(features)


def load_extractor(path: str | Path | None = None):
    """The pinned EG2 `Gemma4AudioFeatureExtractor` (numpy, CPU), from `path` when given (a model
    package's `encoder/`)."""
    from transformers import AutoFeatureExtractor

    if path is not None:
        return AutoFeatureExtractor.from_pretrained(str(path))
    return AutoFeatureExtractor.from_pretrained(EG2[0], revision=EG2[1])


WORKER_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


class _SingleThreadWorkers(DataLoader):
    """Spawned workers start with OMP/OpenBLAS/MKL limited to one thread each: N workers then
    use N cores, not N x cores (the parent's own settings are restored right after the spawn)."""

    def __iter__(self):
        saved = {k: os.environ.get(k) for k in WORKER_THREAD_VARS}
        os.environ.update({k: "1" for k in WORKER_THREAD_VARS})
        try:
            return super().__iter__()
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def feature_loader(dataset: FeatureBatches, workers: int) -> DataLoader:
    return _SingleThreadWorkers(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=4 if workers else None,
        pin_memory=torch.cuda.is_available(),
        multiprocessing_context="spawn" if workers else None,
    )


def to_device(features: dict, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = features["input_features"].to(device, non_blocking=True).to(dtype)
    mask = features["input_features_mask"].to(device, non_blocking=True)
    return inputs, mask


def feature_device(device: torch.device | str) -> torch.device:
    """Where `GpuLogMel` runs for a model on `device`: that device, except Apple MPS, which has no
    float64 (the log-mel's FFT, mel matmul and log run in float64): there it runs on the CPU, in
    the main process, and the features move to MPS (as indicear.transcribe does)."""
    device = torch.device(device)
    return torch.device("cpu") if device.type == "mps" else device


class Featurizer:
    """Main-process half of the feature path: a `FeatureBatches` item -> (input_features in the
    model dtype, input_features_mask) on the model device. For the "gpu" backend it runs
    `GpuLogMel` (copied from the pinned extractor) on the padded waveforms, on `feature_device`
    (the model device, or the CPU for MPS); `check_once` compares it with the extractor itself on
    the first batch."""

    def __init__(self, backend: str, device: torch.device | str, dtype: torch.dtype):
        if backend not in FEATURE_BACKENDS:
            raise ValueError(f"feature backend {backend!r} not in {FEATURE_BACKENDS}")
        self.backend, self.device, self.dtype = backend, torch.device(device), dtype
        self.extractor = self.logmel = None
        self.checked = backend == "cpu"
        if backend == "gpu":
            from indicear.features import GpuLogMel

            self.extractor = load_extractor()
            self.logmel = GpuLogMel.from_extractor(self.extractor).to(feature_device(self.device))

    @classmethod
    def from_extractor(cls, extractor, device: torch.device | str, dtype: torch.dtype) -> Featurizer:
        """A "gpu"-backend featurizer around an extractor already in hand (no Hub download)."""
        from indicear.features import GpuLogMel

        featurizer = cls("cpu", device, dtype)
        featurizer.backend, featurizer.checked, featurizer.extractor = "gpu", False, extractor
        featurizer.logmel = GpuLogMel.from_extractor(extractor).to(feature_device(featurizer.device))
        return featurizer

    def __call__(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        if self.logmel is None:
            return to_device(batch, self.device, self.dtype)
        features, mask = self.logmel(batch["waveforms"], batch["lengths"])
        return features.to(self.device, self.dtype), mask.to(self.device)

    def check_once(self, batch: dict, log: Callable[[str], None], tolerance: float = FEATURE_TOLERANCE) -> float | None:
        """On the first call with the "gpu" backend: compute the batch's features with the
        Hugging Face extractor too, log the max abs difference and stop on a mask or shape
        mismatch or a difference above `tolerance`. A no-op afterwards and for "cpu"."""
        if self.checked:
            return None
        self.checked = True
        audios = [w[:n].numpy() for w, n in zip(batch["waveforms"], batch["lengths"].tolist(), strict=True)]
        reference = self.extractor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        features, mask = self.logmel(batch["waveforms"], batch["lengths"])
        features, mask = features.cpu(), mask.cpu()
        expected, expected_mask = reference["input_features"], reference["input_features_mask"].bool()
        if features.shape != expected.shape or not torch.equal(mask, expected_mask):
            raise SystemExit(
                f"GPU log-mel shape {tuple(features.shape)} / mask differ from the extractor's "
                f"{tuple(expected.shape)}; rerun with --features cpu"
            )
        diff = float((features - expected).abs().max()) if features.numel() else 0.0
        log(f"features: GPU log-mel vs Hugging Face extractor on the first batch max |diff| = {diff:.3g}, mask equal")
        if diff > tolerance:
            raise SystemExit(f"GPU log-mel differs from the extractor by {diff:.3g} > {tolerance:g}; rerun with --features cpu")
        return diff
