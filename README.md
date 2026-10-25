<p align="center">
  <img src="docs/assets/vesper-banner.png" alt="Vesper" width="720">
</p>

<h1 align="center">IndicEAR</h1>

<p align="center"><b>Embedding-Adapter Recognition: the training framework behind Vesper-IndicEAR</b><br>
Part of the Vesper project · research release v0.1</p>

<p align="center">
  <a href="https://huggingface.co/rahulsharmavishwakarma/vesper-indicEAR">Model</a> ·
  <a href="https://huggingface.co/datasets/rahulsharmavishwakarma/vesper-indic-bench">Benchmark</a> ·
  <a href="docs/training-recipe.md">Training recipe</a> ·
  <a href="notebooks/quickstart.ipynb">Quickstart</a>
</p>

IndicEAR turns the frozen audio encoder of an embedding model, EmbeddingGemma 2, into a speech recogniser. Its first model, **Vesper-IndicEAR**, transcribes 11 Indian languages and English, including code-switched speech, writes each word in its own script, and identifies the language. The encoder is never modified; only 11.8M parameters (45 MB) of adapters, CTC heads and language-ID heads are trained.

<p align="center"><img src="docs/figures/fig01_architecture.png" width="100%"></p>

| | |
|---|---|
| Languages | Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, Urdu |
| Model | `google/embeddinggemma-2` audio tower (frozen, 305M) + residual adapters on layers 7–12 + script-factored CTC over 11 scripts + utterance/frame language ID |
| Training | about 6 GPU-hours on one RTX PRO 6000, streaming 8,356 h of audio from the Hugging Face Hub, warm-started from a 4.3 A100-hour pilot |
| Status | **Research release.** It works, but at several times the error rate of dedicated Indic ASR models. Not yet integrated into the Vesper app |

## Results

<p align="center"><img src="docs/figures/fig05_heldout_comparison.png" width="100%"></p>

Script-normalised CER on held-out sets with unseen speakers; every system decodes the same 300 clips per language per set. Vesper-IndicEAR is several times behind IndicConformer and SraVaani, and ahead only on MUCS. The [model card](docs/model-card.md) has the table.

## Use

```bash
pip install indicear
indicear-transcribe --model rahulsharmavishwakarma/vesper-indicEAR talk.wav
indicear-transcribe --model rahulsharmavishwakarma/vesper-indicEAR --json call.m4a   # words, times, language, spans
```

```python
from indicear.transcribe import Transcriber

with Transcriber("rahulsharmavishwakarma/vesper-indicEAR") as model:
    result = model.transcribe_file("talk.wav")
    print(result["text"], result["language"]["top"])
```

The model package (627 MiB) includes the encoder's audio tower with its tensor values unchanged from Google's checkpoint, so nothing else is downloaded. Device `auto` picks CUDA, then Apple MPS, then the CPU.

## Train

The framework streams training audio from the Hugging Face Hub (no corpus download), keeps the GPU saturated with per-stream read-ahead and GPU-side augmentation, resumes exactly from any checkpoint, and evaluates every system on identical clips.

- [Training recipe](docs/training-recipe.md): architecture, data, hyperparameters, hardware and cost, exact commands, and how the framework handles streaming, batching, GPU utilisation and resumable training.
- [Colab notebook](notebooks/train_colab.ipynb): the full recipe (pilot, then the 12-language run) on one Colab GPU.
- [Framework and code guide](docs/framework.md).

## Benchmark

**Vesper Indic Bench** publishes the clips (by source dataset and id), references where their licences allow, and every system's per-clip transcript and scores: Vesper-IndicEAR in three decoding modes, IndicConformer 600M and SraVaani-1.0. `benchmark/build_bench.py` builds it from the evaluation outputs; `benchmark/vesper_slice.py` and `benchmark/score_vesper.py` score the model on Vesper's public ASR benchmark clips.

## Repository

| Path | Contents |
|---|---|
| `indicear/` | The package: model, staged encoder, streaming data, trainer, evaluation, decoding, export, transcription |
| `indicear/runtime/` | Model-agnostic training infrastructure: batching, checkpoints, sharded caches, GPU sampling |
| `benchmark/` | Benchmark dataset builder and the Vesper ASR benchmark scorer |
| `scripts/` | Stage scripts for a GPU VM and helpers for pulling runs to a workstation |
| `notebooks/` | Quickstart, Colab training, and the first ablation |
| `docs/` | Model card, recipe, run reports, design, data licences; `docs/figures/make_figures.py` regenerates every figure |
| `results/` | Run configs, training curves, logs and evaluation summaries |
| `tests/` | CPU test suite (`uv run --locked pytest`) |

## Documentation

- [Model card](docs/model-card.md) · [Training recipe](docs/training-recipe.md) · [Full run](docs/full-run.md) · [Pilot](docs/pilot.md) · [Experiments log](docs/experiments.md)
- [Architecture](docs/architecture.md) · [Framework / code guide](docs/framework.md) · [Original plan](docs/PLAN.md)
- [Data licences](docs/data-licenses.md) · [Loanword normaliser](docs/loanwords.md)
- [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Citation](CITATION.cff)

The project was called Sangam until 25 October 2026; run logs and records under `results/` and the model package's `training/` files keep that name.

## Licence

Code and trained weights: [Apache-2.0](LICENSE). The bundled EmbeddingGemma 2 audio tower is Google's, under Apache-2.0: its tensors are extracted from the checkpoint with values unchanged (see [NOTICE](NOTICE)). Figures use Bricolage Grotesque (SIL Open Font License, `docs/figures/fonts/`). Training data are CC BY (2.0–4.0) and CC0 sources; see [NOTICE](NOTICE) and [docs/data-licenses.md](docs/data-licenses.md).
