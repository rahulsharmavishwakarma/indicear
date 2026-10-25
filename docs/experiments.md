# Experiment log


Every training and evaluation run, with the settings needed to reproduce it or to describe it in a model card or open-weights release. Each run's full config, per-step curves and logs (no weights) are in `results/runs/<run>/`:
- `result.json`: variant, train config, data, environment, metrics;
- `train.csv`: per-step loss and LR;
- `train.log`;
- `gpu.csv`;
- `command.sh`.

Weights and checkpoints are not in git. The released model is [`rahulsharmavishwakarma/vesper-indicEAR`](https://huggingface.co/rahulsharmavishwakarma/vesper-indicEAR) on Hugging Face.

**Common to all runs:**
- Encoder: frozen `google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b`, audio tower only.
- Software: Python 3.14 free-threaded (GIL off), torch 2.14.1+cu130, AdamW.
- "Batch" is a dynamic frame budget: 25 encoder frames per second of audio, sized to free VRAM.

## Runs

| Run | Date | Hardware | Data | Variant | Batch (frames/step) | Steps | Peak LR (schedule) | Audio processed | Wall time | Key result |
|---|---|---|---|---|---:|---:|---|---:|---|---|
| Encoder probes (`results/probe-*`) | 23 Oct | Colab L4 | FLEURS train → dev, ~7 h/lang, 12 langs | Per-layer linear CTC, BPE-256 per language | 100 s of audio | 6 epochs | 2e-3 | — | — | Best-layer CER 16–29% Indic, 29.4% en |
| `fleurs-v1` | 23 Oct | L4 (24 GB) | FLEURS, 12 langs, 88 h | v1: frozen + linear CTC | 111,806 | 880 | 2.36e-3 (warmup 300, cosine to 5%) | 880 h (10 epochs) | 40 min | Dev CER 29–55% |
| Robustness probe | 23 Oct | L4 | FLEURS dev, 200/lang × 17 conditions | v1 heads | — | — | — | — | 15 min | Reverb collapses (`results/pilot-fleurs/robustness-v1`) |
| `fleurs-v2` | 23 Oct | A100 80 GB (3 runs sharing) | Same | v2: + adapters (rank 128, layers 7–12) | 24,226 | 3,540 | 1.10e-3 | 880 h | 79 min | Mean CER 27.67 (ex-ur) |
| `fleurs-v3` | 23 Oct | A100 (shared) | Same | v3: + phonetic self-conditioned CTC at layers 6 and 9 | 19,306 | 4,390 | 9.8e-4 | 880 h | 91 min | 28.17 |
| `fleurs-v4` | 23 Oct | A100 (shared) | Same | v4: + two-level script output | 20,871 | 4,080 | 1.02e-3 | 880 h | 91 min | **27.61**, fewest script errors |
| `pilot-v4lid` | 24 Oct | A100 80 GB (steps 0–3,788 and 3,809–7,423), Apple M5 Pro MPS fp16 (3,788–3,809) | `mixtures/pilot.json`: ~467 h streamed (hi/ta/kn from IndicVoices, Vaani, Kathbath; en from LibriSpeech, People's Speech, VoxPopuli, Vaani-en; mined code-mixed); 50% GPU augmentation, 15% codecs | v4 + utterance and frame LID | 67,045 (splits 1 on A100; 4 on MPS) | 7,423 | 1.83e-3 (warmup 300, cosine to 5%) | 2,664 h | ~4.3 A100-h | Dev CER 22.9 → 16.2; see `docs/pilot.md` |
| `full_warm` | 24–25 Oct | Colab G4, RTX PRO 6000 Blackwell 96 GB | `mixtures/full_warm.json`: 12 languages, 53 streams (Kathbath ml/or/ur/gu not drawn from step 1,079; Hindi ×2.24 from step 3,427; see `docs/full-run.md`) | v4+lid, warm-started from `pilot-v4lid` | 71,676 (2 micro-batches from step ~350) | 13,394 | 6e-4 → 5e-5 cosine (new rows ×3) | 8,356 h (packing 0.784) | ~6.9 h wall incl. restarts; ~6 GPU-h at full speed | Dev macro CER 25.1 (step 13,000). vs IndicConformer (same rows, sn-CER): 22–34 vs 3–12 on Kathbath/IndicVoices/Vaani/IndicContext; ahead only on MUCS. See `docs/full-run.md` |

## Pilot hyperparameters

Full values are in `results/runs/pilot-v4lid/result.json` → `train_config`.

| Group | Settings |
|---|---|
| Optimiser | AdamW, weight decay 0.001, grad clip 1.0, seed 0 |
| Adapters | Residual bottleneck, LN → down 1024→128 → SiLU → up (zero-init), after each block's FFN, layers 7–12 |
| Losses | CTC on the factored output. Self-conditioned phonetic CTC at layers 6 and 9 (weight 0.3, 82 units). LID utterance and frame weights 0.1 each, label smoothing 0.1, class-balanced |
| Tokenizer | Aggregate per-script SentencePiece unigram: deva 1024, latn 1024, other scripts 768, vocab 8,964 |
| Augmentation | clean_prob 0.5. Weights: noise 0.25, babble 0.20, reverb 0.175, reverb + point noise 0.175, speed (0.9/1.1) 0.15, tel8k 0.05. SNR 5–20 dB. RT60 0.3–0.9 s with 128 synthetic RIRs. Codecs: GSM/Opus/AMR, 15% (Mac: Opus only). SpecAugment on |
| Batching | Length-bucketed (30 buckets), frame budget autotuned to 85% VRAM, packing 0.85, live layers 1–6 |
| Precision | bf16 autocast on A100; fp16 + GradScaler on MPS |
| Checkpoints | Every 500 steps or 10 min, kept last 2 + best |

## Evaluations

| Eval | Where | Results |
|---|---|---|
| Pilot step 3,788, 25 sets | Apple M5 Pro | `results/pilot/eval-step3788/` |
| Pilot final, A100 eval | A100 | `results/pilot/eval-final/` |
| Pilot final vs IndicConformer 600M, 600 utterances per set | Apple M5 Pro | `results/pilot/eval-mac-final/`, `results/pilot/baseline-indicconformer/` |
| Pilot final + decode-time script constraints | Apple M5 Pro | `results/pilot/eval-mac-final-constrained/` (`indicear.decoding eval`, k=2, threshold 0.15, Latin split on) |
| Full run final, all 98 sets | G4 | `results/full/eval-g4/` (`ctc`, `ctc_constrained`) |
| Full run final vs IndicConformer 600M, 300 rows per set | Apple M5 Pro | `results/full/comparison-indicconformer.md` (sn-CER), `-strict-cer.md`; per-set results of every system in `results/bench/` |
| SraVaani-1.0 | `f5dd5358` | Run on the full-run eval subset (300 rows per language per set, Apple MPS): sn-CER Kathbath clean unseen 3.2, IndicVoices 7.4, Vaani 8.4; see full-run.md |

## Compute

About 55 Colab units for the pilot (A100 High-RAM at 6.77 units/h), plus about 2 for the L4 probes, and about 67 for the full run (RTX PRO 6000 at 8.9 units/h, including prep, restarts and evaluation). Pilot incidents are in `docs/pilot.md`: the compute balance ran out, the tokenizer was rebuilt, and a cuFFT error was fixed.

## Release

- **Model:** [`rahulsharmavishwakarma/vesper-indicEAR`](https://huggingface.co/rahulsharmavishwakarma/vesper-indicEAR): the trained adapters, heads and tokenizers (45.2 MiB), the EmbeddingGemma 2 audio tower bundled with its tensor values unchanged (Apache-2.0; extracted from Google's checkpoint, keys without the `audio_tower.` prefix), LICENSE and NOTICE, and a sha256 manifest of every file (`indicear-export --verify`).
- **Benchmark:** [`rahulsharmavishwakarma/vesper-indic-bench`](https://huggingface.co/datasets/rahulsharmavishwakarma/vesper-indic-bench): per-clip transcripts and scores of every system on identical clips.
- **Data licences** per source: `docs/data-licenses.md`. MUCS (CC BY-SA) is eval-only, and no non-commercial data was used.
