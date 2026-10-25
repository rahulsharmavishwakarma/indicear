# IndicEAR framework: training, evaluation and packaging

Research code for design A (`docs/architecture.md`) on the frozen EmbeddingGemma 2 audio encoder (`google/embeddinggemma-2@914f7f89`, transformers 5.20.2). The encoder weights are never modified.

- **Layers 1–6** run frozen: either cached once per utterance or computed live every step (`--lower live`, used by both released runs, since their audio streams).
- **Layers 7–12** run every step with trainable residual adapters attached through forward hooks.

**Status (25 October 2026):** the pilot and the full 12-language run are done; results are in [pilot.md](pilot.md) and [full-run.md](full-run.md), and the end-to-end recipe is [training-recipe.md](training-recipe.md). Sections written before the GPU runs (estimates, "unverified") are kept as written. The contextual-biasing experiment (`biasing_test`, `wordspot`) was not used by the released model and is not part of this repository. The pilot (live streaming training of v2, v3, v4 and v4+lid, evaluation and baselines) launches with `bash scripts/pilot.sh`; see "The pilot".


## Layout

| Path | Contents |
| --- | --- |
| `indicear/runtime/` | **Model-agnostic infrastructure, the framework candidate.** No imports from the ASR modules. `batching` (length buckets, frame budget, VRAM sizing, OOM backoff, LR scaling), `checkpoint` (atomic saves, `LATEST.json` manifest, DONE/FAILED, signals), `shards` (sharded row cache with completion markers, df check, VRAM/pinned/mmap placement), `gpu` (5 s GPU sampler: nvidia-smi or Apple `ioreg`; MPS memory accounting and micro-batch sizing), `io`, `parallel` (several jobs on one GPU) |
| `indicear/trainer.py` | Training loop: resume, autotune, prefetch thread, bf16, `torch.compile` on heads. Generic except that it calls `model.losses()`; the next framework candidate. `LiveGroup`: several variants trained from one stream in one process |
| `indicear/stream_train.py` | Live-lower protocol (`StreamSource`, `LoaderSource`), batch preparation, `augment_live` |
| `indicear/gpu_augment.py` | On-the-fly noise, babble, RIR reverb, point noise, speed and tel8k on the GPU, before the log-mel; micro-benchmark |
| `indicear/evaluate.py` | Eval sets, layer-6 lowering of eval audio, decoding several models at once, scoring, `results/<system>/<set>.json`, `eval_summary` |
| `indicear/baselines.py` | Other ASR systems on the same eval sets (IndicConformer, EmbeddingGemma 2 community models); runs in the `baselines` dependency group's Python 3.13 environment |
| `indicear/smoke.py` | CPU smoke harness: tiny tower, synthetic shards, no download; drives `scripts/pilot.sh` with `SMOKE=1` |
| `indicear/encoder.py` | Staged frozen tower (layers 1–6, then 7–12), adapters, the feature path (worker audio loading, `--features cpu\|gpu`) |
| `indicear/features.py` | `GpuLogMel`: the Gemma 4 log-mel extractor in torch, batched on the model's device |
| `indicear/cache.py` | Layer-6 cache build (feature workers, encode, verify the features and the call path on real weights) |
| `indicear/heads.py`, `losses.py`, `model.py` | Self-conditioned CTC, script-factored output, factored RNN-T, CR-CTC, variants v1–v6 |
| `indicear/text.py`, `phonetic.py`, `tokenizer.py` | Normalization, scripts, gloss rewrite, shared phonetic units, per-script SentencePiece |
| `indicear/data.py`, `augment.py`, `metrics.py` | FLEURS, synthetic code-switch set, corruptions, metrics |
| `indicear/sources.py` | Source registry: repo, pinned revision, licence/SPDX, gating, schema, transcript convention, verification notes ([docs/data-licenses.md](data-licenses.md)) |
| `indicear/markup.py` | Per-source transcript markup → IndicEAR convention (Vaani `{gloss}`, IndicVoices `[gloss]` and tags, IndicContextEval `(gloss)`), drop reasons, `code_mixed`, per-word language tags |
| `indicear/ingest.py` | Download path: resumable shards with completion markers, disk guard, parallel decode to 16 kHz FLAC (eval/dev sets, tar-only sources, small full downloads) |
| `indicear/stream.py` | **Streaming training path**: per-(source, language) streams from the Hub, temperature interleave, shuffle buffer, duration-bucketed batches, bounded shard cache, exact resume |
| `indicear/mixture.py`, `indicear/mixtures/{pilot,full,full_warm}.json` | Mixture configs, speaker-disjoint dev, eval-leak checks, temperature sampling, size estimates (`python -m indicear.mixture plan`) and processed hours of a live run (`python -m indicear.mixture hours`) |
| `indicear/warmstart.py`, `indicear/prep_pack.py` | Warm start of a new run from a finished one (`ablate train --init-from`: tokenizer extension, head expansion, row map); packing and restoring a prepared ROOT; see "The full warm-started run" |
| `indicear/loanwords.py`, `indicear/resources/` | Loanword normaliser for native-only transcripts (Kathbath), its lexicons, review samples and the English word list with its licence; see [docs/loanwords.md](loanwords.md) |
| `indicear/ablate.py`, `robustness_probe.py` | The training/evaluation and robustness entry points |
| `indicear/export.py`, `indicear/transcribe.py` | Model packages from trained runs (`--verify` checks them) and transcription of audio files with a package; see "Using a trained model" |
| `indicear/decoding.py` | Decode-time script constraints for the script-factored CTC output (word-boundary script rule, LID script set) and `python -m indicear.decoding eval`, the constrained-versus-unconstrained evaluation of a package; see "Decode-time script constraints" |
| `scripts/` | VM: `pilot` (the whole pilot chain), `full_warm` (the warm-started full run), `setup_env`, `prep`, `cache`, `ablation`, `robustness`, `baselines`, `launch` (nohup, WAIT_FOR), `status`, `stop`, `musan_noise`, `rirs` (real RIRs). Workstation: `resume_local` (resume a run on MPS), `sync_runs`, `fetch_tok` (the ROOT's tokenizer), `resume_on`, `push_init` (a warm-start source to a VM), `colab_ssh`, `lib/transport.sh` |
| `notebooks/` | `quickstart.ipynb`, `train_colab.ipynb` and `06_ablation.ipynb` (the first FLEURS ablation), all generated by `notebooks/make_notebooks.py` |
| `requirements-colab.txt` | `uv export` of `pyproject.toml` + `uv.lock` (torch 2.14.1 CUDA 13 on Linux) |

## Design A as implemented

| Piece | Implementation |
| --- | --- |
| Frozen encoder | Same modules and mask helpers as `Gemma4AudioModel.forward`: `subsample_conv_projection` → `rel_pos_enc` → `create_bidirectional_mask` + `sliding_window_mask_function((left-1, right))` → `_convert_4d_mask_to_blocked_5d` → layers → `output_proj`. Pinned config: chunk 12, left context 13, **right context 0**, so the encoder has no look-ahead. |
| Layer-6 cache | Per utterance, bf16 (or per-row int8), parts with an index file as the completion marker |
| Adapters, layers 7–12 | `LN → down(1024→128) → SiLU → up(→1024)` added after each block's `feed_forward2` (before the clamp and `norm_out`). `up` is zero-initialized, so the model starts as the frozen encoder. Saved to `adapters.safetensors`. |
| Self-conditioned phonetic CTC at 6 and 9 | InterCTC (arXiv 2104.02724), LUPET-style units (arXiv 2401.03689): `x' = x + B·softmax(W·LN(x))` with B zero-initialized. Weight 0.3 on the mean of the two. |
| Phonetic units (82) | Rule tables in `phonetic.py`. Brahmic scripts use ISO 15919-style labels via the shared ISCII block layout. English uses Indian-accent spelling rules (retroflex t/d, w→v, magic-e). Urdu uses a consonant-skeleton romanization, which is lossy. |
| Output units | One SentencePiece unigram model per script (11 scripts: deva 1024, latn 1024, the others 768), each with the script's full character inventory as `required_chars` and `identity` normalization. Any word can be spelled. |
| Two-level output | `log p(token) = log p(script) + log p(token given script)`, with blank as script 0 |
| Final head | v1–v5: CTC. v6: factored RNN-T plus a CTC branch at weight 0.3. TDT is **not** implemented (see open issues). |
| CR-CTC (arXiv 2410.05101) | Two hidden-state SpecAugment views, symmetric KL with stop-gradient, weight 0.2, time masking ×2.5 |
| Decode-time script constraints | `indicear.decoding`, no retraining: a word-boundary script rule and an utterance script set from the LID heads mask the two-level output while decoding (CTC greedy and prefix beam). Off in training and in `DesignA.decode` by default; on by default in the transcribe CLI and `indicear.decoding eval`. See "Decode-time script constraints". |
| RNN-T loss | Exact pure-PyTorch recursion on gathered log-probs; only the target script's segment is softmaxed. It matches `torchaudio.functional.rnnt_loss` (values and gradients) in the tests. |

Variants: **v1** linear CTC on frozen layer 12 · **v2** + adapters · **v3** + phonetic self-conditioned CTC at layers 6 and 9 · **v4** + two-level script output (CTC form) · **v5** + CR-CTC · **v6** + factored RNN-T head.

## Data: sources, mixtures and streaming

Licences, revisions, gating and the transcript conventions found are in [docs/data-licenses.md](data-licenses.md). That page is generated together with `docs/data-licenses.json` from `indicear/sources.py`.

**Mixtures.**
- **`pilot`** is PLAN.md phase 1, 650 h:
  - Hindi 200 h: IndicVoices 100, Vaani 70, Kathbath 30.
  - Tamil 150 h: IndicVoices 113, Vaani 12, Kathbath 25.
  - Kannada 150 h: IndicVoices 80, Vaani 45, Kathbath 25.
  - English 100 h: LibriSpeech 30, People's Speech 30, VoxPopuli 26, Vaani Indian English 14.
  - Code-switched 50 h: MUCS Hi-En 30, plus Vaani code-mixed mining 20 (Hindi 12, Kannada 8).
- **`full`** is the PLAN.md 12-language mixture, about 10.2K h, or 10.7K h with the review sources. Common Voice is blocked; Shrutilipi has no teacher-CER filter yet.
- **Sampling:** temperature sampling over languages plus a code-switched group (`cs`), with α = 0.5. The pilot caps Hindi at a 25% share and floors `cs` at 8%; the full mixture uses 20% and 6%. PLAN.md gives no number for the Hindi cap, so these values are a choice.
- **Licence gate:** `review` sources (MUCS, SPRING-INX) are evaluation-only unless you pass `--allow-review mucs` (recorded in the report). Without it, the pilot trains on 620 h.

**Streaming is the training path.** The full mixture needs 0.7–1.1 TB of audio, and a Colab VM has about 190 GB free. So only the fixed sets are downloaded:
- the eval sets;
- a speaker-disjoint dev set, made of whole speaker groups from one shard per stream;
- tar-only sources (MUCS).

Training audio streams from the Hub shard by shard:

```bash
export HF_TOKEN=...   # read from the environment only; the account must have accepted each gate
python -m indicear.mixture plan --mixture pilot --allow-review mucs          # offline size estimate
python -m indicear.ablate --root R prep --mixture pilot --stream [--allow-review mucs] [--disk-budget-gb 60]
```

`prep --stream` does four things:
1. Downloads and decodes the fixed sets (`manifests/eval_*.jsonl`, `dev.jsonl`, `local_train.jsonl`) and writes `manifests/stream_exclude.json` (the eval and dev speakers, speaker groups, keys and audio hashes).
2. Picks each stream's shard pool and writes `manifests/stream_plan.json`.
3. Reads **only the text columns** of every pool shard, over HTTP range requests, into `train_texts.jsonl`. These feed the tokenizer and the label-rate report.
4. Writes the licence manifest. No training audio is written to disk.

On the VM: `PREP_ARGS="--mixture pilot --stream --allow-review mucs" bash scripts/launch.sh prep scripts/prep.sh`, with `HF_TOKEN` exported in the job's environment.

Without `--stream`, `prep --mixture pilot` downloads and decodes the whole mixture to FLAC instead (about 100 GB downloaded and 55 GB of FLAC for the pilot); use that only for small mixtures. `--mixture fleurs`, the default, is the original FLEURS ablation, unchanged.

**The stream feeds the trainer** (`ablate train --lower live`, below). The loader can also be used directly:

```python
from indicear.augment import CodecAugment
from indicear.stream import StreamLoader, build_planner

setup = build_planner(root, "pilot", frame_budget=budget_frames, allow_review={"mucs"},
                      buffer_seconds=1800, cache_gb=40, augment_prob=0.5)
if resume_state:                                  # from the checkpoint
    setup.planner.load_state(resume_state)
loader = StreamLoader(setup.planner, transform=CodecAugment(0.15), workers=8, cache=setup.cache)
for batch in loader:
    # batch["waveforms"]: [B, N] float32, zero-padded, 16 kHz; batch["lengths"]: [B] int64.
    # batch["items"][i]: key, text, lang, word_langs, code_mixed, source, speaker, seconds, view (+ codec).
    # batch["state"]: JSON; save it in the checkpoint and pass it to load_state to resume.
    ...
setup.close()  # stops the read-ahead threads, then the shard cache
```

How the stream works:
- **Batches** are drawn from the shuffle buffer by duration: a random pivot plus its nearest neighbours in length, up to `frame_budget` padded frames at 25 fps.
- **Augmentation:** a row is augmented with probability `augment_prob`; its `view` (from the planner's batch counter) seeds the corruption, so it differs on each draw and a resumed run repeats it. See "Live training" for the policy.
- **Read-ahead** (`--readahead`, default 64 rows): each Hub stream reads, parses and length-checks rows in its own thread, so the planner's single-threaded draws don't wait on one source at a time. Without it, the 53-stream full mixture supplied only about 266× real time on the G4, against about 740× needed, with the CPU idle. The rows, their order and the resume state are identical with or without read-ahead (tested), so the option is a runtime argument and doesn't change a run's config hash. The drop counters in the report include rows read ahead but not yet drawn.
- **Decoding** to 16 kHz float32 and the augmentation run in a thread pool. Each utterance's audio hash is checked against the eval and dev hashes, and matches are dropped.
- **Resume:** the state holds each stream's (epoch, shard position, row), the RNG, and references to the rows in the shuffle buffer. `load_state` re-reads exactly those rows, so a VM lost mid-epoch continues with the same keys, without repeats or gaps (tested).
- **Shard cache:** each stream reads a fixed seeded pool of shards (enough for its hours), reshuffled every stream-epoch. Shards are prefetched in background threads into `R/stream-cache/`, which is capped at `cache_gb` (default 40 GiB), and each shard is deleted once read. `StreamLoader.status()` logs MB/s, cache fill, buffer fill and consumer wait time.

**Language-ID labels.**
- Every row carries `lang`, the source's language label.
- Rows also carry `word_langs`, one tag per word: Latin words are `en`, words in the language's own script are `lang`, and digits inherit the previous tag.
- Code-mixed rows come from three conventions:
  - the IndicVoices `[gloss]` (50–71% of rows in hi/kn/ta);
  - the Vaani `{gloss}`;
  - MUCS and SPRING-INX, which write English in Latin.
- The labels' reliability by source is in docs/data-licenses.md. In short:
  - English written in native script and never glossed is tagged with the matrix language.
  - Hindi and Urdu are separated by script only.

**Scoring.** AI4Bharat sets often write English in native script. Rows whose glosses were rewritten therefore also carry `refs` = [the native-script form], and `metrics.script_normalised_wer` scores against both forms (see Metrics).

**Measured on an Apple M5 Pro workstation (18 cores):**
- Decode and pad of 16 kHz FLAC through `StreamLoader`: 5,200× real time with one thread and 18,000× with eight. That is 4 h of audio in 0.8 s.
- One ranged GET from the Hub over a home connection: 7.1 MB/s, which at IndicVoices' 61 MB/h is about 420 h of audio per hour per connection.
- Colab network throughput is not measured. Check the `stream:` log lines against the encoder's ~700 h/h on an L4.

**Disk:**
- **Pilot, streaming** (`plan`): about 12 GB of FLAC for the fixed sets, the 40 GiB cache, and the Kathbath test tars. That is about 64 GB in total, plus the layer-6 cache if one is still used.
- **Pilot, full download:** about 100 GB downloaded and 55 GB of FLAC.
- **Full mixture:** about 34 GB of fixed sets plus the cache. One pass over the pools streams about 0.94 TB.

## The pilot: live training, evaluation and baselines

### Launch (one A100)

```bash
# On the VM, from the repository root (the code synced as in "Running on Colab over SSH"); HF_TOKEN in the environment.
export HF_TOKEN=...                      # account that accepted every gate (IndicVoices, Kathbath, Vaani, Svarah, ...)
bash scripts/pilot.sh                    # queues setup -> prep -> train -> eval -> baselines -> report; returns at once
bash scripts/status.sh                   # progress; logs in $ROOT/logs/pilot-<stage>.log
# rerun the same command after any failure or VM loss: DONE stages are skipped, the rest resume

# On a workstation: pull every run (one syncer per run; each exits on DONE or FAILED)
for v in v2 v3 v4 v4+lid; do bash scripts/sync_runs.sh $S /content/indicear/runs/$v & done
```

`scripts/pilot.sh` queues one nohup job per stage with `launch.sh`. Each job waits (`WAIT_FOR`) for `$ROOT/pilot/<previous>.DONE` and gives up if `<previous>.FAILED` appears. The stages:

| Stage | Command | Output |
| --- | --- | --- |
| setup | `scripts/setup_env.sh` | free-threaded 3.14t venv |
| prep | `ablate prep --mixture pilot --stream` (+ `--allow-review mucs` when `ALLOW_REVIEW=mucs`) | eval sets, dev, FLEURS dev, stream plan, tokenizer |
| train | `ablate train --lower live --variants v2,v3,v4,v4+lid --train-hours 6500` | `runs/<v>/`, `results/<v>.json` |
| eval | `ablate eval --variants v2,v3,v4,v4+lid` | `results/<v>/<eval set>.json` |
| baselines | `scripts/baselines.sh` (`indicear.baselines` in the `baselines` group's Python 3.13 environment) | `results/<system>/<eval set>.json`, `results/baselines.json` |
| report | `ablate report` | `results/summary.md`, `results/eval_summary.{md,json}` |

Markers: `$ROOT/pilot/<stage>.DONE` or `.FAILED` (with the exit code), and `$ROOT/pilot/<stage>.gpu.csv` (nvidia-smi every 5 s; each training run also writes `runs/<v>/gpu.csv`). `bash scripts/pilot.sh --foreground` runs the stages in the current shell instead; `--stage NAME` runs one. `scripts/stop.sh pilot-train` stops training after a checkpoint (exit 143, marked FAILED); rerunning `pilot.sh` resumes it. Settings: `MIXTURE`, `VARIANTS`, `TRAIN_HOURS`, `ALLOW_REVIEW`, `BASELINES`, `NOISE_DIR`, `SYNC`, and `PREP_EXTRA` / `TRAIN_EXTRA` / `EVAL_EXTRA` / `BASELINE_EXTRA`.

Smoke mode runs the same chain on a CPU in about 15 s with the tiny tower, synthetic shards and a few steps (`indicear.smoke`):

```bash
SMOKE=1 ROOT=/tmp/pilot-smoke PY=$(uv run python -c 'import sys; print(sys.executable)') bash scripts/pilot.sh --foreground
```

### Live training (`ablate train --lower live`)

- **Data:** the StreamLoader of the prepared mixture, built through the planner (`build_planner`) and wrapped in `LoaderSource`. Its fingerprint, which joins every run's config hash, is the mixture name, the mixture file's sha256, the seed, the allow-review set, the shuffle-buffer size, the augmentation probability and the negatives share. The non-speech negatives (`neg_train`) are streamed too, at a fixed 2% share (`--negatives-share`).
- **Run length:** `--train-hours` of audio (pilot: 650 h × 10 = 6500; default `--epochs` × the mixture's planned hours). The frame budget is only known after the VRAM autotune, so the hours become steps then: `steps = ceil(hours × 3600 × 25 / (budget × packing))`, with `--packing 0.85` the assumed share of the padded budget that holds real frames. The estimate is logged at start, and stored in the checkpoint so a resume keeps it. Each result reports `audio_hours` actually trained and `packing_measured`; if the two disagree, the run trained proportionally more or fewer hours than asked. `--max-steps` fixes the count instead.
- **Variants:** `--variants v2,v3,v4,v4+lid`, any of v1–v6, `<v>+lid` (LID heads on top, detached) and `lid` (LID heads alone on the frozen encoder). LID variants use `LidConfig(class_weights=lid.balance_weights(hours per language))` with the hours of the mixture's streams; `extra["lid"]` enters the config hash only for LID variants, so existing hashes are unchanged.
- **Text:** `text_encoder(text) -> (tokenizer ids, phonetic unit ids)`; a transcript that does not encode is dropped and counted.
- **Several variants on one GPU:** by default all variants train in **one process from one stream** (`trainer.LiveGroup`). Each batch is decoded, augmented, featurized and run through layers 1–6 once; each model then runs its own layers 7–12, heads, backward and optimizer step. A model's adapters are attached to the shared tower only while it runs. Members share the frame budget (autotuned on all of them together), the step count and the stream position. Each keeps its own run directory, checkpoints, `train.csv`, `gpu.csv`, `command.sh` (which reruns the whole group) and config hash; the checkpoint format is that of a solo live run, so a member can also be resumed alone. If a VM is lost between the members' saves, the group resumes every member at the newest step all of them have. Members draw SpecAugment masks one after another from the global RNG, so a member's run is not bit-identical to a solo run of that variant. `--no-share-stream` trains the variants one after another, each with its own stream. This was chosen over N processes with `--vram-share 1/N` (the cached path's `PARALLEL`): separate processes would decode and download the audio N times (about 690 GB per process for the pilot), hold N shard caches (40 GiB each) and run layers 1–6 N times.
- **Dev evaluation:** every `--eval-every` steps on `--eval-items-per-lang` dev rows, whose layer-6 states are computed once and kept in memory; at the end, the full dev set, `cs_dev` and `neg_dev` are decoded by every member from one lowering per chunk.

### Augmentation in live mode

`--clean-prob 0.5`: the planner marks half the rows as augmented draws (view > 0). With `--augment-device gpu` (the default):

- **Stream workers (CPU)** only decode, plus one ffmpeg codec (gsm, opus or amr, those ffmpeg can encode) on `--codec-prob 0.15` of all speech clips, with `-threads 1` (`augment.CodecAugment`).
- **Main process (GPU)**, on the padded batch before the log-mel (`gpu_augment.GpuAugment`), one kind per augmented row: noise 25% (5–20 dB), babble 20% (3–6 other utterances of the same batch, 5–20 dB), reverb 17.5% (RT60 0.3–0.9 s), reverb with point noise 17.5%, speed 15% (0.9 or 1.1, windowed-sinc resampling), tel8k band-limit 5%. `--augment "noise=...,..."` changes the weights. RIRs and coloured noise (or MUSAN files with `--noise-dir`) are pre-generated once into a device bank.
- **Determinism:** every choice comes from numpy `default_rng((seed, crc32(key), view))`, with the view from the planner state; the codec draw uses `(seed, key, view, 1)`. The banks are fixed and no torch RNG is used, and augmentation runs on the whole batch before any OOM split, so a resumed run reproduces it (tested: a killed and resumed run has the same losses as an uninterrupted one).
- **Ordering:** the codec runs first, in the worker, so GPU noise and reverb come after it. Speed changes lengths: rows that no longer fit their transcript under CTC are dropped and counted.
- `--augment-device cpu` runs the whole policy in the workers (`augment.AugmentPolicy`, codecs included).
- `ablate cache --aug-views N` uses the same GPU path by default when the log-mel runs on the GPU (`--augment-device`, `--codec-prob`); `CACHE_WORKERS` sets the DataLoader workers in `scripts/cache.sh`, and every DataLoader worker runs with OMP/OpenBLAS/MKL at one thread.

**Real room impulse responses** (`--rir-dir`, `--real-rir-share`; off by default and in the full run until their licences are reviewed). `scripts/rirs.sh` downloads OpenSLR SLR28 (`rirs_noises.zip`, 1.31 GB) once into `$ROOT/rirs` and extracts only the measured RIRs that `real_rirs_isotropic_noises/rir_list` names, leaving out the isotropic noise recordings.
- **Licences:** the SLR28 page states Apache 2.0, but that covers the packaging. Its README says the 325 real RIRs come from RWCP (182; its own OpenSLR page, SLR13, says "Research and development use only"), AIR (107; licence "Not stated" on its OpenSLR mirror, SLR20) and REVERB 2014 (36; terms not read). `RIR_SETS` defaults to `air,RVB2014` and leaves RWCP out; `RIR_SETS=air,RVB2014,RWCP` takes all 325. Treat AIR and REVERB 2014 as `review` until their terms are confirmed (docs/data-licenses.md).
- **Loading** (`gpu_augment.load_rirs`): each RIR's first channel, resampled to 16 kHz, with its direct-path peak moved to 1 ms from the start, cut to 1.2 s (the longest synthetic RIR) and scaled to unit energy. Each is labelled with its Schroeder RT60 and appended to the bank after the 128 synthetic RIRs.
- **Use:** a reverb or reverb + point-noise row takes a measured RIR with probability `--real-rir-share` (default 0.5), else a synthetic one; the point-noise RIR stays synthetic. The choice is one more numpy draw from the row's seed, so a resumed run repeats it. The number of measured RIRs, the share and a sha256 of the files enter the run's config hash. Without `--rir-dir`, the bank, every draw and the config hash are exactly the synthetic-only pilot's.

**Why:** on the 12-vCPU A100 VM, caching FLEURS with `AUG_VIEWS=2` and the CPU Corruptor ran at about 460 h of audio per hour with the GPU mostly idle (measured on that VM). Measured on an Apple M5 Pro ( `python -m indicear.gpu_augment cpu`), in seconds of audio per CPU-second on one core:

| CPU Corruptor | noise | reverb | reverb + point noise | tel8k | speed | opus codec (ffmpeg, wall) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seconds of audio per core-second | 5,440 | 2,700 | 1,140 | 6,600 | 19,900 | 84 |

The codecs dominate: about 84× real time per core, mostly spawning two ffmpeg processes per clip. With codecs on 15% of clips, the workers' codec cost is about 1 core-second per 560 s of audio, so 12 vCPUs keep up with about 24,000 h/h. That is well above the encoder's rate.

`GpuAugment` measured on an Apple M5 Pro CPU through torch (18 cores), in seconds of audio per second: noise 26,000, babble 8,100, reverb 5,800, reverb + point noise 2,700, speed 3,900, tel8k 21,000. **On an A100, unmeasured:** these are FFT, gather and elementwise kernels. An estimate of ≥ 50,000 s of audio per GPU-second for every kind (at least 10× the CPU figures) puts the pilot's augmentation (about 3,250 h of augmented audio) at a few GPU-minutes in total.

### Evaluation (`ablate eval`) and baselines

**Eval sets.** These are the `manifests/eval_*.jsonl` files written by `prep --stream`; all are downloaded once, and their speakers are excluded from the streams:

- **IndicVoices valid:** hi, ta, kn.
- **Kathbath test:** known and unknown speakers, clean and noisy, for hi, ta, kn (12 sets).
- **Svarah** (Indian English).
- **IndicContextEval:** hi only; the dataset has no kn, ta or en. Scored as the full set and as `indiccontext_hi_code_mixed` (rows with the `code_mixing` flag).
- **MUCS:** test and blind (Hi-En); eval only.
- **Vaani:** test (hi, ta, kn, en).
- **Others:** Lahaja, Vaani Benchmark, LibriSpeech test-clean and test-other, VoxPopuli test and accented.
- **FLEURS dev:** `eval_fleurs_dev`, for continuity, in the mixture's languages; set with `--fleurs-dev-langs`.
- **`cs_dev`, `neg_dev`.**
- **Robustness subset:** 100 utterances per language from the Kathbath clean test-unknown sets plus Svarah, corrupted at load time with the robustness probe's conditions. Defaults: noise 10 and 5 dB, babble 10 dB, reverb 0.6 and 0.9 s, reverb 0.6 s with point noise at 10 dB. Seeded per utterance, so every system hears the same audio.

**How it runs.** Eval audio is lowered through layers 1–6 once per one-hour chunk, then every variant decodes from those states. Finished (variant, set) results are skipped on rerun.

**Baselines** (`indicear.baselines`; revisions pinned; see the module docstring for the card facts):

| System | Repo @ revision | Languages | Status |
| --- | --- | --- | --- |
| `indicconformer-ctc`, `indicconformer-rnnt` | ai4bharat/indic-conformer-600m-multilingual @ 2a77d305 (gated, MIT) | 22 Indic incl. hi, ta, kn; **no English** | Runnable via the repo's `model_onnx.py` (onnxruntime), built on the pinned snapshot because its `from_pretrained` ignores `revision`. Batch 1 |
| `ep-hindi-asr-mixed`, `-native` | epoch-discovery/ep-hindi-asr @ d7822345 (gated, Apache-2.0) | hi | Adapter follows the card's Python API (`load`, `log_probs`, `greedy`; mixed head = English in Latin). The code is not readable without the gate, so it is **unverified** |
| `sravaani` | ARTPARK-IISc/SraVaani-1.0 @ f5dd5358 (gated, MIT) | 20 Indic + en, **no Urdu**; picks the language itself | Runnable via the repo's `modeling_sravaani.py` (TorchScript FastConformer, greedy TDT); about 6× faster on Apple MPS than on the CPU |
| `dulu-malayalam` | serendipitylabs/dulu-malayalam-asr @ 0d90991e (Apache-2.0) | ml | Runnable; runs only when an eval set has ml rows (`--fleurs-dev-langs` with ml) |
| `kothagemma` | nehsan/KothaGemma @ c0e92637 (CC BY-SA 4.0 weights) | bn | Runnable (greedy, no KenLM); runs only with bn rows, so it is not run in the pilot |
| `dulu-malayalam-ctc-778h` | whiletruelearn/dulu-malayalam-ctc-778h @ d6a6b905 | ml | **Skipped:** the card has no inference path and the repo no licence tag |

The baselines execute each repo's own Python code from the pinned snapshot. They need onnxruntime and scipy, and standard CPython ≤ 3.13 (there are no free-threaded onnxruntime wheels), so `scripts/baselines.sh` builds a separate environment from this project's `baselines` dependency group, restricted to CPython 3.13 (`[tool.uv.dependency-groups]`), with `uv sync --locked --python 3.13 --group baselines`; one `uv.lock` pins both environments.

### Pilot compute estimate (live mode, one A100; unverified)

The basis is L4 measurements, with the A100 assumed to be about 2.5× an L4 (not measured):

- **v1 cached head path:** about 33k frames/s, which is 6 frozen layers forward plus a head, about 1,320 h of audio per hour. The A100 is assumed to do about 3,300 h/h.
- **Full-encoder forward:** about 700 h of audio per hour, consistent with the above (12 layers at about half the rate).
- **Per hour of audio on an A100:** the log-mel plus layers 1–6, shared by the group, cost about 1/3,300 GPU-h. Each adapter variant's layers 7–12 forward and backward cost about 2–3× a 6-layer forward (activation gradients only; the base weights are frozen), about 1/1,650–1/1,100 GPU-h.
- **Training v2, v3, v4 and v4+lid on 6,500 h in one group:** (1 + 4 × 2–3) / 3,300 × 6,500 ≈ **18–26 A100-hours**. Sequentially, without the shared stream, it would be about 23–31 A100-hours. LID and phonetic heads are small.
- **Eval:** about 150 h of eval audio (139 h of eval sets by `mixture plan`, plus dev and the robustness subset), lowered once with 4 variants decoding: about 0.25 A100-h. Baselines are batch-1 ONNX/PyTorch inference, roughly 0.5–1.5 A100-h (not measured).
- **Colab units:** at about 7 units per A100-hour (PLAN.md's phase-1 ratio), the pilot needs about 130–190 units against a balance of about 196. `TRAIN_HOURS=3250` (5 epochs) halves the training cost.
- **Data side:** about 69 GB of shards per pass over the pools (`mixture plan`), so about 690 GB over 10 epochs, or about 8–9 MB/s sustained over a 22 h run. Colab's network rate is not measured.
- **CPU side:** decoding runs at 5,200× real time per thread, measured. Codecs on 15% of clips cost about 12 core-hours over the run.

## The full warm-started run (ran 24–25 October 2026; results in [full-run.md](full-run.md))

The continued run starts from the finished pilot (`v4+lid`, step 7,423, config `34ad89478fc81611`) and extends it to all 12 Indian languages plus English. It is a **warm start, not a resume**: a new run with its own config hash, step 0, and fresh AdamW and scheduler state. The plan, the expected hours and cost, and the known limitations are in [docs/PLAN.md](PLAN.md#full-warm-started-run-as-planned-on-24-october-2026).

### Launch (one G4 or A100)

```bash
# On a workstation: the code, then the pilot's final weights and tokenizer (sha256-checked on the VM)
bash scripts/colab_ssh.sh $S 'mkdir -p /content/indicear'
rsync -a -e "ssh -F ~/.config/indicear/ssh-$S.conf" --exclude .venv ./ indicear-$S:/content/indicear/
bash scripts/push_init.sh $S ~/indicear-runs/pilot/v4+lid /content/full/init/v4+lid

# On the VM (HF_TOKEN exported): queue setup -> prep -> train -> eval -> report; returns at once
ROOT=/content/full bash /content/indicear/scripts/full_warm.sh

# On a workstation: pull the run and its tokenizer to a directory of its own
bash scripts/sync_runs.sh $S /content/full/runs/v4+lid ~/indicear-runs/full_warm/v4+lid
```

`scripts/full_warm.sh` works like `pilot.sh`: one nohup job per stage through `launch.sh`, each waiting (`WAIT_FOR`) for the previous `$ROOT/full_warm/<stage>.DONE` and giving up on its `.FAILED`; rerunning it skips finished stages and resumes the rest. `SMOKE=1` runs the whole chain on a CPU in about 20 s (tiny tower, synthetic shards; setup trains a 3-step smoke "pilot" to start from). Its header lists every setting. The defaults:

| Setting | Default | Why |
| --- | --- | --- |
| `TRAIN_HOURS` | 8,000; 7,000 when `nvidia-smi` names an A100 | 10–12 GPU-hours at 620 h of audio per A100-hour, or the G4's estimated 1.15–1.55× |
| `PACKING` | 0.75 | real frames per frame budget, used only for the hours → steps estimate. The pilot's 1,800 s shuffle buffer packed only 0.48 (measured), so the pilot processed about 2,660 h, not the 4,700 h it asked for. A simulation of the planner on the pilot's row durations reproduces 0.47 and gives 0.74–0.78 with a 3 h buffer |
| `BUFFER_SECONDS` | 10,800 | the buffer must hold several batches' worth of audio (a 75–80k-frame budget is 3,000–3,200 s) |
| `PEAK_LR`, `WARMUP`, `FINAL_LR` | 6e-4, 300 steps, 5e-5 | warm-started weights need a gentler peak than the pilot's 1.8e-3 (1e-3 scaled by the budget) |
| `NEW_LR_MULT` | 3 | new-script output rows and class rows (and v6's new embedding rows) |
| `INTER_WEIGHT` | 0.15 (pilot 0.3) | the phonetic CTC loss plateaued at about 1.7 from step 1,500 |
| `LID_FRAME_WEIGHT` | 0.2 (pilot 0.1) | frame-level LID span accuracy was 25–45% |
| `AUGMENT` | reverb 0.30, reverb + point noise 0.175, noise 0.2, babble 0.15, speed 0.125, tel8k 0.05 | heavy reverberation cost +21 CER after the pilot |
| `RIR_SETS` | empty (synthetic RIRs only) | the measured RIRs need a licence decision first (see "Augmentation in live mode") |
| `PREP_PACK`, `PACK_OUT`, `PACK_AUDIO` | empty | restore a packed prep root, or pack this one after prep |

The training command it runs (G4 defaults):

```bash
python -m indicear.ablate --root /content/full train --lower live --variants v4+lid \
  --init-from /content/full/init/v4+lid --train-hours 8000 --packing 0.75 --buffer-seconds 10800 \
  --peak-lr 6e-4 --final-lr 5e-5 --warmup-steps 300 --new-lr-mult 3 --inter-weight 0.15 \
  --lid-frame-weight 0.2 --augment reverb=0.3,reverb_pointnoise=0.175,noise=0.2,babble=0.15,speed=0.125,tel8k=0.05
```

### Warm start (`--init-from`, `indicear.warmstart`)

- **Source:** a finished run directory (`heads.safetensors`, `adapters.safetensors`, `variant.json`, `lid.json`, `tok/`) or one checkpoint file. The variant must have the same structure (adapters, phonetic heads, two-level output, transducer, adapter rank, intermediate layers); loss weights may differ.
- **Copied:** the adapters on layers 7–12, the self-conditioned phonetic CTC heads (82 shared units), the final LayerNorm and the utterance and frame LID heads (all 12 languages already), and for v6 the prediction network and joint.
- **Tokenizer:** `prep --tok-from <pilot>/tok --tok-keep deva,latn,taml,knda` keeps those four SentencePiece models byte for byte and trains new ones for beng, guru, gujr, orya, telu, mlym and arab on the full mixture's text (`AggregateTokenizer.extend`). The script order is unchanged, so `script_range` moves the kept scripts to new offsets but keeps their sizes and pieces; `aggregate.json` records the source tokenizer's sha256 and each kept model's sha256 (`extended_from`).
- **Token head:** rows 0–3 (blank, pad, bos, eos) and each kept script's rows are copied from their old range to the new one. New scripts' rows keep the new model's random initialization.
- **Script classes (two-level output):** blank and the kept scripts are copied. A new script's class row starts fresh: zero weights and bias `--new-script-bias` (default -2, near where the pilot's own unseen scripts settled), so it starts unlikely but trainable. A much lower bias such as -40 never recovers: Adam moves a bias by about the learning rate per step. `--new-script-logits copy` copies the pilot's rows instead.
- **Exactness, measured on the real pilot** (48 pilot dev utterances, 12 each of hi, kn, ta and en, CPU, float32): with `copy`, the log-probabilities of blank and of every deva, latn, taml and knda token, and both LID outputs, are bit-identical before and after expansion. With `fresh`, the largest change is 1.9e-6 nats, because the pilot gave the seven unseen scripts at most 4.1e-7 probability per frame and that mass is renormalized away; greedy transcripts are identical (48 of 48). `tests/test_warmstart.py` checks both cases, including bit-identity with `fresh` for a source that gives the new scripts no probability.
- **Record and resume:** `runs/<v>/init_from.json` and every checkpoint (`warm_start`) record the source's config hash, the weights' and tokenizer's sha256, the kept and new scripts and the row map; the summary is hashed into the run's config (`extra.init_from`). A VM that has lost the source resumes from the checkpoint alone; `resume_on.sh` uploads the synced `tok/` to `<ROOT>/tok` first, so prep never retrains a tokenizer that no longer matches the checkpoints.
- **LR:** `--peak-lr` sets the peak itself (no budget scaling), `--final-lr` the cosine floor, `--warmup-steps` the linear warmup (at most 10% of the steps). `--new-lr-mult` scales the fresh rows' updates: a fresh row lives inside a shared tensor, so `trainer.RowLr` multiplies those elements' AdamW update (Adam direction plus decoupled weight decay) after each step, which equals a step at that LR; whole fresh parameters get their own optimizer groups. Every new option defaults to the old behaviour and enters the config hash only when set: the pilot's checkpoint still hashes to `34ad89478fc81611` with this code (checked).

### The `full_warm` mixture

`indicear/mixtures/full_warm.json`: shares are pinned per sampling group (max = min share), the new languages from IndicVoices, Kathbath and Vaani-transcription-part, a refresher for hi, ta, kn and en, and code-mixed-only streams (IndicVoices for all 11 Indic languages and Vaani for hi, kn, te, bn and mr; `disjoint_code_mixed_pools` keeps their shard pools apart from the same language's main stream). Kathbath streams have `"weight"` 2 (1.5 for hi, ta and kn): within its language, a stream is drawn in proportion to hours × weight. MUCS stays eval-only; Shrutilipi is left out (no access to its rows, no teacher-CER filter).

**Loanword normaliser** (`"loanwords": {"sources": ["kathbath", "kathbath_test"]}` in the mixture; [docs/loanwords.md](loanwords.md) has the method, thresholds and the precision judged by an automated reviewer, 97–100% per language). Kathbath writes English loanwords in native script, while IndicVoices and Vaani after the gloss rewrite write them in Latin. `markup.apply_loanwords` runs inside `ingest.clean_row` (download, streaming and tar paths) and the text pass, for the named sources only:
- training and dev rows take the rewritten text (re-normalized, so `code_mixed` and `word_langs` follow) and keep the original in `refs`;
- eval rows (Kathbath test) keep the published text and carry the rewrite in `refs`, so strict WER stays comparable and the script-normalised WER accepts either form;
- the normaliser's `fingerprint()` (its resource files and parameters) joins the stream fingerprint and so the run's config hash; mixtures without `loanwords` (the pilot) are unchanged.

The planner draws **rows**, so a share of rows is not a share of hours: `python -m indicear.mixture hours --mixture full_warm --train-hours 8000` estimates the processed hours from the row shares and mean row durations (the pilot's dev means; the new languages are assumed to match). The eval sets are Kathbath test known/unknown clean/noisy for 11 languages, IndicVoices valid (11), Vaani test (11 + en), FLEURS dev (12, from `--fleurs-dev-langs mixture`), IndicContextEval (8), MUCS test hi and bn, Svarah, Lahaja, Vaani Benchmark, LibriSpeech and VoxPopuli, plus the robustness subset.

### Prep packs (`indicear.prep_pack`)

```bash
python -m indicear.prep_pack pack --root /content/full --out /content/full-prep.tar --with-audio   # or .tar.gz without audio
python -m indicear.prep_pack restore --root /content/full --archive /content/full-prep.tar
PREP_PACK=/content/full-prep.tar ROOT=/content/full bash scripts/full_warm.sh                    # restores, then prep skips
```

- A pack always holds `tok/`, the text pass (`train_texts.jsonl`, `stream_plan.json`), `mixture.json`, the shard listings and `prep_report.json`; restoring it skips the text pass and the tokenizer, and prep rebuilds only the fixed sets (seeded: the same dev carve-out, tested).
- `--with-audio` adds every other manifest and every audio file they reference under ROOT (eval, dev, cs_dev, negatives, FLEURS dev); after that restore, prep downloads and decodes nothing.
- Manifest audio paths are stored relative to ROOT and rewritten on restore; `mixture.json` is re-pointed at this checkout's mixture file when the sha256 matches. Every file's sha256 is checked before anything moves into place, and a ROOT that already has a tokenizer is refused.

**Cores:** stream decode workers already scale with the host (CPU count - 1); `--fetch-workers` and `prep --download-workers` now default to CPU count / 4 within [4, 16], so 4 on a 12-vCPU A100 VM (as before) and 12 on a 48-vCPU G4 VM.

## Running on Colab over SSH

Colab allows one SSH connection per runtime (a second gets HTTP 429). Every Mac-side script therefore goes through one multiplexed control socket that `scripts/lib/transport.sh` generates in `~/.config/indicear/ssh-<session>.conf`: `ProxyCommand colab ssh --proxy-mode -s <session>`, `ControlMaster auto`, `ControlPersist 10m`.

```bash
# On a workstation (S = Colab session name)
bash scripts/colab_ssh.sh $S 'mkdir -p /content/indicear'
rsync -a -e "ssh -F ~/.config/indicear/ssh-$S.conf" --exclude .venv ./ indicear-$S:/content/indicear/
bash scripts/colab_ssh.sh $S 'bash /content/indicear/scripts/setup_env.sh'

# Queue the whole pipeline. WAIT_FOR chains steps; each job survives disconnects under nohup.
# (WAIT_FOR paths are space-separated, so they must not contain spaces.)
T=/content/indicear/scripts; R=/content/indicear
bash scripts/colab_ssh.sh $S "bash $T/launch.sh prep $T/prep.sh"
bash scripts/colab_ssh.sh $S "WAIT_FOR=$R/prep_report.json bash $T/launch.sh cache $T/cache.sh"
bash scripts/colab_ssh.sh $S "WAIT_FOR=$R/cache/layer6/cache.done VARIANTS=v1 bash $T/launch.sh ablation-v1 $T/ablation.sh"
bash scripts/colab_ssh.sh $S "WAIT_FOR=$R/results/v1.json PARALLEL=3 bash $T/launch.sh robustness $T/robustness.sh"
bash scripts/colab_ssh.sh $S "WAIT_FOR=$R/results/robustness/v1/summary.json VARIANTS='v2 v3 v4' PARALLEL=3 bash $T/launch.sh ablation $T/ablation.sh"

bash scripts/colab_ssh.sh $S "bash $T/status.sh"        # progress at any time

# Pull every run to a workstation (one syncer per run; each exits once its run writes DONE or FAILED)
for v in v1 v2 v3 v4; do bash scripts/sync_runs.sh $S /content/indicear/runs/$v & done
```

### Checkpoints and resume: launch → sync → VM lost → resume_on

1. **On the VM.** Each run directory `runs/<run>/` holds:
   - `ckpt-<step>.pt`: the last 2 checkpoints, plus `best.pt`. Each checkpoint stores the adapters, heads, optimizer, scheduler, RNG states, sampler position, frame budget, OOM split count, config, config hash and git revision. It never stores the encoder or the cache.
   - `LATEST.json`: an atomic manifest listing each complete file with its step, sha256, size and kind, plus whether the checkpoint is the best.
   - `train.csv`, `train.log`, `gpu.csv` (5 s samples), `command.sh` (the exact command), `result.json`, and DONE or FAILED.

   Saves happen every 500 steps, every 10 minutes, after each evaluation, and on SIGTERM/SIGINT. Each one writes a temp file, fsyncs it, then `os.replace`s it.
2. **On a workstation.** `scripts/sync_runs.sh <session> <remote run dir>`:
   - polls every 90 s;
   - pulls only the files `LATEST.json` lists, each to a temporary name;
   - checks each file's sha256 before moving it into place, and rejects a mismatch;
   - keeps the last 2 checkpoints plus the best in `~/indicear-runs/<run>/`, outside any repo;
   - pulls logs and CSVs best effort;
   - stops after a final pull once the run writes DONE or FAILED.

   If SSH fails it uses `colab download`, which needs no SSH slot. `--drive-copy DIR` adds an optional second copy.
3. **VM lost.** Start a new runtime, then run `bash scripts/resume_on.sh <new session> <run>`. It:
   - verifies the local copy;
   - uploads the code, the checkpoints, `LATEST.json`, `command.sh` and `train.csv` to the original paths;
   - runs `command.sh` under nohup.

   `command.sh` rebuilds the environment, the FLEURS data and the layer-6 cache, each step skipping finished parts, then reruns the training command. That command resumes from the uploaded checkpoint with no extra flag. Then restart `sync_runs.sh` for that run.
4. **Rules:**
   - A run refuses to resume when the config hash differs; `--force-new-run` starts over and moves the old checkpoints aside.
   - A truncated checkpoint falls back to the previous one.
   - `ablate.py` skips variants that already have `results/<v>.json`, and `robustness_probe.py` skips conditions that already have a JSON.

### Training on Apple MPS (resuming on an Apple-silicon workstation)

`ablate train` and `ablate eval` pick a device with `--device` or `$INDICEAR_DEVICE` (`auto`: CUDA, then MPS, then the CPU; `encoder.select_device`). The test suite pins `INDICEAR_DEVICE=cpu` in `tests/conftest.py`.

On MPS the defaults are float32, eager execution and fused AdamW. Two options change this, and neither enters the config hash:

- **`--precision fp16`** runs layers 1–12 under float16 autocast over float32 weights, with a dynamic `GradScaler`. The scaler's state is saved as the checkpoint's `grad_scaler`, and a step with non-finite gradients is skipped and logged. The optimizer, the log-softmax and the CTC losses stay in float32.
- **`--precision bf16`** runs bfloat16 autocast with no scaler.
- **Half-precision constants:** both options need `encoder.half_safe_tower`. With float32 weights, the tower's activation clamp (`gradient_clipping = 1e10`, capped with the *weights'* finfo) and its attention fill (`-1e9`) overflow float16 under autocast. They are capped at the autocast dtype's range, which leaves float32 results unchanged.
- **torch.compile (default; `--no-compile` turns it off):**
  - Each tower layer and CTC head is compiled (`encoder.compile_tower_layers`), with the batch dimension marked dynamic and automatic dynamic shapes off.
  - Live micro-batches are padded to multiples of 48 encoder frames (`stream_train.pad_features`). Padding is exact for the valid frames, because the tower has no right context and a causal convolution. Each bucket compiles once, in about 13 s.
  - Evaluation and decoding run under `torch.compiler.set_stance("force_eager")`.
  - Static-T graphs were about 2× faster than dynamic-T ones.

Other MPS differences:

- **CPU work in the main process:** MPS has no float64, so the log-mel (`encoder.feature_device`) and `GpuAugment` (`gpu_augment.augment_device`) run on the CPU in the main process, with the same torch code and numpy plans; the features then move to MPS. The first-batch check against the Hugging Face extractor still runs. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before torch is imported.
- **Micro-batches (`--splits N`, exact):** each step runs as N gradient-accumulation micro-batches, and every loss term of a micro-batch is divided by the **whole** batch's denominator (`DesignA.loss_norms`: target tokens, phonetic units, valid frames, LID class-weight sum and LID rows). The accumulated gradient therefore equals the one-piece gradient, up to float rounding and the per-micro-batch SpecAugment draw (tested). The old row-share weighting was not exact, because CTC is token-normalized. `OomBackoff` doubles the count on a CUDA or MPS out-of-memory error and uses the same rule.
  - Without `--splits`, a checkpoint written on the same device type and precision keeps its count. Otherwise `runtime.gpu.auto_splits` sizes it from `torch.mps.recommended_max_memory()`.
  - The pilot's budget of 67,045 frames on the 64 GB M5 Pro gets 4 micro-batches in half precision and 6 in float32.
- **Hash and resume:** device, precision, splits, compilation and stream workers are not part of the config hash, and `command.sh` omits the host-specific ones. A CUDA run therefore resumes on MPS and back.
  - Checkpoints record `device` and `precision`, and `rng_state` adds the MPS generator.
  - A CUDA generator state cannot be loaded on MPS, so it is skipped with a log line. The MPS generator is then seeded from the checkpoint's CPU generator state (`checkpoint.derived_seed`), so resuming the same checkpoint twice reproduces itself.
- **Monitoring:** `gpu.csv` samples the Apple GPU through `ioreg` (utilization and GPU memory in use). `train.csv` `peak_gib` is the largest sampled `torch.mps.current_allocated_memory()`.
- **Stream workers:** 8 by default on MPS (the CPU count minus 1 on a GPU VM). Codecs stay in the workers with `-threads 1`. `$INDICEAR_TORCH_THREADS` caps torch's CPU threads.
- **Keeping the workstation usable:**
  - `$INDICEAR_MPS_MEM_GB` sets the MPS cap that `auto_splits` sizes micro-batches for, and the launcher sets the matching `PYTORCH_MPS_HIGH_WATERMARK_RATIO`. Past the cap, MPS raises an out-of-memory error and `OomBackoff` doubles the micro-batches.
  - `runtime.gpu.HostMemoryGuard` logs free memory, swap and memory pressure every 5 min. It adds one micro-batch split (at most every 10 min) when swap exceeds `$INDICEAR_SWAP_LIMIT_GB` (default 8) or pressure turns critical.
  - `$INDICEAR_FAST_EXIT=1` makes a stopped run exit right after its checkpoint, without waiting for in-flight shard downloads.
- **Network:** at about 550 s of audio per second, live training needs roughly 20 MB/s from the Hub. A 5–40 MB/s home link stalled training, and an `hf_xet` download hung, so the launcher sets `HF_HUB_DISABLE_XET=1`.
  - `--keep-shards` keeps every downloaded shard in `ROOT/stream-cache` across stream epochs and restarts, instead of deleting it after use. `stream.hub_fetch` then returns a complete local copy without any request.
  - `--prefetch-pool` downloads the whole pool first (`stream.download_pool`; 152 shards, 68 GiB for the pilot) and exits.
- **Rebuilt prep root:** `--lenient-stream-restore` leaves out, and logs, buffered rows that this host's exclusions now reject, instead of stopping the run. A prep root rebuilt on another host can carve a different dev set (see `docs/pilot.md`). The default stays strict.

`scripts/resume_local.sh` (an example for one Apple-silicon workstation; adjust its defaults) resumes the pilot's `v4+lid` run on MPS in fp16 with compilation, writing straight into `~/indicear-runs/pilot/v4+lid/`. Its defaults:

- an MPS cap of 42 GiB, which gives 3 micro-batches;
- 14 CPU threads (10 for torch, 4 stream workers), leaving 4 of the 18 cores free;
- `nice -n 10`;
- a swap limit of 8 GiB;
- `--keep-shards`, after a pool prefetch;
- `--lenient-stream-restore`.

`--help` lists the variables. Before it starts the trainer, it:

- symlinks `$ROOT/runs` to that directory;
- verifies the latest checkpoint against `LATEST.json`, and the tokenizer against the run's `tok/`;
- checks the ffmpeg codecs.

It then starts the trainer with nohup and `caffeinate -i -w <pid>`, and samples the GPU, the trainer's RSS and swap to `host.csv`. `--stop` sends SIGTERM to the trainer only, never to its ffmpeg children; the trainer checkpoints and exits 143. Measured speeds and precision parity are in `docs/pilot.md`.

**Checkpoint sizes** (fp32 weights plus AdamW state, 12 bytes per trainable parameter; vocabulary 8,964, 82 phonetic units):

| Variant | Parameters | Per checkpoint | VM keeps 3 files |
| --- | ---: | ---: | ---: |
| v1 | 9.2M | ~105 MiB | ~316 MiB |
| v2 | 10.8M | ~123 MiB | ~370 MiB |
| v3–v5 | 11.1M | ~127 MiB | ~382 MiB |
| v6 | 27.0M | ~309 MiB | ~926 MiB |

## Using a trained model

A finished run becomes one self-contained **model package** (`indicear.export`), which `indicear.transcribe` runs on CUDA, Apple MPS or the CPU. From the repository root, with `S` the Colab session name and `/content/indicear` the training ROOT (`scripts/env.sh`):

```bash
# 1. Pull the run. Exits after DONE or FAILED; --once pulls once. Lands in ~/indicear-runs/v4+lid/
bash scripts/sync_runs.sh $S /content/indicear/runs/v4+lid

# 2. Fetch the tokenizer, once per training ROOT. sync_runs.sh does not pull it: it lives in <ROOT>/tok/,
#    not in the run directory, and every run trained in that ROOT uses it.
bash scripts/fetch_tok.sh $S /content/indicear ~/indicear-tok/pilot

# 3. Export the package, then check it (re-run --verify any time; transcribe also verifies on load)
uv run --locked python -m indicear.export --run ~/indicear-runs/v4+lid \
    --tok ~/indicear-tok/pilot --out ~/indicear-models/v4+lid --with-encoder
uv run --locked python -m indicear.export --verify ~/indicear-models/v4+lid

# 4. Transcribe (any format soundfile or ffmpeg reads; the transcript on stdout, timing and RTF on stderr)
uv run --locked python -m indicear.transcribe --model ~/indicear-models/v4+lid talk.wav call.m4a
uv run --locked python -m indicear.transcribe --model ~/indicear-models/v4+lid --json talk.wav > talk.json
```

**Export** (`python -m indicear.export --run RUN --tok TOK --out PKG [--checkpoint best|latest|PATH] [--force] [--with-encoder] [--name NAME] [--card MD --card-dir DIR...]`):
- The package holds `heads.safetensors`, `adapters.safetensors` (v2–v6), `variant.json`, `lid.json` (+lid), `tokenizer/` (`aggregate.json` and one SentencePiece model per script), `training/` (`result.json`, `config.json` with the trainer config, config hash and git revision, and `command.sh` with a header marking it as the run's historical launch command), `LICENSE` and `NOTICE`, a `README.md` model card and `model.json`.
- `model.json` lists the sha256 and size of every other file, the pinned encoder (`google/embeddinggemma-2@914f7f89…`, audio tower only, Apache-2.0; with `--with-encoder` the audio tower is bundled as `encoder/` (every audio_tower tensor, values unchanged, keys without the `audio_tower.` prefix), otherwise it is downloaded on first use), the variant and decoder, the tokenizer, the LID languages, the training provenance (config hash, git revision, mixture, steps, hours) and the dev-set eval summary from `result.json`.
- The model card covers what the model is, its languages, training mixture and sources with licences, the metrics, and the known limits (reverberation, Hindi/Urdu script ambiguity, approximate word timings).
- No checkpoint or optimizer state goes into the package; `--verify` fails on any `.pt`, on an unlisted file and on a hash or size mismatch.
- Before writing, export loads the heads and adapters into a model built from `variant.json` and the tokenizer on a weightless stand-in encoder. A tokenizer from another ROOT ("the weights do not match the tokenizer") or a partly synced file (sha256 differs from the run's `LATEST.json`) stops the export.
- `--checkpoint best` (or `latest`, or a path) takes the weights from a checkpoint's model state instead of the published heads: for a run that has not finished, or to ship the best dev checkpoint rather than the final weights. The variant and LID config come from the checkpoint's config.
- Provenance: new runs write `git` and `train_config` into `result.json`; for older runs, export reads the git revision and config from the newest checkpoint.

**Transcribe** (`python -m indicear.transcribe --model PKG FILE... [--json] [--device auto|cpu|mps|cuda] [--dtype auto|fp32|bf16|fp16] [--beam N] [--decoder ctc|rnnt] [--window 30] [--overlap 2] [--script-rules on|off] [--lid-script-set on|off] [--lid-topk 2] [--lid-threshold 0.15] [--lid-frame-share 0]`):
- **Encoder:** loaded once through `encoder.load_tower`: from the package's `encoder/` when it has one, else from the pinned EmbeddingGemma 2 revision (downloaded on first use), keeping only the audio tower. A package trained on another encoder revision is refused.
- **Audio:** soundfile (resampled with soxr, down-mixed to mono), else `ffmpeg` to 16 kHz mono float32.
- **Long audio:** windows of at most 30 s (the feature extractor's limit) overlapping by 2 s, starting on the 40 ms frame grid. The encoder has no look-ahead, so a window's end is as good as a full-length pass and only its start lacks left context. Each window therefore keeps the frames after the previous window's last frame; the overlap only warms it up. The kept frames tile the file exactly, and the decoders run on the merged frame sequence.
- **Decoding:** CTC greedy for v1–v5; `--beam N` runs a CTC prefix beam search (no language model). v6 uses the factored RNN-T greedy decode, carrying the prediction network state across windows, run on the CPU frame by frame. `--decoder ctc` uses v6's CTC branch.
- **Script constraints (on by default for CTC):** the decode-time rules below. `--script-rules off --lid-script-set off` gives the unconstrained decode exactly. `--json` reports the options and the allowed scripts per file under `constraints`.
- **Output:** the tokenizer's text: lowercase, no punctuation, each word in its own script.
- **`--json`:** per file, the words with start and end times (from emission frames: approximate word starts, not forced alignment) and their script. With LID heads it adds each word's language, the utterance language probabilities (for audio over one window, the duration-weighted mean over windows) and the code-switch spans snapped to words, each with its text.
- **Devices:** `auto` picks CUDA, then MPS, then the CPU. dtype `auto` is bf16 on CUDA GPUs that support it and fp32 on MPS and the CPU (`--dtype bf16|fp16` opts into half precision; its accuracy is unverified). The log-mel is computed on the CPU for MPS, which has no float64 FFT. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set. One second of silence goes through the model at load: if it fails, MPS half precision falls back to MPS fp32, and MPS fp32 falls back to the CPU. fp16 on MPS overflowed this way on the tiny tower (a clipped-linear bound does not fit in fp16); bf16 ran.
- **Timing:** each file's real-time factor (wall time / audio duration, including decoding the audio file) is printed on stderr, plus a total for several files.

Measured with the tiny random tower only (not the real encoder): the smoke pilot chain's `v4+lid` run went through sync → fetch_tok → export → verify → transcribe on a workstation. A 75 s FLAC and a 5 s MP3 transcribed identically on MPS and the CPU. MPS and the CPU agree within 1e-3 on the CTC and LID outputs (test). Real-model RTF and MPS numerics on the real encoder are not measured yet.

### Decode-time script constraints (`indicear.decoding`)

Two failure modes of the pilot model's free script choice, fixed at decode time without retraining (training losses, weights and the LID outputs are unchanged):

1. **Mid-word script switches** (`--script-rules`): `familyಿಗೆ`, `royalोd`, `ಪ್ರाणियों`, `स्थानीयಯ`. Every token has a script (its id range) and a kind from its piece: a word start (`▁` + letters), a bare `▁`, neutral (digits, `<unk>`), a continuation starting with a combining mark (Mn/Mc/Me, ZWJ/ZWNJ), or another continuation. The decoder tracks the current word's script (reset by a bare `▁`, kept by neutral pieces). A word start, bare `▁` or neutral piece is always allowed, and so is a same-script continuation (marks included). A full-akshara continuation is allowed after Latin (`companyಗಳು` stays possible). A mark after another script's piece is never allowed, and Indic to another Indic script needs a word boundary. **Latin split** (`--latin-split on`, the default): a Latin continuation glued to an Indic word (`कि` + `etition`, a missed word boundary) becomes a new word, with a Latin `▁` inserted, instead of being blocked. Blocking it dropped the English words and cost code-mixed span accuracy (strict rule in the table below). Known limitation: a Latin stem plus a full Indic akshara (`standeन`) cannot be told from a stem plus an Indic suffix, so it stays allowed.
2. **Indic-Indic bleed** (`--lid-script-set`): Tamil words in Kannada audio, Kannada fragments in Hindi audio. Allowed scripts: Latin, the top-1 utterance-LID language's script, and the scripts of the top `--lid-topk` (2) Indic languages above `--lid-threshold` (0.15). `--lid-frame-share X` also adds languages that are the frame-LID argmax on at least a share X of speech frames (off by default). Models without LID heads skip this rule.

Both rules mask the factored log-probs (a hard constraint, no renormalization). Greedy CTC becomes a constrained best path: per frame, the most likely allowed label given the word state, tracked across blanks and repeats. Each frame is reduced to the best token per (script, kind group), [T, 11, 3], so it is exact and cheap on long audio. The CTC prefix beam search prunes each prefix's extensions by the same rule. On by default in the transcribe CLI and in `python -m indicear.decoding eval`. Off by default in `DesignA.decode` (pass `constraints=decoding.Constrainer.from_tokenizer(tok)`), so training dev reports and `ablate eval` are unchanged. With both rules off, every decoder gives the old output exactly (tested).

**Measured on an Apple M5 Pro** (MPS, fp32, CTC greedy, pilot `v4+lid` final, the first 600 utterances per set, `results/pilot/eval-mac-final-constrained/summary.md`). The unconstrained rerun in the same pass reproduced `eval-mac-final` exactly. With the defaults on:

- **CER and WER:** lower on all 21 Indic sets. LibriSpeech test-clean is unchanged (7.23 CER).
- **Kannada (Kathbath):** CER 20.0–24.1 → 17.8–21.2, WER −6.6 to −8.5, script errors 11.5–13.9% → 3.6–5.0%.
- **Tamil and Hindi (Kathbath):** CER −0.7 to −1.2, script errors down by 1.5–3.8 points.
- **Code-mixed span accuracy:** up on IndicContextEval (80.1 → 81.1), MUCS (61.0 → 62.1), IndicVoices kn/ta and Vaani kn/ta. Vaani hi is unchanged, and IndicVoices hi is 81.06 → 80.98 (one span).
- **Strict rule** (`--latin-split off`): lower WER, but English span accuracy drops (IndicContextEval 71.5 → 68.3, total 80.1 → 78.8). That is why the split is the default.

## GPU use

- **Batches:** length-bucketed and filled to a padded-frame budget. A new run sizes the budget from `torch.cuda.mem_get_info`: it grows until a real worst-case step peaks at 85% of this job's share. On CUDA OOM, the batch is split into twice as many micro-batches and retried, and that split count persists. No edits are needed between a 24 GB L4, an 80 GB A100 and a 96 GB RTX PRO 6000 (bf16; CUDA 13 wheels cover sm_120).
- **Feeding the GPU:**
  - free-threaded Python 3.14t with `PYTHON_GIL=0`;
  - spawn DataLoader workers for audio decoding and corruption (codecs run ffmpeg on the CPU), with pinned memory and non-blocking copies;
  - Gemma 4 log-mel features on the GPU in the main process (`--features gpu`, the default with CUDA, for `ablate cache` and the robustness probe). `--features cpu` keeps the Hugging Face extractor in the workers. The backend is not part of the cache identity, so a cache can resume on either. With `gpu`, the first batch is also run through the Hugging Face extractor, and the run stops on a mask mismatch or a difference above 1e-4. The FFT, mel matmul and log run in float64, as numpy does after the float32 window, so features match it to one float32 ulp (4.8e-7 in the tests);
  - a background thread that gathers the next cached batch;
  - SpecAugment vectorized on the GPU;
  - bf16 autocast and TF32;
  - `torch.compile` on the heads, falling back to eager when it fails.
- **Cache placement:** VRAM when it fits in 45% of free memory, otherwise pinned RAM. Concurrent jobs use `--cache-placement mmap`, which shares the OS page cache.
- **Concurrency:** `PARALLEL=N` runs N variants (or N slices of robustness conditions) with `--vram-share 1/N`.

## Compute and disk (estimates, unverified)

**FLEURS, 12 languages:**
- about 95 h of training audio (7–10 h per language);
- about 14 h of dev;
- about 4 h of synthetic code-switch audio (7 pairs × 100);
- the download is about 20 GB.

**Layer-6 cache:**
- 90,000 frames per hour of audio × 2 KiB per frame (bf16);
- about 20 GB for 113 h (about 10 GB in int8);
- `AUG_VIEWS=2` triples the training part, to about 58 GB in bf16.

**Throughput:**
- The probe measured about 700 h of audio per hour through all 12 layers on an L4.
- Caching layers 1–6 should take 10–20 min on an L4.
- Training runs 6 layers forward and backward per step, at an assumed 1,000–1,500 h per hour on an A100. That is about 40–60 min per variant for v2–v4 at 10 epochs.
- v1 costs about ⅓ of that (no backward through layers 7–12), v5 about 2× (two views), and v6 more again (RNN-T joint).
- **Total:** about 3–5 A100-hours for v1–v4, the robustness probe (17 conditions × 12 languages × 200 utterances) and the biasing test (since removed).

## Metrics

Every report states which definition it uses.

- **CER, WER:** jiwer 4.0.0 on normalized text; CER counts spaces.
- **Script-normalised WER (sn-WER):** WER where an English word written in native script and in Latin both count as correct.
  - Uses the `refs` field the loader stores: the native-script form of glossed IndicVoices, Vaani and IndicContextEval rows.
  - Each alternative reference is word-aligned to the primary one (difflib). A differing block counts as an equivalent spelling only when one side is Latin and the other is only in native script; same-script variants such as Vaani's second transcription do not count.
  - Blocks of equal length pair word by word; otherwise the whole block takes one form.
  - The distance is the minimum word edit distance to any path through the resulting lattice, so one hypothesis may mix both forms. The denominator is the primary reference's word count.
  - Rows without `refs` (Kathbath, FLEURS, MUCS, LibriSpeech) score as strict WER: no transliteration is attempted.
- **Script-error rate:** the share of output words in a script that is absent from the reference.
- **OOV-word recall:** of reference words never seen in the training transcripts, the share recognized exactly.
- **Code-switch span accuracy:**
  - Hypothesis words in a span's script, in order, must have CER ≤ 0.5 against that span.
  - The two scripts must appear in the right order.
  - Per-span CER and order accuracy are reported too.
- **Real code-switch span accuracy** (rows whose `word_langs` hold more than one language run):
  - Each run of one language is a span.
  - The hypothesis words aligned to it (jiwer word alignment, plus insertions strictly inside the span) must be in that language's script, with CER ≤ 0.5.
  - Reported overall and per span language.
- **Language ID** (LID variants): `metrics.lid_report` gives utterance accuracy, macro-F1, the confusion matrix with the hi/ur, hi/mr and pa/hi pairs, and accuracy by duration. `lid_span_report` gives span and sequence accuracy on code-mixed rows. The spans come from `lid.utterance_spans`, snapped to the CTC words when the model has a transcript head.
- **Negatives:** the share of non-speech clips with any output.
- **Robustness:** CER per condition and language, with "collapse" flagged when CER exceeds 2× clean or clean + 20 points.

## Verified locally (CPU, `uv run --locked pytest`)

- **Full warm-started run** (tests/test_warmstart.py, test_full_warm.py, test_loanwords.py, the new cases in test_gpu_augment.py):
  - `AggregateTokenizer.extend` keeps the kept scripts' models byte for byte and the script order;
  - head expansion: blank and kept-script log-probabilities, the phonetic heads and the LID heads are bit-identical after expansion (copied class rows; fresh ones for a source that gives the new scripts no probability), and within-script distributions are exact otherwise; fresh rows get gradient; checkpoint sources; the structure check; a resumed warm-started run needs only its record;
  - `RowLr` equals a separate parameter at 3× the LR over five AdamW steps; the optimizer groups, the warmup peak and the cosine floor; unchanged config hashes for the old options (the pilot's checkpoint still hashes to `34ad89478fc81611`);
  - the `full_warm` mixture: licence-verified training sources, the eval sets per language, row draws matching the pinned shares (±0.6 points), the Kathbath weight, the processed-hours targets, disjoint code-mixed pools, and the pilot's stream weights unchanged;
  - loanword rewrites in training rows (native form kept in `refs`) and eval rows (rewrite as the alternative), only for the named sources;
  - prep packs: a `--with-audio` restore makes prep download and decode nothing, a text-only restore skips the text pass and the tokenizer and reproduces the dev carve-out, a corrupted file fails its sha256;
  - `push_init.sh` (SESSION=local) and `scripts/full_warm.sh` in smoke mode, end to end and rerun.

- **Call path:** the staged path equals `Gemma4AudioModel.forward` exactly, at layers 6 and 12 and `output_proj`. Checked on a randomly initialized 12-layer tower with the pinned attention geometry (chunk 12, left context 13, right context 0).
- **Caching:** per-utterance caching is padding-independent (difference about 1e-6).
- **Adapters:** zero-initialized adapters are an exact identity, the base weights stay byte-identical after training, and the adapters detach cleanly.
- **Losses:**
  - the RNN-T loss matches torchaudio 2.11.0 in values and gradients;
  - the factored output sums to 1;
  - its lean gather equals the full computation.
- **Variants:** all six run forward, backward, decode, save and load.
- **Units and text:** tokenizer round trips in all 11 scripts, mixed-script tokens, and unseen characters spelled without `<unk>`. Phonetic rules, normalization and the gloss rewrite are tested.
- **Batching:** the frame budget, bucket coverage, resume order, auto-tune and OOM backoff are tested.
- **Interrupted runs:**
  - a run killed mid-epoch resumes with the same steps and losses;
  - so does a resume from a synced copy into a fresh run dir;
  - a fresh VM restores from the copy;
  - a corrupt latest checkpoint falls back;
  - a config change is refused.
- **Sync:** `sync_runs.sh` under macOS bash 3.2 pulls only manifest-listed files, rejects sha mismatches, prunes to the last 2 plus best, and exits on DONE.
- **Model packages and transcription** (tests/test_export.py, tiny tower):
  - export → `--verify` → transcribe round trip for v4+lid, v2+lid and v6 packages; a checkpoint export (`--checkpoint best`) has byte-identical tensors and the same transcript as the published one;
  - verification catches a changed byte, an unlisted file and a `.pt`; a tokenizer from another ROOT and a file that differs from `LATEST.json` stop the export;
  - the kept frames of overlapping windows tile a 10.3 s file exactly, and the first window equals a single pass;
  - the CTC prefix beam search finds the prefix that greedy misses; the chunked RNN-T decode equals the one-pass greedy;
- **Decode-time script constraints** (tests/test_decoding.py): synthetic log-probs where the unconstrained best path writes `familyಿಗೆ`, `royalो`, `royalोd`, `ಪ್ರाणियों` and `स्थानीयಯ` and the constrained one cannot; `companyಗಳು`, same-script marks, digits and a bare `▁` before another script stay possible; Tamil words in Kannada audio and a Kannada word in Hindi audio are masked by the LID script set (top-k, threshold, top-1 always kept, the frame-LID share); with both rules off, greedy, the prefix beam search, `DesignA.decode`, `Transcriber.transcribe` and the CLI give the old output exactly (random log-probs and the tiny tower); every reference encoding of the test tokenizer satisfies the rule; constrained greedy and beam outputs on random log-probs and on the tiny tower never break it.
  - the CLI's `--json` on WAV (8 kHz stereo), FLAC and an ffmpeg-only M4A gives LID spans and language probabilities; MPS matches the CPU within 1e-3;
  - `fetch_tok.sh` against a local ROOT.
- **Features:** `GpuLogMel` equals the Hugging Face Gemma 4 extractor (max |diff| ≤ 1e-4, measured 4.8e-7; identical mask) on random, quiet, band-limited and resampled audio, padded batches, frame and padding boundaries and over-30 s truncation. Both feature backends give matching loader output, the same cache metadata and index and the same probe results; a corrupted filterbank stops the first-batch check.
- **End-to-end on CPU:** `ablate train` → results, DONE, skip on rerun and report; and the robustness probe with a fake extractor, both with a tiny tower patched in for the downloaded encoder.
- **Data:** synthetic parquet and tar fixtures that mimic the verified schemas exercise:
  - the markup rewrites (multi-word and glued glosses, tags, misspelt tags, unintelligible drops), `word_langs`, and that every cleaned text tokenizes;
  - hours caps, resume after interruption, the disk-budget stop, and the MUCS and Kathbath tar layouts;
  - the speaker-disjoint dev split, the eval-leak removal and assertion, temperature shares and sampler weights;
  - `prep --mixture` and `prep --stream` end to end;
  - streaming: exact resume mid-stream, the bounded shard cache, interleave frequencies, the frame budget, decoded batch format, and eval-audio drops.
- **Pilot path** (tests/test_pilot.py, test_gpu_augment.py, the new cases in test_metrics, test_lid and test_stream):
  - `prep --stream` → `train --lower live` (v2 and v4+lid sharing one stream, GPU augmentation on the CPU device) → kill at step 3 → resume: the same losses as an uninterrupted run;
  - `eval` on every set type, the skip on rerun, `report`, and the baselines runner;
  - a group member equals a solo run of the same variant, so adapters never leak between members;
  - a group whose members were saved at different steps resumes at the common step;
  - hours-to-steps conversion;
  - `scripts/pilot.sh` in smoke mode, foreground and queued (`launch.sh`/`WAIT_FOR`), with the rerun skip, and the FAILED marker aborting a waiting job;
  - GPU augmentation against the CPU Corruptor: SNR within ±0.5 dB, the RIR bank's Schroeder RT60 within 15% (as the CPU RIRs), power kept by reverb, the tel8k band-limit, speed and length;
  - per-row determinism and kind frequencies;
  - worker augmentation (AugmentPolicy, CodecAugment) identical after a stream resume;
  - script-normalised WER cases, real code-switch spans, LID-only and v3+lid decoding and scoring, and cached-path training of `lid` and `v2+lid`.

## Verified on the GPU runs, and what remains unverified

This section was a list of open checks before the first GPU run. The pilot (A100) and the full run (RTX PRO 6000) have since verified the staged call path on the real weights, the transformers build under free-threaded 3.14t, `torch.compile`, fused AdamW, expandable segments, the GPU log-mel, live streaming of the full mixture, throughput and VRAM autotuning, real model packages (export, verify, transcribe), and the baselines (IndicConformer and SraVaani; see `docs/full-run.md`). Still unverified or approximate:

- **Half precision at inference:** `--dtype bf16|fp16` runs, but its accuracy is not measured.
- **Phonetic rules:** they are hand-written approximations. Medial schwa deletion is not modelled, English is spelling-based, and Urdu has no short vowels. Label rates per language are written to `prep_report.json`.
- **ep-hindi-asr:** its code is gated and was not read; the adapter follows the card's documented API.
- **Real room impulse responses:** the released runs used synthetic RIRs only, pending a licence review of the measured sets.

## Open issues

- **TDT:** NeMo's TDT loss is only importable through `nemo.collections.asr`, which pulls in the ASR stack (and numba on 3.14t). The transducer uses an exact RNN-T loss behind `losses.make_transducer_loss`, where a TDT recursion can plug in.
- **Augmentation:** cached views (`AUG_VIEWS`) and live rows are corrupted on the GPU by default (noise, babble, reverb, speed, tel8k), with only codecs in the CPU workers. The GPU and CPU paths agree statistically, not sample for sample (see "Augmentation in live mode").
- **Code-switch test set:** the synthetic set is inter-sentential only; real intra-sentential sets come later.
- **Data:** the multi-source loaders and the streaming path are written and tested on CPU with synthetic shards that mimic the verified schemas. One real shard (the People's Speech microset) went through the download and decode path. The full 12-language mixture has since streamed through the full run (`docs/full-run.md`).
  - The script-normalised WER has no transliteration fallback, so it equals the strict WER on sets without stored native forms.
- **Framework:** `trainer.py` should move into `runtime/` once its model interface (`losses`, `head_modules`, `state_dict`) is formalized.
