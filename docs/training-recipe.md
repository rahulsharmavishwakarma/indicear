# Training recipe: 12-language warm-started run (and the pilot it started from)

This is how Vesper-IndicEAR (run `full_warm`, variant `v4+lid`) was trained, end to end, with the IndicEAR framework. The detailed run records are [full-run.md](full-run.md) and [pilot.md](pilot.md); the per-run settings table is [experiments.md](experiments.md). Numbers here come from those pages, `results/runs/full_warm/` (`command.sh`, `result.json`, `train.csv`, `gpu.csv`, `init_from.json`) and the code defaults.

## 1. What it is

IndicEAR turns the **frozen audio tower of `google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b`** into a speech recogniser for 11 Indian languages plus English (bn, gu, hi, kn, ml, mr, or, pa, ta, te, ur, en). The encoder's weights are never changed; its text and vision models are dropped. What trains:
- residual bottleneck adapters on encoder layers 7–12;
- self-conditioned phonetic CTC heads at layers 6 and 9, over one shared transliterated unit set;
- a script-factored CTC output: script first (11 scripts), then the subword within that script;
- utterance- and frame-level language-ID heads, which report the language but never route the transcript.

About 11.1M trainable parameters for the transcript path plus about 0.67M for LID ([docs/framework.md](framework.md), checkpoint sizes; [architecture.md](architecture.md)). The exported package is 45.2 MiB.

## 2. Model

| Part | Setting |
|---|---|
| Input | 16 kHz audio → 128-bin log-mel on the GPU → encoder at 25 frames/s |
| Encoder layers 1–6 | Frozen, run under `no_grad` every step (`--lower live`; rows stream, so nothing is cached) |
| Encoder layers 7–12 | Frozen, plus a residual adapter after each block's FFN: LN → down 1024→128 → SiLU → up (zero-init) |
| Phonetic CTC | Self-conditioned, at layers 6 and 9; 82 shared units (ISO 15919 / common-label-set for Brahmic scripts, Latin for English, romanised Urdu) |
| Final head | CTC with a two-level output: p(script) × p(piece \| script); 11 scripts (deva, beng, guru, gujr, orya, taml, telu, knda, mlym, arab, latn) |
| Tokenizer | Aggregate of per-script SentencePiece unigram models; vocabulary 8,964 (deva and latn 1,024 each, the other scripts 768) |
| LID heads | Utterance: learned sum of layers 6 and 12, attentive mean+std pooling, MLP (hidden 256) to 12 languages. Frame: non-speech + 12 languages per frame. Both read detached states, so they do not change transcription training |
| Loss | `0.85 × CTC + 0.15 × phonetic CTC + 0.1 × utterance LID + 0.2 × frame LID` (label smoothing 0.1, class-balanced LID). The pilot used phonetic 0.3 and frame LID 0.1 |
| SpecAugment | On |

**Not trained** although present in the design and the code: the TDT/transducer decoder (`v6`) and CR-CTC consistency (`v5`). The FLEURS ablation picked v4 (mean CER over 11 languages: v2 27.67, v3 28.17, v4 27.61, with the fewest script errors), and the pilot added LID (`v4+lid`).

## 3. Data

All training data streams from the Hugging Face Hub at pinned revisions; only eval and dev sets are downloaded. Licences: [data-licenses.md](data-licenses.md). Everything used for training is CC BY or CC0; MUCS (CC BY-SA) and SPRING-INX are eval-only, Shrutilipi was not readable by the account, and no non-commercial data was used.

**Mixture** `indicear/mixtures/full_warm.json`: 53 streams, 5,504 h of planned pools before speaker exclusion. Rows of 0.5–20 s. "cm" streams keep only code-mixed rows (disjoint from the main pools where possible).

| Lang | Planned pool hours by stream | Pinned share of drawn rows |
|---|---|---:|
| bn | IndicVoices 300, Kathbath 90 (w2), Vaani 100, IV cm 40, Vaani cm 15 | 9% |
| te | IndicVoices 300, Kathbath 150 (w2), Vaani 110, IV cm 40, Vaani cm 25 | 9% |
| ml | IndicVoices 300, Kathbath 140 (w2), Vaani 25, IV cm 40 | 9% |
| pa | IndicVoices 300, Kathbath 130 (w2), Vaani 12, IV cm 40 | 8% |
| mr | IndicVoices 300, Kathbath 150 (w2), Vaani 40, IV cm 40, Vaani cm 10 | 7.5% |
| gu | IndicVoices 280, Kathbath 120 (w2), Vaani 11, IV cm 40 | 7.5% |
| or | IndicVoices 300, Kathbath 80 (w2), Vaani 30, IV cm 40 | 7.5% |
| ur | IndicVoices 300, Kathbath 75 (w2), IV cm 40 (no Vaani: 2 h on its card) | 7.5% |
| hi | IndicVoices 150, Kathbath 120 (w1.5), Vaani 60, IV cm 40, Vaani cm 50 | 4.5% (refresher) |
| ta | IndicVoices 150, Kathbath 150 (w1.5), Vaani 12, IV cm 40 | 5% (refresher) |
| kn | IndicVoices 160, Kathbath 140 (w1.5), Vaani 60, IV cm 60, Vaani cm 25 | 6.5% (refresher) |
| en | LibriSpeech 120 (clean-360 + other-500), People's Speech 80, VoxPopuli 60, Vaani English 14 | 4% |
| code-mixed streams | (the "cm" rows above) | 15% |

- **Sampling:** temperature α = 0.5 inside the pinned per-language shares; non-speech negatives at a fixed 2%; a 3 h shuffle buffer (`--buffer-seconds 10800`).
- **Transcripts:** IndicVoices `[gloss]` and Vaani `{gloss}` English glosses are applied only when the native word sounds like the gloss (`markup.sounds_alike`). Kathbath's native-script English loanwords are rewritten to Latin by the loanword normaliser ([loanwords.md](loanwords.md)).
- **Speaker-disjoint protocol:** every dev and eval speaker (Kathbath test-known, IndicVoices valid, …) is removed from training. For Kathbath this removes 49% (bn) to 100% (or) of train rows ([full-run.md](full-run.md#kathbath-speaker-exclusion)).
- **Mid-run data changes (runtime flags, not hashed):**
  - from step 1,079, `--zero-streams kathbath/ml,kathbath/or,kathbath/ur,kathbath/gu`: the exclusion left these with too few rows (ml would have been repeated about 310 times);
  - from step 3,427, `--stream-scale indicvoices/hi=2.24,vaani/hi=2.24`: Hindi's draw share 4.5% → about 7.5%, equal to Urdu, after Hindi drifted into Urdu script.
- **Dev:** speaker-disjoint, carved from the training sources: 1% per language, 0.5–1 h.
- **Eval:** 83 sets in the mixture (Kathbath clean/noisy × test-known/unknown, IndicVoices valid, Vaani test, Vaani Benchmark, IndicContext, MUCS, Lahaja, Svarah, LibriSpeech, VoxPopuli) plus FLEURS dev and the robustness conditions: 98 sets in the final eval.

**The pilot** (`mixtures/pilot.json`) used the same sources for hi, ta, kn and en only: about 650 h planned, about 467 h streamable after speaker exclusion.

## 4. Hyperparameters

| | Pilot (`pilot-v4lid`) | Full run (`full_warm`) |
|---|---|---|
| Start | From scratch (frozen encoder) | Warm start from the pilot's final weights (step 7,423), new config hash `91e6c3abd66294ff` |
| Batch | Dynamic frame budget, autotuned to 85% VRAM: 67,045 frames (about 45 min of padded audio) | 71,676 frames (about 48 min), as 2 micro-batches from step ~350 after one OOM (same gradient) |
| Packing (real / padded frames) | 0.48 measured (0.5 h buffer) | 0.784 measured (3 h buffer; 0.75 assumed) |
| Steps | 7,423 | 13,394 (`--train-hours 8000`) |
| Audio processed | 2,664 h | 8,356 h |
| LR | Peak 1.83e-3, warmup 300, cosine to 5% | Peak 6e-4, warmup 300, cosine to 5e-5 |
| Warm-start rules | — | Keep the pilot's deva, latn, taml, knda tokenizers and output rows; new scripts get fresh rows with bias −2 and a ×3 LR on those rows (`--new-lr-mult 3`, `trainer.RowLr`) |
| Optimiser | AdamW, weight decay 0.001, grad clip 1.0, seed 0 | Same |
| Precision | bf16 autocast (CUDA); the 21-step Mac resume used fp16 + GradScaler | bf16 autocast; non-finite gradient norm skips the step |
| Augmentation | 50% of rows clean. Weights: noise 0.25, babble 0.20, reverb 0.175, reverb + point noise 0.175, speed 0.15, 8 kHz band 0.05 | 50% clean. Weights: reverb 0.3, reverb + point noise 0.175, noise 0.2, babble 0.15, speed 0.125, 8 kHz band 0.05 |
| Augmentation details | SNR 5–20 dB; RT60 0.3–0.9 s from 128 synthetic RIRs; speed 0.9/1.1; codecs (GSM, Opus, AMR) on 15% of clips; all on the GPU, seeded per row so a resume repeats it | Same; real RIRs off pending licence review |
| Checkpoints | Every 500 steps or 10 min; last 2 + best kept | Same; dev eval every 1,000 steps |

## 5. Hardware and cost

| | Pilot | Full run |
|---|---|---|
| GPU | Colab A100-SXM4-80GB (one 21-step resume on an Apple M5 Pro, MPS) | Colab G4: RTX PRO 6000 Blackwell Server Edition, 96 GB, 48 vCPU |
| Software | Python 3.14 free-threaded (GIL off), torch 2.14.1+cu130 | Same (Python 3.14.8) |
| Colab rate | 6.77 units/h | 8.9 units/h |
| Units used | About 55 (plus about 2 for the L4 probes) | About 67, including prep, restarts and eval |
| Wall time | About 4.3 A100-h of training | About 6.9 h including restarts; about 6 GPU-h at full speed |
| Throughput | About 620 h of audio per GPU-hour; about 29 steps/min; GPU 80–100%, about 76 GB VRAM | About 35,000 frames/s, about 1.6 s per step (median over `train.csv` after step 1,100); GPU 93–99% busy once the loader was fixed (90% mean over the whole log, 95% over the last segment); peak 95 GiB VRAM |

Derived from these: the G4 processed about 1,390 h of audio per GPU-hour (8,356 h in about 6 h), about 2.2× the A100's 620 h/h, at 1.3× the hourly rate. Per hour of audio it was therefore about 40% cheaper. That holds only once the input pipeline keeps the GPU busy (section 8).

## 6. Reproducing it

All commands run from the repository root. A run's exact command is saved as `results/runs/<run>/command.sh`; rerunning it resumes.

**Setup and launch on a GPU VM** (stages setup → prep → train → eval → report, each queued with `launch.sh` and marked `$ROOT/full_warm/<stage>.DONE` or `.FAILED`):

```bash
# on a workstation: upload the finished pilot run (weights, configs, tokenizer) as the warm-start source
bash scripts/push_init.sh SESSION ~/path/to/pilot/v4+lid /content/full/init/v4+lid

# on the VM (HF_TOKEN in the environment, or a saved Hugging Face login)
ROOT=/content/full INIT_FROM=/content/full/init/v4+lid \
TRAIN_EXTRA="--new-script-bias -2 --cache-gb 100 --readahead 64" \
  bash scripts/full_warm.sh            # queues every unfinished stage; rerun to resume
bash scripts/status.sh                 # progress; logs in $ROOT/logs/full_warm-<stage>.log
```

`SMOKE=1` runs the same chain on a CPU in about 20 s with a tiny tower and synthetic shards.

**What the train stage runs** (from `command.sh`; the last two flags were added mid-run):

```bash
python -m indicear.ablate --root /content/full train --lower live --variants v4+lid \
  --init-from /content/full/init/v4+lid --train-hours 8000 --packing 0.75 --buffer-seconds 10800 \
  --peak-lr 6e-4 --final-lr 5e-5 --warmup-steps 300 --new-lr-mult 3 --inter-weight 0.15 \
  --lid-frame-weight 0.2 \
  --augment reverb=0.3,reverb_pointnoise=0.175,noise=0.2,babble=0.15,speed=0.125,tel8k=0.05 \
  --new-script-bias -2 --cache-gb 100 \
  --zero-streams kathbath/ml,kathbath/or,kathbath/ur,kathbath/gu \
  --stream-scale indicvoices/hi=2.24,vaani/hi=2.24
```

Prep for it: `--mixture full_warm --stream --tok-from /content/full/init/v4+lid/tok --tok-keep deva,latn,taml,knda`.

**Resume rules:**
- A run resumes from its newest checkpoint only when its **config hash** matches; `--force-new-run` starts over.
- The hash covers the variant, trainer settings, mixture name and file sha256, seed, buffer size, augmentation and the warm-start source. Editing the mixture file therefore blocks a resume.
- Not hashed, so changeable on resume: host settings (`--device`, `--splits`, `--precision`, `--stream-workers`, `--readahead`) and the data-steering flags `--zero-streams` and `--stream-scale`, which are logged and kept in `command.sh`.
- The stream state (each stream's epoch, shard and row; the shuffle buffer; RNG) is in the checkpoint, so a resumed run sees the same rows without repeats or gaps.

**Eval and baselines:**

```bash
bash scripts/full_warm.sh --stage eval           # every eval set, `ctc` and `ctc_constrained`
bash scripts/full_warm.sh --stage report         # results/eval_summary.md
bash scripts/baselines.sh --systems indicconformer-ctc --max-per-lang 300 --robust-conditions ""
```

**Copying to a workstation and packaging:**

```bash
bash scripts/sync_runs.sh --keep-all SESSION /content/full/runs/v4+lid LOCAL_DIR   # sha256-verified pulls
python -m indicear.export --run LOCAL_DIR --tok LOCAL_DIR/tok --out PKG --with-encoder
python -m indicear.export --verify PKG
python -m indicear.transcribe --model PKG talk.wav
python -m indicear.decoding eval --model PKG --manifests ROOT/manifests --per-lang 300 \
  --out OUT/auto --baseline-out OUT/unconstrained --hint-out OUT/hinted
```

On Apple MPS set `PYTORCH_MPS_HIGH_WATERMARK_RATIO` above `PYTORCH_MPS_LOW_WATERMARK_RATIO` (we used 0.81 and 0.71).

## 7. Decoding and scoring

**Decode-time script rules** (`indicear/decoding.py`; no retraining; on in `transcribe` and `decoding eval`, off in training dev reports):
- **(a) Word-boundary rule:** the script may change only at a word boundary; no combining mark may follow a different script.
- **(b) LID script set:** only Latin plus the scripts of the top-2 utterance-LID languages above 0.15, counted by distinct script.
- **(c) Exclusive scripts:** Devanagari and Urdu (Arabic) script never mix in one utterance; the one whose languages the utterance LID finds likelier wins.
- **Language hint** (`transcribe(..., lang_hint=)`, `decoding eval --hint-out`): replaces the LID with the given language, for comparisons with systems that are told the language.

**Scoring** (`indicear/metrics.py`):
- `score_fold` folds spelling conventions before scoring: Malayalam chillus, Bengali khanda ta, Marathi eyelash ra, Urdu heh, tatweel and vowel marks, NFKC presentation forms, native digits.
- **Strict CER** compares against the reference as written.
- **Script-normalised CER (sn-CER)** accepts an English word in Latin or native script, scoring against the closest whole reference. Use it to compare systems: our IndicVoices and Vaani references write English in Latin, while IndicConformer writes it in native script.

## 8. Training at scale: challenges and how IndicEAR handles them

Training a 12-language model by streaming 8,000+ hours through one GPU raises problems that a small run never shows. Each one below is handled in the framework, so a new run gets the fix by default.

### Data that does not fit on the machine

The full mixture is 0.7–1.1 TB of audio; a Colab VM has about 190 GB free.
- **Streaming from the Hub** (`indicear/stream.py`): each (source, language) pair is its own stream of Parquet shards read at a pinned revision. Only eval and dev sets are downloaded.
- **Bounded shard cache** (`--cache-gb`, 100 GB for 53 streams): shards are fetched ahead and evicted when used.
- **Mixing:** temperature sampling inside pinned per-language shares, a shuffle buffer of 3 hours of audio (`--buffer-seconds`), and duration-bucketed batches.
- **Exhausted streams:** speaker exclusion can leave a stream with almost no usable rows (Kathbath Malayalam kept 6%). A stream that accepts no row in a full pass of its shards is marked dead instead of re-downloading in a loop.

### Keeping the GPU busy

At full speed the GPU consumes about 1,400× real-time audio (35,000 encoder frames per second); a single reading thread delivers about 266×.
- **Free-threaded Python 3.14 (GIL off):** loader threads run truly in parallel.
- **Per-stream read-ahead** (`--readahead 64`): every stream reads, parses and length-checks its next rows on its own thread, so drawing a batch never waits on one stream.
- **Row-parallel decode and augmentation:** a separate thread pool decodes each row's audio and prepares the batch while the GPU trains on the previous one.
- **Augmentation on the GPU** (`indicear/gpu_augment.py`): noise, babble, reverberation, point noise, speed and telephone band are applied on the GPU before the log-mel, seeded per row, so they cost no CPU and repeat exactly on resume.
- **No CPU oversubscription:** every ffmpeg call runs single-threaded with one OpenMP thread.

Effect on the full run: throughput rose from about 13,000 to 35,000 encoder frames per second and the GPU stayed 93–99% busy (figure `docs/figures/fig04_throughput_gpu.png`).

### Batching

- **Dynamic frame budget:** the batch is sized in encoder frames, not utterances. At startup an autotuner grows it until peak memory reaches about 85% of VRAM (71,676 frames, about 48 minutes of padded audio on a 96 GB GPU).
- **Packing:** duration buckets drawn from a large shuffle buffer keep padding low. A 3-hour buffer packs 78% real frames, against 48% with half an hour.
- **Out-of-memory safety:** if a batch does not fit, it is split into micro-batches with gradient accumulation, so the optimizer step and learning rate are unchanged.

### Continuous, resumable training

Long runs on preemptible VMs stop and restart; the framework makes a restart cost minutes, not the run.
- **Checkpoints** every 500 steps or 10 minutes, written atomically, with a `LATEST.json` manifest of file sha256s.
- **Exact data resume:** each stream's shard, row and epoch, the shuffle buffer and the RNG state are in the checkpoint, so a resumed run sees the same rows with no gaps or repeats. Streams are restored in parallel and start reading ahead immediately.
- **Config hash:** a run resumes only when its configuration matches; host settings (device, precision, worker counts, read-ahead) and data steering (below) can change on resume without starting over.
- **Staged pipeline:** setup → prep → train → eval → report, each queued under `nohup` with DONE/FAILED markers, so rerunning the launcher continues where it stopped. The prep stage counts as done only if its report is newer than the stage start.
- **Off-VM copies:** `scripts/sync_runs.sh` pulls verified checkpoints to a workstation; `scripts/resume_on.sh` restarts a run on a fresh VM from them.

### Steering the data mix during a run

Two runtime flags change the mix without changing the config hash, so the run resumes rather than restarts:
- `--zero-streams` stops drawing from named streams. The full run used it from step 1,079 for the Kathbath Malayalam, Odia, Urdu and Gujarati streams, which speaker exclusion had almost emptied.
- `--stream-scale` reweights streams. Hindi and Urdu share most of their spoken form but are written in different scripts; with Urdu at 7.5% of draws and Hindi at 4.5%, the model began writing Hindi in Urdu script (Hindi dev CER 24.8 → 44.5 at step 3,000). Raising Hindi to Urdu's share from step 3,427 brought Hindi back to 23.3 by step 4,000.

### Adding languages to a trained model

The full run warm-starts from the 4-language pilot (`--init-from`, `indicear/warmstart.py`):
- the kept scripts' tokenizers and output rows are copied unchanged;
- new scripts get fresh rows with a bias of −2 and three times the learning rate (`--new-lr-mult 3`). The bias matters: Adam moves a bias by about the learning rate per step, so a strongly negative start (−40 was tried) never recovers.

### Numerical safety

- **bfloat16 without a loss scaler:** a step whose gradient norm is not finite is skipped, counted and logged.
- **GPU FFTs:** band-limiting uses power-of-two FIR convolution, and augmentation retries after a cuFFT memory error instead of failing the step.

### Fair evaluation

- **Speaker-disjoint:** every dev and eval speaker is removed from training.
- **Script-normalised CER:** an English word counts as correct in Latin or native script, since datasets disagree on the convention.
- **Identical clips:** every system decodes exactly the same clips, with per-clip transcripts saved (`<set>.rows.jsonl`).

## 9. Results and limitations

Detail in [full-run.md](full-run.md) and the [model card](model-card.md). Script-normalised CER, the same 300 clips per language per set, mean over the languages every system supports (Urdu excluded):

| Test set | Vesper-IndicEAR | IndicConformer 600M | SraVaani-1.0 |
|---|---:|---:|---:|
| Kathbath clean | 21.0 | 3.4 | 3.2 |
| Kathbath noisy | 24.9 | 4.3 | 4.4 |
| IndicVoices | 24.3 | 7.7 | 7.4 |
| Vaani | 24.8 | 10.9 | 8.4 |
| IndicContext (code-mixed) | 32.4 | 5.2 | 5.3 |
| MUCS (Hindi, Bengali) | 43.5 | 49.0 | 49.3 |

- **Works:** a frozen embedding encoder plus 45 MB of trained weights transcribes 12 languages with built-in LID (FLEURS dev 24.8 CER, LID 97.8%; LibriSpeech 7.2/13.0 CER).
- **Doesn't:** it is several times behind dedicated ASR models on every family except MUCS. Closing that gap likely needs encoder-level changes, not more adapter runs.
- **Weak spots:** Urdu LID labels most Kathbath and Vaani Urdu as Hindi (the hint fixes much of it); Hindi regressed relative to the pilot; reverberation (RT60 0.9 s: 46.2 CER).

**Next-run fixes** (they change targets or tokenizers, so they could not be applied mid-run):
- normalise Malayalam ZWJ chillus and the Marathi eyelash ra before mapping, so targets have one spelling;
- a Marathi-aware Devanagari tokenizer (Marathi currently uses about 3.4 tokens per word against 1.4 for Hindi);
- match the Bengali/Odia "v" → ভ/ବ spelling in gloss matching;
- stricter IndicVoices noise-tag matching (it drops real glosses such as TV, music, door);
- size the Vaani code-mixed pools so they stay disjoint from the main pools;
- LID class weights from the pinned shares, not pool hours;
- a length-feasibility check for phonetic targets;
- plan batches for speed perturbation (pack to budget / 1.11) to avoid the micro-batch split.
