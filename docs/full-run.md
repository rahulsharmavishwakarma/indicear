# Full run: 12 languages, warm-started from the pilot (24 October 2026)


**Status:** complete. Experimentation paused on 25 October 2026.
- **Conclusion:** the frozen encoder with adapters works as multilingual ASR with built-in language ID, but it is 2–7× behind IndicConformer on character error rate. Closing that gap would need changes at the encoder level, not more adapter or decoder runs.
- **Not done:** the next-run fixes below and quantization stay deferred. SraVaani-1.0 was run on the shared subset afterwards (below).
- **Training:** all 13,394 steps, finished 03:44 IST on 25 October.
- **Eval:** 98 sets on the G4, finished 04:07 IST.
- **G4 released:** 04:31 IST, after every copy to a workstation was verified. The run used about 67 Colab units.
- **Comparison with IndicConformer:** run on an Apple M5 Pro workstation, finished 05:32 IST.
- **Audio processed:** 8,356 h; measured packing 0.784.
- **GPU busy:** 95% mean over the last segment.
- **Final step:** loss 1.78; LR 5e-5 at the end of the cosine.
- **Dev CER (training quick eval, unconstrained) at step 13,000:** macro 25.1. By language: bn 27.5, en 6.9, gu 25.4, hi 25.4, kn 33.4, ml 23.9, mr 13.8, or 25.7, pa 25.0, ta 28.7, te 26.4, ur 39.6.
- **Final package:** released as [`rahulsharmavishwakarma/vesper-indicEAR`](https://huggingface.co/rahulsharmavishwakarma/vesper-indicEAR) (45.2 MiB of trained weights, plus the bundled audio tower). Checkpoints from step 2,500 were kept off the VM.

## Final results

### Against IndicConformer (same rows, Mac)

- **Rows:** the first 300 rows per language of every eval set; both systems decode exactly the same rows.
- **Systems:**
  - IndicEAR: the final package. "Auto" means its own utterance LID chooses the script. "Hinted" means it is given the row's language (`decoding eval --hint-out`).
  - IndicConformer: `indic-conformer-600m-multilingual@2a77d305`, CTC, given the row's language.
- **Metric:** the script-normalised CER. It accepts an English word in Latin or in native script: our IndicVoices and Vaani references use Latin, while IndicConformer writes native script.
- **Files:**
  - Per-language tables: `results/full/comparison-indicconformer.md`, plus the strict-CER version `comparison-indicconformer-strict-cer.md`.
  - Raw results: `results/bench/vesper-indicear-{auto,hinted,unconstrained}/` and `results/bench/indicconformer-ctc/`.

Mean CER over the languages both systems cover:

| Eval family | Languages | Vesper-IndicEAR auto | Vesper-IndicEAR hinted | IndicConformer |
|---|---:|---:|---:|---:|
| Kathbath clean, unseen speakers | 11 | 24.0 | 21.9 | **3.3** |
| Kathbath noisy, unseen speakers | 11 | 28.9 | 26.2 | **4.1** |
| IndicVoices valid | 11 | 24.5 | 23.6 | **7.3** |
| Vaani test | 11 | 29.3 | 26.4 | **11.6** |
| IndicContext | 8 | 34.2 | 33.1 | **6.2** |
| MUCS (hi, bn) | 2 | **43.5** | **42.9** | 49.0 |

**Verdict:** IndicConformer is 2–7× lower in CER on every family except MUCS. IndicEAR is lower in only 1 of 65 (set, language) rows (MUCS Bengali).
- **Model sizes:** the frozen EmbeddingGemma 2 tower plus 45 MB of adapters and heads, against a 600M-parameter supervised conformer trained on these corpora.
- **What it shows:** a frozen embedding encoder transcribes 12 Indic languages plus English with built-in LID at 20–35 CER on the main test families (43.5 on MUCS).
- **What it does not show:** that this is close to a dedicated ASR model. It is not, yet.

**Caveats:**
- **Kathbath is in IndicConformer's training data.** Kathbath's test-known speakers are its training speakers by design; whether IndicConformer saw any test sentences is unverified. IndicEAR excluded every Kathbath eval and dev speaker, which removed 49–100% of Kathbath train rows.
- **Strict CER made IndicEAR look competitive.** IndicConformer's strict CER on IndicVoices and Vaani is 15.5 and 25.1, because its native-script English counts as errors against Latin references. On that metric IndicEAR leads on Vaani hi, te, gu and ta. The pilot's earlier "IndicEAR ahead on Vaani and MUCS" came from this artefact (corrected in `docs/pilot.md`).
- **Language hint:** IndicEAR benefits most for Urdu: Kathbath 54 → 33, Vaani 74 → 50, IndicVoices 27 → 20. Its LID labels most Kathbath and Vaani Urdu speech as Hindi.

### Against SraVaani-1.0 (same rows, Mac, 25 October)

SraVaani-1.0 (ARTPARK-IISc, 430M FastConformer + TDT, pinned `f5dd5358`) picks the language itself, like Vesper-IndicEAR auto, and does not support Urdu. It ran on an Apple M5 Pro GPU (MPS) in about 47 min. Means over the languages all four columns share (Urdu excluded), sn-CER:

| Eval family | Vesper-IndicEAR auto | Vesper-IndicEAR hinted | IndicConformer | SraVaani |
|---|---:|---:|---:|---:|
| Kathbath clean, unseen speakers | 21.0 | 20.9 | 3.4 | 3.2 |
| Kathbath noisy, unseen speakers | 24.9 | 24.6 | 4.3 | 4.4 |
| IndicVoices valid | 24.3 | 23.9 | 7.7 | 7.4 |
| Vaani test | 24.8 | 24.1 | 10.9 | 8.4 |
| IndicContext | 32.4 | 31.6 | 5.2 | 5.3 |
| MUCS (hi, bn) | 43.5 | 42.9 | 49.0 | 49.3 |

- SraVaani and IndicConformer are within about 1 point of each other except on Vaani, which is SraVaani's own training source. IndicEAR is lower than SraVaani in 1 of 59 (set, language) rows, on MUCS.
- FLEURS dev (sn-CER, all languages): SraVaani 5.9, IndicEAR 24.8. SraVaani's FLEURS dev WER (hi 12.0, bn 13.6) is in line with its card's test numbers (hi 10.9, bn 15.2).
- Per-language tables: `results/full/comparison-baselines.md`; raw results: `results/bench/sravaani/`.

### IndicEAR alone (full eval sets, G4)

`results/full/eval-g4/eval_summary.md` has 98 sets with the `ctc` and `ctc_constrained` decoders.
- **English:** LibriSpeech clean/other 7.2/13.0 CER (pilot 7.0/12.8); VoxPopuli 12.8; Svarah (Indian-accented English) 25.6.
- **FLEURS dev:** 24.8 CER over all languages, with LID 97.8%.
- **Constrained decoder (decode-time script rules):** lower CER on every Indic set; Hindi improves the most (Kathbath unseen 29.7 → 21.5).
- **Compared with the pilot (hi, ta, kn, unconstrained CER):**
  - English and Tamil are about the same.
  - Kannada is slightly worse.
  - Hindi is worse: Kathbath unseen 17.1 → 29.7, or 21.5 constrained.
  - The pilot was scored before the scoring folds were added.
- **Robustness (all languages, constrained):**
  - Noise at 10 dB: 30.2; at 5 dB: 35.2.
  - Babble at 10 dB: 33.0.
  - Reverb RT60 0.6 s: 38.3; 0.9 s: 46.2; reverb with point noise: 53.2.
- **Code-mixed IndicContext:** hi 35.7 CER, with code-mixed span accuracy 75.7.

## Setup

- **Model:** variant `v4+lid`, the same as the pilot (`docs/pilot.md`).
  - **Warm start:** from the pilot's final weights (`indicear/warmstart.py`).
    - Tokenizers kept: deva, latn, taml, knda.
    - New tokenizers: beng, guru, gujr, orya, telu, mlym, arab.
    - Output rows are mapped by script.
    - New-script bias -2 and new-row LR ×3.
  - **Config hash** `91e6c3abd66294ff`.
- **Data:** `indicear/mixtures/full_warm.json`.
  - 53 streams from IndicVoices, Kathbath, Vaani, LibriSpeech, People's Speech, VoxPopuli and Vaani English, plus mined code-mixed streams and local non-speech negatives.
  - Shares are pinned per language.
  - The loanword normaliser rewrites Kathbath's native-script English loanwords to Latin (`docs/loanwords.md`).
- **Training:**
  - Peak LR 6e-4, cosine to the final LR.
  - Phonetic CTC weight 0.15, LID frame weight 0.2.
  - 3 h shuffle buffer and a frame budget of 71,676 (autotuned).
  - From step about 350, each batch runs as 2 micro-batches after one OOM. The gradient is the same as for one piece.
- **Launch:** `ROOT=/content/full INIT_FROM=/content/full/init/v4+lid TRAIN_EXTRA="--new-script-bias -2 --cache-gb 100 --readahead 64 --zero-streams kathbath/ml,kathbath/or,kathbath/ur,kathbath/gu --stream-scale indicvoices/hi=2.24,vaani/hi=2.24" bash scripts/full_warm.sh` (the stream scale applied from step 3,427)

## Timeline and incidents

| When (UTC) | Step | What happened | Fix |
|---|---:|---|---|
| ~13:30–14:30 | — | ffmpeg used 66 threads per process, so the 48 cores were oversubscribed (load about 620) | `-threads 1` and `OMP_NUM_THREADS=1` for every ffmpeg call |
| ~14:30 | — | Prep was killed with SIGTERM, exited 0 and was marked done | A prep stage counts as done only if `prep_report.json` is newer than the stage start |
| ~14:45 | — | The text pass ran on one thread (about 2 h) | Parallel text pass; `INDICEAR_TEXT_PASS_SHARDS=3` reads a sample of each stream for the tokenizer. Prep then took about 13 min |
| ~15:00 | 0–~60 | With a new-script bias of -40, loss stayed at about 29.7. Adam moves a bias by about the LR per step, so -40 cannot recover | Bias -2, now the default. Restarted from the pilot weights |
| ~15:05 | — | The 40 GiB stream cache filled with 53 streams | `--cache-gb 100` |
| 15:22–15:33 | 0 | The GPU sat idle. One thread read rows from all 53 streams, supplying about 266× real time against about 740× needed | Per-stream read-ahead threads. Restarted from the pilot weights; nothing was lost |
| 15:35–15:40 | 0–99 | The GPU was busy about 15% of the time. Each batch (about 48 min of audio) was decoded on one thread, so about 8 of 47 threads worked | Rows decoded in parallel. Checkpointed at step 99 and resumed |
| 15:40–15:45 | 99 | The resume took 4.5 min: the buffer restore and the first shard opens ran stream by stream | Streams restore and start in parallel. Applied at the next restart |
| 15:55–16:31 | 99–1,079 | The GPU was 60–90% busy. One stream, `kathbath/ml`, had an empty read-ahead queue: excluding the eval speakers removes 94% of its rows. It was on its 18th pass over its shards and re-downloading them; Hugging Face returned 2 rate-limit errors | See "Kathbath speaker exclusion" below |
| 16:31 | 1,079 | Checkpointed and resumed once with all fixes. The restore took 2 min | GPU 93–99% busy, about 35,000 frames/s (about 1.6 s per step) |
| 17:28 | 3,000 | Dev CER: Hindi rose from 24.8 (step 2,000) to 44.5 while Urdu fell from 49.0 to 29.6. On 30 Hindi dev clips, 32% of output words were in Urdu script, and 11 of 30 utterances mixed both scripts (e.g. "आप doctor بن چاہتے ہیں") | Decode rule (c) (below) and a Hindi rebalance (`--stream-scale`) |
| 17:39 | 3,427 | Checkpointed and resumed with `--stream-scale indicvoices/hi=2.24,vaani/hi=2.24`: Hindi's draw share goes from 4.5% to about 7.5%, equal to Urdu. Kathbath Hindi is not scaled (it repeats its usable clips about 6 times already) | Not hashed: same config, resumed |

## Hindi and Urdu

Hindi and Urdu share most of their spoken form (Hindustani) but are written in different scripts; in these datasets the transcript's script is what separates them.
- **Mix:** the full mixture pinned Urdu at 7.5% of draws and Hindi at 4.5%, because Hindi was already trained in the pilot.
- **Effect:** after step 2,000 the model began writing Hindi speech in Urdu script.
- **Decode-time fix** (no retraining): rule (c) in `indicear/decoding.py` never mixes Devanagari and Urdu script in one utterance. It keeps the script whose languages the utterance LID finds likelier.

Measured on 30 Hindi and 30 IndicVoices-valid Urdu clips with the step-3,000 weights:

| Decoder | Hindi CER | Urdu CER | Utterances mixing both scripts |
|---|---:|---:|---:|
| Unconstrained | 50.7 | 25.5 | — |
| Rules a+b | 41.0 | 23.7 | 11 hi, 6 ur |
| + rule (c), choosing by decoded evidence | 44.2 | 22.6 | 0 |
| **+ rule (c), choosing by utterance LID** (default) | **39.7** | 23.6 | 0 |

The remaining Hindi error is mostly whole utterances written in Urdu script where LID chose Urdu (7 of 30). The training rebalance from step 3,427 targets that.

**After the rebalance.** Dev CER (unconstrained) at step 4,000: Hindi 23.3, Urdu 47.8. Hindi recovered and the swing reversed: the model now leans to Devanagari for Hindustani speech. Same 30 + 30 clips with the step-4,000 weights:

| Decoder | Hindi CER | Urdu CER | Utterance LID correct (hi / ur) |
|---|---:|---:|---|
| Unconstrained | 29.1 | 40.1 | 24 / 30, 20 / 30 |
| Rules a+b+c (default) | **26.5** | **31.2** | same |

With equal shares, the script choice rests on the utterance LID, which separates Hindi from Urdu about 75% of the time on these clips. Read the final eval's `ctc_constrained` rows for Hindi and Urdu; the unconstrained `ctc` rows and the training dev metric overstate the error of whichever language is losing the script vote.

## Kathbath speaker exclusion

Kathbath's test-known speakers are, by design, also its training speakers. Prep removes every eval and dev speaker from training (a decision on 24 October to keep the protocol speaker-disjoint). Measured on cached Kathbath train shards, the share of rows this removes:

| bn | gu | hi | kn | ml | mr | or | pa | ta | te | ur |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 49% | 69% | 75% | 85% | 94% | 53% | 100% | 53% | 88% | 57% | 80% |

The streams kept their planned hours and their ×1.5–2 weights. Projected passes over each stream's usable data during the run were:
- ml: about 310;
- or: about 20;
- ur: about 16;
- gu: about 14;
- the rest: 3–8.

**Fix:** from step 1,079, `--zero-streams` stops drawing from `kathbath/ml`, `kathbath/or`, `kathbath/ur` and `kathbath/gu`. That share goes to IndicVoices and Vaani in the same languages.
- **Not hashed:** the option changes no part of the config hash, which is why the run could resume with it.
- **Code:** a stream that accepts no row in a full pass of its pool is now marked dead instead of looping.

**Consequences for results:**
- The Kathbath test-known scores are speaker-disjoint for IndicEAR, so they are not comparable with a system trained on Kathbath's training speakers, as IndicConformer is.
- The pilot had the same exclusion: Kathbath hi 75%, ta 88% and kn 85% of rows removed. This explains much of its Kathbath gap.

## Other fixes made during this run (code-only)

- **Non-finite gradients:** under bf16 an inf or NaN gradient norm now skips the optimizer step. The skip is logged and counted, and the schedule still advances.
- **Per-language dev CER:** logged at every quick eval.
- **Eval:**
  - A `ctc_constrained` decoder, with the decode-time script rules, is scored beside `ctc`.
  - LID top-k counts distinct scripts, not languages.
  - Scoring-only Unicode folds and a script-normalised CER. The folds cover Malayalam chillus, Bengali khanda ta, Marathi eyelash ra, the Urdu heh, tatweel, vowel marks and presentation forms, and native digits.
  - These change scores relative to the pilot's results, so rescore the pilot with the same code before comparing.

## Known limitations of this run (fix in the next run)

These change the training targets or the tokenizer, so they cannot change mid-run.
- **Malayalam and Marathi spellings:** `text.normalize` deletes ZWJ before mapping the old ZWJ chillu and the eyelash ra to their atomic forms. Malayalam and Marathi targets therefore mix two spellings of the same word. The scoring folds handle this at eval.
- **Marathi tokenizer:** Marathi uses the pilot's Hindi-only deva tokenizer, at about 3.4 tokens per word against 1.4 for Hindi. Expect weaker Marathi WER.
- **Bengali and Odia loanwords:** these write English "v" as ভ/ବ (bh), which `markup.sounds_alike` does not match. Those IndicVoices and Vaani glosses stay in native script, the rows lose their code-mixed flag, and the loanword lexicon misses them.
- **IndicVoices glosses dropped as tags:** fuzzy tag matching (cutoff 0.85) removes real English glosses that look like noise tags: TV, music, door, bell, printer, baby.
- **Vaani code-mixed pools:** for te, bn, mr and kn these cannot be disjoint from the main pools at the planned hours, so some rows are drawn by both streams.
- **LID class weights:** these come from planned pool hours, not from the pinned shares, so Hindi and English count about 2× less in the LID loss. LID heads read detached states, so transcription is unaffected.
- **Phonetic targets:** no length-feasibility check. An infeasible row's phonetic loss is zeroed but its units still count in the denominator.
- **Batch length:** speed perturbation lengthens rows after batches are planned, which is why each batch now runs in 2 micro-batches. Packing batches to `budget / 1.11` would avoid it, at about 10% less audio per step. It was not applied mid-run.

## Comparisons

- **IndicConformer:** results come from `indicear.baselines` with the same scoring. It is given the true language, while IndicEAR uses its own LID. Compare per language and on sn-WER/sn-CER.
- **SraVaani-1.0:** run on the shared subset; see "Against SraVaani-1.0" above.
