# Pilot run (24 October 2026)


**Status:** complete. All 7,423 steps were trained: 2,664 h of audio processed (4,700 h planned at an assumed packing of 0.48) from about 467 h of streamed data, with augmentation.
- **Steps 0 → 3,788:** A100 on 24 October overnight. Interrupted when the Colab compute balance ran out; the VM disk was lost (see the incident below).
- **Steps 3,788 → 3,809:** an MPS smoke resume on an Apple-silicon workstation.
- **Steps 3,809 → 7,423:** a new A100 run on 24 October, 08:40–10:55 UTC, with the same config hash `34ad89478fc81611`.
- **Final model:** the pilot package was the warm-start source of the released model. Full results are in `results/pilot/eval-final/`.

## Setup

- **Variant:** v4 + language ID (`v4+lid`): frozen EmbeddingGemma 2 audio tower, adapters on layers 7–12, self-conditioned phonetic CTC at layers 6 and 9, two-level script output, utterance and frame LID heads.
  - Chosen after the FLEURS ablation (`results/pilot-fleurs/`).
  - Mean CER over 11 languages excluding Urdu: v2 27.67, v3 28.17, v4 27.61. v4 had fewer script errors in most languages.
- **Data:** `mixtures/pilot.json`, streamed from the Hugging Face Hub.
  - Hindi, Tamil and Kannada from IndicVoices, Vaani and Kathbath; English from LibriSpeech, People's Speech, VoxPopuli and Vaani English; mined code-mixed rows.
  - About 467 h of streamable training audio after speaker-disjoint exclusion of the dev and eval speakers.
  - IndicVoices `[gloss]` and Vaani `{gloss}` are applied only when the native words sound like the English gloss (`markup.sounds_alike`).
- **Augmentation:** on the GPU for 50% of rows (noise, babble, reverberation with and without point noise, speed, 8 kHz band). Codecs (GSM, Opus) on 15% of clips.
- **Training:** live lower layers, frame budget 67,045, 7,423 planned steps (4,700 h planned; 2,664 h processed). One A100-SXM4-80GB on Colab: GPU 80–100%, about 76 GB VRAM, about 29 steps/min.
- **Code:** the pilot ran the code as it stood on 23 October 2026, plus the markup and token fallback changes for prep and the band-limit and augmentation-retry fix for training.

## Interim results (steps 1,000–3,788)

Dev metric (CER on the pilot's speaker-disjoint dev set) during training:

| Step | 1,000 | 2,000 | 3,000 |
| ---: | ---: | ---: | ---: |
| CER | 22.9 | 19.9 | **18.5** (best checkpoint) |

**Check of the packaged best checkpoint** (step 3,000): 40 random dev utterances per language, `indicear.transcribe` on Apple MPS, CTC greedy.

| Language | CER | WER |
| --- | ---: | ---: |
| en | 13.1 | 29.0 |
| hi | 18.0 | 36.2 |
| kn | 19.7 | 48.9 |
| ta | 26.1 | 59.1 |

- **Speed:** RTF 0.0116 on MPS, about 86× real time; 1,097 s of audio in 12.7 s.

**Full evaluation of the latest checkpoint** (step 3,788, 51% of planned training), run with `indicear.transcribe` on Apple MPS and CTC greedy decoding, scored with `indicear.evaluate.score_transcripts`. There are 25 eval sets, about 118 h, with results in `results/pilot/eval-step3788/`:

| Eval set | Lang | Hours | CER | WER | WER (script-normalised) | Script error % | Code-mixed span acc. | Utterance LID acc. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| indiccontext_hi | hi | 8.26 | 28.26 | 46.62 | 46.44 | 10.38 | 73.91 | 99.21 |
| indicvoices_valid_hi | hi | 8.02 | 19.07 | 33.72 | 33.67 | 1.9 | 79.35 | 96.25 |
| indicvoices_valid_kn | kn | 6.39 | 27.03 | 64.12 | 64.03 | 4.76 | 73.44 | 96.55 |
| indicvoices_valid_ta | ta | 7.47 | 25.74 | 58.86 | 58.78 | 2.43 | 72.56 | 96.71 |
| kathbath_clean_test_known_hi | hi | 3.93 | 17.77 | 38.83 | 38.83 | 4.89 |  | 100.0 |
| kathbath_clean_test_known_kn | kn | 3.4 | 22.48 | 70.15 | 70.15 | 12.81 |  | 99.87 |
| kathbath_clean_test_known_ta | ta | 3.98 | 18.72 | 67.41 | 67.41 | 3.31 |  | 100.0 |
| kathbath_clean_test_unknown_hi | hi | 2.9 | 18.54 | 40.01 | 40.01 | 4.82 |  | 100.0 |
| kathbath_clean_test_unknown_kn | kn | 2.68 | 23.38 | 70.9 | 70.9 | 15.12 |  | 100.0 |
| kathbath_clean_test_unknown_ta | ta | 2.78 | 18.95 | 67.46 | 67.46 | 3.67 |  | 100.0 |
| kathbath_noisy_test_known_hi | hi | 3.68 | 19.45 | 41.4 | 41.4 | 4.97 |  | 99.96 |
| kathbath_noisy_test_known_kn | kn | 3.1 | 24.96 | 71.4 | 71.4 | 14.22 |  | 99.92 |
| kathbath_noisy_test_known_ta | ta | 4.21 | 18.98 | 66.91 | 66.91 | 3.7 |  | 99.95 |
| kathbath_noisy_test_unknown_hi | hi | 1.3 | 21.25 | 42.84 | 42.84 | 6.13 |  | 100.0 |
| kathbath_noisy_test_unknown_kn | kn | 3.45 | 26.49 | 74.64 | 74.64 | 15.94 |  | 99.79 |
| kathbath_noisy_test_unknown_ta | ta | 2.16 | 19.75 | 68.28 | 68.28 | 4.03 |  | 100.0 |
| librispeech_test_clean | en | 4.79 | 7.96 | 21.41 | 21.41 | 0.01 |  | 100.0 |
| librispeech_test_other | en | 5.03 | 13.77 | 31.67 | 31.67 | 0.07 |  | 99.97 |
| mucs_test_hi | hi | 4.91 | 46.93 | 69.91 | 69.91 | 4.66 | 57.11 | 97.77 |
| svarah | en | 8.84 | 27.53 | 49.79 | 49.79 | 9.32 |  | 84.21 |
| vaani_test_en | en | 3.0 | 15.07 | 28.05 | 28.05 | 2.11 |  | 99.51 |
| vaani_test_hi | hi | 3.0 | 13.97 | 24.2 | 24.15 | 1.07 | 88.98 | 99.88 |
| vaani_test_kn | kn | 3.0 | 31.11 | 66.71 | 66.65 | 4.0 | 71.88 | 99.79 |
| vaani_test_ta | ta | 3.0 | 24.71 | 54.26 | 54.09 | 3.06 | 78.01 | 99.59 |
| voxpopuli_test | en | 3.92 | 13.51 | 29.73 | 29.73 | 0.0 |  | 100.0 |

**Reading:**
- **Indic, read speech (Kathbath):** CER 18–25. Noisy versus clean costs only about 1–3 CER, so noise robustness carried over.
- **Indic, spontaneous speech (IndicVoices, Vaani):** CER 14–31.
- **Hindi is best** (Vaani 14.0, Kathbath 17.8).
- **Kannada is weakest:** script-error rate 13–16% on Kathbath, where English loanwords are written in Kannada script in the references.
- **English:** LibriSpeech 8.0 / 13.8 CER (21.4 / 31.7 WER) and VoxPopuli 13.5.
- **Indian English (Svarah):** 27.5 CER, and LID accuracy 84%, because accented English is sometimes labelled Indic.
- **MUCS Hindi–English** (technical lectures): 46.9 CER, the hardest set.
- **Utterance LID:** at least 96% on every Indic set.
- **Code-mixed span accuracy:** 57–89%.

- **Superseded:** see Final results below.

## Final results (step 7,423, A100 eval, CTC greedy)

**Dev metric** (pilot dev CER) by training step:

| Step | 1,000 | 2,000 | 3,000 | 4,000 | 5,000 | 6,000 | 7,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CER | 22.9 | 19.9 | 18.5 | 17.9 | 16.6 | 16.6 | **16.2** |

**CER (%) by eval set,** with WER for English:

| Eval set | hi | kn | ta | en |
| --- | ---: | ---: | ---: | ---: |
| Kathbath clean, known speakers | 16.0 | 20.4 | 17.1 | |
| Kathbath clean, unknown speakers | 17.1 | 21.1 | 17.3 | |
| Kathbath noisy, known speakers | 18.4 | 22.9 | 17.5 | |
| Kathbath noisy, unknown speakers | 20.2 | 24.1 | 18.4 | |
| IndicVoices valid | 17.1 | 25.4 | 24.1 | |
| FLEURS dev | 19.4 | 17.5 | 21.0 | 16.7 |
| IndicContextEval (code-mixed subset) | 27.0 (27.7) | | | |
| MUCS hi-en test | 45.1 | | | |
| LibriSpeech clean / other | | | | 7.0 / 12.8 (WER 19.1 / 29.4) |

**Other results:**
- **Utterance LID:** at least 96% on every Indic and English eval set; 100% on Kathbath.
- **Non-speech:** no false outputs.
- **Robustness subset** (pilot languages, all-language CER; FLEURS dev clean is 18.6):

  | Condition | CER | Change |
  | --- | ---: | ---: |
  | Noise 10 dB | 25.3 | +6.7 |
  | Noise 5 dB | 30.5 | +11.9 |
  | Babble 10 dB | 28.7 | +10.1 |
  | Reverb RT60 0.6 | 32.1 | +13.5 |
  | Reverb RT60 0.9 | 39.5 | +20.9 |
  | Reverb RT60 0.6 + point noise | 48.6 | +30.0 |

## Decode-time script constraints (no retraining)

Same 600-utterance subsets, Mac MPS, final pilot model (`results/pilot/eval-mac-final-constrained/summary.md`). The two rules:
- the script may change only at a word boundary, a Latin continuation after an Indic piece starts a new word, and no combining mark may follow another script;
- only Latin plus the scripts of the top-2 LID languages above 0.15 are allowed.

| Set | CER before → after | Script errors before → after |
| --- | --- | --- |
| Kathbath kn, clean/noisy | 20.0–24.1 → 17.8–21.2 | 11.5–13.9% → 3.6–5.0% |
| Kathbath hi | 16.1–20.3 → 15.4–19.3 | 4.5–6.7% → 2.9–4.5% |
| Kathbath ta | 17.0–18.6 → 16.3–17.5 | 4.3–5.5% → 1.5–1.7% |
| IndicVoices hi/kn/ta | 16.3/25.7/21.6 → 16.0/25.0/21.2 | ≤3.3% → ≤1.0% |
| Vaani hi/kn/ta | 10.8/31.4/22.3 → 10.5/30.8/21.5 | ≤3.9% → ≤1.6% |
| MUCS hi-en | 42.6 → 42.1 | 3.6% → 2.6% |

- CER and WER improve on all 21 Indic sets.
- Code-mixed span accuracy holds or improves on every set except one span on IndicVoices hi (81.06 → 80.98).
- The script-error gate (< 5%) now passes for every pilot language.
- The constraints do not close the gap to IndicConformer on Kathbath or IndicVoices. That gap is a training and data matter.

## Gate check for full training

| Gate | Target | Result | Verdict |
| --- | --- | --- | --- |
| Script errors | < 5% per language | hi 1.7–6.6%, ta 2.7–5.4%, **kn 4.2–14.0%** (Kathbath kn 11.7–14.0%) | Partial: Kannada fails, mostly English loanwords written in Kannada in the references |
| Code-mixed span accuracy | ≥ 90% | Real rows (IndicVoices, IndicContextEval) 74.6–81.0%. Synthetic: hi+en 90.0, kn+en 83.5, ta+en 81.0, hi+kn 74.5 | **Fail** |
| Reverberation | No collapse; ≤ +10 CER heavy reverb | +20.9 at RT60 0.9 and +30.0 with point noise (2.6× clean) | **Fail**: improved over the frozen-encoder probe, but still the weak spot |
| LID | ≥ 97% | 96–100% on clean sets, 90–95% under heavy corruption | Near pass: some clean sets at 96%, below the gate |
| Competitive with IndicConformer | Same eval sets | Run on an Apple M5 Pro (600 rows per set, `results/pilot/baseline-indicconformer/`), scored with strict CER: IndicEAR looked ahead on Vaani hi/ta and MUCS. **Corrected 25 October:** IndicConformer writes English words in native script while our IndicVoices/Vaani references write them in Latin, so strict CER penalised it. Scored with the script-normalised CER (either spelling accepted), IndicConformer is far ahead on every set except MUCS; see `docs/full-run.md` | **Fail** |

**Reading:**
- Accuracy keeps improving with data and steps, and noise robustness holds (Kathbath noisy is only about +2 CER).
- Three things need work before or during full training:
  1. Kannada script conventions in references and scoring.
  2. Intra-sentential code-switch spans: more real code-mixed data, and span-aware training.
  3. Reverberation: more far-field and room-impulse data, stronger reverb augmentation, or a dereverberation front end.
- The IndicConformer comparison, rescored fairly with the full-run model, is in `docs/full-run.md`: not competitive yet.

## Incident: compute balance exhausted

- **What happened:** between about 23:10 and 00:50 UTC the Colab balance went from about 174 units to 0.00. The A100 was replaced by a CPU runtime with a new host key, and the VM disk was lost.
- **Cause unknown:** the night's A100 use was about 25 units, so the rest expired or was consumed elsewhere.
- **What survived:** the best checkpoint (step 3,000) and the latest (step 3,788) were already on a workstation through `sync_runs.sh`.
- **Recovery:** the tokenizer (`<ROOT>/tok`) was not synced. It was rebuilt on a workstation from the same code and the same data.
  - Every source's transcript count matched the VM log.
  - `indicear.export` strict-loaded the weights with it.
  - Decoding real dev audio gives CER about 19, matching the run's 18.5.
- **Fix:** `sync_runs.sh` now pulls `<ROOT>/tok` once per run into `<dest>/tok`.
