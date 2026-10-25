---
license: cc-by-4.0
language:
- bn
- en
- gu
- hi
- kn
- ml
- mr
- or
- pa
- ta
- te
- ur
task_categories:
- automatic-speech-recognition
pretty_name: Vesper Indic Bench
tags:
- asr
- indic
- benchmark
- code-switching
- vesper
size_categories:
- 100K<n<1M
configs:
- config_name: clips
  default: true
  data_files:
  - split: test
    path: data/clips.jsonl
- config_name: transcripts
  data_files:
  - split: test
    path: data/transcripts.jsonl
- config_name: vesper_asr_benchmark_slice
  data_files:
  - split: test
    path: data/vesper_asr_benchmark_slice.jsonl
---

<p align="center">
  <img src="assets/vesper-banner.png" alt="Vesper" width="720">
</p>

<h1 align="center">Vesper Indic Bench</h1>

<p align="center"><b>Per-clip transcripts and scores of Indic speech recognisers on identical held-out clips</b><br>
Part of the Vesper project · 25 October 2026</p>

Vesper Indic Bench is the evaluation behind [Vesper-IndicEAR](https://huggingface.co/rahulsharmavishwakarma/vesper-indicEAR). Every system decodes exactly the same clips: the first 300 per language of each of {{SETS}} held-out evaluation sets across 12 languages ({{CLIPS}} clips, {{HOURS}} hours). The dataset publishes, for every clip, its source, its reference where the source licence allows, and each system's transcript and scores, so any number in the model card can be recomputed or re-scored with a different metric.

**No audio is included.** Clips are referenced by their source dataset, pinned revision and id, with the sha1 of the audio file.

## Systems

| System | Decoding | Language |
|---|---|---|
| Vesper-IndicEAR (automatic language) | CTC greedy, decode-time script rules | from its own language ID |
| Vesper-IndicEAR (language given) | same, with the clip's language as a hint | given |
| Vesper-IndicEAR (no script rules) | unconstrained CTC greedy | — |
| IndicConformer 600M multilingual (`ai4bharat/indic-conformer-600m-multilingual@2a77d305`) | CTC, repository code | given |
| SraVaani-1.0 (`ARTPARK-IISc/SraVaani-1.0@f5dd5358`) | TDT greedy, repository code | automatic; no Urdu |

Each baseline runs its own repository's inference code at the pinned revision. SraVaani has no rows for Urdu and IndicConformer none for English, which they do not support.

## Results

<!-- BEGIN RESULTS -->
Mean script-normalised CER over the languages every system supports (Urdu excluded, as SraVaani has no Urdu); lower is better, best in bold. 300 clips per language per set; per-clip transcripts are in Vesper Indic Bench.

| Eval family | Vesper-IndicEAR (auto) | Vesper-IndicEAR (lang. given) | IndicConformer 600M | SraVaani-1.0 | languages |
|---|---:|---:|---:|---:|---:|
| Kathbath clean, unseen speakers | 21.0 | 20.9 | 3.4 | **3.2** | 10 |
| Kathbath clean, test-known speakers | 20.5 | 20.4 | 3.3 | **3.1** | 10 |
| Kathbath noisy, unseen speakers | 24.9 | 24.6 | **4.3** | 4.4 | 10 |
| IndicVoices valid | 24.3 | 23.9 | 7.7 | **7.4** | 10 |
| Vaani test | 24.8 | 24.1 | 10.9 | **8.4** | 10 |
| IndicContext | 32.4 | 31.6 | **5.2** | 5.3 | 7 |
| MUCS 2021 (hi, bn) | 43.5 | **42.9** | 49.0 | 49.3 | 2 |

Over 65 (set, language) pairs, Vesper-IndicEAR is lower than the better of the two baselines in 1 automatically and 1 with the language given.

<!-- END RESULTS -->

## Files

| File | Rows | Contents |
|---|---:|---|
| `data/clips.jsonl` | {{CLIPS}} | one row per (set, clip) |
| `data/transcripts.jsonl` | {{TRANSCRIPTS}} | one row per (set, clip, system) |
| `data/vesper_asr_benchmark_slice.jsonl` | {{SLICE}} | Vesper-IndicEAR on the clips of Vesper's [ASR benchmark](https://huggingface.co/datasets/rahulsharmavishwakarma/vesper-asr-benchmark) it covers, keyed by that dataset's `sample_id` |
| `summary.json` | | per system, set and language: CER, WER and their script-normalised forms |

**`clips.jsonl`:** `set`, `key` (source / language / split / id), `lang`, `source`, `source_repo`, `source_revision`, `source_licence`, `split`, `seconds`, `audio_sha1`, `code_mixed`, `reference`, `reference_alternatives` (the same sentence with English words in native script, where the source provides one), `reference_withheld` (why the reference is not published).

**`transcripts.jsonl`:** `set`, `key`, `lang`, `system`, `hypothesis`, `detected_language` (Vesper-IndicEAR only), `cer`, `wer`, `cer_script_normalised` (per clip, in %).

## Metrics

Text is normalised before scoring (lowercase, no punctuation, NFKC, native digits, and the folds of Malayalam chillus, Bengali khanda ta, the Marathi eyelash ra and Urdu heh, tatweel and short vowels). **Script-normalised CER** accepts an English word written either in Latin or in native script, scoring against the closest whole reference: IndicVoices and Vaani write English words in Latin, while IndicConformer and SraVaani write them in native script, and strict CER would count that as an error. It is the metric the model card compares systems with. The scorer is `indicear.metrics` in the [code repository](https://github.com/rahulsharmavishwakarma/indicear).

## Sources and licences

| Set family | Source | Reference text |
|---|---|---|
| Kathbath clean / noisy test (known and unseen speakers) | AI4Bharat IndicSUPERB, CC BY 4.0 (Hub tag; see below) | published |
| IndicVoices valid | `ai4bharat/IndicVoices`, CC BY 4.0 | published |
| Vaani test | `ARTPARK-IISc/Vaani-transcription-part`, CC BY 4.0 | published |
| IndicContext (with code-mixed subsets) | `ai4bharat/IndicContextEval`, CC BY 4.0 | published |
| FLEURS dev | `google/fleurs`, CC BY 4.0 | published |
| LibriSpeech test-clean / test-other | `openslr/librispeech_asr`, CC BY 4.0 | published |
| VoxPopuli English test | `facebook/voxpopuli`, CC0 | published |
| MUCS 2021 Hindi-English / Bengali-English test | OpenSLR SLR104, CC BY-SA 4.0 | withheld (share-alike) |
| Svarah | `ai4bharat/Svarah` (gated; CC BY 4.0 only in its card text, no licence tag) | withheld, as in Vesper's ASR benchmark |

Kathbath's licence is stated two ways: its Hub tag says CC BY 4.0 and its card text says CC0. Its read sentences come from the IndicCorp web-text corpus. The references are republished here under CC BY 4.0 with attribution to AI4Bharat.

To get the audio, load each source at the listed revision and match `key`; the code repository's `indicear.sources` and `indicear.ingest` do this for every source. This dataset (ids, transcripts, scores and published references) is CC BY 4.0; references keep their source's licence and attribution.

## Speaker protocol

Vesper-IndicEAR never saw a speaker of these sets in training: every dev and evaluation speaker was removed from its training data. Other systems may have trained on some of these sources and speakers (Kathbath's test-known speakers are its training speakers by design), so compare across systems with that in mind.

## Limitations

- 300 clips per language and set: enough to separate systems several points apart, not to rank close ones.
- Read and prompted speech dominate; conversational speech is limited to IndicVoices, Vaani and MUCS.
- Timings are not published; they came from one Apple silicon Mac and are not comparable across systems run on CPU and GPU.

## Citation

```bibtex
@misc{vesper_indic_bench_2026,
  title  = {Vesper Indic Bench: per-clip transcripts of Indic speech recognisers on identical held-out clips},
  author = {Sharma, Rahul},
  year   = {2026},
  url    = {https://huggingface.co/datasets/rahulsharmavishwakarma/vesper-indic-bench}
}
```
