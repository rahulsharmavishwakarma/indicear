# Data sources and licences

Checked on 24 October 2026 against the Hugging Face Hub API (`dataset_info`, `list_repo_tree`, parquet footers read with HTTP range requests, and the dataset viewer) and the publishers' pages. The machine-readable copy is [`data-licenses.json`](data-licenses.json). It is generated from `indicear/sources.py` with `uv run python -m indicear.sources --out docs/data-licenses.json`, and a test fails when the two drift apart. Each prep run also writes `manifests/sources.json` for the sources it actually used, and lists them in `manifests/data_report.json`.

**Policy** (PLAN.md licensing checklist):

| Status | Meaning | Sources |
| --- | --- | --- |
| `allowed` | CC0 or CC BY. Used for training, model selection and evaluation. | IndicVoices, Vaani, Vaani Benchmark, Kathbath (+ tests), Shrutilipi, LibriSpeech, People's Speech (CC BY configs only), VoxPopuli, Svarah, Lahaja, FLEURS |
| `review` | Share-alike or unconfirmed. Evaluation only, unless the prep is run with `--allow-review <name>`, which the report records. | MUCS (CC BY-SA 4.0), SPRING-INX (licence unconfirmed), Common Voice (new terms) |
| `excluded` | Non-commercial, academic-only, or terms that could not be read. Never loaded, not even for evaluation. | IndicTTS; by policy, CS-FLEURS (CC BY-NC 4.0), HiACC (CC BY-NC), Gram Vaani (academic only) and the MSR sets |
| `blocked` | No plain download. | Common Voice (Mozilla Data Collective) |

"Verified" means a row or parquet footer was read. "Partial" means the repo is gated and the tokens available here had no access: the revision, licence tag, file list and card schema are verified, but no row was read. IndicVoices, the main Vaani repo and Svarah were verified with a read-only account token. That token was loaded in-process only and never printed or written; the loaders read `HF_TOKEN` from the environment. Every loader checks its required columns and stops with a message when they are missing.

## Training sources

| Source | Repo @ revision | Licence | Gated | Verification | Languages and hours |
| --- | --- | --- | --- | --- | --- |
| IndicVoices | `ai4bharat/IndicVoices@c96f9088` | CC BY 4.0 | yes (auto) | **verified** | **Measured** train hours from the duration column: hi 819.4 (2,354 speakers), ta 803.9 (2,407), kn 616.2 (2,131). Other languages, estimated from parquet size at the measured 61 MB/h: pa ~820, or ~710, ml ~675, bn ~665, mr ~615, te ~550, ur ~550, gu ~450. 11.2K transcribed h in total (card). No English. |
| Vaani transcribed part | `ARTPARK-IISc/Vaani-transcription-part@418f8f16` | CC BY 4.0 | yes (form) | **verified** (access granted 24 October; the pilot streamed it, and a shard has the card's columns) | Card table, transcribed hours: hi 1003.8, te 184.5, bn 162.4, kn 151.1, mr 68.1, or 51.2, ml 40.5, ta 24.4, en 22.5, pa 20.4, gu 19.3, ur 2.3 (train + validation + test). |
| Vaani full release (fallback) | `ARTPARK-IISc/Vaani@9e6ed4f9` | CC BY 4.0 | yes (form) | **verified** | 31K h, of which only ~6% is transcribed. Transcribed rows are spread over every row group, so each transcribed hour costs ~18 h of download. Registered as `vaani_main`; not in the mixtures. |
| Kathbath | `ai4bharat/Kathbath@5b9e9284` | CC BY 4.0 tag; CC0 in card text | yes (auto) | partial | 1,684 h over 12 languages (card). Estimate at 103 MB/h: ta ~208, mr ~174, kn ~169, te ~166, hi ~161, ml ~160, pa ~146, gu ~132, bn ~103, or ~91, ur ~88. |
| Shrutilipi | `ai4bharat/Shrutilipi@f4beb2f2` | CC BY 4.0 | yes (auto) | partial: the listing works, a parquet read returns 403 (the account is not authorized); not in `full_warm` | 6,400 h in total. Estimate at 42 MB/h: hi ~2,260, mr ~920, ta ~755, kn ~515, ml ~460, gu ~420, or ~390, te ~140, pa ~75, bn ~45. No Urdu config. |
| SPRING-INX | `SPRINGLab/SPRING_INX_<Lang>_R1/R2` (revisions in the JSON) | unconfirmed | no | partial | ~2,000 h over 10 languages (paper). Hub repos for bn, gu, kn, ml, mr, or, pa, ta; no Hindi audio repo. |
| MUCS 2021 sub-task 2 | OpenSLR SLR104 tarballs, pinned by size and ETag | CC BY-SA 4.0 | no | partial | Hi-En 89.86 h train + 5.18 h test; Bn-En 46.11 h + 7.02 h; blind test set. |
| LibriSpeech | `openslr/librispeech_asr@71cacbfb` | CC BY 4.0 | no | verified | 960 h en |
| People's Speech | `MLCommons/peoples_speech@f10597c5` (`clean/train` only) | CC BY 2.0–4.0 | no | verified | 1.5M clips in `clean/train`; hours not stated on the card |
| VoxPopuli | `facebook/voxpopuli@42f01879` (`en`) | CC0 | no | verified | 543 h en (card) |

## Evaluation-only sources

| Source | Repo @ revision | Licence | Gated | Verification | Size |
| --- | --- | --- | --- | --- | --- |
| Svarah (Indian English) | `ai4bharat/Svarah@ebbf7777` | CC BY 4.0 (card text, no tag) | yes (local login had access) | verified | 9.6 h, 117 speakers |
| Lahaja (Hindi accents) | `ai4bharat/Lahaja@d4ffd2ec` | MIT tag vs CC BY 4.0 in the card text | yes | partial | 12.5 h, 132 speakers |
| Vaani Benchmark v1.0 (Hindi) | `ARTPARK-IISc/Vaani-Benchmark-V1.0@d742f1cc` | CC BY 4.0 | yes | partial | 5,050 segments, 1,103 speakers, ~10.9 h, three references each |
| Kathbath test-known / test-unknown, clean and noisy | IndicSUPERB object store (`clean|noisy/testkn_audio.tar`, `testunk_audio.tar`, `transcripts_n2w.tar`) | as Kathbath | no | partial | clean 3.59 + 2.21 GB, noisy 3.46 + 1.42 GB, transcripts 209 MB; all 12 languages in each tar |
| IndicVoices valid | as above, split `valid` | CC BY 4.0 | yes | verified | hi 8.9 h, kn 7.0 h, ta 8.3 h (measured). 50–71% of rows code-mixed. Its speakers are removed from training: 512/514 hi, 218/220 kn and 546/553 ta valid speakers also occur in train. That costs 24%, 12% and 25% of those train splits. |
| IndicContextEval | `ai4bharat/IndicContextEval@d9c03758` | CC BY 4.0 | no | verified | 55.9 h; bn, gu, hi, ml, mr, or, te, ur (no kn, ta or en). Entity-heavy, with a per-row `code_mixing` flag. English is written `वारंटी (Warranty)`. |
| Indic DiarBench | `sarvamai/indic-diarbench@92877bad` | CC BY 4.0 | no | verified (footer) | ~108 h of multi-speaker long-form recordings. Registered but **not loaded**: it needs segmentation first, and code-mixing is not flagged. |
| Vaani test | as above, split `test` | CC BY 4.0 | yes | partial | capped at 3 h per language in the mixtures |
| MUCS test and blind | SLR104 | CC BY-SA 4.0 | no | partial | 5.18 h Hi-En test, 7.02 h Bn-En test |
| LibriSpeech test-clean / test-other, VoxPopuli test and `en_accented` | as above | CC BY 4.0 / CC0 | no | verified | `en_accented` covers 15 European L2 accents, none Indian |

Missing on the Hub: the IndicVoices **test** split (only `train` and `valid` are published) and the Kathbath tests (only on the object store).

## Attribution

- **IndicVoices:** Javed et al., "IndicVoices: Towards building an Inclusive Multilingual Speech Dataset for Indian Languages", ACL Findings 2024. AI4Bharat, IIT Madras. CC BY 4.0.
- **Vaani and Vaani Benchmark:** Pulikodan et al., "VAANI: Capturing the language landscape for an inclusive digital India", arXiv 2603.28714. IISc Bengaluru and ARTPARK, funded by Google. CC BY 4.0.
- **Kathbath:** Javed et al., "IndicSUPERB", arXiv 2208.11761. AI4Bharat. The read text comes from IndicCorp.
- **Shrutilipi:** Bhogale et al., ICASSP 2023. AI4Bharat. All India Radio news. CC BY 4.0.
- **SPRING-INX:** SPRING Lab, IIT Madras, arXiv 2310.14654.
- **MUCS 2021:** Diwan et al., Interspeech 2021. OpenSLR SLR104; the audio comes from the Spoken Tutorial project, IIT Bombay. CC BY-SA 4.0.
- **LibriSpeech:** Panayotov et al., ICASSP 2015. LibriVox. CC BY 4.0.
- **People's Speech:** Galvez et al., arXiv 2111.09344. MLCommons. CC BY.
- **VoxPopuli:** Wang et al., ACL 2021. Meta AI; European Parliament recordings. CC0.
- **Svarah:** Javed et al., Interspeech 2023. AI4Bharat. CC BY 4.0.
- **Lahaja:** Javed et al., arXiv 2408.11440. AI4Bharat.
- **FLEURS:** Conneau et al., SLT 2022. Google. CC BY 4.0.

## Gating terms

- **AI4Bharat** (IndicVoices, Kathbath, Shrutilipi, Svarah, Lahaja): the Hub's automatic gate, which shares contact details. No extra terms on the cards.
- **ARTPARK-IISc** (Vaani, Vaani Benchmark): a form asking for first name, last name, country, company and purpose. The purpose options are Research, Academic, Commercial and Education.
- **Loaders** read `HF_TOKEN` from the environment, else huggingface_hub's saved login (root-only on the VM), and never print or store it. The account behind the token must have accepted each gate. Without access, a source is reported as unavailable and the rest of the mixture still loads.

## Transcript conventions found

- **Vaani** uses curly braces for English. The convention comes from public Vaani transcripts (the portal's `v1a` JSON, mirrored publicly) and the notation table on the Vaani Benchmark card.
  - `{english}` follows the native-script rendering of an English word: `सैकल{cycle}`, `ब्लैक {black} कलर {color}`.
  - A gloss can cover several words: `ಗವರ್ನಮೆಂಟ್ ಸ್ಕೂಳ್ಳಲೇ {government school}`.
  - A gloss can be glued to the next suffix: `ಕಾರ್ {car}ಆಲ್ಲಿ`.
  - `{native}` and `(native)` give the standard spelling of a dialect form: `पे {पर}`, `बोहोत {बहुत}`.
  - Noisy speech is wrapped in `<noise> … </noise>` and `<static_noise> … </static_noise>`. Other tags: `<PAUSE>`, `[breathing]`, `[lip smacks]`, `[unintelligible]`, and `--` for a cut-off word.
- **Vaani rewrite** (`indicear/markup.py`):
  - A gloss replaces as many preceding native-script tokens as it has words, and the row is flagged `code_mixed`.
  - Standard spellings are dropped, so the reference keeps what was said.
  - Tags are dropped and the speech inside them is kept.
  - Rows with `[unintelligible]` are dropped.
  - **Limitations:** a native token that carries a case suffix loses it (`ಮೈಸೂರಲ್ಲಿ{mysore}` becomes `mysore`). Annotators also gloss some Indian place names (`{mysore}`), which then come out in Latin.
- **IndicVoices** (verified on gated rows):
  - **Fields:** `text` and `normalized` are textbook spellings with English in native script. `verbatim` is the as-spoken spelling.
  - **Glosses:** `unsanitized_normalized` adds a **Latin gloss in square brackets** after English words: `ಆರ್ಡರ್ [order] ಶಿಪ್‌ [ship]`, `हाइवे [Highway]`. Rows with a gloss in the valid split: hi 50%, kn 62%, ta 71%. The most common glosses are sir, madam, hello, okay, time, phone, online and school.
  - **Tags:** the 89 Table-13 names, written as `<tag>` or `[tag]`, sometimes capitalised (`<Persistent-noise-start>`) or misspelt (`[breathinhg]`).
  - **What IndicEAR does:** it reads `unsanitized_normalized`. Tags (and close matches to a tag name) are dropped, glosses are rewritten to Latin, and the row is flagged `code_mixed`. Rows with `<unintelligible>` are dropped (2–4% of rows).
  - So the PLAN.md `[gloss]` convention is real; it lives in the unsanitized field.
- **IndicContextEval:** `वारंटी (Warranty)`, where a Latin parenthesis glosses the preceding word.
- **MUCS** mostly writes English in Latin, but some English words are in Devanagari (`लिबर ऑफिस`, `स्लाइड इन्सर्ट`). Every row with both scripts is flagged `code_mixed`.
- **SPRING-INX** conversational transcripts write English in Latin (`ಹೋ ಮತ್ತೆ ಹೇಗಾಗ್ತದೆ food ಯೆಲ್ಲ`).

## Augmentation and text resources

| Resource | Source | Licence | Status | Use |
| --- | --- | --- | --- | --- |
| MUSAN noise | OpenSLR 17 | CC BY 4.0 | allowed | optional noise bank (`--noise-dir`) |
| Real RIRs: AIR | OpenSLR SLR28 package (Apache 2.0), from the Aachen Impulse Response database | "Not stated" on its OpenSLR mirror (SLR20) | review | `RIR_SETS=air,...` only after the terms are confirmed |
| Real RIRs: REVERB 2014 | SLR28, from the REVERB challenge | terms not read | review | as above |
| Real RIRs: RWCP | SLR28, from the RWCP sound scene database | "Research and development use only" (OpenSLR SLR13) | excluded | never selected by default (`RIR_SETS` must name it) |
| Loanword lexicon | mined from IndicVoices `unsanitized_normalized` glosses (`ai4bharat/IndicVoices@c96f9088`), `indicear/resources/loanwords/lexicon.json` | derived from CC BY 4.0 data: keep the IndicVoices attribution | allowed | rewrites English loanwords in Kathbath transcripts (docs/loanwords.md) |
| English word list | SCOWL 2020.12.07 (wordlist.aspell.net), sizes ≤ 50, `indicear/resources/english_words.tsv` | permissive MIT-like notice (`SCOWL-LICENSE.txt`, shipped with the list); wordfreq avoided (CC BY-SA 4.0) | allowed | loanword normaliser candidates (docs/loanwords.md) |

## Language labels for LID

| Source | `lang` (utterance) | `word_langs` | Notes |
| --- | --- | --- | --- |
| IndicVoices | reliable: per-language config and a `lang` column | English from glosses; English loanwords without a gloss stay in the matrix language | Urdu is in Perso-Arabic script |
| Vaani | the config's language label | English from `{gloss}` | Configs are language labels inside regional speech: "Hindi" spans many districts, and dialect configs (Bhojpuri, Awadhi, …) are not used |
| Kathbath, Shrutilipi, Lahaja | reliable: read text in the target language | monolingual | Kathbath and Shrutilipi Urdu are in Perso-Arabic script |
| MUCS | `hi` or `bn` (matrix language), `code_mixed` | Latin words are `en`; English written in Devanagari is tagged `hi` | spoken technical tutorials |
| SPRING-INX | per-language repo | English in Latin | licence under review |
| LibriSpeech, VoxPopuli, Svarah | `en`, reliable | `en` | Svarah is Indian-accented English |
| People's Speech | `en`, less reliable | `en` | The card notes some non-English audio in the archive.org sources |

**Hindi and Urdu.** Spoken Hindustani is largely shared, so labels follow script: Devanagari rows are `hi` and Perso-Arabic rows are `ur`. An acoustic LID head cannot separate them reliably. Expect hi/ur confusion and treat the two as one acoustic class with a script preference (architecture.md, known risk). Vaani's Urdu config holds only 2.3 h.

## Scoring

AI4Bharat sets (IndicVoices, Kathbath, Lahaja) often write English in native script, while IndicEAR outputs Latin English. Rows whose glosses were rewritten carry `refs` = [the native-script transcript]. Report strict WER and a script-normalised WER (`metrics.script_normalised_wer`, over `text` and `refs`) side by side. In the `full_warm` mixture the loanword normaliser (docs/loanwords.md) rewrites English loanwords in Kathbath transcripts to Latin: training and dev rows take the rewrite as `text` and keep the original in `refs`; Kathbath test rows keep the published text and carry the rewrite in `refs`, so strict WER stays comparable with other systems.

## Open questions

1. **SPRING-INX licence.** The Hub repos carry no licence and `asr.iitm.ac.in/dataset` returns 404. Only the IndiaAI AIKosh listing (Odia, Malayalam) states CC BY 4.0. Confirm with IIT Madras before training.
2. **MUCS share-alike.** CC BY-SA 4.0 needs counsel review of what share-alike means for trained weights before training (PLAN.md checklist 3). Once cleared, MUCS is for internal training only and stays out of anything redistributed.
3. **Kathbath tag conflict.** The Hub tag is CC BY 4.0; the card and the IndicSUPERB README say CC0. The read sentences come from the IndicCorp web crawl, which AI4Bharat does not own. Lahaja has the same kind of conflict (MIT tag vs CC BY 4.0 in the text).
4. **Common Voice.** Since October 2025 it is distributed only through the Mozilla Data Collective, with credentials and new terms. It is blocked until those terms are reviewed, so the plan's 600 h of Common Voice English are missing from the full mixture.
5. **IndicTTS.** The mirrors defer to IIT Madras's "License For Use of Indic TTS" PDF, which timed out from here, and only some mirrors carry a CC BY 4.0 tag. Excluded.
6. **People's Speech attribution.** It is not settled how CC BY 2.0–4.0 attribution applies to the individual archive.org works.
7. **IndicVoices** (measured, not open):
   - The `valid` speakers overlap train almost completely, so using valid as an eval set removes 12–25% of the train hours.
   - Many public models train on IndicVoices train. A model that did may have seen these valid speakers, which affects any comparison with such models on IndicVoices valid.
8. **Shrutilipi access.** The project's account is not on the gated repo's authorized list (403 on a parquet read, 24 October 2026). Vaani-transcription-part, the earlier access question, is resolved.
9. **Unverified layouts.**
   - The split directory names inside the Kathbath test tars, and the noisy transcripts' size and ETag (the loader matches files by name).
   - The MUCS blind-test layout.
   - The Vaani file-name speaker token, which is used as a speaker group because Vaani-transcription-part has no speaker column.
