Script-normalised CER (sn-CER, lower is better: an English word counts as correct in Latin or native script). Every system decodes exactly the same rows: the first 300 per language of each eval set. Speakers in these sets were excluded from training. Means over the languages all four columns share (Urdu excluded: SraVaani does not support it).

| Eval family | IndicEAR (automatic language) | IndicEAR (language given) | IndicConformer 600M (language given) | SraVaani-1.0 (automatic language) |
|---|---:|---:|---:|---:|
| Kathbath clean, unseen speakers | 21.0 | 20.9 | 3.4 | 3.2 |
| Kathbath noisy, unseen speakers | 24.9 | 24.6 | 4.3 | 4.4 |
| IndicVoices valid | 24.3 | 23.9 | 7.7 | 7.4 |
| Vaani test | 24.8 | 24.1 | 10.9 | 8.4 |
| IndicContext | 32.4 | 31.6 | 5.2 | 5.3 |
| MUCS (hi, bn) | 43.5 | 42.9 | 49.0 | 49.3 |

Pinned baselines: `ai4bharat/indic-conformer-600m-multilingual@2a77d305` (CTC) and `ARTPARK-IISc/SraVaani-1.0@f5dd5358` (TDT). Per-language tables: `results/full/comparison-baselines.md` in the code repository.

IndicEAR alone, constrained decoder (full eval sets):

- **English:** LibriSpeech test-clean / test-other 7.2 / 13.0 CER; VoxPopuli 12.8; Svarah (Indian-accented English) 25.6.
- **FLEURS dev:** 24.8 CER over all 12 languages; utterance language ID 97.8%.
- **Robustness:** noise at 10 dB 30.2, babble at 10 dB 33.0, reverberation RT60 0.9 s 46.2.
