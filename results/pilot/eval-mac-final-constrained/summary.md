# Pilot v4+lid: decode-time script constraints on the Mac (MPS, CTC greedy)

Model `pilot-v4lid-final`, the first 600 utterances per set (`sangam.evaluate.subset_per_lang`), scored with `sangam.evaluate.score_transcripts`.
Before: `results/pilot/eval-mac-final/` (constraints off). A rerun of the unconstrained decode in the same pass reproduced every one of those numbers exactly; for `indiccontext_hi_code_mixed` and `librispeech_test_clean`, which have no file there, the before values come from that rerun.
After: `sangam.decoding` defaults: script rules on (with the Latin split), LID script set on, top-k 2, threshold 0.15, no frame-LID share. Command:

```
uv run --locked python -m sangam.decoding eval --model ~/workspace/sangam-models/pilot-v4lid-final \
  --manifests ~/workspace/sangam-pilot-root/manifests --per-lang 600 --out ../results/pilot/eval-mac-final-constrained \
  --sets 'kathbath_*' 'indicvoices_valid_*' vaani_test_hi vaani_test_kn vaani_test_ta 'indiccontext_hi*' mucs_test_hi librispeech_test_clean
```

IC: IndicConformer CTC on the same subsets (`results/pilot/baseline-indicconformer/`). `changed`: utterances whose transcript differs.

| Set | CER before | CER after | WER before | WER after | sn-WER before | sn-WER after | Script err. before | Script err. after | CM span acc. before | CM span acc. after | Changed | IC CER | IC WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| indiccontext_hi | 28.67 | 28.21 | 43.03 | 41.53 | 43.00 | 41.51 | 13.22 | 12.17 | 80.13 | 81.11 | 264 | 11.61 | 18.49 |
| indiccontext_hi_code_mixed | 30.55 | 30.09 | 44.77 | 43.28 | 44.69 | 43.19 | 14.82 | 13.93 | 81.52 | 82.03 | 348 | 14.86 | 20.64 |
| indicvoices_valid_hi | 16.29 | 15.98 | 28.49 | 27.95 | 28.40 | 27.85 | 1.67 | 0.63 | 81.06 | 80.98 | 121 | 12.59 | 18.60 |
| indicvoices_valid_kn | 25.66 | 25.02 | 61.97 | 60.34 | 61.66 | 60.02 | 3.32 | 0.97 | 77.13 | 78.20 | 212 | 20.37 | 49.53 |
| indicvoices_valid_ta | 21.57 | 21.21 | 53.83 | 52.82 | 53.70 | 52.68 | 2.50 | 0.97 | 77.87 | 78.38 | 177 | 15.78 | 40.19 |
| kathbath_clean_test_known_hi | 16.10 | 15.37 | 36.28 | 34.41 | 36.28 | 34.41 | 5.09 | 3.54 |  |  | 146 | 2.40 | 8.71 |
| kathbath_clean_test_known_kn | 20.01 | 17.77 | 65.76 | 59.13 | 65.76 | 59.13 | 11.45 | 3.64 |  |  | 340 | 3.27 | 18.60 |
| kathbath_clean_test_known_ta | 17.04 | 16.26 | 64.08 | 61.15 | 64.08 | 61.15 | 4.28 | 1.64 |  |  | 154 | 3.92 | 23.32 |
| kathbath_clean_test_unknown_hi | 16.56 | 15.82 | 36.98 | 35.13 | 36.98 | 35.13 | 4.48 | 2.85 |  |  | 136 | 2.75 | 9.29 |
| kathbath_clean_test_unknown_kn | 21.02 | 18.21 | 66.13 | 57.67 | 66.13 | 57.67 | 13.91 | 5.03 |  |  | 372 | 2.94 | 16.74 |
| kathbath_clean_test_unknown_ta | 17.82 | 16.83 | 66.65 | 62.83 | 66.65 | 62.83 | 5.19 | 1.73 |  |  | 180 | 4.09 | 23.86 |
| kathbath_noisy_test_known_hi | 18.19 | 17.11 | 39.44 | 37.33 | 39.44 | 37.33 | 5.39 | 3.57 |  |  | 165 | 3.35 | 10.56 |
| kathbath_noisy_test_known_kn | 22.63 | 20.32 | 67.63 | 60.61 | 67.63 | 60.61 | 12.51 | 4.26 |  |  | 353 | 3.39 | 17.44 |
| kathbath_noisy_test_known_ta | 17.26 | 16.48 | 62.82 | 60.08 | 62.82 | 60.08 | 4.31 | 1.45 |  |  | 160 | 4.59 | 24.38 |
| kathbath_noisy_test_unknown_hi | 20.31 | 19.27 | 41.16 | 39.03 | 41.16 | 39.03 | 6.72 | 4.46 |  |  | 191 | 3.45 | 10.54 |
| kathbath_noisy_test_unknown_kn | 24.06 | 21.18 | 69.21 | 61.02 | 69.21 | 61.02 | 13.53 | 4.89 |  |  | 372 | 5.25 | 22.05 |
| kathbath_noisy_test_unknown_ta | 18.60 | 17.53 | 66.39 | 62.57 | 66.39 | 62.57 | 5.51 | 1.71 |  |  | 187 | 4.44 | 24.35 |
| librispeech_test_clean | 7.23 | 7.23 | 19.26 | 19.26 | 19.26 | 19.26 | 0.00 | 0.00 |  |  | 0 |  |  |
| mucs_test_hi | 42.63 | 42.08 | 64.50 | 62.80 | 64.50 | 62.80 | 3.62 | 2.60 | 61.02 | 62.06 | 240 | 50.07 | 54.81 |
| vaani_test_hi | 10.77 | 10.51 | 19.58 | 19.21 | 19.49 | 19.12 | 1.15 | 0.57 | 91.74 | 91.74 | 66 | 17.77 | 23.07 |
| vaani_test_kn | 31.37 | 30.77 | 64.51 | 62.99 | 64.46 | 62.93 | 3.89 | 1.62 | 70.77 | 71.34 | 218 | 32.88 | 62.04 |
| vaani_test_ta | 22.27 | 21.53 | 48.87 | 47.57 | 48.63 | 47.34 | 3.30 | 0.48 | 80.28 | 81.39 | 157 | 39.78 | 70.26 |

Strict rule (`--latin-split off`, Indic to Latin mid-word blocked) on the code-mixed sets, same subsets:

| Set | CER | WER | Script err. | CM span acc. (en spans) |
| --- | ---: | ---: | ---: | ---: |
| indiccontext_hi | 27.82 | 40.00 | 10.93 | 78.83 (68.33) |
| indicvoices_valid_hi | 16.11 | 27.78 | 0.57 | 80.19 (68.53) |
| mucs_test_hi | 42.28 | 61.12 | 2.52 | 60.47 (52.15) |
