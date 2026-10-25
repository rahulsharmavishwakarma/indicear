| system | set | decoder | group | cer | wer | sn-WER | script err | cm span acc | synth span acc | false output | LID acc | LID span acc |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v4+lid | cs_dev | ctc | cs:hi+en | 14.11 | 27.69 |  | 0.42 |  | 90.0 |  | 0.0 | 96.62 |
| v4+lid | cs_dev | ctc | cs:hi+kn | 22.79 | 42.84 |  | 1.5 |  | 74.5 |  | 0.0 | 96.62 |
| v4+lid | cs_dev | ctc | cs:kn+en | 14.46 | 31.27 |  | 0.81 |  | 83.5 |  | 0.0 | 96.62 |
| v4+lid | cs_dev | ctc | cs:ta+en | 16.12 | 34.99 |  | 0.68 |  | 81.0 |  | 0.0 | 96.62 |
| v4+lid | fleurs_dev | ctc | all | 18.61 | 47.97 | 47.97 | 4.71 | 57.46 |  |  | 99.85 | 44.78 |
| v4+lid | fleurs_dev | ctc | en | 16.74 | 36.97 | 36.97 | 0.51 |  |  |  | 99.85 | 44.78 |
| v4+lid | fleurs_dev | ctc | hi | 19.39 | 39.14 | 39.14 | 5.19 | 52.0 |  |  | 99.85 | 44.78 |
| v4+lid | fleurs_dev | ctc | kn | 17.52 | 58.29 | 58.29 | 9.24 | 54.1 |  |  | 99.85 | 44.78 |
| v4+lid | fleurs_dev | ctc | ta | 21.03 | 64.47 | 64.47 | 6.06 | 64.58 |  |  | 99.85 | 44.78 |
| v4+lid | indiccontext_hi | ctc | hi | 27.04 | 44.46 | 44.29 | 10.73 | 76.64 |  |  | 99.35 | 24.91 |
| v4+lid | indiccontext_hi_code_mixed | ctc | hi | 27.71 | 45.28 | 45.02 | 11.0 | 76.53 |  |  | 99.73 | 24.91 |
| v4+lid | indicvoices_valid_hi | ctc | hi | 17.07 | 31.13 | 31.05 | 1.74 | 80.99 |  |  | 96.87 | 26.0 |
| v4+lid | indicvoices_valid_kn | ctc | kn | 25.44 | 61.39 | 61.3 | 4.19 | 74.61 |  |  | 96.22 | 28.33 |
| v4+lid | indicvoices_valid_ta | ctc | ta | 24.12 | 55.86 | 55.76 | 2.66 | 75.88 |  |  | 97.4 | 31.07 |
| v4+lid | kathbath_clean_test_known_hi | ctc | hi | 16.0 | 35.77 | 35.77 | 4.87 |  |  |  | 100.0 |  |
| v4+lid | kathbath_clean_test_known_kn | ctc | kn | 20.37 | 66.1 | 66.1 | 11.67 |  |  |  | 99.93 |  |
| v4+lid | kathbath_clean_test_known_ta | ctc | ta | 17.14 | 64.25 | 64.25 | 4.23 |  |  |  | 99.96 |  |
| v4+lid | kathbath_clean_test_unknown_hi | ctc | hi | 17.08 | 37.56 | 37.56 | 4.91 |  |  |  | 100.0 |  |
| v4+lid | kathbath_clean_test_unknown_kn | ctc | kn | 21.05 | 65.79 | 65.79 | 13.32 |  |  |  | 100.0 |  |
| v4+lid | kathbath_clean_test_unknown_ta | ctc | ta | 17.33 | 64.36 | 64.36 | 4.63 |  |  |  | 100.0 |  |
| v4+lid | kathbath_noisy_test_known_hi | ctc | hi | 18.4 | 40.06 | 40.06 | 5.58 |  |  |  | 100.0 |  |
| v4+lid | kathbath_noisy_test_known_kn | ctc | kn | 22.86 | 67.59 | 67.59 | 13.04 |  |  |  | 100.0 |  |
| v4+lid | kathbath_noisy_test_known_ta | ctc | ta | 17.45 | 63.42 | 63.42 | 4.66 |  |  |  | 100.0 |  |
| v4+lid | kathbath_noisy_test_unknown_hi | ctc | hi | 20.15 | 41.31 | 41.31 | 6.59 |  |  |  | 100.0 |  |
| v4+lid | kathbath_noisy_test_unknown_kn | ctc | kn | 24.1 | 70.13 | 70.13 | 14.01 |  |  |  | 99.86 |  |
| v4+lid | kathbath_noisy_test_unknown_ta | ctc | ta | 18.36 | 66.17 | 66.17 | 5.4 |  |  |  | 99.91 |  |
| v4+lid | librispeech_test_clean | ctc | en | 7.02 | 19.05 | 19.05 | 0.0 |  |  |  | 100.0 |  |
| v4+lid | librispeech_test_other | ctc | en | 12.8 | 29.38 | 29.38 | 0.02 |  |  |  | 99.97 |  |
| v4+lid | mucs_test_hi | ctc | hi | 45.12 | 66.8 | 66.8 | 4.68 | 61.26 |  |  | 97.87 | 39.87 |
| v4+lid | neg_dev | ctc | negatives |  |  |  |  |  |  | 0.0 |  |  |
| v4+lid | robust_babble_10 | ctc | all | 28.66 | 62.83 | 62.83 | 10.5 |  |  |  | 95.5 |  |
| v4+lid | robust_babble_10 | ctc | en | 33.22 | 55.75 | 55.75 | 8.89 |  |  |  | 95.5 |  |
| v4+lid | robust_babble_10 | ctc | hi | 22.58 | 43.61 | 43.61 | 7.14 |  |  |  | 95.5 |  |
| v4+lid | robust_babble_10 | ctc | kn | 28.94 | 74.26 | 74.26 | 16.21 |  |  |  | 95.5 |  |
| v4+lid | robust_babble_10 | ctc | ta | 29.07 | 80.11 | 80.11 | 9.88 |  |  |  | 95.5 |  |
| v4+lid | robust_noise_10 | ctc | all | 25.28 | 59.93 | 59.93 | 7.96 |  |  |  | 94.75 |  |
| v4+lid | robust_noise_10 | ctc | en | 28.37 | 51.39 | 51.39 | 5.98 |  |  |  | 94.75 |  |
| v4+lid | robust_noise_10 | ctc | hi | 20.63 | 41.32 | 41.32 | 4.21 |  |  |  | 94.75 |  |
| v4+lid | robust_noise_10 | ctc | kn | 24.88 | 69.81 | 69.81 | 13.45 |  |  |  | 94.75 |  |
| v4+lid | robust_noise_10 | ctc | ta | 26.41 | 79.65 | 79.65 | 8.54 |  |  |  | 94.75 |  |
| v4+lid | robust_noise_5 | ctc | all | 30.49 | 63.5 | 63.5 | 8.97 |  |  |  | 95.0 |  |
| v4+lid | robust_noise_5 | ctc | en | 33.05 | 55.31 | 55.31 | 7.43 |  |  |  | 95.0 |  |
| v4+lid | robust_noise_5 | ctc | hi | 25.62 | 45.22 | 45.22 | 4.8 |  |  |  | 95.0 |  |
| v4+lid | robust_noise_5 | ctc | kn | 31.37 | 74.26 | 74.26 | 15.73 |  |  |  | 95.0 |  |
| v4+lid | robust_noise_5 | ctc | ta | 30.94 | 81.61 | 81.61 | 8.21 |  |  |  | 95.0 |  |
| v4+lid | robust_reverb_0.6+pointnoise_10 | ctc | all | 48.57 | 78.77 | 78.77 | 14.06 |  |  |  | 90.25 |  |
| v4+lid | robust_reverb_0.6+pointnoise_10 | ctc | en | 51.39 | 73.26 | 73.26 | 12.97 |  |  |  | 90.25 |  |
| v4+lid | robust_reverb_0.6+pointnoise_10 | ctc | hi | 40.79 | 64.1 | 64.1 | 7.83 |  |  |  | 90.25 |  |
| v4+lid | robust_reverb_0.6+pointnoise_10 | ctc | kn | 48.59 | 85.56 | 85.56 | 22.15 |  |  |  | 90.25 |  |
| v4+lid | robust_reverb_0.6+pointnoise_10 | ctc | ta | 51.39 | 94.02 | 94.02 | 13.92 |  |  |  | 90.25 |  |
| v4+lid | robust_reverb_0.6 | ctc | all | 32.11 | 68.08 | 68.08 | 10.08 |  |  |  | 94.5 |  |
| v4+lid | robust_reverb_0.6 | ctc | en | 33.89 | 58.01 | 58.01 | 7.24 |  |  |  | 94.5 |  |
| v4+lid | robust_reverb_0.6 | ctc | hi | 24.71 | 48.43 | 48.43 | 5.26 |  |  |  | 94.5 |  |
| v4+lid | robust_reverb_0.6 | ctc | kn | 32.98 | 78.61 | 78.61 | 16.37 |  |  |  | 94.5 |  |
| v4+lid | robust_reverb_0.6 | ctc | ta | 34.61 | 89.92 | 89.92 | 11.67 |  |  |  | 94.5 |  |
| v4+lid | robust_reverb_0.9 | ctc | all | 39.5 | 73.71 | 73.71 | 11.06 |  |  |  | 93.75 |  |
| v4+lid | robust_reverb_0.9 | ctc | en | 41.77 | 67.42 | 67.42 | 8.66 |  |  |  | 93.75 |  |
| v4+lid | robust_reverb_0.9 | ctc | hi | 31.8 | 55.04 | 55.04 | 6.52 |  |  |  | 93.75 |  |
| v4+lid | robust_reverb_0.9 | ctc | kn | 40.48 | 84.07 | 84.07 | 17.82 |  |  |  | 93.75 |  |
| v4+lid | robust_reverb_0.9 | ctc | ta | 41.76 | 90.57 | 90.57 | 11.59 |  |  |  | 93.75 |  |
| v4+lid | svarah | ctc | en | 25.14 | 45.82 | 45.82 | 7.93 |  |  |  | 79.29 |  |
| v4+lid | vaani_test_en | ctc | en | 13.78 | 25.87 | 25.87 | 2.17 |  |  |  | 98.83 |  |
| v4+lid | vaani_test_hi | ctc | hi | 12.53 | 22.52 | 22.45 | 0.96 | 89.83 |  |  | 100.0 | 31.1 |
| v4+lid | vaani_test_kn | ctc | kn | 30.0 | 64.59 | 64.53 | 3.22 | 73.83 |  |  | 99.93 | 23.65 |
| v4+lid | vaani_test_ta | ctc | ta | 23.67 | 52.27 | 52.06 | 3.46 | 79.5 |  |  | 99.7 | 31.47 |
| v4+lid | voxpopuli_accented | ctc | en | 19.08 | 34.87 | 34.87 | 0.06 |  |  |  | 99.95 |  |
| v4+lid | voxpopuli_test | ctc | en | 12.68 | 28.01 | 28.01 | 0.01 |  |  |  | 100.0 |  |
