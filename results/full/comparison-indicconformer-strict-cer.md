Metric: `cer` (lower is better). Same rows for every system: the first 300 per language of each eval set. Sangam auto = its own utterance LID picks the script; hinted = the row's language is given, as IndicConformer is given it.

Sangam is lower than IndicConformer in 5 of 65 (set, language) rows automatically and in 5 with the language hint.

### Kathbath clean, unseen speakers

| lang | Sangam auto | Sangam hinted | IndicConformer |
|---|---:|---:|---:|
| hi | 19.7 | 19.5 | 2.5 |
| bn | 17.5 | 17.5 | 2.2 |
| te | 20.5 | 20.5 | 3.7 |
| ml | 29.6 | 29.5 | 3.6 |
| mr | 18.3 | 18.3 | 4.3 |
| gu | 33.6 | 33.3 | 3.2 |
| pa | 22.7 | 22.1 | 3.7 |
| or | 26.8 | 26.8 | 3.6 |
| ur | 56.5 | 35.2 | 2.6 |
| ta | 17.8 | 17.7 | 4.2 |
| kn | 18.8 | 18.6 | 2.7 |

### Kathbath clean, test-known speakers

| lang | Sangam auto | Sangam hinted | IndicConformer |
|---|---:|---:|---:|
| hi | 19.9 | 19.8 | 2.4 |
| bn | 21.9 | 21.7 | 2.2 |
| te | 19.6 | 19.6 | 3.8 |
| ml | 30.8 | 30.8 | 3.2 |
| mr | 17.5 | 17.5 | 4.8 |
| gu | 29.4 | 29.3 | 2.9 |
| pa | 20.3 | 19.8 | 3.5 |
| or | 27.0 | 27.0 | 3.6 |
| ur | 59.9 | 37.5 | 2.6 |
| ta | 16.4 | 16.4 | 3.8 |
| kn | 17.9 | 17.9 | 3.0 |

### Kathbath noisy, unseen speakers

| lang | Sangam auto | Sangam hinted | IndicConformer |
|---|---:|---:|---:|
| hi | 25.4 | 24.6 | 3.3 |
| bn | 21.9 | 21.9 | 3.0 |
| te | 25.8 | 25.7 | 4.4 |
| ml | 32.5 | 32.3 | 3.6 |
| mr | 20.2 | 19.9 | 4.2 |
| gu | 38.0 | 37.4 | 4.7 |
| pa | 26.4 | 25.9 | 5.3 |
| or | 31.3 | 31.3 | 4.4 |
| ur | 71.5 | 45.7 | 2.4 |
| ta | 18.9 | 18.8 | 4.5 |
| kn | 25.6 | 25.4 | 5.2 |

### IndicVoices valid

| lang | Sangam auto | Sangam hinted | IndicConformer |
|---|---:|---:|---:|
| hi | 18.5 | 17.1 | 12.1 |
| bn | 19.3 | 19.0 | 6.8 |
| te | 34.0 | 33.6 | 27.9 |
| ml | 29.1 | 29.0 | 17.4 |
| mr | 27.9 | 27.5 | 17.4 |
| gu | 22.2 | 21.9 | 15.6 |
| pa | 16.3 | 16.2 | 15.4 |
| or | 28.6 | 28.1 | 14.2 |
| ur | 27.1 | 20.6 | 7.0 |
| ta | 24.2 | 24.1 | 16.1 |
| kn | 23.9 | 23.7 | 20.9 |

### Vaani test

| lang | Sangam auto | Sangam hinted | IndicConformer |
|---|---:|---:|---:|
| hi | 14.8 | 14.3 | 19.0 |
| bn | 23.1 | 22.8 | 15.5 |
| te | 21.3 | 21.2 | 26.2 |
| ml | 29.3 | 29.2 | 24.7 |
| mr | 26.8 | 26.6 | 24.1 |
| gu | 17.9 | 17.8 | 20.4 |
| pa | 30.1 | 24.4 | 22.1 |
| or | 24.8 | 24.7 | 24.1 |
| ur | 74.2 | 50.2 | 27.6 |
| ta | 28.3 | 28.2 | 39.9 |
| kn | 33.4 | 33.4 | 33.0 |

### IndicContext

| lang | Sangam auto | Sangam hinted | IndicConformer |
|---|---:|---:|---:|
| hi | 28.8 | 23.9 | 15.9 |
| bn | 25.5 | 25.4 | 3.4 |
| te | 26.6 | 26.6 | 3.6 |
| ml | 38.1 | 38.1 | 7.1 |
| mr | 38.4 | 38.0 | 5.2 |
| gu | 34.7 | 34.3 | 2.4 |
| or | 34.7 | 34.7 | 11.9 |
| ur | 47.2 | 43.4 | 13.1 |

### MUCS

| lang | Sangam auto | Sangam hinted | IndicConformer |
|---|---:|---:|---:|
| hi | 52.9 | 52.0 | 51.7 |
| bn | 34.2 | 33.9 | 46.4 |
