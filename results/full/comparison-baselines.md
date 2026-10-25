Metric: `cer_script_normalised` (lower is better). Same rows for every system: the first 300 per language of each eval set. Sangam auto = its own utterance LID picks the script; hinted = the row's language is given. IndicConformer is given the language; SraVaani picks it itself (no Urdu). – = not supported.

Sangam is lower than IndicConformer in 1 of 65 (set, language) rows automatically and in 1 with the language hint.
Sangam is lower than SraVaani in 1 of 59 (set, language) rows automatically and in 1 with the language hint.

### Kathbath clean, unseen speakers

| lang | Sangam auto | Sangam hinted | IndicConformer | SraVaani |
|---|---:|---:|---:|---:|
| hi | 16.9 | 16.6 | 2.5 | 2.2 |
| bn | 16.2 | 16.2 | 2.2 | 1.9 |
| te | 18.5 | 18.5 | 3.7 | 3.5 |
| ml | 28.4 | 28.4 | 3.6 | 3.2 |
| mr | 17.7 | 17.7 | 4.3 | 4.0 |
| gu | 32.2 | 31.9 | 3.2 | 3.2 |
| pa | 19.6 | 18.9 | 3.7 | 3.7 |
| or | 25.3 | 25.3 | 3.6 | 3.5 |
| ur | 53.9 | 32.5 | 2.6 | – |
| ta | 17.3 | 17.2 | 4.2 | 3.8 |
| kn | 18.0 | 17.8 | 2.7 | 2.9 |

### Kathbath clean, test-known speakers

| lang | Sangam auto | Sangam hinted | IndicConformer | SraVaani |
|---|---:|---:|---:|---:|
| hi | 16.9 | 16.7 | 2.4 | 2.2 |
| bn | 19.9 | 19.7 | 2.2 | 2.0 |
| te | 17.6 | 17.6 | 3.8 | 3.6 |
| ml | 29.4 | 29.4 | 3.2 | 2.8 |
| mr | 17.0 | 17.0 | 4.8 | 4.5 |
| gu | 28.0 | 27.9 | 2.9 | 2.8 |
| pa | 17.8 | 17.3 | 3.5 | 3.5 |
| or | 25.2 | 25.2 | 3.6 | 3.6 |
| ur | 57.0 | 34.4 | 2.6 | – |
| ta | 16.2 | 16.1 | 3.8 | 3.3 |
| kn | 17.1 | 17.1 | 3.0 | 2.9 |

### Kathbath noisy, unseen speakers

| lang | Sangam auto | Sangam hinted | IndicConformer | SraVaani |
|---|---:|---:|---:|---:|
| hi | 21.4 | 20.6 | 3.3 | 3.2 |
| bn | 20.1 | 20.0 | 3.0 | 3.1 |
| te | 23.8 | 23.8 | 4.4 | 4.6 |
| ml | 31.0 | 30.8 | 3.6 | 3.4 |
| mr | 19.7 | 19.4 | 4.2 | 4.3 |
| gu | 36.1 | 35.5 | 4.7 | 5.3 |
| pa | 24.4 | 23.8 | 5.3 | 5.9 |
| or | 30.0 | 30.0 | 4.4 | 4.5 |
| ur | 68.0 | 41.8 | 2.4 | – |
| ta | 18.6 | 18.4 | 4.5 | 4.1 |
| kn | 24.3 | 24.1 | 5.2 | 5.2 |

### IndicVoices valid

| lang | Sangam auto | Sangam hinted | IndicConformer | SraVaani |
|---|---:|---:|---:|---:|
| hi | 18.4 | 17.0 | 4.9 | 4.7 |
| bn | 19.2 | 18.9 | 3.4 | 3.5 |
| te | 33.8 | 33.5 | 10.7 | 11.3 |
| ml | 29.1 | 29.0 | 10.2 | 9.6 |
| mr | 27.7 | 27.3 | 6.9 | 6.2 |
| gu | 22.1 | 21.8 | 6.4 | 7.3 |
| pa | 16.3 | 16.2 | 5.5 | 5.3 |
| or | 28.5 | 28.1 | 9.7 | 9.0 |
| ur | 27.0 | 20.4 | 2.7 | – |
| ta | 24.1 | 24.1 | 9.1 | 8.3 |
| kn | 23.8 | 23.7 | 10.4 | 9.1 |

### Vaani test

| lang | Sangam auto | Sangam hinted | IndicConformer | SraVaani |
|---|---:|---:|---:|---:|
| hi | 14.7 | 14.2 | 6.4 | 3.8 |
| bn | 22.9 | 22.5 | 8.7 | 4.5 |
| te | 21.2 | 21.2 | 7.3 | 6.6 |
| ml | 28.7 | 28.7 | 9.9 | 9.0 |
| mr | 26.7 | 26.5 | 11.1 | 9.4 |
| gu | 17.8 | 17.7 | 5.1 | 3.5 |
| pa | 29.8 | 24.1 | 8.9 | 10.6 |
| or | 24.5 | 24.5 | 12.9 | 10.3 |
| ur | 74.2 | 50.2 | 19.2 | – |
| ta | 28.0 | 27.9 | 18.4 | 8.3 |
| kn | 33.3 | 33.3 | 20.2 | 18.7 |

### IndicContext

| lang | Sangam auto | Sangam hinted | IndicConformer | SraVaani |
|---|---:|---:|---:|---:|
| hi | 28.7 | 23.9 | 5.9 | 6.2 |
| bn | 25.5 | 25.4 | 3.4 | 3.2 |
| te | 26.6 | 26.6 | 3.6 | 3.6 |
| ml | 38.1 | 38.1 | 7.1 | 7.5 |
| mr | 38.4 | 38.0 | 5.2 | 5.5 |
| gu | 34.7 | 34.3 | 2.4 | 2.2 |
| or | 34.7 | 34.7 | 9.0 | 9.2 |
| ur | 47.2 | 43.4 | 13.1 | – |

### MUCS

| lang | Sangam auto | Sangam hinted | IndicConformer | SraVaani |
|---|---:|---:|---:|---:|
| hi | 52.9 | 52.0 | 51.7 | 51.9 |
| bn | 34.2 | 33.9 | 46.4 | 46.8 |
