# Benchmark tools

Scripts that build **Vesper Indic Bench** (every system's per-clip transcripts on identical held-out clips) and score Vesper-IndicEAR on the clips of Vesper's public ASR benchmark.

| File | What it does |
|---|---|
| `build_bench.py` | Builds the dataset: `data/clips.jsonl`, `data/transcripts.jsonl`, `summary.json` from the evaluation outputs. No audio; clips are referenced by source, revision and id. |
| `vesper_slice.py` | Writes eval manifests for the clips of [Vesper's ASR benchmark](https://huggingface.co/datasets/rahulsharmavishwakarma/vesper-asr-benchmark) that the model covers (FLEURS Hindi and US English, Svarah, MUCS). |
| `score_vesper.py` | Scores Vesper-IndicEAR and every Vesper speech model on those clips with one scorer; writes `results/vesper-benchmark/`. |
| `fill_tables.py` | Fills the result tables of the dataset card from the evaluation outputs. |
| `dataset-card.md` | Template of the Hugging Face dataset card (`{{...}}` counts are filled at build time). |

## Rebuilding

```bash
# 1. decode every system on the same 300 clips per language per set
uv run --locked python -m indicear.decoding eval --model rahulsharmavishwakarma/vesper-indicEAR \
    --manifests ROOT/manifests --per-lang 300 --out ROOT/results/vesper-indicear-auto \
    --baseline-out ROOT/results/vesper-indicear-unconstrained --hint-out ROOT/results/vesper-indicear-hinted
bash scripts/baselines.sh --systems indicconformer-ctc,sravaani --max-per-lang 300 --robust-conditions ""

# 2. build the dataset and fill its card
uv run --locked python benchmark/build_bench.py --manifests ROOT/manifests --results ROOT/results --out DATASET
uv run --locked python benchmark/fill_tables.py --results ROOT/results --vesper results/vesper-benchmark \
    --bench-dir DATASET --out DATASET
```

The per-set scores used in the model card are in `results/bench/`.
