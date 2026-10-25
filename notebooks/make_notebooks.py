"""Generate the notebooks in this directory (regenerate after editing: `uv run python notebooks/make_notebooks.py`).

- quickstart.ipynb: install the package and transcribe audio with the released model.
- train_colab.ipynb: the released model's recipe on one Colab GPU (pilot, then the full warm-started run).
- 06_ablation.ipynb: the record of the first FLEURS ablation (v1-v4 trained; v5 and v6 implemented), which chose variant v4.

The training notebooks run the same scripts that are launched over SSH (scripts/*.sh under nohup), so
both paths do identical work.
"""

import json
from pathlib import Path

ABLATION: list[tuple[str, str]] = [
    ("markdown", """# IndicEAR design A: FLEURS ablation and robustness probe (historical record)

Frozen EmbeddingGemma 2 audio encoder (`914f7f89`), never modified. Layers 1-6 are cached once; layers 7-12 run every step with trainable adapters. FLEURS train/dev for 12 languages; **FLEURS test is never read**.

| Variant | What it adds |
| --- | --- |
| v1 | linear CTC on frozen layer 12 (the probe baseline) |
| v2 | adapters on layers 7-12 + final CTC |
| v3 | + self-conditioned phonetic CTC at layers 6 and 9 |
| v4 | + two-level script output p(script) x p(subword given script), CTC form |
| v5 | + CR-CTC (optional) |
| v6 | + factored RNN-T head with a CTC branch at weight 0.3 (design A with RNN-T standing in for TDT; optional) |

This is the first ablation (23 October 2026) that chose variant v4; the released model's recipe is `train_colab.ipynb`.\n\n**How to run.** Every long step is a bash script started with `scripts/launch.sh` under `nohup`, so it survives notebook and SSH disconnects. The same commands work over SSH (see README.md). Every step is resumable: rerunning a cell continues where it stopped, and checkpoints are synced to `$SYNC` (Google Drive) after each save.

**Runtime:** an A100 (40/80 GB), an L4 (24 GB) or an RTX PRO 6000 (96 GB). Batch budgets are sized from free VRAM at startup, so no edits are needed."""),
    ("markdown", """**Checkpoints can be pulled to a workstation.** Each run directory (`$ROOT/runs/<variant>/`) holds `LATEST.json`, a manifest of complete files with sha256. On the workstation, `scripts/sync_runs.sh <session> /content/indicear/runs/<variant>` pulls only those files over one multiplexed SSH connection, verifies them and keeps the last 2 checkpoints plus the best in `~/indicear-runs/<variant>/`. If the VM is lost: `scripts/resume_on.sh <new session> <variant>` uploads the verified copy and reruns the run's `command.sh`. Google Drive (`$SYNC`) is only an optional second copy."""),
    ("code", """# @title 1. GPU, CPU, RAM and disk
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
!nproc; free -g | head -2; df -h /content | tail -1"""),
    ("code", """# @title 2. Code and paths
!git clone -q https://github.com/rahulsharmavishwakarma/indicear /content/indicear-code
%env CODE=/content/indicear-code
%env ROOT=/content/indicear
%env VENV=/content/venv314t
# Optional second copy on Drive (mount it first, cell 3).
%env SYNC=/content/drive/MyDrive/indicear
!ls $CODE/scripts"""),
    ("code", """# @title 3. Mount Google Drive (persistent checkpoints) and the Hugging Face token
from google.colab import drive, userdata
drive.mount('/content/drive')
import os
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')  # FLEURS is ungated; the token avoids rate limits"""),
    ("code", """# @title 4. Free-threaded Python 3.14t environment (uv, exact pins)
!bash $CODE/scripts/setup_env.sh"""),
    ("code", """# @title 5. Optional: MUSAN noise (about 11 GB download, only noise/ kept)
# Without it the robustness probe uses synthetic colored noise, and the reports say so.
!bash $CODE/scripts/launch.sh musan $CODE/scripts/musan_noise.sh
%env NOISE_DIR=/content/indicear/musan/musan/noise"""),
    ("code", """# @title 6. Prepare data: FLEURS manifests, tokenizer, code-switch dev, negatives, label rates (background)
!bash $CODE/scripts/launch.sh prep $CODE/scripts/prep.sh"""),
    ("code", """# @title Progress of prep (rerun any time)
!tail -n 20 $ROOT/logs/prep.log"""),
    ("code", """# @title 7. Cache layers 1-6 once (background; resumable per part, df checked first)
# AUG_VIEWS=2 adds two augmented copies per training utterance (speed, reverb + point noise,
# codecs, babble, noise): about 3x the cache size.
!WAIT_FOR=$ROOT/prep_report.json AUG_VIEWS=0 STORAGE=bf16 bash $CODE/scripts/launch.sh cache $CODE/scripts/cache.sh"""),
    ("code", """# @title Progress of the cache (throughput in hours of audio per hour, GPU utilization)
!tail -n 8 $ROOT/logs/cache.log; tail -n 3 $ROOT/cache/layer6/gpu.csv"""),
    ("code", """# @title 8. Train v1, the frozen-encoder baseline (background)
!WAIT_FOR=$ROOT/cache/layer6/cache.done VARIANTS=v1 PARALLEL=1 bash $CODE/scripts/launch.sh ablation-v1 $CODE/scripts/ablation.sh"""),
    ("code", """# @title 9. Robustness probe of the frozen encoder with the v1 heads (background, 3 jobs on one GPU)
# Clean, SNR 20/10/5 (noise and babble), reverberation, 8 kHz telephone, AMR-NB/GSM/Opus, speed 0.9/1.1.
!WAIT_FOR=$ROOT/results/v1.json PARALLEL=3 bash $CODE/scripts/launch.sh robustness $CODE/scripts/robustness.sh"""),
    ("code", """# @title 10. Train v2-v4 (background). PARALLEL=3 shares the GPU between three variants.
# Add v5 v6 to VARIANTS for CR-CTC and the RNN-T head.
# Queued behind the robustness probe so the jobs do not compete for VRAM.
!WAIT_FOR=$ROOT/results/robustness/v1/summary.json VARIANTS="v2 v3 v4" PARALLEL=3 bash $CODE/scripts/launch.sh ablation $CODE/scripts/ablation.sh"""),
    ("code", """# @title Status of every job (rerun any time)
!bash $CODE/scripts/status.sh"""),
    ("code", """# @title 11. Results
import json, pathlib
root = pathlib.Path('/content/indicear/results')
for name in ['summary.md', 'robustness/v1/summary.md']:
    path = root / name
    print(f'== {name}'); print(path.read_text() if path.exists() else 'not yet')
print(json.loads((pathlib.Path('/content/indicear') / 'prep_report.json').read_text())['label_rates'])"""),
    ("code", """# @title 12. Copy results to Drive, then release the runtime
!mkdir -p $SYNC/results && cp -r $ROOT/results/. $SYNC/results/ && ls $SYNC/results
print('Results copied. Release the runtime now: Runtime > Disconnect and delete runtime.')"""),
]


QUICKSTART: list[tuple[str, str]] = [
    ("markdown", """# Vesper-IndicEAR quickstart

Transcribe speech in 11 Indian languages and English with Vesper-IndicEAR (part of the Vesper project): a frozen EmbeddingGemma 2 audio encoder with small trained adapters and heads. The output is lowercase text without punctuation, each word in its own script (Latin for English words), plus the detected language.

Runs on a CPU, an NVIDIA GPU or Apple MPS. The model download is 627 MiB (it includes the encoder's audio tower)."""),
    ("code", """# @title 1. Install
!pip install -q indicear"""),
    ("code", """# @title 2. Load the model (downloads it once)
from indicear.transcribe import Transcriber

model = Transcriber("rahulsharmavishwakarma/vesper-indicEAR")  # device and precision are picked automatically
print(model.manifest["languages"], model.device)"""),
    ("code", """# @title 3. An English sample (LibriSpeech, CC BY 4.0)
import io, soundfile as sf
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

parquet = hf_hub_download("hf-internal-testing/librispeech_asr_dummy", "clean/validation-00000-of-00001.parquet",
                          repo_type="dataset", revision="5be91486e11a2d616f4ec5db8d3fd248585ac07a")
row = pq.read_table(parquet).slice(0, 1).to_pylist()[0]
audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
sf.write("sample.wav", audio, rate)
print("reference:", row["text"].lower())"""),
    ("code", """# @title 4. Transcribe a file
result = model.transcribe_file("sample.wav")
print(result["text"])
print("language:", result["language"]["top"])"""),
    ("code", """# @title 5. Your own audio: upload any file soundfile or ffmpeg can read (wav, flac, mp3, m4a, ...)
# In Colab: from google.colab import files; name = next(iter(files.upload()))
# result = model.transcribe_file(name)
# If you know the language, a hint helps (most of all for Urdu, which is often detected as Hindi):
# from indicear.transcribe import read_audio
# result = model.transcribe(read_audio(name), lang_hint="ur")"""),
    ("code", """# @title 6. Words, times and code-switch spans
for word in result["words"][:10]:
    print(f'{word["start"]:6.2f} {word["end"]:6.2f}  {word["word"]}  {word.get("lang", "")}')
model.close()"""),
]

TRAIN_COLAB: list[tuple[str, str]] = [
    ("markdown", """# IndicEAR: train the released model on Colab

The recipe of the released model (`docs/training-recipe.md`), in two stages on one GPU runtime:

1. **Pilot** (`scripts/pilot.sh`): Hindi, Tamil, Kannada and English from the frozen encoder; about 4.3 A100-hours (less on a G4).
2. **Full warm-started run** (`scripts/full_warm.sh`): all 12 languages, warm-started from the pilot; about 6 hours on an RTX PRO 6000 (Colab G4, about 55 units for training).

Every long step is a bash script started with `scripts/launch.sh` under `nohup`, so it survives notebook disconnects; every step resumes when rerun. Training audio streams from the Hugging Face Hub, so a Hugging Face token (Colab secret `HF_TOKEN`, with the gated datasets' terms accepted) is needed.

**Runtime:** Colab G4 (RTX PRO 6000, 96 GB) recommended, or an A100 (80 GB). Batch budgets are sized from free VRAM, so no edits are needed."""),
    ("code", """# @title 1. GPU, CPU, RAM and disk
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
!nproc; free -g | head -2; df -h /content | tail -1"""),
    ("code", """# @title 2. Code, token and environment (free-threaded Python 3.14t, exact pins)
!git clone -q https://github.com/rahulsharmavishwakarma/indicear /content/indicear-code
%env CODE=/content/indicear-code
%env VENV=/content/venv314t
from google.colab import userdata
import os
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
!bash $CODE/scripts/setup_env.sh"""),
    ("code", """# @title 3. Stage 1: the pilot (background: setup, prep, train, eval, report)
%env ROOT=/content/pilot
!bash $CODE/scripts/pilot.sh"""),
    ("code", """# @title Progress (rerun any time)
!bash $CODE/scripts/status.sh"""),
    ("code", """# @title 4. Warm-start source: the pilot's final v4+lid weights and tokenizer
!mkdir -p /content/full/init/v4+lid
!cd /content/pilot/runs/v4+lid && cp heads.safetensors adapters.safetensors variant.json lid.json result.json /content/full/init/v4+lid/
!cp -r /content/pilot/tok /content/full/init/v4+lid/tok && ls /content/full/init/v4+lid"""),
    ("code", """# @title 5. Stage 2: the full warm-started run (background: setup, prep, train, eval, report)
# The released run added --zero-streams at step 1,079 and --stream-scale at step 3,427 (see the recipe's
# "Training at scale"); here they apply from the start.
%env ROOT=/content/full
%env INIT_FROM=/content/full/init/v4+lid
%env TRAIN_EXTRA=--new-script-bias -2 --cache-gb 100 --readahead 64 --zero-streams kathbath/ml,kathbath/or,kathbath/ur,kathbath/gu --stream-scale indicvoices/hi=2.24,vaani/hi=2.24
!bash $CODE/scripts/full_warm.sh"""),
    ("code", """# @title Progress (rerun any time): step, loss, throughput, GPU use
!bash $CODE/scripts/status.sh
!tail -n 3 /content/full/runs/v4+lid/train.csv"""),
    ("code", """# @title 6. Package the model (with the audio tower) and try it
!cd $CODE && $VENV/bin/python -m indicear.export --run /content/full/runs/v4+lid --tok /content/full/tok --out /content/indicear-model --with-encoder
import json
clip = json.loads(open('/content/full/manifests/eval_kathbath_clean_test_unknown_hi.jsonl').readline())
print('reference:', clip['text'])
!cd $CODE && $VENV/bin/python -m indicear.transcribe --model /content/indicear-model {clip['audio']}"""),
    ("code", """# @title 7. Results; copy what you want to keep, then release the runtime
!cat /content/full/results/eval_summary.md | head -60
print('Copy /content/indicear-model and /content/full/results, then: Runtime > Disconnect and delete runtime.')"""),
]


def write(name: str, cells_source: list[tuple[str, str]], gpu: str | None) -> None:
    cells = []
    for kind, source in cells_source:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    metadata = {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
    if gpu:
        metadata |= {"accelerator": "GPU", "colab": {"gpuType": gpu, "machine_shape": "hm"}}
    notebook = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 0}
    path = Path(__file__).parent / name
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    write("quickstart.ipynb", QUICKSTART, None)
    write("train_colab.ipynb", TRAIN_COLAB, "A100")
    write("06_ablation.ipynb", ABLATION, "A100")


if __name__ == "__main__":
    main()
