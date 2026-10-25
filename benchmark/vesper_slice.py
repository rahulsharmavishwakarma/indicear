"""Build eval manifests for the clips of Vesper's public ASR benchmark that Vesper-IndicEAR covers, so it
can be scored on exactly the clips Vesper's speech models were scored on.

    uv run --locked python benchmark/vesper_slice.py --out ROOT [--svarah-mucs-manifests DIR ...]

Sources (clean conditions only, i.e. sample ids without a `--condition` suffix or with `--clean`):
- FLEURS hi_in test and en_us test (`google/fleurs@70bb2e84`, the revision Vesper pinned): audio extracted
  from the split's tar; reference = the benchmark row's `reference_normalized`.
- Svarah (Indian English) and MUCS 2021 Hindi-English test: the benchmark publishes no reference text
  for them (licence), so audio and reference come from local IndicEAR eval manifests that hold the
  same utterance ids (`svarah/en/test/<id>`, `mucs/hi/test/<id>`).

Writes ROOT/manifests/eval_vesper_<source>.jsonl (key = the benchmark's sample_id) and
ROOT/vesper_rows.jsonl: the benchmark's own rows for those clips (every Vesper model's transcript).
"""

import argparse
import json
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

BENCH = ("rahulsharmavishwakarma/vesper-asr-benchmark", "a277de9046e44bd03bbf2fb7b2af7c978348487c")
FLEURS = ("google/fleurs", "70bb2e84b976b7e960aa89f1c648e09c59f894dd")
SOURCES = {  # benchmark source_id -> (set name, language, how the audio is found)
    "fleurs-hi_in-test": ("fleurs_hi", "hi", "fleurs:hi_in"),
    "fleurs-en_us-test": ("fleurs_en", "en", "fleurs:en_us"),
    "svarah": ("svarah", "en", "local:svarah/en/test/"),
    "openslr-104-mucs-hi-en-test": ("mucs_hi_en", "hi", "local:mucs/hi/test/"),
}


def clean(sample_id: str) -> bool:
    return "--" not in sample_id or sample_id.endswith("--clean")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local-manifests", type=Path, nargs="+", default=[],
                        help="IndicEAR manifest directories holding the Svarah and MUCS eval rows")
    args = parser.parse_args()
    out = args.out.expanduser()
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in
            open(hf_hub_download(BENCH[0], "data/transcripts.jsonl", repo_type="dataset", revision=BENCH[1]),
                 encoding="utf-8")]
    rows = [r for r in rows if r["source_id"] in SOURCES and clean(r["sample_id"])]
    local = {}
    for directory in args.local_manifests:
        for path in Path(directory).expanduser().glob("eval_*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if Path(row["audio"]).is_file():
                    local[row["key"]] = row
    kept = []
    for source, (name, lang, how) in SOURCES.items():
        clips = {}
        for r in rows:
            if r["source_id"] == source:
                sid = r["sample_id"].removesuffix("--clean")
                if sid not in clips or (clips[sid].get("reference_normalized") is None and r.get("reference_normalized")):
                    clips[sid] = r  # prefer a row that carries the reference
        manifest = []
        if how.startswith("fleurs:"):
            locale = how.split(":", 1)[1]
            tar = hf_hub_download(FLEURS[0], f"data/{locale}/audio/test.tar.gz", repo_type="dataset", revision=FLEURS[1])
            audio_dir = out / "audio" / name
            audio_dir.mkdir(parents=True, exist_ok=True)
            wanted = {sid.rsplit("-", 1)[-1] + ".wav": sid for sid in clips}
            with tarfile.open(tar) as archive:
                for member in archive:
                    base = Path(member.name).name
                    if base in wanted:
                        target = audio_dir / base
                        if not target.exists():
                            target.write_bytes(archive.extractfile(member).read())
                        sid = wanted[base]
                        if not clips[sid].get("reference_normalized"):
                            continue  # no published reference for this clip
                        manifest.append({"key": sid, "audio": str(target), "lang": lang,
                                         "text": clips[sid]["reference_normalized"]})
        else:
            prefix = how.split(":", 1)[1]
            for sid in clips:
                row = local.get(prefix + sid.removeprefix("svarah-"))
                if row:
                    manifest.append({"key": sid, "audio": row["audio"], "lang": lang, "text": row["text"],
                                     **({"seconds": row["seconds"]} if "seconds" in row else {})})
        manifest.sort(key=lambda m: m["key"])
        (out / "manifests" / f"eval_vesper_{name}.jsonl").write_text(
            "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest), encoding="utf-8")
        print(f"{name}: {len(manifest)} of {len(clips)} benchmark clips")
        keep = {m["key"] for m in manifest}
        kept += [r | {"set": name} for r in rows if r["source_id"] == source and r["sample_id"].removesuffix("--clean") in keep]
    (out / "vesper_rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    print(f"{len(kept)} benchmark rows kept")


if __name__ == "__main__":
    main()
