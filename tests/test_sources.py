import json
import re
from pathlib import Path

from indicear.mixture import load_mixture
from indicear.sources import SOURCES, allowed, license_manifest

DOC = Path(__file__).resolve().parents[1] / "docs" / "data-licenses.json"
SHA = re.compile(r"^[0-9a-f]{40}$")


def test_registry_entries_are_complete():
    for name, source in SOURCES.items():
        assert source.name == name
        assert source.license_status in ("allowed", "review", "excluded")
        assert source.access in ("open", "gated", "blocked")
        assert source.verification in ("verified", "partial", "unverified")
        assert source.attribution and source.license and source.spdx
        if source.kind == "hf-parquet":
            assert source.schema and source.langs and source.splits and source.mb_per_hour > 0
            for lang in source.langs:
                for repo, revision in source.repo_for(lang):
                    assert "/" in repo and SHA.match(revision), (name, lang)
        if source.kind == "url-tar":
            assert source.files and all(f.get("url") or f.get("urls") for f in source.files.values())


def test_licence_gate():
    assert allowed(SOURCES["librispeech"])[0]
    assert not allowed(SOURCES["mucs"])[0] and allowed(SOURCES["mucs"], {"mucs"})[0]
    assert not allowed(SOURCES["common_voice"], {"common_voice"})[0]  # blocked stays blocked
    assert not allowed(SOURCES["indictts"], {"indictts"})[0]  # excluded stays excluded


def test_machine_readable_manifest_matches_registry():
    """docs/data-licenses.json is generated: `uv run python -m indicear.sources --out docs/data-licenses.json`."""
    assert json.loads(DOC.read_text(encoding="utf-8")) == json.loads(json.dumps(license_manifest()))


def test_mixtures_use_only_loadable_sources():
    for name in ("pilot", "full"):
        mixture = load_mixture(name)
        for entry in mixture["train"] + mixture["eval"]:
            source = SOURCES[entry["source"]]
            assert source.license_status != "excluded" and source.access != "blocked"
            assert ("eval" if entry in mixture["eval"] else "train") in source.use
