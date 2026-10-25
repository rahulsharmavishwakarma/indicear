"""Pinned sources, shared constants and small I/O helpers."""

from __future__ import annotations

from indicear.runtime.gpu import GpuSampler, gil_enabled, runtime_banner  # noqa: F401 (re-export)
from indicear.runtime.io import read_jsonl, write_json, write_jsonl  # noqa: F401 (re-export)

# Pinned revisions. Kathbath, IndicVoices and the LibriSpeech mirror were pinned from the Hub
# API on 2026-10-23 and re-checked on 2026-10-24 (still the head commits); schemas, licences and
# the other sources live in indicear.sources.
EG2 = ("google/embeddinggemma-2", "914f7f89142e33e77833254d9c9b90c3cef7303b")
FLEURS = ("google/fleurs", "70bb2e84b976b7e960aa89f1c648e09c59f894dd")
KATHBATH = ("ai4bharat/Kathbath", "5b9e92849222026d9141acba4e8434fe816396bf")
INDICVOICES = ("ai4bharat/IndicVoices", "c96f9088f138cf89d419da7e8e643e1f05c00a87")
LIBRISPEECH = ("openslr/librispeech_asr", "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1")
MUCS_MIRRORS = (
    "https://openslr.trmal.net/resources/104/",
    "https://openslr.elda.org/resources/104/",
    "https://openslr.magicdatatech.com/resources/104/",
)

SAMPLE_RATE = 16_000
FRAME_RATE = 25  # encoder output frames per second (40 ms hop)
ENCODER_LAYERS = 12
ENCODER_DIM = 1024
PROJ_DIM = 1536
CACHE_LAYER = 6  # layers 1-6 are frozen with no adapters: their output is cached once

# FLEURS locales for the twelve design-A languages (FLEURS test is never read here).
FLEURS_LOCALES = {
    "hi": "hi_in", "mr": "mr_in", "bn": "bn_in", "pa": "pa_in", "gu": "gu_in", "or": "or_in",
    "ur": "ur_pk", "ta": "ta_in", "te": "te_in", "kn": "kn_in", "ml": "ml_in", "en": "en_us",
}
LANGS = tuple(FLEURS_LOCALES)
# Synthetic code-switch pairs: FLEURS dev utterances from two languages, concatenated.
CODE_SWITCH_PAIRS = (
    ("hi", "en"), ("kn", "en"), ("hi", "kn"), ("ta", "en"), ("te", "en"), ("bn", "en"), ("ur", "en"),
)

# Licence, gating and schema per source: indicear.sources.SOURCES (verified 2026-10-24).
