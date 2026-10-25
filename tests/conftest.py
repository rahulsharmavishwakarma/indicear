"""Shared fixtures: a tiny random-weight Gemma4 audio tower (no download; indicear.smoke owns it)."""

import os

import pytest

# The suite runs on the CPU unless a test asks for a device itself (auto would pick MPS or CUDA).
os.environ.setdefault("INDICEAR_DEVICE", "cpu")

from indicear.smoke import make_tiny_tower


@pytest.fixture(scope="session")
def tiny_tower():
    return make_tiny_tower()
