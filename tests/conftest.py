# tests/conftest.py
"""Shared test isolation."""
import pytest


@pytest.fixture(autouse=True)
def isolated_trust_file(tmp_path, monkeypatch):
    """
    Keep workspace trust out of the real ~/.pixlang/trusted.toml.

    Set in the environment rather than patched in, so CLI subprocesses started
    by a test inherit the same store.
    """
    monkeypatch.setenv("PIXLANG_TRUST_FILE", str(tmp_path / "trusted.toml"))
