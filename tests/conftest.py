"""pytest never makes API calls: every test runs without API keys, and reading .env or constructing a real SDK
client fails the test. Adapters built with an injected fake `client` are unaffected."""

from __future__ import annotations

import pytest

KEY_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY")


def _forbid(what: str):
    def fail(*args, **kwargs):
        # pytest.fail raises a BaseException, so an `except Exception` in the code under test can't swallow it.
        pytest.fail(f"a test {what}; tests must use a fake adapter (tests/agent/fakes.py) and no API keys")
    return fail


@pytest.fixture(autouse=True)
def no_api_calls(monkeypatch):
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    import dotenv.main

    # Every `from dotenv import load_dotenv` binding goes through DotEnv, so this catches them all.
    monkeypatch.setattr(dotenv.main.DotEnv, "dict", _forbid("read .env"))
    import anthropic

    monkeypatch.setattr(anthropic.Anthropic, "__init__", _forbid("constructed a real Anthropic client"))
    monkeypatch.setattr(anthropic.AsyncAnthropic, "__init__", _forbid("constructed a real Anthropic client"))
    try:
        from google import genai
    except ImportError:
        return
    monkeypatch.setattr(genai.Client, "__init__", _forbid("constructed a real Gemini client"))
