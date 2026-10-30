"""The conftest guard (tests/conftest.py): a test that would reach a real API fails instead."""

import os

import pytest
from dotenv import load_dotenv

from agent.models import AnthropicAdapter, ModelConfig
from agent.run import ENV_FILE


def test_api_keys_are_removed():
    assert not any(os.environ.get(v) for v in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"))


def test_reading_dotenv_fails():
    with pytest.raises(pytest.fail.Exception, match="read .env"):
        load_dotenv(ENV_FILE)


def test_a_real_anthropic_adapter_fails():
    with pytest.raises(pytest.fail.Exception, match="real Anthropic client"):
        AnthropicAdapter(ModelConfig(model="claude-haiku-5-5"))


def test_a_real_gemini_adapter_fails(monkeypatch):
    pytest.importorskip("google.genai")
    from agent.gemini import GeminiAdapter

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    with pytest.raises(pytest.fail.Exception, match="real Gemini client"):
        GeminiAdapter(ModelConfig(model="gemini-3.8-flash"))


def test_an_injected_client_is_allowed():
    AnthropicAdapter(ModelConfig(model="claude-haiku-5-5"), client=object())
