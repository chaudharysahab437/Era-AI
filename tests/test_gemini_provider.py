from __future__ import annotations

from unittest import mock

import pytest
import requests

from tests.conftest import make_frame
from vision.providers.base import ProviderAuthError, ProviderResponseError
from vision.providers.gemini_provider import GeminiProvider


def _fake_response(status_code, json_body=None, text=""):
    resp = mock.Mock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    resp.text = text
    return resp


def test_not_configured_raises_auth_error():
    provider = GeminiProvider(api_key="", model="gemini-2.5-flash")
    with pytest.raises(ProviderAuthError):
        provider.analyze(make_frame().image, prompt="hi")


def test_successful_response_parses_to_screen_state():
    provider = GeminiProvider(api_key="k", model="gemini-2.5-flash", max_retries=0)
    body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '{"application": "Notepad", "summary": "text editor open", '
                            '"confidence": 0.8, "important_text": ["hello.txt"]}'
                        }
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }
    with mock.patch("requests.post", return_value=_fake_response(200, body)):
        state = provider.analyze(make_frame().image, prompt="describe")
    assert state.application == "Notepad"
    assert state.confidence == 0.8
    assert "hello.txt" in state.important_text


def test_401_raises_auth_error():
    provider = GeminiProvider(api_key="bad", model="gemini-2.5-flash", max_retries=0)
    with mock.patch("requests.post", return_value=_fake_response(401)):
        with pytest.raises(ProviderAuthError):
            provider.analyze(make_frame().image, prompt="describe")


def test_empty_candidates_raises_response_error():
    provider = GeminiProvider(api_key="k", model="gemini-2.5-flash", max_retries=0)
    with mock.patch("requests.post", return_value=_fake_response(200, {"candidates": []})):
        with pytest.raises(ProviderResponseError):
            provider.analyze(make_frame().image, prompt="describe")


def test_malformed_json_text_raises_response_error():
    provider = GeminiProvider(api_key="k", model="gemini-2.5-flash", max_retries=0)
    body = {"candidates": [{"content": {"parts": [{"text": "not json"}]}, "finishReason": "STOP"}]}
    with mock.patch("requests.post", return_value=_fake_response(200, body)):
        with pytest.raises(ProviderResponseError):
            provider.analyze(make_frame().image, prompt="describe")


def test_network_error_retried_then_raises():
    provider = GeminiProvider(api_key="k", model="gemini-2.5-flash", max_retries=1)
    with mock.patch("requests.post", side_effect=requests.ConnectionError("boom")):
        with pytest.raises(Exception):
            provider.analyze(make_frame().image, prompt="describe")


def test_is_configured():
    assert GeminiProvider(api_key="k", model="m").is_configured() is True
    assert GeminiProvider(api_key="", model="m").is_configured() is False
