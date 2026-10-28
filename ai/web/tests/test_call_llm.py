"""
Unit tests for web.app.call_llm (lines 10-49)
"""
import pytest
from unittest.mock import patch, MagicMock

from web.app import call_llm


def _mock_response(status_code=200, json_data=None, text=""):
    """Helper to build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text or str(json_data)
    resp.json.return_value = json_data or {}
    return resp


# ── Normal path ──────────────────────────────────────────────

@patch("web.app.requests.post")
def test_call_llm_default_base_url(mock_post):
    """When base_url is None, should use OpenAI default."""
    mock_post.return_value = _mock_response(
        json_data={"choices": [{"message": {"content": "hello"}}]}
    )

    result = call_llm("key", "gpt-4", "hi")

    assert result == "hello"
    call_args = mock_post.call_args
    assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"


@patch("web.app.requests.post")
def test_call_llm_custom_base_url(mock_post):
    """Custom base_url should be used instead of default."""
    mock_post.return_value = _mock_response(
        json_data={"choices": [{"message": {"content": "ok"}}]}
    )

    result = call_llm("key", "deepseek-chat", "hi", base_url="https://api.deepseek.com/v1")

    assert result == "ok"
    call_args = mock_post.call_args
    assert call_args[0][0] == "https://api.deepseek.com/v1/chat/completions"


@patch("web.app.requests.post")
def test_call_llm_request_structure(mock_post):
    """Verify headers, model, messages, and temperature sent to the API."""
    mock_post.return_value = _mock_response(
        json_data={"choices": [{"message": {"content": "x"}}]}
    )

    call_llm("my-api-key", "qwen-plus", "test prompt", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    _, kwargs = mock_post.call_args
    # headers
    assert kwargs["headers"]["Authorization"] == "Bearer my-api-key"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    # payload
    payload = kwargs["json"]
    assert payload["model"] == "qwen-plus"
    assert payload["temperature"] == 0.7
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "You are an expert NPC persona designer."
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"] == "test prompt"


@patch("web.app.requests.post")
def test_call_llm_returns_content(mock_post):
    """Should return the content string from the first choice."""
    mock_post.return_value = _mock_response(
        json_data={"choices": [{"message": {"content": "NPC persona output"}}]}
    )

    result = call_llm("k", "m", "p")

    assert result == "NPC persona output"


# ── Error paths ──────────────────────────────────────────────

@patch("web.app.requests.post")
def test_call_llm_non_200_raises(mock_post):
    """Non-200 status code should raise Exception with API error text."""
    mock_post.return_value = _mock_response(
        status_code=429, text="Rate limit exceeded"
    )

    with pytest.raises(Exception, match="API Error: Rate limit exceeded"):
        call_llm("k", "m", "p")


@patch("web.app.requests.post")
def test_call_llm_500_raises(mock_post):
    """500 status code should raise Exception."""
    mock_post.return_value = _mock_response(
        status_code=500, text="Internal Server Error"
    )

    with pytest.raises(Exception, match="API Error: Internal Server Error"):
        call_llm("k", "m", "p")


@patch("web.app.requests.post")
def test_call_llm_missing_choices_raises(mock_post):
    """200 but no 'choices' key in response should raise Exception."""
    mock_post.return_value = _mock_response(
        status_code=200, json_data={"error": "bad"}
    )

    with pytest.raises(Exception, match="Invalid response"):
        call_llm("k", "m", "p")


@patch("web.app.requests.post")
def test_call_llm_empty_choices_raises(mock_post):
    """200 with empty choices list should raise (IndexError propagated)."""
    mock_post.return_value = _mock_response(
        status_code=200, json_data={"choices": []}
    )

    # Accessing [0] on empty list raises IndexError
    with pytest.raises(IndexError):
        call_llm("k", "m", "p")


# ── Boundary / edge cases ────────────────────────────────────

@patch("web.app.requests.post")
def test_call_llm_empty_prompt(mock_post):
    """Empty prompt string should still be sent normally."""
    mock_post.return_value = _mock_response(
        json_data={"choices": [{"message": {"content": "empty received"}}]}
    )

    result = call_llm("k", "m", "")

    assert result == "empty received"
    payload = mock_post.call_args[1]["json"]
    assert payload["messages"][1]["content"] == ""


@patch("web.app.requests.post")
def test_call_llm_empty_api_key(mock_post):
    """Empty API key is still passed to the header (server will reject)."""
    mock_post.return_value = _mock_response(
        json_data={"choices": [{"message": {"content": "x"}}]}
    )

    call_llm("", "m", "p")

    headers = mock_post.call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer "


@patch("web.app.requests.post")
def test_call_llm_200_with_choices_key(mock_post):
    """Verify exact response parsing with multiple choices present."""
    mock_post.return_value = _mock_response(
        status_code=200,
        json_data={
            "choices": [
                {"message": {"content": "first"}},
                {"message": {"content": "second"}}
            ]
        }
    )

    result = call_llm("k", "m", "p")

    # Should return content from the first choice only
    assert result == "first"


@patch("web.app.requests.post")
def test_call_llm_debug_print(mock_post, capsys):
    """Debug prints for STATUS and RAW RESPONSE should be emitted."""
    mock_post.return_value = _mock_response(
        status_code=200,
        json_data={"choices": [{"message": {"content": "x"}}]},
        text='{"choices": [{"message": {"content": "x"}}]}'
    )

    call_llm("k", "m", "p")

    captured = capsys.readouterr()
    assert "STATUS: 200" in captured.out
    assert "RAW RESPONSE:" in captured.out
