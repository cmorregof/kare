import json
from collections.abc import Callable

import httpx
import pytest

from app.llm.client import (
    AnthropicClient,
    LLMError,
    OpenAICompatClient,
    extract_json_object,
)
from app.schemas.evidence import DraftAnswer

Handler = Callable[[httpx.Request], httpx.Response]


def scripted(responses: list[httpx.Response], seen: list[httpx.Request]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses.pop(0)

    return handler


def anthropic_text(text: str, stop_reason: str = "end_turn") -> httpx.Response:
    return httpx.Response(
        200,
        json={"content": [{"type": "text", "text": text}], "stop_reason": stop_reason},
    )


def anthropic(
    responses: list[httpx.Response], seen: list[httpx.Request]
) -> AnthropicClient:
    return AnthropicClient(
        api_key="clave-de-prueba",
        model="claude-de-prueba",
        transport=httpx.MockTransport(scripted(responses, seen)),
        retry_delays=(0.0, 0.0),
    )


def test_extract_json_object_tolerates_fences_and_trailing_text() -> None:
    raw = 'Aquí va:\n```json\n{"abstain": true}\n```\nFin.'
    assert extract_json_object(raw) == {"abstain": True}
    with pytest.raises(ValueError):
        extract_json_object("sin json")


async def test_anthropic_request_shape_and_parsing() -> None:
    seen: list[httpx.Request] = []
    client = anthropic(
        [
            anthropic_text(
                '```json\n{"claims": [], "abstain": true, "abstain_reason": "x"}\n```'
            )
        ],
        seen,
    )
    result = await client.complete_json(system="S", user="U", schema=DraftAnswer)
    assert result.abstain is True
    assert result.abstain_reason == "x"
    request = seen[0]
    assert request.url.path == "/v1/messages"
    assert request.headers["x-api-key"] == "clave-de-prueba"
    assert request.headers["anthropic-version"] == "2023-06-01"
    payload = json.loads(request.content)
    assert payload["model"] == "claude-de-prueba"
    assert payload["system"] == "S"
    assert payload["messages"][0]["role"] == "user"
    assert "JSON Schema" in payload["messages"][0]["content"]
    assert "temperature" not in payload


async def test_invalid_output_gets_one_retry_with_the_error() -> None:
    seen: list[httpx.Request] = []
    client = anthropic(
        [anthropic_text("Lo siento, no puedo."), anthropic_text('{"abstain": true}')],
        seen,
    )
    result = await client.complete_json(system="S", user="U", schema=DraftAnswer)
    assert result.abstain is True
    assert len(seen) == 2
    assert (
        "no cumplió el formato" in json.loads(seen[1].content)["messages"][0]["content"]
    )


async def test_two_invalid_outputs_raise() -> None:
    client = anthropic([anthropic_text("nada"), anthropic_text('{"claims": 3}')], [])
    with pytest.raises(LLMError):
        await client.complete_json(system="S", user="U", schema=DraftAnswer)


async def test_overloaded_api_is_retried() -> None:
    seen: list[httpx.Request] = []
    client = anthropic(
        [httpx.Response(529, json={"error": "overloaded"}), anthropic_text("{}")], seen
    )
    result = await client.complete_json(system="S", user="U", schema=DraftAnswer)
    assert result.claims == []
    assert len(seen) == 2


async def test_client_errors_are_not_retried() -> None:
    seen: list[httpx.Request] = []
    client = anthropic([httpx.Response(400, json={"error": "bad"})], seen)
    with pytest.raises(LLMError):
        await client.complete_json(system="S", user="U", schema=DraftAnswer)
    assert len(seen) == 1


async def test_truncated_output_raises() -> None:
    client = anthropic([anthropic_text('{"claims": [', stop_reason="max_tokens")], [])
    with pytest.raises(LLMError):
        await client.complete_json(system="S", user="U", schema=DraftAnswer)


async def test_openai_compatible_local_server_without_key() -> None:
    seen: list[httpx.Request] = []
    response = httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": '{"abstain": false}'}, "finish_reason": "stop"}
            ]
        },
    )
    client = OpenAICompatClient(
        model="modelo-local",
        base_url="http://localhost:11434/v1",
        transport=httpx.MockTransport(scripted([response], seen)),
        retry_delays=(0.0,),
    )
    result = await client.complete_json(system="S", user="U", schema=DraftAnswer)
    assert result.abstain is False
    request = seen[0]
    assert request.url.path == "/v1/chat/completions"
    assert "authorization" not in request.headers
    payload = json.loads(request.content)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 2048
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
