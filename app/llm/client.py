"""Clientes de modelo de lenguaje sobre httpx, sin SDK.

Un solo contrato: `complete_json(system, user, schema)` devuelve una instancia
validada del esquema Pydantic pedido o lanza LLMError. Hay un reintento si la
salida no valida y reintentos con espera si la API está saturada (429, 5xx).
El modelo nunca recibe herramientas ni decide el flujo: transforma texto.
"""

import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.http_errors import describe_error_response

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

ANTHROPIC_VERSION = "2023-06-01"
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})
_DEFAULT_RETRY_DELAYS: tuple[float, ...] = (2.0, 6.0)
_MAX_RETRY_AFTER = 30.0


class LLMError(RuntimeError):
    """Falla de red, de la API o del formato de salida del modelo."""


class LLMClient(Protocol):
    @property
    def model_name(self) -> str: ...

    async def complete_json(self, *, system: str, user: str, schema: type[T]) -> T: ...


def schema_instructions(schema: type[BaseModel]) -> str:
    compact = json.dumps(
        schema.model_json_schema(), ensure_ascii=False, separators=(",", ":")
    )
    return (
        "Responde únicamente con un objeto JSON válido, sin texto antes ni después "
        "y sin bloques de código, que cumpla este JSON Schema:\n" + compact
    )


def extract_json_object(raw: str) -> dict[str, Any]:
    """Toma el primer objeto JSON del texto, aunque venga entre ```json."""
    text = raw.strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no hay un objeto JSON en la respuesta")
    value, _ = json.JSONDecoder().raw_decode(text, start)
    if not isinstance(value, dict):
        raise ValueError("la respuesta no es un objeto JSON")
    return value


def _summarize(exc: Exception) -> str:
    """Qué falló en el formato, sin copiar la salida del modelo.

    Pydantic incluye por defecto el valor rechazado, y ese valor (texto del
    modelo, derivado de la pregunta) terminaría en los registros del servidor.
    """
    if isinstance(exc, ValidationError):
        text = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'objeto'}: "
            f"{error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        )
    else:
        text = str(exc)
    return " ".join(text.split())[:300]


def _retry_after(response: httpx.Response, fallback: float) -> float:
    header = response.headers.get("retry-after", "")
    try:
        return min(max(float(header), 0.0), _MAX_RETRY_AFTER)
    except ValueError:
        return fallback


async def _post_json(
    *,
    provider: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    transport: httpx.AsyncBaseTransport | None,
    delays: Sequence[float],
) -> dict[str, Any]:
    attempts = len(delays) + 1
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        for attempt in range(attempts):
            last = attempt == attempts - 1
            try:
                response = await client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                if last:
                    raise LLMError(
                        f"{provider}: error de red ({exc.__class__.__name__})."
                    ) from exc
                await asyncio.sleep(delays[attempt])
                continue
            if response.status_code in _RETRYABLE_STATUS and not last:
                wait = _retry_after(response, delays[attempt])
                logger.warning(
                    "%s respondió %d; reintento en %.1f s.",
                    provider,
                    response.status_code,
                    wait,
                )
                await asyncio.sleep(wait)
                continue
            if response.status_code != 200:
                # Sin el cuerpo completo: algunos proveedores devuelven ahí
                # parte de lo que se les envió.
                raise LLMError(
                    f"{provider} respondió {describe_error_response(response)}"
                )
            try:
                data = response.json()
            except ValueError as exc:
                raise LLMError(f"{provider}: la respuesta no es JSON.") from exc
            if not isinstance(data, dict):
                raise LLMError(f"{provider}: respuesta inesperada.")
            return data
    raise LLMError(f"{provider}: sin respuesta.")


class _JSONCompletion:
    """Validación contra el esquema con un reintento, común a los proveedores."""

    async def _complete_text(self, *, system: str, user: str) -> str:
        raise NotImplementedError

    async def complete_json(self, *, system: str, user: str, schema: type[T]) -> T:
        instructions = schema_instructions(schema)
        prompt = f"{user}\n\n{instructions}"
        problem = ""
        for _ in range(2):
            raw = await self._complete_text(system=system, user=prompt)
            try:
                return schema.model_validate(extract_json_object(raw))
            except ValueError as exc:  # incluye JSONDecodeError y ValidationError
                problem = _summarize(exc)
                prompt = (
                    f"{user}\n\n{instructions}\n\nTu respuesta anterior no cumplió "
                    f"el formato ({problem}). Devuelve solo el objeto JSON."
                )
        raise LLMError(f"Salida del modelo inválida tras un reintento: {problem}")


class AnthropicClient(_JSONCompletion):
    """API de Mensajes de Anthropic (POST /v1/messages)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 2048,
        timeout: float = 90.0,
        temperature: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        retry_delays: Sequence[float] = _DEFAULT_RETRY_DELAYS,
    ) -> None:
        if not api_key:
            raise ValueError("Falta ANTHROPIC_API_KEY.")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._temperature = temperature
        self._transport = transport
        self._delays = tuple(retry_delays)

    @property
    def model_name(self) -> str:
        return self._model

    async def _complete_text(self, *, system: str, user: str) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        data = await _post_json(
            provider="Anthropic",
            url=f"{self._base_url}/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            payload=payload,
            timeout=self._timeout,
            transport=self._transport,
            delays=self._delays,
        )
        if data.get("stop_reason") == "max_tokens":
            raise LLMError("Anthropic: respuesta truncada; sube LLM_MAX_TOKENS.")
        content = data.get("content")
        if not isinstance(content, list):
            raise LLMError("Anthropic: respuesta sin contenido.")
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )


class OpenAICompatClient(_JSONCompletion):
    """Chat Completions de OpenAI o de un servidor compatible (Ollama, Groq...)."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        json_mode: bool = True,
        max_tokens_field: str = "max_tokens",
        max_tokens: int = 2048,
        timeout: float = 90.0,
        temperature: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        retry_delays: Sequence[float] = _DEFAULT_RETRY_DELAYS,
    ) -> None:
        if not model:
            raise ValueError("Falta OPENAI_MODEL.")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._json_mode = json_mode
        self._max_tokens_field = max_tokens_field
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._temperature = temperature
        self._transport = transport
        self._delays = tuple(retry_delays)

    @property
    def model_name(self) -> str:
        return self._model

    async def _complete_text(self, *, system: str, user: str) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            self._max_tokens_field: self._max_tokens,
        }
        if self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        data = await _post_json(
            provider="OpenAI-compatible",
            url=f"{self._base_url}/chat/completions",
            headers=headers,
            payload=payload,
            timeout=self._timeout,
            transport=self._transport,
            delays=self._delays,
        )
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("OpenAI-compatible: respuesta sin contenido.") from exc
        if isinstance(choice, dict) and choice.get("finish_reason") == "length":
            raise LLMError(
                "OpenAI-compatible: respuesta truncada; sube LLM_MAX_TOKENS."
            )
        if not isinstance(content, str):
            raise LLMError("OpenAI-compatible: contenido vacío.")
        return content


def build_llm(
    settings: Settings, *, role: Literal["generator", "verifier"] = "generator"
) -> LLMClient | None:
    """Cliente según la configuración, o None si falta algo para usarlo."""
    if settings.llm_provider == "none":
        return None
    use_verifier_model = role == "verifier" and bool(settings.verifier_model.strip())
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            logger.warning("LLM_PROVIDER=anthropic pero falta ANTHROPIC_API_KEY.")
            return None
        return AnthropicClient(
            api_key=settings.anthropic_api_key,
            model=(
                settings.verifier_model.strip()
                if use_verifier_model
                else settings.anthropic_model
            ),
            base_url=settings.anthropic_base_url,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            temperature=settings.llm_temperature,
        )
    model = (
        settings.verifier_model.strip() if use_verifier_model else settings.openai_model
    )
    if not model:
        logger.warning("LLM_PROVIDER=openai_compat pero falta OPENAI_MODEL.")
        return None
    return OpenAICompatClient(
        model=model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        json_mode=settings.openai_json_mode,
        max_tokens_field=settings.openai_max_tokens_field,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        temperature=settings.llm_temperature,
    )
