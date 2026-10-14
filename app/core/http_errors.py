"""Resumen de una respuesta de error de una API, sin copiar su cuerpo.

Algunos proveedores devuelven en el error parte de lo que se les envió
(OpenRouter, por ejemplo, pone en metadata.flagged_input el texto que marcó su
moderación). Ese cuerpo terminaría en los registros, y en el servidor del bot
los registros no deben llevar texto de preguntas. Del cuerpo solo se toman el
tipo o código del error y su mensaje, recortado.
"""

from typing import Any

import httpx

_MAX_MESSAGE = 160


def _error_object(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    for key in ("error", "detail"):  # Anthropic y OpenAI: error; ElevenLabs: detail
        value = payload.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return {"message": value}
    return None


def describe_error_response(response: httpx.Response) -> str:
    """'HTTP 400 (invalid_request_error: …)', sin otros campos del cuerpo."""
    try:
        error = _error_object(response.json())
    except ValueError:
        error = None
    parts: list[str] = []
    if error is not None:
        kind = error.get("type") or error.get("code") or error.get("status")
        if isinstance(kind, str | int) and str(kind).strip():
            parts.append(str(kind).strip())
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            parts.append(" ".join(message.split())[:_MAX_MESSAGE])
    detail = f" ({': '.join(parts)})" if parts else ""
    return f"HTTP {response.status_code}{detail}"
