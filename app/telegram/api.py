"""Cliente mínimo de la Bot API de Telegram sobre httpx.

El token va en la ruta de cada URL, así que nada de este módulo puede llegar
a un registro con la URL puesta: los errores de httpx se reemplazan por uno
propio sin URL y sin la excepción original encadenada (que la traería en el
traceback). Quien use este cliente debe además silenciar el registro INFO de
httpx, que imprime cada URL.
"""

from typing import Any

import httpx

API_BASE = "https://api.telegram.org"
# Telegram corta los mensajes de más de 4096 caracteres.
MESSAGE_LIMIT = 4096


class TelegramError(RuntimeError):
    """Falla de Telegram; el mensaje nunca incluye el token."""


class TelegramAPI:
    def __init__(
        self,
        token: str,
        *,
        base_url: str = API_BASE,
        timeout: float = 75.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not token:
            raise ValueError("Falta TELEGRAM_BOT_TOKEN.")
        self._token = token
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def call(self, method: str, **params: Any) -> Any:
        try:
            response = await self._client.post(
                f"/bot{self._token}/{method}", json=params
            )
        except httpx.HTTPError as exc:
            raise TelegramError(f"{method}: {type(exc).__name__}") from None
        try:
            payload = response.json()
        except ValueError:
            raise TelegramError(
                f"{method}: respuesta inválida (HTTP {response.status_code})"
            ) from None
        if not isinstance(payload, dict) or not payload.get("ok"):
            description = (
                payload.get("description", "sin detalle")
                if isinstance(payload, dict)
                else "sin detalle"
            )
            raise TelegramError(
                f"{method}: {description} (HTTP {response.status_code})"
            )
        return payload.get("result")

    async def get_updates(self, offset: int | None, timeout: int) -> list[Any]:
        params: dict[str, Any] = {
            "timeout": timeout,
            "allowed_updates": ["message"],
        }
        if offset is not None:
            params["offset"] = offset
        result = await self.call("getUpdates", **params)
        return result if isinstance(result, list) else []

    async def send_message(self, chat_id: int, text: str, *, html: bool = True) -> None:
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "link_preview_options": {"is_disabled": True},
        }
        if html:
            params["parse_mode"] = "HTML"
        await self.call("sendMessage", **params)

    async def send_typing(self, chat_id: int) -> None:
        await self.call("sendChatAction", chat_id=chat_id, action="typing")

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        await self.call("deleteMessage", chat_id=chat_id, message_id=message_id)

    async def download_file(self, file_id: str, *, max_bytes: int) -> bytes:
        info = await self.call("getFile", file_id=file_id)
        if not isinstance(info, dict) or not info.get("file_path"):
            raise TelegramError("getFile: sin ruta de archivo")
        size = info.get("file_size")
        if isinstance(size, int) and size > max_bytes:
            raise TelegramError("getFile: archivo demasiado grande")
        try:
            response = await self._client.get(
                f"/file/bot{self._token}/{info['file_path']}"
            )
        except httpx.HTTPError as exc:
            raise TelegramError(f"descarga: {type(exc).__name__}") from None
        if response.status_code != 200:
            raise TelegramError(f"descarga: HTTP {response.status_code}")
        if len(response.content) > max_bytes:
            raise TelegramError("descarga: archivo demasiado grande")
        return response.content
