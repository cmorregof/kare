"""KARE-E en Telegram: un bot privado para preguntas clínicas generales.

Telegram es zona personal: guarda los mensajes en sus servidores antes de que
el bot los lea. Por eso:

- Solo responde a los IDs autorizados. A cualquier otro le dice su ID y nada
  más; su mensaje no pasa por la guardia ni por el modelo.
- La guardia de identificadores corre antes de cualquier llamada al modelo.
  Si un mensaje trae identificadores, el bot lo borra del chat, no lo procesa
  y no repite el dato en su respuesta.
- Una pregunta muy larga suele ser una nota clínica pegada: no se procesa.
- El registro (que en el servidor ve quien administre el despliegue) nunca
  lleva el texto de una pregunta, una respuesta ni una transcripción: solo
  estados, conteos y tiempos.
"""

import asyncio
import contextlib
import logging
import mimetypes
import re
import tempfile
import time
import traceback
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from html import escape, unescape
from pathlib import Path
from typing import Any, Protocol

from app.answer.factory import build_pipeline
from app.core.config import Settings, parse_user_ids
from app.corpus.store import CorpusStore
from app.guard.identifiers import describe, kinds, scan
from app.schemas.evidence import AnswerStatus, FinalAnswer
from app.telegram.api import TelegramAPI, TelegramError
from app.telegram.format import render_telegram
from app.transcription import Transcriber, get_transcriber, transcribe_safely

logger = logging.getLogger(__name__)

PRIVATE = (
    "Este bot es privado. Tu ID de Telegram es {user_id}: pásaselo a quien lo "
    "administra si debes tener acceso."
)
YOUR_ID = "Tu ID de Telegram es {user_id}."
UNKNOWN_COMMAND = "No conozco ese comando. Escribe /ayuda."
ONLY_TEXT = (
    "Solo entiendo preguntas escritas o notas de voz; no proceso imágenes ni "
    "documentos. Si lo que enviaste trae datos de un paciente, bórralo del chat."
)
TOO_LONG = (
    "La pregunta es muy larga ({length} caracteres; el máximo es {limit}). "
    "Escribe una pregunta corta y general, sin notas clínicas."
)
DELETED = "Borré tu mensaje de este chat."
NO_VOICE = "Las notas de voz no están activadas. Escribe la pregunta."
VOICE_TOO_LONG = "La nota dura más de {limit} segundos. Haz una pregunta corta."
VOICE_FAILED = "No pude transcribir la nota. Intenta de nuevo o escribe la pregunta."
VOICE_BLOCKED = (
    "<b>Nota de voz bloqueada</b>\n"
    "La transcripción trae datos que identifican a una persona ({kinds}). Borré "
    "la nota de este chat y el texto no se envió al modelo. El audio sí pasó por "
    "Telegram y por el servicio de transcripción: dicta solo preguntas generales."
)
HEARD = "Escuché: «{text}»"
FAILED = "No pude responder por un error interno. Intenta de nuevo en un momento."

_COMMAND = re.compile(r"^/([a-z_]+)(?:@\w+)?(?:\s|$)", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def welcome(*, voice: bool) -> str:
    how = (
        "Escribe la pregunta o manda una nota de voz corta."
        if voice
        else "Escribe la pregunta."
    )
    return (
        "<b>KARE</b> · evidencia citada para cuidados paliativos oncológicos\n\n"
        "Respondo preguntas clínicas generales con afirmaciones citadas a guías y "
        "normas, y me abstengo cuando las fuentes no alcanzan. Un verificador "
        "revisa cada afirmación contra su fuente; lo que no queda sustentado no "
        "se muestra.\n\n"
        "<b>Reglas</b>\n"
        "• Solo preguntas generales. Nada que identifique a un paciente: nombres, "
        "documentos, fechas exactas, teléfonos, direcciones ni números de "
        "historia. Si detecto algo así, borro el mensaje y no lo proceso.\n"
        "• Lo que escribes pasa por los servidores de Telegram y del modelo de "
        "lenguaje.\n"
        "• Es apoyo a la decisión: verifica cada afirmación en la fuente citada.\n\n"
        f"{how}\n"
        "/fuentes · documentos que consulto\n"
        "/id · tu ID de Telegram"
    )


def describe_sources(store: CorpusStore) -> str:
    counts = store.chunk_counts()
    lines = ["<b>Fuentes indexadas</b>"]
    for source_id, info in sorted(store.sources().items()):
        indexed = counts.get(source_id, 0)
        if not indexed:
            continue
        year = f", {info.year}" if info.year else ""
        lines.append(
            f"• {escape(info.title, quote=False)} "
            f"({escape(info.publisher, quote=False)}{year}) · {indexed} fragmentos"
        )
    if len(lines) == 1:
        return "Todavía no hay fuentes indexadas: no puedo responder nada."
    return "\n".join(lines)


def plain(html: str) -> str:
    """Texto sin formato, para reenviar si Telegram no acepta el HTML."""
    return unescape(_TAG.sub("", html))


class Answerer(Protocol):
    async def answer(self, question: str) -> FinalAnswer: ...


@dataclass(frozen=True)
class BotConfig:
    allowed_users: frozenset[int]
    max_question_chars: int = 600
    max_voice_seconds: int = 90
    max_voice_bytes: int = 5 * 1024 * 1024
    poll_timeout: int = 50
    typing_every: float = 4.0


def _where(exc: BaseException) -> str:
    """Archivo y línea donde falló, sin el mensaje (que podría traer texto)."""
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return type(exc).__name__
    last = frames[-1]
    return f"{type(exc).__name__} en {Path(last.filename).name}:{last.lineno}"


class KareBot:
    def __init__(
        self,
        api: TelegramAPI,
        answerer: Answerer,
        config: BotConfig,
        *,
        sources_text: str,
        transcriber: Transcriber | None = None,
    ) -> None:
        self._api = api
        self._answerer = answerer
        self._config = config
        self._sources_text = sources_text
        self._transcriber = transcriber

    async def run(self) -> None:
        # getUpdates no funciona si quedó un webhook configurado.
        await self._api.call("deleteWebhook")
        offset: int | None = None
        while True:
            try:
                updates = await self._api.get_updates(offset, self._config.poll_timeout)
            except TelegramError as exc:
                logger.warning("Falló getUpdates (%s); reintento en 5 s.", exc)
                await asyncio.sleep(5)
                continue
            for update in updates:
                if not isinstance(update, dict):
                    continue
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    offset = update_id + 1
                await self.handle_update(update)

    async def handle_update(self, update: Mapping[str, Any]) -> None:
        """Nunca lanza: un mensaje que falla no tumba el bot."""
        message = update.get("message")
        if not isinstance(message, dict):
            return
        chat = message.get("chat")
        sender = message.get("from")
        if not isinstance(chat, dict) or not isinstance(sender, dict):
            return
        chat_id = chat.get("id")
        user_id = sender.get("id")
        if chat.get("type") != "private" or not isinstance(chat_id, int):
            return
        if not isinstance(user_id, int):
            return
        try:
            await self._dispatch(chat_id, user_id, message)
        except Exception as exc:
            logger.error("Falló un mensaje: %s", _where(exc))
            with contextlib.suppress(TelegramError):
                await self._api.send_message(chat_id, FAILED, html=False)

    async def _dispatch(
        self, chat_id: int, user_id: int, message: Mapping[str, Any]
    ) -> None:
        text = message.get("text")
        command = None
        if isinstance(text, str):
            match = _COMMAND.match(text.strip())
            command = match.group(1).lower() if match else None
        if command == "id":
            await self._send(chat_id, YOUR_ID.format(user_id=user_id))
            return
        if user_id not in self._config.allowed_users:
            logger.info("Mensaje de un ID no autorizado: %d", user_id)
            await self._send(chat_id, PRIVATE.format(user_id=user_id))
            return
        message_id = message.get("message_id")
        if not isinstance(message_id, int):
            return
        audio = message.get("voice") or message.get("audio")
        if command in {"start", "ayuda", "help"}:
            await self._send(chat_id, welcome(voice=self._transcriber is not None))
        elif command == "fuentes":
            await self._send(chat_id, self._sources_text)
        elif command:
            await self._send(chat_id, UNKNOWN_COMMAND)
        elif isinstance(audio, dict):
            await self._voice(chat_id, message_id, audio)
        elif isinstance(text, str) and text.strip():
            await self._question(chat_id, message_id, text.strip())
        else:
            await self._send(chat_id, ONLY_TEXT)

    async def _question(self, chat_id: int, message_id: int, text: str) -> None:
        limit = self._config.max_question_chars
        if len(text) > limit:
            findings = scan(text)
            if findings:
                await self._block(chat_id, message_id, describe(findings))
            else:
                await self._send(
                    chat_id, TOO_LONG.format(length=len(text), limit=limit)
                )
            logger.info("Pregunta demasiado larga, no procesada.")
            return
        started = time.monotonic()
        async with self._typing(chat_id):
            answer = await self._answerer.answer(text)
        if answer.status == AnswerStatus.BLOCKED:
            await self._block(chat_id, message_id, answer.reason or "")
            logger.info("Pregunta bloqueada: %s", ", ".join(answer.blocked_kinds))
            return
        for part in render_telegram(answer):
            await self._send(chat_id, part)
        logger.info(
            "Respuesta: estado=%s, afirmaciones=%d, descartadas=%d, %.1f s%s",
            answer.status.value,
            len(answer.claims),
            len(answer.discarded),
            time.monotonic() - started,
            ", con error del modelo" if answer.error else "",
        )

    async def _block(self, chat_id: int, message_id: int, reason: str) -> None:
        deleted = await self._delete(chat_id, message_id)
        lines = ["<b>Pregunta bloqueada</b>", escape(reason, quote=False)]
        if deleted:
            lines.append(DELETED)
        await self._send(chat_id, "\n".join(lines))

    async def _voice(
        self, chat_id: int, message_id: int, audio: Mapping[str, Any]
    ) -> None:
        if self._transcriber is None:
            await self._send(chat_id, NO_VOICE)
            return
        limit = self._config.max_voice_seconds
        duration = audio.get("duration")
        file_id = audio.get("file_id")
        if isinstance(duration, int) and duration > limit:
            await self._send(chat_id, VOICE_TOO_LONG.format(limit=limit))
            return
        if not isinstance(file_id, str):
            await self._send(chat_id, VOICE_FAILED)
            return
        mime = audio.get("mime_type")
        guessed = mimetypes.guess_extension(mime) if isinstance(mime, str) else None
        suffix = guessed or ".ogg"
        async with self._typing(chat_id):
            data = await self._api.download_file(
                file_id, max_bytes=self._config.max_voice_bytes
            )
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / f"nota{suffix}"
                path.write_bytes(data)
                heard = await transcribe_safely(self._transcriber, str(path))
        if not heard:
            await self._send(chat_id, VOICE_FAILED)
            return
        findings = scan(heard)
        if findings:
            await self._delete(chat_id, message_id)
            names = ", ".join(kind.value for kind in kinds(findings))
            await self._send(chat_id, VOICE_BLOCKED.format(kinds=escape(names)))
            logger.info("Nota de voz bloqueada: %s", names)
            return
        await self._send(chat_id, HEARD.format(text=escape(heard, quote=False)))
        await self._question(chat_id, message_id, heard)

    async def _send(self, chat_id: int, html: str) -> None:
        try:
            await self._api.send_message(chat_id, html)
        except TelegramError as exc:
            if "parse" not in str(exc).lower():
                raise
            # El HTML no pasó (por ejemplo, un bloque recortado a la mitad de
            # una etiqueta): mismo texto, sin formato.
            await self._api.send_message(chat_id, plain(html), html=False)

    async def _delete(self, chat_id: int, message_id: int) -> bool:
        try:
            await self._api.delete_message(chat_id, message_id)
        except TelegramError as exc:
            logger.warning("No se pudo borrar un mensaje bloqueado (%s).", exc)
            return False
        return True

    @contextlib.asynccontextmanager
    async def _typing(self, chat_id: int) -> AsyncIterator[None]:
        """Mantiene el "escribiendo…" mientras el pipeline trabaja."""

        async def keep_typing() -> None:
            while True:
                await asyncio.sleep(self._config.typing_every)
                with contextlib.suppress(TelegramError):
                    await self._api.send_typing(chat_id)

        with contextlib.suppress(TelegramError):
            await self._api.send_typing(chat_id)
        task = asyncio.create_task(keep_typing())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def run_bot(settings: Settings) -> int:
    # httpx registra cada URL en INFO, y en Telegram el token va en la URL.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if not settings.telegram_bot_token:
        logger.error("Falta TELEGRAM_BOT_TOKEN.")
        return 2
    allowed = parse_user_ids(settings.telegram_allowed_users)
    with CorpusStore(settings.db_path) as store:
        pipeline, indexed = build_pipeline(settings, store)
        sources_text = describe_sources(store)
    if not pipeline.has_model:
        logger.error("El bot necesita un modelo: revisa LLM_PROVIDER y la clave.")
        return 2
    if not allowed:
        logger.warning("Sin TELEGRAM_ALLOWED_USERS: el bot solo responde /id.")
    if indexed == 0:
        logger.warning("Corpus vacío: toda pregunta quedará sin respuesta.")
    transcriber = get_transcriber(settings)
    api = TelegramAPI(settings.telegram_bot_token)
    config = BotConfig(
        allowed_users=allowed,
        max_question_chars=settings.telegram_max_question_chars,
    )
    bot = KareBot(
        api, pipeline, config, sources_text=sources_text, transcriber=transcriber
    )
    try:
        me = await api.call("getMe")
        username = me.get("username") if isinstance(me, dict) else None
        logger.info(
            "Bot @%s listo: %d IDs autorizados, %d fragmentos, notas de voz: %s.",
            username,
            len(allowed),
            indexed,
            "sí" if transcriber else "no",
        )
        await bot.run()
    except TelegramError as exc:
        # Token inválido, red caída: el mensaje ya viene sin el token.
        logger.error("Telegram no respondió como se esperaba: %s", exc)
        return 1
    finally:
        await api.aclose()
    return 0
