"""KARE-E en Telegram: un bot privado para preguntas clínicas generales.

Telegram es zona personal: guarda los mensajes en sus servidores antes de que
el bot los lea. Por eso:

- Solo responde a los IDs autorizados. A cualquier otro le dice su ID y nada
  más; su mensaje no pasa por la guardia ni por el modelo.
- La guardia corre sobre todo texto (pregunta, comando o pie de foto) antes de
  cualquier otra cosa. Si encuentra identificadores, el bot borra el mensaje
  del chat, no lo procesa y no repite el dato. Si no logra borrarlo, lo dice.
- Un texto muy largo suele ser una nota clínica pegada: se borra sin
  procesarlo, y también los pedazos en que Telegram parte un texto de más de
  4096 caracteres, que llegan seguidos como mensajes aparte.
- Imágenes, documentos y otros archivos no se procesan y se borran.
- El registro (que en el servidor ve quien administre el despliegue) nunca
  lleva el texto de una pregunta, una respuesta ni una transcripción: solo
  estados, conteos y tiempos.
"""

import asyncio
import contextlib
import logging
import mimetypes
import re
import signal
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
DELETED = "Borré tu mensaje de este chat."
NOT_DELETED = "No pude borrarlo: bórralo tú del chat."
BLOCKED = "<b>Pregunta bloqueada</b>\n{reason}"
TOO_LONG = (
    "<b>Mensaje muy largo</b>\n"
    "Tiene {length} caracteres y el máximo para una pregunta es {limit}. Un texto "
    "así suele ser una nota clínica, que no debe ir aquí. Escribe una pregunta "
    "corta y general."
)
CONTINUATION = (
    "<b>Continuación de un mensaje largo</b>\n"
    "Telegram partió un texto largo en varios mensajes; este tampoco lo proceso."
)
NOT_PROCESSED = (
    "<b>Archivo no procesado</b>\n"
    "Solo respondo preguntas escritas o notas de voz; no proceso imágenes, "
    "documentos ni otros archivos, por si traen datos de un paciente."
)
NO_VOICE = "Las notas de voz no están activadas. Escribe la pregunta."
VOICE_TOO_LONG = "La nota dura más de {limit} segundos. Haz una pregunta corta."
VOICE_FAILED = "No pude transcribir la nota. Intenta de nuevo o escribe la pregunta."
VOICE_BLOCKED = (
    "<b>Nota de voz bloqueada</b>\n"
    "La transcripción trae datos que identifican a una persona ({kinds}). El "
    "texto no se envió al modelo. El audio sí pasó por Telegram y por el servicio "
    "de transcripción: dicta solo preguntas generales."
)
VOICE_TRANSCRIPT_TOO_LONG = (
    "<b>Nota muy larga</b>\n"
    "La transcripción tiene {length} caracteres y el máximo para una pregunta es "
    "{limit}. Haz una pregunta corta y general."
)
HEARD = "Escuché: «{text}»"
FAILED = "No pude responder por un error interno. Intenta de nuevo en un momento."

# Los pedazos de un texto partido por Telegram llegan en el mismo segundo o en
# el siguiente.
CONTINUATION_SECONDS = 3

_COMMAND = re.compile(r"^/([a-z_]+)(?:@\w+)?(?:\s|$)", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def welcome(*, voice: bool) -> str:
    how = (
        "Escribe la pregunta o manda una nota de voz corta."
        if voice
        else "Escribe la pregunta."
    )
    audio = (
        "• Una nota de voz pasa por el servicio de transcripción antes de que yo "
        "pueda revisarla: dicta solo preguntas generales.\n"
        if voice
        else ""
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
        f"{audio}"
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
    # Una nota de voz de Telegram (Opus) de 90 s pesa unos 300 KB.
    max_voice_bytes: int = 2 * 1024 * 1024
    poll_timeout: int = 50
    typing_every: float = 4.0
    retry_seconds: float = 5.0


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
        # Por chat: fecha (en segundos) del último texto que pasó del límite.
        self._last_long: dict[int, int] = {}

    async def run(self) -> None:
        # getUpdates no funciona si quedó un webhook configurado.
        await self._api.call("deleteWebhook")
        offset: int | None = None
        while True:
            try:
                updates = await self._api.get_updates(offset, self._config.poll_timeout)
            except TelegramError as exc:
                if exc.status in {401, 403, 404}:
                    # Token inválido o bot borrado: que el despliegue falle a la
                    # vista en lugar de reintentar para siempre.
                    raise
                if exc.status == 409:
                    # Quedó un webhook u otra copia del bot lee el mismo token.
                    logger.warning(
                        "Conflicto en getUpdates (%s): ¿hay otra copia del bot "
                        "corriendo con este token?",
                        exc,
                    )
                    with contextlib.suppress(TelegramError):
                        await self._api.call("deleteWebhook")
                else:
                    logger.warning("Falló getUpdates (%s); reintento.", exc)
                await asyncio.sleep(self._config.retry_seconds)
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
        if user_id not in self._config.allowed_users:
            # A un desconocido solo se le dice su ID: su texto no pasa por la
            # guardia, el modelo ni nada más.
            logger.info("Mensaje de un ID no autorizado: %d", user_id)
            reply = YOUR_ID if command == "id" else PRIVATE
            await self._send(chat_id, reply.format(user_id=user_id))
            return
        message_id = message.get("message_id")
        if not isinstance(message_id, int):
            return
        date = message.get("date")
        limit = self._config.max_question_chars

        if isinstance(text, str):
            if self._continues_long_text(chat_id, date):
                self._mark_long(chat_id, date)
                await self._refuse(chat_id, message_id, CONTINUATION)
                logger.info("Continuación de un texto largo, borrada.")
                return
            if len(text) > limit:
                self._mark_long(chat_id, date)

        written = " ".join(
            part for part in (text, message.get("caption")) if isinstance(part, str)
        )
        findings = scan(written) if written else []
        if findings:
            await self._refuse(
                chat_id,
                message_id,
                BLOCKED.format(reason=escape(describe(findings), quote=False)),
            )
            names = ", ".join(kind.value for kind in kinds(findings))
            logger.info("Mensaje bloqueado: %s", names)
            return
        if isinstance(text, str) and len(text) > limit:
            await self._refuse(
                chat_id, message_id, TOO_LONG.format(length=len(text), limit=limit)
            )
            logger.info("Texto largo borrado sin procesar.")
            return

        audio = message.get("voice") or message.get("audio")
        if command in {"start", "ayuda", "help"}:
            await self._send(chat_id, welcome(voice=self._transcriber is not None))
        elif command == "id":
            await self._send(chat_id, YOUR_ID.format(user_id=user_id))
        elif command == "fuentes":
            await self._send(chat_id, self._sources_text)
        elif command:
            await self._send(chat_id, UNKNOWN_COMMAND)
        elif isinstance(audio, dict):
            await self._voice(chat_id, message_id, audio)
        elif isinstance(text, str) and text.strip():
            await self._question(chat_id, message_id, text.strip())
        else:
            await self._refuse(chat_id, message_id, NOT_PROCESSED)
            logger.info("Mensaje sin texto ni voz, borrado.")

    def _continues_long_text(self, chat_id: int, date: object) -> bool:
        last = self._last_long.get(chat_id)
        return (
            isinstance(date, int)
            and last is not None
            and 0 <= date - last <= CONTINUATION_SECONDS
        )

    def _mark_long(self, chat_id: int, date: object) -> None:
        if isinstance(date, int):
            self._last_long[chat_id] = date

    async def _question(self, chat_id: int, message_id: int, text: str) -> None:
        started = time.monotonic()
        async with self._typing(chat_id):
            answer = await self._answerer.answer(text)
        if answer.status == AnswerStatus.BLOCKED:
            # La guardia ya corrió antes; esto es solo por si el pipeline
            # bloquea algo más.
            reason = escape(answer.reason or "", quote=False)
            await self._refuse(chat_id, message_id, BLOCKED.format(reason=reason))
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

    async def _voice(
        self, chat_id: int, message_id: int, audio: Mapping[str, Any]
    ) -> None:
        if self._transcriber is None:
            await self._send(chat_id, NO_VOICE)
            return
        duration = audio.get("duration")
        file_id = audio.get("file_id")
        if isinstance(duration, int) and duration > self._config.max_voice_seconds:
            await self._send(
                chat_id, VOICE_TOO_LONG.format(limit=self._config.max_voice_seconds)
            )
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
            names = ", ".join(kind.value for kind in kinds(findings))
            await self._refuse(
                chat_id, message_id, VOICE_BLOCKED.format(kinds=escape(names))
            )
            logger.info("Nota de voz bloqueada: %s", names)
            return
        limit = self._config.max_question_chars
        if len(heard) > limit:
            await self._refuse(
                chat_id,
                message_id,
                VOICE_TRANSCRIPT_TOO_LONG.format(length=len(heard), limit=limit),
            )
            logger.info("Nota de voz larga borrada sin procesar.")
            return
        await self._send(chat_id, HEARD.format(text=escape(heard, quote=False)))
        await self._question(chat_id, message_id, heard)

    async def _refuse(self, chat_id: int, message_id: int, html: str) -> None:
        """Borra el mensaje y responde; si no pudo borrarlo, pide hacerlo."""
        deleted = await self._delete(chat_id, message_id)
        await self._send(chat_id, f"{html}\n{DELETED if deleted else NOT_DELETED}")

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
            logger.warning("No se pudo borrar un mensaje (%s).", exc)
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
    # En el contenedor el bot es el proceso 1, que ignora SIGTERM por omisión:
    # sin esto, cada redespliegue espera hasta que lo maten a la fuerza.
    task = asyncio.current_task()
    if task is not None:
        with contextlib.suppress(NotImplementedError, RuntimeError):
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
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
    except asyncio.CancelledError:
        logger.info("Bot detenido.")
        return 0
    finally:
        await api.aclose()
    return 0
