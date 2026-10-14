"""Transcripción de preguntas dictadas.

Solo para preguntas de la médica sin datos de pacientes: el audio sale al
proveedor antes de que la guardia de identificadores pueda leerlo. La guardia
corre después sobre el texto, pero el audio ya salió. Audio con pacientes
pertenece a la zona clínica (modelo local o proveedor con retención cero y
contrato), que este repositorio no implementa.

Mismo patrón que insight_engine: interfaz `Transcriber`, implementación
intercambiable, y fallas no fatales para quien llama (`transcribe_safely`).
"""

import logging
import mimetypes
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx

from app.core.config import Settings
from app.core.http_errors import describe_error_response

logger = logging.getLogger(__name__)

# Reglas de ElevenLabs para keyterms (documentación de la API, oct 2026).
MAX_KEYTERMS = 1000
_KEYTERM_MAX_CHARS = 50
_KEYTERM_MAX_WORDS = 5
_KEYTERM_FORBIDDEN = frozenset("<>{}[]\\")


class TranscriptionError(RuntimeError):
    pass


@runtime_checkable
class Transcriber(Protocol):
    async def transcribe(self, audio_path: str) -> str: ...


def is_valid_keyterm(term: str) -> bool:
    return (
        0 < len(term) < _KEYTERM_MAX_CHARS
        and len(term.split()) <= _KEYTERM_MAX_WORDS
        and not any(char in _KEYTERM_FORBIDDEN for char in term)
    )


def load_keyterms(path: str | Path) -> list[str]:
    """Un término por línea; '#' comenta. Descarta los que la API rechazaría."""
    file = Path(path)
    if not file.exists():
        return []
    terms: list[str] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if not term or term.startswith("#"):
            continue
        if not is_valid_keyterm(term):
            logger.warning(
                "Término omitido (no cumple las reglas de keyterms): %r", term
            )
            continue
        terms.append(term)
    unique = list(dict.fromkeys(terms))
    if len(unique) > MAX_KEYTERMS:
        logger.warning(
            "Hay %d términos; se usan los primeros %d.", len(unique), MAX_KEYTERMS
        )
    return unique[:MAX_KEYTERMS]


def _audio_part(path: Path) -> tuple[str, bytes, str]:
    if not path.exists():
        raise TranscriptionError(f"No existe el archivo de audio {path}.")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return path.name, path.read_bytes(), mime


class ElevenLabsTranscriber:
    """Speech to Text de ElevenLabs (Scribe), POST /v1/speech-to-text."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "scribe_v2",
        language: str = "es",
        keyterms: Sequence[str] = (),
        base_url: str = "https://api.elevenlabs.io",
        timeout: float = 120.0,
        zero_retention: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Falta ELEVENLABS_API_KEY.")
        self._zero_retention = zero_retention
        self._api_key = api_key
        self._model = model
        self._language = language
        self._keyterms = [term for term in keyterms if is_valid_keyterm(term)][
            :MAX_KEYTERMS
        ]
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport

    async def transcribe(self, audio_path: str) -> str:
        data: dict[str, str | list[str]] = {
            "model_id": self._model,
            "language_code": self._language,
            "tag_audio_events": "false",
            "diarize": "false",
        }
        if self._keyterms:
            data["keyterms"] = list(self._keyterms)
        async with httpx.AsyncClient(
            timeout=self._timeout, transport=self._transport
        ) as client:
            response = await client.post(
                f"{self._base_url}/v1/speech-to-text",
                params={"enable_logging": "false"} if self._zero_retention else None,
                headers={"xi-api-key": self._api_key},
                data=data,
                files={"file": _audio_part(Path(audio_path))},
            )
        if response.status_code != 200:
            raise TranscriptionError(
                f"ElevenLabs respondió {describe_error_response(response)}"
            )
        text = response.json().get("text")
        if not isinstance(text, str):
            raise TranscriptionError("ElevenLabs: respuesta sin texto.")
        return text.strip()


class WhisperTranscriber:
    """Whisper de OpenAI o un servidor compatible (/audio/transcriptions)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "whisper-1",
        language: str = "es",
        prompt: str = "",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._language = language
        self._prompt = prompt
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport

    async def transcribe(self, audio_path: str) -> str:
        data = {"model": self._model, "language": self._language}
        if self._prompt:
            data["prompt"] = self._prompt
        headers = {"authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        async with httpx.AsyncClient(
            timeout=self._timeout, transport=self._transport
        ) as client:
            response = await client.post(
                f"{self._base_url}/audio/transcriptions",
                headers=headers,
                data=data,
                files={"file": _audio_part(Path(audio_path))},
            )
        if response.status_code != 200:
            raise TranscriptionError(
                f"Whisper respondió {describe_error_response(response)}"
            )
        text = response.json().get("text")
        if not isinstance(text, str):
            raise TranscriptionError("Whisper: respuesta sin texto.")
        return text.strip()


def _same_host(first: str, second: str) -> bool:
    return httpx.URL(first).host == httpx.URL(second).host


def get_transcriber(settings: Settings) -> Transcriber | None:
    """ElevenLabs si hay clave; si no, Whisper; si no, None."""
    keyterms = load_keyterms(settings.keyterms_path)
    if settings.elevenlabs_api_key:
        return ElevenLabsTranscriber(
            api_key=settings.elevenlabs_api_key,
            model=settings.elevenlabs_stt_model,
            language=settings.transcription_language,
            keyterms=keyterms,
            base_url=settings.elevenlabs_base_url,
            timeout=settings.transcription_timeout_seconds,
            zero_retention=settings.elevenlabs_zero_retention,
        )
    if settings.openai_api_key and _same_host(
        settings.openai_base_url, settings.whisper_base_url
    ):
        # Solo si la clave es del mismo servidor de Whisper: OPENAI_API_KEY puede
        # ser de OpenRouter o Groq, y mandarla (con el audio) a OpenAI sería
        # entregarle la clave y la nota a un tercero.
        # Whisper admite un prompt corto con vocabulario; unos pocos términos bastan.
        return WhisperTranscriber(
            api_key=settings.openai_api_key,
            model=settings.whisper_model,
            language=settings.transcription_language,
            prompt=", ".join(keyterms[:40]),
            base_url=settings.whisper_base_url,
            timeout=settings.transcription_timeout_seconds,
        )
    return None


async def transcribe_safely(transcriber: Transcriber, audio_path: str) -> str | None:
    """Nunca lanza: devuelve None y deja la falla en el registro."""
    try:
        return await transcriber.transcribe(audio_path)
    except Exception as exc:
        logger.error("Falló la transcripción de %s: %s", audio_path, exc)
        return None
