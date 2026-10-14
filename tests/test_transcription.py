from pathlib import Path

import httpx
import pytest

from app.core.config import Settings
from app.transcription import (
    ElevenLabsTranscriber,
    TranscriptionError,
    WhisperTranscriber,
    get_transcriber,
    load_keyterms,
    transcribe_safely,
)


async def test_elevenlabs_request_carries_model_language_keyterms_and_file(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "pregunta.m4a"
    audio.write_bytes(b"audio-de-prueba")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, json={"text": "  ¿Dosis de morfina en falla renal?  "}
        )

    transcriber = ElevenLabsTranscriber(
        api_key="clave",
        keyterms=["morfina", "sedación paliativa", "malo[x]"],
        transport=httpx.MockTransport(handler),
    )
    assert (
        await transcriber.transcribe(str(audio)) == "¿Dosis de morfina en falla renal?"
    )
    request = seen[0]
    assert request.url.path == "/v1/speech-to-text"
    assert request.headers["xi-api-key"] == "clave"
    body = request.content.decode("utf-8", errors="replace")
    assert 'name="model_id"' in body and "scribe_v2" in body
    assert 'name="language_code"' in body
    assert body.count('name="keyterms"') == 2
    assert "sedación paliativa" in body
    assert "malo[x]" not in body
    assert 'filename="pregunta.m4a"' in body


def test_load_keyterms_filters_what_the_api_would_reject(tmp_path: Path) -> None:
    path = tmp_path / "terminos.txt"
    path.write_text(
        "# comentario\n"
        "morfina\n"
        "morfina\n"
        "rotación de opioides\n"
        "una frase de más de cinco palabras aquí\n"
        "con {llaves}\n" + "x" * 60 + "\n",
        encoding="utf-8",
    )
    assert load_keyterms(path) == ["morfina", "rotación de opioides"]
    assert load_keyterms(tmp_path / "no-existe.txt") == []


async def test_transcribe_safely_never_raises() -> None:
    class Broken:
        async def transcribe(self, audio_path: str) -> str:
            raise RuntimeError("sin red")

    assert await transcribe_safely(Broken(), "nota.m4a") is None


def test_whisper_fallback_never_sends_another_providers_key() -> None:
    # Hallazgo de la revisión del bot: con OPENAI_API_KEY de OpenRouter, el
    # respaldo mandaba esa clave y el audio a api.openai.com.
    # Cada campo que decide va explícito: así ni el entorno ni un .env real
    # cambian el resultado.
    whisper = "https://api.openai.com/v1"
    elsewhere = Settings(
        elevenlabs_api_key="",
        openai_api_key="clave-de-otro-proveedor",
        openai_base_url="https://openrouter.example/api/v1",
        whisper_base_url=whisper,
    )
    assert get_transcriber(elsewhere) is None
    openai = Settings(
        elevenlabs_api_key="",
        openai_api_key="clave-de-openai",
        openai_base_url=whisper,
        whisper_base_url=whisper,
    )
    assert isinstance(get_transcriber(openai), WhisperTranscriber)


async def test_transcription_errors_do_not_copy_the_response_body(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "pregunta.m4a"
    audio.write_bytes(b"audio-de-prueba")
    leaked = "transcripcion sintetica"

    def handler(request: httpx.Request) -> httpx.Response:
        detail = {"status": "invalid", "message": "bad audio", "echo": leaked}
        return httpx.Response(422, json={"detail": detail})

    transcriber = ElevenLabsTranscriber(
        api_key="clave", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(TranscriptionError) as failed:
        await transcriber.transcribe(str(audio))
    assert "HTTP 422 (invalid: bad audio)" in str(failed.value)
    assert leaked not in str(failed.value)


async def test_zero_retention_is_opt_in(tmp_path: Path) -> None:
    audio = tmp_path / "pregunta.m4a"
    audio.write_bytes(b"audio-de-prueba")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "hola"})

    for zero_retention in (False, True):
        transcriber = ElevenLabsTranscriber(
            api_key="clave",
            zero_retention=zero_retention,
            transport=httpx.MockTransport(handler),
        )
        await transcriber.transcribe(str(audio))
    assert "enable_logging" not in seen[0].url.params
    assert seen[1].url.params["enable_logging"] == "false"
