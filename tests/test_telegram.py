"""Bot de Telegram con la API simulada. Todos los textos son sintéticos."""

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.answer.pipeline import EvidencePipeline
from app.core.config import parse_user_ids
from app.schemas.evidence import (
    AnswerStatus,
    FinalAnswer,
    Passage,
    VerifiedClaim,
)
from app.telegram.api import MESSAGE_LIMIT, TelegramAPI, TelegramError
from app.telegram.bot import BotConfig, KareBot
from app.telegram.format import render_telegram
from tests.fakes import FakeLLM, synthetic_retriever

TOKEN = "123456:TOKEN-SINTETICO-DE-PRUEBA"
ALLOWED = 1001
STRANGER = 2002
SYNTHETIC_ID = "1053456789"  # número inventado, como en las pruebas de la guardia


class FakeTelegram:
    def __init__(self, *, reject_html: bool = False, fail: str | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.reject_html = reject_html
        self.fail = fail

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith(f"/file/bot{TOKEN}/"):
            return httpx.Response(200, content=b"audio sintetico")
        method = path.rsplit("/", 1)[-1]
        payload = json.loads(request.content or b"{}")
        self.calls.append((method, payload))
        if method == self.fail:
            return httpx.Response(
                401, json={"ok": False, "description": "Unauthorized"}
            )
        if self.reject_html and method == "sendMessage" and "parse_mode" in payload:
            return httpx.Response(
                400,
                json={"ok": False, "description": "Bad Request: can't parse entities"},
            )
        if method == "getFile":
            result: Any = {"file_path": "voice/nota.oga", "file_size": 15}
        else:
            result = True
        return httpx.Response(200, json={"ok": True, "result": result})

    def api(self) -> TelegramAPI:
        return TelegramAPI(TOKEN, transport=httpx.MockTransport(self.handler))

    def sent(self) -> list[str]:
        return [
            payload["text"] for method, payload in self.calls if method == "sendMessage"
        ]

    def methods(self) -> list[str]:
        return [method for method, _ in self.calls]


class RecordingAnswerer:
    def __init__(self, answer: FinalAnswer | Exception) -> None:
        self._answer = answer
        self.questions: list[str] = []

    async def answer(self, question: str) -> FinalAnswer:
        self.questions.append(question)
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer


class FakeTranscriber:
    def __init__(self, text: str) -> None:
        self._text = text
        self.audio: list[bytes] = []

    async def transcribe(self, audio_path: str) -> str:
        self.audio.append(Path(audio_path).read_bytes())
        return self._text


PASSAGE = Passage(
    chunk_id="guia-a:p12:c0",
    source_id="guia-a",
    source_title="Guía sintética A",
    source_year=2024,
    page=12,
    language="es",
    text="Texto sintético para pruebas sobre opioides en insuficiencia renal.",
)

ANSWER = FinalAnswer(
    question="¿Qué opioide se prefiere en insuficiencia renal grave?",
    status=AnswerStatus.ANSWERED,
    claims=[
        VerifiedClaim(
            text="Texto sintético: dosis <ajustada> & vigilada.",
            chunk_ids=["guia-a:p12:c0"],
            label="supported",
        )
    ],
    passages=[PASSAGE],
    model="modelo-falso",
)


def message(
    text: str | None = None,
    *,
    user_id: int = ALLOWED,
    chat_type: str = "private",
    voice: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "message_id": 7,
        "chat": {"id": user_id, "type": chat_type},
        "from": {"id": user_id},
    }
    if text is not None:
        body["text"] = text
    if voice is not None:
        body["voice"] = voice
    return {"update_id": 1, "message": body}


def bot(
    telegram: FakeTelegram,
    answerer: Any,
    *,
    transcriber: FakeTranscriber | None = None,
) -> KareBot:
    return KareBot(
        telegram.api(),
        answerer,
        BotConfig(allowed_users=frozenset({ALLOWED}), max_question_chars=200),
        sources_text="<b>Fuentes indexadas</b>",
        transcriber=transcriber,
    )


async def test_strangers_only_learn_their_own_id() -> None:
    telegram = FakeTelegram()
    answerer = RecordingAnswerer(ANSWER)
    await bot(telegram, answerer).handle_update(
        message("¿Qué opioide se prefiere?", user_id=STRANGER)
    )
    assert answerer.questions == []
    assert telegram.methods() == ["sendMessage"]
    assert str(STRANGER) in telegram.sent()[0]
    assert "privado" in telegram.sent()[0]


async def test_anyone_can_ask_for_their_id_and_groups_are_ignored() -> None:
    telegram = FakeTelegram()
    kare = bot(telegram, RecordingAnswerer(ANSWER))
    await kare.handle_update(message("/id@kare_bot", user_id=STRANGER))
    assert telegram.sent() == [f"Tu ID de Telegram es {STRANGER}."]
    await kare.handle_update(message("¿Pregunta?", chat_type="group"))
    assert len(telegram.calls) == 1


async def test_a_question_gets_html_claims_and_folded_sources() -> None:
    telegram = FakeTelegram()
    answerer = RecordingAnswerer(ANSWER)
    await bot(telegram, answerer).handle_update(message("  ¿Qué opioide?  "))
    assert answerer.questions == ["¿Qué opioide?"]
    assert "sendChatAction" in telegram.methods()
    first, sources = telegram.sent()
    assert first.startswith("<b>Respuesta con evidencia citada</b>")
    assert "dosis &lt;ajustada&gt; &amp; vigilada. [1]" in first
    assert "verifica cada afirmación" in first
    assert sources.startswith("<b>Fuentes</b>\n<blockquote expandable>[1] Guía")
    assert sources.endswith("</blockquote>")


async def test_identifiers_are_deleted_and_never_repeated() -> None:
    telegram = FakeTelegram()
    llm = FakeLLM({})
    pipeline = EvidencePipeline(synthetic_retriever(), llm)
    await bot(telegram, pipeline).handle_update(
        message(f"¿Qué opioide para la paciente con CC {SYNTHETIC_ID}?")
    )
    assert llm.calls == []
    assert ("deleteMessage", {"chat_id": ALLOWED, "message_id": 7}) in telegram.calls
    [reply] = telegram.sent()
    assert reply.startswith("<b>Pregunta bloqueada</b>")
    assert "Borré tu mensaje" in reply
    assert SYNTHETIC_ID not in reply


async def test_long_texts_are_not_processed() -> None:
    telegram = FakeTelegram()
    answerer = RecordingAnswerer(ANSWER)
    kare = bot(telegram, answerer)
    await kare.handle_update(message("Nota clínica sintética. " * 20))
    assert answerer.questions == []
    assert "muy larga" in telegram.sent()[-1]
    assert "deleteMessage" not in telegram.methods()
    await kare.handle_update(message("Nota sintética. " * 15 + f"CC {SYNTHETIC_ID}"))
    assert answerer.questions == []
    assert "deleteMessage" in telegram.methods()
    assert SYNTHETIC_ID not in telegram.sent()[-1]


async def test_voice_needs_a_transcriber() -> None:
    telegram = FakeTelegram()
    voice = {"file_id": "f1", "duration": 5, "mime_type": "audio/ogg"}
    await bot(telegram, RecordingAnswerer(ANSWER)).handle_update(message(voice=voice))
    assert telegram.sent() == [
        "Las notas de voz no están activadas. Escribe la pregunta."
    ]


async def test_a_dictated_question_is_transcribed_and_answered() -> None:
    telegram = FakeTelegram()
    answerer = RecordingAnswerer(ANSWER)
    heard = "¿Qué opioide se prefiere en insuficiencia renal grave?"
    transcriber = FakeTranscriber(heard)
    voice = {"file_id": "f1", "duration": 5, "mime_type": "audio/ogg"}
    await bot(telegram, answerer, transcriber=transcriber).handle_update(
        message(voice=voice)
    )
    assert transcriber.audio == [b"audio sintetico"]
    assert answerer.questions == [heard]
    assert telegram.sent()[0] == f"Escuché: «{heard}»"


async def test_a_dictated_identifier_is_deleted_and_not_echoed() -> None:
    telegram = FakeTelegram()
    answerer = RecordingAnswerer(ANSWER)
    transcriber = FakeTranscriber(f"La paciente con cédula {SYNTHETIC_ID} tiene dolor")
    voice = {"file_id": "f1", "duration": 5, "mime_type": "audio/ogg"}
    await bot(telegram, answerer, transcriber=transcriber).handle_update(
        message(voice=voice)
    )
    assert answerer.questions == []
    assert "deleteMessage" in telegram.methods()
    [reply] = telegram.sent()
    assert reply.startswith("<b>Nota de voz bloqueada</b>")
    assert SYNTHETIC_ID not in reply


async def test_long_voice_notes_are_refused_before_downloading() -> None:
    telegram = FakeTelegram()
    transcriber = FakeTranscriber("no debería oírse")
    voice = {"file_id": "f1", "duration": 600, "mime_type": "audio/ogg"}
    await bot(
        telegram, RecordingAnswerer(ANSWER), transcriber=transcriber
    ).handle_update(message(voice=voice))
    assert "getFile" not in telegram.methods()
    assert transcriber.audio == []


async def test_rejected_html_is_resent_as_plain_text() -> None:
    telegram = FakeTelegram(reject_html=True)
    await bot(telegram, RecordingAnswerer(ANSWER)).handle_update(message("¿Opioide?"))
    plain = [
        p for m, p in telegram.calls if m == "sendMessage" and "parse_mode" not in p
    ]
    assert plain[0]["text"].startswith("Respuesta con evidencia citada")
    assert "dosis <ajustada> & vigilada." in plain[0]["text"]


async def test_a_failure_is_reported_without_logging_the_question(
    caplog: pytest.LogCaptureFixture,
) -> None:
    telegram = FakeTelegram()
    answerer = RecordingAnswerer(RuntimeError("texto sintético que no debe salir"))
    with caplog.at_level(logging.INFO):
        await bot(telegram, answerer).handle_update(message("¿Pregunta sintética?"))
    assert "error interno" in telegram.sent()[-1]
    assert "Pregunta sintética" not in caplog.text
    assert "no debe salir" not in caplog.text
    assert "RuntimeError" in caplog.text


async def test_api_errors_never_carry_the_token() -> None:
    telegram = FakeTelegram(fail="sendMessage")
    with pytest.raises(TelegramError) as rejected:
        await telegram.api().send_message(ALLOWED, "hola")
    assert TOKEN not in str(rejected.value)
    assert "Unauthorized" in str(rejected.value)

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"no se pudo conectar a {request.url}")

    api = TelegramAPI(TOKEN, transport=httpx.MockTransport(unreachable))
    with pytest.raises(TelegramError) as failed:
        await api.call("getMe")
    assert TOKEN not in str(failed.value)
    assert failed.value.__cause__ is None
    assert failed.value.__suppress_context__


def test_long_answers_are_split_without_breaking_the_markup() -> None:
    claims = [
        VerifiedClaim(
            text=f"Afirmación sintética {n} <{n}>. " + "texto " * 100,
            chunk_ids=[f"guia-a:p{n}:c0"],
            label="supported",
        )
        for n in range(1, 9)
    ]
    passages = [
        PASSAGE.model_copy(
            update={
                "chunk_id": f"guia-a:p{n}:c0",
                "page": n,
                "text": "Pasaje sintético & largo. " * 60,
            }
        )
        for n in range(1, 9)
    ]
    answer = ANSWER.model_copy(update={"claims": claims, "passages": passages})
    parts = render_telegram(answer, snippet_chars=900)
    assert len(parts) >= 4
    assert all(len(part) <= MESSAGE_LIMIT for part in parts)
    sources = [part for part in parts if part.startswith("<b>Fuentes</b>")]
    assert len(sources) > 1
    for part in sources:
        assert part.count("<blockquote expandable>") == part.count("</blockquote>") == 1


def test_allowed_users_are_parsed_strictly() -> None:
    assert parse_user_ids(" 123, 456;789 ,") == frozenset({123, 456, 789})
    assert parse_user_ids("") == frozenset()
    with pytest.raises(ValueError):
        parse_user_ids("123, @karen")
