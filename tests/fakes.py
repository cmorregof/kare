"""Dobles de prueba: un modelo con respuestas guionadas y textos sintéticos.

Los textos del corpus de prueba son inventados y lo dicen; no son guías.
"""

from collections.abc import Mapping, Sequence
from typing import TypeVar

from pydantic import BaseModel

from app.corpus.store import SourceInfo, StoredChunk
from app.llm.client import LLMError
from app.retrieval.retriever import Retriever
from app.retrieval.text import analyze
from app.schemas.evidence import FinalAnswer, Language

T = TypeVar("T", bound=BaseModel)


class FakeLLM:
    def __init__(
        self,
        script: Mapping[type[BaseModel], Sequence[BaseModel | Exception]],
        model_name: str = "modelo-falso",
    ) -> None:
        self._script = {schema: list(items) for schema, items in script.items()}
        self._model = model_name
        self.calls: list[tuple[str, str, str]] = []

    @property
    def model_name(self) -> str:
        return self._model

    def calls_for(self, schema: type[BaseModel]) -> list[tuple[str, str, str]]:
        return [call for call in self.calls if call[0] == schema.__name__]

    async def complete_json(self, *, system: str, user: str, schema: type[T]) -> T:
        self.calls.append((schema.__name__, system, user))
        queue = self._script.get(schema)
        if not queue:
            raise LLMError(f"sin respuesta guionada para {schema.__name__}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return schema.model_validate(item.model_dump())


class StaticAnswerer:
    """Devuelve respuestas fijas por pregunta (para el arnés de evaluación)."""

    def __init__(self, answers: Mapping[str, FinalAnswer]) -> None:
        self._answers = dict(answers)

    async def answer(self, question: str) -> FinalAnswer:
        return self._answers[question]


SOURCES = {
    "guia-a": SourceInfo(
        "guia-a", "Guía sintética A", "Pruebas", 2024, "es", "abierta", None
    ),
    "guia-b": SourceInfo(
        "guia-b", "Synthetic guideline B", "Tests", 2023, "en", "abierta", None
    ),
    "norma-c": SourceInfo(
        "norma-c", "Norma sintética C", "Pruebas", 2018, "es", "abierta", None
    ),
}

_CHUNKS: list[tuple[str, str, int | None, Language, str]] = [
    (
        "guia-a:p12:c0",
        "guia-a",
        12,
        "es",
        "Texto sintético para pruebas. En pacientes con insuficiencia renal grave se "
        "prefieren opioides como la hidromorfona o el fentanilo y se recomienda evitar "
        "la morfina.",
    ),
    (
        "guia-a:p30:c1",
        "guia-a",
        30,
        "es",
        "Texto sintético para pruebas. La sedación paliativa se indica ante síntomas "
        "refractarios en los últimos días de vida, con consentimiento del paciente o "
        "de su representante.",
    ),
    (
        "guia-a:p14:c2",
        "guia-a",
        14,
        "es",
        "Texto sintético para pruebas. La dosis inicial de hidromorfona oral descrita "
        "en este ejemplo es de 2 mg cada 4 horas.",
    ),
    (
        "guia-b:p5:c0",
        "guia-b",
        5,
        "en",
        "Synthetic test text. In advanced cancer, opioids can reduce breathlessness, "
        "and a fan directed to the face is a useful non-pharmacological measure.",
    ),
    (
        "norma-c:c0",
        "norma-c",
        None,
        "es",
        "Texto sintético para pruebas. El documento de voluntad anticipada permite "
        "dejar por escrito las decisiones sobre los tratamientos que una persona desea "
        "o no recibir al final de la vida.",
    ),
]


def synthetic_chunks() -> list[StoredChunk]:
    return [
        StoredChunk(
            chunk_id=cid,
            source_id=source,
            page=page,
            ordinal=index,
            language=language,
            text=text,
            terms=tuple(analyze(text, language)),
        )
        for index, (cid, source, page, language, text) in enumerate(_CHUNKS)
    ]


def synthetic_retriever() -> Retriever:
    return Retriever(synthetic_chunks(), SOURCES)
