"""Contratos de datos del módulo de evidencia (KARE-E).

Dos familias: lo que el modelo debe devolver, validado con Pydantic antes de
usarse, y lo que el sistema entrega. El modelo nunca decide el estado final de
una respuesta; lo deciden reglas sobre estos objetos.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

Language = Literal["es", "en"]
VerifierLabel = Literal["supported", "partial", "unsupported"]
ClaimLabel = Literal["supported", "partial", "unsupported", "unverified"]


class Passage(BaseModel):
    """Fragmento del corpus entregado al generador, con su procedencia."""

    chunk_id: str
    source_id: str
    source_title: str
    source_year: int | None = None
    page: int | None = None
    language: Language
    text: str
    score: float = 0.0
    coverage: float = 0.0


# --- Salidas del modelo ---------------------------------------------------------


class QueryPlan(BaseModel):
    """Consultas léxicas cortas para BM25, en español y en inglés."""

    queries_es: list[str] = Field(default_factory=list)
    queries_en: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)


class DraftAnswer(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    abstain: bool = False
    abstain_reason: str | None = None


class SupportVerdict(BaseModel):
    claim_index: int = Field(ge=1)
    label: VerifierLabel
    rationale: str = ""


class VerificationReport(BaseModel):
    verdicts: list[SupportVerdict] = Field(default_factory=list)


# --- Resultado --------------------------------------------------------------------


class Stage(StrEnum):
    """Etapa que decidió la suerte de una afirmación."""

    CITATIONS = "citas"
    NUMBERS = "cifras"
    VERIFIER = "verificador"


class VerifiedClaim(BaseModel):
    text: str
    chunk_ids: list[str] = Field(default_factory=list)
    label: ClaimLabel
    decided_by: Stage | None = None
    rationale: str = ""
    # Gancho para la factualidad conforme (P4): un puntaje de soporte que se
    # pueda calibrar sobre el banco de evaluación. En v0 queda vacío.
    support_score: float | None = None


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    ABSTAINED = "abstained"
    BLOCKED = "blocked"


class FinalAnswer(BaseModel):
    question: str
    status: AnswerStatus
    claims: list[VerifiedClaim] = Field(default_factory=list)
    discarded: list[VerifiedClaim] = Field(default_factory=list)
    passages: list[Passage] = Field(default_factory=list)
    # Texto fijo escrito por el sistema; nunca texto libre del modelo.
    reason: str | None = None
    # Lo que el modelo dijo al abstenerse. Sin verificar: no se muestra como
    # respuesta; queda para depurar y para el análisis de la evaluación.
    model_note: str | None = None
    model: str | None = None
    # Solo los tipos de identificador; nunca el valor encontrado.
    blocked_kinds: list[str] = Field(default_factory=list)
    error: str | None = None
