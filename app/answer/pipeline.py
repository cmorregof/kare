"""Pipeline de evidencia citada con abstención (P3).

Orden fijo, sin agentes:

1. Guardia de identificadores: si hay datos de una persona, se bloquea y no
   sale nada del equipo.
2. Plan de búsqueda: el modelo propone consultas en español e inglés; la
   pregunta literal siempre se busca también.
3. Recuperación BM25 por idioma con fusión RRF y umbral de cobertura. Sin
   pasajes pertinentes, abstención sin llamar al generador.
4. Borrador: afirmaciones con citas a los pasajes [P1]..[Pn].
5. Chequeos deterministas: citas que existan y cifras presentes en los
   pasajes citados.
6. Verificación independiente: otro llamado, que solo ve cada afirmación con
   sus pasajes, sin la pregunta ni el resto de la respuesta.
7. Estado final por reglas. Si la verificación falla, no se muestra nada.

El modelo transforma texto; decidir qué se muestra y qué se calla es de las
reglas.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.answer import prompts
from app.answer.numbers import missing_numbers, strip_citation_marks
from app.guard.identifiers import describe, kinds, redact, scan
from app.llm.client import LLMClient, LLMError
from app.retrieval.retriever import Query, Retriever
from app.schemas.evidence import (
    AnswerStatus,
    DraftAnswer,
    FinalAnswer,
    Passage,
    QueryPlan,
    Stage,
    SupportVerdict,
    VerificationReport,
    VerifiedClaim,
)

logger = logging.getLogger(__name__)

_LABEL_NUMBER = re.compile(r"\d+")

NO_PASSAGES = "No encontré pasajes pertinentes en el corpus para esta pregunta."
NO_MODEL = "No hay modelo configurado: solo se muestran los pasajes recuperados."
NO_SUPPORT = "Ninguna afirmación quedó sustentada por los pasajes del corpus."
NOT_COVERED = "El corpus no responde esta pregunta con suficiente respaldo."
PARTLY_COVERED = "Solo una parte de la pregunta quedó sustentada por el corpus."
GENERATION_FAILED = "No se pudo generar una respuesta; intenta de nuevo."
VERIFICATION_FAILED = (
    "No se pudo verificar la respuesta; por seguridad no se muestra. Intenta de nuevo."
)

# Si el verificador repite una afirmación con veredictos distintos, gana el
# más severo.
_SEVERITY = {"supported": 0, "partial": 1, "unsupported": 2}


@dataclass(frozen=True)
class PipelineConfig:
    top_k: int = 8
    per_query_k: int = 20
    min_coverage: float = 0.3
    keep_partial: bool = False
    max_claims: int = 8
    max_queries_per_language: int = 3


@dataclass(frozen=True)
class _Candidate:
    text: str
    passages: tuple[Passage, ...]

    @property
    def chunk_ids(self) -> list[str]:
        return [passage.chunk_id for passage in self.passages]


def parse_labels(citations: Sequence[str]) -> list[str]:
    """Normaliza "P2", "[P2]", "p 2" o "2" a "P2", sin repetir."""
    labels: list[str] = []
    for citation in citations:
        labels.extend(f"P{number}" for number in _LABEL_NUMBER.findall(citation))
    return list(dict.fromkeys(labels))


class EvidencePipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: LLMClient | None,
        verifier: LLMClient | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._verifier = verifier or generator
        self._config = config or PipelineConfig()

    async def answer(self, question: str) -> FinalAnswer:
        question = question.strip()
        if not question:
            return FinalAnswer(
                question="",
                status=AnswerStatus.ABSTAINED,
                reason="La pregunta está vacía.",
            )

        findings = scan(question)
        if findings:
            return FinalAnswer(
                question=redact(question, findings),
                status=AnswerStatus.BLOCKED,
                reason=describe(findings),
                blocked_kinds=[kind.value for kind in kinds(findings)],
            )

        queries = await self._plan(question)
        passages = self._retriever.search(
            queries,
            top_k=self._config.top_k,
            per_query_k=self._config.per_query_k,
            min_coverage=self._config.min_coverage,
        )
        if not passages:
            return FinalAnswer(
                question=question, status=AnswerStatus.ABSTAINED, reason=NO_PASSAGES
            )
        if self._generator is None:
            return FinalAnswer(
                question=question,
                status=AnswerStatus.ABSTAINED,
                reason=NO_MODEL,
                passages=passages,
            )

        model = self._generator.model_name
        try:
            draft = await self._generator.complete_json(
                system=prompts.GENERATOR_SYSTEM,
                user=prompts.generator_user(question, passages),
                schema=DraftAnswer,
            )
        except LLMError as exc:
            logger.error("Falló la generación: %s", exc)
            return FinalAnswer(
                question=question,
                status=AnswerStatus.ABSTAINED,
                reason=GENERATION_FAILED,
                passages=passages,
                model=model,
                error=str(exc),
            )

        candidates, discarded = self._deterministic_checks(draft, passages)
        try:
            kept, rejected = await self._verify(candidates)
        except LLMError as exc:
            logger.error("Falló la verificación: %s", exc)
            return FinalAnswer(
                question=question,
                status=AnswerStatus.ABSTAINED,
                reason=VERIFICATION_FAILED,
                passages=passages,
                model=model,
                error=str(exc),
            )
        discarded.extend(rejected)
        return self._finalize(question, draft, kept, discarded, passages, model)

    async def _plan(self, question: str) -> list[Query]:
        queries = [Query("es", question)]
        if self._generator is None:
            return queries
        try:
            plan = await self._generator.complete_json(
                system=prompts.PLANNER_SYSTEM,
                user=prompts.planner_user(question),
                schema=QueryPlan,
            )
        except LLMError as exc:
            logger.warning(
                "Sin plan de búsqueda (%s); se usa la pregunta literal.", exc
            )
            return queries
        limit = self._config.max_queries_per_language
        queries += [Query("es", q) for q in plan.queries_es[:limit] if q.strip()]
        queries += [Query("en", q) for q in plan.queries_en[:limit] if q.strip()]
        return queries

    def _deterministic_checks(
        self, draft: DraftAnswer, passages: Sequence[Passage]
    ) -> tuple[list[_Candidate], list[VerifiedClaim]]:
        by_label = {f"P{number}": p for number, p in enumerate(passages, start=1)}
        candidates: list[_Candidate] = []
        discarded: list[VerifiedClaim] = []
        for index, claim in enumerate(draft.claims):
            text = strip_citation_marks(claim.text)
            if not text:
                continue
            if index >= self._config.max_claims:
                discarded.append(
                    VerifiedClaim(
                        text=text,
                        label="unsupported",
                        rationale="Excede el máximo de afirmaciones por respuesta.",
                    )
                )
                continue
            labels = parse_labels(claim.citations)
            unknown = [label for label in labels if label not in by_label]
            cited = tuple(by_label[label] for label in labels if label in by_label)
            if not cited or unknown:
                # Citar un pasaje que no existe delata una afirmación inventada,
                # aunque también cite uno real.
                rationale = (
                    f"Cita pasajes que no existen: {', '.join(unknown)}."
                    if unknown
                    else "No cita ninguno de los pasajes recuperados."
                )
                discarded.append(
                    VerifiedClaim(
                        text=text,
                        chunk_ids=[p.chunk_id for p in cited],
                        label="unsupported",
                        decided_by=Stage.CITATIONS,
                        rationale=rationale,
                    )
                )
                continue
            evidence = [p.text for p in cited] + [
                f"{p.source_title} {p.source_year or ''}" for p in cited
            ]
            missing = missing_numbers(text, evidence)
            if missing:
                discarded.append(
                    VerifiedClaim(
                        text=text,
                        chunk_ids=[p.chunk_id for p in cited],
                        label="unsupported",
                        decided_by=Stage.NUMBERS,
                        rationale="Cifras que no aparecen en los pasajes citados: "
                        + ", ".join(missing)
                        + ".",
                    )
                )
                continue
            candidates.append(_Candidate(text=text, passages=cited))
        return candidates, discarded

    async def _verify(
        self, candidates: Sequence[_Candidate]
    ) -> tuple[list[VerifiedClaim], list[VerifiedClaim]]:
        if not candidates:
            return [], []
        if self._verifier is None:
            raise LLMError("No hay verificador configurado.")
        report = await self._verifier.complete_json(
            system=prompts.VERIFIER_SYSTEM,
            user=prompts.verifier_user([(c.text, c.passages) for c in candidates]),
            schema=VerificationReport,
        )
        verdicts: dict[int, SupportVerdict] = {}
        for verdict in report.verdicts:
            current = verdicts.get(verdict.claim_index)
            if current is None or _SEVERITY[verdict.label] > _SEVERITY[current.label]:
                verdicts[verdict.claim_index] = verdict
        kept: list[VerifiedClaim] = []
        rejected: list[VerifiedClaim] = []
        for number, candidate in enumerate(candidates, start=1):
            found = verdicts.get(number)
            if found is None:
                rejected.append(
                    VerifiedClaim(
                        text=candidate.text,
                        chunk_ids=candidate.chunk_ids,
                        label="unsupported",
                        decided_by=Stage.VERIFIER,
                        rationale="El verificador no se pronunció sobre ella.",
                    )
                )
                continue
            claim = VerifiedClaim(
                text=candidate.text,
                chunk_ids=candidate.chunk_ids,
                label=found.label,
                decided_by=Stage.VERIFIER,
                rationale=found.rationale,
            )
            accepted = found.label == "supported" or (
                found.label == "partial" and self._config.keep_partial
            )
            (kept if accepted else rejected).append(claim)
        return kept, rejected

    @staticmethod
    def _finalize(
        question: str,
        draft: DraftAnswer,
        kept: list[VerifiedClaim],
        discarded: list[VerifiedClaim],
        passages: list[Passage],
        model: str,
    ) -> FinalAnswer:
        # El motivo que da el modelo es texto libre sin verificar: puede traer
        # justo la dosis inventada que el resto del pipeline bloquea. Se guarda
        # como nota y el usuario ve solo textos fijos del sistema.
        model_note = (draft.abstain_reason or "").strip() or None
        if not kept:
            return FinalAnswer(
                question=question,
                status=AnswerStatus.ABSTAINED,
                reason=NOT_COVERED if draft.abstain else NO_SUPPORT,
                model_note=model_note,
                discarded=discarded,
                passages=passages,
                model=model,
            )
        complete = (
            not discarded
            and not draft.abstain
            and all(claim.label == "supported" for claim in kept)
        )
        if complete:
            return FinalAnswer(
                question=question,
                status=AnswerStatus.ANSWERED,
                claims=kept,
                passages=passages,
                model_note=model_note,
                model=model,
            )
        notes = []
        if draft.abstain:
            notes.append(PARTLY_COVERED)
        if len(discarded) == 1:
            notes.append("Se omitió 1 afirmación sin soporte suficiente.")
        elif discarded:
            notes.append(
                f"Se omitieron {len(discarded)} afirmaciones sin soporte suficiente."
            )
        return FinalAnswer(
            question=question,
            status=AnswerStatus.PARTIAL,
            claims=kept,
            discarded=discarded,
            passages=passages,
            reason=" ".join(notes) or "Respuesta parcialmente sustentada.",
            model_note=model_note,
            model=model,
        )
