"""Comparador del ensayo: el mismo modelo, sin corpus y sin verificación.

Sirve para medir cuánto aporta KARE-E frente a preguntarle al modelo a secas.
Comparte la guardia: tampoco envía preguntas con datos de una persona.
"""

import logging

from app.answer import prompts
from app.answer.numbers import strip_citation_marks
from app.guard.identifiers import describe, kinds, redact, scan
from app.llm.client import LLMClient, LLMError
from app.schemas.evidence import AnswerStatus, DraftAnswer, FinalAnswer, VerifiedClaim

logger = logging.getLogger(__name__)

MAX_CLAIMS = 8


class BaselineAnswerer:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def answer(self, question: str) -> FinalAnswer:
        question = question.strip()
        findings = scan(question)
        if findings:
            return FinalAnswer(
                question=redact(question, findings),
                status=AnswerStatus.BLOCKED,
                reason=describe(findings),
                blocked_kinds=[kind.value for kind in kinds(findings)],
            )
        model = self._llm.model_name
        try:
            draft = await self._llm.complete_json(
                system=prompts.BASELINE_SYSTEM,
                user=prompts.baseline_user(question),
                schema=DraftAnswer,
            )
        except LLMError as exc:
            logger.error("Falló el comparador: %s", exc)
            return FinalAnswer(
                question=question,
                status=AnswerStatus.ABSTAINED,
                reason="Error del modelo.",
                model=model,
                error=str(exc),
            )
        texts = [
            strip_citation_marks(claim.text) for claim in draft.claims[:MAX_CLAIMS]
        ]
        claims = [
            VerifiedClaim(text=text, label="unverified") for text in texts if text
        ]
        reason = (draft.abstain_reason or "").strip() or None
        if not claims:
            return FinalAnswer(
                question=question,
                status=AnswerStatus.ABSTAINED,
                reason=reason or "El modelo se abstuvo.",
                model=model,
            )
        return FinalAnswer(
            question=question,
            status=AnswerStatus.PARTIAL if draft.abstain else AnswerStatus.ANSWERED,
            claims=claims,
            reason=reason,
            model=model,
        )
