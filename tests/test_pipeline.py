from app.answer.pipeline import (
    NO_MODEL,
    NO_PASSAGES,
    NO_SUPPORT,
    NOT_COVERED,
    PARTLY_COVERED,
    VERIFICATION_FAILED,
    EvidencePipeline,
    PipelineConfig,
    parse_labels,
)
from app.answer.render import DISCLAIMER, render_text
from app.llm.client import LLMError
from app.schemas.evidence import (
    AnswerStatus,
    Claim,
    DraftAnswer,
    QueryPlan,
    Stage,
    SupportVerdict,
    VerificationReport,
)
from tests.fakes import FakeLLM, synthetic_retriever

RENAL_QUESTION = "¿Qué opioide se prefiere en insuficiencia renal grave?"
RENAL_PLAN = QueryPlan(
    queries_es=["opioide insuficiencia renal grave"],
    queries_en=["opioid renal impairment"],
)
RENAL_CHUNK = "guia-a:p12:c0"


def verdicts(*labels: str) -> VerificationReport:
    return VerificationReport(
        verdicts=[
            SupportVerdict(claim_index=i, label=label, rationale="ok")  # type: ignore[arg-type]
            for i, label in enumerate(labels, start=1)
        ]
    )


def test_parse_labels() -> None:
    assert parse_labels(["P2", "[P3]", "p 2", "4"]) == ["P2", "P3", "P4"]


async def test_blocks_identifiers_without_calling_any_model() -> None:
    llm = FakeLLM({})
    pipeline = EvidencePipeline(synthetic_retriever(), llm)
    answer = await pipeline.answer("Paciente con CC 1.053.456.789, ¿qué opioide uso?")
    assert answer.status == AnswerStatus.BLOCKED
    assert "1.053.456.789" not in answer.question
    assert answer.blocked_kinds == ["documento de identidad"]
    assert llm.calls == []


async def test_abstains_without_generating_when_corpus_has_nothing() -> None:
    plan = QueryPlan(queries_es=["vacuna fiebre amarilla"], queries_en=["yellow fever"])
    llm = FakeLLM({QueryPlan: [plan]})
    pipeline = EvidencePipeline(synthetic_retriever(), llm)
    answer = await pipeline.answer("¿Qué vacuna necesita un viajero a zona amarilla?")
    assert answer.status == AnswerStatus.ABSTAINED
    assert answer.reason == NO_PASSAGES
    assert llm.calls_for(DraftAnswer) == []


async def test_answers_with_verified_citations() -> None:
    draft = DraftAnswer(
        claims=[
            Claim(
                text="En insuficiencia renal grave se prefieren la hidromorfona o el "
                "fentanilo.",
                citations=["P1"],
            ),
            Claim(text="Se recomienda evitar la morfina [P1].", citations=["[P1]"]),
        ]
    )
    llm = FakeLLM(
        {
            QueryPlan: [RENAL_PLAN],
            DraftAnswer: [draft],
            VerificationReport: [verdicts("supported", "supported")],
        }
    )
    answer = await EvidencePipeline(synthetic_retriever(), llm).answer(RENAL_QUESTION)
    assert answer.status == AnswerStatus.ANSWERED
    assert [claim.chunk_ids for claim in answer.claims] == [
        [RENAL_CHUNK],
        [RENAL_CHUNK],
    ]
    assert answer.claims[1].text == "Se recomienda evitar la morfina."
    rendered = render_text(answer)
    assert "Guía sintética A (2024), p. 12" in rendered
    assert DISCLAIMER in rendered


async def test_verifier_sees_claims_and_passages_but_not_the_question() -> None:
    draft = DraftAnswer(
        claims=[Claim(text="Se recomienda evitar la morfina.", citations=["P1"])]
    )
    llm = FakeLLM(
        {
            QueryPlan: [RENAL_PLAN],
            DraftAnswer: [draft],
            VerificationReport: [verdicts("supported")],
        }
    )
    await EvidencePipeline(synthetic_retriever(), llm).answer(RENAL_QUESTION)
    (_, _, verifier_prompt) = llm.calls_for(VerificationReport)[0]
    assert "Se recomienda evitar la morfina." in verifier_prompt
    assert "hidromorfona" in verifier_prompt
    assert RENAL_QUESTION not in verifier_prompt


async def test_claims_citing_unknown_passages_are_discarded() -> None:
    draft = DraftAnswer(
        claims=[
            Claim(text="Se recomienda evitar la morfina.", citations=["P1"]),
            Claim(text="La metadona es de primera línea.", citations=["P9"]),
        ]
    )
    llm = FakeLLM(
        {
            QueryPlan: [RENAL_PLAN],
            DraftAnswer: [draft],
            VerificationReport: [verdicts("supported")],
        }
    )
    answer = await EvidencePipeline(synthetic_retriever(), llm).answer(RENAL_QUESTION)
    assert answer.status == AnswerStatus.PARTIAL
    assert [c.text for c in answer.claims] == ["Se recomienda evitar la morfina."]
    assert answer.discarded[0].decided_by == Stage.CITATIONS


async def test_invented_numbers_are_discarded_before_verification() -> None:
    draft = DraftAnswer(
        claims=[
            Claim(text="Se recomienda evitar la morfina.", citations=["P1"]),
            Claim(
                text="La hidromorfona se inicia con 4 mg cada 6 horas.",
                citations=["P1"],
            ),
        ]
    )
    llm = FakeLLM(
        {
            QueryPlan: [RENAL_PLAN],
            DraftAnswer: [draft],
            VerificationReport: [verdicts("supported")],
        }
    )
    answer = await EvidencePipeline(synthetic_retriever(), llm).answer(RENAL_QUESTION)
    assert answer.status == AnswerStatus.PARTIAL
    discarded = answer.discarded[0]
    assert discarded.decided_by == Stage.NUMBERS
    assert "4" in discarded.rationale and "6" in discarded.rationale
    (_, _, verifier_prompt) = llm.calls_for(VerificationReport)[0]
    assert "4 mg" not in verifier_prompt


async def test_numbers_present_in_the_source_pass() -> None:
    question = "¿Cuál es la dosis inicial de hidromorfona oral?"
    plan = QueryPlan(queries_es=["dosis inicial hidromorfona oral"], queries_en=[])
    draft = DraftAnswer(
        claims=[
            Claim(
                text="La dosis inicial de hidromorfona oral descrita es de 2 mg cada 4 "
                "horas.",
                citations=["P1"],
            )
        ]
    )
    llm = FakeLLM(
        {
            QueryPlan: [plan],
            DraftAnswer: [draft],
            VerificationReport: [verdicts("supported")],
        }
    )
    answer = await EvidencePipeline(synthetic_retriever(), llm).answer(question)
    assert answer.status == AnswerStatus.ANSWERED
    assert answer.claims[0].chunk_ids == ["guia-a:p14:c2"]


async def test_unsupported_verdict_leads_to_abstention() -> None:
    draft = DraftAnswer(
        claims=[Claim(text="La morfina es segura en diálisis.", citations=["P1"])]
    )
    llm = FakeLLM(
        {
            QueryPlan: [RENAL_PLAN],
            DraftAnswer: [draft],
            VerificationReport: [verdicts("unsupported")],
        }
    )
    answer = await EvidencePipeline(synthetic_retriever(), llm).answer(RENAL_QUESTION)
    assert answer.status == AnswerStatus.ABSTAINED
    assert answer.reason == NO_SUPPORT
    assert answer.claims == []
    assert answer.discarded[0].decided_by == Stage.VERIFIER


async def test_verifier_failure_fails_closed() -> None:
    draft = DraftAnswer(
        claims=[Claim(text="Se recomienda evitar la morfina.", citations=["P1"])]
    )
    llm = FakeLLM(
        {
            QueryPlan: [RENAL_PLAN],
            DraftAnswer: [draft],
            VerificationReport: [LLMError("caída")],
        }
    )
    answer = await EvidencePipeline(synthetic_retriever(), llm).answer(RENAL_QUESTION)
    assert answer.status == AnswerStatus.ABSTAINED
    assert answer.reason == VERIFICATION_FAILED
    assert answer.claims == []
    assert answer.error == "caída"


async def test_partial_verdicts_are_dropped_unless_configured() -> None:
    def script() -> FakeLLM:
        draft = DraftAnswer(
            claims=[
                Claim(text="Se recomienda evitar la morfina.", citations=["P1"]),
                Claim(
                    text="El fentanilo es el único opioide seguro.", citations=["P1"]
                ),
            ]
        )
        return FakeLLM(
            {
                QueryPlan: [RENAL_PLAN],
                DraftAnswer: [draft],
                VerificationReport: [verdicts("supported", "partial")],
            }
        )

    strict = await EvidencePipeline(synthetic_retriever(), script()).answer(
        RENAL_QUESTION
    )
    assert strict.status == AnswerStatus.PARTIAL
    assert len(strict.claims) == 1

    lenient = await EvidencePipeline(
        synthetic_retriever(), script(), config=PipelineConfig(keep_partial=True)
    ).answer(RENAL_QUESTION)
    assert lenient.status == AnswerStatus.PARTIAL
    assert [claim.label for claim in lenient.claims] == ["supported", "partial"]
    assert "parcialmente sustentada" in render_text(lenient)


async def test_missing_verdict_discards_the_claim() -> None:
    draft = DraftAnswer(
        claims=[
            Claim(text="Se recomienda evitar la morfina.", citations=["P1"]),
            Claim(text="Se prefieren hidromorfona o fentanilo.", citations=["P1"]),
        ]
    )
    llm = FakeLLM(
        {
            QueryPlan: [RENAL_PLAN],
            DraftAnswer: [draft],
            VerificationReport: [verdicts("supported")],
        }
    )
    answer = await EvidencePipeline(synthetic_retriever(), llm).answer(RENAL_QUESTION)
    assert answer.status == AnswerStatus.PARTIAL
    assert answer.discarded[0].text == "Se prefieren hidromorfona o fentanilo."


INVENTED_NOTE = "No cubre la dosis: inicie hidromorfona 8 mg cada 4 horas."


async def test_model_free_text_is_never_shown_as_the_reason() -> None:
    # Hallazgo de la revisión independiente: el motivo de abstención del
    # modelo se mostraba tal cual y podía traer una dosis inventada.
    abstained = DraftAnswer(abstain=True, abstain_reason=INVENTED_NOTE)
    llm = FakeLLM({QueryPlan: [RENAL_PLAN], DraftAnswer: [abstained]})
    answer = await EvidencePipeline(synthetic_retriever(), llm).answer(RENAL_QUESTION)
    assert answer.status == AnswerStatus.ABSTAINED
    assert answer.reason == NOT_COVERED
    assert answer.model_note == INVENTED_NOTE
    assert "8 mg" not in render_text(answer)
    assert llm.calls_for(VerificationReport) == []

    partial = DraftAnswer(
        claims=[Claim(text="Se recomienda evitar la morfina.", citations=["P1"])],
        abstain=True,
        abstain_reason=INVENTED_NOTE,
    )
    llm = FakeLLM(
        {
            QueryPlan: [RENAL_PLAN],
            DraftAnswer: [partial],
            VerificationReport: [verdicts("supported")],
        }
    )
    answer = await EvidencePipeline(synthetic_retriever(), llm).answer(RENAL_QUESTION)
    assert answer.status == AnswerStatus.PARTIAL
    assert answer.reason == PARTLY_COVERED
    assert "8 mg" not in render_text(answer)
    assert "sin verificar" in render_text(answer, show_discarded=True)


async def test_a_gap_noted_without_abstain_still_makes_the_answer_partial() -> None:
    # Primera prueba real: el modelo respondió en parte, explicó lo que faltaba
    # y dejó abstain = false; el título decía "Respuesta con evidencia citada".
    draft = DraftAnswer(
        claims=[Claim(text="Se recomienda evitar la morfina.", citations=["P1"])],
        abstain=False,
        abstain_reason="No compara los opioides según el grado de función renal.",
    )
    llm = FakeLLM(
        {
            QueryPlan: [RENAL_PLAN],
            DraftAnswer: [draft],
            VerificationReport: [verdicts("supported")],
        }
    )
    answer = await EvidencePipeline(synthetic_retriever(), llm).answer(RENAL_QUESTION)
    assert answer.status == AnswerStatus.PARTIAL
    assert answer.reason == PARTLY_COVERED

    blank_note = draft.model_copy(update={"abstain_reason": "  "})
    llm = FakeLLM(
        {
            QueryPlan: [RENAL_PLAN],
            DraftAnswer: [blank_note],
            VerificationReport: [verdicts("supported")],
        }
    )
    answer = await EvidencePipeline(synthetic_retriever(), llm).answer(RENAL_QUESTION)
    assert answer.status == AnswerStatus.ANSWERED


async def test_a_claim_citing_any_missing_passage_is_discarded() -> None:
    draft = DraftAnswer(
        claims=[Claim(text="Se recomienda evitar la morfina.", citations=["P1", "P9"])]
    )
    llm = FakeLLM({QueryPlan: [RENAL_PLAN], DraftAnswer: [draft]})
    answer = await EvidencePipeline(synthetic_retriever(), llm).answer(RENAL_QUESTION)
    assert answer.status == AnswerStatus.ABSTAINED
    assert answer.discarded[0].decided_by == Stage.CITATIONS
    assert "P9" in answer.discarded[0].rationale
    assert llm.calls_for(VerificationReport) == []


async def test_conflicting_verdicts_keep_the_most_severe() -> None:
    draft = DraftAnswer(
        claims=[Claim(text="Se recomienda evitar la morfina.", citations=["P1"])]
    )
    report = VerificationReport(
        verdicts=[
            SupportVerdict(claim_index=1, label="supported"),
            SupportVerdict(claim_index=1, label="unsupported"),
        ]
    )
    llm = FakeLLM(
        {QueryPlan: [RENAL_PLAN], DraftAnswer: [draft], VerificationReport: [report]}
    )
    answer = await EvidencePipeline(synthetic_retriever(), llm).answer(RENAL_QUESTION)
    assert answer.status == AnswerStatus.ABSTAINED


async def test_claims_over_the_limit_are_reported_not_silently_dropped() -> None:
    claim = Claim(text="Se recomienda evitar la morfina.", citations=["P1"])
    llm = FakeLLM(
        {
            QueryPlan: [RENAL_PLAN],
            DraftAnswer: [DraftAnswer(claims=[claim, claim, claim])],
            VerificationReport: [verdicts("supported", "supported")],
        }
    )
    pipeline = EvidencePipeline(
        synthetic_retriever(), llm, config=PipelineConfig(max_claims=2)
    )
    answer = await pipeline.answer(RENAL_QUESTION)
    assert answer.status == AnswerStatus.PARTIAL
    assert len(answer.claims) == 2
    assert "máximo" in answer.discarded[0].rationale


async def test_retrieval_only_mode_without_model() -> None:
    answer = await EvidencePipeline(synthetic_retriever(), None).answer(RENAL_QUESTION)
    assert answer.status == AnswerStatus.ABSTAINED
    assert answer.reason == NO_MODEL
    assert answer.passages[0].chunk_id == RENAL_CHUNK
    assert "Pasajes más cercanos" in render_text(answer)


async def test_planner_failure_falls_back_to_the_literal_question() -> None:
    draft = DraftAnswer(
        claims=[Claim(text="Se recomienda evitar la morfina.", citations=["P1"])]
    )
    llm = FakeLLM(
        {
            QueryPlan: [LLMError("sin plan")],
            DraftAnswer: [draft],
            VerificationReport: [verdicts("supported")],
        }
    )
    answer = await EvidencePipeline(synthetic_retriever(), llm).answer(RENAL_QUESTION)
    assert answer.status == AnswerStatus.ANSWERED
