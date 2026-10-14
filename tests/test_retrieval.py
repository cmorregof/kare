import pytest

from app.corpus.store import StoredChunk
from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.retriever import Query, Retriever
from app.retrieval.text import analyze
from app.schemas.evidence import Language
from tests.fakes import SOURCES, synthetic_retriever


def test_accents_do_not_change_terms() -> None:
    assert analyze("Sedación paliativa", "es") == analyze("sedacion paliativa", "es")
    assert analyze("medicación", "es") == analyze("medicacion", "es")
    assert analyze("Oncología", "es") == analyze("oncologia", "es")


def test_stopwords_and_question_marks_are_dropped() -> None:
    assert analyze("¿Qué es la disnea?", "es") == analyze("disnea", "es")


def test_numbers_are_kept_with_decimal_point() -> None:
    assert analyze("2,5 mg", "es") == ["2.5", "mg"]


def test_english_stemming() -> None:
    assert analyze("opioids", "en") == analyze("opioid", "en")


def test_bm25_prefers_documents_matching_more_terms() -> None:
    docs = [
        analyze("morfina oral en dolor oncológico", "es"),
        analyze("morfina e insuficiencia renal grave", "es"),
        analyze("haloperidol para delirium", "es"),
    ]
    index = BM25Index(docs)
    hits = index.search(analyze("morfina insuficiencia renal", "es"), k=3)
    assert hits[0].doc_index == 1
    assert hits[0].matched == 3
    assert {hit.doc_index for hit in hits} == {0, 1}
    assert index.search(analyze("vacuna", "es"), k=3) == []


def test_rrf_rewards_agreement() -> None:
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert max(scores, key=lambda item: scores[item]) == "b"


def test_rrf_weights_scale_each_ranking() -> None:
    scores = reciprocal_rank_fusion([["a"], ["b"]], weights=[0.5, 1.0])
    assert scores["b"] == 2 * scores["a"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_a_language_with_fewer_queries_still_reaches_the_top() -> None:
    # Textos sintéticos. Tres consultas en español contra una en inglés: con la
    # suma simple, los cuatro pasajes en español llenaban el top 3.
    texts: list[tuple[Language, str]] = [
        ("es", "Texto sintético. Morfina e insuficiencia renal: ajuste de dosis."),
        ("es", "Texto sintético. Morfina, insuficiencia renal y metabolitos."),
        ("es", "Texto sintético. Insuficiencia renal grave y morfina oral."),
        ("es", "Texto sintético. Morfina en insuficiencia renal moderada."),
        ("en", "Synthetic text. Morphine accumulates in renal impairment."),
    ]
    chunks = [
        StoredChunk(
            chunk_id=f"s{index}",
            source_id="guia-b" if language == "en" else "guia-a",
            page=index + 1,
            ordinal=index,
            language=language,
            text=text,
            terms=tuple(analyze(text, language)),
        )
        for index, (language, text) in enumerate(texts)
    ]
    retriever = Retriever(chunks, SOURCES)
    plan = [
        Query("es", "morfina insuficiencia renal"),
        Query("es", "insuficiencia renal morfina dosis"),
        Query("es", "morfina renal metabolitos"),
        Query("en", "morphine renal impairment"),
    ]
    top = retriever.search(plan, top_k=3)
    assert "en" in {passage.language for passage in top}
    # Dentro de un idioma el orden no cambia por agregar consultas en otro.
    spanish_order = [p.chunk_id for p in retriever.search(plan[:3], top_k=4)]
    mixed = retriever.search(plan, top_k=5)
    assert [p.chunk_id for p in mixed if p.language == "es"] == spanish_order


def test_retriever_returns_relevant_passage_with_provenance() -> None:
    retriever = synthetic_retriever()
    passages = retriever.search([Query("es", "opioide insuficiencia renal grave")])
    assert [p.chunk_id for p in passages] == ["guia-a:p12:c0"]
    assert passages[0].source_title == "Guía sintética A"
    assert passages[0].page == 12
    assert passages[0].coverage == 1.0


def test_retriever_searches_each_language_in_its_index() -> None:
    retriever = synthetic_retriever()
    passages = retriever.search([Query("en", "opioids breathlessness fan")])
    assert [p.chunk_id for p in passages] == ["guia-b:p5:c0"]
    assert passages[0].language == "en"


def test_low_coverage_passages_are_filtered() -> None:
    retriever = synthetic_retriever()
    # Solo "hidromorfona" coincide: 1 de 4 términos no alcanza el umbral.
    query = Query("es", "hidromorfona prurito urticaria exantema")
    assert retriever.search([query], min_coverage=0.3) == []


def test_single_word_queries_never_pass_the_gate() -> None:
    # Hallazgo de la revisión independiente: una consulta de una palabra
    # ("pacientes") abría la puerta al generador en preguntas fuera del corpus.
    retriever = synthetic_retriever()
    assert retriever.search([Query("es", "pacientes")]) == []
    plan = [Query("es", "vacuna fiebre amarilla"), Query("es", "pacientes")]
    assert retriever.search(plan) == []
