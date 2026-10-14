from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.retriever import Query
from app.retrieval.text import analyze
from tests.fakes import synthetic_retriever


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
