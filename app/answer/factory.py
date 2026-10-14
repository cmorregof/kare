"""Arma el pipeline de evidencia desde la configuración (terminal y bot)."""

from app.answer.pipeline import EvidencePipeline, PipelineConfig
from app.core.config import Settings
from app.corpus.store import CorpusStore
from app.llm.client import build_llm
from app.retrieval.retriever import Retriever


def build_pipeline(
    settings: Settings, store: CorpusStore
) -> tuple[EvidencePipeline, int]:
    """El pipeline y cuántos fragmentos tiene el índice cargado en memoria."""
    retriever = Retriever.from_store(store)
    generator = build_llm(settings, role="generator")
    verifier = build_llm(settings, role="verifier") or generator
    config = PipelineConfig(
        top_k=settings.top_k,
        per_query_k=settings.per_query_k,
        min_coverage=settings.min_coverage,
        keep_partial=settings.keep_partial_claims,
    )
    return EvidencePipeline(retriever, generator, verifier, config), len(retriever)
