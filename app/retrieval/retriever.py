"""Recuperación léxica bilingüe con umbral de pertinencia.

Un índice BM25 por idioma; cada consulta busca en el índice de su idioma; los
rankings se fusionan con RRF. Un pasaje solo cuenta si contiene una fracción
mínima de los términos de la consulta que lo trajo: es la primera compuerta
de abstención, y es determinista.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.corpus.store import CorpusStore, SourceInfo, StoredChunk
from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.text import analyze
from app.schemas.evidence import Language, Passage


@dataclass(frozen=True)
class Query:
    language: Language
    text: str


class Retriever:
    def __init__(
        self, chunks: Sequence[StoredChunk], sources: Mapping[str, SourceInfo]
    ) -> None:
        self._sources = dict(sources)
        self._by_language: dict[Language, list[StoredChunk]] = {"es": [], "en": []}
        for chunk in chunks:
            self._by_language[chunk.language].append(chunk)
        self._indexes: dict[Language, BM25Index] = {
            language: BM25Index([chunk.terms for chunk in items])
            for language, items in self._by_language.items()
        }
        self._by_id = {chunk.chunk_id: chunk for chunk in chunks}

    @classmethod
    def from_store(cls, store: CorpusStore) -> "Retriever":
        return cls(store.chunks(), store.sources())

    def __len__(self) -> int:
        return len(self._by_id)

    def search(
        self,
        queries: Sequence[Query],
        *,
        top_k: int = 8,
        per_query_k: int = 20,
        min_coverage: float = 0.3,
    ) -> list[Passage]:
        rankings: list[list[str]] = []
        coverage: dict[str, float] = {}
        for query in queries:
            terms = list(dict.fromkeys(analyze(query.text, query.language)))
            if not terms:
                continue
            chunks = self._by_language[query.language]
            ranking: list[str] = []
            for hit in self._indexes[query.language].search(terms, per_query_k):
                share = hit.matched / len(terms)
                if hit.matched < min(2, len(terms)) or share < min_coverage:
                    continue
                chunk_id = chunks[hit.doc_index].chunk_id
                ranking.append(chunk_id)
                coverage[chunk_id] = max(coverage.get(chunk_id, 0.0), share)
            if ranking:
                rankings.append(ranking)
        fused = reciprocal_rank_fusion(rankings)
        ordered = sorted(fused, key=lambda cid: (-fused[cid], cid))[:top_k]
        return [self._passage(cid, fused[cid], coverage[cid]) for cid in ordered]

    def _passage(self, chunk_id: str, score: float, share: float) -> Passage:
        chunk = self._by_id[chunk_id]
        source = self._sources.get(chunk.source_id)
        return Passage(
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            source_title=source.title if source else chunk.source_id,
            source_year=source.year if source else None,
            page=chunk.page,
            language=chunk.language,
            text=chunk.text,
            score=round(score, 6),
            coverage=round(share, 3),
        )
