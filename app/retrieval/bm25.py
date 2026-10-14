"""BM25 de Okapi, en memoria y sin dependencias.

Para un corpus de decenas de guías (miles de fragmentos) es instantáneo y,
sobre todo, explicable: se puede decir por qué un pasaje salió primero.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    doc_index: int
    score: float
    # Términos distintos de la consulta presentes en el documento.
    matched: int


class BM25Index:
    def __init__(
        self,
        documents: Sequence[Sequence[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._tfs: list[Counter[str]] = [Counter(doc) for doc in documents]
        self._lengths: list[int] = [len(doc) for doc in documents]
        self._n = len(documents)
        total = sum(self._lengths)
        self._avgdl = total / self._n if self._n and total else 1.0
        df: Counter[str] = Counter()
        postings: dict[str, list[int]] = {}
        for index, tf in enumerate(self._tfs):
            df.update(tf.keys())
            for term in tf:
                postings.setdefault(term, []).append(index)
        self._postings = postings
        # Variante de Lucene: el idf nunca es negativo.
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (self._n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def __len__(self) -> int:
        return self._n

    def search(self, query_terms: Sequence[str], k: int) -> list[Hit]:
        scores: dict[int, float] = {}
        matched: Counter[int] = Counter()
        for term in dict.fromkeys(query_terms):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index in self._postings[term]:
                tf = self._tfs[index][term]
                norm = tf + self._k1 * (
                    1.0 - self._b + self._b * self._lengths[index] / self._avgdl
                )
                scores[index] = (
                    scores.get(index, 0.0) + idf * tf * (self._k1 + 1.0) / norm
                )
                matched[index] += 1
        ranked = sorted(scores, key=lambda i: (-scores[i], i))[:k]
        return [Hit(doc_index=i, score=scores[i], matched=matched[i]) for i in ranked]
