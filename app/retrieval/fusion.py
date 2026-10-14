"""Fusión de rankings por rango recíproco (RRF, Cormack et al., 2009)."""

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, k: int = 60
) -> dict[str, float]:
    """Suma 1/(k + rango) de cada identificador en cada ranking."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores
