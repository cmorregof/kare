"""Fusión de rankings por rango recíproco (RRF, Cormack et al., 2009)."""

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Suma peso/(k + rango) de cada identificador en cada ranking.

    Sin pesos, cada ranking pesa 1 (la RRF original).
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("Se necesita exactamente un peso por ranking.")
    scores: dict[str, float] = {}
    for position, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[position]
        for rank, item in enumerate(dict.fromkeys(ranking), start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return scores
