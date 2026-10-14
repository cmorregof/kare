"""Chequeo determinista de cifras.

Toda cifra de una afirmación (dosis, intervalos, umbrales, años) debe aparecer
en los pasajes que cita. Si no aparece, la afirmación se descarta sin preguntar
a ningún modelo. Es la defensa más barata contra la dosis inventada.

Se aceptan las equivalencias de notación "2,5" = "2.5" y "1.000" = "1000".
Una conversión de unidades ("1 g" frente a "1000 mg") no se acepta: el
generador tiene instrucción de copiar las cifras tal como están.
"""

import re
from collections.abc import Iterable

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_CITATION_MARK = re.compile(r"\[\s*P\d+(?:\s*[,;]\s*P?\d+)*\s*\]|\bP\d+\b")


def strip_citation_marks(text: str) -> str:
    """Quita marcas como [P2] o P3 que el modelo deje dentro del texto."""
    cleaned = _CITATION_MARK.sub("", text)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _interpretations(token: str) -> set[float]:
    parts = re.split(r"[.,]", token)
    if len(parts) == 1:
        return {float(token)}
    values: set[float] = set()
    if all(len(part) == 3 for part in parts[1:]):
        values.add(float("".join(parts)))  # separador de miles
    if len(parts) == 2:
        values.add(float(f"{parts[0]}.{parts[1]}"))  # decimal
    elif all(len(part) == 3 for part in parts[1:-1]):
        values.add(float("".join(parts[:-1]) + "." + parts[-1]))  # 1.000,5
    return values or {float(parts[0])}


def number_values(text: str) -> set[float]:
    values: set[float] = set()
    for match in _NUMBER.finditer(text):
        values |= _interpretations(match.group(0))
    return values


def missing_numbers(claim: str, evidence: Iterable[str]) -> list[str]:
    """Cifras de la afirmación que no aparecen en ninguno de los textos."""
    available: set[float] = set()
    for text in evidence:
        available |= number_values(text)
    missing = [
        match.group(0)
        for match in _NUMBER.finditer(strip_citation_marks(claim))
        if not _interpretations(match.group(0)) & available
    ]
    return list(dict.fromkeys(missing))
