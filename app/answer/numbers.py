"""Chequeo determinista de cifras.

Toda cifra de una afirmación (dosis, intervalos, umbrales, años) debe aparecer
en los pasajes que cita. Si no aparece, la afirmación se descarta sin preguntar
a ningún modelo. Es la defensa más barata contra la dosis inventada.

Notación: un separador seguido de exactamente tres cifras es de miles
("6.000" = 6000, "1,000" = 1000), salvo detrás de un cero ("0,125" es
decimal); con otro número de cifras es decimal ("2,5" = "2.5"). Así un error
de mil veces ("6.000 mg" frente a "6 horas") no pasa. Una conversión de
unidades ("1 g" frente a "1000 mg") tampoco: el generador copia las cifras tal
como están.

Las cantidades escritas con palabras ("cada ocho horas", "dos veces al día",
"every eight hours", "twice daily") cuentan como cifras.

Es una condición necesaria, no suficiente: que el número aparezca en el pasaje
no garantiza que aparezca en el mismo contexto ("10 mg" en la afirmación y
"10 años" en la fuente). Eso lo juzga el verificador.
"""

import re
from collections.abc import Iterable

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# Marcas de cita: entre corchetes o paréntesis ("[P2]", "(P1, P3)"), o sueltas
# con una o dos cifras ("P2"). "P450" o "P2Y12" no son marcas.
_CITATION_MARK = re.compile(
    r"[\[(]\s*P\d+(?:\s*(?:[,;]|y)\s*P?\d+)*\s*[\])]|\bP\d{1,2}\b"
)
_PASSAGE_PHRASE = re.compile(
    r"(?i)\b(?:seg[uú]n|como\s+(?:indica|se\s+[nñ]ala\s+en)|de\s+acuerdo\s+con)\s+"
    r"(?:el|los)\s+pasajes?\s*,?\s*"
)

_SPANISH_WORDS = {
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "quince": 15,
    "veinte": 20,
    "veinticuatro": 24,
    "treinta": 30,
    "cuarenta y ocho": 48,
    "setenta y dos": 72,
}
_ENGLISH_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "twenty-four": 24,
    "thirty": 30,
    "forty-eight": 48,
    "seventy-two": 72,
}
_SPANISH_UNITS = (
    r"veces|horas?|d[ií]as?|semanas?|meses|minutos?|mg|ml|gotas|tabletas|"
    r"comprimidos|dosis|ampollas|puffs?|inhalaciones|litros?"
)
_ENGLISH_UNITS = (
    r"times|hours?|days?|weeks?|months?|minutes?|mg|ml|drops|tablets|doses|"
    r"ampoules|puffs?|litres?|liters?"
)


def _word_pattern(words: dict[str, int], units: str) -> re.Pattern[str]:
    alternatives = "|".join(
        re.escape(word) for word in sorted(words, key=len, reverse=True)
    )
    return re.compile(rf"(?i)\b({alternatives})\s+(?:{units})\b")


_SPANISH_QUANTITY = _word_pattern(_SPANISH_WORDS, _SPANISH_UNITS)
_ENGLISH_QUANTITY = _word_pattern(_ENGLISH_WORDS, _ENGLISH_UNITS)
_TWICE = re.compile(r"(?i)\btwice\b")


def strip_citation_marks(text: str) -> str:
    """Quita marcas como [P2], (P1) o "según el pasaje" que el modelo deje."""
    cleaned = _CITATION_MARK.sub("", text)
    cleaned = _PASSAGE_PHRASE.sub("", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    if cleaned and text.strip()[:1].isupper():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _interpretations(token: str) -> set[float]:
    separators = re.findall(r"[.,]", token)
    parts = re.split(r"[.,]", token)
    if not separators:
        return {float(token)}
    if len(separators) == 1:
        whole, fraction = parts
        if len(fraction) == 3 and whole != "0":
            return {float(whole + fraction)}  # miles: "6.000", "1,000"
        return {float(f"{whole}.{fraction}")}  # decimal: "2,5", "0,125"
    if len(set(separators)) == 1:
        if all(len(part) == 3 for part in parts[1:]):
            return {float("".join(parts))}  # miles: "2.400.000"
        return set()
    last_is_decimal = separators[-1] not in separators[:-1]
    if last_is_decimal and all(len(part) == 3 for part in parts[1:-1]):
        return {float("".join(parts[:-1]) + "." + parts[-1])}  # "1.000,5"
    return set()


def _quantities(text: str, *, english: bool) -> list[tuple[str, set[float]]]:
    found: list[tuple[str, set[float]]] = []
    for match in _NUMBER.finditer(text):
        found.append((match.group(0), _interpretations(match.group(0))))
    for match in _SPANISH_QUANTITY.finditer(text):
        word = match.group(1).lower()
        found.append((match.group(1), {float(_SPANISH_WORDS[word])}))
    if english:
        for match in _ENGLISH_QUANTITY.finditer(text):
            word = match.group(1).lower()
            found.append((match.group(1), {float(_ENGLISH_WORDS[word])}))
        for match in _TWICE.finditer(text):
            found.append((match.group(0), {2.0}))
    return found


def number_values(text: str) -> set[float]:
    """Todas las cantidades del texto, en cifras o en palabras (es y en)."""
    values: set[float] = set()
    for _, interpretations in _quantities(text, english=True):
        values |= interpretations
    return values


def missing_numbers(claim: str, evidence: Iterable[str]) -> list[str]:
    """Cifras de la afirmación que no aparecen en ninguno de los textos."""
    available: set[float] = set()
    for text in evidence:
        available |= number_values(text)
    missing = [
        token
        for token, values in _quantities(strip_citation_marks(claim), english=False)
        if not values & available
    ]
    return list(dict.fromkeys(missing))
