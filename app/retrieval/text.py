"""Análisis léxico para BM25: minúsculas, sin palabras vacías, raíz sin tildes.

Las preguntas escritas a la carrera suelen venir sin tildes ("sedacion",
"medicacion") y las guías vienen bien escritas. Para que ambas formas lleguen
a la misma raíz, se restauran las tildes de los sufijos más comunes antes de
aplicar el stemmer de Snowball y se quitan después.
"""

import re
import unicodedata
from functools import lru_cache

import snowballstemmer

from app.schemas.evidence import Language

_TOKEN = re.compile(r"\d+(?:[.,]\d+)?|[^\W\d_]+")

SPANISH_STOPWORDS = frozenset(
    """
    a al algo algun alguna algunas alguno algunos ante antes asi aun aunque bajo
    cada como con contra cual cuales cualquier cuando cuanto cuanta cuantos
    cuantas de del desde donde dos durante e el ella ellas ello ellos en entre
    era eran es esa esas ese eso esos esta estaba estan estar estas este esto
    estos fue fueron ha habia han hasta hay la las le les lo los mas me mi mis
    mismo muy ni no nos o os otra otras otro otros para pero poco por porque que
    quien quienes se sea segun ser si sin sobre son su sus tambien tan tanto te
    tiene tienen todo todos tras tu tus u un una unas uno unos y ya
    """.split()
)

ENGLISH_STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be because been
    before being below between both but by can could did do does doing down
    during each few for from further had has have having he her here hers him
    his how i if in into is it its itself just me more most my no nor not of off
    on once only or other our ours out over own same she should so some such
    than that the their theirs them then there these they this those through to
    too under until up very was we were what when where which while who whom why
    will with within without would you your
    """.split()
)

_STOPWORDS: dict[Language, frozenset[str]] = {
    "es": SPANISH_STOPWORDS,
    "en": ENGLISH_STOPWORDS,
}
_STEMMERS = {
    "es": snowballstemmer.stemmer("spanish"),
    "en": snowballstemmer.stemmer("english"),
}

# Sufijos que casi siempre llevan tilde en español clínico.
_ACCENTED_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("cion", "ción"),
    ("sion", "sión"),
    ("xion", "xión"),
    ("logia", "logía"),
    ("logias", "logías"),
)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _restore_spanish_accents(word: str) -> str:
    if word != strip_accents(word):
        return word
    for plain, accented in _ACCENTED_SUFFIXES:
        if word.endswith(plain):
            return word[: -len(plain)] + accented
    return word


@lru_cache(maxsize=200_000)
def _stem(word: str, language: Language) -> str:
    if language == "es":
        word = _restore_spanish_accents(word)
    stem = str(_STEMMERS[language].stemWord(word))
    return strip_accents(stem)


def analyze(text: str, language: Language) -> list[str]:
    """Convierte texto en la lista de términos que indexa y busca BM25."""
    stopwords = _STOPWORDS[language]
    terms: list[str] = []
    for match in _TOKEN.finditer(text.lower()):
        token = match.group(0)
        if token[0].isdigit():
            terms.append(token.replace(",", "."))
            continue
        plain = strip_accents(token)
        if len(plain) < 2 or plain in stopwords:
            continue
        terms.append(_stem(token, language))
    return terms
