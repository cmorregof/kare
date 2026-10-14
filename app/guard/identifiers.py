"""Guardia de identificadores.

Corre antes de cualquier llamada a un servicio externo y bloquea la pregunta si
parece contener datos que identifican a una persona. Es conservadora a
propósito: un falso positivo cuesta una reformulación; un falso negativo saca
datos de un paciente del equipo.

Límites conocidos: un nombre escrito todo en minúsculas ("paciente rosa
gomez") o sin una palabra que lo anuncie ("María tiene dolor"), y las
combinaciones de cuasi identificadores (edad extrema, municipio pequeño, fecha
de un evento) pasan. La regla de uso no cambia: preguntas clínicas generales,
nunca casos con datos.
"""

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum


class IdentifierKind(StrEnum):
    DOCUMENT_ID = "documento de identidad"
    PHONE = "teléfono"
    EMAIL = "correo electrónico"
    DATE = "fecha exacta"
    RECORD_NUMBER = "número de historia o ingreso"
    ADDRESS = "dirección"
    PERSON_NAME = "nombre de persona"


_PLACEHOLDERS: dict[IdentifierKind, str] = {
    IdentifierKind.DOCUMENT_ID: "[DOCUMENTO]",
    IdentifierKind.PHONE: "[TELÉFONO]",
    IdentifierKind.EMAIL: "[CORREO]",
    IdentifierKind.DATE: "[FECHA]",
    IdentifierKind.RECORD_NUMBER: "[HISTORIA]",
    IdentifierKind.ADDRESS: "[DIRECCIÓN]",
    IdentifierKind.PERSON_NAME: "[NOMBRE]",
}


@dataclass(frozen=True)
class Finding:
    kind: IdentifierKind
    start: int
    end: int


# --- Piezas -------------------------------------------------------------------

# Un número no empieza ni sigue pegado a otro dígito, ni a un separador decimal
# o de miles seguido de dígito ("1.500", "2,5"). La puntuación normal de una
# frase (una coma o un punto antes de un espacio) no cuenta como parte del
# número: "su cédula es 1053456789, tiene dolor" se detecta.
_NO_DIGIT_BEFORE = r"(?<!\d)(?<!\d[.,])"
_NO_DIGIT_AFTER = r"(?!\d)(?![.,]\d)"

# Una cifra seguida de una unidad es una cantidad, no un identificador:
# "2.400.000 UI", "300-600-1200 mg", "3-6-12 meses", "100000 habitantes".
_UNIT_AFTER = (
    r"(?!(?i:\s*(?:ui\b|u\b|unidades|mg\b|mcg\b|µg|ug\b|g\b|kg\b|ml\b|cc\b|mmol|"
    r"meq|gotas|tabletas|comprimidos|ampollas|c[eé]lulas|/mm|/µl|/ul|x\s?10|%|"
    r"h\b|hrs?\b|horas?\b|min\b|minutos|d[ií]as?\b|semanas?\b|meses\b|años\b|"
    r"habitantes|personas|pacientes)))"
)

_DAY = r"(?:0?[1-9]|[12]\d|3[01])"
_MONTH_NUMBER = r"(?:0?[1-9]|1[0-2])"
_MONTHS = (
    r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|"
    r"octubre|noviembre|diciembre|ene|feb|mar|abr|may|jun|jul|ago|sept|sep|set|"
    r"oct|nov|dic)\.?(?![a-záéíóúñ])"
)
_NUMBER_WORD = r"(?:n[uú]mero|num\.?|nro\.?|n[°º.o]*)"

_DOCUMENT_CUES = (
    r"c\.\s?c\.|cc|c[eé]dula(?:\s+de\s+(?:ciudadan[ií]a|extranjer[ií]a))?|"
    r"c\.\s?e\.|ce|t\.\s?i\.|ti|tarjeta(?:\s+de\s+identidad)?|registro\s+civil|"
    r"r\.\s?c\.|rc|pasaporte|nuip|ppt|pep|documento(?:\s+de\s+identidad)?|"
    r"identificad[oa]\s+con"
)
_PHONE_CUES = (
    r"tel[eé]fono|tel|cel(?:ular)?|fijo|m[oó]vil|whats(?:app)?|contacto|"
    r"llamar\s+al"
)
_STREETS = (
    r"calle|cll|cl|carrera|cra|kra|kr|cr|ak|ac|avenida(?:\s+(?:calle|carrera))?|"
    r"av|diagonal|dg|transversal|tv|circular|cq"
)
_NAME_WORD = r"[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+"
_TITLE_NAME = rf"{_NAME_WORD}(?:(?:\s+(?:de|del|la|las|los|y))*\s+{_NAME_WORD}){{0,4}}"
_CAPS_NAME = r"[A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,}){1,4}"
_FULL_CUES = (
    r"paciente|señor|señora|señorita|don|doña|nombre|usuari[oa]|hij[oa]|"
    r"espos[oa]|madre|padre|herman[oa]|cuidador(?:a)?|niet[oa]|acompañante"
)
_ABBREVIATED_CUES = r"pcte|pte|sr|sra|srta|dña"

# Palabras con mayúscula que siguen a "paciente" sin ser un nombre: epónimos,
# escalas y palabras corrientes. Se comparan sin tildes y en minúscula.
_NOT_NAMES = frozenset(
    """
    child pugh karnofsky alzheimer parkinson hodgkin barthel zarit glasgow cushing
    crohn down duchenne kaposi wilms ewing burkitt pancoast horner guillain
    charcot huntington lewy creutzfeldt ramsay richmond edmonton palliative
    ecog esas pps pap ppi nyha gold tnm epoc vih sida covid con sin que tiene
    presenta refiere masculino femenino oncologico oncologica adulto adulta
    mayor joven terminal paliativo paliativa ambulatorio ambulatoria
    hospitalizado hospitalizada en de el la los las
    """.split()
)

# --- Excepciones (estrechas a propósito) -------------------------------------

# Una fecha es legítima solo como fecha de expedición de una norma, pegada a
# ella: "Resolución 2665 del 25 de junio de 2018", "Resolución 2665 de 2018,
# expedida el 25 de junio". Una palabra como "ley" más atrás en la frase no
# exime nada.
_LEGAL_DATE_CONTEXT = re.compile(
    r"(?i:\b(?:ley|decreto|resoluci[oó]n|res\.|circular|acuerdo|sentencia)\s*"
    r"(?:n[°º.o]*\s*)?[\w-]+(?:\s+de\s+\d{4})?,?\s+"
    r"(?:expedida\s+el|de\s+fecha|del?|el)\s+)$"
)
# Un número largo es legítimo pegado a un código o a un examen de laboratorio:
# "CUPS 890201", "plaquetas 150000".
_NUMBER_CONTEXT = re.compile(
    r"(?i:\b(?:cups|cum|invima|registro\s+sanitario|plaquetas|leucocitos|"
    r"neutr[oó]filos|linfocitos|recuento|conteo|carga\s+viral|cd4|incidencia|"
    r"prevalencia)\b[^.\n\d]{0,15})$"
)

_Exemption = re.Pattern[str] | None

# (tipo, patrón, contexto que lo exime o None)
_PATTERNS: tuple[tuple[IdentifierKind, re.Pattern[str], _Exemption], ...] = (
    (IdentifierKind.EMAIL, re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), None),
    (
        IdentifierKind.DOCUMENT_ID,
        re.compile(
            rf"(?i:\b(?:{_DOCUMENT_CUES})\s*(?:{_NUMBER_WORD}\s*)?[:#]?\s*"
            r"[a-z]{0,3}\d[\d.,\s-]{3,16}\d)"
        ),
        None,
    ),
    # Con separadores de miles: 24.345.678, 1.053.456.789 o 1,053,456,789. Con
    # un solo separador ("1.500 ml", "plaquetas 150.000") es una cantidad.
    (
        IdentifierKind.DOCUMENT_ID,
        re.compile(
            rf"{_NO_DIGIT_BEFORE}\d{{1,3}}(?:[.,]\d{{3}}){{2,3}}{_NO_DIGIT_AFTER}"
            rf"{_UNIT_AFTER}"
        ),
        None,
    ),
    # Celulares (3xx) y fijos (60x) colombianos, con o sin +57, paréntesis,
    # espacios, guiones o puntos.
    (
        IdentifierKind.PHONE,
        re.compile(
            rf"{_NO_DIGIT_BEFORE}(?:\+?57[\s.-]?)?\(?(?:3\d{{2}}|60\d)\)?[\s.-]?"
            rf"\d{{3}}[\s.-]?\d{{2}}[\s.-]?\d{{2}}{_NO_DIGIT_AFTER}{_UNIT_AFTER}"
        ),
        None,
    ),
    # Cualquier número de siete o más cifras después de una señal de teléfono,
    # incluidos los fijos antiguos de siete cifras ("fijo 881 2345").
    (
        IdentifierKind.PHONE,
        re.compile(
            rf"(?i:\b(?:{_PHONE_CUES})\.?\s*(?:{_NUMBER_WORD}\s*)?[:#]?\s*"
            r"\+?\(?\d[\d\s().-]{5,16}\d)"
        ),
        None,
    ),
    # Números de 6 a 12 cifras sin contexto: cédulas, celulares, historias.
    (
        IdentifierKind.DOCUMENT_ID,
        re.compile(rf"{_NO_DIGIT_BEFORE}\d{{6,12}}{_NO_DIGIT_AFTER}{_UNIT_AFTER}"),
        _NUMBER_CONTEXT,
    ),
    # Día/mes/año con barras, guiones o puntos. No si es parte de una serie
    # ("6-12-18-24 horas") ni si lleva unidad ("3-6-12 meses").
    (
        IdentifierKind.DATE,
        re.compile(
            rf"{_NO_DIGIT_BEFORE}(?<!\d[/-])(?<!\d\s[/-]){_DAY}\s?[/.-]\s?"
            rf"{_MONTH_NUMBER}\s?[/.-]\s?(?:\d{{4}}|\d{{2}})(?!\d)"
            rf"(?!\s?[/.-]\s?\d){_UNIT_AFTER}"
        ),
        _LEGAL_DATE_CONTEXT,
    ),
    # Año primero: 2024-03-12 o 2024/03/12.
    (
        IdentifierKind.DATE,
        re.compile(
            rf"{_NO_DIGIT_BEFORE}(?:19|20)\d{{2}}\s?[/.-]\s?{_MONTH_NUMBER}\s?[/.-]\s?"
            rf"{_DAY}(?!\d)(?!\s?[/.-]\s?\d)"
        ),
        _LEGAL_DATE_CONTEXT,
    ),
    # Día/mes sin año, con el mes en dos cifras ("3/05"). "1/2 tableta" no.
    (
        IdentifierKind.DATE,
        re.compile(
            rf"{_NO_DIGIT_BEFORE}(?<!\d[/-]){_DAY}/(?:0[1-9]|1[0-2])(?!\d)"
            rf"(?!\s?[/.-]\s?\d){_UNIT_AFTER}"
        ),
        _LEGAL_DATE_CONTEXT,
    ),
    # Día y mes en palabras o abreviado, con o sin año: "3 de mayo",
    # "15 de dic.", "3/may/2024", "03-MAY-24", "3 mayo 2024".
    (
        IdentifierKind.DATE,
        re.compile(
            rf"(?i:\b{_DAY}\s*(?:de\s+|[/.-]\s*|\s)?{_MONTHS}"
            r"(?:(?:\s*(?:de|del|,)\s*|\s*[/.-]\s*|\s+)(?:\d{4}|\d{2})(?!\d))?)"
        ),
        _LEGAL_DATE_CONTEXT,
    ),
    # Mes primero, con año: "mayo 3 de 2024".
    (
        IdentifierKind.DATE,
        re.compile(rf"(?i:\b{_MONTHS}\s+{_DAY}\s*(?:de|del|,)\s*\d{{4}}(?!\d))"),
        _LEGAL_DATE_CONTEXT,
    ),
    (
        IdentifierKind.RECORD_NUMBER,
        re.compile(
            r"(?i:\b(?:h\.\s?c\.|hc|historia(?:\s+cl[ií]nica)?|ingreso|episodio|"
            rf"admisi[oó]n)\s*(?:{_NUMBER_WORD}\s*)?[:#]?\s*\d{{4,}}\b)"
        ),
        None,
    ),
    (
        IdentifierKind.ADDRESS,
        re.compile(
            rf"(?i:\b(?:{_STREETS})\.?\s*(?:\d{{1,3}}\s?[a-z]?(?:\s*bis)?|"
            r"[a-záéíóúñ]+(?:\s+[a-záéíóúñ]+)?)(?:\s+(?:sur|norte|este|oeste))?\s*"
            r"(?:#|n[°º.o]\.?|no\.?|n[uú]mero)\s*\d{1,3}\s?[a-z]?(?:\s*bis)?\s*-\s*"
            r"\d{1,3}\b)"
        ),
        None,
    ),
    (
        IdentifierKind.ADDRESS,
        re.compile(
            r"(?i:\b(?:mz|manzana)\.?\s*[a-z0-9]{1,4}\s*,?\s*"
            r"(?:casa|cs|lote|lt|apto|apartamento|interior|int)\.?\s*\d{1,4}\b)"
        ),
        None,
    ),
    # Nombre después de una señal de persona. La señal ignora mayúsculas; el
    # nombre va con mayúscula inicial ("doña Rosa", "Paciente: Rosa Gómez") o
    # todo en mayúsculas si son dos palabras o más ("ROSA GÓMEZ PÉREZ", como
    # sale de una historia). Una sigla sola (ECOG, EPOC) no cuenta.
    (
        IdentifierKind.PERSON_NAME,
        re.compile(
            rf"(?i:\b(?:(?:{_FULL_CUES})(?:\s*:\s*|\s+)|"
            rf"(?:{_ABBREVIATED_CUES})\.?\s*:?\s*))"
            rf"(?P<name>{_TITLE_NAME}|{_CAPS_NAME})"
        ),
        None,
    ),
)


def _plain(word: str) -> str:
    decomposed = unicodedata.normalize("NFKD", word.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _is_name(words: str) -> bool:
    first = re.split(r"[\s-]+", words.strip(), maxsplit=1)[0]
    return _plain(first) not in _NOT_NAMES


def scan(text: str) -> list[Finding]:
    """Posibles identificadores en orden; descarta los contenidos en otro."""
    found: list[Finding] = []
    for kind, pattern, exemption in _PATTERNS:
        for match in pattern.finditer(text):
            if "name" in pattern.groupindex:
                start, end = match.span("name")
                if not _is_name(match.group("name")):
                    continue
            else:
                start, end = match.span()
            window = text[max(0, start - 80) : start]
            if exemption is not None and exemption.search(window):
                continue
            found.append(Finding(kind=kind, start=start, end=end))
    return _without_contained(found)


def _without_contained(findings: list[Finding]) -> list[Finding]:
    ordered = sorted(findings, key=lambda f: (f.start, -(f.end - f.start)))
    result: list[Finding] = []
    for finding in ordered:
        if any(
            kept.start <= finding.start and finding.end <= kept.end for kept in result
        ):
            continue
        result.append(finding)
    return result


def kinds(findings: list[Finding]) -> list[IdentifierKind]:
    return list(dict.fromkeys(f.kind for f in findings))


def describe(findings: list[Finding]) -> str:
    """Explica el bloqueo sin repetir el dato encontrado."""
    names = ", ".join(kind.value for kind in kinds(findings))
    return (
        f"La pregunta parece contener datos que identifican a una persona "
        f"({names}). No se envió al modelo. Reformúlala como pregunta clínica "
        "general: edad, sexo y diagnóstico en términos amplios, sin nombres, "
        "documentos, fechas exactas ni direcciones."
    )


def redact(text: str, findings: list[Finding]) -> str:
    """Reemplaza cada tramo detectado (unidos los que se solapan) por su etiqueta."""
    spans: list[tuple[int, int, IdentifierKind]] = []
    for finding in sorted(findings, key=lambda f: (f.start, -f.end)):
        if spans and finding.start < spans[-1][1]:
            start, end, kind = spans[-1]
            spans[-1] = (start, max(end, finding.end), kind)
        else:
            spans.append((finding.start, finding.end, finding.kind))
    parts: list[str] = []
    cursor = 0
    for start, end, kind in spans:
        parts.append(text[cursor:start])
        parts.append(_PLACEHOLDERS[kind])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)
