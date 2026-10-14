"""Guardia de identificadores.

Corre antes de cualquier llamada a un servicio externo y bloquea la pregunta si
parece contener datos que identifican a una persona. Es conservadora a
propósito: un falso positivo cuesta una reformulación; un falso negativo saca
datos de un paciente del equipo.

No detecta todo: un nombre sin señal previa ("María tiene dolor") o una
combinación de cuasi identificadores (edad extrema, municipio pequeño, fecha
de un evento) pasan. La regla de uso no cambia: preguntas clínicas genéricas,
nunca casos con datos.
"""

import re
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


_MONTHS = (
    "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|"
    "octubre|noviembre|diciembre"
)

# Una fecha o un número largo son legítimos cuando citan una norma o un código
# ("Resolución 2665 del 25 de junio de 2018", "código CUPS 890201"). La señal
# debe estar en la misma oración, a pocos caracteres.
_REFERENCE_CONTEXT = re.compile(
    r"(?i:\b(?:ley|decreto|resoluci[oó]n|circular|acuerdo|sentencia|gu[ií]a|"
    r"norma|cups|cum|invima|registro\s+sanitario|c[oó]digo)\b|\bres\.)"
    r"[^.\n]{0,60}$"
)

_NAME = r"[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+){0,3}"

# (tipo, patrón, ¿admite la excepción por contexto normativo?)
_PATTERNS: tuple[tuple[IdentifierKind, re.Pattern[str], bool], ...] = (
    (IdentifierKind.EMAIL, re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), False),
    (
        IdentifierKind.DOCUMENT_ID,
        re.compile(
            r"(?i:\b(?:c\.\s?c\.|cc|c[eé]dula(?:\s+de\s+ciudadan[ií]a)?|t\.\s?i\.|"
            r"tarjeta\s+de\s+identidad|pasaporte|nuip|documento)"
            r"\s*(?:n[°º.o]*\s*)?[:#]?\s*\d[\d.\s]{3,14}\d)"
        ),
        False,
    ),
    # Cédula escrita con puntos de miles: 24.345.678 o 1.053.456.789. Con un
    # solo punto ("1.500 ml") no se marca: es una cantidad.
    (
        IdentifierKind.DOCUMENT_ID,
        re.compile(r"(?<![\d.,])\d{1,3}(?:\.\d{3}){2,3}(?![\d,]|\.\d)"),
        False,
    ),
    # Celulares (3xx) y fijos (60x) colombianos, con o sin +57 y separadores.
    (
        IdentifierKind.PHONE,
        re.compile(
            r"(?<![\d.,])(?:\+?57[\s-]?)?(?:3\d{2}|60\d)[\s-]?\d{3}[\s-]?\d{4}(?![\d,]|\.\d)"
        ),
        False,
    ),
    # Números de 6 a 10 cifras sin contexto: cédulas, celulares, historias.
    (
        IdentifierKind.DOCUMENT_ID,
        re.compile(r"(?<![\d.,])\d{6,10}(?![\d,]|\.\d)"),
        True,
    ),
    (
        IdentifierKind.DATE,
        re.compile(r"(?<![\d.,])\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})(?![\d,]|\.\d)"),
        True,
    ),
    (IdentifierKind.DATE, re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), True),
    (
        IdentifierKind.DATE,
        re.compile(
            rf"(?i:\b\d{{1,2}}\s+de\s+(?:{_MONTHS})(?:\s+(?:de|del)\s+\d{{4}})?\b)"
        ),
        True,
    ),
    (
        IdentifierKind.RECORD_NUMBER,
        re.compile(
            r"(?i:\b(?:h\.\s?c\.|hc|historia(?:\s+cl[ií]nica)?|ingreso|episodio|"
            r"admisi[oó]n)\s*(?:n[°º.o]*\s*)?[:#]?\s*\d{4,}\b)"
        ),
        True,
    ),
    (
        IdentifierKind.ADDRESS,
        re.compile(
            r"(?i:\b(?:calle|cl|cll|carrera|cra|cr|kr|kra|avenida|av|diagonal|dg|"
            r"transversal|tv)\.?\s*\d{1,3}\s?[a-z]?(?:\s*bis)?\s*"
            r"(?:#|n[°º.o]\.?|no\.?)\s*\d{1,3}\s?[a-z]?\s*-\s*\d{1,3}\b)"
        ),
        False,
    ),
    # Nombre propio después de una señal de persona. La señal ignora
    # mayúsculas; el nombre debe empezar con mayúscula ("doña Rosa", "paciente
    # Juan Pérez"). Siglas como ECOG o EPOC no cuentan como nombre.
    (
        IdentifierKind.PERSON_NAME,
        re.compile(
            r"(?i:\b(?:paciente|pcte|pte|sr|sra|srta|señor|señora|señorita|don|doña)"
            rf"\.?)\s+(?P<name>{_NAME})"
        ),
        False,
    ),
)


def scan(text: str) -> list[Finding]:
    """Devuelve los posibles identificadores, sin solapamientos, en orden."""
    found: list[Finding] = []
    for kind, pattern, allows_reference in _PATTERNS:
        for match in pattern.finditer(text):
            if "name" in pattern.groupindex:
                start, end = match.span("name")
            else:
                start, end = match.span()
            window = text[max(0, start - 80) : start]
            if allows_reference and _REFERENCE_CONTEXT.search(window):
                continue
            found.append(Finding(kind=kind, start=start, end=end))
    return _without_overlaps(found)


def _without_overlaps(findings: list[Finding]) -> list[Finding]:
    ordered = sorted(findings, key=lambda f: (f.start, -(f.end - f.start)))
    result: list[Finding] = []
    last_end = -1
    for finding in ordered:
        if finding.start >= last_end:
            result.append(finding)
            last_end = finding.end
    return result


def kinds(findings: list[Finding]) -> list[IdentifierKind]:
    return list(dict.fromkeys(f.kind for f in findings))


def describe(findings: list[Finding]) -> str:
    """Explica el bloqueo sin repetir el dato encontrado."""
    names = ", ".join(kind.value for kind in kinds(findings))
    return (
        f"La pregunta parece contener datos que identifican a una persona "
        f"({names}). No se envió a ningún servicio. Reformúlala como pregunta "
        "clínica general: edad, sexo y diagnóstico en términos amplios, sin "
        "nombres, documentos, fechas exactas ni direcciones."
    )


def redact(text: str, findings: list[Finding]) -> str:
    """Reemplaza cada hallazgo por una etiqueta como [DOCUMENTO]."""
    parts: list[str] = []
    cursor = 0
    for finding in sorted(findings, key=lambda f: f.start):
        parts.append(text[cursor : finding.start])
        parts.append(_PLACEHOLDERS[finding.kind])
        cursor = finding.end
    parts.append(text[cursor:])
    return "".join(parts)
