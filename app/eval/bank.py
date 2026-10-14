"""Banco de preguntas PallQA-es: carga y validación.

Las filas cuyo id empieza por "EJ-" son ejemplos de formato y se ignoran. Si
una pregunta parece traer datos de una persona, el banco entero se rechaza:
ninguna pregunta del banco puede salir a un servicio externo con
identificadores.
"""

from dataclasses import dataclass
from pathlib import Path

from app.eval.csvio import read_rows
from app.guard.identifiers import kinds, scan

BANK_COLUMNS = (
    "id",
    "pregunta",
    "tipo",
    "tema",
    "respondible_esperado",
    "respuesta_de_referencia",
    "fuente_de_referencia",
    "variante_de",
    "autor",
    "notas",
)
EXAMPLE_PREFIX = "EJ-"

_YES = frozenset({"si", "sí", "s", "yes", "y", "1", "x"})
_NO = frozenset({"no", "n", "0"})


class BankError(ValueError):
    """El banco tiene errores que impiden correr la evaluación."""


@dataclass(frozen=True)
class BankQuestion:
    id: str
    question: str
    kind: str = ""
    topic: str = ""
    answerable: bool | None = None
    reference_answer: str = ""
    reference_source: str = ""
    variant_of: str = ""
    author: str = ""
    notes: str = ""


def parse_yes_no(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in _YES:
        return True
    if normalized in _NO:
        return False
    return None


def load_bank(path: Path) -> list[BankQuestion]:
    rows = read_rows(path)
    if rows and not {"id", "pregunta"} <= rows[0].keys():
        raise BankError("El banco necesita al menos las columnas id y pregunta.")
    questions: list[BankQuestion] = []
    errors: list[str] = []
    seen: set[str] = set()
    for number, row in enumerate(rows, start=2):
        qid = row.get("id", "")
        text = row.get("pregunta", "")
        if qid.upper().startswith(EXAMPLE_PREFIX):
            continue
        if not qid or not text:
            errors.append(f"fila {number}: falta id o pregunta")
            continue
        if qid in seen:
            errors.append(f"fila {number}: id repetido ({qid})")
            continue
        seen.add(qid)
        findings = scan(text)
        if findings:
            names = ", ".join(kind.value for kind in kinds(findings))
            errors.append(
                f"fila {number} ({qid}): parece contener datos de una persona ({names})"
            )
            continue
        raw_answerable = row.get("respondible_esperado", "")
        answerable = parse_yes_no(raw_answerable)
        if raw_answerable and answerable is None:
            errors.append(
                f"fila {number} ({qid}): respondible_esperado debe ser si o no"
            )
            continue
        questions.append(
            BankQuestion(
                id=qid,
                question=text,
                kind=row.get("tipo", ""),
                topic=row.get("tema", ""),
                answerable=answerable,
                reference_answer=row.get("respuesta_de_referencia", ""),
                reference_source=row.get("fuente_de_referencia", ""),
                variant_of=row.get("variante_de", ""),
                author=row.get("autor", ""),
                notes=row.get("notas", ""),
            )
        )
    if errors:
        raise BankError("\n".join(errors))
    return questions
