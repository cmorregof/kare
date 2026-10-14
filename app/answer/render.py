"""Presentación de una respuesta en texto plano para la terminal."""

from app.answer.prompts import locator
from app.schemas.evidence import AnswerStatus, FinalAnswer, Passage

DISCLAIMER = (
    "Apoyo a la decisión para profesionales de la salud. No reemplaza el juicio "
    "clínico: verifica cada afirmación en la fuente citada."
)

TITLES = {
    AnswerStatus.ANSWERED: "Respuesta con evidencia citada",
    AnswerStatus.PARTIAL: "Respuesta parcial: solo lo que el corpus sustenta",
    AnswerStatus.ABSTAINED: "Sin respuesta",
    AnswerStatus.BLOCKED: "Pregunta bloqueada",
}


def snippet(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit].rsplit(" ", 1)[0]
    return f"{cut}…"


def source_line(passage: Passage) -> str:
    line = locator(passage)
    if passage.language == "en":
        line += " [en inglés]"
    return line


def render_text(
    answer: FinalAnswer, *, show_discarded: bool = False, snippet_chars: int = 280
) -> str:
    lines = [TITLES[answer.status]]
    if answer.reason:
        lines.append(answer.reason)

    by_id = {passage.chunk_id: passage for passage in answer.passages}
    numbering: dict[str, int] = {}
    if answer.claims:
        lines.append("")
        for index, claim in enumerate(answer.claims, start=1):
            marks = "".join(
                f"[{numbering.setdefault(cid, len(numbering) + 1)}]"
                for cid in claim.chunk_ids
            )
            note = " (parcialmente sustentada)" if claim.label == "partial" else ""
            lines.append(f"{index}. {claim.text}{note} {marks}".rstrip())
        if numbering:
            lines += ["", "Fuentes"]
            for cid, number in numbering.items():
                passage = by_id.get(cid)
                if passage is None:
                    lines.append(f"[{number}] {cid}")
                    continue
                lines.append(f"[{number}] {source_line(passage)}")
                lines.append(f"    «{snippet(passage.text, snippet_chars)}»")
    elif (
        answer.status == AnswerStatus.ABSTAINED and answer.passages and not answer.model
    ):
        lines += ["", "Pasajes más cercanos (sin generar texto)"]
        for number, passage in enumerate(answer.passages, start=1):
            lines.append(f"[{number}] {source_line(passage)}")
            lines.append(f"    «{snippet(passage.text, snippet_chars)}»")

    if show_discarded and answer.discarded:
        lines += ["", "Descartadas"]
        for claim in answer.discarded:
            stage = claim.decided_by.value if claim.decided_by else "límite"
            lines.append(f"- {claim.text} ({stage}: {claim.rationale})")
    if show_discarded and answer.model_note:
        lines += [
            "",
            "Nota del modelo (sin verificar; no usar como evidencia):",
            f"  {answer.model_note}",
        ]

    if answer.status in {AnswerStatus.ANSWERED, AnswerStatus.PARTIAL}:
        lines += ["", DISCLAIMER]
    return "\n".join(lines)
