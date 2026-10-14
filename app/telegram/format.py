"""Una respuesta de KARE-E como mensajes HTML de Telegram.

Mismo contenido que la terminal: título fijo del sistema, afirmaciones
verificadas con sus marcas de cita, fuentes y aviso. Las fuentes van en un
segundo mensaje, plegadas, para que la respuesta se lea primero. Todo texto
variable se escapa: una afirmación con "<" no puede romper el formato.
"""

from collections.abc import Sequence
from html import escape

from app.answer.render import DISCLAIMER, TITLES, snippet, source_line
from app.schemas.evidence import AnswerStatus, FinalAnswer
from app.telegram.api import MESSAGE_LIMIT

# Margen bajo el límite de Telegram: se mide el HTML con etiquetas, que
# siempre es más largo que el texto que Telegram cuenta.
_BUDGET = MESSAGE_LIMIT - 196
_CUT_NOTE = "… (recortado: ver la fuente)"


def _e(text: str) -> str:
    return escape(text, quote=False)


def _fit(block: str, budget: int) -> str:
    """Un bloque que solo no cabe se recorta, y lo dice."""
    if len(block) <= budget:
        return block
    return block[: budget - len(_CUT_NOTE)].rsplit(" ", 1)[0] + _CUT_NOTE


def pack(
    blocks: Sequence[str],
    *,
    separator: str = "\n",
    prefix: str = "",
    suffix: str = "",
    budget: int = _BUDGET,
) -> list[str]:
    """Agrupa bloques en mensajes que caben, sin partir ningún bloque."""
    room = budget - len(prefix) - len(suffix)
    messages: list[str] = []
    current: list[str] = []
    size = 0
    for raw in blocks:
        block = _fit(raw, room)
        extra = len(block) + (len(separator) if current else 0)
        if current and size + extra > room:
            messages.append(prefix + separator.join(current) + suffix)
            current, size = [], 0
            extra = len(block)
        current.append(block)
        size += extra
    if current:
        messages.append(prefix + separator.join(current) + suffix)
    return messages


def render_telegram(answer: FinalAnswer, *, snippet_chars: int = 280) -> list[str]:
    head = [f"<b>{_e(TITLES[answer.status])}</b>"]
    if answer.reason:
        head.append(_e(answer.reason))
    blocks = ["\n".join(head)]

    numbering: dict[str, int] = {}
    for index, claim in enumerate(answer.claims, start=1):
        marks = "".join(
            f"[{numbering.setdefault(cid, len(numbering) + 1)}]"
            for cid in claim.chunk_ids
        )
        note = " <i>(parcialmente sustentada)</i>" if claim.label == "partial" else ""
        blocks.append(f"{index}. {_e(claim.text)}{note} {marks}".rstrip())
    if answer.status in {AnswerStatus.ANSWERED, AnswerStatus.PARTIAL}:
        blocks.append(f"<i>{_e(DISCLAIMER)}</i>")
    messages = pack(blocks, separator="\n\n")

    by_id = {passage.chunk_id: passage for passage in answer.passages}
    sources: list[str] = []
    for cid, number in numbering.items():
        passage = by_id.get(cid)
        if passage is None:
            sources.append(f"[{number}] {_e(cid)}")
            continue
        sources.append(
            f"[{number}] {_e(source_line(passage))}\n"
            f"«{_e(snippet(passage.text, snippet_chars))}»"
        )
    if sources:
        messages += pack(
            sources,
            separator="\n\n",
            prefix="<b>Fuentes</b>\n<blockquote expandable>",
            suffix="</blockquote>",
        )
    return messages
