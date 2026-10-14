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
# Lugar que se reserva en una afirmación para el número, la nota y las citas.
_CLAIM_RESERVE = 200
_CUT_NOTE = "… (recortado: ver la fuente)"


def _e(text: str) -> str:
    return escape(text, quote=False)


def _units(text: str) -> int:
    """Largo como lo cuenta Telegram: unidades UTF-16 (un emoji vale dos)."""
    return len(text.encode("utf-16-le")) // 2


def _fit(block: str, budget: int) -> str:
    """Último recurso para un bloque que solo no cabe: se recorta y lo dice.

    Las afirmaciones nunca llegan aquí (se parten antes); solo un título o una
    fuente desmesurados.
    """
    if _units(block) <= budget:
        return block
    cut = block
    while cut and _units(cut) > budget - _units(_CUT_NOTE):
        cut = cut[:-64]
    return cut.rsplit(" ", 1)[0] + _CUT_NOTE


def _pieces(text: str, size: int) -> list[str]:
    """Parte un texto en trozos cuyo HTML escapado cabe en size, por espacios."""
    pieces: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and _units(_e(candidate)) > size:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current or not pieces:
        pieces.append(current)
    return pieces


def pack(
    blocks: Sequence[str],
    *,
    separator: str = "\n",
    prefix: str = "",
    suffix: str = "",
    budget: int = _BUDGET,
) -> list[str]:
    """Agrupa bloques en mensajes que caben, sin partir ningún bloque."""
    room = budget - _units(prefix) - _units(suffix)
    messages: list[str] = []
    current: list[str] = []
    size = 0
    for raw in blocks:
        block = _fit(raw, room)
        extra = _units(block) + (_units(separator) if current else 0)
        if current and size + extra > room:
            messages.append(prefix + separator.join(current) + suffix)
            current, size = [], 0
            extra = _units(block)
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
        # Una afirmación nunca se recorta: si no cabe en un mensaje, sigue en el
        # siguiente, y la nota y las citas van con su último trozo.
        pieces = _pieces(claim.text, _BUDGET - _CLAIM_RESERVE)
        for position, piece in enumerate(pieces):
            block = f"{index}. {_e(piece)}" if position == 0 else _e(piece)
            if position == len(pieces) - 1:
                block = f"{block}{note} {marks}".rstrip()
            blocks.append(block)
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
