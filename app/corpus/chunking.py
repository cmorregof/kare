"""Limpieza de texto extraído y partición en fragmentos con página.

Los fragmentos no cruzan páginas: así cada cita apunta a una página exacta.
Se arman por oraciones hasta un tamaño objetivo, con un pequeño solapamiento
para no cortar una recomendación por la mitad.
"""

import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    target_chars: int = 1000
    max_chars: int = 1400
    overlap_chars: int = 150
    min_chars: int = 40

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_chars < self.target_chars <= self.max_chars:
            raise ValueError("Se requiere 0 <= overlap < target <= max.")


@dataclass(frozen=True)
class Chunk:
    page: int | None
    ordinal: int
    text: str


_SENTENCE_BREAK = re.compile(r"(?<=[.!?;:])\s+(?=[A-ZÁÉÍÓÚÑ¿¡•\d(\-–])|\s+(?=•)")

_EDGE_LINES = 3
_BARE_NUMBER = re.compile(r"\s*(\d{1,4})\s*")
_LETTER = re.compile(r"[^\W\d_]")


def _line_key(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line)).strip().lower()


def _filled(lines: Sequence[str]) -> list[int]:
    return [index for index, line in enumerate(lines) if line.strip()]


def _edge_drops(
    order: Sequence[int],
    lines: Sequence[str],
    *,
    running: set[str],
    is_page_number: Callable[[str], bool],
) -> set[int]:
    """Líneas de un borde que salen, recorriendo desde el extremo hacia adentro.

    Mientras el bloque desde el extremo sea todo encabezado o número de página,
    sale completo aunque pase de tres líneas (la OMS imprime su encabezado dos
    veces). Después, dentro de las tres primeras líneas sale solo lo que es
    encabezado; un número suelto ahí ya no se toca.
    """
    drops: set[int] = set()
    contiguous = True
    for position, index in enumerate(order):
        line = lines[index]
        repeated = _line_key(line) in running
        if contiguous and (repeated or is_page_number(line)):
            drops.add(index)
            continue
        contiguous = False
        if position >= _EDGE_LINES:
            break
        if repeated:
            drops.add(index)
    return drops


def strip_running_lines(
    pages: Sequence[tuple[int | None, str]],
    *,
    min_pages: int = 8,
    share: float = 0.3,
) -> list[tuple[int | None, str]]:
    """Quita encabezados, pies y números de página repetidos en el borde.

    Una línea es encabezado o pie si, con las cifras normalizadas ("MINSALUD -
    IETS 41" y "MINSALUD - IETS 42" son la misma), aparece entre las tres
    primeras o las tres últimas líneas de al menos el 30 % de las páginas.
    Solo se quita en el borde: la misma frase en mitad de una página es
    contenido. Un número suelto sale solo si está pegado al borde y sigue la
    numeración de las páginas (página 41 impresa como "39"), para no borrar la
    cifra de una tabla. Además de limpiar la búsqueda, esto evita que un número
    de página haga pasar una cifra inventada en el chequeo de cifras.
    """
    split = [(number, text.splitlines()) for number, text in pages]
    if len(split) < min_pages:
        return [(number, text) for number, text in pages]
    threshold = share * len(split)

    counts: Counter[str] = Counter()
    offsets: Counter[int] = Counter()
    for number, lines in split:
        filled = _filled(lines)
        edges = filled[:_EDGE_LINES] + filled[-_EDGE_LINES:]
        counts.update({_line_key(lines[index]) for index in edges})
        if number is not None:
            seen_here = set()
            for index in edges:
                bare = _BARE_NUMBER.fullmatch(lines[index])
                if bare:
                    seen_here.add(int(bare.group(1)) - number)
            offsets.update(seen_here)
    running = {
        key for key, seen in counts.items() if seen >= threshold and _LETTER.search(key)
    }
    offset = None
    if offsets:
        candidate, seen = offsets.most_common(1)[0]
        if seen >= threshold:
            offset = candidate

    cleaned: list[tuple[int | None, str]] = []
    for number, lines in split:

        def is_page_number(line: str, number: int | None = number) -> bool:
            bare = _BARE_NUMBER.fullmatch(line)
            return (
                bare is not None
                and offset is not None
                and number is not None
                and int(bare.group(1)) - number == offset
            )

        filled = _filled(lines)
        drops = set()
        for order in (filled, filled[::-1]):
            drops |= _edge_drops(
                order, lines, running=running, is_page_number=is_page_number
            )
        kept = [line for index, line in enumerate(lines) if index not in drops]
        cleaned.append((number, "\n".join(kept)))
    return cleaned


def clean_page_text(raw: str) -> str:
    """Une guiones de fin de línea y líneas partidas; conserva párrafos."""
    text = raw.replace("\xad", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t\xa0]+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        for piece in _SENTENCE_BREAK.split(paragraph.strip()):
            piece = piece.strip()
            if piece:
                sentences.append(piece)
    return sentences


def _split_long(sentence: str, limit: int) -> list[str]:
    if len(sentence) <= limit:
        return [sentence]
    pieces: list[str] = []
    current = ""
    for word in sentence.split():
        while len(word) > limit:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:limit])
            word = word[limit:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) > limit:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _tail(sentences: Sequence[str], budget: int) -> list[str]:
    """Últimas oraciones que caben en el presupuesto de solapamiento."""
    kept: list[str] = []
    size = 0
    for sentence in reversed(sentences):
        extra = len(sentence) + (1 if kept else 0)
        if size + extra > budget:
            break
        kept.insert(0, sentence)
        size += extra
    return kept


def pack_sentences(sentences: Sequence[str], config: ChunkingConfig) -> list[str]:
    limit = max(config.max_chars - config.overlap_chars - 1, 1)
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        for piece in _split_long(sentence, limit):
            has_new_content = len(current) > len(_tail(current, config.overlap_chars))
            if (
                current
                and has_new_content
                and size + 1 + len(piece) > config.target_chars
            ):
                chunks.append(" ".join(current))
                current = _tail(current, config.overlap_chars)
                size = len(" ".join(current))
            current.append(piece)
            size = len(" ".join(current))
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(
    pages: Iterable[tuple[int | None, str]], config: ChunkingConfig
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page, raw in pages:
        text = clean_page_text(raw)
        if not text:
            continue
        for piece in pack_sentences(split_sentences(text), config):
            if len(piece) >= config.min_chars:
                chunks.append(Chunk(page=page, ordinal=len(chunks), text=piece))
    return chunks
