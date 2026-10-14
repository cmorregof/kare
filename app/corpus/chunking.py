"""Limpieza de texto extraído y partición en fragmentos con página.

Los fragmentos no cruzan páginas: así cada cita apunta a una página exacta.
Se arman por oraciones hasta un tamaño objetivo, con un pequeño solapamiento
para no cortar una recomendación por la mitad.
"""

import re
from collections.abc import Iterable, Sequence
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
