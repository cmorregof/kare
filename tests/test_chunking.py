import pytest

from app.corpus.chunking import (
    ChunkingConfig,
    chunk_pages,
    clean_page_text,
    split_sentences,
)


def test_clean_joins_hyphenation_and_lines_but_keeps_paragraphs() -> None:
    raw = "La medica-\nción se ajusta\nsegún la función renal.\n\nOtro párrafo."
    assert clean_page_text(raw) == (
        "La medicación se ajusta según la función renal.\n\nOtro párrafo."
    )


def test_split_sentences_handles_bullets() -> None:
    text = "Primera recomendación. Segunda recomendación. • Viñeta uno • Viñeta dos"
    assert split_sentences(text) == [
        "Primera recomendación.",
        "Segunda recomendación.",
        "• Viñeta uno",
        "• Viñeta dos",
    ]


def test_chunks_respect_size_pages_and_overlap() -> None:
    sentence = "La recomendación número {n} describe una conducta clínica concreta."
    page_text = " ".join(sentence.format(n=n) for n in range(40))
    config = ChunkingConfig(target_chars=400, max_chars=600, overlap_chars=120)
    short_page = "Página corta, pero con texto suficiente para quedarse."
    chunks = chunk_pages([(7, page_text), (8, short_page)], config)
    page_seven = [chunk for chunk in chunks if chunk.page == 7]
    assert len(page_seven) > 1
    assert all(len(chunk.text) <= config.max_chars for chunk in chunks)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    last_sentence = page_seven[0].text.split(". ")[-1]
    assert page_seven[1].text.startswith(last_sentence.rstrip("."))
    assert chunks[-1].page == 8


def test_tiny_pages_are_dropped() -> None:
    config = ChunkingConfig()
    assert chunk_pages([(1, "12"), (2, "   ")], config) == []


def test_long_sentences_are_split_within_limit() -> None:
    config = ChunkingConfig(target_chars=200, max_chars=300, overlap_chars=50)
    long_sentence = " ".join(["palabra"] * 200)
    chunks = chunk_pages([(1, long_sentence)], config)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= config.max_chars for chunk in chunks)


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(target_chars=500, max_chars=400)
