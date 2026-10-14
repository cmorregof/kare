import pytest

from app.corpus.chunking import (
    ChunkingConfig,
    chunk_pages,
    clean_page_text,
    split_sentences,
    strip_running_lines,
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


def _page(number: int, body: list[str]) -> tuple[int, str]:
    return number, "\n".join(body)


_WORDS = (
    "morfina hidromorfona fentanilo metadona oxicodona tramadol "
    "haloperidol midazolam dexametasona lactulosa octreotida ketamina "
    "lorazepam furosemida gabapentina pregabalina"
).split()


def test_running_headers_and_page_numbers_are_removed() -> None:
    # Contenido distinto en cada página (como en un PDF real) y la misma
    # etiqueta de contenido en mitad de todas: solo sale lo que es borde.
    pages = [
        _page(
            n,
            [
                str(n - 2),
                f"MINSALUD - IETS {n}",
                f"La {_WORDS[n]} abre la página.",
                f"Sobre {_WORDS[n + 1]} y su uso.",
                "Recomendación D",
                f"Notas de {_WORDS[n - 1]}.",
                f"Dosis de {_WORDS[n - 2]} en la tabla.",
                f"La {_WORDS[n]} cierra la página.",
            ],
        )
        for n in range(3, 15)
    ]
    cleaned = dict(strip_running_lines(pages))
    page = cleaned[7]
    assert "MINSALUD" not in page
    assert not page.startswith("5")
    assert "Recomendación D" in page
    assert f"La {_WORDS[7]} abre la página." in page
    assert page.endswith(f"La {_WORDS[7]} cierra la página.")


def test_table_numbers_at_the_page_edge_are_kept() -> None:
    pages = [
        _page(
            n,
            [
                str(n),
                f"Tabla de {_WORDS[n]}.",
                f"Dosis máxima de {_WORDS[n + 1]}:",
                "500",
            ],
        )
        for n in range(1, 11)
    ]
    cleaned = dict(strip_running_lines(pages))
    assert cleaned[4].endswith("500")
    assert cleaned[4].startswith(f"Tabla de {_WORDS[4]}.")


def test_a_header_printed_twice_is_removed_entirely() -> None:
    # La guía de la OMS trae el encabezado dos veces en las páginas pares, y
    # la segunda copia queda fuera de las tres primeras líneas.
    header = ["WHO GUIDELINES FOR THE PHARMACOLOGICAL", "OF CANCER PAIN IN ADULTS"]
    pages = [
        _page(
            n,
            [
                str(n - 2),
                *header,
                *header,
                f"The {_WORDS[n]} paragraph starts here.",
                f"It mentions {_WORDS[n + 1]} as well.",
                f"And it ends with {_WORDS[n - 1]}.",
            ],
        )
        for n in range(3, 15)
    ]
    page = dict(strip_running_lines(pages))[8]
    assert "WHO GUIDELINES" not in page
    assert "OF CANCER PAIN" not in page
    assert page.startswith(f"The {_WORDS[8]} paragraph starts here.")


def test_a_header_line_after_content_is_kept_outside_the_edge() -> None:
    # Fuera de las tres primeras líneas, la frase repetida solo sale si está
    # pegada al borde: después de contenido es contenido.
    pages = [
        _page(
            n,
            [
                "MINSALUD - IETS",
                f"La {_WORDS[n]} abre.",
                f"Sigue {_WORDS[n + 1]}.",
                f"Luego {_WORDS[n - 1]}.",
                "MINSALUD - IETS",
                f"Más sobre {_WORDS[n]}.",
                f"Otra línea de {_WORDS[n + 1]}.",
                f"Cierre de {_WORDS[n - 1]}.",
                f"Final de {_WORDS[n]}.",
            ],
        )
        for n in range(1, 15)
    ]
    page = dict(strip_running_lines(pages))[5]
    assert page.count("MINSALUD - IETS") == 1
    assert page.startswith(f"La {_WORDS[5]} abre.")


def test_short_documents_are_left_alone() -> None:
    pages = [_page(n, ["MINSALUD - IETS", "Texto."]) for n in range(1, 4)]
    assert strip_running_lines(pages) == pages
