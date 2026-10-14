from pathlib import Path

import pytest
from pydantic import ValidationError

from app.corpus.chunking import ChunkingConfig
from app.corpus.ingest import ingest_manifest, read_pages
from app.corpus.manifest import LicenseStatus, Manifest, load_manifest
from app.corpus.store import CorpusStore

MANIFEST = """
sources:
  - id: norma-prueba
    title: "Norma de prueba"
    publisher: Pruebas
    year: 2014
    jurisdiction: Colombia
    kind: norma
    language: es
    file: norma.txt
    license_status: abierta
  - id: guia-pendiente
    title: "Guía pendiente"
    publisher: Pruebas
    jurisdiction: internacional
    kind: guia
    language: en
    file: guia.pdf
    license_status: por_verificar
  - id: guia-sin-archivo
    title: "Guía sin archivo"
    publisher: Pruebas
    jurisdiction: internacional
    kind: guia
    language: es
    file: falta.pdf
    license_status: abierta
"""

NORM_TEXT = (
    "Artículo 1. Texto sintético de prueba. El documento de voluntad anticipada "
    "permite dejar por escrito decisiones sobre tratamientos al final de la vida. "
)


def minimal_pdf(lines: list[str]) -> bytes:
    """PDF de una página con texto en Helvetica, armado a mano para las pruebas."""
    text_ops = " ".join(f"({line}) Tj 0 -16 Td" for line in lines)
    stream = f"BT /F1 12 Tf 72 720 Td {text_ops} ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    body = b"%PDF-1.4\n"
    offsets = []
    for number, content in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{number} 0 obj\n".encode() + content + b"\nendobj\n"
    xref_at = len(body)
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        body += f"{offset:010d} 00000 n \n".encode()
    body += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return body


def test_read_pages_from_pdf_keeps_page_numbers(tmp_path: Path) -> None:
    pdf = tmp_path / "guia.pdf"
    pdf.write_bytes(
        minimal_pdf(["Opioids and renal impairment", "synthetic test text"])
    )
    pages = read_pages(pdf)
    assert len(pages) == 1
    number, text = pages[0]
    assert number == 1
    assert "renal impairment" in text


def test_ingest_respects_licenses_and_withdraws_downgraded_sources(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "fuentes"
    corpus.mkdir()
    (corpus / "norma.txt").write_text(NORM_TEXT * 4, encoding="utf-8")
    (corpus / "guia.pdf").write_bytes(minimal_pdf(["no deberia entrar"]))
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(MANIFEST, encoding="utf-8")
    manifest = load_manifest(manifest_path)

    with CorpusStore(tmp_path / "kare.db") as store:
        reports = ingest_manifest(
            manifest, corpus_dir=corpus, store=store, config=ChunkingConfig()
        )
        assert {r.source_id: r.status for r in reports} == {
            "norma-prueba": "ingerida",
            "guia-pendiente": "omitida",
            "guia-sin-archivo": "error",
        }
        assert set(store.chunk_counts()) == {"norma-prueba"}
        chunk = store.chunks()[0]
        assert chunk.page is None
        assert chunk.chunk_id.startswith("norma-prueba:c")
        assert store.sources()["norma-prueba"].year == 2014

        restricted = manifest.sources[0].model_copy(
            update={"license_status": LicenseStatus.RESTRICTED}
        )
        downgraded = Manifest(sources=[restricted, *manifest.sources[1:]])
        reports = ingest_manifest(
            downgraded, corpus_dir=corpus, store=store, config=ChunkingConfig()
        )
        assert ("norma-prueba", "retirada") in {
            (r.source_id, r.status) for r in reports
        }
        assert store.chunk_counts() == {}


def test_manifest_rejects_duplicates_and_bad_ids(tmp_path: Path) -> None:
    source = {
        "id": "fuente-uno",
        "title": "T",
        "publisher": "P",
        "jurisdiction": "Colombia",
        "kind": "guia",
        "language": "es",
        "file": "x.pdf",
        "license_status": "abierta",
    }
    with pytest.raises(ValidationError):
        Manifest.model_validate({"sources": [source, source]})
    with pytest.raises(ValidationError):
        Manifest.model_validate({"sources": [source | {"id": "Con Espacios"}]})


def test_only_unknown_ids_are_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(MANIFEST, encoding="utf-8")
    with CorpusStore(":memory:") as store, pytest.raises(ValueError):
        ingest_manifest(
            load_manifest(manifest_path),
            corpus_dir=tmp_path,
            store=store,
            config=ChunkingConfig(),
            only=["no-existe"],
        )
