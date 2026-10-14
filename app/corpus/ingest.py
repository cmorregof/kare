"""Ingesta: manifiesto → páginas → fragmentos → términos → SQLite.

Respeta la licencia declarada: lo que no está autorizado no entra, y lo que
deja de estarlo sale del índice en la siguiente ingesta completa.
"""

import hashlib
import logging
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pypdf import PdfReader

from app.corpus.chunking import ChunkingConfig, chunk_pages, strip_running_lines
from app.corpus.manifest import Manifest, SourceSpec
from app.corpus.store import CorpusStore
from app.retrieval.text import analyze

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".pdf", ".txt", ".md")

MISMATCH = (
    "el archivo no es la versión revisada (su sha256 no coincide con el del "
    "manifiesto): revisa la versión nueva y actualiza sha256 antes de indexarla"
)


@dataclass(frozen=True)
class IngestReport:
    source_id: str
    status: Literal["ingerida", "omitida", "pendiente", "retirada", "error"]
    message: str
    chunks: int = 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pages(path: Path) -> list[tuple[int | None, str]]:
    """Texto por página. Los .txt/.md no tienen páginas salvo saltos \\f."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        # pypdf avisa de fuentes que no decodifica del todo (fontTools); no
        # cambia el texto que se extrae y llena la terminal de ruido.
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            reader.decrypt("")
        return [
            (number, page.extract_text() or "")
            for number, page in enumerate(reader.pages, start=1)
        ]
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
        parts = text.split("\f")
        if len(parts) == 1:
            return [(None, text)]
        return [(number, part) for number, part in enumerate(parts, start=1)]
    raise ValueError(f"Formato no soportado: {path.suffix}")


def ingest_source(
    spec: SourceSpec,
    *,
    corpus_dir: Path,
    store: CorpusStore,
    config: ChunkingConfig,
) -> IngestReport:
    if not spec.ingestible:
        return IngestReport(
            spec.id, "omitida", f"licencia «{spec.license_status.value}»: no se indexa"
        )
    path = corpus_dir / spec.file
    if not path.exists():
        # Autorizada pero sin archivo todavía: no es un error, queda pendiente.
        return IngestReport(spec.id, "pendiente", f"falta el archivo {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return IngestReport(spec.id, "error", f"formato no soportado: {path.suffix}")
    sha256 = file_sha256(path)
    if spec.sha256 and sha256 != spec.sha256:
        return IngestReport(spec.id, "error", MISMATCH)
    try:
        pages = read_pages(path)
    except Exception as exc:  # pypdf lanza tipos muy variados
        logger.exception("No se pudo leer %s", path)
        return IngestReport(spec.id, "error", f"no se pudo leer el archivo: {exc}")
    chunks = chunk_pages(strip_running_lines(pages), config)
    if not chunks:
        return IngestReport(
            spec.id,
            "error",
            "sin texto extraíble (¿PDF escaneado? necesita OCR antes de ingerirse)",
        )
    terms = [analyze(chunk.text, spec.language) for chunk in chunks]
    store.save_source(spec, sha256=sha256, chunks=chunks, terms=terms)
    paged = len({chunk.page for chunk in chunks if chunk.page is not None})
    detail = f"{len(chunks)} fragmentos" + (f" de {paged} páginas" if paged else "")
    return IngestReport(spec.id, "ingerida", detail, chunks=len(chunks))


def ingest_manifest(
    manifest: Manifest,
    *,
    corpus_dir: Path,
    store: CorpusStore,
    config: ChunkingConfig,
    only: Collection[str] | None = None,
) -> list[IngestReport]:
    known = {source.id for source in manifest.sources}
    if only:
        unknown = sorted(set(only) - known)
        if unknown:
            raise ValueError(f"No están en el manifiesto: {', '.join(unknown)}")
    reports = [
        ingest_source(spec, corpus_dir=corpus_dir, store=store, config=config)
        for spec in manifest.sources
        if not only or spec.id in only
    ]
    if not only:
        allowed = {source.id for source in manifest.sources if source.ingestible}
        for source_id in store.delete_sources_except(allowed):
            reports.append(
                IngestReport(
                    source_id,
                    "retirada",
                    "salió del índice: ya no está autorizada en el manifiesto",
                )
            )
    return reports
