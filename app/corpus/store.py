"""Persistencia del corpus en SQLite, con SQL explícito.

Guarda las fuentes (con su licencia y la huella del archivo) y los fragmentos
ya analizados, para que cargar el índice no exija volver a procesar los PDF.
"""

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from app.corpus.chunking import Chunk
from app.corpus.manifest import SourceSpec
from app.schemas.evidence import Language

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    publisher TEXT NOT NULL,
    year INTEGER,
    version TEXT,
    jurisdiction TEXT NOT NULL,
    kind TEXT NOT NULL,
    language TEXT NOT NULL,
    url TEXT,
    license_status TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    page INTEGER,
    ordinal INTEGER NOT NULL,
    language TEXT NOT NULL,
    text TEXT NOT NULL,
    terms TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);
"""


@dataclass(frozen=True)
class SourceInfo:
    source_id: str
    title: str
    publisher: str
    year: int | None
    language: Language
    license_status: str
    url: str | None


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    source_id: str
    page: int | None
    ordinal: int
    language: Language
    text: str
    terms: tuple[str, ...]


def chunk_id(source_id: str, chunk: Chunk) -> str:
    page = f"p{chunk.page}:" if chunk.page is not None else ""
    return f"{source_id}:{page}c{chunk.ordinal}"


def as_language(value: object) -> Language:
    if value == "es":
        return "es"
    if value == "en":
        return "en"
    raise ValueError(f"Idioma no soportado: {value!r}")


def _optional_int(value: object) -> int | None:
    return None if value is None else int(str(value))


class CorpusStore:
    def __init__(self, db_path: str | Path) -> None:
        path = str(db_path)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CorpusStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def save_source(
        self,
        spec: SourceSpec,
        *,
        sha256: str,
        chunks: Sequence[Chunk],
        terms: Sequence[Sequence[str]],
    ) -> None:
        """Reemplaza la fuente y todos sus fragmentos en una sola transacción."""
        if len(chunks) != len(terms):
            raise ValueError("Cada fragmento necesita su lista de términos.")
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE source_id = ?", (spec.id,))
            self._conn.execute(
                """
                INSERT INTO sources (
                    id, title, publisher, year, version, jurisdiction, kind,
                    language, url, license_status, file_sha256, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    publisher = excluded.publisher,
                    year = excluded.year,
                    version = excluded.version,
                    jurisdiction = excluded.jurisdiction,
                    kind = excluded.kind,
                    language = excluded.language,
                    url = excluded.url,
                    license_status = excluded.license_status,
                    file_sha256 = excluded.file_sha256,
                    ingested_at = excluded.ingested_at
                """,
                (
                    spec.id,
                    spec.title,
                    spec.publisher,
                    spec.year,
                    spec.version,
                    spec.jurisdiction,
                    spec.kind,
                    spec.language,
                    spec.url,
                    spec.license_status.value,
                    sha256,
                    now,
                ),
            )
            self._conn.executemany(
                """
                INSERT INTO chunks (id, source_id, page, ordinal, language, text, terms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk_id(spec.id, chunk),
                        spec.id,
                        chunk.page,
                        chunk.ordinal,
                        spec.language,
                        chunk.text,
                        " ".join(chunk_terms),
                    )
                    for chunk, chunk_terms in zip(chunks, terms, strict=True)
                ],
            )

    def delete_sources_except(self, keep_ids: Iterable[str]) -> list[str]:
        """Retira del índice toda fuente que no esté en `keep_ids`."""
        keep = set(keep_ids)
        current = [
            str(row["id"]) for row in self._conn.execute("SELECT id FROM sources")
        ]
        removed = sorted(source_id for source_id in current if source_id not in keep)
        with self._conn:
            for source_id in removed:
                self._conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        return removed

    def sources(self) -> dict[str, SourceInfo]:
        rows = self._conn.execute(
            "SELECT id, title, publisher, year, language, license_status, url "
            "FROM sources ORDER BY id"
        )
        return {
            str(row["id"]): SourceInfo(
                source_id=str(row["id"]),
                title=str(row["title"]),
                publisher=str(row["publisher"]),
                year=_optional_int(row["year"]),
                language=as_language(row["language"]),
                license_status=str(row["license_status"]),
                url=None if row["url"] is None else str(row["url"]),
            )
            for row in rows
        }

    def chunks(self) -> list[StoredChunk]:
        rows = self._conn.execute(
            "SELECT id, source_id, page, ordinal, language, text, terms "
            "FROM chunks ORDER BY source_id, ordinal"
        )
        return [
            StoredChunk(
                chunk_id=str(row["id"]),
                source_id=str(row["source_id"]),
                page=_optional_int(row["page"]),
                ordinal=int(row["ordinal"]),
                language=as_language(row["language"]),
                text=str(row["text"]),
                terms=tuple(str(row["terms"]).split()),
            )
            for row in rows
        ]

    def chunk_counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT source_id, COUNT(*) AS n FROM chunks GROUP BY source_id"
        )
        return {str(row["source_id"]): int(row["n"]) for row in rows}
