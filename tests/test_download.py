"""Descarga de fuentes con huella. El contenido es sintético."""

import hashlib
from pathlib import Path

import httpx

from app.corpus.chunking import ChunkingConfig
from app.corpus.download import download_sources
from app.corpus.ingest import MISMATCH, ingest_source
from app.corpus.manifest import Manifest
from app.corpus.store import CorpusStore

CONTENT = b"Texto sintetico de prueba: no es una guia real."
URL = "https://fuente.example/guia.txt"


def manifest(**overrides: object) -> Manifest:
    source: dict[str, object] = {
        "id": "guia-sintetica",
        "title": "Guía sintética",
        "publisher": "Pruebas",
        "jurisdiction": "Colombia",
        "kind": "guia",
        "language": "es",
        "file": "guia.txt",
        "download_url": URL,
        "sha256": hashlib.sha256(CONTENT).hexdigest(),
        "license_status": "abierta",
    }
    source.update(overrides)
    return Manifest.model_validate({"sources": [source]})


class Server:
    def __init__(self, status: int = 200, content: bytes = CONTENT) -> None:
        self.status = status
        self.content = content
        self.requests = 0

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests += 1
            assert str(request.url) == URL
            return httpx.Response(self.status, content=self.content)

        return httpx.MockTransport(handler)


def test_the_reviewed_version_is_downloaded_once(tmp_path: Path) -> None:
    server = Server()
    [report] = download_sources(
        manifest(), corpus_dir=tmp_path, transport=server.transport()
    )
    assert report.status == "descargada"
    assert (tmp_path / "guia.txt").read_bytes() == CONTENT
    [again] = download_sources(
        manifest(), corpus_dir=tmp_path, transport=server.transport()
    )
    assert again.status == "presente"
    assert server.requests == 1


def test_another_version_is_rejected_and_nothing_is_left(tmp_path: Path) -> None:
    server = Server(content=b"Otra version sintetica")
    [report] = download_sources(
        manifest(), corpus_dir=tmp_path, transport=server.transport()
    )
    assert report.status == "error"
    assert "versión revisada" in report.message
    assert list(tmp_path.iterdir()) == []


def test_http_errors_and_missing_hashes_do_not_write_files(tmp_path: Path) -> None:
    server = Server(status=404)
    [report] = download_sources(
        manifest(), corpus_dir=tmp_path, transport=server.transport()
    )
    assert report.status == "error"
    assert "404" in report.message
    blind = Server()
    [unpinned] = download_sources(
        manifest(sha256=None), corpus_dir=tmp_path, transport=blind.transport()
    )
    assert unpinned.status == "error"
    assert blind.requests == 0
    assert list(tmp_path.iterdir()) == []


def test_restricted_sources_are_never_downloaded(tmp_path: Path) -> None:
    server = Server()
    [report] = download_sources(
        manifest(license_status="restringida"),
        corpus_dir=tmp_path,
        transport=server.transport(),
    )
    assert report.status == "omitida"
    assert server.requests == 0


def test_ingest_refuses_a_file_that_is_not_the_reviewed_version(
    tmp_path: Path,
) -> None:
    (tmp_path / "guia.txt").write_bytes(b"Version sintetica cambiada en disco.")
    [spec] = manifest().sources
    with CorpusStore(tmp_path / "kare.db") as store:
        report = ingest_source(
            spec, corpus_dir=tmp_path, store=store, config=ChunkingConfig()
        )
        assert store.chunk_counts() == {}
    assert report.status == "error"
    assert report.message == MISMATCH
