"""Descarga de las fuentes autorizadas desde su enlace oficial.

Para el servidor del bot, donde no hay quien copie los PDF a mano. Solo baja
fuentes que se pueden indexar, y solo acepta el archivo si su sha256 coincide
con el de la versión revisada que declara el manifiesto. Nunca reemplaza un
archivo que ya existe.
"""

import hashlib
import os
import tempfile
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from app import __version__
from app.corpus.manifest import Manifest, SourceSpec

MAX_BYTES = 100 * 1024 * 1024
# Algunos servidores rechazan clientes sin agente de usuario de navegador.
_HEADERS = {
    "user-agent": (
        f"Mozilla/5.0 (compatible; KARE/{__version__}; "
        "+https://github.com/cmorregof/kare)"
    )
}

DownloadStatus = Literal["descargada", "presente", "sin_enlace", "omitida", "error"]


@dataclass(frozen=True)
class DownloadReport:
    source_id: str
    status: DownloadStatus
    message: str


class DownloadError(RuntimeError):
    pass


def _fetch(url: str, target: Path, *, client: httpx.Client, max_bytes: int) -> str:
    """Baja a un temporal junto al destino; devuelve su sha256 sin moverlo."""
    digest = hashlib.sha256()
    size = 0
    with client.stream("GET", url) as response:
        if response.status_code != 200:
            raise DownloadError(f"el servidor respondió {response.status_code}")
        with target.open("wb") as fh:
            for block in response.iter_bytes():
                size += len(block)
                if size > max_bytes:
                    raise DownloadError(f"pasa de {max_bytes // (1024 * 1024)} MB")
                digest.update(block)
                fh.write(block)
    return digest.hexdigest()


def download_source(
    spec: SourceSpec,
    *,
    corpus_dir: Path,
    client: httpx.Client,
    max_bytes: int = MAX_BYTES,
) -> DownloadReport:
    if not spec.ingestible:
        return DownloadReport(
            spec.id, "omitida", f"licencia «{spec.license_status.value}»: no se baja"
        )
    target = corpus_dir / spec.file
    if target.exists():
        return DownloadReport(spec.id, "presente", f"ya está en {target}")
    if not spec.download_url:
        return DownloadReport(
            spec.id, "sin_enlace", "sin enlace de descarga: hay que copiarla a mano"
        )
    if not spec.sha256:
        # Sin huella no hay cómo saber que es la versión que alguien revisó.
        return DownloadReport(
            spec.id, "error", "falta sha256 en el manifiesto: no se baja a ciegas"
        )
    corpus_dir.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=corpus_dir, suffix=".parcial")
    os.close(handle)
    partial = Path(temporary)
    try:
        sha256 = _fetch(spec.download_url, partial, client=client, max_bytes=max_bytes)
        if sha256 != spec.sha256:
            return DownloadReport(
                spec.id,
                "error",
                "lo descargado no es la versión revisada (otro sha256): "
                "revisa la fuente y actualiza el manifiesto",
            )
        partial.replace(target)
    except (httpx.HTTPError, DownloadError) as exc:
        reason = str(exc) if isinstance(exc, DownloadError) else type(exc).__name__
        return DownloadReport(spec.id, "error", f"no se pudo bajar: {reason}")
    finally:
        partial.unlink(missing_ok=True)
    size = target.stat().st_size / (1024 * 1024)
    return DownloadReport(spec.id, "descargada", f"{size:.1f} MB en {target}")


def download_sources(
    manifest: Manifest,
    *,
    corpus_dir: Path,
    only: Collection[str] | None = None,
    transport: httpx.BaseTransport | None = None,
    timeout: float = 120.0,
) -> list[DownloadReport]:
    known = {source.id for source in manifest.sources}
    if only:
        unknown = sorted(set(only) - known)
        if unknown:
            raise ValueError(f"No están en el manifiesto: {', '.join(unknown)}")
    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers=_HEADERS,
        transport=transport,
    ) as client:
        return [
            download_source(spec, corpus_dir=corpus_dir, client=client)
            for spec in manifest.sources
            if not only or spec.id in only
        ]
