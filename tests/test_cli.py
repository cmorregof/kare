from collections.abc import Iterator
from pathlib import Path

import pytest

from app.cli import main
from app.core.config import get_settings

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
"""

NORM_TEXT = (
    "Artículo 1. Texto sintético de prueba. El documento de voluntad anticipada "
    "permite dejar por escrito decisiones sobre tratamientos al final de la vida."
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    corpus = tmp_path / "fuentes"
    corpus.mkdir()
    (corpus / "norma.txt").write_text(NORM_TEXT, encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(MANIFEST, encoding="utf-8")
    monkeypatch.setenv("MANIFEST_PATH", str(manifest))
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "kare.db"))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_ingest_status_and_retrieval_only_answer(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["ingerir"]) == 0
    assert main(["estado"]) == 0
    assert "norma-prueba" in capsys.readouterr().out
    assert main(["preguntar", "¿Qué es el documento de voluntad anticipada?"]) == 0
    output = capsys.readouterr().out
    assert "Pasajes más cercanos" in output
    assert "Norma de prueba (2014)" in output


def test_scan_reports_without_echoing(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["escanear", "CC", "1053456789"]) == 1
    assert "1053456789" not in capsys.readouterr().out
    assert main(["escanear", "dolor", "irruptivo"]) == 0


def test_questions_with_identifiers_are_blocked(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["preguntar", "doña Rosa tiene disnea, ¿qué hago?"]) == 0
    output = capsys.readouterr().out
    assert "Pregunta bloqueada" in output
    assert "Rosa" not in output
