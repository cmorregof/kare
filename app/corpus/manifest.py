"""Manifiesto del corpus: qué fuentes hay y bajo qué condiciones se usan."""

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.evidence import Language


class LicenseStatus(StrEnum):
    OPEN = "abierta"
    INTERNAL = "uso_interno"
    PENDING = "por_verificar"
    RESTRICTED = "restringida"


# Solo estas categorías entran al índice.
INGESTIBLE = frozenset({LicenseStatus.OPEN, LicenseStatus.INTERNAL})

SourceKind = Literal["guia", "norma", "consenso", "revision", "protocolo"]


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    title: str
    publisher: str
    year: int | None = None
    version: str | None = None
    jurisdiction: str
    kind: SourceKind
    language: Language
    file: str
    url: str | None = None
    license_status: LicenseStatus
    license_note: str = ""
    evidence_level: str | None = None
    reviewed_on: date | None = None

    @property
    def ingestible(self) -> bool:
        return self.license_status in INGESTIBLE


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[SourceSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "Manifest":
        seen: set[str] = set()
        for source in self.sources:
            if source.id in seen:
                raise ValueError(f"id repetido en el manifiesto: {source.id}")
            seen.add(source.id)
        return self


def load_manifest(path: str | Path) -> Manifest:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"No encuentro el manifiesto en {manifest_path}.")
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    return Manifest.model_validate(data)
