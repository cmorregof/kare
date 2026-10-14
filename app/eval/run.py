"""Corrida de evaluación: KARE-E frente al mismo modelo sin corpus.

Escribe en una carpeta nueva:

- ciego.csv: pregunta, referencia y dos respuestas en orden aleatorio, con
  columnas para calificar. Es lo único que ven las calificadoras (una copia
  por persona).
- clave.csv: qué sistema es la respuesta 1 y cuál la 2. No se comparte.
- citas.csv: cada afirmación de KARE-E con los pasajes que cita, para
  calificar si los pasajes la sostienen. Revela el brazo, así que se califica
  después de ciego.csv.
- salidas.jsonl: las respuestas completas de ambos brazos, para el análisis.
- corrida.json: modelo, semilla, fecha, fuentes y conteos.
"""

import asyncio
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from app.answer.prompts import locator
from app.eval.bank import BankQuestion
from app.eval.blind import ARMS, Arm, assign_order, blinded_text
from app.eval.csvio import write_rows
from app.schemas.evidence import FinalAnswer

BLIND_HEADER = (
    "id",
    "pregunta",
    "referencia",
    "respuesta_1",
    "respuesta_2",
    "r1_correcta",
    "r1_dano",
    "r1_debio_abstenerse",
    "r2_correcta",
    "r2_dano",
    "r2_debio_abstenerse",
    "preferida",
    "comentario",
)
KEY_HEADER = ("id", "respuesta_1", "respuesta_2")
CITATION_HEADER = (
    "id",
    "afirmacion_n",
    "afirmacion",
    "fuentes",
    "pasajes",
    "soporta",
    "comentario",
)
RUN_FILES = ("ciego.csv", "clave.csv", "citas.csv", "salidas.jsonl", "corrida.json")


class Answerer(Protocol):
    async def answer(self, question: str) -> FinalAnswer: ...


@dataclass(frozen=True)
class RunOutcome:
    paths: list[Path]
    # Ids de preguntas cuya respuesta falló por error del modelo, por brazo.
    failed: dict[Arm, list[str]]


@dataclass(frozen=True)
class QuestionResult:
    question: BankQuestion
    kare: FinalAnswer
    base: FinalAnswer

    def by_arm(self, arm: Arm) -> FinalAnswer:
        return self.kare if arm == "kare" else self.base


async def collect(
    bank: Sequence[BankQuestion],
    kare: Answerer,
    base: Answerer,
    *,
    concurrency: int = 3,
    on_progress: Callable[[str], None] | None = None,
) -> list[QuestionResult]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(question: BankQuestion) -> QuestionResult:
        async with semaphore:
            kare_answer = await kare.answer(question.question)
            base_answer = await base.answer(question.question)
        if on_progress is not None:
            on_progress(question.id)
        return QuestionResult(question=question, kare=kare_answer, base=base_answer)

    return list(await asyncio.gather(*(one(question) for question in bank)))


def _ensure_empty(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} ya tiene archivos; usa otra carpeta para no pisar "
            "calificaciones."
        )


def write_run(
    out_dir: Path,
    results: Sequence[QuestionResult],
    *,
    seed: int,
    metadata: Mapping[str, object],
) -> list[Path]:
    _ensure_empty(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    order = assign_order([result.question.id for result in results], seed)
    blind_rows: list[tuple[object, ...]] = []
    key_rows: list[tuple[object, ...]] = []
    citation_rows: list[tuple[object, ...]] = []
    for result in results:
        question = result.question
        first, second = order[question.id]
        blind_rows.append(
            (
                question.id,
                question.question,
                question.reference_answer,
                blinded_text(result.by_arm(first)),
                blinded_text(result.by_arm(second)),
                *([""] * 8),
            )
        )
        key_rows.append((question.id, first, second))
        passages = {passage.chunk_id: passage for passage in result.kare.passages}
        for number, claim in enumerate(result.kare.claims, start=1):
            cited = [passages[cid] for cid in claim.chunk_ids if cid in passages]
            citation_rows.append(
                (
                    question.id,
                    number,
                    claim.text,
                    " | ".join(locator(passage) for passage in cited),
                    "\n---\n".join(passage.text for passage in cited),
                    "",
                    "",
                )
            )

    paths = {name: out_dir / name for name in RUN_FILES}
    write_rows(paths["ciego.csv"], BLIND_HEADER, blind_rows)
    write_rows(paths["clave.csv"], KEY_HEADER, key_rows)
    write_rows(paths["citas.csv"], CITATION_HEADER, citation_rows)
    with paths["salidas.jsonl"].open("w", encoding="utf-8") as fh:
        for result in results:
            for arm in ARMS:
                record = {
                    "id": result.question.id,
                    "brazo": arm,
                    "respondible_esperado": result.question.answerable,
                    "tipo": result.question.kind,
                    "variante_de": result.question.variant_of,
                    "respuesta": result.by_arm(arm).model_dump(mode="json"),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary: dict[str, object] = {
        "fecha": datetime.now(UTC).isoformat(timespec="seconds"),
        "semilla": seed,
        "preguntas": len(results),
        "estados": {
            arm: dict(Counter(result.by_arm(arm).status.value for result in results))
            for arm in ARMS
        },
        "errores": {
            arm: sum(1 for result in results if result.by_arm(arm).error)
            for arm in ARMS
        },
        **metadata,
    }
    paths["corrida.json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return list(paths.values())


async def run_evaluation(
    bank: Sequence[BankQuestion],
    kare: Answerer,
    base: Answerer,
    out_dir: Path,
    *,
    seed: int = 13,
    concurrency: int = 3,
    metadata: Mapping[str, object] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> RunOutcome:
    if not bank:
        raise ValueError("El banco no tiene preguntas.")
    _ensure_empty(out_dir)
    results = await collect(
        bank, kare, base, concurrency=concurrency, on_progress=on_progress
    )
    paths = write_run(out_dir, results, seed=seed, metadata=metadata or {})
    failed = {
        arm: [result.question.id for result in results if result.by_arm(arm).error]
        for arm in ARMS
    }
    return RunOutcome(paths=paths, failed=failed)
