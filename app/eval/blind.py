"""Cegamiento de la comparación KARE-E frente al modelo solo.

Las dos respuestas se muestran con el mismo formato: afirmaciones numeradas,
sin citas ni fuentes, y la misma frase cuando cualquiera de los dos se
abstiene. El cegamiento es parcial: el estilo de una respuesta puede delatar
su origen. Las citas de KARE-E se califican aparte, después.
"""

import random
from collections.abc import Sequence
from typing import Literal

from app.schemas.evidence import AnswerStatus, FinalAnswer

Arm = Literal["kare", "base"]
ARMS: tuple[Arm, Arm] = ("kare", "base")

ABSTENTION_TEXT = "No puedo responder esta pregunta con suficiente seguridad."


def as_arm(value: str) -> Arm:
    if value == "kare":
        return "kare"
    if value == "base":
        return "base"
    raise ValueError(f"Brazo desconocido: {value!r}")


def is_abstention(answer: FinalAnswer) -> bool:
    return (
        answer.status in {AnswerStatus.ABSTAINED, AnswerStatus.BLOCKED}
        or not answer.claims
    )


def blinded_text(answer: FinalAnswer) -> str:
    if is_abstention(answer):
        return ABSTENTION_TEXT
    return "\n".join(
        f"{number}. {claim.text}" for number, claim in enumerate(answer.claims, start=1)
    )


def assign_order(question_ids: Sequence[str], seed: int) -> dict[str, tuple[Arm, Arm]]:
    """Orden balanceado y reproducible: la mitad ve KARE-E como respuesta 1."""
    rng = random.Random(seed)
    half = len(question_ids) // 2
    kare_first = [True] * half + [False] * (len(question_ids) - half)
    rng.shuffle(kare_first)
    order: dict[str, tuple[Arm, Arm]] = {}
    for qid, first in zip(question_ids, kare_first, strict=True):
        order[qid] = ("kare", "base") if first else ("base", "kare")
    return order
