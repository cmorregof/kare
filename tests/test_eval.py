import csv
from pathlib import Path

import pytest

from app.eval.bank import BANK_COLUMNS, BankError, BankQuestion, load_bank
from app.eval.blind import ABSTENTION_TEXT, assign_order, blinded_text
from app.eval.csvio import read_rows, write_rows
from app.eval.run import BLIND_HEADER, run_evaluation
from app.eval.score import cohen_kappa, render_report, score_run
from app.schemas.evidence import AnswerStatus, FinalAnswer, Passage, VerifiedClaim
from tests.fakes import StaticAnswerer

CHUNK = "guia-a:p12:c0"


def supported(question: str) -> FinalAnswer:
    return FinalAnswer(
        question=question,
        status=AnswerStatus.ANSWERED,
        claims=[
            VerifiedClaim(
                text="Afirmación sustentada.", chunk_ids=[CHUNK], label="supported"
            )
        ],
        passages=[
            Passage(
                chunk_id=CHUNK,
                source_id="guia-a",
                source_title="Guía sintética A",
                source_year=2024,
                page=12,
                language="es",
                text="Texto sintético del pasaje.",
            )
        ],
        model="m",
    )


def unsourced(question: str) -> FinalAnswer:
    return FinalAnswer(
        question=question,
        status=AnswerStatus.ANSWERED,
        claims=[VerifiedClaim(text="Afirmación sin fuente.", label="unverified")],
        model="m",
    )


def abstained(question: str) -> FinalAnswer:
    return FinalAnswer(question=question, status=AnswerStatus.ABSTAINED, reason="Nada.")


def test_assign_order_is_balanced_and_reproducible() -> None:
    ids = [f"Q{i}" for i in range(10)]
    order = assign_order(ids, seed=7)
    assert order == assign_order(ids, seed=7)
    assert sum(1 for first, _ in order.values() if first == "kare") == 5


def test_blinded_text_hides_sources_and_unifies_abstention() -> None:
    assert blinded_text(supported("q")) == "1. Afirmación sustentada."
    assert blinded_text(abstained("q")) == ABSTENTION_TEXT
    blocked = FinalAnswer(question="q", status=AnswerStatus.BLOCKED)
    assert blinded_text(blocked) == ABSTENTION_TEXT


def test_load_bank_skips_examples_and_reads_semicolons(tmp_path: Path) -> None:
    path = tmp_path / "banco.csv"
    path.write_text(
        ";".join(BANK_COLUMNS)
        + "\nEJ-01;Ejemplo;;;si;;;;;\n"
        + "Q1;¿Qué opioide en falla renal?;eleccion_farmaco;dolor;si;Hidromorfona;"
        "Guía p. 12;;;\n"
        + "Q2;¿Supervivencia local este año?;dato_local;pronostico;no;;;;;\n",
        encoding="utf-8-sig",
    )
    bank = load_bank(path)
    assert [question.id for question in bank] == ["Q1", "Q2"]
    assert bank[0].answerable is True
    assert bank[1].answerable is False


def test_load_bank_rejects_identifiers_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "banco.csv"
    empty = ("",) * 8
    write_rows(
        path,
        BANK_COLUMNS,
        [
            ("Q1", "Llamar al 310 555 1234 para control", *empty),
            ("Q2", "¿Dosis de haloperidol en delirium?", *empty),
            ("Q2", "¿Otra pregunta?", *empty),
        ],
    )
    with pytest.raises(BankError) as error:
        load_bank(path)
    message = str(error.value)
    assert "teléfono" in message
    assert "310" not in message
    assert "repetido" in message


def _rate(run_dir: Path) -> Path:
    """Simula a una calificadora que guarda con punto y coma, como Excel en español."""
    key = {
        row["id"]: (row["respuesta_1"], row["respuesta_2"])
        for row in read_rows(run_dir / "clave.csv")
    }
    rated = run_dir / "ciego_karen.csv"
    with rated.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(BLIND_HEADER)
        for row in read_rows(run_dir / "ciego.csv"):
            qid = row["id"]
            should_abstain = "si" if qid == "Q2" else "no"
            cells = dict(row)
            for slot, arm in enumerate(key[qid], start=1):
                if arm == "kare":
                    correct = "" if qid == "Q2" else "2"
                    harm = "ninguno"
                else:
                    correct = "1"
                    harm = "mayor" if qid == "Q1" else "menor"
                cells[f"r{slot}_correcta"] = correct
                cells[f"r{slot}_dano"] = harm
                cells[f"r{slot}_debio_abstenerse"] = should_abstain
            cells["preferida"] = str(key[qid].index("kare") + 1)
            writer.writerow([cells.get(column, "") for column in BLIND_HEADER])
    return rated


async def test_run_and_score_roundtrip(tmp_path: Path) -> None:
    bank = [
        BankQuestion(
            id="Q1", question="Pregunta uno", answerable=True, reference_answer="Ref"
        ),
        BankQuestion(id="Q2", question="Pregunta dos", answerable=False),
    ]
    kare = StaticAnswerer(
        {
            "Pregunta uno": supported("Pregunta uno"),
            "Pregunta dos": abstained("Pregunta dos"),
        }
    )
    base = StaticAnswerer(
        {
            "Pregunta uno": unsourced("Pregunta uno"),
            "Pregunta dos": unsourced("Pregunta dos"),
        }
    )
    run_dir = tmp_path / "r1"
    paths = await run_evaluation(
        bank, kare, base, run_dir, seed=5, metadata={"modelo": "m"}
    )
    assert {path.name for path in paths} == {
        "ciego.csv",
        "clave.csv",
        "citas.csv",
        "salidas.jsonl",
        "corrida.json",
    }
    blind = (run_dir / "ciego.csv").read_text(encoding="utf-8-sig")
    for leak in ("kare", "base", "Guía sintética", "[1]"):
        assert leak not in blind
    assert ABSTENTION_TEXT in blind

    citations = run_dir / "citas_karen.csv"
    rows = read_rows(run_dir / "citas.csv")
    assert len(rows) == 1 and "Guía sintética A (2024), p. 12" in rows[0]["fuentes"]
    write_rows(
        citations,
        list(rows[0].keys()),
        [[*(row | {"soporta": "sí"}).values()] for row in rows],
    )

    rated = _rate(run_dir)
    report = score_run(run_dir, [rated], [citations])
    kare_score, base_score = report.arms["kare"], report.arms["base"]
    assert kare_score.mean_correctness == 2.0
    assert base_score.mean_correctness == 1.0
    assert base_score.harm["mayor"] == 1
    assert kare_score.abstention_agreement == 1.0
    assert base_score.abstention_agreement == 0.5
    assert (kare_score.abstained_on_unanswerable, kare_score.unanswerable) == (1, 1)
    assert base_score.abstained_on_unanswerable == 0
    assert kare_score.preferred == 2
    assert report.citation_precision == 1.0
    text = render_report(report)
    assert "- [x] Ningún error con daño potencial mayor en KARE-E" in text
    assert "- [x] Precisión de citas" in text

    double = score_run(run_dir, [rated, rated])
    assert double.agreement == 1.0
    assert double.kappa == pytest.approx(1.0)

    with pytest.raises(FileExistsError):
        await run_evaluation(bank, kare, base, run_dir)


def test_cohen_kappa() -> None:
    assert cohen_kappa(["a", "b"], ["b", "a"]) == pytest.approx(-1.0)
    assert cohen_kappa(["a", "a"], ["a", "a"]) is None
    assert cohen_kappa([], []) is None
