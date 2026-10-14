import pytest

from app.guard.identifiers import IdentifierKind, describe, redact, scan


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("paciente con CC 1053456789 y dolor", IdentifierKind.DOCUMENT_ID),
        ("cédula 24.345.678", IdentifierKind.DOCUMENT_ID),
        ("su documento es 24.345.678", IdentifierKind.DOCUMENT_ID),
        ("número 1053456789", IdentifierKind.DOCUMENT_ID),
        ("llamar al 310 555 1234", IdentifierKind.PHONE),
        ("contacto +57 310-555-1234", IdentifierKind.PHONE),
        ("correo ana.perez@gmail.com", IdentifierKind.EMAIL),
        ("nació el 03/05/1950", IdentifierKind.DATE),
        ("ingresó el 12 de marzo de 2024", IdentifierKind.DATE),
        ("control 2024-03-12", IdentifierKind.DATE),
        ("HC 123456", IdentifierKind.RECORD_NUMBER),
        ("historia clínica n° 98765", IdentifierKind.RECORD_NUMBER),
        ("vive en la Calle 23 # 45-67", IdentifierKind.ADDRESS),
        ("vive en la Cra 5 No. 12-30", IdentifierKind.ADDRESS),
        ("la paciente Rosa Gómez tiene disnea", IdentifierKind.PERSON_NAME),
        ("doña Rosa refiere dolor", IdentifierKind.PERSON_NAME),
        ("el señor Pérez no come", IdentifierKind.PERSON_NAME),
    ],
)
def test_detects_identifiers(text: str, kind: IdentifierKind) -> None:
    assert kind in {finding.kind for finding in scan(text)}


@pytest.mark.parametrize(
    "text",
    [
        "¿Qué opioide se prefiere en insuficiencia renal con TFG de 25 ml/min?",
        "Paciente de 75 años con cáncer de páncreas y ascitis, ¿cuándo paracentesis?",
        "Infundir 1.500 ml en 24 horas por vía subcutánea",
        "morfina 2,5 mg cada 4 horas",
        "Paciente ECOG 3 con disnea en reposo",
        "¿Qué dice la Resolución 2665 del 25 de junio de 2018?",
        "Ley 1733 de 2014 y cuidados paliativos",
        "¿Qué recomienda la guía de la OMS de 2018 para dolor irruptivo?",
        "Código CUPS 890201 para consulta de cuidado paliativo",
        "Escala PPS de 30 % y PaP score",
    ],
)
def test_clinical_questions_pass(text: str) -> None:
    assert scan(text) == []


def test_redact_and_describe_never_echo_the_value() -> None:
    text = "CC 1053456789 y doña Rosa, celular 3105551234"
    findings = scan(text)
    redacted = redact(text, findings)
    message = describe(findings)
    for secret in ("1053456789", "Rosa", "3105551234"):
        assert secret not in redacted
        assert secret not in message
    assert "[DOCUMENTO]" in redacted
    assert "[NOMBRE]" in redacted


def test_findings_do_not_overlap() -> None:
    findings = scan("HC 123456")
    assert len(findings) == 1
    assert findings[0].kind == IdentifierKind.RECORD_NUMBER
