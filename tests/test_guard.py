"""La guardia, probada con entradas colombianas realistas.

Todos los datos son inventados. Buena parte de los casos viene de la revisión
independiente del 14 de octubre de 2026, que encontró fugas reales en v0.1.
"""

import pytest

from app.guard.identifiers import IdentifierKind, describe, redact, scan

MUST_BLOCK = [
    # Documento, también seguido de puntuación normal
    "paciente con CC 1053456789 y dolor",
    "su cédula es 1053456789, tiene dolor óseo",
    "identificada con 1053456789, ¿qué opioide?",
    "su documento es 24.345.678, refiere disnea",
    "cédula número 24345678, con ascitis",
    "el paciente 1053456789, con dolor",
    "Paciente de 70 años, su cédula es 1053456789, ¿qué opioide?",
    "TI 98052312345 niño con leucemia",
    "tarjeta 98052312345",
    "CC 1,053,456,789 con dolor",
    "pasaporte AB12345",
    "PPT 1234567",
    "Según la guía, el paciente 1053456789 tiene dolor",
    "Ley 1733 y el documento del paciente: 1053456789",
    "Código interno 1053456789 de la paciente",
    # Teléfonos
    "llamar al 310 555 1234, gracias",
    "cel 3105551234, hija",
    "cel.3105551234",
    "tel.6068812345",
    "celular 310.555.1234",
    "celular 310 555 12 34",
    "celular (310) 555-1234",
    "contacto +57 310-555-1234",
    "fijo (606) 881 2345",
    "fijo 881 2345",
    "fijo 881-2345",
    "+57 606 881 2345",
    # Correo
    "correo rosa_gomez@hotmail.com",
    # Fechas
    "nació el 03/05/1950, tiene dolor",
    "ingresó el 3 de mayo",
    "ingresó el 12 de marzo de 2024",
    "ingresó el 3/may/2024",
    "ingresó el 3-may-2024",
    "ingresó el 03-MAY-24",
    "ingresó el 3 may 2024",
    "ingresó el 3 mayo 2024",
    "ingresó mayo 3 de 2024",
    "ingresó el 15 de dic.",
    "ingresó el 2024/03/12",
    "control 2024-03-12",
    "ingresó el 12 / 03 / 2024",
    "ingresó el 3/05",
    "Por la Resolución 2665, nació el 03/05/1950",
    # Historia o ingreso
    "HC 123456",
    "historia clínica número 54321",
    "HC número 54321",
    "historia clínica No. 54321",
    # Direcciones
    "vive en la Cra 23 #45-67",
    "vive en la Calle 23 # 45-67",
    "vive en la Calle 5 Sur # 10-20",
    "vive en la Carrera 23 Este # 45-67",
    "vive en la Carrera 23 número 45-67",
    "vive en la Avenida Santander # 52-14",
    "vive en la AK 7 # 32-16",
    "vive en Mz 5 Casa 12 barrio La Enea",
    "vive en la Calle 65A Sur # 10-20",
    # Nombres
    "paciente Rosa Gómez",
    "paciente: Rosa Gómez, 68 años",
    "Paciente: ROSA GÓMEZ PÉREZ",
    "paciente ROSA GOMEZ con dolor",
    "la usuaria Rosa Gómez",
    "Nombre: Rosa Gómez Pérez",
    "doña Rosa refiere dolor",
    "el señor Pérez no come",
    "sra. Gómez",
    "Sr.Pérez",
    "la paciente María de los Ángeles Gómez tiene disnea",
]

MUST_PASS = [
    "¿Qué opioide se prefiere en insuficiencia renal con TFG de 25 ml/min?",
    "Paciente de 75 años con cáncer de páncreas y ascitis, ¿cuándo paracentesis?",
    "Infundir 1.500 ml en 24 horas por vía subcutánea",
    "morfina 2,5 mg cada 4 horas",
    "creatinina 1,8 y TFG 25",
    "plaquetas 150.000",
    "plaquetas 150000",
    "leucocitos 12.500",
    "leucocitos 125000",
    "nistatina 100000 UI/ml 5 ml cada 6 horas",
    "nistatina 100.000 UI/ml",
    "penicilina benzatínica 2.400.000 UI",
    "Paciente ECOG 3 con disnea en reposo",
    "Escala PPS de 30 % y PaP score",
    "Ley 1733 de 2014 y cuidados paliativos",
    "¿Qué dice la Resolución 2665 del 25 de junio de 2018?",
    "Resolución 2665 de 2018, expedida el 25 de junio de 2018",
    "¿Qué recomienda la guía de la OMS de 2018 para dolor irruptivo?",
    "Código CUPS 890201 para consulta de cuidado paliativo",
    "¿Qué opioide en paciente Child-Pugh C?",
    "Paciente Karnofsky 40 % con disnea",
    "¿Sedación paliativa en paciente Alzheimer avanzado?",
    "Tengo una paciente. Tiene cáncer de mama con metástasis óseas.",
    "morfina 5 mg a las 6-12-18-24 horas",
    "metadona cada 8 horas (6-14-22)",
    "supervivencia a 3-6-12 meses según PPI",
    "titulación de gabapentina 300-600-1200 mg",
    "gabapentina 300 600 1200 mg/día",
    "incidencia 1 en 100000 habitantes",
    "Hb 7,5 y albúmina 2,1",
    "media tableta: 1/2 cada 12 horas",
    "a los 80 años",
]


@pytest.mark.parametrize("text", MUST_BLOCK)
def test_identifiers_are_blocked(text: str) -> None:
    assert scan(text), f"fuga: {text!r}"


@pytest.mark.parametrize("text", MUST_PASS)
def test_clinical_questions_pass(text: str) -> None:
    assert scan(text) == [], f"falso positivo: {text!r}"


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("cédula 24.345.678", IdentifierKind.DOCUMENT_ID),
        ("llamar al 310 555 1234", IdentifierKind.PHONE),
        ("correo ana.perez@gmail.com", IdentifierKind.EMAIL),
        ("nació el 03/05/1950", IdentifierKind.DATE),
        ("HC 123456", IdentifierKind.RECORD_NUMBER),
        ("vive en la Calle 23 # 45-67", IdentifierKind.ADDRESS),
        ("doña Rosa refiere dolor", IdentifierKind.PERSON_NAME),
    ],
)
def test_kinds_are_named(text: str, kind: IdentifierKind) -> None:
    assert kind in {finding.kind for finding in scan(text)}


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


def test_overlapping_findings_are_redacted_together() -> None:
    text = "paciente Rosa Gómez Calle 23 # 45-67"
    redacted = redact(text, scan(text))
    assert redacted == "paciente [NOMBRE]"


def test_contained_findings_collapse() -> None:
    findings = scan("HC 123456")
    assert len(findings) == 1
    assert findings[0].kind == IdentifierKind.RECORD_NUMBER
