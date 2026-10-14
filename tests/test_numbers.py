from app.answer.numbers import missing_numbers, strip_citation_marks


def test_strip_citation_marks() -> None:
    assert strip_citation_marks("Evitar la morfina [P1].") == "Evitar la morfina."
    assert strip_citation_marks("Según P2, usar ventilador [P1, P3].") == (
        "Según, usar ventilador."
    )


def test_decimal_comma_matches_decimal_point() -> None:
    assert missing_numbers("2,5 mg cada 8 horas", ["2.5 mg every 8 hours"]) == []


def test_thousands_separator_matches_plain_number() -> None:
    assert missing_numbers("1.000 mg al día", ["1000 mg daily"]) == []


def test_invented_number_is_reported() -> None:
    assert missing_numbers("15 mg cada 4 horas", ["10 mg cada 4 horas"]) == ["15"]


def test_citation_labels_are_not_numbers() -> None:
    assert missing_numbers("Usar ventilador [P2].", ["Un ventilador ayuda."]) == []


def test_numbers_from_source_metadata_count() -> None:
    assert missing_numbers("La guía de 2018 lo recomienda.", ["Guía X 2018"]) == []
