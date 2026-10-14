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


def test_thousand_fold_errors_are_caught() -> None:
    # Hallazgos de la revisión independiente: estos pasaban en v0.1.
    passage = "hasta 1 g cada 6 horas; máximo 4 g/día"
    assert missing_numbers("Paracetamol hasta 6.000 mg al día.", [passage]) == ["6.000"]
    assert missing_numbers("1 mg", ["1.000 mg"]) == ["1"]
    assert missing_numbers("2.400 UI", ["2.400.000 UI"]) == ["2.400"]
    assert missing_numbers("0,125 mg", ["125 mg"]) == ["0,125"]
    assert missing_numbers("0,125 mg", ["0.125 mg"]) == []
    assert missing_numbers("1.000,5 ml", ["1,000.5 ml"]) == []


def test_quantities_in_words_are_checked() -> None:
    assert missing_numbers("Morfina cada ocho horas.", ["morfina cada 4 horas"]) == [
        "ocho"
    ]
    assert missing_numbers("Morfina cada ocho horas.", ["every eight hours"]) == []
    assert missing_numbers("Dos veces al día.", ["twice daily"]) == []
    assert missing_numbers("Una revisión de dos estudios.", ["Una revisión."]) == []


def test_passage_mentions_are_removed() -> None:
    assert strip_citation_marks("Según el pasaje, la morfina se evita (P1).") == (
        "La morfina se evita."
    )


def test_numbers_from_source_metadata_count() -> None:
    assert missing_numbers("La guía de 2018 lo recomienda.", ["Guía X 2018"]) == []


def test_drug_and_gene_names_are_not_citation_marks() -> None:
    assert strip_citation_marks("Metabolizado por el citocromo P450.") == (
        "Metabolizado por el citocromo P450."
    )
    assert strip_citation_marks("Inhibe el receptor P2Y12 [P1].") == (
        "Inhibe el receptor P2Y12."
    )
