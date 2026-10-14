"""CSV pensado para hojas de cálculo.

Se escribe en UTF-8 con BOM para que Excel muestre bien las tildes. Al leer se
acepta coma, punto y coma o tabulador (Excel en español guarda con punto y
coma), se normalizan los encabezados y se ignoran las filas vacías.
"""

import csv
import io
from collections.abc import Iterable, Sequence
from pathlib import Path

_DELIMITERS = (",", ";", "\t")


def read_rows(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []
    header_line = text.splitlines()[0]
    delimiter = max(_DELIMITERS, key=header_line.count)
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    header = [cell.strip().lower() for cell in rows[0]]
    records: list[dict[str, str]] = []
    for raw in rows[1:]:
        if not any(cell.strip() for cell in raw):
            continue
        cells = list(raw) + [""] * (len(header) - len(raw))
        records.append(
            {name: cells[i].strip() for i, name in enumerate(header) if name}
        )
    return records


def write_rows(
    path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if value is None else value for value in row])
