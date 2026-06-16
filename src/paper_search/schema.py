from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


PAPER_COLUMNS = [
    "paper_id",
    "conference",
    "year",
    "title",
    "abstract",
    "authors",
    "url",
    "pdf_url",
    "source_file",
    "parser",
]


@dataclass(slots=True)
class PaperRecord:
    paper_id: str
    conference: str
    year: int | str
    title: str
    abstract: str = ""
    authors: str = ""
    url: str = ""
    pdf_url: str = ""
    source_file: str = ""
    parser: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {column: data.get(column, "") for column in PAPER_COLUMNS}
