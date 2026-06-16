from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .parsers import parse_file
from .schema import PAPER_COLUMNS
from .utils import guess_conference_year_from_path, iter_html_files, safe_year


def ingest_html(
    input_path: str | Path,
    *,
    conference: str = "",
    year: int | str = "",
    base_url: str = "",
    parser_hint: str = "auto",
) -> pd.DataFrame:
    files = iter_html_files(input_path)
    rows: list[dict] = []
    for file in files:
        guessed_conf, guessed_year = guess_conference_year_from_path(file)
        conf = conference or guessed_conf
        yr = safe_year(year or guessed_year)
        if not conf or not yr:
            raise ValueError(
                f"Cannot determine conference/year for {file}. "
                "Pass --conference and --year, or store files as data/html/CONF/YEAR/*.html."
            )
        records = parse_file(file, conf, yr, base_url=base_url, parser_hint=parser_hint)
        rows.extend(record.to_dict() for record in records)

    df = pd.DataFrame(rows, columns=PAPER_COLUMNS)
    if not df.empty:
        df = normalize_dataframe(df)
    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for column in PAPER_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df = df[PAPER_COLUMNS].copy()
    for column in ["conference", "title", "abstract", "authors", "url", "pdf_url", "source_file", "parser"]:
        df[column] = df[column].fillna("").astype(str).str.strip()
    df = df[df["title"].astype(bool)]
    df = df.drop_duplicates(subset=["conference", "year", "title"], keep="first")
    return df.reset_index(drop=True)


def save_papers_csv(df: pd.DataFrame, output: str | Path, *, append: bool = False) -> pd.DataFrame:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if append and output_path.exists():
        old = pd.read_csv(output_path)
        combined = pd.concat([old, df], ignore_index=True)
        combined = normalize_dataframe(combined)
    else:
        combined = normalize_dataframe(df)
    combined.to_csv(output_path, index=False)
    return combined


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse saved conference HTML into a normalized paper CSV.")
    parser.add_argument("--input", required=True, help="HTML file or directory containing .html/.htm files")
    parser.add_argument("--conference", default="", help="Conference name, e.g. ICML, NeurIPS, ACL. Optional if path includes CONF/YEAR.")
    parser.add_argument("--year", default="", help="Conference year. Optional if path includes CONF/YEAR.")
    parser.add_argument("--base-url", default="", help="Base URL used to resolve relative links in saved HTML")
    parser.add_argument(
        "--parser",
        default="auto",
        choices=["auto", "iclr_virtual", "cvf", "pmlr", "acl", "openreview", "generic"],
        help="Parser hint. 'auto' tries all known parser families.",
    )
    parser.add_argument("--output", default="data/processed/papers.csv", help="Output CSV path")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV and deduplicate")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    df = ingest_html(
        args.input,
        conference=args.conference,
        year=args.year,
        base_url=args.base_url,
        parser_hint=args.parser,
    )
    combined = save_papers_csv(df, args.output, append=args.append)
    print(f"Parsed {len(df)} paper rows from {args.input}")
    print(f"Saved {len(combined)} total rows to {args.output}")
    if not combined.empty:
        print(combined.groupby(["conference", "year"]).size().to_string())


if __name__ == "__main__":
    main()
