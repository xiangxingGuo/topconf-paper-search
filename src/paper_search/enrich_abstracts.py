"""Enrich papers.csv by visiting each paper URL and filling missing abstracts.

This module is intentionally simple and direct:
- read a normalized papers.csv
- for rows whose abstract is missing, visit row[url]
- extract an abstract from common conference/proceedings HTML patterns
- write the updated CSV back, with a backup if updating in place

Example:
    python -m paper_search.enrich_abstracts \
        --csv data/processed/papers.csv \
        --inplace \
        --min-wait 2.0 \
        --max-wait 6.0 \
        --long-pause-every 100 \
        --long-pause-min 60 \
        --long-pause-max 180 \
        --save-every 50

For testing:
    python -m paper_search.enrich_abstracts \
        --csv data/processed/papers.csv \
        --output data/processed/papers.enriched.sample.csv \
        --max-rows 30
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; TopConfPaperSearch/0.1; "
    "+https://github.com/local/topconf-paper-search)"
)

ABSTRACT_MIN_CHARS = 80
ABSTRACT_MAX_CHARS = 12000

BINARY_EXTENSIONS = (
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".tgz",
    ".ppt",
    ".pptx",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
)

BAD_ABSTRACT_PREFIXES = (
    "click here",
    "download pdf",
    "poster",
    "video",
    "supplementary",
    "openreview",
    "abstract details",
)


@dataclass
class FetchResult:
    url: str
    ok: bool
    status_code: Optional[int] = None
    html_text: str = ""
    error: str = ""


def is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return not str(value).strip()


def clean_text(text: object) -> str:
    if text is None:
        return ""
    s = html.unescape(str(text))
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # Remove common leading label.
    s = re.sub(r"^(abstract|summary)\s*[:\-–—]?\s+", "", s, flags=re.I).strip()
    return s


def looks_like_abstract(text: str) -> bool:
    s = clean_text(text)
    if len(s) < ABSTRACT_MIN_CHARS or len(s) > ABSTRACT_MAX_CHARS:
        return False
    low = s.lower()
    if any(low.startswith(prefix) for prefix in BAD_ABSTRACT_PREFIXES):
        return False
    # Avoid accidentally using full page/navigation blobs.
    nav_words = ["schedule", "sponsors", "registration", "organizers", "code of conduct"]
    if sum(w in low for w in nav_words) >= 3:
        return False
    # Abstracts usually contain sentence punctuation.
    return "." in s or ";" in s


def best_candidate(candidates: Iterable[str]) -> str:
    cleaned = []
    for c in candidates:
        s = clean_text(c)
        if looks_like_abstract(s):
            cleaned.append(s)
    if not cleaned:
        return ""
    # Prefer a medium-length paragraph, not a huge page blob.
    cleaned.sort(key=lambda x: (abs(len(x) - 1200), len(x)))
    return cleaned[0]


def meta_content(soup: BeautifulSoup, names: Iterable[str]) -> list[str]:
    out: list[str] = []
    names_lower = {n.lower() for n in names}
    for meta in soup.find_all("meta"):
        key = ""
        for attr in ("name", "property", "itemprop"):
            if meta.get(attr):
                key = str(meta.get(attr)).lower()
                break
        if key in names_lower and meta.get("content"):
            out.append(str(meta.get("content")))
    return out


def extract_json_like_abstracts(html_text: str) -> list[str]:
    candidates: list[str] = []

    # Common serialized JSON field: "abstract": "..."
    patterns = [
        r'"abstract"\s*:\s*"((?:\\.|[^"\\])*)"',
        r"'abstract'\s*:\s*'((?:\\.|[^'\\])*)'",
        r'"description"\s*:\s*"((?:\\.|[^"\\])*)"',
    ]
    for pat in patterns:
        for m in re.finditer(pat, html_text, flags=re.I | re.S):
            raw = m.group(1)
            try:
                raw = json.loads(f'"{raw}"')
            except Exception:
                raw = raw.encode("utf-8", errors="ignore").decode("unicode_escape", errors="ignore")
            candidates.append(raw)

    # Some pages include escaped HTML/JSON, so try one unescape pass too.
    unescaped = html.unescape(html_text)
    if unescaped != html_text:
        for m in re.finditer(r'"abstract"\s*:\s*"((?:\\.|[^"\\])*)"', unescaped, flags=re.I | re.S):
            raw = m.group(1)
            try:
                raw = json.loads(f'"{raw}"')
            except Exception:
                pass
            candidates.append(raw)

    return candidates


def text_from_element(el: Tag) -> str:
    # Avoid collecting hidden scripts/styles/buttons.
    for bad in el.find_all(["script", "style", "nav", "button", "svg"]):
        bad.decompose()
    return clean_text(el.get_text(" ", strip=True))


def extract_by_selectors(soup: BeautifulSoup) -> list[str]:
    selectors = [
        "#abstract",
        ".abstract",
        ".paper-abstract",
        ".abstract-content",
        "div[class*='abstract']",
        "section[class*='abstract']",
        "p[class*='abstract']",
        "span[class*='abstract']",
        "div#paper-abstract",
        "div.paper-abstract",
        "div.abstractSection",
        "div.abstract-section",
    ]
    candidates: list[str] = []
    for sel in selectors:
        for el in soup.select(sel):
            candidates.append(text_from_element(el))
    return candidates


def extract_after_abstract_heading(soup: BeautifulSoup) -> list[str]:
    candidates: list[str] = []
    heading_tags = ["h1", "h2", "h3", "h4", "h5", "strong", "b", "div", "span"]
    for h in soup.find_all(heading_tags):
        label = clean_text(h.get_text(" ", strip=True)).lower().strip(":")
        if label not in {"abstract", "summary"}:
            continue

        pieces: list[str] = []
        # First, sometimes the abstract is in the same parent right after a strong label.
        parent = h.parent if isinstance(h.parent, Tag) else None
        if parent is not None:
            parent_text = text_from_element(parent)
            parent_text = re.sub(r"^(abstract|summary)\s*[:\-–—]?\s*", "", parent_text, flags=re.I).strip()
            if parent_text and parent_text.lower() not in {"abstract", "summary"}:
                candidates.append(parent_text)

        # Then collect next siblings until another heading-like block.
        for sib in h.next_siblings:
            if not isinstance(sib, Tag):
                continue
            if sib.name in {"h1", "h2", "h3", "h4", "h5"}:
                break
            text = text_from_element(sib)
            if not text:
                continue
            if clean_text(text).lower().strip(":") in {"introduction", "references", "paper", "reviews"}:
                break
            pieces.append(text)
            if sum(len(x) for x in pieces) > 4000:
                break
        if pieces:
            candidates.append(" ".join(pieces))
    return candidates


def extract_openreview_note_content(soup: BeautifulSoup) -> list[str]:
    candidates: list[str] = []
    text = soup.get_text("\n", strip=True)
    # OpenReview pages often render a label "Abstract:" or separate "Abstract" line.
    m = re.search(
        r"(?:^|\n)Abstract\s*[:\n]\s*(.+?)(?:\n(?:Keywords|TL;DR|Primary Area|Submission Number|BibTeX|Reviews|Decision)\b|$)",
        text,
        flags=re.I | re.S,
    )
    if m:
        candidates.append(m.group(1))
    return candidates


def extract_abstract(html_text: str, url: str = "") -> str:
    soup = BeautifulSoup(html_text, "lxml")

    candidates: list[str] = []

    # 1. High-quality metadata used by ACL Anthology, CVF, PMLR, many proceedings sites.
    candidates.extend(
        meta_content(
            soup,
            [
                "citation_abstract",
                "dc.description",
                "dcterms.abstract",
                "description",
                "og:description",
                "twitter:description",
            ],
        )
    )

    # 2. Common HTML containers.
    candidates.extend(extract_by_selectors(soup))

    # 3. Text after an Abstract heading.
    candidates.extend(extract_after_abstract_heading(soup))

    # 4. OpenReview-like rendered pages.
    candidates.extend(extract_openreview_note_content(soup))

    # 5. Serialized JSON inside scripts or hydration data.
    candidates.extend(extract_json_like_abstracts(html_text))

    return best_candidate(candidates)


def should_skip_url(url: object) -> tuple[bool, str]:
    if is_missing(url):
        return True, "missing url"
    u = str(url).strip()
    parsed = urlparse(u)
    if parsed.scheme not in {"http", "https"}:
        return True, f"unsupported scheme: {parsed.scheme}"
    if parsed.path.lower().endswith(BINARY_EXTENSIONS):
        return True, "binary/pdf url"
    return False, ""


def fetch_url(session: requests.Session, url: str, timeout: float, retries: int) -> FetchResult:
    last_error = ""
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, timeout=timeout, allow_redirects=True)
            content_type = resp.headers.get("content-type", "").lower()
            if resp.status_code >= 400:
                return FetchResult(url=url, ok=False, status_code=resp.status_code, error=f"HTTP {resp.status_code}")
            if "text/html" not in content_type and "application/xhtml" not in content_type and content_type:
                return FetchResult(
                    url=url,
                    ok=False,
                    status_code=resp.status_code,
                    error=f"non-html content-type: {content_type}",
                )
            return FetchResult(url=url, ok=True, status_code=resp.status_code, html_text=resp.text)
        except Exception as exc:  # network errors, timeout, SSL, etc.
            last_error = str(exc)
            if attempt < retries:
                time.sleep(1.0 + attempt)
    return FetchResult(url=url, ok=False, error=last_error)


def atomic_write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, output_path)


def enrich_csv(
    csv_path: Path,
    output_path: Path,
    inplace: bool = False,
    backup: bool = True,
    url_col: str = "url",
    abstract_col: str = "abstract",
    max_rows: Optional[int] = None,
    start_row: int = 0,
    delay: float = 1.0,
    jitter: float = 0.3,
    min_wait: Optional[float] = None,
    max_wait: Optional[float] = None,
    long_pause_every: int = 0,
    long_pause_min: float = 30.0,
    long_pause_max: float = 90.0,
    timeout: float = 20.0,
    retries: int = 1,
    save_every: int = 25,
    overwrite_existing: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
) -> None:
    # Important: many parsed CSVs have an empty abstract column. Pandas may infer
    # that column as float64 because empty cells are read as NaN. If we later write
    # a real abstract string into a float64 column, pandas can raise:
    #   TypeError: Invalid value ... for dtype 'float64'
    # So we read normally, then force URL/abstract-related columns to text/object.
    df = pd.read_csv(csv_path)
    if url_col not in df.columns:
        raise ValueError(f"CSV has no URL column named '{url_col}'. Columns: {list(df.columns)}")
    if abstract_col not in df.columns:
        df[abstract_col] = ""

    # Force mutable text columns. Keep NaN semantics usable via is_missing(), but
    # make the destination column capable of storing long strings.
    df[url_col] = df[url_col].astype("object")
    df[abstract_col] = df[abstract_col].astype("object")
    df[abstract_col] = df[abstract_col].where(~df[abstract_col].isna(), "")

    if inplace:
        output_path = csv_path
        if backup:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_path = csv_path.with_suffix(csv_path.suffix + f".bak-{ts}")
            shutil.copy2(csv_path, backup_path)
            print(f"Backup written: {backup_path}")

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"})

    processed = 0
    filled = 0
    failed = 0
    skipped = 0

    candidate_indices = []
    for idx, row in df.iterrows():
        if idx < start_row:
            continue
        if not overwrite_existing and not is_missing(row.get(abstract_col)):
            skipped += 1
            continue
        skip, reason = should_skip_url(row.get(url_col))
        if skip:
            skipped += 1
            continue
        candidate_indices.append(idx)
        if max_rows is not None and len(candidate_indices) >= max_rows:
            break

    total = len(candidate_indices)
    print(f"Rows to visit: {total}")

    for n, idx in enumerate(candidate_indices, start=1):
        url = str(df.at[idx, url_col]).strip()
        title = clean_text(df.at[idx, "title"] if "title" in df.columns else "")
        print(f"[{n}/{total}] row={idx} {title[:80]} | {url}")

        result = fetch_url(session, url, timeout=timeout, retries=retries)
        processed += 1

        if result.ok:
            abstract = extract_abstract(result.html_text, url=url)
            if abstract:
                df.at[idx, abstract_col] = abstract
                filled += 1
                print(f"  OK abstract_len={len(abstract)}")
            else:
                failed += 1
                print("  NO_ABSTRACT_FOUND")
        else:
            failed += 1
            print(f"  FETCH_FAILED {result.error}")

        if save_every > 0 and processed % save_every == 0:
            atomic_write_csv(df, output_path)
            print(f"  Saved progress to {output_path}")

        # Polite randomized waiting. Two modes are supported:
        #   1) default: wait = delay + Uniform(0, jitter)
        #   2) safer:   wait = Uniform(min_wait, max_wait) when both are provided
        if min_wait is not None or max_wait is not None:
            lo = 0.0 if min_wait is None else float(min_wait)
            hi = lo if max_wait is None else float(max_wait)
            if hi < lo:
                raise ValueError(f"--max-wait ({hi}) must be >= --min-wait ({lo})")
            sleep_for = random.uniform(lo, hi)
        else:
            sleep_for = max(0.0, delay + random.uniform(0.0, max(0.0, jitter)))

        if sleep_for:
            print(f"  Sleep {sleep_for:.2f}s")
            time.sleep(sleep_for)

        # Optional longer pause after every N visited URLs. This helps avoid
        # looking like a constant-rate crawler when enriching thousands of rows.
        if long_pause_every and processed % long_pause_every == 0 and n < total:
            lo = min(long_pause_min, long_pause_max)
            hi = max(long_pause_min, long_pause_max)
            pause_for = random.uniform(lo, hi)
            print(f"  Long pause {pause_for:.2f}s after {processed} requests")
            time.sleep(pause_for)

    atomic_write_csv(df, output_path)
    print("Done.")
    print(f"Processed: {processed}; filled: {filled}; failed: {failed}; skipped_existing_or_invalid: {skipped}")
    print(f"Output: {output_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visit paper URLs in papers.csv and fill missing abstracts.")
    p.add_argument("--csv", required=True, type=Path, help="Input papers.csv path.")
    p.add_argument("--output", type=Path, default=None, help="Output CSV path. Required unless --inplace is used.")
    p.add_argument("--inplace", action="store_true", help="Update --csv directly. A backup is created by default.")
    p.add_argument("--no-backup", action="store_true", help="Do not create a backup when using --inplace.")
    p.add_argument("--url-col", default="url")
    p.add_argument("--abstract-col", default="abstract")
    p.add_argument("--max-rows", type=int, default=None, help="Only visit this many rows; useful for testing.")
    p.add_argument("--start-row", type=int, default=0, help="Start from this dataframe row index.")
    p.add_argument("--delay", type=float, default=1.0, help="Base delay between requests, seconds. Used as delay + random(0, jitter) unless --min-wait/--max-wait are set.")
    p.add_argument("--jitter", type=float, default=0.3, help="Extra random delay between 0 and jitter seconds.")
    p.add_argument("--min-wait", type=float, default=None, help="Minimum random wait between requests. If set with --max-wait, overrides --delay/--jitter.")
    p.add_argument("--max-wait", type=float, default=None, help="Maximum random wait between requests. If set with --min-wait, overrides --delay/--jitter.")
    p.add_argument("--long-pause-every", type=int, default=0, help="After every N visited URLs, take a longer random pause. 0 disables it.")
    p.add_argument("--long-pause-min", type=float, default=30.0, help="Minimum seconds for long pauses.")
    p.add_argument("--long-pause-max", type=float, default=90.0, help="Maximum seconds for long pauses.")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--save-every", type=int, default=25, help="Save progress every N visited URLs.")
    p.add_argument("--overwrite-existing", action="store_true", help="Re-fetch even if abstract is already present.")
    p.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    if not args.inplace and args.output is None:
        raise SystemExit("Use either --inplace or --output PATH.")
    output = args.output or args.csv
    enrich_csv(
        csv_path=args.csv,
        output_path=output,
        inplace=args.inplace,
        backup=not args.no_backup,
        url_col=args.url_col,
        abstract_col=args.abstract_col,
        max_rows=args.max_rows,
        start_row=args.start_row,
        delay=args.delay,
        jitter=args.jitter,
        min_wait=args.min_wait,
        max_wait=args.max_wait,
        long_pause_every=args.long_pause_every,
        long_pause_min=args.long_pause_min,
        long_pause_max=args.long_pause_max,
        timeout=args.timeout,
        retries=args.retries,
        save_every=args.save_every,
        overwrite_existing=args.overwrite_existing,
        user_agent=args.user_agent,
    )


if __name__ == "__main__":
    main()
