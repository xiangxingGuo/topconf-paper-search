from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urljoin


def clean_text(value: object) -> str:
    """Normalize whitespace and common HTML entities."""
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_author_text(value: object) -> str:
    text = clean_text(value)
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return text.strip(" ,;")


def normalize_url(href: str | None, base_url: str = "") -> str:
    href = clean_text(href)
    if not href:
        return ""
    if base_url:
        return urljoin(base_url, href)
    return href


def stable_paper_id(conference: str, year: int | str, title: str, url: str = "") -> str:
    raw = f"{clean_text(conference).lower()}|{year}|{clean_text(title).lower()}|{clean_text(url)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def read_html(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="latin-1")


def iter_html_files(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    suffixes = {".html", ".htm"}
    return sorted(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in suffixes)


def guess_conference_year_from_path(path: str | Path) -> tuple[str, str]:
    """Best-effort fallback: /.../ICML/2024/page.html -> (ICML, 2024)."""
    parts = [p for p in Path(path).parts if p]
    year = ""
    conference = ""
    for part in reversed(parts):
        if not year and re.fullmatch(r"20\d{2}", part):
            year = part
            continue
        if year and not conference:
            conference = re.sub(r"[^A-Za-z0-9_-]", "", part).upper()
            break
    return conference, year


def split_people(value: object) -> list[str]:
    text = clean_author_text(value)
    if not text:
        return []
    if ";" in text:
        return [clean_text(x) for x in text.split(";") if clean_text(x)]
    return [clean_text(x) for x in text.split(",") if clean_text(x)]


def looks_like_author_line(text: str) -> bool:
    """Heuristic to avoid treating author lists as paper titles."""
    text = clean_text(text)
    if not text or not re.search(r"[,;]", text):
        return False
    if re.search(r"[:?!]|\b(for|with|via|using|towards?|toward|from|under|over|through)\b", text, re.I):
        return False
    parts = [p.strip() for p in re.split(r"\s*(?:,|;|\band\b)\s*", text) if p.strip()]
    if len(parts) < 2 or len(parts) > 20:
        return False
    total_words = len(re.findall(r"[A-Za-z]+", text))
    if total_words > 40:
        return False
    name_like = 0
    for part in parts:
        words = re.findall(r"[A-Za-z][A-Za-z.'’-]*", part)
        if 1 <= len(words) <= 4 and all(re.match(r"^[A-ZÁÉÍÓÚÜÑÄÖ][A-Za-zÁÉÍÓÚÜÑÄÖáéíóúüñäö.'’-]*$", w) for w in words):
            name_like += 1
    return name_like / len(parts) >= 0.75


def is_probable_title(text: str) -> bool:
    text = clean_text(text)
    if len(text) < 8 or len(text) > 280:
        return False
    lower = text.lower().strip(" :")
    blocked_exact = {
        "accepted papers",
        "accepted main conference papers",
        "long papers",
        "short papers",
        "best papers",
        "calls",
        "program",
        "registration",
        "committees",
        "sponsors",
        "participants info",
        "faq",
        "tutorials",
        "workshops",
        "conference overview",
        "download pdf",
        "openreview",
        "abstract",
        "bibtex",
    }
    if lower in blocked_exact:
        return False
    if lower.startswith(("proceedings of ", "volume ", "filter authors", "filter titles")):
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    if looks_like_author_line(text):
        return False
    # Real paper titles normally have several words. Allow compact acronym-heavy titles too.
    word_count = len(re.findall(r"[A-Za-z0-9]+", text))
    return word_count >= 3


def safe_year(value: object) -> int | str:
    text = clean_text(value)
    if re.fullmatch(r"20\d{2}", text):
        return int(text)
    return text
