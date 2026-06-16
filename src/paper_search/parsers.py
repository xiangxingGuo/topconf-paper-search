from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from .schema import PaperRecord
from .utils import clean_author_text, clean_text, is_probable_title, normalize_url, stable_paper_id


TITLE_SELECTORS = [
    "dt.ptitle",
    "p.title",
    ".paper-title",
    ".maincardBody .maincardBodyTitle",
    ".note_content_title",
    "h1.title",
    "h2.title",
    "h3.title",
    "h4.title",
]

ABSTRACT_SELECTORS = [
    ".abstract",
    "p.abstract",
    "div.abstract",
    ".note_content_abstract",
    "#abstract",
]

VIRTUAL_HOSTS = {
    "ICLR": "https://iclr.cc",
    "ICML": "https://icml.cc",
    "NEURIPS": "https://nips.cc",
    "NIPS": "https://nips.cc",
    "MLSYS": "https://mlsys.org",
}


def parse_html(
    html: str,
    conference: str,
    year: int | str,
    *,
    base_url: str = "",
    source_file: str = "",
    parser_hint: str = "auto",
) -> list[PaperRecord]:
    """Parse one saved conference HTML page into normalized paper records.

    This file is designed for your current saved-HTML layout, e.g.

        data/html/ICLR/ICLR 2025 Papers.html
        data/html/ICML/ICML 2025 Papers.html
        data/html/NeurIPS/NeurIPS 2024 Papers.html
        data/html/ACL/ACL 2024.html
        data/html/EMNLP/EMNLP 2024.html

    Main parser families:
      - conference_virtual: iclr.cc / icml.cc / nips.cc / mlsys.org virtual paper pages
      - acl: ACL/EMNLP official accepted-paper pages using <strong>title</strong> + authors
      - cvf: CVPR/ICCV CVF OpenAccess pages
      - pmlr: PMLR proceedings pages
      - openreview: OpenReview JSON/visible note pages
      - generic: fallback for simple accepted-paper lists
    """
    soup = BeautifulSoup(html, "html.parser")
    parser_hint = (parser_hint or "auto").lower().strip()

    mapping: dict[str, Any] = {
        "conference_virtual": parse_conference_virtual,
        "virtual": parse_conference_virtual,
        "iclr_virtual": parse_conference_virtual,  # backward compatible with previous version
        "icml_virtual": parse_conference_virtual,
        "neurips_virtual": parse_conference_virtual,
        "nips_virtual": parse_conference_virtual,
        "mlsys_virtual": parse_conference_virtual,
        "openreview": parse_openreview,
        "cvf": parse_cvf,
        "pmlr": parse_pmlr,
        "acl": parse_acl_style,
        "emnlp": parse_acl_style,
        "generic": parse_generic,
    }

    if parser_hint == "auto":
        parsers: list[tuple[str, Any]] = [
            ("conference_virtual", parse_conference_virtual),
            ("cvf", parse_cvf),
            ("pmlr", parse_pmlr),
            ("acl", parse_acl_style),
            ("openreview", parse_openreview),
            ("generic", parse_generic),
        ]
    else:
        if parser_hint not in mapping:
            raise ValueError(f"Unknown parser hint: {parser_hint}. Available: {', '.join(sorted(mapping))}")
        parsers = [(parser_hint, mapping[parser_hint])]

    candidates: list[PaperRecord] = []
    for parser_name, parser_fn in parsers:
        parsed = parser_fn(soup, conference, year, base_url=base_url, source_file=source_file)
        if parsed:
            candidates.extend(parsed)
            # Specific parsers are more reliable than the generic fallback.
            if parser_name != "generic":
                break

    return dedupe_records(candidates)


def make_record(
    *,
    conference: str,
    year: int | str,
    title: str,
    abstract: str = "",
    authors: str | Iterable[str] = "",
    url: str = "",
    pdf_url: str = "",
    source_file: str = "",
    parser: str = "",
    trust_title: bool = False,
) -> PaperRecord | None:
    title = clean_text(title)
    title = strip_trailing_noise(title)
    if trust_title:
        if not title or len(title) > 350 or not re.search(r"[A-Za-z]", title):
            return None
        if is_bad_title(title):
            return None
    elif not is_probable_title(title) or is_bad_title(title):
        return None

    if isinstance(authors, str):
        authors_text = clean_author_text(authors)
    else:
        authors_text = clean_author_text("; ".join(clean_text(x) for x in authors if clean_text(x)))

    url = clean_text(url)
    pdf_url = clean_text(pdf_url)
    return PaperRecord(
        paper_id=stable_paper_id(conference, year, title, url),
        conference=clean_text(conference).upper(),
        year=year,
        title=title,
        abstract=clean_text(abstract),
        authors=authors_text,
        url=url,
        pdf_url=pdf_url,
        source_file=str(source_file),
        parser=parser,
    )


def dedupe_records(records: list[PaperRecord]) -> list[PaperRecord]:
    seen: dict[tuple[str, str, str], PaperRecord] = {}
    for rec in records:
        key = (rec.conference, str(rec.year), rec.title.lower())
        old = seen.get(key)
        if old is None:
            seen[key] = rec
            continue
        old_score = bool(old.abstract) + bool(old.authors) + bool(old.url) + bool(old.pdf_url)
        new_score = bool(rec.abstract) + bool(rec.authors) + bool(rec.url) + bool(rec.pdf_url)
        if new_score > old_score:
            seen[key] = rec
    return list(seen.values())


def strip_trailing_noise(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\s+\|\s+.*$", "", text)
    text = re.sub(r"\s+Download\s+PDF\s*$", "", text, flags=re.I)
    return text.strip(" -–—\t\n\r")


def is_bad_title(title: str) -> bool:
    t = clean_text(title).lower()
    bad_exact = {
        "login", "logout", "help/faq", "contact", "privacy policy", "code of conduct",
        "calls", "program", "registration", "participants info", "accepted", "award",
        "events", "long papers", "short papers", "main conference", "on this page",
        "conference overview", "workshops", "tutorials", "best papers", "feed",
    }
    if t in bad_exact:
        return True
    nav_phrases = [
        "select year:", "add/remove bookmark", "filter all papers", "i agree",
        "we use cookies", "skip to", "powered by", "all rights reserved",
    ]
    return any(p in t for p in nav_phrases)


def text_of_first(element: Tag, selectors: list[str]) -> str:
    for selector in selectors:
        node = element.select_one(selector)
        if node:
            return clean_text(node.get_text(" ", strip=True))
    return ""


def href_of_first(element: Tag, selectors: list[str], base_url: str = "") -> str:
    for selector in selectors:
        node = element.select_one(selector)
        if node and node.has_attr("href"):
            return normalize_url(str(node.get("href")), base_url)
    return ""


def find_pdf_url(element: Tag, base_url: str = "") -> str:
    for a in element.find_all("a", href=True):
        href = str(a.get("href"))
        label = clean_text(a.get_text(" ", strip=True)).lower()
        if href.lower().endswith(".pdf") or label == "pdf" or " pdf" in f" {label} ":
            return normalize_url(href, base_url)
    return ""


def nearby_abstract(element: Tag) -> str:
    for selector in ABSTRACT_SELECTORS:
        node = element.select_one(selector)
        if node:
            text = clean_text(node.get_text(" ", strip=True))
            return clean_text(re.sub(r"^abstract\s*[:\-]?\s*", "", text, flags=re.I))

    sibling = element.next_sibling
    steps = 0
    while sibling is not None and steps < 5:
        if isinstance(sibling, Tag):
            txt = clean_text(sibling.get_text(" ", strip=True))
            if re.match(r"^abstract\s*[:\-]", txt, re.I):
                return clean_text(re.sub(r"^abstract\s*[:\-]?\s*", "", txt, flags=re.I))
            cls = " ".join(str(x).lower() for x in sibling.get("class", []))
            if "abstract" in cls:
                return clean_text(txt)
        sibling = sibling.next_sibling
        steps += 1
    return ""


# ---------------------------------------------------------------------------
# iclr.cc / icml.cc / nips.cc / mlsys.org virtual paper pages
# ---------------------------------------------------------------------------


def default_virtual_base(conference: str, base_url: str) -> str:
    if base_url:
        return base_url
    return VIRTUAL_HOSTS.get(clean_text(conference).upper(), "")


def is_virtual_poster_href(href: str) -> bool:
    if not href:
        return False
    parsed = urlparse(href)
    path = parsed.path or href
    # Exclude login links like /accounts/login?nextp=/virtual/2025/poster/...
    return "/virtual/" in path and "/poster/" in path


def parse_conference_virtual(
    soup: BeautifulSoup,
    conference: str,
    year: int | str,
    *,
    base_url: str = "",
    source_file: str = "",
) -> list[PaperRecord]:
    """Parse ICLR/ICML/NeurIPS/MLSys virtual paper pages.

    These pages typically contain:
      - 400 visible pp-card entries with title + authors;
      - a <noscript> list containing the complete accepted-paper list;
      - JS references to orals-posters.json and abstracts.json.

    The saved HTML often does not embed all abstracts, so abstract may remain empty.
    """
    html_head = str(soup)[:50000].lower()
    if "/virtual/" not in html_head and not soup.select('a[href*="/virtual/"][href*="/poster/"]'):
        return []

    resolved_base = default_virtual_base(conference, base_url)
    records_by_key: dict[str, PaperRecord] = {}

    # 1) Rich visible cards. Usually first 400 papers only, but with authors/session.
    for card in soup.select("div.pp-card-header, div.pp-card"):
        paper_link = None
        for a in card.find_all("a", href=True):
            href = str(a.get("href", ""))
            if is_virtual_poster_href(href):
                paper_link = a
                break
        if not paper_link:
            continue

        href = str(paper_link.get("href"))
        url = normalize_url(href, resolved_base)
        title_node = paper_link.select_one("h5.card-title, .card-title")
        title = clean_text(title_node.get_text(" ", strip=True) if title_node else paper_link.get_text(" ", strip=True))

        author_nodes = card.select('h6.card-subtitle a[href*="filter=author"], .card-subtitle a[href*="filter=author"]')
        authors = "; ".join(clean_text(a.get_text(" ", strip=True)) for a in author_nodes if clean_text(a.get_text(" ", strip=True)))

        rec = make_record(
            conference=conference,
            year=year,
            title=title,
            authors=authors,
            url=url,
            source_file=source_file,
            parser="conference-virtual-card",
            trust_title=True,
        )
        if rec:
            key = rec.url or rec.title.lower()
            records_by_key[key] = rec

    # 2) Complete fallback list, especially inside <noscript>.
    for a in soup.find_all("a", href=True):
        href = str(a.get("href"))
        if not is_virtual_poster_href(href):
            continue
        url = normalize_url(href, resolved_base)
        title = clean_text(a.get_text(" ", strip=True))
        rec = make_record(
            conference=conference,
            year=year,
            title=title,
            url=url,
            source_file=source_file,
            parser="conference-virtual-list",
            trust_title=True,
        )
        if not rec:
            continue
        key = rec.url or rec.title.lower()
        old = records_by_key.get(key)
        if old is None:
            records_by_key[key] = rec
        elif not old.authors and rec.authors:
            records_by_key[key] = rec

    return list(records_by_key.values())


# ---------------------------------------------------------------------------
# CVF OpenAccess: CVPR / ICCV
# ---------------------------------------------------------------------------


def parse_cvf(
    soup: BeautifulSoup,
    conference: str,
    year: int | str,
    *,
    base_url: str = "",
    source_file: str = "",
) -> list[PaperRecord]:
    records: list[PaperRecord] = []
    if not soup.select("dt.ptitle") and "openaccess.thecvf.com" not in str(soup)[:20000].lower():
        return []

    resolved_base = base_url or "https://openaccess.thecvf.com/"
    for dt in soup.select("dt.ptitle"):
        a = dt.find("a", href=True)
        title = clean_text(a.get_text(" ", strip=True) if a else dt.get_text(" ", strip=True))
        url = normalize_url(str(a.get("href")), resolved_base) if a else ""
        authors = ""
        abstract = ""
        pdf_url = ""

        # CVF structure is often dt title, dd authors, dd links/abstract.
        sib = dt.find_next_sibling()
        steps = 0
        while isinstance(sib, Tag) and steps < 5:
            txt = clean_text(sib.get_text(" ", strip=True))
            cls = " ".join(str(c).lower() for c in sib.get("class", []))
            if sib.name == "dd" and not authors and "author" in cls or (sib.name == "dd" and not authors and "pdf" not in txt.lower()):
                authors = txt
            if "abstract" in cls or txt.lower().startswith("abstract"):
                abstract = clean_text(re.sub(r"^abstract\s*[:\-]?\s*", "", txt, flags=re.I))
            found_pdf = find_pdf_url(sib, resolved_base)
            if found_pdf:
                pdf_url = found_pdf
            sib = sib.find_next_sibling()
            steps += 1

        parent = dt.parent if isinstance(dt.parent, Tag) else dt
        pdf_url = pdf_url or find_pdf_url(parent, resolved_base)
        rec = make_record(
            conference=conference,
            year=year,
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
            pdf_url=pdf_url,
            source_file=source_file,
            parser="cvf",
        )
        if rec:
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# PMLR: ICML/MLSys proceedings pages when saved from proceedings.mlr.press
# ---------------------------------------------------------------------------


def parse_pmlr(
    soup: BeautifulSoup,
    conference: str,
    year: int | str,
    *,
    base_url: str = "",
    source_file: str = "",
) -> list[PaperRecord]:
    records: list[PaperRecord] = []
    if not soup.select("div.paper, div.paper-card, li.paper") and "proceedings.mlr.press" not in str(soup)[:20000].lower():
        return []

    for node in soup.select("div.paper, div.paper-card, li.paper"):
        title = text_of_first(node, ["p.title", ".title", "h2", "h3", "a"])
        authors = text_of_first(node, ["p.details", ".details", "p.authors", ".authors"])
        abstract = nearby_abstract(node)
        url = href_of_first(node, ["p.links a[href]", "a[href]"], base_url)
        pdf_url = find_pdf_url(node, base_url)
        rec = make_record(
            conference=conference,
            year=year,
            title=title,
            abstract=abstract,
            authors=authors,
            url=url,
            pdf_url=pdf_url,
            source_file=source_file,
            parser="pmlr",
        )
        if rec:
            records.append(rec)

    # PMLR older pages can use <p class="title"> without a wrapping card.
    if not records:
        for title_node in soup.select("p.title"):
            container = title_node.parent if isinstance(title_node.parent, Tag) else title_node
            rec = make_record(
                conference=conference,
                year=year,
                title=title_node.get_text(" ", strip=True),
                authors=text_of_first(container, ["p.details", ".details", "p.authors", ".authors"]),
                abstract=nearby_abstract(container),
                url=href_of_first(container, ["a[href]"], base_url),
                pdf_url=find_pdf_url(container, base_url),
                source_file=source_file,
                parser="pmlr-title",
            )
            if rec:
                records.append(rec)
    return records


# ---------------------------------------------------------------------------
# ACL / EMNLP official accepted pages
# ---------------------------------------------------------------------------


def parse_acl_style(
    soup: BeautifulSoup,
    conference: str,
    year: int | str,
    *,
    base_url: str = "",
    source_file: str = "",
) -> list[PaperRecord]:
    """Parse ACL/EMNLP accepted-paper pages.

    Current ACL 2024 sample uses <li><strong>Title</strong> Authors</li>.
    Current EMNLP 2024 sample uses <p><strong>Title</strong> Authors</p>.
    """
    records: list[PaperRecord] = []
    main = soup.select_one("article") or soup.select_one("main") or soup.body or soup

    # Strong-title structure: robust for ACL 2024 and EMNLP 2024.
    for strong in main.find_all("strong"):
        parent = strong.parent if isinstance(strong.parent, Tag) else None
        if parent is None or parent.name not in {"li", "p", "div"}:
            continue
        title = clean_text(strong.get_text(" ", strip=True))
        if is_bad_title(title) or len(title.split()) < 3:
            continue

        full_text = clean_text(parent.get_text(" ", strip=True))
        authors = full_text
        if authors.startswith(title):
            authors = clean_text(authors[len(title):])
        authors = authors.strip(" :;-–—")

        # Avoid obvious page chrome; most author strings contain comma-separated names,
        # but keep single-author cases if the remaining text still looks like a name.
        if not authors or len(authors) < 4:
            continue

        a = parent.find("a", href=True)
        url = normalize_url(str(a.get("href")), base_url) if a else ""
        rec = make_record(
            conference=conference,
            year=year,
            title=title,
            authors=authors,
            url=url,
            pdf_url=find_pdf_url(parent, base_url),
            source_file=source_file,
            parser="acl-strong",
            trust_title=True,
        )
        if rec:
            records.append(rec)

    if records:
        return records

    # Card/list fallback for other ACL Anthology-style pages.
    card_selectors = [
        "p.d-sm-flex",
        ".acl-paper",
        ".paper",
        "li.paper",
        "div.paper-card",
        "div.card",
    ]
    for selector in card_selectors:
        for node in main.select(selector):
            title = text_of_first(node, ["span.d-block", "p.title", ".title", "a"])
            if not title:
                title = clean_text(node.get_text(" ", strip=True).split("\n")[0])
            a = node.find("a", href=True)
            url = normalize_url(str(a.get("href")), base_url) if a else ""
            authors = text_of_first(node, ["p.authors", ".authors", ".text-muted"])
            rec = make_record(
                conference=conference,
                year=year,
                title=title,
                authors=authors,
                url=url,
                pdf_url=find_pdf_url(node, base_url),
                source_file=source_file,
                parser="acl-card",
            )
            if rec:
                records.append(rec)
        if records:
            return records

    # Last ACL fallback: line pairs title/authors inside article/main.
    lines = [clean_text(x) for x in main.get_text("\n", strip=True).split("\n")]
    lines = [x for x in lines if x and not is_bad_title(x)]
    skip = {"accepted main conference papers", "main conference"}
    i = 0
    while i < len(lines) - 1:
        title = lines[i]
        authors = lines[i + 1]
        if title.lower() in skip:
            i += 1
            continue
        if is_probable_title(title) and looks_like_authors(authors):
            rec = make_record(
                conference=conference,
                year=year,
                title=title,
                authors=authors,
                source_file=source_file,
                parser="acl-lines",
            )
            if rec:
                records.append(rec)
                i += 2
                continue
        i += 1
    return records


def looks_like_authors(text: str) -> bool:
    text = clean_text(text)
    if not text or len(text) > 400:
        return False
    if any(x in text.lower() for x in ["abstract", "poster session", "select year", "conference"]):
        return False
    return "," in text or " and " in text.lower() or (2 <= len(text.split()) <= 30 and not text.endswith("."))


# ---------------------------------------------------------------------------
# OpenReview
# ---------------------------------------------------------------------------


def parse_openreview(
    soup: BeautifulSoup,
    conference: str,
    year: int | str,
    *,
    base_url: str = "",
    source_file: str = "",
) -> list[PaperRecord]:
    records: list[PaperRecord] = []

    # JSON data in __NEXT_DATA__ or similar scripts.
    for script in soup.find_all("script"):
        text = script.string or script.get_text(" ", strip=True)
        if not text or ("title" not in text.lower() and "abstract" not in text.lower()):
            continue
        objects: list[Any] = []
        if script.get("id") == "__NEXT_DATA__" or text.strip().startswith("{"):
            try:
                objects.append(json.loads(text))
            except Exception:
                pass
        for obj in objects:
            for note in walk_openreview_notes(obj):
                rec = note_to_record(
                    note,
                    conference=conference,
                    year=year,
                    base_url=base_url,
                    source_file=source_file,
                )
                if rec:
                    records.append(rec)

    page_marker = soup.get_text(" ", strip=True).lower() + " " + " ".join(
        " ".join(str(c) for c in tag.get("class", [])) for tag in soup.find_all(True)
    ).lower()
    if "openreview" in page_marker or "note_content" in page_marker or "maincard" in page_marker:
        for node in soup.select(".note, .forum-note, .note-content, .maincard"):
            title = text_of_first(node, [".note_content_title", ".maincardBodyTitle", ".title", "h4", "h3", "a"])
            abstract = text_of_first(node, [".note_content_abstract", ".abstract"])
            authors = text_of_first(node, [".note_content_authors", ".authors", ".maincardBodyAuthors"])
            a = node.find("a", href=True)
            url = normalize_url(str(a.get("href")), base_url) if a else ""
            rec = make_record(
                conference=conference,
                year=year,
                title=title,
                abstract=abstract,
                authors=authors,
                url=url,
                source_file=source_file,
                parser="openreview-visible",
            )
            if rec:
                records.append(rec)
    return records


def walk_openreview_notes(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        content = obj.get("content")
        if isinstance(content, dict):
            maybe_title = extract_openreview_value(content.get("title"))
            if maybe_title:
                yield obj
        for value in obj.values():
            yield from walk_openreview_notes(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_openreview_notes(item)


def extract_openreview_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "value" in value:
            return value.get("value")
        if "values" in value:
            return value.get("values")
    return value


def note_to_record(
    note: dict[str, Any],
    *,
    conference: str,
    year: int | str,
    base_url: str,
    source_file: str,
) -> PaperRecord | None:
    content = note.get("content", {}) if isinstance(note, dict) else {}
    title = extract_openreview_value(content.get("title"))
    abstract = extract_openreview_value(content.get("abstract"))
    authors = extract_openreview_value(content.get("authors")) or extract_openreview_value(content.get("authorids"))
    forum = note.get("forum") or note.get("id") or ""
    url = normalize_url(f"/forum?id={forum}", "https://openreview.net") if forum else base_url
    return make_record(
        conference=conference,
        year=year,
        title=clean_text(title),
        abstract=clean_text(abstract),
        authors=authors if isinstance(authors, list) else clean_text(authors),
        url=url,
        source_file=source_file,
        parser="openreview-json",
    )


# ---------------------------------------------------------------------------
# Generic fallback
# ---------------------------------------------------------------------------


def parse_generic(
    soup: BeautifulSoup,
    conference: str,
    year: int | str,
    *,
    base_url: str = "",
    source_file: str = "",
) -> list[PaperRecord]:
    records: list[PaperRecord] = []

    for selector in TITLE_SELECTORS:
        for node in soup.select(selector):
            a = node.find("a", href=True) if isinstance(node, Tag) else None
            title = clean_text(a.get_text(" ", strip=True) if a else node.get_text(" ", strip=True))
            url = normalize_url(str(a.get("href")), base_url) if a else ""
            container = node.parent if isinstance(node.parent, Tag) else node
            authors = text_of_first(container, [".authors", "p.authors", ".details", "p.details"])
            abstract = nearby_abstract(container)
            rec = make_record(
                conference=conference,
                year=year,
                title=title,
                abstract=abstract,
                authors=authors,
                url=url,
                pdf_url=find_pdf_url(container, base_url),
                source_file=source_file,
                parser="generic-selector",
            )
            if rec:
                records.append(rec)

    for a in soup.find_all("a", href=True):
        title = clean_text(a.get_text(" ", strip=True))
        if not is_probable_title(title) or is_bad_title(title):
            continue
        container = a.parent if isinstance(a.parent, Tag) else a
        text_lines = [clean_text(x) for x in container.get_text("\n", strip=True).split("\n")]
        text_lines = [x for x in text_lines if x and x != title]
        authors = text_lines[0] if text_lines else ""
        rec = make_record(
            conference=conference,
            year=year,
            title=title,
            authors=authors,
            url=normalize_url(str(a.get("href")), base_url),
            pdf_url=find_pdf_url(container, base_url),
            source_file=source_file,
            parser="generic-anchor",
        )
        if rec:
            records.append(rec)

    if len(records) < 3:
        main = soup.select_one("main") or soup.select_one("article") or soup.body or soup
        lines = [clean_text(x) for x in main.get_text("\n", strip=True).split("\n")]
        lines = [x for x in lines if x and not is_bad_title(x)]
        for i, line in enumerate(lines[:-1]):
            if not is_probable_title(line):
                continue
            next_line = lines[i + 1]
            if not looks_like_authors(next_line):
                continue
            rec = make_record(
                conference=conference,
                year=year,
                title=line,
                authors=next_line,
                source_file=source_file,
                parser="generic-lines",
            )
            if rec:
                records.append(rec)
    return records


def parse_file(
    path: str | Path,
    conference: str,
    year: int | str,
    *,
    base_url: str = "",
    parser_hint: str = "auto",
) -> list[PaperRecord]:
    from .utils import read_html

    html = read_html(path)
    return parse_html(
        html,
        conference,
        year,
        base_url=base_url,
        source_file=str(path),
        parser_hint=parser_hint,
    )
