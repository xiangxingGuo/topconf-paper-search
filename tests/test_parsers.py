from pathlib import Path

from paper_search.parsers import parse_html


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_parse_cvf():
    html = (EXAMPLES / "sample_cvf.html").read_text(encoding="utf-8")
    records = parse_html(html, "CVPR", 2026, base_url="https://openaccess.thecvf.com/", parser_hint="cvf")
    assert len(records) == 2
    assert records[0].conference == "CVPR"
    assert "Vision Transformer" in records[0].title
    assert records[0].url.startswith("https://openaccess.thecvf.com/")


def test_parse_pmlr():
    html = (EXAMPLES / "sample_pmlr.html").read_text(encoding="utf-8")
    records = parse_html(html, "ICML", 2026, base_url="https://proceedings.mlr.press/v999/", parser_hint="pmlr")
    assert len(records) == 2
    assert records[0].abstract.startswith("We study calibration")
    assert records[0].pdf_url.endswith("lovelace26a.pdf")


def test_parse_acl():
    html = (EXAMPLES / "sample_acl.html").read_text(encoding="utf-8")
    records = parse_html(html, "ACL", 2026, base_url="https://aclanthology.org/", parser_hint="acl")
    assert len(records) == 2
    assert "Knowledge Editing" in records[0].title


def test_parse_openreview():
    html = (EXAMPLES / "sample_openreview.html").read_text(encoding="utf-8")
    records = parse_html(html, "ICLR", 2026, parser_hint="openreview")
    assert len(records) == 2
    assert records[0].url.startswith("https://openreview.net/forum?id=")
    assert "Reward Models" in records[0].title


def test_parse_iclr_virtual():
    html = (EXAMPLES / "sample_iclr_virtual.html").read_text(encoding="utf-8")
    records = parse_html(html, "ICLR", 2025, parser_hint="iclr_virtual")
    assert len(records) == 3
    assert records[0].url == "https://iclr.cc/virtual/2025/poster/31505"
    assert "SakethaNath Jagarlapudi" in records[0].authors
    assert any(r.title == "Differential Transformer" for r in records)
    assert any(r.title == "Hyper-Connections" for r in records)
