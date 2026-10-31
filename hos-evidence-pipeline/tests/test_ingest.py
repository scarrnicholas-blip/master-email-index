"""Stage 1 tests on invented decisions: archive, HTML parsing, paragraph references."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from hos_pipeline.config import Settings
from hos_pipeline.ingest.archive import html_path, load_html, read_manifest
from hos_pipeline.ingest.paragraphs import convert_file, number_to_pos, pos_label, rewrite_citations
from hos_pipeline.ingest.parse import describe_html, find_date, parse_decision, parse_text_decision, section_kind
from hos_pipeline.ingest.run import ingest_archive
from hos_pipeline.io_utils import read_jsonl, write_jsonl
from hos_pipeline.models import Decision

FIXTURES = Path(__file__).parent / "fixtures" / "decisions"
V1 = ("v1_literal_entry_content.html", "202312345")
V2 = ("v2_ol_start_nested.html", "202298765")
V3 = ("v3_ol_continuing.html", "202154321")
V4 = ("v4_older_findings.html", "201901234")


def parse_fixture(name: str, case_id: str) -> Decision:
    return parse_decision((FIXTURES / name).read_text(encoding="utf-8"), case_id)


def numbered(d: Decision) -> list[tuple[int | None, str | None]]:
    return [(p.number, p.sub) for p in d.paragraphs if p.number is not None]


def by_number(d: Decision, number: int, sub: str | None = None):
    return next(p for p in d.paragraphs if p.number == number and p.sub == sub)


# --------------------------------------------------------------------------
# Archive
# --------------------------------------------------------------------------


def _write_page(archive: Path, case_id: str, html: str) -> str:
    path = html_path(archive, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = html.encode("utf-8")
    path.write_bytes(gzip.compress(data))
    return hashlib.sha256(data).hexdigest()


def _write_manifest(archive: Path, rows: list[tuple[str, str, str]]) -> None:
    lines = ["case_id\turl\thttp_status\tbytes\tsha256\tfetched_utc"]
    for i, (case_id, url, sha) in enumerate(rows):
        lines.append(f"{case_id}\t{url}\t200\t100\t{sha}\t2026-10-0{i + 1}T10:00:00Z")
    (archive / "manifest.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_manifest_last_row_wins_and_sha_check(tmp_path: Path) -> None:
    good = _write_page(tmp_path, "202312345", "<p>first page</p>")
    _write_page(tmp_path, "202298765", "<p>second page</p>")
    _write_page(tmp_path, "202000001", "<p>on disk only</p>")
    _write_manifest(tmp_path, [
        ("202312345", "https://example.invalid/old-202312345/", "0" * 64),
        ("202298765", "https://example.invalid/b-202298765/", "f" * 64),
        ("202312345", "https://example.invalid/new-202312345/", good),
    ])

    entries = read_manifest(tmp_path)
    assert list(entries) == ["202000001", "202298765", "202312345"]
    e = entries["202312345"]
    assert e.url == "https://example.invalid/new-202312345/"
    assert e.fetched_utc == "2026-10-03T10:00:00Z"
    assert e.path == tmp_path / "html" / "2023" / "202312345.html.gz"
    assert load_html(e) == ("<p>first page</p>", True)
    assert load_html(entries["202298765"])[1] is False
    assert load_html(entries["202298765"], verify_sha=False)[1] is None
    orphan = entries["202000001"]
    assert orphan.url is None and orphan.sha256 is None
    assert load_html(orphan) == ("<p>on disk only</p>", None)


def test_manifest_missing_globs_html_and_bad_utf8(tmp_path: Path) -> None:
    path = html_path(tmp_path, "202400002")
    path.parent.mkdir(parents=True)
    path.write_bytes(gzip.compress(b"<p>caf\xe9</p>"))
    entries = read_manifest(tmp_path)
    assert list(entries) == ["202400002"]
    html, ok = load_html(entries["202400002"])
    assert html == "<p>caf�</p>" and ok is None


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_v1_literal_numbers_inside_entry_content() -> None:
    d = parse_fixture(*V1)
    assert d.landlord == "Exampleshire Borough Council"
    assert d.title == "Exampleshire Borough Council (202312345)"
    assert d.decision_date == date(2024, 6, 17) and d.year == 2024
    assert d.url.endswith("/exampleshire-borough-council-202312345/")
    assert d.parse_warnings == []
    assert [n for n, s in numbered(d) if s is None] == list(range(1, 14))
    assert by_number(d, 12, "a").text.startswith("£500 for the distress")
    assert by_number(d, 12, "b").text == "£150 for its complaint handling."
    assert by_number(d, 13, "a").text.startswith("A qualified surveyor")  # split at <br>
    assert by_number(d, 2).text.startswith("We have also looked")  # "2.&nbsp;" prefix stripped
    # Noise is gone; boilerplate is unnumbered under its heading.
    text = d.text()
    for noise in ("cookies", "Related decisions", "Contact us", "Published 3 July", "Posted in", "builtOn"):
        assert noise not in text
    approach = d.section("Our approach")
    assert len(approach) == 2 and all(p.number is None for p in approach)
    assert by_number(d, 10).section == "Determination"
    assert by_number(d, 7).section == "The landlord’s handling of the reports of damp and mould"
    quote = d.paragraphs[by_number(d, 7).pos]  # the blockquote after paragraph 7
    assert quote.number is None and quote.text.startswith("“Our clothes")


def test_v2_ol_start_split_lists_with_nested_letters() -> None:
    d = parse_fixture(*V2)
    assert d.landlord == "Northfield Housing Association"
    assert d.decision_date == date(2023, 2, 2)
    assert d.parse_warnings == []
    heads = [n for n, s in numbered(d) if s is None]
    assert sorted(set(heads)) == list(range(1, 11))
    # Paragraph 4 spans two <p> inside one <li>.
    assert [p.text[:3] for p in d.paragraphs if p.number == 4] == ["In ", "She"]
    assert by_number(d, 9, "a").text == "pay the resident £400 compensation;"
    assert by_number(d, 9, "b").section == "Orders"
    assert [p.number for p in d.section("Determination")] == [7, 8]


def test_v3_ol_without_start_continues_across_headings() -> None:
    d = parse_fixture(*V3)
    assert d.landlord == "Ashdown Vale Housing Trust"
    assert d.decision_date == date(2022, 9, 30)  # dd/mm/yyyy
    assert d.parse_warnings == []
    assert sorted({n for n, _ in numbered(d)}) == list(range(1, 11))
    assert [p.number for p in d.section("Determination")] == [8, 9]
    # Bullets inside paragraph 4 keep its number; the plain short line before an
    # <ol> became a heading.
    assert [p.number for p in d.paragraphs if p.text.startswith("it would")] == [4, 4]
    assert by_number(d, 5).section == "The resident’s window repairs"
    assert by_number(d, 1).section == "The complaint"  # <p><strong> heading


def test_v4_older_layout() -> None:
    d = parse_fixture(*V4)
    assert d.landlord == "Westmere District Council"
    assert d.decision_date == date(2020, 3, 5) and d.year == 2020
    assert d.parse_warnings == []
    assert by_number(d, 1).section == "Complaint definition"
    assert section_kind("Complaint definition") == "complaint"
    assert section_kind(by_number(d, 5).section) == "determination"  # "Findings"
    assert [s for n, s in numbered(d) if n == 9] == [None, "a", "b"]  # "a)" style


def test_plain_text_decision() -> None:
    d = parse_text_decision((FIXTURES / "exampleshire_202420001.txt").read_text(encoding="utf-8"))
    assert d.case_id == "202420001"
    assert d.landlord == "Exampleshire Borough Council"
    assert d.url == "https://www.housing-ombudsman.org.uk/decisions/exampleshire-borough-council-202420001/"
    assert d.decision_date == date(2025, 3, 4)
    assert d.parse_warnings == []
    assert d.paragraphs[0].text == "REPORT"  # the repeated title line is dropped
    assert sorted({n for n, _ in numbered(d)}) == list(range(1, 13))
    assert by_number(d, 11, "b").text == "£200 for its complaint handling."
    assert by_number(d, 6).section == "The landlord's handling of the resident's reports of damp and mould"
    assert [p.number for p in d.section("Determination")] == [9, 10]


def test_numbering_warnings() -> None:
    html = ("<title>Gapford Council (202400001)</title><div class='entry-content'><p>17 May 2024</p>"
            "<p>1. a</p><p>2. b</p><p>4. c</p><p>4. d</p><p>3. e</p><p>5. f</p></div>")
    d = parse_decision(html, "202400001")
    assert d.parse_warnings == [
        "numbering gap 2 -> 4",
        "duplicate paragraph number 4",
        "paragraph 3 out of order after 4",
        "no Determination section found",
    ]


def test_missing_landlord_numbers_and_date() -> None:
    d = parse_decision("<html><body><p>Page not found</p></body></html>", "202400001")
    assert d.landlord == "" and d.year == 2024 and d.decision_date is None
    for w in ("landlord not found", "no numbered paragraphs", "no Determination section found",
              "decision date not found; year from case id"):
        assert w in d.parse_warnings


def test_landlord_from_url_and_bare_markers() -> None:
    html = ("<div class='entry-content'><h2>Determination</h2><p>1.</p><p>The first paragraph.</p>"
            "<p>2. Line one<br>soft continuation</p><table><tr><td>3.</td><td>In a table cell.</td></tr></table>"
            "<p>(a)</p><p>sub-item text</p></div>")
    url = "https://www.housing-ombudsman.org.uk/decisions/borough-of-somewhere-upon-sea-202400001/"
    d = parse_decision(html, "202400001", url)
    assert d.landlord == "Borough of Somewhere upon Sea"
    assert "landlord taken from URL slug" in d.parse_warnings
    assert [(p.number, p.sub, p.text) for p in d.paragraphs] == [
        (1, None, "The first paragraph."),
        (2, None, "Line one"),
        (2, None, "soft continuation"),
        (3, None, "In a table cell."),
        (3, "a", "sub-item text"),
    ]


def test_li_value_literal_override_and_nested_list_first() -> None:
    html = """<h1>Listshire Homes Limited (202400001)</h1><article><p>1 May 2024</p>
      <h2>Background</h2><ol><li>one</li><li value="2">two</li><li>three</li></ol>
      <h2>Determination</h2><ol><li>4. four (literal)</li><li>five</li>
      <li><ol type="a"><li>(c) lettered c</li><li>next item</li></ol></li></ol></article>"""
    d = parse_decision(html, "202400001")
    assert d.parse_warnings == []
    assert [(p.number, p.sub) for p in d.paragraphs if p.number] == [
        (1, None), (2, None), (3, None), (4, None), (5, None), (6, "c"), (6, "d")
    ]


def test_find_date_formats() -> None:
    assert find_date("Decision of 17 June 2025") == date(2025, 6, 17)
    assert find_date("June 3rd, 2021") == date(2021, 6, 3)
    assert find_date("on 1/2/2020 and 17 June 2025") == date(2020, 2, 1)
    assert find_date("31 February 2020") is None


def test_describe_html_reports_structure() -> None:
    out = describe_html((FIXTURES / V1[0]).read_text(encoding="utf-8"))
    assert "selector '.entry-content'" in out
    assert "literal 'N.' prefixes" in out
    assert "landlord='Exampleshire Borough Council'" in out
    assert "warnings:    none" in out
    out2 = describe_html((FIXTURES / V2[0]).read_text(encoding="utf-8"))
    assert "<ol>/<li> counters" in out2 and "start=['1', '3', '5', '7', '9']" in out2


# --------------------------------------------------------------------------
# Paragraph references
# --------------------------------------------------------------------------


def test_pos_label_and_number_to_pos() -> None:
    d = parse_fixture(*V1)
    p12a = by_number(d, 12, "a").pos
    assert pos_label(d, by_number(d, 10).pos) == "paragraph 10"
    assert pos_label(d, p12a) == "paragraph 12(a)"
    assert pos_label(d, d.section("Our approach")[0].pos) == "unnumbered text under 'Our approach'"
    assert pos_label(d, by_number(d, 7).pos + 1) == "unnumbered text after paragraph 7"
    with pytest.raises(KeyError):
        pos_label(d, 999)
    assert number_to_pos(d, 12) == by_number(d, 12).pos
    assert number_to_pos(d, 12, "a") == p12a
    assert number_to_pos(d, 12, "(A)") == p12a
    assert number_to_pos(d, 99) is None
    assert number_to_pos(d, 12, "z") is None


def test_rewrite_citations_forms() -> None:
    d = parse_fixture(*V1)
    p = {n: by_number(d, n).pos for n in (8, 9, 10)}
    pa = by_number(d, 12, "a").pos
    text = f"Late [p{p[10]}]; see [p{p[8]}, p{p[9]}] and [p{p[8]}-p{p[10]}], [p{pa}], [p9999]."
    assert rewrite_citations(text, d) == (
        "Late [para 10]; see [para 8, 9] and [para 8-10], [para 12(a)], [p9999]."
    )
    assert rewrite_citations(f"[p{p[10]}]", d, template="({long})") == "(paragraph 10)"
    assert rewrite_citations(f"[p{p[10]}]", d, template="{pos}") == f"p{p[10]}"


def test_convert_file_csv_and_jsonl(tmp_path: Path) -> None:
    v1, v2 = parse_fixture(*V1), parse_fixture(*V2)
    decisions = tmp_path / "decisions.jsonl"
    write_jsonl(decisions, [v2, v1])
    p10, p9a = by_number(v1, 10).pos, by_number(v2, 9, "a").pos

    src = tmp_path / "claims.csv"
    with src.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "claim", "note"])
        w.writerow([v1.case_id, f"Finding at [p{p10}]", "x"])
        w.writerow([v2.case_id, f"Order [p{p9a}]", "y"])
        w.writerow(["209999999", "[p1]", "unknown case"])
    out = tmp_path / "claims_official.csv"
    assert convert_file(src, decisions, out, columns=["claim"]) == 3
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == ["case_id", "claim", "claim_official", "note"]
    assert rows[0]["claim"] == f"Finding at [p{p10}]"
    assert rows[0]["claim_official"] == "Finding at [para 10]"
    assert rows[1]["claim_official"] == "Order [para 9(a)]"
    assert rows[2]["claim_official"] == "[p1]"

    src_j = tmp_path / "coded.jsonl"
    write_jsonl(src_j, [{"case_id": v2.case_id, "quotes": [f"[p{p9a}]"], "rationale": f"see [p{p9a}]"}])
    out_j = tmp_path / "coded_official.jsonl"
    assert convert_file(src_j, decisions, out_j, columns=["quotes", "rationale"]) == 1
    rec = next(read_jsonl(out_j))
    assert rec["quotes_official"] == ["[para 9(a)]"] and rec["rationale_official"] == "see [para 9(a)]"

    with pytest.raises(ValueError):
        convert_file(src, decisions, out, columns=["missing_column"])


# --------------------------------------------------------------------------
# Stage runner
# --------------------------------------------------------------------------


def test_ingest_archive_resumable_with_failures(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    rows = []
    for name, case_id in (V1, V2, V3):
        sha = _write_page(archive, case_id, (FIXTURES / name).read_text(encoding="utf-8"))
        rows.append((case_id, f"https://example.invalid/decisions/x-{case_id}/", sha))
    rows[1] = (rows[1][0], rows[1][1], "0" * 64)  # checksum mismatch, still ingested
    broken = html_path(archive, "202400009")
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"\x1f\x8b not really gzip")
    rows.append(("202400009", "https://example.invalid/decisions/broken-202400009/", "1" * 64))
    _write_manifest(archive, rows)

    settings = Settings()
    settings.paths.archive_dir = archive
    settings.paths.work_dir = tmp_path / "work"

    out = ingest_archive(settings, limit=2)
    assert [d["case_id"] for d in read_jsonl(out)] == ["202154321", "202298765"]
    out = ingest_archive(settings)
    decisions = {d.case_id: d for d in read_jsonl(out, Decision)}
    assert sorted(decisions) == ["202154321", "202298765", "202312345"]  # no duplicates
    assert decisions["202312345"].sha256_ok is True
    assert decisions["202298765"].sha256_ok is False
    assert "sha256 does not match manifest" in decisions["202298765"].parse_warnings
    assert decisions["202312345"].source_path.endswith("202312345.html.gz")
    assert decisions["202312345"].url == "https://example.invalid/decisions/x-202312345/"

    with settings.work("ingest_report.csv").open() as fh:
        report = {r["case_id"]: r for r in csv.DictReader(fh)}
    assert report["202312345"]["n_numbered"] == "13"
    assert report["202312345"]["has_determination"] == "True"
    assert report["202400009"]["warnings"].startswith("error:")

    # A retry of the failed page replaces its report row rather than adding one.
    ingest_archive(settings)
    with settings.work("ingest_report.csv").open() as fh:
        assert [r["case_id"] for r in csv.DictReader(fh)].count("202400009") == 1
    assert json.loads(out.read_text().splitlines()[0])["case_id"] == "202154321"

    # An interrupted run can leave a half-written last line; the next run drops and redoes it.
    lines = out.read_text().splitlines(keepends=True)
    out.write_text("".join(lines[:-1]) + lines[-1][:50])
    ingest_archive(settings)
    assert sorted(d["case_id"] for d in read_jsonl(out)) == ["202154321", "202298765", "202312345"]
