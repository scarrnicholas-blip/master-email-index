"""Parse a published Housing Ombudsman decision page into a ``Decision``.

Nobody building this has seen the archive's HTML, so the parser follows the
published text layout and accepts several markups for paragraph numbers:

* literal numbers on ``<p>`` paragraphs ("12. The landlord ...");
* ``<ol>``/``<li>`` counters, honouring ``start`` and ``value`` and carrying the
  count across lists that headings split apart;
* sub-items as nested ``<ol>`` or literal "(a)" lines, attached to the parent number.

Literal number prefixes ("12.", "(a)") are stripped from ``Paragraph.text``
because the number is kept in ``Paragraph.number`` / ``Paragraph.sub``.
``describe_html`` reports what the parser saw, for tuning against real pages.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

from ..models import Decision, Paragraph

_BR = ""  # stands in for <br> so line breaks survive whitespace clean-up
_ZERO_WIDTH = re.compile("[­​-‍⁠﻿]")
_WS = re.compile(r"\s+")

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_LIST_TAGS = {"ol", "ul", "menu"}
_BLOCK_TAGS = _HEADING_TAGS | _LIST_TAGS | {
    "p", "div", "section", "article", "main", "aside", "nav", "header", "footer", "blockquote",
    "li", "dl", "dt", "dd", "pre", "address", "figure", "figcaption", "table", "caption",
    "thead", "tbody", "tfoot", "tr", "td", "th", "form", "fieldset", "details", "summary", "center",
}
_LEAF_TAGS = _HEADING_TAGS | {"p", "li", "td", "th", "dt", "dd", "pre"}
_CODE_TAGS = ["script", "style", "noscript", "template", "svg", "iframe", "object", "embed",
              "canvas", "video", "audio", "button", "input", "select", "textarea"]

_NOISE_TAGS = {"nav", "aside", "header", "footer", "form", "dialog"}
_NOISE_ROLES = {"navigation", "banner", "contentinfo", "search", "complementary", "dialog", "alertdialog"}
_NOISE_TOKENS = {
    "cookie", "cookies", "consent", "gdpr", "banner", "breadcrumb", "breadcrumbs", "share", "sharing",
    "sharedaddy", "social", "skip", "menu", "navbar", "navigation", "nav", "sidebar", "newsletter",
    "search", "modal", "popup", "related", "feedback", "masthead", "colophon", "header", "footer",
    "widget", "pagination", "postnav",
}
_CONTAINER_SELECTORS = (
    ".entry-content", "[itemprop='articleBody']", ".post-content", ".article-content", ".page-content",
    ".single-content", ".decision-content", "article", "main", "[role='main']", "#content", ".content",
    "#main", ".content-area",
)

TITLE_RE = re.compile(r"^\s*(?P<landlord>.+?)\s*\(\s*(?P<ids>\d{6,}(?:\s*(?:,|&|and|/|\+)\s*\d{6,})*)\s*\)")
_LITERAL_NUM = re.compile(r"^(\d{1,3})\.(?!\d)\s*")
_SUB_ITEM = re.compile(r"^(?:\((?P<a>[a-z]{1,2}|[ivx]{1,5}|\d{1,2})\)|(?P<b>[a-z])\))\s*(?=\S)", re.I)
_BARE_MARKER = re.compile(r"^(?:\d{1,3}\.|\((?:[a-z]{1,2}|[ivx]{1,5}|\d{1,2})\)|[a-z]\))$", re.I)
_PARA_BREAK = re.compile(f"{_BR}\\s*{_BR}")
_COMPLAINT_LINE = re.compile(r"^complaint\s+(?:no\.?|number|ref(?:erence)?)?\s*:?\s*\d{6,}$", re.I)

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9,
    "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE_RES = (
    re.compile(rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<m>{_MONTH})\.?,?\s+(?P<y>\d{{4}})\b", re.I),
    re.compile(rf"\b(?P<m>{_MONTH})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y>\d{{4}})\b", re.I),
    re.compile(r"\b(?P<d>\d{1,2})[/.\-](?P<m>\d{1,2})[/.\-](?P<y>\d{4})\b"),
)

_SECTION_KINDS = (
    ("complaint", re.compile(
        r"^(?:the |what the |summary of (?:the )?|the resident's )?complaints?"
        r"(?: definition| (?:is|are|was|were) about| being (?:considered|investigated))?$")),
    ("determination", re.compile(r"^(?:the )?(?:ombudsman's )?(?:determination|decision|findings?)\b")),
    ("background", re.compile(r"^(?:background|summary of events|chronology|events|(?:the |factual )?facts)\b")),
    ("assessment", re.compile(r"^(?:assessment|scope of (?:the )?investigation|reasons|analysis|investigation)\b")),
    ("jurisdiction", re.compile(r"\bjurisdiction\b")),
    ("orders", re.compile(r"^(?:orders?|recommendations?)\b")),
    ("approach", re.compile(r"^our approach\b")),
)
_OTHER_HEADINGS = re.compile(
    r"^(?:complaint handling|the landlord's (?:complaint handling|handling|response|record[- ]keeping|"
    r"policies|obligations|repairs?)\b.*|(?:relevant )?policies and procedures|legal and policy framework|"
    r"conclusions?|summary|damp and mould|the resident's concerns?)$"
)


def section_kind(name: str) -> str:
    """Classify a heading: complaint, determination, background, assessment,
    jurisdiction, orders, approach or other."""
    key = _heading_key(name)
    for kind, rx in _SECTION_KINDS:
        if rx.search(key):
            return kind
    return "other"


def find_date(text: str) -> date | None:
    """The earliest full date in ``text`` ("17 June 2025", "June 17, 2025", "17/06/2025")."""
    best: tuple[int, date] | None = None
    for rx in _DATE_RES:
        for m in rx.finditer(text):
            found = _to_date(m)
            if found:
                if best is None or m.start() < best[0]:
                    best = (m.start(), found)
                break
    return best[1] if best else None


def parse_decision(html: str, case_id: str, url: str | None = None) -> Decision:
    """Parse one archived decision page. Problems go to ``parse_warnings``; it never raises on odd markup."""
    return _parse_html(html, case_id, url)[0]


def parse_text_decision(text: str, case_id: str | None = None, url: str | None = None) -> Decision:
    """Parse the plain-text copy: a "<Landlord> (<id>) - Housing Ombudsman" line,
    a "Source: <url>" line, then one paragraph or heading per line."""
    lines = [ln for ln in (_clean(raw) for raw in text.splitlines()) if ln]
    titles: list[str] = []
    body_start = 0
    for i, ln in enumerate(lines[:5]):
        if ln.lower().startswith("source:"):
            url = url or ln.split(":", 1)[1].strip() or None
            body_start = i + 1
        elif i == 0 and TITLE_RE.match(ln):
            titles.append(ln)
            body_start = 1
    case_id = case_id or _case_id_from(titles, url)
    if not case_id:
        raise ValueError("no case id: pass case_id or start the text with a '<Landlord> (<id>)' line")
    b = _Builder(case_id, literal_mode=True)
    for ln in lines[body_start:]:
        b.line(ln, b.group())
    return _finish(b, case_id, url, titles)


def describe_html(html: str) -> str:
    """A short report of how the parser sees a page, for tuning against real archive pages."""
    soup = BeautifulSoup(html or "", "lxml")
    titles = _title_candidates(soup)
    url = _canonical_url(soup)
    case_id = _case_id_from(titles, url) or "000000000"
    d, diag = _parse_html(html, case_id, url, diagnose=True)

    numbered = [p for p in d.paragraphs if p.number is not None and p.sub is None]
    numbers = sorted({p.number for p in d.paragraphs if p.number is not None})
    subs = [p for p in d.paragraphs if p.sub is not None]
    style = "literal 'N.' prefixes" if diag.literal_mode else "<ol>/<li> counters"
    if not diag.literal_paragraphs and not diag.list_items:
        style = "none detected"
    counts = " ".join(f"{k}={diag.counts.get(k, 0)}" for k in
                      ("h1", "h2", "h3", "h4", "p", "ol", "ul", "li", "table", "blockquote", "br"))
    out = [
        f"titles:      {' | '.join(repr(t) for t in titles) or '(none)'}",
        f"parsed:      landlord={d.landlord!r} case_id={case_id} date={d.decision_date} year={d.year}",
        f"url:         {d.url or '(none)'}",
        f"removed:     {len(diag.removed)} noise elements: {_tally(diag.removed)}",
        f"container:   {diag.container}",
        f"  children:  {_tally(diag.children)}",
        f"elements:    {counts} bold-only-p={diag.counts.get('bold_p', 0)}",
        f"lists:       top-level <ol> start={diag.ol_starts} type={diag.ol_types}; nested <ol> {diag.nested_ols}",
        f"numbering:   {diag.literal_paragraphs} literal 'N.' lines, {diag.list_items} top-level <ol> items"
        f" -> {style}",
        f"headings:    {' | '.join(diag.headings[:14]) or '(none)'}",
        f"header:      {' | '.join(diag.header[:8]) or '(none)'}",
        f"paragraphs:  {len(d.paragraphs)} extracted; {len(numbers)} official numbers"
        f"{f' ({numbers[0]}-{numbers[-1]})' if numbers else ''}; {len(subs)} sub-items;"
        f" {sum(p.number is None for p in d.paragraphs)} unnumbered",
    ]
    for p in numbered[:3]:
        out.append(f"  [p{p.pos}] {p.number} ({p.section}): {p.text[:90]}")
    det = [p for p in d.paragraphs if section_kind(p.section) == "determination"]
    for p in det[:2]:
        out.append(f"  determination [p{p.pos}] {p.number}: {p.text[:90]}")
    out.append(f"warnings:    {'; '.join(d.parse_warnings) or 'none'}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# HTML pipeline
# --------------------------------------------------------------------------


@dataclass
class _Diagnostics:
    container: str = ""
    removed: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    literal_mode: bool = False
    literal_paragraphs: int = 0
    list_items: int = 0
    counts: Counter = field(default_factory=Counter)
    ol_starts: list[str] = field(default_factory=list)
    ol_types: list[str] = field(default_factory=list)
    nested_ols: int = 0
    headings: list[str] = field(default_factory=list)
    header: list[str] = field(default_factory=list)


def _parse_html(html: str, case_id: str, url: str | None, diagnose: bool = False) -> tuple[Decision, _Diagnostics]:
    soup = BeautifulSoup(html or "", "lxml")
    titles = _title_candidates(soup)
    url = url or _canonical_url(soup)
    meta_date = _meta_date(soup)
    for el in soup.find_all(_CODE_TAGS):
        if not el.decomposed:
            el.decompose()
    page_text = _clean(soup.get_text(" "))
    for br in soup.find_all("br"):
        br.replace_with(_BR)

    root = soup.body or soup
    diag = _Diagnostics()
    diag.removed = _drop_noise(root)
    container, diag.container = _choose_container(root)
    diag.literal_mode, diag.literal_paragraphs, diag.list_items = _literal_mode(container)
    if diagnose:
        _count_elements(container, diag)

    b = _Builder(case_id, diag.literal_mode)
    _Walker(b).walk(container)
    diag.headings, diag.header = b.headings, b.header
    return _finish(b, case_id, url, titles, page_text, meta_date), diag


def _drop_noise(root: Tag) -> list[str]:
    """Remove navigation, headers, footers, cookie banners and the like."""
    total = len(root.get_text()) or 1
    candidates = [el for el in root.find_all(True) if _is_noise(el)]
    removed = []
    for el in candidates:
        if el.decomposed or _protected(el, total):
            continue
        removed.append(_describe_el(el))
        el.decompose()
    return removed


def _is_noise(el: Tag) -> bool:
    if el.name in _NOISE_TAGS or str(el.get("role") or "").lower() in _NOISE_ROLES:
        return True
    names = " ".join([*(el.get("class") or []), str(el.get("id") or "")]).lower()
    return bool(_NOISE_TOKENS.intersection(re.split(r"[\s_\-]+", names)))


def _protected(el: Tag, total: int) -> bool:
    """Never drop a wrapper that holds the decision itself."""
    if el.name in ("html", "body", "main", "article"):
        return True
    if el.find(["main", "article"]) is not None or el.select_one(".entry-content, [itemprop='articleBody']"):
        return True
    return len(el.get_text()) > 0.5 * total


def _choose_container(root: Tag) -> tuple[Tag, str]:
    """Try known content selectors, else the smallest element holding most of the text."""
    scores = _block_text_scores(root)
    total = scores.get(id(root), 0)
    if total == 0:
        return root, "whole page (no paragraph elements found)"
    for sel in _CONTAINER_SELECTORS:
        matches = root.select(sel)
        if not matches:
            continue
        best = max(matches, key=lambda el: scores.get(id(el), 0))
        score = scores.get(id(best), 0)
        if score >= 0.5 * total:
            return best, f"selector {sel!r} -> {_describe_el(best)}; {score:,} of {total:,} chars of block text"
    candidates = [el for el in root.find_all(True) if scores.get(id(el), 0) >= 0.85 * total]
    best = max(candidates, key=lambda el: len(list(el.parents)), default=root)
    score = scores.get(id(best), 0)
    return best, f"fallback (no selector matched) -> {_describe_el(best)}; {score:,} of {total:,} chars of block text"


def _block_text_scores(root: Tag) -> dict[int, int]:
    """Characters of leaf-block text (p, li, td, headings) under each element, keyed by id()."""
    scores: dict[int, int] = {}
    for el in _tags(root, _LEAF_TAGS):
        if _contains(el, _LEAF_TAGS):
            continue
        n = len(_clean(el.get_text().replace(_BR, " ")))
        node: Tag | None = el
        while node is not None and n:
            scores[id(node)] = scores.get(id(node), 0) + n
            if node is root:
                break
            node = node.parent
    return scores


def _literal_mode(container: Tag) -> tuple[bool, int, int]:
    """Are paragraphs numbered by literal "N." text rather than by <ol> counters?"""
    n_literal = 0
    for el in container.find_all(["p", "div", "td", "dd"]):
        if _has_block(el) or _inside(el, "li"):
            continue
        n_literal += sum(1 for ln in _split(el.get_text()) if _LITERAL_NUM.match(ln))
    n_items = sum(
        1 for li in container.find_all("li")
        if li.parent is not None and li.parent.name == "ol" and not _inside(li, "li")
    )
    return n_literal >= 3 and n_literal >= n_items, n_literal, n_items


def _count_elements(container: Tag, diag: _Diagnostics) -> None:
    for el in container.find_all(True):
        diag.counts[el.name] += 1
    diag.counts["br"] = container.get_text().count(_BR)
    diag.counts["bold_p"] = sum(1 for p in container.find_all("p") if _bold_only(p))
    diag.children = [_describe_el(el) for el in container.children if isinstance(el, Tag)]
    for ol in container.find_all("ol"):
        if not _inside(ol, "li"):
            diag.ol_starts.append(str(ol.get("start", "-")))
            diag.ol_types.append(_list_style(ol) or "-")
        else:
            diag.nested_ols += 1


# --------------------------------------------------------------------------
# Paragraph builder: numbering, sections and warnings
# --------------------------------------------------------------------------


@dataclass
class _Item:
    """An <li> being walked. Its first line registers the official number."""

    number: int | None
    sub: str | None = None
    kind: str = "plain"  # "main" numbered paragraph, "sub" lettered sub-item, "plain" bullet
    pending: bool = False
    prefix: str | None = None  # parent's sub label, for nested sub-items ("a.i")
    parent: _Item | None = None


class _Builder:
    def __init__(self, case_id: str, literal_mode: bool) -> None:
        self.case_id = case_id
        self.literal_mode = literal_mode
        self.paras: list[Paragraph] = []
        self.section = ""
        self.headings: list[str] = []
        self.header: list[str] = []  # lines before the first numbered paragraph or known heading
        self.last_main: int | None = None  # last number in document order; lists continue from it
        self.current: int | None = None  # the number "(a)" lines attach to; headings reset it
        self.numbering: list[str] = []
        self._header_open = True
        self._max = 0
        self._seen: set[int] = set()
        self._overflow = 0
        self._groups = 0
        self._last_group: int | None = None
        self._last_ref: tuple[int | None, str | None] = (None, None)
        self._maybe_heading: Paragraph | None = None
        self._marker: str | None = None  # a bare "12." or "(a)" waiting for its text

    def group(self) -> int:
        """A new id for the lines of one source element."""
        self._groups += 1
        return self._groups

    def heading(self, text: str) -> None:
        text = _clean(text).rstrip(":").strip()
        if not text or self._is_title(text):
            return
        if self._header_open:
            self.header.append(text)
            if section_kind(text) != "other" or _OTHER_HEADINGS.match(_heading_key(text)):
                self._header_open = False
        self.section = text
        self.headings.append(text)
        self.current = None
        self._last_group = None
        self._maybe_heading = None

    def line(self, text: str, group: int, item: _Item | None = None, bold: bool = False) -> None:
        text = _clean(text)
        if not text:
            return
        if item is None and _BARE_MARKER.match(text):
            self._marker = text
            return
        if item is None and self._marker:
            text, self._marker = f"{self._marker} {text}", None
        number: int | None = None
        sub: str | None = None
        if item is not None:
            number, sub, text = self._item_line(item, text)
        elif self._header_open and self._is_title(text):
            return
        elif _is_heading(text, bold):
            self.heading(text)
            return
        elif m := _LITERAL_NUM.match(text):
            number, text = int(m[1]), text[m.end():]
            self._head(number)
        elif (m := _SUB_ITEM.match(text)) and self.current is not None:
            number, sub, text = self.current, m["a"] or m["b"], text[m.end():]
        elif group == self._last_group:
            number, sub = self._last_ref
        self._emit(text, number, sub, group, plain=item is None and number is None)

    def _item_line(self, item: _Item, text: str) -> tuple[int | None, str | None, str]:
        parent = item.parent
        while parent is not None:  # a sub-list can come before its parent item's own text
            if parent.pending and parent.kind == "main":
                parent.pending = False
                self._head(parent.number)
            parent = parent.parent
        if item.pending:
            item.pending = False
            if item.kind == "main":
                if m := _LITERAL_NUM.match(text):
                    item.number, text = int(m[1]), text[m.end():]
                self._head(item.number)
            elif m := _SUB_ITEM.match(text):
                label = m["a"] or m["b"]
                item.sub = f"{item.prefix}.{label}" if item.prefix else label
                text = text[m.end():]
        return item.number, item.sub, text

    def _head(self, n: int | None) -> None:
        """Register an official paragraph number and check the sequence."""
        if n is None:
            return
        # A short plain line just before a numbered paragraph is an unmarked heading.
        if self._maybe_heading is not None and self.paras and self.paras[-1] is self._maybe_heading:
            text = self.paras.pop().text
            if self.header and self.header[-1] == text:
                self.header.pop()
            self.heading(text)
        if n in self._seen:
            self._warn(f"duplicate paragraph number {n}")
        elif n < self._max:
            self._warn(f"paragraph {n} out of order after {self._max}")
        elif n > self._max + 1:
            self._warn(f"numbering starts at {n}" if self._max == 0 else f"numbering gap {self._max} -> {n}")
        self._seen.add(n)
        self._max = max(self._max, n)
        self.last_main = self.current = n
        self._header_open = False

    def _warn(self, msg: str) -> None:
        if len(self.numbering) < 12:
            self.numbering.append(msg)
        else:
            self._overflow += 1

    def _emit(self, text: str, number: int | None, sub: str | None, group: int, plain: bool) -> None:
        first_line = group != self._last_group
        self._last_group, self._last_ref = group, (number, sub)
        if not text:
            return
        para = Paragraph(pos=len(self.paras) + 1, number=number, sub=sub, section=self.section, text=text)
        self.paras.append(para)
        if self._header_open:
            self.header.append(text)
        self._maybe_heading = para if plain and first_line and _heading_like(text) else None

    def _is_title(self, text: str) -> bool:
        m = TITLE_RE.match(text)
        return bool(m) and self.case_id in m["ids"]

    def warnings(self) -> list[str]:
        extra = [f"... and {self._overflow} more numbering warnings"] if self._overflow else []
        return self.numbering + extra


class _Walker:
    """Walks the content container in document order, feeding lines to the builder."""

    def __init__(self, builder: _Builder) -> None:
        self.b = builder

    def walk(self, el: Tag, item: _Item | None = None) -> None:
        buf: list[str] = []
        for child in el.children:
            if isinstance(child, Tag):
                if child.name in _BLOCK_TAGS or _has_block(child):
                    self._flush(buf, item)
                    buf = []
                    self.block(child, item)
                else:
                    buf.append(child.get_text())
            elif type(child) is NavigableString:  # skips comments, doctype, CDATA
                buf.append(str(child))
        self._flush(buf, item)

    def _flush(self, buf: list[str], item: _Item | None) -> None:
        self._lines("".join(buf), item)

    def _lines(self, text: str, item: _Item | None, bold: bool = False) -> None:
        """Lines split at <br>; a blank line (<br><br>) starts a new paragraph group."""
        for chunk in _PARA_BREAK.split(text):
            g = self.b.group()
            for ln in _split(chunk):
                self.b.line(ln, g, item, bold=bold)

    def block(self, el: Tag, item: _Item | None) -> None:
        name = el.name
        if name in _HEADING_TAGS:
            text = " ".join(_split(el.get_text()))
            if item is None:
                self.b.heading(text)
            else:
                self.b.line(text, self.b.group(), item)
        elif name in _LIST_TAGS:
            self.list(el, item)
        elif name == "tr":
            self.row(el, item)
        elif not _has_block(el):
            text = el.get_text()
            self._lines(text, item, bold=len(_split(text)) == 1 and _bold_only(el))
        else:
            self.walk(el, item)

    def row(self, tr: Tag, item: _Item | None) -> None:
        cells = tr.find_all(["td", "th"], recursive=False)
        if any(_has_block(c) for c in cells):
            for c in cells:
                self.walk(c, item)
            return
        texts = [t for t in (" ".join(_split(c.get_text())) for c in cells) if t]
        if not texts:
            return
        # A layout table may hold the number in its own cell: "12." | "The landlord ..."
        if len(texts) > 1 and re.fullmatch(r"\d{1,3}\.?|\([a-z]{1,2}\)", texts[0]):
            first = texts[0] if texts[0].endswith((".", ")")) else texts[0] + "."
            text = f"{first} {' | '.join(texts[1:])}"
        else:
            text = " | ".join(texts)
        self.b.line(text, self.b.group(), item)

    def list(self, el: Tag, parent: _Item | None) -> None:
        b = self.b
        ordered = el.name == "ol"
        start = _int_attr(el, "start")
        style = _list_style(el)
        lettered = style in ("a", "A", "i", "I")
        if parent is None and ordered and not lettered and (start is not None or not b.literal_mode):
            counter = start if start is not None else (b.last_main or 0) + 1
            for child in el.children:
                if isinstance(child, Tag) and child.name == "li":
                    value = _int_attr(child, "value")
                    counter = value if value is not None else counter
                    it = _Item(number=counter, kind="main", pending=True)
                    self.walk(child, it)
                    counter = (it.number if it.number is not None else counter) + 1
                else:
                    self._other(child, None)
            return

        number = parent.number if parent is not None else b.current
        prefix = parent.sub if parent is not None else None
        as_sub = ordered and number is not None
        k = start if start is not None else 1
        for child in el.children:
            if not (isinstance(child, Tag) and child.name == "li"):
                self._other(child, parent)
                continue
            if as_sub:
                value = _int_attr(child, "value")
                k = value if value is not None else k
                label = _marker(k, style or "a")
                it = _Item(number=number, sub=f"{prefix}.{label}" if prefix else label, kind="sub",
                           pending=True, prefix=prefix, parent=parent)
                self.walk(child, it)
                # A literal "(c)" in the item text overrides the counter; carry on from it.
                shown = (it.sub or "").rsplit(".", 1)[-1].lower()
                k = ord(shown) - 96 + 1 if shown != label.lower() and len(shown) == 1 and shown.isalpha() else k + 1
            else:
                self.walk(child, _Item(number=parent.number if parent else None, sub=prefix, parent=parent))

    def _other(self, child: object, item: _Item | None) -> None:
        """Stray content directly inside a list (a heading between <li>s, loose text)."""
        if isinstance(child, Tag):
            self.block(child, item)
        elif type(child) is NavigableString:
            self._lines(str(child), item)


# --------------------------------------------------------------------------
# Assembling the Decision
# --------------------------------------------------------------------------


def _finish(
    b: _Builder,
    case_id: str,
    url: str | None,
    titles: list[str],
    page_text: str | None = None,
    meta_date: date | None = None,
) -> Decision:
    warnings = b.warnings()
    landlord, title = "", ""
    hit = _title_match(titles, case_id)
    if hit:
        landlord, title, ids = hit
        if case_id not in ids:
            warnings.append(f"title case id {'/'.join(ids)} differs from {case_id}")
    if not landlord:
        landlord = _landlord_after_complaint_line(b.header) or ""
    if not landlord and url:
        landlord = _landlord_from_url(url) or ""
        if landlord:
            warnings.append("landlord taken from URL slug")
    if not landlord:
        warnings.append("landlord not found")

    decision_date = next((d for d in map(find_date, b.header[:40]) if d), None)
    if decision_date is None and page_text:
        m = re.search(rf"complaint\s+{re.escape(case_id)}", page_text, re.I)
        decision_date = find_date(page_text[m.end(): m.end() + 300]) if m else None
    if decision_date is None and meta_date:
        decision_date = meta_date
        warnings.append("decision date taken from page metadata (publication date)")
    if decision_date is None:
        warnings.append("decision date not found; year from case id")
    year = decision_date.year if decision_date else (int(case_id[:4]) if case_id[:4].isdigit() else None)

    paras = b.paras
    if not paras:
        warnings.append("no paragraphs extracted")
    if not any(p.number is not None for p in paras):
        warnings.append("no numbered paragraphs")
    if not any(section_kind(p.section) == "determination" for p in paras):
        warnings.append("no Determination section found")
    return Decision(
        case_id=case_id,
        landlord=landlord,
        title=title or (titles[-1] if titles else ""),
        url=url,
        decision_date=decision_date,
        year=year,
        paragraphs=paras,
        parse_warnings=warnings,
    )


def _title_candidates(soup: BeautifulSoup) -> list[str]:
    out = [_clean(h1.get_text(" ")) for h1 in soup.find_all("h1")]
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        out.append(_clean(str(og["content"])))
    if soup.title:
        out.append(_clean(soup.title.get_text()))
    return [t for t in out if t]


def _title_match(titles: list[str], case_id: str) -> tuple[str, str, list[str]] | None:
    fallback = None
    for t in titles:
        m = TITLE_RE.match(t)
        if not m:
            continue
        ids = re.findall(r"\d{6,}", m["ids"])
        hit = (m["landlord"].strip(" -–:|"), _clean(m.group(0)), ids)
        if case_id in ids:
            return hit
        fallback = fallback or hit
    return fallback


def _case_id_from(titles: list[str], url: str | None) -> str | None:
    for t in titles:
        if m := TITLE_RE.match(t):
            return re.findall(r"\d{6,}", m["ids"])[0]
    if url and (m := re.search(r"(\d{6,})/?$", urlparse(url).path)):
        return m[1]
    return None


def _canonical_url(soup: BeautifulSoup) -> str | None:
    link = soup.find("link", rel="canonical")
    if link and link.get("href"):
        return str(link["href"])
    og = soup.find("meta", attrs={"property": "og:url"})
    return str(og["content"]) if og and og.get("content") else None


def _meta_date(soup: BeautifulSoup) -> date | None:
    meta = soup.find("meta", attrs={"property": "article:published_time"})
    value = meta.get("content") if meta else None
    if not value and (t := soup.find("time", attrs={"datetime": True})):
        value = t["datetime"]
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(value or ""))
    try:
        return date(int(m[1]), int(m[2]), int(m[3])) if m else None
    except ValueError:
        return None


def _landlord_after_complaint_line(header: list[str]) -> str | None:
    """Fallback: the line after "COMPLAINT 202420779" in the report header."""
    for i, line in enumerate(header[:-1]):
        if _COMPLAINT_LINE.match(line):
            nxt = header[i + 1]
            if len(nxt) < 150 and not find_date(nxt) and not re.search(r"\d{6,}", nxt):
                return nxt
    return None


_SMALL_WORDS = {"of", "and", "the", "for", "in", "on", "upon", "at", "by", "to", "a", "an", "under"}


def _landlord_from_url(url: str) -> str | None:
    """Fallback: ".../decisions/london-borough-of-sutton-202420779/" -> "London Borough of Sutton"."""
    slug = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"(?:-\d{6,})+$", "", slug)
    words = [w for w in slug.split("-") if w]
    if not words:
        return None
    return " ".join(w if i and w in _SMALL_WORDS else w.capitalize() for i, w in enumerate(words))


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _clean(text: str) -> str:
    return _WS.sub(" ", _ZERO_WIDTH.sub("", text)).strip()


def _split(text: str) -> list[str]:
    return [ln for ln in (_clean(part) for part in text.split(_BR)) if ln]


def _heading_key(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    return _clean(text).rstrip(":").strip().lower()


# bs4's find() with a list of names rebuilds its matcher on every call, which
# dominates the run time on 18k pages; these plain walks avoid that.
def _tags(el: Tag, names: set[str]):
    return (d for d in el.descendants if isinstance(d, Tag) and d.name in names)


def _contains(el: Tag, names: set[str]) -> bool:
    return next(_tags(el, names), None) is not None


def _has_block(el: Tag) -> bool:
    return _contains(el, _BLOCK_TAGS)


def _inside(el: Tag, name: str) -> bool:
    return any(p.name == name for p in el.parents)


def _is_heading(text: str, bold: bool) -> bool:
    if len(text) > 200 or _LITERAL_NUM.match(text) or _SUB_ITEM.match(text):
        return False
    words = len(text.split())
    plain_ending = not text.rstrip().endswith((".", ";", ","))
    key = _heading_key(text)
    if words <= 20 and plain_ending and not re.search(r"\d", text):
        if section_kind(key) != "other" or _OTHER_HEADINGS.match(key):
            return True
    return bold and words <= 15 and plain_ending and not find_date(text) and not re.search(r"\d{6,}", text)


def _heading_like(text: str) -> bool:
    """Could this short plain line be a heading? Confirmed only if a numbered paragraph follows."""
    words = text.split()
    return (
        1 <= len(words) <= 12
        and text[0].isupper()
        and not re.search(r"\d", text)
        and not text.endswith((".", ":", ";", ",", "?", "!", '"', "”", "’", "'", ")"))
    )


def _bold_only(el: Tag) -> bool:
    found = False
    for s in el.find_all(string=True):
        if type(s) is not NavigableString or not s.replace(_BR, "").strip():
            continue
        found = True
        if not _is_bold(s, el):
            return False
    return found


def _is_bold(s: NavigableString, stop: Tag) -> bool:
    for p in s.parents:
        style = str(p.get("style") or "").replace(" ", "").lower()
        if p.name in ("strong", "b") or re.search(r"font-weight:(bold|[6-9]00)", style):
            return True
        if p is stop:
            return False
    return False


def _int_attr(el: Tag, name: str) -> int | None:
    try:
        return int(str(el.get(name)).strip())
    except (TypeError, ValueError):
        return None


_STYLE_TYPES = {
    "decimal": "1", "lower-alpha": "a", "lower-latin": "a", "upper-alpha": "A", "upper-latin": "A",
    "lower-roman": "i", "upper-roman": "I",
}


def _list_style(el: Tag) -> str | None:
    t = str(el.get("type") or "").strip()
    if t in ("1", "a", "A", "i", "I"):
        return t
    m = re.search(r"list-style(?:-type)?\s*:\s*([a-z-]+)", str(el.get("style") or ""), re.I)
    return _STYLE_TYPES.get(m[1].lower()) if m else None


def _marker(k: int, style: str) -> str:
    """The label an <ol> of this type shows for item k (1-based)."""
    if style == "1":
        return str(k)
    if style in ("i", "I"):
        out, n = "", k
        for value, sym in ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
                           (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
            while n >= value:
                out, n = out + sym, n - value
        return out.upper() if style == "I" else out
    out, n = "", k
    while n > 0:
        n, r = divmod(n - 1, 26)
        out = chr(97 + r) + out
    return out.upper() if style == "A" else out


def _to_date(m: re.Match) -> date | None:
    month = m["m"]
    mo = int(month) if month.isdigit() else _MONTHS.get(month.lower())
    year = int(m["y"])
    if not mo or not 1990 <= year <= 2100:
        return None
    try:
        return date(year, mo, int(m["d"]))
    except ValueError:
        return None


def _describe_el(el: Tag) -> str:
    out = el.name
    if el.get("id"):
        out += f"#{el['id']}"
    classes = el.get("class") or []
    if classes:
        out += "." + ".".join(classes[:2])
    return out


def _tally(items: list[str]) -> str:
    if not items:
        return "(none)"
    return ", ".join(f"{k} x{n}" for k, n in Counter(items).most_common(10))
