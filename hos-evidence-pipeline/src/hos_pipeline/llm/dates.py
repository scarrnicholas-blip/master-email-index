"""Deterministic date bounds for event dates as written in decisions.

``parse_date_bounds`` turns ``Event.date_text`` into (precision, earliest,
latest). The coder recomputes every event's bounds with it rather than trusting
the model's own reading, so that "late June 2023" never silently becomes a day.

Conventions
-----------
- Day-first numeric dates (UK): "17/06/2025", "17.06.25", "2025-06-17".
- Qualifiers never narrow the bounds: "early/mid/late June 2025" and "end of
  June 2025" are the whole month; "late 2022" and "mid-2023" are the whole year.
- Seasons are the one exception to full-year bounds, because the season is
  explicit: spring = Mar-May, summer = Jun-Aug, autumn/fall = Sep-Nov. Winter
  is Dec-Feb: "winter 2022/23" is 1 Dec 2022 - 28 Feb 2023, and a bare "winter
  2022" is ambiguous, so it gets the union of both readings (1 Dec 2021 - end
  Feb 2023). Seasons are labelled precision "year" (coarser than a month); the
  earliest/latest bounds are always the authoritative interval.
- Ranges ("between June and August 2023", "3 to 10 March 2025") give the
  coarser precision of the two ends and span from the first to the last.
- A day or month without a year uses ``context_year`` when given; otherwise
  it is unknown.
- Anything else, including impossible dates, is ("unknown", None, None).
"""

from __future__ import annotations

import calendar
import re
from datetime import date

from ..models import DatePrecision

Bounds = tuple[DatePrecision, date | None, date | None]
UNKNOWN: Bounds = ("unknown", None, None)

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH = r"(?P<month>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_DAY = r"(?P<day>\d{1,2})(?:st|nd|rd|th)?"
_YEAR = r"(?P<year>(?:19|20)\d{2})"
_WEEKDAY = r"(?:(?:mon|tues|wednes|thurs|fri|satur|sun)day,?\s+)?"
_RANK = {"day": 0, "month": 1, "year": 2}

# Leading words that say nothing about the bounds. "by"/"before"/"after" are
# deliberately absent: they change the interval, so such text stays unknown.
_FILLER = re.compile(
    r"^(?:(?:on\s+or\s+(?:about|around)|on|in|during|around|about|approximately|approx\.?"
    r"|circa|c\.|roughly|sometime\s+in|some\s+time\s+in|at\s+the\s+time\s+of|the)\s+)+"
)
# Qualifiers inside a month or year that must not narrow it.
_QUALIFIER = re.compile(
    r"^(?:the\s+)?(?:first\s+half|second\s+half|beginning|middle|early|start|late|end|mid)(?:\s+of)?[\s-]*"
)

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("iso", re.compile(r"^(?P<year>\d{4})-(?P<m>\d{1,2})-(?P<day>\d{1,2})$")),
    ("numeric", re.compile(r"^(?P<day>\d{1,2})[/.\-](?P<m>\d{1,2})[/.\-](?P<year>\d{2}|\d{4})$")),
    ("dmy", re.compile(rf"^{_WEEKDAY}{_DAY}(?:\s+of)?\s+{_MONTH},?\s+{_YEAR}$")),
    ("mdy", re.compile(rf"^{_WEEKDAY}{_MONTH}\s+{_DAY},?\s+{_YEAR}$")),
    ("dm", re.compile(rf"^{_WEEKDAY}{_DAY}(?:\s+of)?\s+{_MONTH}$")),
    ("md", re.compile(rf"^{_WEEKDAY}{_MONTH}\s+{_DAY}$")),
    ("my", re.compile(rf"^{_MONTH},?\s+(?:of\s+)?{_YEAR}$")),
    ("m", re.compile(rf"^{_MONTH}$")),
    ("season", re.compile(r"^(?P<season>spring|summer|autumn|fall|winter)\s+(?:of\s+)?"
                          r"(?P<year>(?:19|20)\d{2})(?:\s*[/-]\s*(?P<year2>\d{2}|\d{4}))?$")),
    ("y", re.compile(rf"^{_YEAR}$")),
]


def parse_date_bounds(text: str, context_year: int | None = None) -> Bounds:
    """(precision, earliest, latest) for a date as written; see module docstring."""
    s = _normalise(text)
    if not s:
        return UNKNOWN
    single = _parse_single(s, context_year)
    if single[0] != "unknown":
        return single
    return _parse_range(s, context_year)


def bounds_consistent(precision: str, earliest: date | None, latest: date | None) -> bool:
    """Whether model-supplied bounds are internally consistent with their precision."""
    if precision not in _RANK or earliest is None or latest is None or earliest > latest:
        return False
    span = (latest - earliest).days
    if precision == "day":
        return span == 0
    if precision == "month":
        return earliest.day == 1 and latest == _month_end(earliest.year, earliest.month)
    return span <= 456  # a year, or a winter that straddles two years


# --------------------------------------------------------------------------


def _normalise(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("–", "-").replace("—", "-").replace(" ", " ")
    s = re.sub(r"\s+", " ", s).strip(" .,;:()")
    s = _FILLER.sub("", s)
    return s.strip()


def _parse_single(s: str, context_year: int | None) -> Bounds:
    stripped = _QUALIFIER.sub("", s).strip()
    qualified = stripped != s
    for kind, pat in _PATTERNS:
        m = pat.match(stripped)
        if not m:
            continue
        g = m.groupdict()
        # A qualifier only makes sense on a month or year ("late June 2025");
        # "late 17 June 2025" is not a date we can bound.
        if qualified and kind not in ("my", "m", "y", "season"):
            return UNKNOWN
        try:
            return _build(kind, g, context_year)
        except ValueError:
            return UNKNOWN
    return UNKNOWN


def _build(kind: str, g: dict[str, str | None], context_year: int | None) -> Bounds:
    if kind in ("iso", "numeric"):
        year = int(g["year"])
        if year < 100:
            year += 2000
        d = date(year, int(g["m"]), int(g["day"]))
        return ("day", d, d)
    if kind in ("dmy", "mdy", "dm", "md"):
        year = int(g["year"]) if g.get("year") else context_year
        if year is None:
            return UNKNOWN
        d = date(year, _MONTHS[g["month"]], int(g["day"]))
        return ("day", d, d)
    if kind in ("my", "m"):
        year = int(g["year"]) if g.get("year") else context_year
        if year is None:
            return UNKNOWN
        month = _MONTHS[g["month"]]
        return ("month", date(year, month, 1), _month_end(year, month))
    if kind == "season":
        return _season(g["season"], int(g["year"]), g.get("year2"))
    if kind == "y":
        year = int(g["year"])
        return ("year", date(year, 1, 1), date(year, 12, 31))
    raise ValueError(kind)


def _season(season: str, year: int, year2: str | None) -> Bounds:
    if season == "winter":
        if year2:
            end_year = int(year2) + (2000 if len(year2) == 2 else 0)
            if end_year != year + 1:
                raise ValueError("winter must span consecutive years")
            return ("year", date(year, 12, 1), _month_end(end_year, 2))
        return ("year", date(year - 1, 12, 1), _month_end(year + 1, 2))
    if year2:
        raise ValueError("only winter spans two years")
    start, end = {"spring": (3, 5), "summer": (6, 8), "autumn": (9, 11), "fall": (9, 11)}[season]
    return ("year", date(year, start, 1), _month_end(year, end))


_RANGE = re.compile(
    r"^(?:between\s+(?P<a1>.+?)\s+and\s+(?P<b1>.+)|(?:from\s+)?(?P<a2>.+?)\s*(?:\s(?:to|until|till)\s|-)\s*(?P<b2>.+))$"
)


def _parse_range(s: str, context_year: int | None) -> Bounds:
    """Ranges span both ends; precision is set by the width of the interval
    (one day -> day, up to 31 days -> month, wider -> year)."""
    m = _RANGE.match(s)
    if not m:
        return UNKNOWN
    a, b = (m["a1"], m["b1"]) if m["a1"] else (m["a2"], m["b2"])
    a = _normalise(a)
    end = _parse_single(_normalise(b), context_year)
    if end[0] == "unknown" or end[1] is None or end[2] is None:
        return UNKNOWN
    if re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)?", a) and end[0] == "day":
        try:  # "3 to 10 March 2025": the start borrows the end's month and year
            first = end[1].replace(day=int(re.sub(r"\D", "", a)))
        except ValueError:
            return UNKNOWN
        start: Bounds = ("day", first, first)
    else:
        start = _parse_single(a, context_year)
        if start[0] == "unknown":
            start = _parse_single(a, end[2].year)  # "June to August 2023"
            if start[1] is not None and start[1] > end[2]:
                start = _parse_single(a, end[2].year - 1)  # "November to February 2023"
    if start[1] is None or start[1] > end[2]:
        return UNKNOWN
    span = (end[2] - start[1]).days
    precision: DatePrecision = "day" if span == 0 else "month" if span <= 31 else "year"
    return (precision, start[1], end[2])


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])
