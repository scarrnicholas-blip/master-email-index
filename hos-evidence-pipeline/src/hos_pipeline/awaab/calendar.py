"""England and Wales working-day calendar for Awaab's Law deadlines.

Convention
----------
The trigger day (e.g. the day the landlord becomes aware) is day 0 and working
day 1 is the first working day after it. A 10-working-day duty is therefore met
if the action date <= ``add_working_days(aware, 10)``, and
``working_days_between(aware, action)`` counts the working days after ``aware``
up to and including ``action``.

A working day is any day except Saturday, Sunday and an England and Wales bank
holiday. Christmas Day and Good Friday are common-law holidays rather than
Banking and Financial Dealings Act 1971 bank holidays; gov.uk lists them as bank
holidays and this calendar always excludes them, even when a supplied
bank-holidays.json omits them.

Built-in holidays follow the regular rules for any year; the moved and one-off
holidays below are complete for 2010-2026 and none are assumed after that, so
for future years a gov.uk bank-holidays.json should be supplied when available.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

BUILTIN_YEARS = range(2010, 2041)

# Regular dates moved by royal proclamation: regular date -> replacement date(s).
_MOVED: dict[date, tuple[date, ...]] = {
    date(2012, 5, 28): (date(2012, 6, 4),),  # spring holiday moved for the Diamond Jubilee
    date(2020, 5, 4): (date(2020, 5, 8),),  # early May holiday moved to Friday 8 May (VE Day 75)
    date(2022, 5, 30): (date(2022, 6, 2),),  # spring holiday moved to Thursday 2 June (Platinum Jubilee)
}

# Additional one-off bank holidays.
_EXTRA: tuple[date, ...] = (
    date(2011, 4, 29),  # wedding of Prince William and Catherine Middleton
    date(2012, 6, 5),  # Queen's Diamond Jubilee
    date(2022, 6, 3),  # Queen's Platinum Jubilee
    date(2022, 9, 19),  # State Funeral of Queen Elizabeth II
    date(2023, 5, 8),  # Coronation of King Charles III
)


def easter_sunday(year: int) -> date:
    """Western Easter Sunday by the Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741 - the algorithm's own name
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def good_friday(year: int) -> date:
    return easter_sunday(year) - timedelta(days=2)


def _first_monday(year: int, month: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(0 - d.weekday()) % 7)


def _last_monday(year: int, month: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1)
    return d - timedelta(days=d.weekday())


def _new_year(year: int) -> date:
    d = date(year, 1, 1)
    # Saturday -> Monday 3rd, Sunday -> Monday 2nd.
    return d + timedelta(days=7 - d.weekday()) if d.weekday() >= 5 else d


def _christmas(year: int) -> tuple[date, date]:
    """Christmas Day and Boxing Day, or their substitute days."""
    weekday = date(year, 12, 25).weekday()
    if weekday == 4:  # Boxing Day falls on Saturday: substitute Monday 28th
        return date(year, 12, 25), date(year, 12, 28)
    if weekday == 5:  # Christmas Saturday: substitutes Monday 27th and Tuesday 28th
        return date(year, 12, 27), date(year, 12, 28)
    if weekday == 6:  # Christmas Sunday: Boxing Day Monday 26th, Christmas substitute Tuesday 27th
        return date(year, 12, 26), date(year, 12, 27)
    return date(year, 12, 25), date(year, 12, 26)


def builtin_bank_holidays(year: int) -> list[date]:
    """England and Wales bank holidays for ``year`` from the built-in rules."""
    easter = easter_sunday(year)
    regular = [
        _new_year(year),
        easter - timedelta(days=2),  # Good Friday
        easter + timedelta(days=1),  # Easter Monday
        _first_monday(year, 5),  # early May bank holiday
        _last_monday(year, 5),  # spring bank holiday
        _last_monday(year, 8),  # summer bank holiday
        *_christmas(year),
    ]
    days = {moved for d in regular for moved in _MOVED.get(d, (d,))}
    days.update(d for d in _EXTRA if d.year == year)
    return sorted(days)


def load_govuk_bank_holidays(path: Path) -> dict[int, set[date]]:
    """England and Wales dates from a gov.uk bank-holidays.json, grouped by year."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    events = data.get("england-and-wales", data).get("events", [])
    by_year: dict[int, set[date]] = {}
    for event in events:
        d = date.fromisoformat(str(event["date"])[:10])
        by_year.setdefault(d.year, set()).add(d)
    return by_year


class WorkingCalendar:
    """England and Wales working days.

    If ``bank_holidays_json`` (gov.uk format) is given, its dates take precedence
    over the built-in list for every year the file covers; other years use the
    built-in rules. Christmas Day and Good Friday are always non-working.
    """

    def __init__(self, bank_holidays_json: Path | None = None) -> None:
        self._govuk: dict[int, set[date]] = (
            load_govuk_bank_holidays(bank_holidays_json) if bank_holidays_json is not None else {}
        )
        self._cache: dict[int, frozenset[date]] = {}
        self.source = str(bank_holidays_json) if bank_holidays_json is not None else "built-in"

    def _holidays(self, year: int) -> frozenset[date]:
        cached = self._cache.get(year)
        if cached is None:
            days = set(self._govuk[year]) if year in self._govuk else set(builtin_bank_holidays(year))
            for d in (good_friday(year), date(year, 12, 25)):
                if d.weekday() < 5:
                    days.add(d)
            cached = self._cache[year] = frozenset(days)
        return cached

    def bank_holidays(self, year: int) -> list[date]:
        return sorted(self._holidays(year))

    def is_working_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self._holidays(d.year)

    def roll_forward(self, d: date) -> date:
        """``d`` if it is a working day, else the next working day."""
        while not self.is_working_day(d):
            d += timedelta(days=1)
        return d

    def add_working_days(self, start: date, n: int) -> date:
        """The n-th working day after ``start`` (``start`` itself is day 0)."""
        if n < 0:
            raise ValueError("n must be >= 0")
        d = start
        while n:
            d += timedelta(days=1)
            if self.is_working_day(d):
                n -= 1
        return d

    def working_days_between(self, start: date, end: date) -> int:
        """Working days after ``start`` up to and including ``end``; 0 if end <= start."""
        if end <= start:
            return 0
        weeks, rem = divmod((end - start).days, 7)
        count = weeks * 5  # every run of 7 consecutive days holds 5 weekdays
        d = start + timedelta(days=weeks * 7)
        for _ in range(rem):
            d += timedelta(days=1)
            if d.weekday() < 5:
                count += 1
        for year in range(start.year, end.year + 1):
            count -= sum(1 for h in self._holidays(year) if start < h <= end and h.weekday() < 5)
        return count
