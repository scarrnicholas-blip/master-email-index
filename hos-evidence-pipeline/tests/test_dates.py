from __future__ import annotations

from datetime import date

import pytest

from hos_pipeline.llm.dates import bounds_consistent, parse_date_bounds

D = date


@pytest.mark.parametrize(
    "text, context_year, expected",
    [
        # day precision
        ("17 June 2025", None, ("day", D(2025, 6, 17), D(2025, 6, 17))),
        ("17/06/2025", None, ("day", D(2025, 6, 17), D(2025, 6, 17))),
        ("17th June 2025", None, ("day", D(2025, 6, 17), D(2025, 6, 17))),
        ("17th of June, 2025", None, ("day", D(2025, 6, 17), D(2025, 6, 17))),
        ("Tuesday 17 June 2025", None, ("day", D(2025, 6, 17), D(2025, 6, 17))),
        ("June 17, 2025", None, ("day", D(2025, 6, 17), D(2025, 6, 17))),
        ("2025-06-17", None, ("day", D(2025, 6, 17), D(2025, 6, 17))),
        ("17/6/25", None, ("day", D(2025, 6, 17), D(2025, 6, 17))),
        ("on or around 17 June 2025", None, ("day", D(2025, 6, 17), D(2025, 6, 17))),
        ("29 February 2024", None, ("day", D(2024, 2, 29), D(2024, 2, 29))),
        ("17 June", 2024, ("day", D(2024, 6, 17), D(2024, 6, 17))),
        # month precision; qualifiers never narrow
        ("June 2025", None, ("month", D(2025, 6, 1), D(2025, 6, 30))),
        ("in Sept 2023", None, ("month", D(2023, 9, 1), D(2023, 9, 30))),
        ("early June 2025", None, ("month", D(2025, 6, 1), D(2025, 6, 30))),
        ("mid-June 2025", None, ("month", D(2025, 6, 1), D(2025, 6, 30))),
        ("late June 2025", None, ("month", D(2025, 6, 1), D(2025, 6, 30))),
        ("the end of February 2024", None, ("month", D(2024, 2, 1), D(2024, 2, 29))),
        ("the middle of June 2025", None, ("month", D(2025, 6, 1), D(2025, 6, 30))),
        ("June", 2024, ("month", D(2024, 6, 1), D(2024, 6, 30))),
        # year precision: full year unless a season is explicit
        ("2025", None, ("year", D(2025, 1, 1), D(2025, 12, 31))),
        ("late 2022", None, ("year", D(2022, 1, 1), D(2022, 12, 31))),
        ("mid-2023", None, ("year", D(2023, 1, 1), D(2023, 12, 31))),
        ("summer 2023", None, ("year", D(2023, 6, 1), D(2023, 8, 31))),
        ("the spring of 2023", None, ("year", D(2023, 3, 1), D(2023, 5, 31))),
        ("autumn 2023", None, ("year", D(2023, 9, 1), D(2023, 11, 30))),
        ("winter 2022/23", None, ("year", D(2022, 12, 1), D(2023, 2, 28))),
        ("winter 2022", None, ("year", D(2021, 12, 1), D(2023, 2, 28))),  # both readings
        # ranges: precision follows the width
        ("3 to 10 March 2025", None, ("month", D(2025, 3, 3), D(2025, 3, 10))),
        ("between June and August 2023", None, ("year", D(2023, 6, 1), D(2023, 8, 31))),
        ("November to February 2023", None, ("year", D(2022, 11, 1), D(2023, 2, 28))),
        # unknown
        ("17 June", None, ("unknown", None, None)),
        ("31 June 2025", None, ("unknown", None, None)),
        ("29/02/2023", None, ("unknown", None, None)),
        ("by 17 June 2025", None, ("unknown", None, None)),
        ("late 17 June 2025", None, ("unknown", None, None)),
        ("shortly afterwards", None, ("unknown", None, None)),
        ("", None, ("unknown", None, None)),
    ],
)
def test_parse_date_bounds(text, context_year, expected):
    assert parse_date_bounds(text, context_year) == expected


def test_bounds_consistent():
    assert bounds_consistent("day", D(2024, 1, 5), D(2024, 1, 5))
    assert not bounds_consistent("day", D(2024, 1, 5), D(2024, 1, 6))
    assert bounds_consistent("month", D(2024, 2, 1), D(2024, 2, 29))
    assert not bounds_consistent("month", D(2024, 2, 3), D(2024, 2, 29))
    assert bounds_consistent("year", D(2024, 1, 1), D(2024, 12, 31))
    assert not bounds_consistent("year", D(2024, 12, 31), D(2024, 1, 1))
    assert not bounds_consistent("unknown", None, None)
