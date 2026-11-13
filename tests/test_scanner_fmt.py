"""Number and date formatting in both languages."""

import math
from datetime import date, datetime

import pandas as pd
import pytest

from tradingagents.scanner.fmt import fmt_date, fmt_num, fmt_pct, fmt_points, fmt_price


@pytest.mark.unit
@pytest.mark.parametrize("bad", [None, float("nan"), math.inf, -math.inf, "abc"])
def test_missing_values_render_a_dash(bad):
    for lang in ("en", "de"):
        assert fmt_num(bad, lang=lang) == "-"
        assert fmt_pct(bad, lang=lang) == "-"
        assert fmt_price(bad, lang=lang) == "-"
        assert fmt_points(bad, lang=lang) == "-"
    assert fmt_date(None) == "-"


@pytest.mark.unit
def test_fmt_num_thousands_and_decimals():
    assert fmt_num(1234567.891) == "1,234,567.89"
    assert fmt_num(1234567.891, lang="de") == "1.234.567,89"
    assert fmt_num(3.14159, digits=3) == "3.142"
    assert fmt_num(0, digits=0) == "0"


@pytest.mark.unit
def test_fmt_pct_is_a_fraction_and_signed():
    assert fmt_pct(0.123) == "+12.3 %"
    assert fmt_pct(0.123, lang="de") == "+12,3 %"
    assert fmt_pct(-0.0456, digits=2) == "-4.56 %"
    assert fmt_pct(0.123, signed=False) == "12.3 %"
    assert fmt_pct(12.34, lang="de") == "+1.234,0 %"


@pytest.mark.unit
def test_fmt_price_digits_follow_size():
    assert fmt_price(69271.61) == "69,271.61"
    assert fmt_price(69271.61, lang="de") == "69.271,61"
    assert fmt_price(226.95) == "226.95"
    assert fmt_price(0.123456) == "0.1235"
    assert fmt_price(0.123456, lang="de") == "0,1235"


@pytest.mark.unit
def test_fmt_points():
    assert fmt_points(4.0) == "+4.0 pts"
    assert fmt_points(4.0, lang="de") == "+4,0 Punkte"
    assert fmt_points(-9.94) == "-9.9 pts"


@pytest.mark.unit
def test_fmt_date_accepts_several_types():
    assert fmt_date("2026-11-13") == "2026-11-13"
    assert fmt_date("2026-11-13", "de") == "13.11.2026"
    assert fmt_date(date(2026, 1, 5), "de") == "05.01.2026"
    assert fmt_date(datetime(2026, 1, 5, 9, 30)) == "2026-01-05"
    assert fmt_date(pd.Timestamp("2026-01-05"), "de") == "05.01.2026"
    assert fmt_date(pd.NaT) == "-"
