"""Number and date formatting for scanner output, in English and German.

One place decides how a figure looks so every later report agrees. The locale module
is deliberately not used: it is process-global (not thread-safe) and depends on which
locales the OS has installed, so the same run would print differently on two machines.
Instead the English rendering is built first and its separators are swapped.

A missing value (None, NaN, infinity) renders as "-": a blank or a "0" would read as a
measurement that was never made.
"""

from __future__ import annotations

import math
from datetime import date, datetime

MISSING = "-"


def _ok(x) -> bool:
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _localize(text: str, lang: str) -> str:
    """Swap '.' and ',' for German; the placeholder keeps the two swaps from colliding."""
    if lang == "de":
        return text.replace(",", "\0").replace(".", ",").replace("\0", ".")
    return text


def fmt_num(x, digits: int = 2, lang: str = "en") -> str:
    """A plain number with thousands separators (en 1,234.50 / de 1.234,50)."""
    if not _ok(x):
        return MISSING
    return _localize(f"{float(x):,.{digits}f}", lang)


def fmt_pct(x, digits: int = 1, signed: bool = True, lang: str = "en") -> str:
    """A fraction as a percentage: 0.123 -> "+12.3 %" (de "+12,3 %")."""
    if not _ok(x):
        return MISSING
    sign = "+" if signed else ""
    return _localize(f"{float(x) * 100:{sign},.{digits}f} %", lang)


def fmt_price(x, lang: str = "en") -> str:
    """A price with digits suited to its size: >= 1000 two decimals, < 1 four, else two.

    Sub-1 prices (some crypto, penny stocks) lose meaning at two decimals.
    """
    if not _ok(x):
        return MISSING
    v = float(x)
    digits = 4 if abs(v) < 1 else 2
    return _localize(f"{v:,.{digits}f}", lang)


def fmt_points(x, digits: int = 1, lang: str = "en") -> str:
    """Percentage points, signed: "+4.0 pts" / "+4,0 Punkte"."""
    if not _ok(x):
        return MISSING
    unit = "Punkte" if lang == "de" else "pts"
    return _localize(f"{float(x):+,.{digits}f}", lang) + f" {unit}"


def fmt_date(d, lang: str = "en") -> str:
    """ISO date for English, dd.mm.yyyy for German. Accepts date, datetime, Timestamp or ISO text."""
    if d is None:
        return MISSING
    if isinstance(d, str):
        try:
            d = datetime.strptime(d[:10], "%Y-%m-%d")
        except ValueError:
            return d
    try:
        if d != d:  # NaT
            return MISSING
        d = date(d.year, d.month, d.day)
    except (AttributeError, ValueError):
        return MISSING
    return d.strftime("%d.%m.%Y") if lang == "de" else d.isoformat()
