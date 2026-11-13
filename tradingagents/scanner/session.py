"""Is a daily bar finished? Yahoo returns today's bar with the live price while the market is open.

A scan run at 11:50 Berlin time would otherwise take the running Xetra session's price as
"the close" and compute ATR, entry and stop from it, and a US scan would pair yesterday's
prices with a live VIX. A bar therefore counts only once its exchange session has ended
plus ``BUFFER_MINUTES`` (final auctions, late prints, Yahoo's own delay).

Session end by symbol shape (local time; the buffer is added on top):

=====================================  ==================================
no suffix (US), ``^GSPC``              16:00 America/New_York
``^VIX``                               16:15 America/New_York
``.DE``, ``^GDAXI``                    17:45 Europe/Berlin
``.L``                                 16:35 Europe/London
``.PA .AS .MI .MC .SW``                17:35 Paris / Amsterdam / Rome / Madrid / Zurich
``.T``                                 15:30 Asia/Tokyo
``.HK``                                16:10 Asia/Hong_Kong
``.TO``                                16:00 America/Toronto
``.AX``                                16:12 Australia/Sydney
``.NS .BO``                            15:30 Asia/Kolkata
``.SS .SZ``                            15:00 Asia/Shanghai
``.WA``                                17:05 Europe/Warsaw
``*-USD`` (crypto)                     the UTC day is over (UTC date > bar date)
``=F`` futures, anything unknown       the UTC date is past the bar date
=====================================  ==================================

Time zones come from ``zoneinfo``, so daylight saving is handled by the database.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

BUFFER_MINUTES = 20

_US = (time(16, 0), "America/New_York")
_BY_SUFFIX: dict[str, tuple[time, str]] = {
    "DE": (time(17, 45), "Europe/Berlin"),
    "L": (time(16, 35), "Europe/London"),
    "PA": (time(17, 35), "Europe/Paris"),
    "AS": (time(17, 35), "Europe/Amsterdam"),
    "MI": (time(17, 35), "Europe/Rome"),
    "MC": (time(17, 35), "Europe/Madrid"),
    "SW": (time(17, 35), "Europe/Zurich"),
    "T": (time(15, 30), "Asia/Tokyo"),
    "HK": (time(16, 10), "Asia/Hong_Kong"),
    "TO": (time(16, 0), "America/Toronto"),
    "AX": (time(16, 12), "Australia/Sydney"),
    "NS": (time(15, 30), "Asia/Kolkata"),
    "BO": (time(15, 30), "Asia/Kolkata"),
    "SS": (time(15, 0), "Asia/Shanghai"),
    "SZ": (time(15, 0), "Asia/Shanghai"),
    "WA": (time(17, 5), "Europe/Warsaw"),
}
_BY_INDEX = {"^GSPC": _US, "^VIX": (time(16, 15), "America/New_York"),
             "^GDAXI": _BY_SUFFIX["DE"]}


def _session_end(symbol: str) -> tuple[time, str] | None:
    """``(local close time, zone)`` for an exchange-traded symbol, None for the UTC-day rule."""
    sym = symbol.strip().upper()
    if sym in _BY_INDEX:
        return _BY_INDEX[sym]
    if sym.startswith("^") or sym.endswith("=F") or sym.endswith("=X") or sym.endswith("-USD"):
        return None
    if "." in sym:
        return _BY_SUFFIX.get(sym.rsplit(".", 1)[1])
    return _US


def _utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    return now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)


def is_settled(symbol: str, bar_date, now_utc: datetime | None = None) -> bool:
    """True once the daily bar dated ``bar_date`` is final for ``symbol`` at ``now_utc``."""
    day = pd.Timestamp(bar_date).date()
    now = _utc(now_utc)
    rule = _session_end(symbol)
    if rule is None:
        return now.date() > day
    close, zone = rule
    ended = datetime.combine(day, close, tzinfo=ZoneInfo(zone)) + timedelta(
        minutes=BUFFER_MINUTES)
    return now >= ended


def drop_unsettled(
    frames: dict[str, pd.DataFrame], now_utc: datetime | None = None
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """``(frames without running bars, symbols that had one removed)``.

    A frame left empty is dropped, so the caller sees the symbol as having no data.
    """
    out: dict[str, pd.DataFrame] = {}
    trimmed: list[str] = []
    for symbol, frame in frames.items():
        keep = len(frame)
        while keep and not is_settled(symbol, frame.index[keep - 1], now_utc):
            keep -= 1
        if keep < len(frame):
            trimmed.append(symbol)
        if keep:
            out[symbol] = frame.iloc[:keep]
    return out, trimmed


def settled(frames: dict[str, pd.DataFrame], now_utc: datetime | None = None
            ) -> dict[str, pd.DataFrame]:
    """``frames`` without any trailing bar whose session is not yet over."""
    return drop_unsettled(frames, now_utc)[0]


__all__ = ["BUFFER_MINUTES", "drop_unsettled", "is_settled", "settled"]
