"""Intraday bars (5-minute and 60-minute) for the short-horizon research.

Yahoo serves 5-minute bars for the last 60 days only and 60-minute bars for the last
730, so a study of intraday rules can never be longer than that; the limits are
enforced here (``days`` is clamped one day inside them) instead of being left to a
confusing Yahoo error. Bars are the regular session only (``prepost=False``): the
rules in the study are about the 09:30-16:00 session and pre/post-market quotes are
thin and not tradable for a retail account.

What Yahoo returns, as checked on real data: each bar is stamped with its START
time in exchange-local wall clock (first US bar 09:30; last 5-minute bar 15:55; the
60-minute bars start 09:30, 10:30 ... 15:30 and the last one covers only 30 minutes);
half days simply have fewer bars; a few bars go missing. The loader does not paper
over any of that: it returns what Yahoo has, sorted and de-duplicated, and the
study decides which sessions are complete enough to trade on.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.common import yf_retry

logger = logging.getLogger(__name__)

BATCH_SIZE = 25
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
# interval -> the longest look-back (in calendar days) we dare to request
MAX_DAYS = {"5m": 59, "60m": 729}


def _extract(raw: pd.DataFrame, symbol: str, single: bool, tz: str) -> pd.DataFrame | None:
    """One symbol's intraday OHLCV out of a (usually multi-symbol) download, or None."""
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol not in raw.columns.get_level_values(0):
            return None
        frame = raw[symbol]
    elif single:
        frame = raw
    else:
        return None
    if "Close" not in frame.columns:
        return None
    frame = frame.reindex(columns=OHLCV_COLUMNS).apply(pd.to_numeric, errors="coerce")
    idx = pd.DatetimeIndex(frame.index)
    # Yahoo stamps UTC or exchange-local; a naive index is read as UTC, then every bar
    # is put on the exchange wall clock so "09:30" means the open wherever we run.
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    frame.index = idx.tz_convert(tz)
    frame = frame.dropna(subset=["Close"])
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame if not frame.empty else None


def load_intraday_ohlcv(
    symbols: list[str],
    interval: str,
    days: int,
    tz: str = "America/New_York",
    pause: float = 0.5,
    end: str | pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Regular-session intraday OHLCV for each symbol, NOT dividend-adjusted.

    Yahoo adjusts daily bars for dividends (with ``auto_adjust=True``) but serves
    intraday bars as traded: the price level of an intraday bar is the one that was on
    the screen, so it differs from the adjusted daily close of the same day by the
    dividends paid since (several percent for high yielders over two years). Never
    compare an intraday price with a daily-bar price; take every reference level
    (previous close, ...) from the intraday frame itself.

    ``interval`` is ``"5m"`` (at most the last 59 days are asked for) or ``"60m"``
    (at most 729). Returns ``{symbol: frame}`` with Open/High/Low/Close/Volume and a
    tz-aware ascending DatetimeIndex in ``tz`` (bar START times), without duplicates
    and without rows that lack a close. A symbol Yahoo has nothing for is absent;
    no symbol at all raises ``VendorUnavailableError``, which says Yahoo failed
    rather than that every symbol is empty. A batch that keeps failing does not abort
    the load: the symbols of the failed batches are logged and left out, and the error
    is raised only when nothing at all came back. ``pause`` seconds are slept between
    batches to stay gentle with Yahoo. ``end`` (a date, default: now) anchors the
    window on the last session wanted, inclusive, instead of the wall clock; the
    window is then ``days`` calendar days up to it.
    """
    if interval not in MAX_DAYS:
        raise ValueError(f"interval must be one of {sorted(MAX_DAYS)}, got {interval!r}")
    days = max(1, min(int(days), MAX_DAYS[interval]))
    wanted = list(dict.fromkeys(normalize_symbol(s) for s in symbols))
    last_day = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() if end is None \
        else pd.Timestamp(end).tz_localize(None).normalize()
    start = (last_day - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    stop = (last_day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")    # Yahoo's end is exclusive

    out: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    last_error: Exception | None = None
    for i in range(0, len(wanted), BATCH_SIZE):
        batch = wanted[i:i + BATCH_SIZE]
        if i and pause:
            time.sleep(pause)
        try:
            raw = yf_retry(lambda batch=batch: yf.download(
                batch, start=start, end=stop, interval=interval, prepost=False,
                auto_adjust=True, progress=False, threads=True, group_by="ticker",
            ))
        except VendorUnavailableError as exc:
            logger.warning("intraday batch %d-%d failed, symbols left out: %s (%s)",
                           i, i + len(batch) - 1, ", ".join(batch), exc)
            failed.extend(batch)
            last_error = exc
            continue
        for symbol in batch:
            frame = _extract(raw, symbol, single=len(batch) == 1, tz=tz)
            if frame is not None:
                out[symbol] = frame
    if failed:
        logger.warning("intraday load kept %d of %d symbols; failed batches: %s",
                       len(out), len(wanted), ", ".join(failed))
    if wanted and not out:
        raise VendorUnavailableError(
            f"Yahoo Finance returned no {interval} bars for any of {len(wanted)} symbols"
        ) from last_error
    return out
