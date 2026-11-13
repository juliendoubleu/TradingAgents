"""Daily bars for many symbols in a few requests, for scanning a whole market.

``load_ohlcv`` serves one symbol into a per-symbol cache; a scan of 60 symbols
through it would make 60 sequential requests. Here the symbols go to Yahoo in
batches of ``BATCH_SIZE``. Nothing is cached: a scan wants the latest bars, and a
stale cache would silently age the scan.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.errors import VendorError, VendorUnavailableError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.common import yf_retry

logger = logging.getLogger(__name__)

BATCH_SIZE = 50
RETRY_PAUSE_S = 0.3     # between the single-symbol retries, so a throttle can ease
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _naive_dates(index) -> pd.DatetimeIndex:
    """Each bar's own local date, as a naive midnight timestamp.

    Dropping the zone (not converting to UTC) keeps a Tokyo or Frankfurt bar on the
    day it traded, and lets bars from different exchanges be compared with a naive
    ``as_of_date``.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _extract(raw: pd.DataFrame, symbol: str, single: bool) -> pd.DataFrame | None:
    """One symbol's OHLCV out of a (usually multi-symbol) download, or None if empty."""
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        # group_by="ticker" puts the symbol on level 0
        if symbol not in raw.columns.get_level_values(0):
            return None
        frame = raw[symbol]
    elif single:
        frame = raw
    else:
        return None
    if "Close" not in frame.columns:
        return None
    frame = frame.reindex(columns=OHLCV_COLUMNS)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame.index = _naive_dates(frame.index)
    # Rows without a close (holidays, gaps) go. A running session is NOT among them: Yahoo
    # returns today's bar with the live price as its "Close", so the caller must cut such
    # bars itself (``tradingagents.scanner.session``) -- this loader does not know the clock.
    frame = frame.dropna(subset=["Close"])
    frame = frame[~frame.index.isna()]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame if not frame.empty else None


def load_bulk_ohlcv(
    symbols: list[str], as_of_date: str, lookback_days: int
) -> dict[str, pd.DataFrame]:
    """Daily OHLCV (split/dividend adjusted) for each symbol, up to ``as_of_date``.

    Returns ``{symbol: frame}`` with Open/High/Low/Close/Volume columns, a naive
    ascending DatetimeIndex and no row without a close. No row is dated after
    ``as_of_date`` (the repo treats look-ahead as a bug), so a scan of a past date
    sees what was known then. ``yf.download`` swallows per-symbol failures (HTTP 429),
    so a symbol that came back empty from its batch is asked for once more on its own
    through ``yf_retry`` before it counts as missing. A symbol Yahoo has nothing for is
    absent from the result; the caller reports it as skipped. Raises ``VendorUnavailableError`` when
    no symbol comes back at all, which says Yahoo failed rather than that every
    symbol is empty.
    """
    as_of = pd.Timestamp(as_of_date).normalize()
    wanted = list(dict.fromkeys(normalize_symbol(s) for s in symbols))
    start = (as_of - pd.Timedelta(days=int(lookback_days))).strftime("%Y-%m-%d")
    # yfinance's ``end`` is exclusive; the as_of_date cut below is what prevents look-ahead.
    end = (as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    out: dict[str, pd.DataFrame] = {}
    empty: list[str] = []          # nothing came back for these (not even before the cut)
    for i in range(0, len(wanted), BATCH_SIZE):
        batch = wanted[i:i + BATCH_SIZE]
        raw = yf_retry(lambda batch=batch: yf.download(
            batch, start=start, end=end, auto_adjust=True, progress=False,
            threads=True, group_by="ticker",
        ))
        for symbol in batch:
            frame = _extract(raw, symbol, single=len(batch) == 1)
            if frame is None:
                empty.append(symbol)
                continue
            frame = frame[frame.index <= as_of]
            if not frame.empty:
                out[symbol] = frame
    for symbol in empty:
        time.sleep(RETRY_PAUSE_S)
        try:
            raw = yf_retry(lambda symbol=symbol: yf.download(
                [symbol], start=start, end=end, auto_adjust=True, progress=False,
                threads=False, group_by="ticker",
            ))
        except VendorError as exc:
            logger.warning("Retry for %s failed: %s", symbol, exc)
            continue
        frame = _extract(raw, symbol, single=True)
        if frame is not None:
            frame = frame[frame.index <= as_of]
            if not frame.empty:
                out[symbol] = frame
    if wanted and not out:
        raise VendorUnavailableError(
            f"Yahoo Finance returned no price data for any of {len(wanted)} symbols"
        )
    return out
