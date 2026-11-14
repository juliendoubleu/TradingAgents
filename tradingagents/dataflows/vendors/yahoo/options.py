"""The move the options market prices in, read off at-the-money straddles.

Buying the call and the put at the strike nearest the spot costs roughly the absolute
move the market expects until that expiry, so ``straddle / spot`` is a measured,
forward-looking size for an event that falls before the expiry. Yahoo's quotes are the
weak point: outside market hours bid and ask are zero and only the last trade remains,
which may be hours old and not simultaneous for the two legs. That is reported
(``price_basis``/``stale``) rather than hidden.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import yfinance as yf

from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.common import yf_retry

logger = logging.getLogger(__name__)

_NY = ZoneInfo("America/New_York")
_OPTIONS_CLOSE = time(16, 0)      # listed US options stop trading at 16:00 New York time

# Yahoo reports 1e-05 as the implied volatility of a contract it has no quote for.
_MIN_IV = 1e-3


@dataclass
class ImpliedMove:
    """One expiry's at-the-money straddle. ``move_pct`` is a fraction (0.05 = 5 %)."""

    symbol: str
    expiry: str
    days_to_expiry: int
    spot: float
    strike: float
    call_price: float
    put_price: float
    straddle: float
    move_pct: float
    price_basis: str  # "mid" | "last" | "mixed"
    stale: bool
    call_open_interest: int | None = None
    put_open_interest: int | None = None
    call_iv: float | None = None
    put_iv: float | None = None
    expires_today: bool = False   # dated today in New York, the session is still running

    def to_dict(self) -> dict:
        return asdict(self)


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _leg_price(row) -> tuple[float | None, str | None]:
    """A leg's price and its basis: the bid/ask mid if both are quoted, else the last trade."""
    bid, ask = _num(row.get("bid")), _num(row.get("ask"))
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0, "mid"
    last = _num(row.get("lastPrice"))
    if last is not None and last > 0:
        return last, "last"
    return None, None


def _iv(row) -> float | None:
    v = _num(row.get("impliedVolatility"))
    return v if v is not None and v > _MIN_IV else None


def _oi(row) -> int | None:
    v = _num(row.get("openInterest"))
    return int(v) if v is not None else None


def _spot(symbol: str) -> float | None:
    """Last price; each Yahoo call gets its own Ticker (one whose fast_info failed makes the
    next history() raise TypeError, which would read as an outage instead of an unknown symbol)."""
    try:
        price = _num(yf.Ticker(symbol).fast_info["lastPrice"])
    except Exception:
        price = None
    if price is not None and price > 0:
        return price
    hist = yf.Ticker(symbol).history(period="5d", auto_adjust=False)
    if hist is None or hist.empty or "Close" not in hist:
        return None
    closes = hist["Close"].dropna()
    return float(closes.iloc[-1]) if len(closes) else None


def load_implied_moves(symbol: str, max_expiries: int = 4,
                       now: datetime | None = None) -> list[ImpliedMove]:
    """Straddle-implied moves for the next ``max_expiries`` expiries dated today or later.

    An expiry dated today (New York) is dropped once its session has ended (16:00), and
    marked ``expires_today`` before that.  A symbol without listed options (most non-US
    tickers, Yahoo's indices) returns [], which is an answer; a symbol Yahoo does not know
    at all raises ``NoMarketDataError``.  An expiry where a leg has no usable price is
    skipped, never guessed. Yahoo being unreachable raises ``VendorUnavailableError``.
    ``now`` (UTC) makes the clock injectable for tests.
    """
    canonical = normalize_symbol(symbol)
    # a fresh Ticker per attempt: one whose fetch failed keeps a broken state (see yf_retry)
    expiries = yf_retry(lambda: yf.Ticker(canonical).options)
    if not expiries:
        if yf_retry(lambda: _spot(canonical)) is None:
            raise NoMarketDataError(symbol, canonical, "no price history")
        return []
    now_ny = (now or datetime.now(UTC)).astimezone(_NY)
    today = now_ny.date()
    closed = now_ny.time() >= _OPTIONS_CLOSE

    def alive(e: str) -> bool:
        d = date.fromisoformat(e)
        return d > today or (d == today and not closed)

    upcoming = [e for e in expiries if alive(e)][:max(0, max_expiries)]
    if not upcoming:
        return []
    spot = yf_retry(lambda: _spot(canonical))
    if spot is None or spot <= 0:
        raise VendorUnavailableError(f"Yahoo Finance returned no price for {canonical}")

    out: list[ImpliedMove] = []
    for expiry in upcoming:
        chain = yf_retry(lambda expiry=expiry: yf.Ticker(canonical).option_chain(expiry))
        if chain is None:
            continue
        calls = chain.calls.set_index("strike", drop=False)
        puts = chain.puts.set_index("strike", drop=False)
        common = sorted(set(calls.index) & set(puts.index))
        if not common:
            continue
        strike = min(common, key=lambda k: (abs(k - spot), k))
        crow, prow = calls.loc[strike], puts.loc[strike]
        if hasattr(crow, "ndim") and crow.ndim > 1:  # duplicate strike rows: take the first
            crow = crow.iloc[0]
        if hasattr(prow, "ndim") and prow.ndim > 1:
            prow = prow.iloc[0]
        cprice, cbasis = _leg_price(crow)
        pprice, pbasis = _leg_price(prow)
        if cprice is None or pprice is None:
            logger.info("skipping %s %s: no usable quote at strike %s", canonical, expiry, strike)
            continue
        basis = cbasis if cbasis == pbasis else "mixed"
        straddle = cprice + pprice
        out.append(ImpliedMove(
            symbol=canonical,
            expiry=expiry,
            days_to_expiry=(date.fromisoformat(expiry) - today).days,
            spot=spot,
            strike=float(strike),
            call_price=cprice,
            put_price=pprice,
            straddle=straddle,
            move_pct=straddle / spot,
            price_basis=basis,
            stale=basis != "mid",
            call_open_interest=_oi(crow),
            put_open_interest=_oi(prow),
            call_iv=_iv(crow),
            put_iv=_iv(prow),
            expires_today=expiry == today.isoformat(),
        ))
    return out
