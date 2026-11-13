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
from datetime import date

import yfinance as yf

from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.common import yf_retry

logger = logging.getLogger(__name__)

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


def _spot(ticker) -> float | None:
    try:
        price = _num(ticker.fast_info["lastPrice"])
    except Exception:
        price = None
    if price is not None and price > 0:
        return price
    hist = ticker.history(period="5d", auto_adjust=False)
    if hist is None or hist.empty or "Close" not in hist:
        return None
    closes = hist["Close"].dropna()
    return float(closes.iloc[-1]) if len(closes) else None


def load_implied_moves(symbol: str, max_expiries: int = 4) -> list[ImpliedMove]:
    """Straddle-implied moves for the next ``max_expiries`` expiries dated today or later.

    A symbol without listed options (most non-US tickers, Yahoo's indices) returns [],
    which is an answer. An expiry where a leg has no usable price is skipped, never
    guessed. Yahoo being unreachable raises ``VendorUnavailableError``.
    """
    canonical = normalize_symbol(symbol)
    ticker = yf.Ticker(canonical)
    expiries = yf_retry(lambda: ticker.options)
    if not expiries:
        return []
    today = date.today()
    upcoming = [e for e in expiries if date.fromisoformat(e) >= today][:max(0, max_expiries)]
    if not upcoming:
        return []
    spot = yf_retry(lambda: _spot(ticker))
    if spot is None or spot <= 0:
        raise VendorUnavailableError(f"Yahoo Finance returned no price for {canonical}")

    out: list[ImpliedMove] = []
    for expiry in upcoming:
        chain = yf_retry(lambda expiry=expiry: ticker.option_chain(expiry))
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
        ))
    return out
