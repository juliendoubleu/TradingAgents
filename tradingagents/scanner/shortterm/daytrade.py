"""Day-trade candidates (daily, 60-minute and 5-minute bars) and the shared runners.

All rules were written down BEFORE any result was seen (see the plan the research
follows) and use only information available before the entry. Three kinds:

* daily-bar day trades D1-D5: enter at a day's open, leave at its close, stop
  ``stop_atr`` x ATR14 from the entry. D3-D5 read the previous close only (known
  before the open); D1/D2 read the open itself, which on daily bars is optimistic
  (you cannot know the opening print before it exists), so M2 re-tests both with a
  realistic 09:35 entry on 5-minute bars.
* 60-minute day trades H1-H3 and 5-minute day trades M1-M3.

The daily signal functions take the table from ``daily_features`` and return +1/-1/0
per row; ``run_daily`` turns them into simulated trades and is also what ``swing``
uses, and ``live`` calls the same signal functions on today's last bar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradingagents.scanner.shortterm import COSTS_BPS, StrategySpec
from tradingagents.scanner.shortterm.simulate import Level, next_bar_entry, simulate_trade

# --------------------------------------------------------------------------- daily signals


def _sides(sig: pd.Series, p: dict) -> pd.Series:
    """Restrict to one side when a variant asks for it (long-only / short-only)."""
    side = p.get("sides", "both")
    if side == "long":
        return sig.where(sig > 0, 0)
    if side == "short":
        return sig.where(sig < 0, 0)
    return sig


def _dir(long_mask: pd.Series, short_mask: pd.Series) -> pd.Series:
    return (long_mask.astype(int) - short_mask.astype(int)).astype(int)


def sig_d1(f: pd.DataFrame, p: dict) -> pd.Series:
    """Gap fade: a moderate gap AGAINST the daily trend is expected to close."""
    a = f["gap_atr"].abs()
    mid = (a >= p.get("lo", 0.3)) & (a < p.get("hi", 1.5))
    return _sides(_dir(mid & (f["gap"] < 0) & f["up_prev"], mid & (f["gap"] > 0) & f["down_prev"]), p)


def sig_d2(f: pd.DataFrame, p: dict) -> pd.Series:
    """Gap continuation: a gap of at least 1.5 ATR% is followed in its direction."""
    big = f["gap_atr"].abs() >= p.get("min", 1.5)
    return _sides(_dir(big & (f["gap"] > 0), big & (f["gap"] < 0)), p)


def sig_d3(f: pd.DataFrame, p: dict) -> pd.Series:
    """Washout reversal: a >= 2 ATR drop closing in the bottom quarter bounces next day."""
    k, pos = p.get("chg", 2.0), p.get("pos", 0.25)
    return _sides(_dir((f["chg_atr"] <= -k) & (f["pos"] <= pos),
                       (f["chg_atr"] >= k) & (f["pos"] >= 1 - pos)), p)


def sig_d4(f: pd.DataFrame, p: dict) -> pd.Series:
    """Trend-day follow-through: close at the extreme on heavy volume continues next day."""
    pos, vol = p.get("pos", 0.10), p.get("vol", 1.5)
    heavy = f["vol_ratio"] >= vol
    return _sides(_dir((f["pos"] >= 1 - pos) & heavy & (f["close"] > f["sma50"]),
                       (f["pos"] <= pos) & heavy & (f["close"] < f["sma50"])), p)


def sig_d5(f: pd.DataFrame, p: dict) -> pd.Series:
    """RSI(2) extreme with the long-term trend: oversold in an uptrend, overbought in a downtrend."""
    lo, hi = p.get("lo", 5), p.get("hi", 95)
    return _sides(_dir((f["rsi2"] < lo) & f["up"], (f["rsi2"] > hi) & f["down"]), p)


DAILY_SIGNALS: dict[str, Callable[[pd.DataFrame, dict], pd.Series]] = {
    "D1": sig_d1, "D2": sig_d2, "D3": sig_d3, "D4": sig_d4, "D5": sig_d5,
}
# Signals that read the open of the entry day itself fill at that open; every
# other daily signal is known at a close and fills at the NEXT bar's open.
SAME_OPEN_FAMILIES = {"D1", "D2"}


def _post_filters(sig: pd.Series, f: pd.DataFrame, p: dict, same_open: bool) -> pd.Series:
    """Optional variant filters, applied on top of a family's signal.

    For same-open families only ``*_prev`` columns may be used (the entry day's own
    volume or range does not exist yet at its open).
    """
    atr_pct = f["atr_pct_prev"] if same_open else f["atr_pct"]
    if "atr_pct_min" in p:
        sig = sig.where(atr_pct >= p["atr_pct_min"], 0)
    if "atr_pct_max" in p:
        sig = sig.where(atr_pct <= p["atr_pct_max"], 0)
    if "min_vol_ratio" in p and not same_open:
        sig = sig.where(f["vol_ratio"] >= p["min_vol_ratio"], 0)
    return sig


def _atr_distance_stop(direction: int, dist: float) -> Callable[[float], float]:
    """Stop ``dist`` price units against the position, priced at the actual fill."""
    return lambda entry: entry - direction * dist


def run_daily(spec: StrategySpec, feats: dict[str, pd.DataFrame],
              signal_fn: Callable[[pd.DataFrame, dict], pd.Series], *,
              first_day: pd.Timestamp, cost_bps: float | None = None,
              exit_flags: Callable[[pd.DataFrame, int], np.ndarray | None] | None = None,
              ) -> list[dict]:
    """Simulate a daily-bar strategy on every symbol; returns one dict per trade.

    ``signal_fn(features, params)`` must already include the variant filters
    (``swing.daily_signal``), the very function ``live`` calls on today's bar.

    No pyramiding: a symbol already in a trade ignores new signals until it is out.
    A trade whose last allowed bar lies beyond the data is dropped, not guessed.
    """
    p = spec.params
    same_open = spec.family in SAME_OPEN_FAMILIES
    hold = int(p.get("hold", 1))
    stop_atr = p.get("stop_atr")
    cost = COSTS_BPS[spec.horizon] if cost_bps is None else cost_bps
    out: list[dict] = []
    for symbol, f in feats.items():
        sig = signal_fn(f, p).to_numpy()
        o, h, lo, c = (f[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        atr_ref = (f["atr_prev"] if same_open else f["atr"]).to_numpy(dtype=float)
        dates = f.index
        n = len(f)
        busy_until = -1
        for i in np.flatnonzero(sig):
            entry_idx = i if same_open else next_bar_entry(i, n)
            if entry_idx is None or entry_idx <= busy_until or dates[entry_idx] < first_day:
                continue
            last_idx = entry_idx + hold - 1
            if last_idx >= n or not np.isfinite(atr_ref[i]):
                continue
            direction = int(sig[i])
            stop: Level = None
            if stop_atr:
                stop = _atr_distance_stop(direction, float(stop_atr) * atr_ref[i])
            flags = exit_flags(f, direction) if exit_flags else None
            res = simulate_trade(o, h, lo, c, entry_idx=entry_idx, direction=direction,
                                 last_idx=last_idx, cost_bps=cost, stop=stop,
                                 exit_on_close=flags)
            busy_until = res.exit_idx
            out.append({
                "symbol": symbol, "date": dates[entry_idx].normalize(),
                "entry_ts": dates[entry_idx], "exit_ts": dates[res.exit_idx],
                "direction": direction, "raw": res.exit_price / res.entry_price - 1.0,
                "window": c[res.exit_idx] / res.entry_price - 1.0, "reason": res.exit_reason,
            })
    return out


# --------------------------------------------------------------------------- intraday setups


@dataclass(frozen=True)
class Setup:
    """One intraday trade decided on a session: fill bar, side, protective levels, last bar."""

    entry_idx: int
    direction: int
    last_idx: int
    stop: Level = None
    target: Level = None


def _allowed(direction: int, p: dict) -> bool:
    side = p.get("sides", "both")
    return side == "both" or (side == "long") == (direction > 0)


def _atr_stop(ctx: dict, mult: float, direction: int) -> Level:
    return _atr_distance_stop(direction, mult * ctx["atr"])


def setup_h1(o, h, low, c, v, mins, ctx, p) -> Setup | None:
    """Late-day momentum: follow the day's drift into the last hour (15:30 bar).

    Direction is the sign of the move from the previous close to the CLOSE of the
    14:30 bar (the last print known when the 15:30 bar opens); fill at that open.
    """
    k = len(o) - 1
    d = int(np.sign(c[k - 1] / ctx["prev_close"] - 1.0))
    return Setup(k, d, k) if d and _allowed(d, p) else None


def setup_h1b(o, h, low, c, v, mins, ctx, p) -> Setup | None:
    """First-hour momentum: previous close -> close of the first bar, hold to the close."""
    d = int(np.sign(c[0] / ctx["prev_close"] - 1.0))
    return Setup(1, d, len(o) - 1) if d and _allowed(d, p) else None


def setup_h2(o, h, low, c, v, mins, ctx, p) -> Setup | None:
    """First-hour breakout: a later 60m close beyond the first hour's range, stop at its midpoint."""
    hi, lo = h[0], low[0]
    mid = (hi + lo) / 2.0
    for k in range(1, len(o) - 1):
        d = 1 if c[k] > hi else -1 if c[k] < lo else 0
        if d:
            if not _allowed(d, p):
                return None
            return Setup(k + 1, d, len(o) - 1, stop=mid)
    return None


def setup_h3(o, h, low, c, v, mins, ctx, p) -> Setup | None:
    """Morning fade: a first-hour move of >= 1 ATR% is faded from 10:30, stop at its extreme."""
    ret = c[0] / o[0] - 1.0
    thr = p.get("mult", 1.0) * ctx["atr_pct"]
    d = 1 if ret <= -thr else -1 if ret >= thr else 0
    if not d or not _allowed(d, p):
        return None
    stop = low[0] if d > 0 else h[0]
    return Setup(1, d, len(o) - 1, stop=stop)


def setup_m1(o, h, low, c, v, mins, ctx, p) -> Setup | None:
    """Opening-range breakout (5m): close beyond the first 30 minutes' range, 10:00-12:00."""
    first = np.flatnonzero(mins < 600)
    hi, lo = h[first].max(), low[first].min()
    width, mid = hi - lo, (hi + lo) / 2.0
    for k in np.flatnonzero((mins >= 600) & (mins + 5 <= 720)):
        d = 1 if c[k] > hi else -1 if c[k] < lo else 0
        if d:
            if not _allowed(d, p) or width <= 0:
                return None
            return Setup(k + 1, d, len(o) - 1, stop=mid,
                         target=lambda entry, d=d, width=width: entry + d * width)
    return None


def _gap_setup(o, c, ctx, p, *, family: str) -> Setup | None:
    """D1/D2 on 5m bars: the gap is read at the first bar's open, entry at the 2nd bar's open."""
    gap = o[0] / ctx["prev_close"] - 1.0
    g = gap / ctx["atr_pct"]
    a = abs(g)
    if family == "D1":
        if not (p.get("lo", 0.3) <= a < p.get("hi", 1.5)):
            return None
        d = 1 if (gap < 0 and ctx["up"]) else -1 if (gap > 0 and ctx["down"]) else 0
    else:
        d = (1 if gap > 0 else -1) if a >= p.get("min", 1.5) else 0
    if not d or not _allowed(d, p):
        return None
    return Setup(1, d, len(o) - 1, stop=_atr_stop(ctx, p.get("stop_atr", 1.0), d))


def setup_m2a(o, h, low, c, v, mins, ctx, p) -> Setup | None:
    return _gap_setup(o, c, ctx, p, family="D1")


def setup_m2b(o, h, low, c, v, mins, ctx, p) -> Setup | None:
    return _gap_setup(o, c, ctx, p, family="D2")


def setup_m3(o, h, low, c, v, mins, ctx, p) -> Setup | None:
    """VWAP reversion (5m): 2 sigma away from session VWAP, with the daily trend, back to VWAP."""
    tp = (h + low + c) / 3.0
    cum_v = np.cumsum(v)
    vwap = np.cumsum(tp * v) / np.where(cum_v > 0, cum_v, np.nan)
    for k in np.flatnonzero((mins >= 660) & (mins < 840)):
        if cum_v[k] <= 0:
            continue
        diff = c[: k + 1] - vwap[: k + 1]
        sigma = float(np.nanstd(diff, ddof=1))
        if not sigma > 0:
            continue
        z = diff[k] / sigma
        d = 1 if (z < -2.0 and ctx["up"]) else -1 if (z > 2.0 and ctx["down"]) else 0
        if d:
            if not _allowed(d, p):
                return None
            return Setup(k + 1, d, len(o) - 1, stop=_atr_distance_stop(d, sigma),
                         target=float(vwap[k]))
    return None


INTRADAY_SETUPS: dict[str, Callable] = {
    "H1": setup_h1, "H1b": setup_h1b, "H2": setup_h2, "H3": setup_h3,
    "M1": setup_m1, "M2a": setup_m2a, "M2b": setup_m2b, "M3": setup_m3,
}
# a "full" session: the bar minutes (since midnight, exchange clock) a regular day has
_FULL_MINUTES = {
    "60m": np.arange(570, 570 + 7 * 60, 60),
    "5m": np.arange(570, 570 + 78 * 5, 5),
}


def minutes_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.hour * 60 + index.minute).to_numpy()


def full_session(mins: np.ndarray, interval: str) -> bool:
    """True for a complete regular session: no half day, no missing bar."""
    full = _FULL_MINUTES[interval]
    return len(mins) == len(full) and bool((mins == full).all())


def day_context(f: pd.DataFrame, day: pd.Timestamp) -> dict | None:
    """What the daily bars say before ``day`` opens (the previous trading day's values)."""
    pos = f.index.searchsorted(day) - 1
    if pos < 0:
        return None
    row = f.iloc[pos]
    if not (np.isfinite(row["atr"]) and np.isfinite(row["close"]) and row["close"] > 0):
        return None
    return {"prev_close": float(row["close"]), "atr": float(row["atr"]),
            "atr_pct": float(row["atr_pct"]), "up": bool(row["up"]), "down": bool(row["down"])}


def session_days(df: pd.DataFrame, interval: str):
    """Yield ``(date, slice)`` for each complete session of an intraday frame."""
    dates = pd.DatetimeIndex(df.index.tz_localize(None)).normalize()
    day_ints = dates.asi8
    mins_all = minutes_of_day(df.index)
    bounds = np.flatnonzero(np.diff(day_ints)) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(df)]
    for s, e in zip(starts, ends, strict=True):
        if full_session(mins_all[s:e], interval):
            yield dates[s], slice(s, e)


def run_intraday(spec: StrategySpec, intraday: dict[str, pd.DataFrame],
                 feats: dict[str, pd.DataFrame], *, cost_bps: float | None = None) -> list[dict]:
    """Simulate an intraday strategy on every complete session of every symbol."""
    setup_fn = INTRADAY_SETUPS[spec.family]
    cost = COSTS_BPS[spec.horizon] if cost_bps is None else cost_bps
    interval = spec.timeframe
    out: list[dict] = []
    for symbol, df in intraday.items():
        f = feats.get(symbol)
        if f is None:
            continue
        o, h, lo, c, v = (df[k].to_numpy(dtype=float)
                          for k in ("Open", "High", "Low", "Close", "Volume"))
        mins_all = minutes_of_day(df.index)
        for day, sl in session_days(df, interval):
            ctx = day_context(f, day)
            if ctx is None:
                continue
            args = (o[sl], h[sl], lo[sl], c[sl], v[sl], mins_all[sl])
            setup = setup_fn(*args, ctx, spec.params)
            if setup is None:
                continue
            res = simulate_trade(*args[:4], entry_idx=setup.entry_idx, direction=setup.direction,
                                 last_idx=setup.last_idx, cost_bps=cost, stop=setup.stop,
                                 target=setup.target)
            ts = df.index[sl]
            out.append({
                "symbol": symbol, "date": day, "entry_ts": ts[res.entry_idx],
                "exit_ts": ts[res.exit_idx], "direction": setup.direction,
                "raw": res.exit_price / res.entry_price - 1.0,
                "window": args[3][res.exit_idx] / res.entry_price - 1.0,
                "reason": res.exit_reason,
            })
    return out


# --------------------------------------------------------------------------- registry

_D = {"stop_atr": 1.0, "hold": 1}
DAY_SPECS: list[StrategySpec] = [
    StrategySpec("D1", "D1", "day", "1d", "Gap fade (daily bars)",
                 "Gap-Fade (Tagesbalken)", {"lo": 0.3, "hi": 1.5, **_D}),
    StrategySpec("D2", "D2", "day", "1d", "Gap continuation (daily bars)",
                 "Gap-Fortsetzung (Tagesbalken)", {"min": 1.5, **_D}),
    StrategySpec("D3", "D3", "day", "1d", "Washout reversal",
                 "Ausverkaufs-Umkehr", {"chg": 2.0, "pos": 0.25, **_D}),
    StrategySpec("D4", "D4", "day", "1d", "Trend-day follow-through",
                 "Trendtag-Fortsetzung", {"pos": 0.10, "vol": 1.5, **_D}),
    StrategySpec("D5", "D5", "day", "1d", "RSI(2) extreme, day trade",
                 "RSI(2)-Extrem, Daytrade", {"lo": 5, "hi": 95, **_D}),
    StrategySpec("H1", "H1", "day", "60m", "Late-day momentum (15:30 hour)",
                 "Spätes Momentum (Schlussstunde)", {}),
    StrategySpec("H1b", "H1b", "day", "60m", "First-hour momentum, held to the close",
                 "Momentum der ersten Stunde bis Handelsschluss", {}),
    StrategySpec("H2", "H2", "day", "60m", "First-hour breakout",
                 "Ausbruch aus der ersten Handelsstunde", {}),
    StrategySpec("H3", "H3", "day", "60m", "Morning fade",
                 "Morgen-Fade", {"mult": 1.0}),
    StrategySpec("M1", "M1", "day", "5m", "Opening-range breakout (5m)",
                 "Opening-Range-Ausbruch (5 Min.)", {}),
    StrategySpec("M2a", "M2a", "day", "5m", "Gap fade, entry 09:35 (5m)",
                 "Gap-Fade, Einstieg 09:35 (5 Min.)", {"lo": 0.3, "hi": 1.5, "stop_atr": 1.0}),
    StrategySpec("M2b", "M2b", "day", "5m", "Gap continuation, entry 09:35 (5m)",
                 "Gap-Fortsetzung, Einstieg 09:35 (5 Min.)", {"min": 1.5, "stop_atr": 1.0}),
    StrategySpec("M3", "M3", "day", "5m", "VWAP reversion (5m)",
                 "VWAP-Rückkehr (5 Min.)", {}),
]
