"""Technical indicators on daily bars: plain pandas/numpy, no vendor and no state.

Every function takes a series (or the OHLCV columns it needs) and returns a series
aligned to the input, with NaN where there is not yet enough history. Nothing here
looks forward: the value at a bar uses that bar and earlier ones only, which is what
lets the walk-forward validation reuse the live scan's code unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(close: pd.Series, n: int) -> pd.Series:
    """Simple moving average; NaN until ``n`` bars exist."""
    return close.rolling(n, min_periods=n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    """Exponential moving average (span ``n``), seeded with the first value."""
    return close.ewm(span=n, adjust=False).mean()


def _wilder(values: np.ndarray, n: int, first: int) -> np.ndarray:
    """Wilder's smoothing: the seed is the plain mean of the ``n`` values starting at
    ``first``, then ``avg = (avg * (n - 1) + x) / n``. NaN before the seed bar."""
    out = np.full(len(values), np.nan)
    seed_at = first + n - 1
    if len(values) <= seed_at:
        return out
    avg = float(np.mean(values[first:first + n]))
    out[seed_at] = avg
    for i in range(seed_at + 1, len(values)):
        avg = (avg * (n - 1) + values[i]) / n
        out[i] = avg
    return out


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder smoothing, 0..100.

    No losses over the window reads 100 (or 50 when there were no gains either: a
    flat series has no strength to report), so the division by zero never leaks out.
    """
    delta = close.diff().to_numpy(dtype=float)
    gains = np.where(np.isnan(delta), 0.0, np.maximum(delta, 0.0))
    losses = np.where(np.isnan(delta), 0.0, np.maximum(-delta, 0.0))
    # delta[0] is NaN, so the first usable change is bar 1
    avg_gain = _wilder(gains, n, first=1)
    avg_loss = _wilder(losses, n, first=1)
    out = np.full(len(close), np.nan)
    valid = ~np.isnan(avg_gain)
    flat = valid & (avg_loss == 0)
    out[flat] = np.where(avg_gain[flat] > 0, 100.0, 50.0)
    regular = valid & (avg_loss > 0)
    rs = avg_gain[regular] / avg_loss[regular]
    out[regular] = 100.0 - 100.0 / (1.0 + rs)
    return pd.Series(out, index=close.index)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's true range; the first bar has no previous close, so it is high - low."""
    prev = close.shift(1)
    parts = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1)
    return parts.max(axis=1, skipna=True)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average true range with Wilder smoothing, in price units."""
    tr = true_range(high, low, close).to_numpy(dtype=float)
    return pd.Series(_wilder(tr, n, first=0), index=close.index)


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram: (EMA fast - EMA slow) minus its own ``signal``-span EMA."""
    line = ema(close, fast) - ema(close, slow)
    return line - ema(line, signal)


def pct_return(close: pd.Series, n: int) -> pd.Series:
    """Return over the last ``n`` bars, as a fraction (0.05 = +5 %)."""
    return close / close.shift(n) - 1.0


def realized_vol(close: pd.Series, n: int = 63) -> pd.Series:
    """Annualised volatility of daily log returns over ``n`` bars (sqrt(252) scaling)."""
    logret = np.log(close / close.shift(1))
    return logret.rolling(n, min_periods=n).std() * np.sqrt(TRADING_DAYS)


def dist_from_high(close: pd.Series, n: int = TRADING_DAYS) -> pd.Series:
    """Close relative to the highest close of the last ``n`` bars: 0 at the high, else negative."""
    return close / close.rolling(n, min_periods=min(n, 60)).max() - 1.0


def dist_from_low(close: pd.Series, n: int = TRADING_DAYS) -> pd.Series:
    """Close relative to the lowest close of the last ``n`` bars: 0 at the low, else positive."""
    return close / close.rolling(n, min_periods=min(n, 60)).min() - 1.0


def volume_ratio(volume: pd.Series, short: int = 20, long: int = 60) -> pd.Series:
    """Average volume of the last ``short`` bars over that of the last ``long`` bars.

    NaN when there is no volume to speak of (indices and some FX report none or zero),
    so callers can tell "no volume data" from "volume is normal".
    """
    long_avg = volume.rolling(long, min_periods=long).mean()
    short_avg = volume.rolling(short, min_periods=short).mean()
    ratio = short_avg / long_avg.where(long_avg > 0)
    return ratio.replace([np.inf, -np.inf], np.nan)
