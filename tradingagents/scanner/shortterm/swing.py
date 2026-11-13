"""Swing-trade candidates on daily bars: signal at a close, fill at the NEXT open.

Holding periods of a few days (5-10 trading days) with a stop of 2-2.5 ATR. The
signal functions return +1/-1/0 per row of ``daily_features`` and read that row and
earlier ones only; ``daytrade.run_daily`` simulates them and ``live`` calls the same
functions on today's last bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.scanner.shortterm import StrategySpec
from tradingagents.scanner.shortterm.daytrade import (
    DAILY_SIGNALS,
    SAME_OPEN_FAMILIES,
    _dir,
    _post_filters,
    _sides,
)


def sig_s1(f: pd.DataFrame, p: dict) -> pd.Series:
    """RSI(2) pullback inside the long-term trend (the classic short-term mean reversion)."""
    return _sides(_dir((f["rsi2"] < p.get("lo", 10)) & f["up"],
                       (f["rsi2"] > p.get("hi", 90)) & f["down"]), p)


def sig_s2(f: pd.DataFrame, p: dict) -> pd.Series:
    """5-day reversal: an extreme 5-day move (|z| > 2 vs its own year) against the trend."""
    z = p.get("z", 2.0)
    return _sides(_dir((f["z5"] < -z) & f["up"], (f["z5"] > z) & f["down"]), p)


def sig_s3(f: pd.DataFrame, p: dict) -> pd.Series:
    """Breakout: a 20-day closing extreme on heavy volume, with the 50-day trend."""
    heavy = f["vol_ratio"] >= p.get("vol", 1.5)
    return _sides(_dir((f["close"] >= f["hi20"]) & heavy & (f["close"] > f["sma50"]),
                       (f["close"] <= f["lo20"]) & heavy & (f["close"] < f["sma50"])), p)


def sig_s4(f: pd.DataFrame, p: dict) -> pd.Series:
    """Post-gap drift: a big, high-volume gap that the day then confirmed (close vs open)."""
    g, vol = p.get("gap", 1.5), p.get("vol", 2.0)
    heavy = f["vol_ratio"] >= vol
    return _sides(_dir((f["gap_atr"] >= g) & heavy & (f["close"] >= f["open"]),
                       (f["gap_atr"] <= -g) & heavy & (f["close"] <= f["open"])), p)


SWING_SIGNALS = {"S1": sig_s1, "S2": sig_s2, "S3": sig_s3, "S4": sig_s4}


def daily_signal(family: str, f: pd.DataFrame, params: dict) -> pd.Series:
    """The +1/-1/0 signal of any daily-bar family including its variant filters.

    The single entry point used by the study AND by ``live``: there is no second
    implementation of a rule that could drift from the one that was measured.
    """
    sig = ALL_DAILY_SIGNALS[family](f, params)
    return _post_filters(sig, f, params, same_open=family in SAME_OPEN_FAMILIES)


def exit_flags_s1(f: pd.DataFrame, direction: int) -> np.ndarray:
    """S1 leaves at the first close beyond the 5-day average (above for longs, below for shorts)."""
    c, m = f["close"].to_numpy(dtype=float), f["sma5"].to_numpy(dtype=float)
    return (c > m) if direction > 0 else (c < m)


EXIT_FLAGS = {"S1": exit_flags_s1}
# every daily-bar signal function, by strategy family
ALL_DAILY_SIGNALS = {**DAILY_SIGNALS, **SWING_SIGNALS}

SWING_SPECS: list[StrategySpec] = [
    StrategySpec("S1", "S1", "swing", "1d", "RSI(2) pullback", "RSI(2)-Rücksetzer",
                 {"lo": 10, "hi": 90, "hold": 5, "stop_atr": 2.5}),
    StrategySpec("S2", "S2", "swing", "1d", "5-day reversal", "5-Tage-Umkehr",
                 {"z": 2.0, "hold": 5, "stop_atr": 2.5}),
    StrategySpec("S3", "S3", "swing", "1d", "Volume breakout", "Ausbruch mit Volumen",
                 {"vol": 1.5, "hold": 10, "stop_atr": 2.0}),
    StrategySpec("S4", "S4", "swing", "1d", "Post-gap drift", "Drift nach Kurslücke",
                 {"gap": 1.5, "vol": 2.0, "hold": 5, "stop_atr": 2.0}),
]
