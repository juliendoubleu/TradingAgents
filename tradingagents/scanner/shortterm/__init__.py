"""Short-horizon research: simulate day and swing trades on Yahoo bars, measure honestly.

Nothing here talks to a broker. The package holds a strict execution simulator
(``simulate``), the in-sample / out-of-sample evaluation and the pre-registered
verdict rule (``evaluate``), the candidate strategies (``daytrade``, ``swing``), the
study driver and report (``research``) and the translation of validated results
into labelled ideas for the daily list (``live``).

This module holds what several of them share: the daily-bar feature table every
daily signal reads, so the study and the live ideas run the SAME signal code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradingagents.scanner import indicators as ind

# Round-trip costs in basis points (see the report for why retail reality is higher).
COSTS_BPS = {"day": 5.0, "swing": 10.0}
COST_MULTIPLES = (1.0, 2.0, 4.0)
IS_FRACTION = 0.70


@dataclass(frozen=True)
class StrategySpec:
    """One candidate: which signal family it runs, with which parameters.

    Variants of a strategy share its ``family`` (hence its signal function) and
    differ only in ``params``, so the live ideas and the study can never diverge.
    """

    id: str
    family: str
    horizon: str                 # "day" | "swing"
    timeframe: str               # "1d" | "60m" | "5m"
    name_en: str
    name_de: str
    params: dict = field(default_factory=dict)
    note: str = ""               # for variants: the one-sentence motivation


def daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Everything the daily signals read, one row per trading day.

    Every column at row ``t`` uses bars up to ``t`` only, EXCEPT the ones that say
    otherwise by name: ``gap`` and ``gap_atr`` at ``t`` use the open of day ``t``
    (known when the session opens), and the ``*_prev`` columns are the previous
    day's values, i.e. what is known BEFORE day ``t``'s open. A signal that is
    "known at the close of t" reads the plain columns; a signal that is "known at
    the open of t" reads ``*_prev`` and the gap.
    """
    o, h, low, c, v = (df[k].astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    atr = ind.atr(h, low, c, 14)
    sma200 = ind.sma(c, 200)
    sma50 = ind.sma(c, 50)
    rng = (h - low).where(h > low)
    avg_vol = v.shift(1).rolling(20, min_periods=20).mean()
    ret5 = c / c.shift(5) - 1.0
    z5 = (ret5 - ret5.rolling(252, min_periods=252).mean()) / ret5.rolling(
        252, min_periods=252).std()
    prev_close = c.shift(1)
    out = pd.DataFrame({
        "open": o, "high": h, "low": low, "close": c, "volume": v,
        "atr": atr, "atr_pct": atr / c,
        "sma5": ind.sma(c, 5), "sma50": sma50, "sma200": sma200,
        "rsi2": ind.rsi(c, 2),
        "pos": (c - low) / rng,                      # 0 = closed at the low, 1 = at the high
        "chg_atr": (c - prev_close) / atr,           # close-to-close move in ATRs
        "vol_ratio": (v / avg_vol.where(avg_vol > 0)).replace([np.inf, -np.inf], np.nan),
        "hi20": c.rolling(20, min_periods=20).max(),
        "lo20": c.rolling(20, min_periods=20).min(),
        "z5": z5,
        "up": (c > sma200).fillna(False),
        "down": (c < sma200).fillna(False),
        "gap": o / prev_close - 1.0,
    })
    out["atr_prev"] = out["atr"].shift(1)
    out["atr_pct_prev"] = out["atr_pct"].shift(1)
    out["up_prev"] = out["up"].shift(1, fill_value=False).astype(bool)
    out["down_prev"] = out["down"].shift(1, fill_value=False).astype(bool)
    out["prev_close"] = prev_close
    out["gap_atr"] = out["gap"] / out["atr_pct_prev"]
    return out
