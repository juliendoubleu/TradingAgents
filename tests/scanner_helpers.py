"""Synthetic price series for the scanner tests: no network, fully deterministic."""

import numpy as np
import pandas as pd

END = "2025-06-30"


def frame(closes, volume=1_000_000.0, spread=0.01, end=END) -> pd.DataFrame:
    """Daily OHLCV on business days ending ``end``; high/low are close +- spread."""
    closes = np.asarray(closes, dtype=float)
    index = pd.bdate_range(end=end, periods=len(closes))
    return pd.DataFrame({
        "Open": closes, "High": closes * (1 + spread), "Low": closes * (1 - spread),
        "Close": closes, "Volume": float(volume),
    }, index=index)


def rising(n=400, daily=0.002, start=100.0, wiggle=0.0, **kw) -> pd.DataFrame:
    i = np.arange(n)
    return frame(start * np.exp(np.log1p(daily) * i + wiggle * np.sin(i * 0.9)), **kw)


def falling(n=400, daily=0.002, start=100.0, wiggle=0.0, **kw) -> pd.DataFrame:
    return rising(n, -daily, start, wiggle, **kw)


def flat(n=400, start=100.0, **kw) -> pd.DataFrame:
    return frame(np.full(n, start), **kw)


def noisy(n=400, start=100.0, amplitude=0.02, **kw) -> pd.DataFrame:
    i = np.arange(n)
    return frame(start * (1 + amplitude * np.sin(i * 0.9)), **kw)
