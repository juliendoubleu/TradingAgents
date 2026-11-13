"""Execution simulator: the strict, deterministic heart of the short-horizon study.

The rules are deliberately pessimistic, because a simulator that is kind to itself
manufactures edges:

* A signal computed from a bar's CLOSE can only be filled at the OPEN of the NEXT bar
  (``next_bar_entry``); never at the same bar's close, never at a price that was not
  known yet. No next bar in the session means no trade.
* Bar by bar after entry (the entry bar included) a stop is checked before a target:
  when both lie inside one bar we cannot know the order, so we assume the worse one.
* A bar that OPENS beyond the stop (a gap) fills at that open, not at the stop price,
  because that is what a stop order does in a gap. Targets are the mirror: a gap
  through a target fills at the better open.
* Otherwise the position is closed at the close of the last allowed bar (time exit).
* ``net_return = direction * (exit / entry - 1) - cost_bps / 10_000``.

Pure functions on numpy arrays: no pandas, no I/O, so every rule can be tested on a
handful of hand-written bars.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

Level = float | Callable[[float], float] | None


@dataclass(frozen=True)
class TradeResult:
    """One simulated trade; indices refer to the bar arrays that were passed in."""

    direction: int
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    exit_reason: str          # "stop" | "target" | "signal" | "time"
    gross_return: float
    net_return: float


def next_bar_entry(signal_idx: int, n_bars: int) -> int | None:
    """Index of the bar a close-based signal can be filled on (its open), if any."""
    nxt = signal_idx + 1
    return nxt if nxt < n_bars else None


def _level(value: Level, entry: float) -> float | None:
    """A stop/target given as a price, or as a function of the entry price."""
    if value is None:
        return None
    return float(value(entry)) if callable(value) else float(value)


def simulate_trade(
    open_: Sequence[float],
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    *,
    entry_idx: int,
    direction: int,
    last_idx: int,
    cost_bps: float,
    stop: Level = None,
    target: Level = None,
    exit_on_close: Sequence[bool] | None = None,
) -> TradeResult:
    """Fill at ``open_[entry_idx]`` and walk the bars until an exit rule fires.

    A stop or target that the fill itself already crosses (a breakout bar followed by an
    open back through the stop) is not an error: the gap rule fills it at that open,
    so the decision to trade never depends on the entry bar's price.

    ``stop``/``target`` are prices, or callables of the entry price (for rules like
    "1 ATR from the entry", which cannot be priced before the fill). ``exit_on_close``
    flags bars at whose CLOSE the position is closed when neither stop nor target
    was hit (e.g. "first close above the 5-day average").
    """
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    o, h, lo, c = (np.asarray(x, dtype=float) for x in (open_, high, low, close))
    n = len(o)
    if not 0 <= entry_idx <= last_idx < n:
        raise ValueError(f"need 0 <= entry_idx <= last_idx < {n}, got {entry_idx}, {last_idx}")
    entry = float(o[entry_idx])
    stop_px, target_px = _level(stop, entry), _level(target, entry)

    exit_idx, exit_px, reason = last_idx, float(c[last_idx]), "time"
    for i in range(entry_idx, last_idx + 1):
        if direction == 1:
            if stop_px is not None and lo[i] <= stop_px:
                exit_idx, exit_px, reason = i, min(stop_px, float(o[i])), "stop"
                break
            if target_px is not None and h[i] >= target_px:
                exit_idx, exit_px, reason = i, max(target_px, float(o[i])), "target"
                break
        else:
            if stop_px is not None and h[i] >= stop_px:
                exit_idx, exit_px, reason = i, max(stop_px, float(o[i])), "stop"
                break
            if target_px is not None and lo[i] <= target_px:
                exit_idx, exit_px, reason = i, min(target_px, float(o[i])), "target"
                break
        if exit_on_close is not None and bool(exit_on_close[i]):
            exit_idx, exit_px, reason = i, float(c[i]), "signal"
            break

    gross = direction * (exit_px / entry - 1.0)
    return TradeResult(
        direction=direction, entry_idx=entry_idx, exit_idx=exit_idx, entry_price=entry,
        exit_price=exit_px, exit_reason=reason, gross_return=gross,
        net_return=gross - cost_bps / 10_000.0,
    )
