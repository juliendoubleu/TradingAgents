"""Walk-forward check of the scanner's own signals: did its calls pay off afterwards?

At each rebalance date (every ``horizon`` trading days, so windows do not overlap) the
regime and the signals are computed by ``scan.analyse`` on bars up to that date and no
later, then the close-to-close return over the next ``horizon`` bars is measured for
every instrument and the benchmark. Per direction it reports how many calls were made,
the average forward return, the average excess over the benchmark and the hit rate.

This is a sanity check, not proof of an edge: see ``CAVEATS``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.bulk import load_bulk_ohlcv
from tradingagents.scanner.scan import VIX, analyse, parse_date
from tradingagents.scanner.session import settled
from tradingagents.scanner.signals import SignalParams
from tradingagents.scanner.universe import universe_info

WARMUP_MONTHS = 14
MIN_DATES_FOR_TSTAT = 3

CAVEATS = {
    "en": [
        "Survivorship bias: today's universe members are applied to the past, so names "
        "that fell out of the list or were delisted are missing, which flatters Longs.",
        "No transaction costs, slippage, taxes or short borrow fees are included.",
        "Past results do not guarantee future results.",
        "Small sample: few non-overlapping windows, and calls on one date are correlated "
        "with each other, so the effective number of independent observations is lower "
        "than the call counts suggest.",
    ],
    "de": [
        "Survivorship-Bias: Die heutigen Mitglieder des Universums werden auf die Vergangenheit "
        "angewendet; ausgeschiedene oder delistete Werte fehlen, was Longs begünstigt.",
        "Keine Transaktionskosten, Slippage, Steuern oder Leihgebühren für Shorts enthalten.",
        "Vergangene Ergebnisse garantieren keine zukünftigen Ergebnisse.",
        "Kleine Stichprobe: wenige überschneidungsfreie Zeitfenster, und Signale desselben "
        "Tages sind untereinander korreliert; die Zahl unabhängiger Beobachtungen ist daher "
        "kleiner, als die Anzahl der Signale vermuten lässt.",
    ],
}


@dataclass
class DirectionStats:
    direction: str
    calls: int = 0
    avg_return: float | None = None          # the instrument's own forward return
    avg_excess: float | None = None          # instrument minus benchmark, same window
    hit_rate: float | None = None            # Long: ret > 0; Short: ret < 0; Neutral: none
    avg_pnl: float | None = None             # P&L of the position: ret (Long), -ret (Short)
    avg_pnl_excess: float | None = None      # position's P&L vs the benchmark (Short: -(ret - bench))

    def to_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in self.__dict__.items()}


@dataclass
class ValidationResult:
    universe: str
    benchmark: str
    as_of_date: str
    months: int
    horizon: int
    first_rebalance: str
    last_rebalance: str
    rebalances: int
    instruments: int
    stats: dict[str, DirectionStats]
    spread: float | None                     # Long minus Short return, per date, then averaged
    spread_tstat: float | None               # of the per-date spreads, when enough dates
    spread_dates: int                        # dates that had both Long and Short calls
    universe_avg_return: float | None        # all calls, equal weight: the no-skill baseline
    benchmark_avg_return: float | None       # benchmark over the same windows
    caveats: list[str] = field(default_factory=lambda: list(CAVEATS["en"]))
    spread_pooled: float | None = None       # pooled Long average minus pooled Short average

    def to_dict(self) -> dict:
        return {
            "universe": self.universe, "benchmark": self.benchmark, "as_of_date": self.as_of_date,
            "months": self.months, "horizon": self.horizon,
            "first_rebalance": self.first_rebalance, "last_rebalance": self.last_rebalance,
            "rebalances": self.rebalances, "instruments": self.instruments,
            "stats": {k: v.to_dict() for k, v in self.stats.items()},
            "spread": self.spread, "spread_pooled": self.spread_pooled, "spread_tstat": self.spread_tstat,
            "spread_dates": self.spread_dates,
            "universe_avg_return": self.universe_avg_return,
            "benchmark_avg_return": self.benchmark_avg_return,
            "caveats": self.caveats, "caveats_de": CAVEATS["de"],
        }


Loader = Callable[[list[str], str, int], dict[str, pd.DataFrame]]


def rebalance_positions(n_bars: int, horizon: int, window_bars: int) -> list[int]:
    """Calendar positions to rebalance at, every ``horizon`` bars, ending ``horizon`` bars
    before the last bar so the last window has an outcome. At most ``window_bars`` back."""
    last = n_bars - 1 - horizon
    if last < 0:
        return []
    positions = list(range(last, max(last - window_bars, -1), -horizon))
    return sorted(positions)


def _price_at(close: pd.Series, when: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None]:
    """Last close on or before ``when`` and the date of that bar."""
    upto = close.loc[:when]
    if upto.empty:
        return None, None
    return float(upto.iloc[-1]), upto.index[-1]


def validate(
    universe: str,
    as_of_date: str,
    tickers: list[str] | None = None,
    months: int = 24,
    horizon: int | None = None,
    loader: Loader | None = None,
    now: datetime | None = None,
) -> ValidationResult:
    """Walk-forward test of the signals over the last ``months`` months.

    ``horizon`` defaults to the universe's month in bars (21 stocks, 30 crypto). Bars
    whose session is not finished at ``now`` are removed first, exactly as in the scan.
    """
    info = universe_info(universe, tickers)
    horizon = info.month_bars if horizon is None else horizon
    if months < 1 or horizon < 1:
        raise ValueError("months and horizon must be at least 1")
    parse_date(as_of_date)
    benchmark = info.benchmark
    members = list(dict.fromkeys(normalize_symbol(t) for t in info.members))
    wanted = list(dict.fromkeys([*members, benchmark, *([VIX] if info.use_vix else [])]))
    lookback = int((months + WARMUP_MONTHS) * 30.5)
    frames = (loader or load_bulk_ohlcv)(wanted, as_of_date, lookback)
    frames = settled(frames, now or datetime.now(UTC))
    return validate_frames(universe, as_of_date, members, benchmark, frames, months, horizon,
                           params=SignalParams(info.month_bars, info.bars_per_year),
                           use_vix=info.use_vix)


def validate_frames(
    universe: str,
    as_of_date: str,
    members: list[str],
    benchmark: str,
    frames: dict[str, pd.DataFrame],
    months: int,
    horizon: int,
    params: SignalParams | None = None,
    use_vix: bool = True,
) -> ValidationResult:
    """The walk-forward over bars already downloaded (split out so tests can feed their own)."""
    as_of = pd.Timestamp(as_of_date).normalize()
    # Calendar of rebalance and outcome dates: the benchmark's trading days, else the union.
    if benchmark in frames:
        calendar = frames[benchmark]["Close"].dropna().index
    else:
        calendar = pd.DatetimeIndex(sorted(set().union(
            *(f["Close"].dropna().index for s, f in frames.items() if s in members))))
    calendar = calendar[calendar <= as_of]
    window_start = as_of - pd.DateOffset(months=months)
    in_window = int((calendar >= window_start).sum())
    # ``in_window`` counts the last ``horizon`` bars too, which cannot be rebalance dates
    # (they have no outcome yet), so the grid may reach back before the window: drop those.
    positions = [p for p in rebalance_positions(len(calendar), horizon, in_window)
                 if calendar[p] >= window_start]

    rows = []                       # (direction, ret, bench_ret) per call
    per_date: list[tuple[list[float], list[float]]] = []   # (long rets, short rets) per date
    bench_window_returns: list[float] = []
    seen: set[str] = set()
    dates_used: list[pd.Timestamp] = []

    for pos in positions:
        at, end = calendar[pos], calendar[pos + horizon]
        # Everything below sees bars up to `at` only; `end` is used solely to read outcomes.
        regime, signals, _skipped = analyse(members, frames, benchmark, at=at, params=params,
                                           use_vix=use_vix)
        bench_ret = None
        if benchmark in frames:
            b0, _ = _price_at(frames[benchmark]["Close"].dropna(), at)
            b1, _ = _price_at(frames[benchmark]["Close"].dropna(), end)
            if b0 and b1:
                bench_ret = b1 / b0 - 1
        longs, shorts = [], []
        used = 0
        for sig in signals:
            close = frames[sig.symbol]["Close"].dropna()
            p0, d0 = _price_at(close, at)
            p1, d1 = _price_at(close, end)
            # An outcome needs a later bar for the same instrument: a name whose data
            # ends here (delisted, gap) has no forward return and is not counted.
            if p0 is None or p1 is None or d1 <= d0 or p0 <= 0:
                continue
            ret = p1 / p0 - 1
            rows.append((sig.direction, ret, bench_ret))
            seen.add(sig.symbol)
            used += 1
            if sig.direction == "Long":
                longs.append(ret)
            elif sig.direction == "Short":
                shorts.append(ret)
        if used:
            dates_used.append(at)
            per_date.append((longs, shorts))
            if bench_ret is not None:
                bench_window_returns.append(bench_ret)

    stats = {d: _stats(d, [r for r in rows if r[0] == d]) for d in ("Long", "Short", "Neutral")}
    all_rets = [r[1] for r in rows]
    date_spreads = [float(np.mean(lg) - np.mean(sh)) for lg, sh in per_date if lg and sh]
    # Averaged per date, like its t-statistic, so a date with many calls does not outweigh the rest.
    spread = float(np.mean(date_spreads)) if date_spreads else None
    # Pooled: every call weighs the same, so it can differ in sign from the per-date mean.
    pooled = (stats["Long"].avg_return - stats["Short"].avg_return
              if stats["Long"].avg_return is not None and stats["Short"].avg_return is not None
              else None)
    tstat = None
    if len(date_spreads) >= MIN_DATES_FOR_TSTAT:
        sd = float(np.std(date_spreads, ddof=1))
        if sd > 0:
            tstat = float(np.mean(date_spreads) / (sd / math.sqrt(len(date_spreads))))
    return ValidationResult(
        universe=universe, benchmark=benchmark, as_of_date=as_of_date, months=months,
        horizon=horizon,
        first_rebalance=dates_used[0].strftime("%Y-%m-%d") if dates_used else "",
        last_rebalance=dates_used[-1].strftime("%Y-%m-%d") if dates_used else "",
        rebalances=len(dates_used), instruments=len(seen), stats=stats,
        spread=spread, spread_pooled=pooled, spread_tstat=tstat, spread_dates=len(date_spreads),
        universe_avg_return=float(np.mean(all_rets)) if all_rets else None,
        benchmark_avg_return=(float(np.mean(bench_window_returns))
                              if bench_window_returns else None),
    )


def _stats(direction: str, rows: list[tuple[str, float, float | None]]) -> DirectionStats:
    out = DirectionStats(direction=direction, calls=len(rows))
    if not rows:
        return out
    rets = np.array([r[1] for r in rows])
    out.avg_return = float(rets.mean())
    excess = np.array([r[1] - r[2] for r in rows if r[2] is not None])
    out.avg_excess = float(excess.mean()) if len(excess) else None
    if direction == "Long":
        out.hit_rate = float((rets > 0).mean())
        out.avg_pnl = out.avg_return
        out.avg_pnl_excess = out.avg_excess
    elif direction == "Short":
        out.hit_rate = float((rets < 0).mean())
        out.avg_pnl = -out.avg_return
        out.avg_pnl_excess = -out.avg_excess if out.avg_excess is not None else None
    return out
