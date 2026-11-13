"""Scan a universe: regime, a signal per instrument, and the ranked Long / Short ideas.

``analyse`` is the single place where bars become a regime and signals. The live scan
calls it on everything downloaded; the walk-forward validation calls it once per
rebalance date with ``at`` set, which cuts every series at that date first. Sharing it
is what makes the validation a test of the scan's own logic rather than of a copy.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.bulk import load_bulk_ohlcv
from tradingagents.scanner.regime import MarketRegime, breadth_above_sma200, compute_regime
from tradingagents.scanner.signals import Signal, SkipSignal, compute_signal
from tradingagents.scanner.universe import resolve_universe

logger = logging.getLogger(__name__)

VIX = "^VIX"
LOOKBACK_DAYS = 600          # ~410 trading days: 220 for scoring plus the 252-day range
STALE_DAYS = 10              # an instrument whose last bar lags the newest by more is skipped
WARN_OLDER_THAN_DAYS = 5     # the newest bar this far before the requested date is flagged

Loader = Callable[[list[str], str, int], dict[str, pd.DataFrame]]


@dataclass
class ScanResult:
    universe: str
    description: str
    benchmark: str
    requested_date: str
    as_of_date: str                       # date of the newest bar actually used
    regime: MarketRegime
    signals: list[Signal]
    longs: list[Signal]
    shorts: list[Signal]
    skipped: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    top: int = 10
    generated: str = ""

    def to_dict(self) -> dict:
        return {
            "universe": self.universe, "description": self.description,
            "benchmark": self.benchmark, "requested_date": self.requested_date,
            "as_of_date": self.as_of_date, "generated": self.generated,
            "regime": self.regime.to_dict(), "top": self.top,
            "longs": [s.to_dict() for s in self.longs],
            "shorts": [s.to_dict() for s in self.shorts],
            "signals": [s.to_dict() for s in self.signals],
            "skipped": [{"symbol": s, "reason": r} for s, r in self.skipped],
            "warnings": list(self.warnings),
        }


def parse_date(value: str | None, today: date | None = None) -> str:
    """``YYYY-MM-DD`` for a scan date: today when empty; a bad or future date is a ValueError."""
    today = today or date.today()
    if not value:
        return today.strftime("%Y-%m-%d")
    try:
        parsed = datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"the date must be in YYYY-MM-DD format, got {value!r}") from None
    if parsed.strftime("%Y-%m-%d") != str(value):
        raise ValueError(f"the date must be in YYYY-MM-DD format, got {value!r}")
    if parsed > today:
        raise ValueError(f"{value} is in the future; the scan needs prices that already exist")
    return str(value)


def _cut(frame: pd.DataFrame | None, at: pd.Timestamp | None) -> pd.DataFrame | None:
    if frame is None:
        return None
    return frame if at is None else frame.loc[:at]


def analyse(
    members: list[str],
    frames: dict[str, pd.DataFrame],
    benchmark: str,
    at: pd.Timestamp | None = None,
) -> tuple[MarketRegime, list[Signal], list[tuple[str, str]]]:
    """Regime and a signal per member from ``frames``, seeing only bars up to ``at``.

    ``at=None`` uses every bar given (the live scan). Members with no data, too little
    history, or a last bar far behind the newest one are returned as ``(symbol, reason)``
    skips instead of being scored on stale prices.
    """
    cut = {s: _cut(frames.get(s), at) for s in members}
    cut = {s: f for s, f in cut.items() if f is not None and not f["Close"].dropna().empty}
    skipped: list[tuple[str, str]] = [
        (s, "no price data from Yahoo") for s in members if s not in cut
    ]

    newest = max((f["Close"].dropna().index[-1] for f in cut.values()), default=None)
    live: dict[str, pd.DataFrame] = {}
    for symbol, frame in cut.items():
        last = frame["Close"].dropna().index[-1]
        if (newest - last).days > STALE_DAYS:
            skipped.append((symbol, f"last bar {last.date()} is stale"))
        else:
            live[symbol] = frame

    bench_frame = _cut(frames.get(benchmark), at)
    bench_close = None if bench_frame is None else bench_frame["Close"].dropna()
    vix_frame = _cut(frames.get(VIX), at)
    vix_close = None if vix_frame is None else vix_frame["Close"].dropna()
    breadth = breadth_above_sma200({s: f["Close"].dropna() for s, f in live.items()})
    regime = compute_regime(benchmark, bench_close, breadth, vix_close)

    signals: list[Signal] = []
    for symbol, frame in live.items():
        try:
            signals.append(compute_signal(symbol, frame, bench_close, regime.label, benchmark))
        except SkipSignal as exc:
            skipped.append((symbol, str(exc)))
    return regime, signals, skipped


def run_scan(
    universe: str,
    as_of_date: str,
    tickers: list[str] | None = None,
    top: int = 10,
    loader: Loader = load_bulk_ohlcv,
) -> ScanResult:
    """Scan ``universe`` as of ``as_of_date`` and rank the Long and Short ideas.

    ``loader(symbols, as_of_date, lookback_days)`` supplies the bars (Yahoo by default,
    injectable so tests run offline). The result states the date of the newest bar that
    was really used, which is earlier than the requested date on weekends, holidays
    and before the close.
    """
    if top < 1:
        raise ValueError("top must be at least 1")
    parse_date(as_of_date)
    description, benchmark, members = resolve_universe(universe, tickers)
    members = list(dict.fromkeys(normalize_symbol(t) for t in members))
    wanted = list(dict.fromkeys([*members, benchmark, VIX]))
    frames = loader(wanted, as_of_date, LOOKBACK_DAYS)

    regime, signals, skipped = analyse(members, frames, benchmark)
    warnings: list[str] = []
    if benchmark not in frames:
        warnings.append(f"Benchmark {benchmark} had no data: relative strength is 0 for all "
                        "and the regime has no benchmark facts.")
    if VIX not in frames:
        warnings.append("VIX had no data: it adds no points to the regime.")
    for item in regime.missing:
        if item not in ("VIX",):
            warnings.append(f"Regime input not available: {item}.")
    if skipped:
        warnings.append(f"{len(skipped)} of {len(members)} instruments were skipped (see list).")

    as_of = max((s.last_date for s in signals), default="")
    if as_of:
        lag = (pd.Timestamp(as_of_date) - pd.Timestamp(as_of)).days
        if lag > WARN_OLDER_THAN_DAYS:
            warnings.append(f"The newest price bar is {as_of}, {lag} days before the requested "
                            f"{as_of_date}.")
        if as_of == date.today().strftime("%Y-%m-%d"):
            warnings.append("The newest bar is today's session; it may be incomplete if the "
                            "market is still open.")

    longs = sorted((s for s in signals if s.direction == "Long"),
                   key=lambda s: (-s.score, s.symbol))[:top]
    shorts = sorted((s for s in signals if s.direction == "Short"),
                    key=lambda s: (s.score, s.symbol))[:top]
    return ScanResult(
        universe=universe.strip().lower(), description=description, benchmark=benchmark,
        requested_date=as_of_date, as_of_date=as_of, regime=regime,
        signals=sorted(signals, key=lambda s: -s.score), longs=longs, shorts=shorts,
        skipped=sorted(skipped), warnings=warnings, top=top,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

