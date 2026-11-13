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
from datetime import UTC, date, datetime

import pandas as pd

from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.bulk import load_bulk_ohlcv
from tradingagents.scanner.notices import Notice
from tradingagents.scanner.regime import MarketRegime, breadth_above_sma200, compute_regime
from tradingagents.scanner.session import drop_unsettled
from tradingagents.scanner.signals import Signal, SignalParams, SkipSignal, compute_signal
from tradingagents.scanner.universe import universe_info

logger = logging.getLogger(__name__)

VIX = "^VIX"
LOOKBACK_DAYS = 600          # ~410 trading days: 220 for scoring plus the 252-day range
STALE_DAYS = 10              # an instrument whose last bar lags the newest by more is skipped
WARN_OLDER_THAN_DAYS = 5     # the newest bar this far before the requested date is flagged
MAX_MISSING_SHARE = 0.20     # more members than this without data: not a market scan

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
    skipped: list[tuple[str, Notice]] = field(default_factory=list)   # Notice is a str too
    warnings: list[Notice] = field(default_factory=list)
    running_bars_removed: int = 0         # members whose still-running daily bar was cut
    top: int = 10
    generated: str = ""
    custom: bool = False                  # a --tickers list replaced the universe's members

    def to_dict(self) -> dict:
        return {
            "universe": self.universe, "description": self.description,
            "benchmark": self.benchmark, "requested_date": self.requested_date,
            "as_of_date": self.as_of_date, "generated": self.generated, "custom": self.custom,
            "regime": self.regime.to_dict(), "top": self.top,
            "longs": [s.to_dict() for s in self.longs],
            "shorts": [s.to_dict() for s in self.shorts],
            "signals": [s.to_dict() for s in self.signals],
            "skipped": [{"symbol": s, "reason": str(r), "reason_de": r.text("de"),
                         "code": r.code} for s, r in self.skipped],
            "warnings": [str(w) for w in self.warnings],
            "warnings_de": [w.text("de") for w in self.warnings],
            "warning_items": [{"code": w.code, "params": w.params} for w in self.warnings],
            "running_bars_removed": self.running_bars_removed,
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
    params: SignalParams | None = None,
    use_vix: bool = True,
) -> tuple[MarketRegime, list[Signal], list[tuple[str, Notice]]]:
    """Regime and a signal per member from ``frames``, seeing only bars up to ``at``.

    ``at=None`` uses every bar given (the live scan). Members with no data, too little
    history, or a last bar far behind the newest one are returned as ``(symbol, reason)``
    skips instead of being scored on stale prices. ``params`` carries the universe's
    calendar (crypto vs stocks); ``use_vix=False`` leaves the VIX out of the regime.
    """
    params = params or SignalParams()
    cut = {s: _cut(frames.get(s), at) for s in members}
    cut = {s: f for s, f in cut.items() if f is not None and not f["Close"].dropna().empty}
    skipped: list[tuple[str, Notice]] = [
        (s, Notice("no_data")) for s in members if s not in cut
    ]

    newest = max((f["Close"].dropna().index[-1] for f in cut.values()), default=None)
    live: dict[str, pd.DataFrame] = {}
    for symbol, frame in cut.items():
        last = frame["Close"].dropna().index[-1]
        if (newest - last).days > STALE_DAYS:
            skipped.append((symbol, Notice("stale", last=str(last.date()))))
        else:
            live[symbol] = frame

    bench_frame = _cut(frames.get(benchmark), at)
    bench_close = None if bench_frame is None else bench_frame["Close"].dropna()
    vix_frame = _cut(frames.get(VIX), at) if use_vix else None
    vix_close = None if vix_frame is None else vix_frame["Close"].dropna()
    breadth = breadth_above_sma200({s: f["Close"].dropna() for s, f in live.items()})
    regime = compute_regime(benchmark, bench_close, breadth, vix_close, use_vix=use_vix)

    signals: list[Signal] = []
    for symbol, frame in live.items():
        try:
            signals.append(compute_signal(
                symbol, frame, bench_close, regime.label, benchmark, params))
        except SkipSignal as exc:
            skipped.append((symbol, exc.notice))
    return regime, signals, skipped


def run_scan(
    universe: str,
    as_of_date: str,
    tickers: list[str] | None = None,
    top: int = 10,
    loader: Loader | None = None,
    now: datetime | None = None,
) -> ScanResult:
    """Scan ``universe`` as of ``as_of_date`` and rank the Long and Short ideas.

    ``loader(symbols, as_of_date, lookback_days)`` supplies the bars (Yahoo when None,
    injectable so tests run offline). ``now`` (UTC, default the current time) decides
    which daily bars are finished: a bar whose exchange session is still running is
    removed from every series, the VIX and the benchmark included, so close, ATR and
    stop never come from a live price. The result states the date of the newest bar
    that was really used. Unless ``tickers`` replaces the members, a universe with more
    than ``MAX_MISSING_SHARE`` of its members without data is refused: breadth and the
    ranking of a partial market would mislead.
    """
    if top < 1:
        raise ValueError("top must be at least 1")
    parse_date(as_of_date)
    now = now or datetime.now(UTC)
    info = universe_info(universe, tickers)
    description, benchmark = info.description["en"], info.benchmark
    params = SignalParams(info.month_bars, info.bars_per_year)
    members = list(dict.fromkeys(normalize_symbol(t) for t in info.members))
    wanted = list(dict.fromkeys([*members, benchmark, *([VIX] if info.use_vix else [])]))
    frames = (loader or load_bulk_ohlcv)(wanted, as_of_date, LOOKBACK_DAYS)
    frames, trimmed = drop_unsettled(frames, now)
    running = len([s for s in trimmed if s in members])

    no_data = [s for s in members if s not in frames]
    if not tickers and len(no_data) / len(members) > MAX_MISSING_SHARE:
        raise VendorUnavailableError(
            f"No price data for {len(no_data)} of {len(members)} {universe.strip().lower()} "
            f"symbols ({', '.join(no_data[:8])}{', ...' if len(no_data) > 8 else ''}); "
            "Yahoo is probably throttling, and a partial market is not a market scan. "
            "Try again in a few minutes."
        )

    regime, signals, skipped = analyse(members, frames, benchmark, params=params,
                                       use_vix=info.use_vix)
    warnings: list[Notice] = []
    if benchmark not in frames:
        warnings.append(Notice("benchmark_missing", bench=benchmark))
    if info.use_vix and VIX not in frames:
        warnings.append(Notice("vix_missing"))
    for item in regime.missing:
        if item != "VIX":
            warnings.append(Notice("regime_input_missing", item=item))
    if skipped:
        warnings.append(Notice("skipped_summary", n=len(skipped), total=len(members)))
    if running:
        warnings.append(Notice("running_bar_removed", n=running, total=len(members)))

    as_of = max((s.last_date for s in signals), default="")
    if as_of:
        lag = (pd.Timestamp(as_of_date) - pd.Timestamp(as_of)).days
        if lag > WARN_OLDER_THAN_DAYS:
            warnings.append(Notice("stale_data", asof=as_of, lag=lag, requested=as_of_date))
    if as_of_date < now.strftime("%Y-%m-%d"):
        warnings.append(Notice("past_date", date=as_of_date))

    longs = sorted((s for s in signals if s.direction == "Long"),
                   key=lambda s: (-s.score, s.symbol))[:top]
    shorts = sorted((s for s in signals if s.direction == "Short"),
                    key=lambda s: (s.score, s.symbol))[:top]
    return ScanResult(
        universe=universe.strip().lower(), description=description, benchmark=benchmark,
        requested_date=as_of_date, as_of_date=as_of, regime=regime,
        signals=sorted(signals, key=lambda s: -s.score), longs=longs, shorts=shorts,
        skipped=sorted(skipped), warnings=warnings, top=top,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"), custom=bool(tickers),
        running_bars_removed=running,
    )
