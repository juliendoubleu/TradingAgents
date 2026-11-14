"""Event study, performance snapshot and their Markdown renderings.

Three measured facts back an answer to "what is to be expected around event X":
how the share behaved around comparable past events (``event_study``), where the
affected instruments stand now (``performance_snapshot``) and, via the options vendor,
what move is priced in (``render_implied_moves`` formats it). Everything is plain
pandas on daily closes; the price loader is injectable so tests need no network.

Nothing is extrapolated: a window that needs bars before the history or after the last
bar is None, and an average only counts the events that have a value.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.scanner import indicators as ind, session
from tradingagents.scanner.fmt import fmt_date, fmt_pct, fmt_points, fmt_price

# Offsets in the instrument's own trading bars, relative to the event bar (index 0).
WINDOWS: dict[str, tuple[int, int]] = {
    "runup_60": (-61, -1), "runup_20": (-21, -1), "event_day": (-1, 0),
    "after_5": (-1, 5), "after_10": (-1, 10), "after_20": (-1, 20), "after_60": (-1, 60),
}
_RUNUPS = ("runup_60", "runup_20")
_WARMUP_DAYS = 130  # calendar days before the earliest event: 61 bars plus holidays

CAVEATS = {
    "en": [
        "A handful of events is an indication, not statistics.",
        "Other news inside the window can dominate the effect of the event itself.",
        "The company's size and business may have changed since the older events.",
    ],
    "de": [
        "Wenige Ereignisse sind ein Hinweis, keine Statistik.",
        "Andere Nachrichten im Zeitfenster können die Wirkung des Ereignisses überlagern.",
        "Größe und Geschäft des Unternehmens können sich seit den älteren Ereignissen "
        "verändert haben.",
    ],
}


@dataclass
class WindowReturn:
    """One window of one event. ``ret``/``bench_ret`` are fractions, ``excess_pts`` points."""

    start_date: str | None = None
    end_date: str | None = None
    ret: float | None = None
    bench_ret: float | None = None
    excess_pts: float | None = None


@dataclass
class EventResult:
    date: str
    label: str
    event_bar_date: str | None  # first trading bar on or after ``date``; None if upcoming/uncovered
    upcoming: bool
    windows: dict[str, WindowReturn] = field(default_factory=dict)
    covered: bool = True  # False: the event lies before the history that exists

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EventStudy:
    symbol: str
    benchmark: str | None
    as_of_date: str
    last_bar_date: str
    benchmark_available: bool
    events: list[EventResult]
    summary: dict[str, dict]  # window -> n, mean, median, mean_excess_pts, n_pos, n_neg
    caveats: dict[str, list[str]]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Snapshot:
    symbol: str
    last_close: float | None = None
    last_date: str | None = None
    ret_5: float | None = None
    ret_21: float | None = None
    ret_63: float | None = None
    ret_126: float | None = None
    ret_252: float | None = None
    dist_high_252: float | None = None  # <= 0
    dist_low_252: float | None = None  # >= 0
    atr14_pct: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- loading

def _default_loader(symbols, as_of_date, lookback_days):
    from tradingagents.dataflows.vendors.yahoo.bulk import load_bulk_ohlcv

    return load_bulk_ohlcv(symbols, as_of_date, lookback_days)


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _parse_date(text: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(datetime.strptime(str(text), "%Y-%m-%d"))
    except ValueError as exc:
        raise ValueError(f"bad date {text!r}: expected YYYY-MM-DD") from exc


def _as_of(as_of_date) -> pd.Timestamp:
    if as_of_date is None:
        return pd.Timestamp(date.today())
    return _parse_date(as_of_date)


def _norm(symbol: str) -> str:
    from tradingagents.dataflows.symbols import normalize_symbol

    return normalize_symbol(symbol)


# --------------------------------------------------------------------------- event study

def _close_on_or_before(close: pd.Series, ts: pd.Timestamp) -> float | None:
    pos = close.index.searchsorted(ts, side="right") - 1
    return float(close.iloc[pos]) if pos >= 0 else None


def _window(close: pd.Series, bench: pd.Series | None, i0: int, start: int, end: int) -> WindowReturn:
    a, b = i0 + start, i0 + end
    n = len(close)
    if a < 0 or b >= n or a >= b:
        return WindowReturn()
    d0, d1 = close.index[a], close.index[b]
    ret = float(close.iloc[b] / close.iloc[a] - 1.0)
    bench_ret = None
    if bench is not None and len(bench):
        c0, c1 = _close_on_or_before(bench, d0), _close_on_or_before(bench, d1)
        if c0 and c1:
            bench_ret = c1 / c0 - 1.0
    excess = None if bench_ret is None else (ret - bench_ret) * 100.0
    return WindowReturn(_iso(d0), _iso(d1), ret, bench_ret, excess)


def _summarise(events: list[EventResult]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    done = [e for e in events if not e.upcoming and e.covered]
    for name in WINDOWS:
        wins = [e.windows[name] for e in done if e.windows[name].ret is not None]
        rets = [w.ret for w in wins]
        excess = [w.excess_pts for w in wins if w.excess_pts is not None]
        out[name] = {
            "n": len(rets),
            "mean": statistics.fmean(rets) if rets else None,
            "median": statistics.median(rets) if rets else None,
            "mean_excess_pts": statistics.fmean(excess) if excess else None,
            "n_excess": len(excess),
            "n_pos": sum(1 for r in rets if r > 0),
            "n_neg": sum(1 for r in rets if r < 0),
        }
    return out


def event_study(symbol, events, benchmark="^GSPC", as_of_date=None, loader=None) -> EventStudy:
    """How ``symbol`` behaved around each dated event, against ``benchmark``.

    The event bar is the first trading bar on or after the date, so a weekend event maps
    to the next session. Windows are measured in the instrument's own bars (see WINDOWS);
    the benchmark is read between the same two calendar dates. A window lacking bars is
    None. An event after the last bar is ``upcoming`` and gets only the run-up so far,
    measured to the latest bar. Raises ``ValueError`` for a bad date and
    ``NoMarketDataError`` when the symbol has no data.
    """
    if not events:
        raise ValueError("no events given")
    parsed = [(_parse_date(d), label) for d, label in events]
    as_of = _as_of(as_of_date)
    loader = loader or _default_loader
    sym = _norm(symbol)
    bench_sym = _norm(benchmark) if benchmark else None

    earliest = min(d for d, _ in parsed)
    lookback = max((as_of - earliest).days, 0) + _WARMUP_DAYS
    wanted = [sym] + ([bench_sym] if bench_sym and bench_sym != sym else [])
    data = loader(wanted, as_of.strftime("%Y-%m-%d"), lookback)
    frame = data.get(sym)
    if frame is None or frame.empty:
        raise NoMarketDataError(symbol, sym, "no price history")
    close = frame["Close"].dropna()
    close = close[close.index <= as_of]
    if close.empty:
        raise NoMarketDataError(symbol, sym, "no price history")
    bench = None
    if bench_sym:
        bframe = data.get(bench_sym) if bench_sym != sym else frame
        if bframe is not None and not bframe.empty:
            bench = bframe["Close"].dropna()
            bench = bench[bench.index <= as_of]
            bench = bench if len(bench) else None

    n = len(close)
    first_bar, last_bar = close.index[0], close.index[-1]
    results: list[EventResult] = []
    for (edate, label), (raw_date, _) in zip(parsed, events, strict=True):
        windows = {name: WindowReturn() for name in WINDOWS}
        if edate > last_bar:
            # Upcoming: the virtual event bar sits just after the last bar, so the run-up
            # windows end on the last bar and have their usual length.
            for name in _RUNUPS:
                windows[name] = _window(close, bench, n, *WINDOWS[name])
            results.append(EventResult(raw_date, label, None, True, windows))
            continue
        i0 = int(close.index.searchsorted(edate, side="left"))
        # An event before the first bar we hold: the event bar would be wrong, so say so.
        if i0 == 0 and edate < first_bar - timedelta(days=4):
            results.append(EventResult(raw_date, label, None, False, windows, covered=False))
            continue
        for name, (s, e) in WINDOWS.items():
            windows[name] = _window(close, bench, i0, s, e)
        results.append(EventResult(raw_date, label, _iso(close.index[i0]), False, windows))

    return EventStudy(
        symbol=sym, benchmark=bench_sym, as_of_date=_iso(as_of), last_bar_date=_iso(last_bar),
        benchmark_available=bench is not None, events=results,
        summary=_summarise(results), caveats=CAVEATS,
    )


# --------------------------------------------------------------------------- snapshot

# bars per 1 week / 1 / 3 / 6 / 12 months and in a year: stocks trade 252 days, crypto 365
_BARS_STOCK = (5, 21, 63, 126, 252)
_BARS_CRYPTO = (7, 30, 91, 182, 365)


def calendar_bars(symbol: str) -> tuple[int, int, int, int, int]:
    """The bar counts of the 1-week/1/3/6/12-month windows of ``symbol`` (crypto trades every
    day of the year, so its windows are longer in bars)."""
    return _BARS_CRYPTO if _norm(symbol).endswith("-USD") else _BARS_STOCK


def performance_snapshot(symbols, as_of_date=None, loader=None, now=None) -> list[Snapshot]:
    """Where each instrument stands: last close, the 1-week/1/3/6/12-month returns, distance
    to the 12-month high/low, ATR14 as % of the close.

    The windows are counted in bars of the instrument's own calendar (stocks 5/21/63/126/252,
    crypto 7/30/91/182/365).  A bar whose session is not over yet at ``now`` (default: the
    current time, see ``session``) is dropped, like in the scan.  Anything needing more bars
    than exist is None; a symbol without data comes back with ``error`` set instead of being
    dropped.  The ``ret_5 ... ret_252`` fields hold the five windows in that order.
    """
    as_of = _as_of(as_of_date)
    loader = loader or _default_loader
    names = [_norm(s) for s in symbols]
    data = loader(names, as_of.strftime("%Y-%m-%d"), 440) if names else {}
    out: list[Snapshot] = []
    for sym in names:
        frame = data.get(sym)
        if frame is None or frame.empty or frame["Close"].dropna().empty:
            out.append(Snapshot(symbol=sym, error="no price data"))
            continue
        frame = frame[frame.index <= as_of].dropna(subset=["Close"])
        frame = session.settled({sym: frame}, now).get(sym) if not frame.empty else frame
        if frame is None or frame.empty:
            out.append(Snapshot(symbol=sym, error="no price data"))
            continue
        close = frame["Close"]
        last = float(close.iloc[-1])
        bars = calendar_bars(sym)

        def ret(k, close=close):
            return float(close.iloc[-1] / close.iloc[-1 - k] - 1.0) if len(close) > k else None

        dist_high = dist_low = None
        if len(close) >= bars[4]:
            tail = close.iloc[-bars[4]:]
            dist_high = last / float(tail.max()) - 1.0
            dist_low = last / float(tail.min()) - 1.0
        atr_pct = None
        if {"High", "Low"} <= set(frame.columns):
            atr = ind.atr(frame["High"], frame["Low"], close, 14).iloc[-1]
            if pd.notna(atr) and last:
                atr_pct = float(atr) / last
        out.append(Snapshot(
            symbol=sym, last_close=last, last_date=_iso(close.index[-1]),
            ret_5=ret(bars[0]), ret_21=ret(bars[1]), ret_63=ret(bars[2]), ret_126=ret(bars[3]),
            ret_252=ret(bars[4]),
            dist_high_252=dist_high, dist_low_252=dist_low, atr14_pct=atr_pct,
        ))
    return out


# --------------------------------------------------------------------------- rendering

_T = {
    "en": {
        "event": "Event", "day": "Day", "window": "Window", "mean": "Mean", "median": "Median",
        "excess": "vs index", "pos": "up/down", "title": "Event study", "returns": "Returns",
        "excess_h": "Excess vs benchmark", "summary": "Summary (past events)",
        "upcoming": "upcoming", "runup_sofar": "run-up so far", "caveats": "Caveats",
        "nobench": "No benchmark data: excess returns are not available.",
        "uncovered": "No price history that far back",
        "asof": "as of", "vs": "vs", "spot": "Spot",
        "w": {"runup_60": "-60d", "runup_20": "-20d", "event_day": "Day", "after_5": "+5d",
              "after_10": "+10d", "after_20": "+20d", "after_60": "+60d"},
        "snap": "Snapshot", "symbol": "Symbol", "close": "Close", "high": "to high",
        "low": "to low", "atr": "ATR14", "err": "no data",
        "snap_note": "Returns over about 1 week / 1 / 3 / 6 / 12 months (crypto: 7/30/91/182/365 "
                     "daily bars, stocks: 5/21/63/126/252); high/low = 12-month range.",
        "imp": "Implied move", "expiry": "Expiry", "days": "Days", "strike": "Strike",
        "straddle": "Straddle", "move": "Move", "basis": "Basis",
        "imp_note": ("The price of the at-the-money straddle (call + put) roughly equals the "
                     "absolute move the market prices in until expiry."),
        "stale_note": ("Some quotes are last-trade prices (no live bid/ask), possibly hours "
                       "old and not simultaneous."),
        "expires_today": "expires today",
        "none": "No listed options.", "mid": "mid", "last": "last", "mixed": "mixed",
    },
    "de": {
        "event": "Ereignis", "day": "Tag", "window": "Fenster", "mean": "Mittel",
        "median": "Median", "excess": "ggü. Index", "pos": "auf/ab", "title": "Ereignisstudie",
        "returns": "Renditen", "excess_h": "Überrendite ggü. Vergleichsindex",
        "summary": "Zusammenfassung (vergangene Ereignisse)", "upcoming": "bevorstehend",
        "runup_sofar": "Anlauf bisher", "caveats": "Einschränkungen",
        "nobench": "Keine Vergleichsdaten: Überrenditen sind nicht verfügbar.",
        "uncovered": "Keine Kurshistorie so weit zurück",
        "asof": "Stand", "vs": "ggü.", "spot": "Kurs",
        "w": {"runup_60": "-60T", "runup_20": "-20T", "event_day": "Tag", "after_5": "+5T",
              "after_10": "+10T", "after_20": "+20T", "after_60": "+60T"},
        "snap": "Momentaufnahme", "symbol": "Symbol", "close": "Schluss", "high": "zum Hoch",
        "low": "zum Tief", "atr": "ATR14", "err": "keine Daten",
        "snap_note": ("Renditen über etwa 1 Woche / 1 / 3 / 6 / 12 Monate (Krypto: 7/30/91/182/365 "
                      "Tageskurse, Aktien: 5/21/63/126/252); Hoch/Tief = 12-Monats-Spanne."),
        "imp": "Implizite Bewegung", "expiry": "Verfall", "days": "Tage", "strike": "Strike",
        "straddle": "Straddle", "move": "Bewegung", "basis": "Basis",
        "imp_note": ("Der Preis des At-the-money-Straddles (Call + Put) entspricht etwa der "
                     "absoluten Bewegung, die der Markt bis zum Verfall einpreist."),
        "stale_note": ("Einige Kurse sind letzte Handelspreise (kein aktuelles Bid/Ask), "
                       "möglicherweise Stunden alt und nicht zeitgleich."),
        "expires_today": "läuft heute aus",
        "none": "Keine gelisteten Optionen.", "mid": "Mitte", "last": "letzter", "mixed": "gemischt",
    },
}


def _tr(lang: str) -> dict:
    return _T["de" if lang == "de" else "en"]


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render_event_study(study: EventStudy, lang: str = "en") -> str:
    """Compact Markdown: returns per event, excess per event, summary, caveats."""
    t = _tr(lang)
    cols = ["runup_60", "runup_20", "event_day", "after_5", "after_10", "after_20", "after_60"]
    head = [t["event"]] + [t["w"][c] for c in cols]
    done = [e for e in study.events if not e.upcoming]
    up = [e for e in study.events if e.upcoming]
    parts = [f"### {t['title']}: {study.symbol}",
             f"{t['asof']} {fmt_date(study.last_bar_date, lang)}"
             + (f", {t['vs']} {study.benchmark}" if study.benchmark and study.benchmark_available else "")]

    def row(e, fn):
        if not e.covered:
            return [e.label] + [t["uncovered"]] + ["-"] * (len(cols) - 1)
        return [e.label] + [fn(e.windows[c]) for c in cols]

    if done:
        parts += [f"**{t['returns']}**",
                  _table(head, [row(e, lambda w: fmt_pct(w.ret, 1, True, lang)) for e in done])]
        if study.benchmark_available:
            parts += [f"**{t['excess_h']}**",
                      _table(head, [row(e, lambda w: fmt_points(w.excess_pts, 1, lang).replace(
                          " Punkte", "").replace(" pts", "")) for e in done]),
                      "(" + ("Prozentpunkte" if lang == "de" else "percentage points") + ")"]
        else:
            parts.append(t["nobench"])
    for e in up:
        r60, r20 = e.windows["runup_60"], e.windows["runup_20"]
        parts.append(
            f"**{e.label}** ({fmt_date(e.date, lang)}, {t['upcoming']}): {t['runup_sofar']} "
            f"{t['w']['runup_60']} {fmt_pct(r60.ret, 1, True, lang)}, "
            f"{t['w']['runup_20']} {fmt_pct(r20.ret, 1, True, lang)}")
    srows = []
    for name in WINDOWS:
        s = study.summary[name]
        srows.append([t["w"][name], str(s["n"]), fmt_pct(s["mean"], 1, True, lang),
                      fmt_pct(s["median"], 1, True, lang),
                      fmt_points(s["mean_excess_pts"], 1, lang),
                      f"{s['n_pos']}/{s['n_neg']}"])
    parts += [f"**{t['summary']}**",
              _table([t["window"], "n", t["mean"], t["median"], t["excess"], t["pos"]], srows),
              f"**{t['caveats']}**"]
    bullets = "\n".join(f"- {c}" for c in study.caveats["de" if lang == "de" else "en"])
    return "\n\n".join(parts) + "\n\n" + bullets + "\n"


def render_snapshot(snapshots: list[Snapshot], lang: str = "en") -> str:
    """Compact Markdown table of the snapshots, then a one-line note."""
    t = _tr(lang)
    unit = "T" if lang == "de" else "d"
    head = [t["symbol"], t["close"], *(f"{n}{unit}" for n in (5, 21, 63, 126, 252)),
            t["high"], t["low"], t["atr"]]
    rows = []
    for s in snapshots:
        if s.error:
            rows.append([s.symbol, t["err"]] + ["-"] * 8)
            continue
        rows.append([
            s.symbol, fmt_price(s.last_close, lang),
            *(fmt_pct(v, 1, True, lang) for v in (s.ret_5, s.ret_21, s.ret_63, s.ret_126, s.ret_252)),
            fmt_pct(s.dist_high_252, 1, True, lang), fmt_pct(s.dist_low_252, 1, True, lang),
            fmt_pct(s.atr14_pct, 1, False, lang),
        ])
    dates = sorted({s.last_date for s in snapshots if s.last_date})
    stamp = f"{t['asof']} " + ", ".join(fmt_date(d, lang) for d in dates) if dates else ""
    return "\n\n".join(x for x in [f"### {t['snap']}", stamp, _table(head, rows), t["snap_note"]] if x) + "\n"


def render_implied_moves(moves, lang: str = "en") -> str:
    """Markdown table of straddle-implied moves, with what the number means and a stale note."""
    t = _tr(lang)
    if not moves:
        return f"### {t['imp']}\n\n{t['none']}\n"
    head = [t["expiry"], t["days"], t["strike"], t["straddle"], t["move"], t["basis"]]
    rows = [[fmt_date(m.expiry, lang),
             t["expires_today"] if m.expires_today else str(m.days_to_expiry),
             fmt_price(m.strike, lang),
             fmt_price(m.straddle, lang), "±" + fmt_pct(m.move_pct, 1, False, lang),
             t[m.price_basis]] for m in moves]
    parts = [f"### {t['imp']}: {moves[0].symbol}",
             f"{t['spot']} {fmt_price(moves[0].spot, lang)}", _table(head, rows), t["imp_note"]]
    if any(m.stale for m in moves):
        parts.append(t["stale_note"])
    return "\n\n".join(parts) + "\n"
