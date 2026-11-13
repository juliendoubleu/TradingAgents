"""Medium- and long-horizon rules, shared by the study and the live scan.

A rule answers one question: "given everything known at the close of day t, which
instruments are Long (+1), Short (-1) or nothing (0), and how strongly?".  The study
(``horizon_research``) evaluates these rules over twelve years of history and the idea
engine (``horizons``) asks the SAME functions for the sides at the latest settled bar.
There is no second implementation that could drift from the one that was measured.

How look-ahead is excluded: every feature of a symbol is computed on that symbol's own
bars (a rolling window or a shift in its own bar count), so row ``t`` uses bars up to
``t`` only; only afterwards are the columns aligned to the common date index.  A symbol
with no bar at ``t`` has NaN there and therefore takes no part in the cross-sectional
ranks of day ``t``.  Fills are not decided here: a signal of day ``t`` is filled at the
open of ``t + 1`` by the study's simulator.

The pre-registered rules (fixed before any result was seen) are T0 to T5; ``V1`` ...
are variants added after the in-sample results, each with the sentence that motivated it.
Scores are "bullishness": high = Long candidate, low = Short candidate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradingagents.scanner import indicators as ind
from tradingagents.scanner.signals import SignalParams

MIN_RANK_NAMES = 5          # a quantile of fewer names than this is not a ranking
MIN_ROW_SHARE = 0.5         # a date counts only when this share of the live symbols has a bar
COMPOSITE_WINDOW = 410      # bars the live scan sees (LOOKBACK_DAYS = 600 calendar days)
ATR_N = 14

MEDIUM, LONG = "medium", "long"


@dataclass(frozen=True)
class RuleSpec:
    """One rule (or variant) with the numbers that define it and its bilingual texts."""

    id: str
    base: str                          # T0 .. T5: which function computes it
    name_en: str
    name_de: str
    desc_en: str
    desc_de: str
    horizons: tuple[str, ...]
    params: dict = field(default_factory=dict)   # overrides of the base defaults
    variant: bool = False
    note_en: str = ""                  # variants: the motivation, chosen on in-sample data
    note_de: str = ""


_DEFAULTS: dict[str, dict] = {
    "T0": {},
    "T1": {"q": 0.20, "lookback": 12, "skip": 1},
    "T2": {"fast": 50, "slow": 200},
    "T3": {"q": 0.30, "lookback": 12, "skip": 1, "slow": 200, "market": True},
    "T4": {"up": 0.95, "down": 0.70, "slow": 200},
    "T5": {"q": 0.20, "slow": 200},
}

RULES: dict[str, RuleSpec] = {r.id: r for r in [
    RuleSpec(
        "T0", "T0", "Technical composite (baseline)", "Technischer Gesamtscore (Basis)",
        "The scanner's own score (trend, momentum, relative strength, MACD, RSI, volume) "
        "with the regime thresholds.",
        "Der Gesamtscore des Scanners (Trend, Momentum, relative Stärke, MACD, RSI, Volumen) "
        "mit den Regime-Schwellen.",
        (MEDIUM,)),
    RuleSpec(
        "T1", "T1", "Momentum 12-1", "Momentum 12-1",
        "Return from 12 months ago to 1 month ago; Long = top 20 % of the universe, "
        "Short = bottom 20 %.",
        "Rendite von vor 12 bis vor 1 Monat; Long = oberste 20 % des Universums, "
        "Short = unterste 20 %.",
        (MEDIUM, LONG)),
    RuleSpec(
        "T2", "T2", "Trend state", "Trendzustand",
        "Long when price is above the 200-day average and the 50-day average is above it too; "
        "Short for the mirror image.",
        "Long, wenn der Kurs über dem 200-Tage-Schnitt und der 50-Tage-Schnitt darüber liegt; "
        "Short im Spiegelbild.",
        (MEDIUM, LONG)),
    RuleSpec(
        "T3", "T3", "Momentum with trend and market filter",
        "Momentum mit Trend- und Marktfilter",
        "Long = top 30 % by 12-1 momentum and above the 200-day average; Short = bottom 30 %, "
        "below it, and the benchmark below its own 200-day average.",
        "Long = oberste 30 % nach 12-1-Momentum und über dem 200-Tage-Schnitt; Short = unterste "
        "30 %, darunter, und der Vergleichsindex unter seinem eigenen 200-Tage-Schnitt.",
        (MEDIUM, LONG)),
    RuleSpec(
        "T4", "T4", "52-week high", "52-Wochen-Hoch",
        "Long when the close is within 5 % of the 52-week closing high; Short when it is 30 % "
        "or more below it and under the 200-day average.",
        "Long, wenn der Schlusskurs höchstens 5 % unter dem 52-Wochen-Schlusshoch liegt; Short, "
        "wenn er mindestens 30 % darunter und unter dem 200-Tage-Schnitt liegt.",
        (MEDIUM,)),
    RuleSpec(
        "T5", "T5", "Pullback in the trend", "Rücksetzer im Trend",
        "Long = the 20 % weakest 1-month returns among names above their 200-day average; "
        "Short = the 20 % strongest among names below it.",
        "Long = die 20 % schwächsten 1-Monats-Renditen unter den Werten über dem 200-Tage-"
        "Schnitt; Short = die 20 % stärksten unter den Werten darunter.",
        (MEDIUM,)),
]}


# Variants: each added AFTER the in-sample results of T0-T5 were seen, chosen on those results
# only and never pruned afterwards (failures stay in the evidence).  Maximum six.
_VARIANTS = [
    RuleSpec(
        "V1", "T1", "Momentum 12-1, 10 % tails", "Momentum 12-1, 10 % Ränder",
        "Like T1 but only the top and bottom 10 % of the universe.",
        "Wie T1, aber nur die obersten und untersten 10 % des Universums.",
        (MEDIUM, LONG), {"q": 0.10}, True,
        "In-sample, the momentum premium appeared only at the extremes (US long horizon), so "
        "tighter tails might concentrate it.",
        "Im In-Sample-Teil zeigte sich die Momentum-Prämie nur an den Rändern (USA, lang): "
        "engere Ränder könnten sie bündeln."),
    RuleSpec(
        "V2", "T1", "Momentum 6-1", "Momentum 6-1",
        "Like T1 but the return from 6 months ago to 1 month ago.",
        "Wie T1, aber die Rendite von vor 6 bis vor 1 Monat.",
        (MEDIUM, LONG), {"lookback": 6}, True,
        "The 12-month window was weak in-sample on most families; the shorter classic "
        "6-month formation reacts faster.",
        "Das 12-Monats-Fenster war im In-Sample-Teil bei den meisten Familien schwach; die "
        "klassische 6-Monats-Bildung reagiert schneller."),
    RuleSpec(
        "V3", "T3", "Momentum with trend filter, no market filter",
        "Momentum mit Trendfilter, ohne Marktfilter",
        "Like T3 but the Short side no longer requires the benchmark below its 200-day average.",
        "Wie T3, aber die Short-Seite verlangt nicht mehr, dass der Index unter seinem "
        "200-Tage-Schnitt liegt.",
        (MEDIUM, LONG), {"market": False}, True,
        "In-sample the market filter left the Short side nearly empty (13 to 85 of 432 dates), "
        "so T3 could not be judged there.",
        "Im In-Sample-Teil ließ der Marktfilter die Short-Seite fast leer (13 bis 85 von 432 "
        "Terminen): T3 war dort nicht beurteilbar."),
    RuleSpec(
        "V4", "T2", "Faster trend state (20/100)", "Schnellerer Trendzustand (20/100)",
        "Like T2 with 20- and 100-day averages instead of 50 and 200.",
        "Wie T2 mit 20- und 100-Tage-Schnitt statt 50 und 200.",
        (MEDIUM, LONG), {"fast": 20, "slow": 100}, True,
        "The 200-day average lags; in-sample T2 earned no more than the universe, a faster "
        "state might turn earlier.",
        "Der 200-Tage-Schnitt hinkt hinterher; im In-Sample-Teil verdiente T2 nicht mehr als "
        "das Universum, ein schnellerer Zustand könnte früher drehen."),
    RuleSpec(
        "V5", "T5", "Pullback, 10 % tails", "Rücksetzer, 10 % Ränder",
        "Like T5 but the 10 % weakest/strongest 1-month returns.",
        "Wie T5, aber die 10 % schwächsten/stärksten 1-Monats-Renditen.",
        (MEDIUM,), {"q": 0.10}, True,
        "In-sample the short side of T5 lagged the market on ETFs; deeper extremes might "
        "make the reversal clearer.",
        "Im In-Sample-Teil blieb die Short-Seite von T5 bei ETFs hinter dem Markt zurück: "
        "tiefere Extreme könnten die Umkehr deutlicher machen."),
]
RULES.update({r.id: r for r in _VARIANTS})


def register_variant(spec: RuleSpec) -> None:
    """Add a variant to the registry (used by the study module for its logged variants)."""
    RULES[spec.id] = spec


def rule_params(rule_id: str) -> dict:
    spec = RULES[rule_id]
    return {**_DEFAULTS[spec.base], **spec.params}


def horizon_plan(horizon: str, month_bars: int) -> tuple[int, int]:
    """``(hold bars, decision step in bars)``: medium holds 1 month and decides every 5 bars
    (crypto 7); long holds 6 months and decides monthly."""
    if horizon == MEDIUM:
        return month_bars, 7 if month_bars >= 30 else 5
    if horizon == LONG:
        return 6 * month_bars, month_bars
    raise ValueError(f"unknown horizon {horizon!r}")


# ------------------------------------------------------------------------------ panel


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    """Bars with a positive close; a missing high/low falls back to the close, an unusable
    open (NaN or <= 0) stays NaN so that no fill can happen on it."""
    if df is None or "Close" not in df.columns:
        return None
    f = df[df["Close"].notna() & (df["Close"] > 0)].copy()
    if f.empty:
        return None
    f = f[~f.index.duplicated(keep="last")].sort_index()
    for col in ("High", "Low"):
        f[col] = f[col].fillna(f["Close"]) if col in f.columns else f["Close"]
    f["Open"] = f["Open"].where(f["Open"] > 0) if "Open" in f.columns else np.nan
    return f


class Panel:
    """Dates x symbols price panels plus the cached per-symbol features rules read.

    ``index`` is the union of the members' bar dates, minus dates on which fewer than
    ``MIN_ROW_SHARE`` of the symbols already listed have a bar (a stray print is not a
    trading day).  Features come from each symbol's OWN bars and are aligned afterwards,
    so a hole in one symbol's history cannot shift its windows.
    """

    def __init__(self, frames: dict[str, pd.DataFrame], members: list[str], benchmark: str,
                 params: SignalParams):
        self.params = params
        self.benchmark = benchmark
        self.clean = {s: c for s in members
                      if (c := _clean_frame(frames.get(s))) is not None}
        self.members = list(self.clean)
        if not self.members:
            raise ValueError("no instrument has usable bars")
        union = sorted(set().union(*(set(c.index) for c in self.clean.values())))
        idx = pd.DatetimeIndex(union)
        firsts = np.sort(np.array([c.index[0] for c in self.clean.values()], dtype="datetime64[ns]"))
        alive = np.searchsorted(firsts, idx.to_numpy(dtype="datetime64[ns]"), side="right")
        have = self._counts(idx)
        keep = have >= np.maximum(1, np.ceil(MIN_ROW_SHARE * alive))
        self.index = idx[keep]
        self._cache: dict = {}
        self.open, self.high, self.low, self.close = (
            self._stack(lambda c, col=col: c[col]) for col in ("Open", "High", "Low", "Close"))
        bench = _clean_frame(frames.get(benchmark))
        self.bench = bench
        self.bench_open = (bench["Open"].reindex(self.index) if bench is not None
                           else pd.Series(np.nan, index=self.index))
        self.bench_close = (bench["Close"].reindex(self.index) if bench is not None
                            else pd.Series(np.nan, index=self.index))

    def _counts(self, idx: pd.DatetimeIndex) -> np.ndarray:
        total = np.zeros(len(idx), dtype=int)
        for c in self.clean.values():
            total += idx.isin(c.index)
        return total

    def _stack(self, fn) -> pd.DataFrame:
        cols = {s: fn(c) for s, c in self.clean.items()}
        return pd.DataFrame(cols).reindex(self.index)[self.members]

    def _memo(self, key, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    # -- features (row t uses own bars <= t only)
    def sma(self, n: int) -> pd.DataFrame:
        return self._memo(("sma", n), lambda: self._stack(lambda c: ind.sma(c["Close"], n)))

    def momentum(self, lookback: int, skip: int) -> pd.DataFrame:
        m = self.params.month_bars
        return self._memo(("mom", lookback, skip), lambda: self._stack(
            lambda c: c["Close"].shift(skip * m) / c["Close"].shift(lookback * m) - 1.0))

    def ret_months(self, months: int) -> pd.DataFrame:
        m = self.params.month_bars
        return self._memo(("ret", months), lambda: self._stack(
            lambda c: ind.pct_return(c["Close"], months * m)))

    def high_n(self, n: int) -> pd.DataFrame:
        return self._memo(("hi", n), lambda: self._stack(
            lambda c: c["Close"].rolling(n, min_periods=n).max()))

    def atr(self) -> pd.DataFrame:
        return self._memo(("atr",), lambda: self._stack(
            lambda c: ind.atr(c["High"], c["Low"], c["Close"], ATR_N)))

    def bench_sma(self, n: int) -> pd.Series:
        if self.bench is None:
            return pd.Series(np.nan, index=self.index)
        return self._memo(("bsma", n), lambda: ind.sma(self.bench["Close"], n).reindex(self.index))

    def warm_row(self) -> int:
        """First row from which every pre-registered rule has its features for >= 5 names."""
        valid = (self.momentum(12, 1).notna().sum(axis=1) >= MIN_RANK_NAMES) \
            & (self.high_n(self.params.bars_per_year).notna().sum(axis=1) >= MIN_RANK_NAMES) \
            & (self.sma(200).notna().sum(axis=1) >= MIN_RANK_NAMES)
        rows = np.flatnonzero(valid.to_numpy())
        return int(rows[0]) if len(rows) else len(self.index)


build_panel = Panel


# ------------------------------------------------------------------------------ rules


def _quantile_mask(values: pd.DataFrame, q: float, top: bool) -> pd.DataFrame:
    """True for the best ``q`` share of the names that HAVE a value on that date.

    ``top=True`` selects the highest values.  Names without a value never count in the
    denominator, and a date with fewer than ``MIN_RANK_NAMES`` valid names selects none.
    Ties are broken by column order, which keeps the result deterministic.
    """
    n = values.notna().sum(axis=1)
    rank = values.rank(axis=1, method="first", ascending=not top)
    k = np.ceil(np.round(q * n.to_numpy(dtype=float), 9)).clip(min=1)
    ok = (n.to_numpy() >= MIN_RANK_NAMES)
    mask = rank.le(k, axis=0) & values.notna()
    return mask & ok[:, None]


def _sides(long_mask: pd.DataFrame, short_mask: pd.DataFrame) -> pd.DataFrame:
    return long_mask.astype(np.int8) - short_mask.astype(np.int8)


def rule_sides(rule_id: str, panel: Panel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(sides, scores)`` for every date of the panel; ``sides`` is +1/-1/0.

    ``T0`` (the composite) cannot be computed vectorised: use ``composite_panel``.
    """
    spec = RULES[rule_id]
    p = rule_params(rule_id)
    close = panel.close
    base = spec.base
    if base == "T0":
        raise ValueError("T0 is computed by composite_panel")
    if base in ("T1", "T3"):
        mom = panel.momentum(p["lookback"], p["skip"])
        scores = mom
        long_m = _quantile_mask(mom, p["q"], True)
        short_m = _quantile_mask(mom, p["q"], False)
        if base == "T3":
            sma = panel.sma(p["slow"])
            long_m &= close > sma
            short_m &= close < sma
            if p["market"]:
                below = (panel.bench_close < panel.bench_sma(p["slow"])).to_numpy()
                short_m &= below[:, None]
        return _sides(long_m, short_m), scores
    if base == "T2":
        fast, slow = panel.sma(p["fast"]), panel.sma(p["slow"])
        long_m = (close > slow) & (fast > slow)
        short_m = (close < slow) & (fast < slow)
        return _sides(long_m, short_m), close / slow - 1.0
    if base == "T4":
        hi = panel.high_n(panel.params.bars_per_year)
        near = close / hi
        slow = panel.sma(p["slow"])
        long_m = near >= p["up"]
        short_m = (near <= p["down"]) & (close < slow)
        return _sides(long_m, short_m), near - 1.0
    if base == "T5":
        r1 = panel.ret_months(1)
        slow = panel.sma(p["slow"])
        long_m = _quantile_mask((-r1).where(close > slow), p["q"], True)
        short_m = _quantile_mask(r1.where(close < slow), p["q"], True)
        return _sides(long_m, short_m), -r1
    raise ValueError(f"unknown rule {rule_id!r}")


def composite_panel(
    frames: dict[str, pd.DataFrame], members: list[str], benchmark: str,
    dates: pd.DatetimeIndex, params: SignalParams, use_vix: bool = True,
    window_bars: int = COMPOSITE_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """T0: the scanner's own ``analyse`` called at each date on bars cut at that date.

    Each call sees the last ``window_bars`` bars, which is what the live scan sees
    (600 calendar days), so the baseline measured here is the live composite.
    """
    from tradingagents.scanner.scan import VIX, analyse  # lazy: scan imports horizons

    sides = pd.DataFrame(0, index=dates, columns=members, dtype=np.int8)
    scores = pd.DataFrame(np.nan, index=dates, columns=members)
    keys = [s for s in [*members, benchmark, *([VIX] if use_vix else [])] if s in frames]
    indexes = {s: frames[s].index for s in keys}
    for d in dates:
        cut = {}
        for s in keys:
            k = int(indexes[s].searchsorted(d, side="right"))
            if k:
                cut[s] = frames[s].iloc[max(0, k - window_bars):k]
        _, signals, _ = analyse(members, cut, benchmark, at=d, params=params, use_vix=use_vix)
        for sig in signals:
            scores.at[d, sig.symbol] = sig.score
            sides.at[d, sig.symbol] = 1 if sig.direction == "Long" else (
                -1 if sig.direction == "Short" else 0)
    return sides, scores


def current_members(rule_id: str, panel: Panel) -> list[tuple[str, int, float, int]]:
    """``(symbol, side, score, rank)`` of the latest date for a non-composite rule.

    ``rank`` counts from 1 within the side (best candidate first).  Names that have no
    bar on the panel's last date are not members.
    """
    sides, scores = rule_sides(rule_id, panel)
    row_s, row_v = sides.iloc[-1], scores.iloc[-1]
    out: list[tuple[str, int, float, int]] = []
    for side in (1, -1):
        names = [s for s in panel.members if row_s[s] == side and math.isfinite(row_v[s])]
        names.sort(key=lambda s: (-row_v[s] if side == 1 else row_v[s], s))
        out.extend((s, side, float(row_v[s]), i + 1) for i, s in enumerate(names))
    return out


__all__ = ["RULES", "RuleSpec", "Panel", "build_panel", "composite_panel", "current_members",
           "horizon_plan", "register_variant", "rule_params", "rule_sides", "MEDIUM", "LONG"]
