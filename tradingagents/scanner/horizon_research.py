"""Medium- and long-horizon study: do rules with a prior in the literature beat the market?

Run it with ``python -m tradingagents.scanner.horizon_research --out DIR``.  It downloads
daily bars (as many as Yahoo has, up to twelve years of decisions) for four instrument
families, evaluates the pre-registered rules of ``rules`` and writes ``evidence.json``
plus a German report.  The idea engine (``horizons``) reads only that snapshot, so
"what the program claims" is exactly "what was measured here".

The discipline, in the order it is applied:

* A signal of day t is filled at the OPEN of t + 1 and the position is closed at the close
  of bar t + hold.  Per decision date the Long basket L, the Short basket S and the equal-
  weight universe U (the no-skill drift) are averaged over the same window.  Costs are a
  fixed round-trip number of basis points per position taken.
* Chronological 70/30 split of the decision dates with a purge: in-sample dates whose
  holding window reaches into the out-of-sample period are dropped.  Parameters and
  variants are chosen on the in-sample part only; the out-of-sample part is evaluated once
  per final candidate and reported whatever it shows.
* Windows overlap (a one-month hold decided every five days), so a t-stat over all dates
  would be far too confident.  After the independent review the VERDICT uses the plain t
  of the non-overlapping subsample (every k-th date, counted back from the last decision
  date of the segment, so the phase is explicit and deterministic).  The Newey-West t
  (lag = windows per hold - 1) stays in the snapshot as information only: the review
  showed it is optimistic (a 6-window overlap gets a variance factor of 4.06, not 6, and
  its false-positive rate under the null is 5 to 10 % instead of 2.3 %).  A moving-block
  bootstrap interval and the share of the k phase offsets that would pass are reported
  for transparency.
* Verdicts use the pre-registered thresholds of ``shortterm.evaluate``; "validated" needs
  the edge on BOTH segments.  A label above ``unproven`` also needs an average basket of at
  least ``MIN_BASKET`` instruments (a basket of one ETF is an anecdote) and survives only
  if the series recomputed WITHOUT the ``ROBUST_K`` members with the highest total return
  still has a pooled t >= ``ROBUST_MIN_T``: today's list is applied backwards, and a rule
  that merely buys what later became a mega cap would otherwise be mistaken for skill.
* The number of DISTINCT evaluations is recorded because with enough tests something
  passes by chance.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.dataflows.vendors.yahoo.bulk import load_bulk_ohlcv
from tradingagents.scanner import rules
from tradingagents.scanner.fmt import fmt_num
from tradingagents.scanner.notices import LABEL_NAMES
from tradingagents.scanner.rules import LONG, MEDIUM, RULES, Panel
from tradingagents.scanner.session import settled
from tradingagents.scanner.shortterm.evaluate import (
    EXPERIMENTAL_T_IS,
    EXPERIMENTAL_T_OOS,
    STRONG_T,
    VALIDATED_T,
)
from tradingagents.scanner.shortterm.simulate import simulate_trade
from tradingagents.scanner.signals import SignalParams
from tradingagents.scanner.universe import universe_info

logger = logging.getLogger(__name__)

SNAPSHOT_NAME = "evidence.json"
SNAPSHOT_PATH = Path(__file__).with_name(SNAPSHOT_NAME)
SNAPSHOT_VERSION = 3
IS_FRACTION = 0.70
N_BOOT = 2000
SEED = 20261113
VALIDATED_MIN_OOS_DATES = 30
EXPERIMENTAL_MIN_OOS_DATES = 20       # added guard: a t >= 1 on a handful of dates is noise
MIN_BASKET = 3                        # average basket below this: no label above unproven
ROBUST_K = 3                          # members with the highest total return removed (fixed now)
ROBUST_MIN_T = 2.0                    # pooled t that must survive their removal
STOP_ATR = {MEDIUM: 2.5, LONG: 4.0}
COSTS_BPS = {"stocks_us": 10.0, "stocks_dax": 10.0, "etfs": 10.0, "crypto": 30.0}
COST_MULTIPLES = (1.0, 2.0)
HISTORY_DAYS = int(13.5 * 365)        # 12 years of decisions after ~1.3 years of warm-up
SEGMENTS = ("IS", "OOS", "ALL")
SIDE_SERIES = {"long": ("long_excess", "long_abs", "spread"),
               "short": ("short_pnl", "short_excess", "spread")}
JUDGED = {"long": "long_excess", "short": "short_pnl"}


@dataclass(frozen=True)
class FamilySpec:
    name: str
    universe: str
    tickers: tuple[str, ...] | None = None


def _etf_members() -> tuple[str, ...]:
    return tuple(dict.fromkeys([*universe_info("global").members,
                                *universe_info("sectors").members]))


def family_specs() -> dict[str, FamilySpec]:
    return {
        "stocks_us": FamilySpec("stocks_us", "us"),
        "stocks_dax": FamilySpec("stocks_dax", "dax"),
        "etfs": FamilySpec("etfs", "global", _etf_members()),
        "crypto": FamilySpec("crypto", "crypto"),
    }


# --------------------------------------------------------------------------- statistics


def nw_tstat(series, lags: int) -> float:
    """Newey-West t-statistic of the mean (Bartlett kernel, ``lags`` autocovariances).

    Overlapping holding windows make neighbouring dates share most of their return, so
    the variance of the mean is the long-run variance
    ``gamma0 + 2 * sum_k (1 - k / (lags + 1)) * gamma_k``, divided by n.  Returns 0.0 when
    it is undefined (fewer than two values, no variance).
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return 0.0
    e = x - x.mean()
    lags = max(0, min(int(lags), n - 1))
    long_run = float(e @ e) / n
    for k in range(1, lags + 1):
        long_run += 2.0 * (1.0 - k / (lags + 1.0)) * float(e[k:] @ e[:-k]) / n
    if not long_run > 0.0:
        return 0.0
    return float(x.mean() / math.sqrt(long_run / n))


def plain_tstat(series) -> float:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    sd = float(x.std(ddof=1))
    return 0.0 if not sd > 0 else float(x.mean() / (sd / math.sqrt(len(x))))


def block_bootstrap_ci(series, block: int, n_boot: int = N_BOOT,
                       seed: int = SEED) -> tuple[float, float]:
    """95 % percentile interval of the mean from a moving-block bootstrap.

    Blocks of ``block`` consecutive dates keep the overlap-induced dependence inside a
    resample; the seed is fixed so the interval is reproducible.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    b = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / b)
    starts = rng.integers(0, n - b + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def nonoverlap_tail(x, k: int) -> np.ndarray:
    """Every ``k``-th value counted back from the LAST one.

    Windows of ``k`` consecutive dates cover the holding period, so these values do not
    overlap.  Anchoring at the end (the last decision date of the segment) makes the phase
    explicit and deterministic: the same data always give the same subsample.
    """
    x = np.asarray(x, dtype=float)
    return x[::-1][::max(1, int(k))][::-1]


def series_stats(values, hold: int, step: int, boot: bool = True) -> dict:
    """The statistics block of one series on one segment; values are fractions, output bps.

    ``t_nonoverlap`` is what the verdicts judge: the MEDIAN, over all ``k`` possible phases
    (start offsets) of the non-overlapping subsample, of the plain t.  Judging one fixed
    phase made the verdict depend on an arbitrary choice (an independent review showed
    t from 0.7 to 2.6 across the phases of the same cell), so the rule was changed to the
    median.  ``phase_t`` keeps every phase's t (the snapshot shows them), ``t_tail`` is the
    old single-phase value (subsample ending at the last date), ``phases_pass`` counts the
    phases that reach ``VALIDATED_T``.  ``nw_t`` is kept for information and is optimistic.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    k = math.ceil(hold / step)
    out = {"n": int(len(x)), "mean_bps": None, "median_bps": None, "pos_share": None,
           "nw_t": 0.0, "t_nonoverlap": 0.0, "t_tail": 0.0, "phase_t": [],
           "n_nonoverlap": 0, "mean_nonoverlap_bps": None, "phases": k, "phases_pass": 0,
           "ci_low_bps": None, "ci_high_bps": None}
    if len(x) == 0:
        return out
    out["mean_bps"] = float(x.mean() * 1e4)
    out["median_bps"] = float(np.median(x) * 1e4)
    out["pos_share"] = float((x > 0).mean())
    out["nw_t"] = nw_tstat(x, k - 1)
    phase_t = [plain_tstat(x[o::k]) for o in range(k)]
    out["phase_t"] = [round(float(t), 3) for t in phase_t]
    out["t_nonoverlap"] = float(np.median(phase_t))
    out["t_tail"] = plain_tstat(nonoverlap_tail(x, k))
    # the mean that belongs to the judged t: the phase(s) holding the median t
    order = np.argsort(phase_t, kind="stable")
    mid = order[(k - 1) // 2: k // 2 + 1]
    out["mean_nonoverlap_bps"] = float(np.mean([x[o::k].mean() for o in mid]) * 1e4)
    out["n_nonoverlap"] = int(round(float(np.median([len(x[o::k]) for o in range(k)]))))
    out["phases_pass"] = int(sum(t >= VALIDATED_T for t in phase_t))
    if boot and len(x) >= 2:
        lo, hi = block_bootstrap_ci(x, k)
        out["ci_low_bps"], out["ci_high_bps"] = lo * 1e4, hi * 1e4
    return out


def split_positions(pos: np.ndarray, hold: int, is_fraction: float = IS_FRACTION
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices into the decision dates: ``(in-sample kept, out-of-sample, purged)``.

    The first ``is_fraction`` of the dates (a count, fixed by the data alone) is in-sample.
    An in-sample date whose window, which ends at bar ``pos + hold``, reaches the first
    out-of-sample decision bar is purged, so no return that overlaps the out-of-sample
    period leaks into the in-sample statistics.
    """
    n = len(pos)
    n_is = int(math.floor(n * is_fraction))
    oos = np.arange(n_is, n)
    if n_is >= n:
        return np.arange(n_is), oos, np.array([], dtype=int)
    first_oos = pos[n_is]
    head = np.arange(n_is)
    purged_mask = pos[:n_is] + hold >= first_oos
    return head[~purged_mask], oos, head[purged_mask]


def return_series(L, S, U, cost: float) -> dict[str, np.ndarray]:
    """The five per-date series, net of ``cost`` (a fraction per round trip) where a
    position is taken; NaN where a basket is empty."""
    return {
        "long_excess": L - U - cost,
        "long_abs": L - cost,
        "short_pnl": -S - cost,
        "short_excess": U - S - cost,
        "spread": L - S - 2.0 * cost,
    }


def daily_max_drawdown(paths) -> float | None:
    """Maximum drawdown of consecutive holding periods marked every bar.

    ``paths`` are the value paths of the periods, each relative to its own start (1.0 at
    the fill); the end of one period is the start of the next.  Marking only at the
    period ends hides every dip inside a half year (the review: -6 % at half-year points
    against -31 % on daily marks).
    """
    level, peak, worst = 1.0, 1.0, 0.0
    seen = False
    for path in paths:
        p = np.asarray(path, dtype=float)
        if len(p) == 0 or not np.all(np.isfinite(p)):
            continue
        seen = True
        marks = level * p
        running = np.maximum(peak, np.maximum.accumulate(marks))
        worst = min(worst, float((marks / running - 1.0).min()))
        peak = float(running[-1])
        level = float(marks[-1])
    return worst if seen else None


def risk_stats(period_returns, periods_per_year: float, daily_paths=None) -> dict:
    """CAGR, annualised volatility, Sharpe (risk-free 0) and maximum drawdown of a series of
    non-overlapping period returns; the periods are compounded.  With ``daily_paths`` the
    drawdown is measured on the daily marks inside the periods, otherwise at period ends."""
    x = np.asarray(period_returns, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {"n": int(n), "cagr": None, "vol": None, "sharpe": None, "max_drawdown": None}
    equity = np.cumprod(1.0 + x)
    years = n / periods_per_year
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0 else -1.0
    sd = float(x.std(ddof=1))
    vol = sd * math.sqrt(periods_per_year)
    sharpe = float(x.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else None
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    drawdown = float(((equity / peak) - 1.0).min())
    if daily_paths is not None:
        drawdown = daily_max_drawdown(daily_paths)
    return {"n": int(n), "cagr": cagr, "vol": float(vol), "sharpe": sharpe,
            "max_drawdown": drawdown}


# --------------------------------------------------------------------------- verdict


PLAIN, NEWEY_WEST = "t_nonoverlap", "nw_t"


def _t(stats: dict, key: str = PLAIN) -> float:
    """The judged t: plain t of the non-overlapping subsample (see ``series_stats``);
    ``key=NEWEY_WEST`` gives the optimistic statistic the first version judged by."""
    return float(stats.get(key) or 0.0)


def _meets(level: str, is_s: dict, oos_s: dict, oos_mean_2x_bps: float | None,
           key: str = PLAIN) -> bool:
    oos_mean = oos_s.get("mean_bps")
    if oos_mean is None:
        return False
    if level == "validated":
        return (_t(is_s, key) >= VALIDATED_T and _t(oos_s, key) >= VALIDATED_T
                and oos_mean_2x_bps is not None and oos_mean_2x_bps > 0
                and oos_s["n"] >= VALIDATED_MIN_OOS_DATES)
    return (_t(is_s, key) >= EXPERIMENTAL_T_IS and oos_mean > 0
            and _t(oos_s, key) >= EXPERIMENTAL_T_OOS
            and oos_s["n"] >= EXPERIMENTAL_MIN_OOS_DATES)


def _level(is_s: dict, oos_s: dict, oos_2x: float | None, key: str = PLAIN) -> str | None:
    for level in ("validated", "experimental"):
        if _meets(level, is_s, oos_s, oos_2x, key):
            return level
    return None


def judge_side(side: str, series: dict[str, dict], oos_2x: dict[str, float | None],
               variant: bool = False, avg_basket: float | None = None,
               key: str = PLAIN) -> dict:
    """Pre-registered verdict of one side from its series statistics.

    Long is judged on ``long_excess`` (beating the universe, not riding it).  Short is
    judged on ``short_pnl`` (the price must really fall); failing that, ``short_excess``
    meeting the same thresholds gives ``relative`` (the name lags the market: a reason to
    avoid it, no evidence for an outright short).  A long that fails against the universe
    but passes on ``long_abs`` is ``beta``: it made money only because the market did.  The
    label needs the out-of-sample surplus over the universe to be absent (mean <= 0 or
    t < 1); a rule that did beat the universe out of sample but failed in-sample stays
    ``unproven``, because beta would misdescribe it.  Every t is the plain t of the
    non-overlapping subsample (``key`` selects another statistic, used only to show what
    the first version of the study would have said).  An average basket below
    ``MIN_BASKET`` instruments leaves the verdict at ``unproven`` (``basket_too_small``).
    ``strong`` needs a pooled t >= 3 on the judged series and some verdict above unproven;
    a variant is never strong, because it was added after the in-sample results had been
    seen.
    """
    judged = JUDGED[side]
    s = series[judged]
    level = _level(s["IS"], s["OOS"], oos_2x.get(judged), key)
    verdict = level or "unproven"
    if level is None and side == "short":
        ex = series["short_excess"]
        if _level(ex["IS"], ex["OOS"], oos_2x.get("short_excess"), key):
            verdict = "relative"
    if level is None and side == "long":
        ab, ex = series["long_abs"], series["long_excess"]["OOS"]
        no_surplus = (ex.get("mean_bps") is None or ex["mean_bps"] <= 0
                      or _t(ex, key) < EXPERIMENTAL_T_OOS)
        if no_surplus and _level(ab["IS"], ab["OOS"], oos_2x.get("long_abs"), key):
            verdict = "beta"
    small = avg_basket is not None and avg_basket < MIN_BASKET
    if small and verdict != "unproven":
        verdict = "unproven"
    pooled = _t(s["ALL"], key)
    return {"verdict": verdict, "judged": judged, "pooled_t": pooled,
            "pooled_t_nw": float(s["ALL"].get("nw_t") or 0.0),
            "basket_too_small": bool(small),
            "strong": bool(verdict != "unproven" and pooled >= STRONG_T and not variant)}


def phase_verdicts(side: str, series: dict[str, dict], oos_2x: dict[str, float | None],
                   variant: bool = False, avg_basket: float | None = None) -> list[str]:
    """The verdict the cell would get if the plain t of each single phase were judged
    (phase p of in-sample with phase p of out-of-sample), before the robustness check.
    Transparency only: the verdict itself uses the median over the phases."""
    k = max((len(st.get("phase_t") or []) for s in series.values() for st in s.values()),
            default=0)
    out = []
    for p in range(k):
        shifted = {}
        for name, segs in series.items():
            shifted[name] = {
                seg: {**st, PLAIN: (st["phase_t"][p] if p < len(st.get("phase_t") or [])
                                    else st.get(PLAIN))}
                for seg, st in segs.items()}
        out.append(judge_side(side, shifted, oos_2x, variant, avg_basket)["verdict"])
    return out


# --------------------------------------------------------------------------- grid


def _atr_stop(direction: int, distance: float):
    """A stop ``distance`` against the position, priced from the entry fill."""
    def level(entry: float) -> float:
        return entry - direction * distance
    return level


class Grid:
    """Decision dates of one family and horizon with every per-date quantity precomputed.

    ``R[p, s]`` is the window return of symbol s for decision p (open of bar t+1 to close
    of bar t+hold, NaN when either bar is missing) -- the no-look-ahead fill; ``U`` the
    equal-weight mean of all valid R of that date; ``B`` the benchmark over the same window.
    """

    def __init__(self, panel: Panel, horizon: str, is_fraction: float = IS_FRACTION):
        self.panel, self.horizon = panel, horizon
        self.hold, self.step = rules.horizon_plan(horizon, panel.params.month_bars)
        n = len(panel.index)
        self.pos = np.arange(panel.warm_row(), n - self.hold, self.step)
        self.dates = panel.index[self.pos]
        o, c = panel.open.to_numpy(float), panel.close.to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = c[self.pos + self.hold] / o[self.pos + 1] - 1.0
            bo = panel.bench_open.to_numpy(float)[self.pos + 1]
            bc = panel.bench_close.to_numpy(float)[self.pos + self.hold]
            self.B = bc / bo - 1.0
        r[~np.isfinite(r)] = np.nan
        self.R = r
        valid = np.isfinite(r)
        cnt = valid.sum(axis=1)
        self.U = np.where(cnt > 0, np.nansum(r, axis=1) / np.maximum(cnt, 1), np.nan)
        self.is_idx, self.oos_idx, self.purged_idx = split_positions(
            self.pos, self.hold, is_fraction)
        self.all_idx = np.concatenate([self.is_idx, self.oos_idx])
        self._arrays = None
        self._marks = None

    def segments(self, include_oos: bool = True) -> dict[str, np.ndarray]:
        seg = {"IS": self.is_idx}
        if include_oos:
            seg["OOS"], seg["ALL"] = self.oos_idx, self.all_idx
        return seg

    def info(self) -> dict:
        d = self.dates

        def day(i):
            return str(d[i].date()) if len(d) else None
        return {
            "hold": self.hold, "step": self.step, "n_dates": int(len(d)),
            "first_decision": day(0) if len(d) else None,
            "last_decision": day(-1) if len(d) else None,
            "is_n": int(len(self.is_idx)), "oos_n": int(len(self.oos_idx)),
            "purged_n": int(len(self.purged_idx)),
            "is_first": day(self.is_idx[0]) if len(self.is_idx) else None,
            "is_last": day(self.is_idx[-1]) if len(self.is_idx) else None,
            "oos_first": day(self.oos_idx[0]) if len(self.oos_idx) else None,
            "oos_last": day(self.oos_idx[-1]) if len(self.oos_idx) else None,
        }

    def ohlc_by_symbol(self):
        if self._arrays is None:
            p = self.panel
            self._arrays = tuple(np.ascontiguousarray(f.to_numpy(float).T)
                                 for f in (p.open, p.high, p.low, p.close))
        return self._arrays

    def window_path(self, p: int, member_mask: np.ndarray) -> np.ndarray | None:
        """Equal-weight buy-and-hold value path of the masked members over the holding
        window of decision ``p``, marked at every close and relative to the fill (the open
        of t + 1).  A missing bar inside the window carries the previous close."""
        if self._marks is None:
            self._marks = (self.panel.close.ffill().to_numpy(float),
                           self.panel.bench_close.ffill().to_numpy(float))
        i = int(self.pos[p])
        rows = slice(i + 1, i + self.hold + 1)
        if member_mask is None:
            fill = self.panel.bench_open.to_numpy(float)[i + 1]
            rel = self._marks[1][rows] / fill
            return rel if np.all(np.isfinite(rel)) else None
        if not member_mask.any():
            return None
        fill = self.panel.open.to_numpy(float)[i + 1, member_mask]
        rel = self._marks[0][rows][:, member_mask] / fill
        return np.nanmean(rel, axis=1)

    def basket_means(self, sides_pos: np.ndarray, returns: np.ndarray | None = None):
        """``(L, S, nL, nS)``: mean window return of the Long and Short baskets per date."""
        r = self.R if returns is None else returns
        ok = np.isfinite(r)
        out = []
        for side in (1, -1):
            m = (sides_pos == side) & ok
            cnt = m.sum(axis=1)
            tot = np.where(m, np.nan_to_num(r), 0.0).sum(axis=1)
            out.append((np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), cnt))
        (lo, n_lo), (sh, n_sh) = out
        return lo, sh, n_lo, n_sh

    def stop_returns(self, sides_pos: np.ndarray, atr_mult: float) -> np.ndarray:
        """Window returns when every position carries a protective stop ``atr_mult`` x ATR14
        from the entry fill (stop-first, gaps fill at the open: ``simulate_trade``).

        Returned as the raw-return equivalent (``exit / entry - 1``), so the same basket
        formulas apply to Long and Short.  ATR is the value at the signal close.
        """
        o, h, lo, c = self.ohlc_by_symbol()
        atr = self.panel.atr().to_numpy(float)
        out = np.full_like(self.R, np.nan)
        for p in range(len(self.pos)):
            i = int(self.pos[p])
            for s in np.flatnonzero((sides_pos[p] != 0) & np.isfinite(self.R[p])):
                d = int(sides_pos[p, s])
                a = atr[i, s]
                stop = _atr_stop(d, atr_mult * a) if np.isfinite(a) and a > 0 else None
                tr = simulate_trade(o[s], h[s], lo[s], c[s], entry_idx=i + 1, direction=d,
                                    last_idx=i + self.hold, cost_bps=0.0, stop=stop)
                out[p, s] = d * tr.gross_return
        return out


# --------------------------------------------------------------------------- cell


def _stats_for(series: dict[str, np.ndarray], grid: Grid, include_oos: bool, boot: bool,
               names) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in names:
        out[name] = {seg: series_stats(series[name][idx], grid.hold, grid.step, boot)
                     for seg, idx in grid.segments(include_oos).items()}
    return out


def _present(paths) -> list:
    return [q for q in paths if q is not None]


def _risk_view(grid: Grid, sides_pos: np.ndarray, L: np.ndarray, nL: np.ndarray, cost: float,
               include_oos: bool, bars_per_year: int) -> dict:
    """The long basket held at non-overlapping dates (cash when empty) against the equal-
    weight universe and the benchmark held over the same windows.

    Returns, volatility and Sharpe come from the period returns; the maximum drawdown is
    measured on the daily marks inside the periods (marking only at the period ends hid
    most of it).
    """
    k = math.ceil(grid.hold / grid.step)
    ppy = bars_per_year / (k * grid.step)
    strat = np.where(nL > 0, np.nan_to_num(L) - cost, 0.0)
    seg_idx = {"ALL": np.arange(len(grid.pos)), **({"OOS": grid.oos_idx} if include_oos else {})}
    ok_r = np.isfinite(grid.R)
    flat = np.ones(grid.hold)
    out = {}
    for name, idx in seg_idx.items():
        sel = idx[::k]
        ok = np.isfinite(grid.U[sel]) & np.isfinite(grid.B[sel])
        sel = sel[ok]
        basket_paths, universe_paths, bench_paths = [], [], []
        for p in sel:
            held = (sides_pos[p] == 1) & ok_r[p]
            path = grid.window_path(int(p), held)
            basket_paths.append(flat if path is None else path - cost)
            universe_paths.append(grid.window_path(int(p), ok_r[p]))
            bench_paths.append(grid.window_path(int(p), None))
        out[name] = {
            "basket": risk_stats(strat[sel], ppy, basket_paths),
            "universe": risk_stats(grid.U[sel], ppy, _present(universe_paths)),
            "benchmark": risk_stats(grid.B[sel], ppy, _present(bench_paths)),
            "in_market": float((nL[sel] > 0).mean()) if len(sel) else None,
            "drawdown_marks": "daily",
        }
    return out


def _mean_held(counts: np.ndarray) -> float | None:
    """Average basket size over the dates on which a basket exists (an empty basket is
    cash, not a basket of size zero)."""
    held = counts[counts > 0]
    return float(held.mean()) if len(held) else None


def evaluate_cell(grid: Grid, sides_pos: np.ndarray, cost_bps: float, bars_per_year: int, *,
                  include_oos: bool = True, boot: bool = True, with_stop: bool = True) -> dict:
    """Everything measured for one (family, rule, horizon): series statistics in both
    execution modes, the 2x-cost out-of-sample means, basket sizes and the risk view.
    ``with_stop=False`` skips the (slow) protective-stop mode, which the robustness
    recomputation does not need."""
    cost = cost_bps / 1e4
    L, S, nL, nS = grid.basket_means(sides_pos)
    ser = return_series(L, S, grid.U, cost)
    all_names = ("long_excess", "long_abs", "short_pnl", "short_excess", "spread")
    result = {"series": _stats_for(ser, grid, include_oos, boot, all_names)}
    if with_stop:
        atr_mult = STOP_ATR[grid.horizon]
        Ls, Ss, _, _ = grid.basket_means(sides_pos, grid.stop_returns(sides_pos, atr_mult))
        result["stop"] = _stats_for(return_series(Ls, Ss, grid.U, cost), grid, include_oos,
                                    False, all_names)
    if include_oos:
        # 2x costs: every series subtracts its costs once more (spread pays two legs)
        extra = {"spread": 2.0 * cost_bps}
        result["oos_mean_2x_bps"] = {
            n: (None if result["series"][n]["OOS"]["mean_bps"] is None
                else result["series"][n]["OOS"]["mean_bps"] - extra.get(n, cost_bps))
            for n in all_names}
    result["baskets"] = {
        seg: {"long": _mean_held(nL[idx]), "short": _mean_held(nS[idx])}
        for seg, idx in grid.segments(include_oos).items()}
    result["risk"] = _risk_view(grid, sides_pos, L, nL, cost, include_oos, bars_per_year)
    return result


def entries_for(family: str, rule_id: str, horizon: str, cell: dict, include_oos: bool) -> list[dict]:
    """The flat result entries (one per side) of a cell, with verdicts when OOS is in."""
    out = []
    variant = bool(RULES[rule_id].variant)
    for side in ("long", "short"):
        names = SIDE_SERIES[side]
        entry = {"family": family, "rule": rule_id, "horizon": horizon, "side": side,
                 "judged": JUDGED[side],
                 "series": {n: cell["series"][n] for n in names},
                 "stop": {n: cell["stop"][n] for n in names},
                 "baskets": {seg: v[side] for seg, v in cell["baskets"].items()}}
        if include_oos:
            entry.update(judge_side(side, cell["series"], cell["oos_mean_2x_bps"], variant,
                                    cell["baskets"].get("ALL", {}).get(side)))
            entry["oos_mean_2x_bps"] = {n: cell["oos_mean_2x_bps"][n] for n in names}
            entry["phase_verdicts"] = phase_verdicts(
                side, cell["series"], cell["oos_mean_2x_bps"], variant,
                cell["baskets"].get("ALL", {}).get(side))
            # what the first version of the study (Newey-West t, no basket rule) would say
            entry["legacy_verdict"] = judge_side(
                side, cell["series"], cell["oos_mean_2x_bps"], variant, None, NEWEY_WEST)["verdict"]
            entry["robust"] = None
            entry["robustness"] = None
            entry["caveat"] = None
        out.append(entry)
    return out


# --------------------------------------------------------------------------- robustness

_DOWNGRADE = {"validated": "experimental", "experimental": "unproven"}
_NOUN = {"stocks_us": ("Aktien", "stocks"), "stocks_dax": ("Aktien", "stocks"),
         "etfs": ("ETFs", "ETFs"), "crypto": ("Coins", "coins")}


def top_winners(frames_panel: Panel, k: int = ROBUST_K) -> list[str]:
    """The ``k`` members with the highest total return over their data in the panel.

    Fixed in advance (k = 3): today's list contains the names that later became mega
    caps, so a momentum rule that buys winners is flattered by exactly these.
    """
    totals = {}
    for sym in frames_panel.members:
        c = frames_panel.close[sym].dropna()
        if len(c) >= 2 and c.iloc[0] > 0:
            totals[sym] = float(c.iloc[-1] / c.iloc[0] - 1.0)
    return sorted(totals, key=lambda s: (-totals[s], s))[:k]


def caveat_texts(family: str, removed: list[str], t: float) -> dict:
    """The mandatory bilingual caveat of a cell that does not survive the removal."""
    de_noun, en_noun = _NOUN.get(family, ("Werte", "names"))
    names = ", ".join(removed)
    t_de, t_en = _f(t, 1), f"{t:.1f}"
    k = len(removed)
    return {
        "de": f"Ergebnis hängt an {k} {de_noun} ({names}), die in der Liste stehen, weil sie "
              f"gestiegen sind; ohne sie t = {t_de}",
        "en": f"The result depends on {k} {en_noun} ({names}) that are in the list because they "
              f"went up; without them t = {t_en}",
        "short_de": f"hängt an {names}", "short_en": f"depends on {names}"}


def robustness_check(fam: Family, rule_id: str, horizon: str, side: str,
                     as_of: str) -> dict:
    """Recompute the judged series without the ``ROBUST_K`` top-return members.

    The universe mean, the rank cut-offs and the baskets are all rebuilt on the reduced
    list (as a reader would do who says "without NVDA"), with the same costs and split
    rule.  ``kept`` is True when the pooled t of the reduced series is at least
    ``ROBUST_MIN_T``.
    """
    removed = top_winners(fam.panel)
    members = [m for m in fam.panel.members if m not in removed]
    panel = Panel(fam.frames, members, fam.info.benchmark, fam.params)
    reduced = Family(fam.name, fam.info, fam.frames, panel, fam.cost_bps, fam.params)
    grid = Grid(panel, horizon)
    sides = rule_sides_at(reduced, grid, rule_id, None, as_of)
    cell = evaluate_cell(grid, sides, fam.cost_bps, fam.params.bars_per_year, boot=False,
                         with_stop=False)
    judged = JUDGED[side]
    ser = cell["series"][judged]
    pooled = _t(ser["ALL"])
    return {
        "k": ROBUST_K, "removed": removed, "judged": judged, "min_t": ROBUST_MIN_T,
        "IS": {"mean_bps": ser["IS"]["mean_bps"], "t": _t(ser["IS"]), "n": ser["IS"]["n"]},
        "OOS": {"mean_bps": ser["OOS"]["mean_bps"], "t": _t(ser["OOS"]), "n": ser["OOS"]["n"]},
        "pooled_mean_bps": ser["ALL"]["mean_bps"], "pooled_t": pooled,
        "pooled_t_nw": float(ser["ALL"].get("nw_t") or 0.0),
        "kept": bool(pooled >= ROBUST_MIN_T),
    }


def apply_robustness(entry: dict, check: dict, family: str) -> None:
    """Attach the check and, when it fails, downgrade the label one step with its caveat.

    A cell that passes only under the first version's statistic (``legacy_verdict``) is
    checked too, for the record (``info_only``), but is already ``unproven``: nothing to
    downgrade and no caveat.
    """
    entry["robustness"] = check
    if entry["verdict"] not in _DOWNGRADE:
        check["info_only"] = True
        return
    entry["verdict_pre_robustness"] = entry["verdict"]
    if check["kept"]:
        entry["robust"] = True
        return
    entry["robust"] = False
    entry["verdict"] = _DOWNGRADE[entry["verdict"]]
    entry["strong"] = False
    entry["caveat"] = caveat_texts(family, check["removed"], check["pooled_t"])


# --------------------------------------------------------------------------- data


def _cached(cache_dir: Path | None, name: str, build):
    return _cached_download(cache_dir, name, build)[0]


def _cached_download(cache_dir: Path | None, name: str, build) -> tuple[object, datetime]:
    """``(value, time the value was downloaded)``.

    The download time is stored next to the cache file (the file's own modification time
    for a cache written before this existed).  The study cuts every series at the last bar
    that was settled AT THAT TIME: a cache taken during a session holds the partial bar of
    that day, which must not become the last "close" of the study.
    """
    if cache_dir is not None:
        path = Path(cache_dir) / name
        if path.exists():
            meta = path.with_name(name + ".meta.json")
            try:
                stamp = datetime.fromisoformat(json.loads(meta.read_text())["downloaded_at"])
            except (OSError, ValueError, KeyError):
                stamp = datetime.fromtimestamp(path.stat().st_mtime, UTC)
            return pickle.loads(path.read_bytes()), stamp
    value = build()
    stamp = datetime.now(UTC)
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        path = Path(cache_dir) / name
        path.write_bytes(pickle.dumps(value))
        path.with_name(name + ".meta.json").write_text(
            json.dumps({"downloaded_at": stamp.isoformat()}), encoding="utf-8")
    return value, stamp


@dataclass
class Family:
    name: str
    info: object
    frames: dict[str, pd.DataFrame]
    panel: Panel
    cost_bps: float
    params: SignalParams
    downloaded_at: str = ""


def load_family(spec: FamilySpec, as_of: str, cache_dir: Path | None,
                now: datetime | None = None) -> Family:
    info = universe_info(spec.universe, list(spec.tickers) if spec.tickers else None)
    vix = ["^VIX"] if info.use_vix else []
    wanted = list(dict.fromkeys([*info.members, info.benchmark, *vix]))
    raw, downloaded = _cached_download(cache_dir, f"{spec.name}_{as_of}.pkl",
                                       lambda: load_bulk_ohlcv(wanted, as_of, HISTORY_DAYS))
    frames = settled(raw, now or downloaded)
    params = SignalParams(info.month_bars, info.bars_per_year)
    members = [m for m in info.members if m in frames]
    panel = Panel(frames, members, info.benchmark, params)
    return Family(spec.name, info, frames, panel, COSTS_BPS[spec.name], params,
                  downloaded.strftime("%Y-%m-%dT%H:%M:%SZ"))


def composite_sides(fam: Family, grid: Grid, cache_dir: Path | None, as_of: str) -> np.ndarray:
    """T0 sides at the grid's decision dates (the slow part, so it is cached)."""
    name = f"T0_{fam.name}_{grid.horizon}_{as_of}_{len(grid.pos)}.pkl"

    def build():
        sides, _ = rules.composite_panel(
            fam.frames, fam.panel.members, fam.info.benchmark, grid.dates, fam.params,
            use_vix=fam.info.use_vix)
        return sides.reindex(columns=fam.panel.members).to_numpy()
    return _cached(cache_dir, name, build)


def rule_sides_at(fam: Family, grid: Grid, rule_id: str, cache_dir: Path | None,
                  as_of: str) -> np.ndarray:
    if RULES[rule_id].base == "T0":
        return composite_sides(fam, grid, cache_dir, as_of)
    sides, _ = rules.rule_sides(rule_id, fam.panel)
    return sides.to_numpy()[grid.pos]


# --------------------------------------------------------------------------- study


@dataclass
class Study:
    as_of: str
    include_oos: bool
    families: dict[str, dict]
    entries: list[dict]
    risk: list[dict]
    rule_ids: list[str]


def run_study(as_of: str, cache_dir: str | Path | None = None, families: list[str] | None = None,
              rule_ids: list[str] | None = None, include_oos: bool = True,
              boot: bool = True, loaded: dict[str, Family] | None = None) -> Study:
    """Evaluate every (rule, horizon) of ``rule_ids`` (default: all registered rules,
    pre-registered and variants) on each family.  ``include_oos=False`` computes the
    in-sample part only, so the exploration phase cannot see out-of-sample numbers."""
    cache = Path(cache_dir) if cache_dir else None
    specs = family_specs()
    names = families or list(specs)
    rule_ids = rule_ids or list(RULES)
    fam_info: dict[str, dict] = {}
    entries: list[dict] = []
    risk: list[dict] = []
    for name in names:
        fam = (loaded or {}).get(name) or load_family(specs[name], as_of, cache)
        if loaded is not None:
            loaded[name] = fam
        panel = fam.panel
        logger.info("%s: %d symbols, bars %s .. %s", name, len(panel.members),
                    panel.index[0].date(), panel.index[-1].date())
        grids = {h: Grid(panel, h) for h in (MEDIUM, LONG)}
        fam_info[name] = {
            "universe": fam.info.name, "benchmark": fam.info.benchmark,
            "n_symbols": len(panel.members), "symbols_missing": sorted(
                set(fam.info.members) - set(panel.members)),
            "first_bar": str(panel.index[0].date()), "last_bar": str(panel.index[-1].date()),
            "downloaded_at": fam.downloaded_at,
            "month_bars": fam.params.month_bars, "bars_per_year": fam.params.bars_per_year,
            "cost_bps": fam.cost_bps,
            "horizons": {h: g.info() for h, g in grids.items()},
            "drift": {h: {seg: series_stats(g.U[idx], g.hold, g.step, boot=False)
                          for seg, idx in g.segments(include_oos).items()}
                      for h, g in grids.items()},
        }
        for rid in rule_ids:
            for h in RULES[rid].horizons:
                grid = grids[h]
                if len(grid.pos) < 10:
                    logger.warning("%s %s %s: only %d decision dates, skipped",
                                   name, rid, h, len(grid.pos))
                    continue
                sides_pos = rule_sides_at(fam, grid, rid, cache, as_of)
                cell = evaluate_cell(grid, sides_pos, fam.cost_bps, fam.params.bars_per_year,
                                     include_oos=include_oos, boot=boot)
                new = entries_for(name, rid, h, cell, include_oos)
                for entry in new:
                    if include_oos and (entry["verdict"] in _DOWNGRADE
                                        or entry["legacy_verdict"] in _DOWNGRADE):
                        apply_robustness(entry, robustness_check(
                            fam, rid, h, entry["side"], as_of), name)
                entries.extend(new)
                risk.append({"family": name, "rule": rid, "horizon": h, **cell["risk"]})
                logger.info("%s %s %s done", name, rid, h)
    return Study(as_of, include_oos, fam_info, entries, risk, rule_ids)


# --------------------------------------------------------------------------- output


def _clean(value):
    """JSON-safe: NaN/inf become None, numpy scalars become Python numbers."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, 4) if math.isfinite(value) else None
    return value


def _fingerprint(entry: dict) -> tuple:
    """What an evaluation measured: family, horizon, side and every series' numbers.  Two
    rules that select the same instruments (V3's Long side is T3's) have the same one."""
    parts = []
    for name in SIDE_SERIES[entry["side"]][:2]:      # the side's own series, not the spread
        for seg in ("IS", "OOS", "ALL"):
            st = entry["series"][name].get(seg) or {}
            parts.append((name, seg, st.get("n"), st.get("mean_bps"), st.get("t_nonoverlap")))
    return (entry["family"], entry["horizon"], entry["side"], tuple(parts))


def count_evaluations(study: Study) -> dict:
    pre = [r for r in study.rule_ids if not RULES[r].variant]
    var = [r for r in study.rule_ids if RULES[r].variant]
    n_fam = len(study.families)

    def cells(ids):
        return sum(len(RULES[r].horizons) for r in ids)
    distinct = len({_fingerprint(e) for e in study.entries})
    return {
        "families": n_fam,
        "rule_horizons_preregistered": cells(pre),
        "variants": len(var),
        "rule_horizons_variants": cells(var),
        "evaluations_preregistered": 2 * n_fam * cells(pre),
        "evaluations_variants": 2 * n_fam * cells(var),
        "evaluations_total": 2 * n_fam * (cells(pre) + cells(var)),
        "evaluations_distinct": distinct if study.entries else None,
        "evaluations_duplicate": (len(study.entries) - distinct) if study.entries else None,
    }


def snapshot_dict(study: Study, generated_at: str | None = None) -> dict:
    rule_meta = {}
    for rid in study.rule_ids:
        r = RULES[rid]
        rule_meta[rid] = {
            "base": r.base, "name_en": r.name_en, "name_de": r.name_de, "desc_en": r.desc_en,
            "desc_de": r.desc_de, "horizons": list(r.horizons), "variant": r.variant,
            "params": rules.rule_params(rid), "note_en": r.note_en, "note_de": r.note_de}
    return _clean({
        "version": SNAPSHOT_VERSION,
        "study_date": study.as_of,
        "generated_at": generated_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "is_fraction": IS_FRACTION,
        "purged": True,
        "costs_bps": COSTS_BPS,
        "stop_atr": STOP_ATR,
        "thresholds": {
            "validated_t": VALIDATED_T, "experimental_t_is": EXPERIMENTAL_T_IS,
            "experimental_t_oos": EXPERIMENTAL_T_OOS, "strong_t": STRONG_T,
            "validated_min_oos_dates": VALIDATED_MIN_OOS_DATES,
            "experimental_min_oos_dates": EXPERIMENTAL_MIN_OOS_DATES,
            "min_basket": MIN_BASKET, "robust_k": ROBUST_K, "robust_min_t": ROBUST_MIN_T,
            "statistic": "median over all phases of the plain t of the non-overlapping "
                         "subsample (every phase t is stored as phase_t); the Newey-West t is "
                         "information only (optimistic)"},
        "counts": count_evaluations(study),
        "rules": rule_meta,
        "variants": [{"id": k, **v} for k, v in rule_meta.items() if v["variant"]],
        "families": study.families,
        "results": study.entries,
        "risk": study.risk,
    })


def write_snapshot(study: Study, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot_dict(study), ensure_ascii=False, separators=(",", ":")),
                      encoding="utf-8")
    return target


def load_snapshot(path: str | Path | None = None) -> dict | None:
    """The persisted study, or None when there is none (or it cannot be read)."""
    try:
        return json.loads(Path(path or SNAPSHOT_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("evidence snapshot unavailable: %s", exc)
        return None


# ---- German report

FAMILY_DE = {"stocks_us": "US-Aktien", "stocks_dax": "DAX-Aktien", "etfs": "ETFs (Welt + Sektoren)",
             "crypto": "Kryptowährungen"}
VERDICT_DE = {k: v[1][0].lower() + v[1][1:] for k, v in LABEL_NAMES.items()
              if k != "not_studied"}
HORIZON_DE = {"medium": "mittel (1 Monat)", "long": "lang (6 Monate)"}
SIDE_DE = {"long": "Long", "short": "Short"}
LABELS = ("validated", "experimental", "relative", "beta", "unproven")


def _f(x, nd=1) -> str:
    return fmt_num(x, nd, "de")


def _pct(x, nd=1) -> str:
    return "-" if x is None else _f(x * 100, nd) + " %"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def label_counts(results: list[dict]) -> dict[str, int]:
    """How many (family, rule, horizon, side) cells carry each label."""
    return {k: sum(1 for r in results if r.get("verdict") == k) for k in LABELS}


def chance_rates(results: list[dict], n_sim: int = 100_000, seed: int = SEED) -> dict:
    """Expected number of passes by chance alone, from the study's own sample sizes.

    Under the null the plain t of ``n`` independent observations follows a t distribution
    with ``n - 1`` degrees of freedom; draws are seeded.  For every DISTINCT evaluation the
    probability of ``validated`` is P(t_IS >= 2) * P(t_OOS >= 2) and of validated-or-
    experimental P(t_IS >= 2) * P(t_OOS >= 1); the cost and robustness conditions are left
    out, so both are upper bounds.  The two segments are taken as independent.
    """
    rng = np.random.default_rng(seed)
    cache: dict[tuple[int, float], float] = {}

    def tail(n: int, thr: float) -> float:
        key = (n, thr)
        if key not in cache:
            cache[key] = (float((rng.standard_t(n - 1, size=n_sim) >= thr).mean())
                          if n >= 2 else 0.0)
        return cache[key]

    seen, validated, either = set(), 0.0, 0.0
    per_segment = []
    for r in results:
        fp = _fingerprint(r)
        if fp in seen:
            continue
        seen.add(fp)
        ser = r["series"][r["judged"]]
        n_is, n_oos = ser["IS"].get("n_nonoverlap", 0), ser["OOS"].get("n_nonoverlap", 0)
        validated += tail(n_is, VALIDATED_T) * tail(n_oos, VALIDATED_T)
        either += tail(n_is, EXPERIMENTAL_T_IS) * tail(n_oos, EXPERIMENTAL_T_OOS)
        per_segment.append((n_is, n_oos))
    typical = {}
    for h in ("medium", "long"):
        ns = [(r["series"][r["judged"]]["IS"].get("n_nonoverlap", 0),
               r["series"][r["judged"]]["OOS"].get("n_nonoverlap", 0))
              for r in results if r["horizon"] == h]
        if ns:
            n_is, n_oos = ns[len(ns) // 2]
            typical[h] = {"n_is": n_is, "n_oos": n_oos, "p_is": tail(n_is, 2.0),
                          "p_oos": tail(n_oos, 2.0)}
    return {"distinct": len(seen), "validated": validated, "either": either, "typical": typical}


def _label_of(r: dict | None) -> str:
    return "-" if r is None else VERDICT_DE[r["verdict"]]


def _before_after(s: dict, before: dict | None) -> list[str]:
    """The table of every cell whose label changed against the pre-review snapshot."""
    out: list[str] = []
    if not before:
        return out
    old = {(r["family"], r["rule"], r["horizon"], r["side"]): r for r in before["results"]}
    rows = []
    for r in s["results"]:
        key = (r["family"], r["rule"], r["horizon"], r["side"])
        o = old.get(key)
        if o is None or o["verdict"] == r["verdict"]:
            continue
        j = r["judged"]
        so, sn = o["series"][o["judged"]], r["series"][j]
        rob = r.get("robustness")
        pre = r.get("verdict_pre_robustness") or r["verdict"]
        rows.append([
            f"{FAMILY_DE[r['family']]}, {r['rule']}, {HORIZON_DE[r['horizon']]}, "
            f"{SIDE_DE[r['side']]}",
            VERDICT_DE[o["verdict"]], VERDICT_DE[r["verdict"]],
            f"{_f(so['IS']['nw_t'])} / {_f(so['OOS']['nw_t'])} / {_f(o.get('pooled_t'))}",
            f"{_f(sn['IS']['t_nonoverlap'])} / {_f(sn['OOS']['t_nonoverlap'])} / "
            f"{_f(r.get('pooled_t'))}",
            ("ohne " + ", ".join(rob["removed"]) + f": gepooltes t = {_f(rob['pooled_t'])} "
             + ("(hätte bestanden)" if rob["kept"] else "(wäre durchgefallen)")
             if rob.get("info_only") else
             "ohne " + ", ".join(rob["removed"]) + f": gepooltes t = {_f(rob['pooled_t'])} "
             + ("(bleibt)" if rob["kept"] else "(fällt durch)")) if rob else
            ("Korb < 3 Werte" if r.get("basket_too_small") and pre != r["verdict"]
             else "-")])
    if not rows:
        return ["Kein Urteil hat sich gegenüber dem Lauf vor der Prüfung geändert."]
    out.append(_table(["Zelle", "Urteil vorher", "Urteil nachher",
                       "t vorher (Newey-West: IS / OOS / gepoolt)",
                       "t nachher (einfach, nicht überlappend: IS / OOS / gepoolt)",
                       "Robustheitsprüfung"], rows))
    return out


def _phase_section(s: dict) -> list[str]:
    """The German section on the phase dependence of the verdicts (review finding M5)."""
    out = ["## Phasenabhängigkeit des Urteils\n"]
    out.append(
        "Bei Halbjahres- und Monatsfenstern überlappen sich benachbarte Termine; das einfache t "
        "nimmt deshalb nur jeden k-ten Termin. Welcher Termin der erste ist (die Phase), ist "
        "willkürlich, und das t kann sich dadurch stark ändern. Beurteilt wird darum der Median "
        "des t über alle k Phasen (In-Sample, Out-of-Sample und gepoolt); diese Regel wurde "
        "festgelegt, nachdem eine unabhängige Prüfung die Phasenabhängigkeit gezeigt hatte, und "
        "die Schwellen blieben unverändert. Unten die Zellen, die in mindestens einer einzelnen "
        "Phase \"schwach belegt\" oder besser erreicht hätten (vor der Robustheitsprüfung; reine "
        "Transparenz, keine Beurteilung):\n")
    rows = []
    for r in s["results"]:
        pv = r.get("phase_verdicts") or []
        hit = [i for i, v in enumerate(pv) if v in ("validated", "experimental")]
        if not hit:
            continue
        ser = r["series"][r["judged"]]
        rows.append([
            f"{FAMILY_DE[r['family']]}, {r['rule']}, {HORIZON_DE[r['horizon']]}, "
            f"{SIDE_DE[r['side']]}",
            f"in {len(hit)} von {len(pv)} Phasen wäre die Zelle schwach belegt oder besser",
            "/".join(_f(t) for t in ser["IS"].get("phase_t", [])),
            "/".join(_f(t) for t in ser["OOS"].get("phase_t", [])),
            f"{_f(ser['IS']['t_nonoverlap'])} / {_f(ser['OOS']['t_nonoverlap'])}",
            VERDICT_DE[r["verdict"]]])
    if rows:
        out.append(_table(["Zelle", "Phasen", "IS-t je Phase", "OOS-t je Phase",
                           "Median IS / OOS", "Urteil (Median)"], rows))
    else:
        out.append("Keine Zelle hätte in einer einzelnen Phase \"schwach belegt\" oder besser "
                   "erreicht.")
    out.append("")
    return out


def write_report(snapshot: dict, path: str | Path, before: dict | None = None) -> Path:
    from tradingagents.scanner.horizons import STALE_AFTER_DAYS

    s = snapshot
    res = s["results"]
    counts = s["counts"]
    th = s["thresholds"]
    best = [r for r in res if r["verdict"] in ("validated", "experimental")]
    rel = [r for r in res if r["verdict"] == "relative"]
    beta = [r for r in res if r["verdict"] == "beta"]
    lc = label_counts(res)
    out: list[str] = []
    w = out.append
    w("# Mittel- und Langfrist: Was ist wirklich belegt?")
    w(f"\nStudie vom {s['study_date']}. Alle Zahlen stammen aus einem echten Testlauf auf "
      "Yahoo-Kursdaten. Verworfene Regeln und Varianten stehen in den Tabellen mit drin.\n")
    w("**Dieser Bericht wurde nach einer unabhängigen Prüfung korrigiert.** Die erste Fassung "
      "stufte genau eine Zelle als \"belegt\" ein (US-Aktien, Momentum 12-1, langfristig, Long). "
      "Die Prüfung reproduzierte jede Zahl, zeigte aber, dass dieser Befund nicht trägt: er hing "
      "an drei Aktien, die mit Rückschau in der Liste stehen, sein t-Wert ignorierte die "
      "Überlappung der Fenster, und der mitgelieferte Stop zerstörte den gemessenen Vorteil. "
      "Beurteilt wird jetzt (1) mit dem Median des einfachen t auf nicht überlappenden Terminen über alle Phasen (siehe Abschnitt Phasenabhängigkeit), (2) nur "
      "mit Körben von im Schnitt mindestens drei Werten und (3) nur, wenn das Ergebnis auch "
      "ohne die drei Werte mit der höchsten Gesamtrendite besteht. Unten steht die Gegenüberstellung "
      "vorher / nachher.\n")
    w("## Kurzfassung\n")
    w(f"Einstufungen der {len(res)} Regel-Seiten-Zellen (Mehrfachnennungen derselben Messung "
      f"inbegriffen): {VERDICT_DE['validated']} {lc['validated']}, {VERDICT_DE['experimental']} "
      f"{lc['experimental']}, {VERDICT_DE['relative']} {lc['relative']}, {VERDICT_DE['beta']} "
      f"{lc['beta']}, {VERDICT_DE['unproven']} {lc['unproven']}.\n")
    if not best:
        w("**Keine der getesteten Regeln hat die korrigierte Prüfung bestanden.** Das ist das "
          "Ergebnis, kein Fehler: Die Anlageideen des Programms für Monate und Halbjahre sind "
          "deshalb als Beobachtungsliste gekennzeichnet, nicht als Empfehlung.")
    else:
        w(f"Von {counts['evaluations_distinct']} verschiedenen Prüfungen bestehen "
          f"{len(best)} Zellen den vorab festgelegten Test (siehe Tabellen). Wegen der vielen Tests "
          "kann ein Teil davon Zufall sein; Details unten.")
        for r in best:
            w(f"- {FAMILY_DE[r['family']]}, {s['rules'][r['rule']]['name_de']}, "
              f"{HORIZON_DE[r['horizon']]}, {SIDE_DE[r['side']]}: {VERDICT_DE[r['verdict']]}"
              f"{' (stark)' if r['strong'] else ''}"
              + (f" – Achtung: {r['caveat']['de']}" if r.get("caveat") else ""))
    if rel:
        w(f"\nNur **relativ** (der Wert hinkt dem Markt hinterher, fällt aber nicht belegbar "
          f"absolut; ein Grund zum Meiden/Untergewichten, kein Grund für ein Short): {len(rel)} "
          "Fälle.")
    if beta:
        w(f"\nNur **Marktrendite** (Gewinn in absoluten Zahlen, aber nicht mehr als der "
          f"Durchschnitt aller Werte verdient hat): {len(beta)} Fälle.")
    w("\n## Korrektur nach der unabhängigen Prüfung\n")
    w("Gegenüberstellung aller Zellen, deren Urteil sich geändert hat:\n")
    out.extend(_before_after(s, before))
    w("\nWas geändert wurde und warum:\n")
    for line in [
        "**Drei Aktien tragen den Befund.** Die Listen enthalten die heutigen Mitglieder; ein "
        "Momentum-Verfahren kauft verlässlich jene, die später zu den größten Werten wurden. Ohne "
        "NVDA, AVGO und AMD fiel das gepoolte Newey-West-t der US-Zelle von 2,97 auf 1,84, ohne die "
        "fünf besten auf 1,20 (Zahlen der Prüfung); NVDA allein lieferte 36 % des Bruttovorsprungs, NVDA, AMD, AVGO und "
        "TSLA zusammen 85 %. In den Jahren 2018 und 2022 lag der Vorsprung bei etwa -200 bzw. "
        f"-266 Basispunkten. Jetzt gilt für jede Zelle, die belegt oder schwach belegt wäre: "
        f"die Reihe wird ohne die {th['robust_k']} Werte mit der höchsten Gesamtrendite im "
        f"Datenfenster der Familie neu berechnet (vorab festgelegt, k = {th['robust_k']}, fest). "
        f"Besteht das gepoolte t dann nicht mehr mit mindestens {_f(th['robust_min_t'])}, wird "
        "das Urteil um eine Stufe herabgesetzt und trägt eine Warnung mit den Namen.",
        "**Das Newey-West-t war zu optimistisch.** Bei Haltedauern, die ein Mehrfaches des "
        "Terminabstands sind, setzt die verwendete Verzögerung die Überlappung nicht voll an "
        "(bei sechs überlappenden Fenstern ein Varianzfaktor von 4,06 statt 6, das t ist um "
        "etwa das 1,22-fache zu hoch). In einer Monte-Carlo-Prüfung ohne echten Effekt erreichten "
        "7,2 % (lang, 94 Termine), 9,6 % (lang, 44 Termine) und 5,0 % (mittel) ein t von mindestens "
        "2 – statt der nominalen 2,3 %. Beurteilt wird deshalb jetzt mit dem **einfachen t der "
        "nicht überlappenden Termine** (jeder k-te Termin; maßgeblich ist der **Median über alle "
        "k möglichen Phasen**, siehe Abschnitt Phasenabhängigkeit), getrennt für In-Sample, "
        "Out-of-Sample und gepoolt. Das Newey-West-t steht weiter in den Daten, als "
        "optimistisch gekennzeichnet; zusätzlich stehen im Datensatz die t-Werte aller Phasen "
        "(Startversätze).",
        "**Das Urteil hing an einer willkürlichen Phase.** Die erste korrigierte Fassung "
        "beurteilte nur eine feste Phase der nicht überlappenden Termine (die am letzten Termin "
        "endende). Eine weitere unabhängige Prüfung zeigte, dass sich bei derselben Zelle (US-Aktien, "
        "Momentum, lang, Long) das t je nach Phase stark ändert und die gewählte Phase die mit dem "
        "niedrigsten In-Sample-t war. Diese Regel wurde deshalb nach jener Prüfung festgelegt: "
        "beurteilt wird der **Median des einfachen t über alle Phasen**, für In-Sample, "
        "Out-of-Sample, gepoolt und ebenso in der Robustheitsprüfung; die Schwellen blieben "
        "unverändert. Der Mittelwert, der neben dem t genannt wird, stammt aus derselben "
        "Phase (bzw. dem Mittel der beiden mittleren Phasen).",
        "**Der Stop zerstörte den Vorteil.** Im eigenen Stop-Modus der Studie (siehe unten) "
        "wurde aus +597 Basispunkten außerhalb der Stichprobe für die frühere US-Zelle ein Minus. "
        "Für die Zeit-Ausstiegs-Regeln (mittel, lang) schlägt das Programm deshalb keinen Stop "
        "mehr vor, und die Positionsgröße ist gleichgewichtet im gemessenen Korb statt aus "
        "einem Stopabstand abgeleitet.",
        f"**Mindestgröße des Korbs.** Ein Urteil über \"nicht belegt\" hinaus verlangt einen "
        f"durchschnittlichen Korb von mindestens {th['min_basket']} Werten; die Short-Seite der "
        "Variante V5 bei ETFs stützte sich im Schnitt auf etwa einen ETF je Korb.",
        "**Verschiedene Prüfungen.** Die Varianten V3 und T3 haben dieselbe Long-Seite (8 "
        "identische Zellen). Gezählt werden deshalb verschiedene Messungen, nicht Zellen "
        f"({counts['evaluations_distinct']} verschiedene von {counts['evaluations_total']} "
        "Zellen). Eine Variante bekommt nie das Siegel \"stark\".",
        "**Letzter abgeschlossener Kurs.** Der Download fand während einer Börsensitzung statt "
        "und enthielt Teilkurse des Tages. Die Studie schneidet jede Reihe beim letzten Kurs, der "
        "zum Zeitpunkt des Downloads abgeschlossen war (der Zeitpunkt wird mit dem Zwischenspeicher "
        "abgelegt).",
        "**Rückgang auf Tageskursen.** Die Risikosicht maß den größten Rückgang nur an den "
        "Halbjahrespunkten (etwa -6 % statt -10 %; auf Tageskursen desselben Pfads -30,7 % "
        "statt -34,5 %). Er wird jetzt auf täglichen Kursen gemessen.",
    ]:
        w(f"- {line}")
    w("\n## Was wurde getestet\n")
    w("Regeln, für die es in der Literatur einen Grund gibt, und die jeweils ohne Blick in die "
      "Zukunft berechnet werden (Signal am Schlusskurs von Tag t, Einstieg zur Eröffnung von "
      "t+1, Ausstieg nach einem Monat bzw. sechs Monaten):\n")
    rows = [[k, v["name_de"], v["desc_de"], ", ".join(HORIZON_DE[h] for h in v["horizons"]),
             "Variante" if v["variant"] else "vorab festgelegt"]
            for k, v in s["rules"].items()]
    w(_table(["Kürzel", "Regel", "Beschreibung", "Horizont", "Art"], rows))
    w("\nVergleichsmaßstab ist immer der Durchschnitt **aller** Werte des Universums im selben "
      "Zeitraum (\"Drift\"): Wer nur mit dem steigenden Markt mitschwimmt, hat keinen Vorteil "
      "gezeigt. Gerechnet wird nach Kosten (10 Basispunkte je Hin- und Rückweg bei Aktien/ETFs, "
      "30 bei Krypto); der Test wird zusätzlich mit doppelten Kosten geprüft.\n")
    out.extend(_phase_section(s))
    w("## Daten\n")
    rows = []
    for name, f in s["families"].items():
        h = f["horizons"]
        rows.append([FAMILY_DE[name], str(f["n_symbols"]), f"{f['first_bar']} bis {f['last_bar']}",
                     f"{h['medium']['first_decision']} bis {h['medium']['last_decision']}",
                     f"{h['medium']['n_dates']} / {h['long']['n_dates']}",
                     f"{h['medium']['oos_first']} bis {h['medium']['oos_last']}"])
    w(_table(["Familie", "Werte", "Kursdaten", "Entscheidungstage", "Termine mittel / lang",
              "Testzeitraum (OOS, mittel)"], rows))
    w("\nAufteilung: die ersten 70 % der Entscheidungstage dienen zum Auswählen (In-Sample, IS), "
      "die letzten 30 % werden genau einmal zum Prüfen verwendet (Out-of-Sample, OOS). IS-Tage, "
      "deren Haltefenster in den OOS-Zeitraum hineinreicht, werden entfernt (Purge). Die Kursdaten "
      "enden beim letzten Tag, der zum Zeitpunkt des Downloads abgeschlossen war.\n")
    w("## Ergebnisse je Familie\n")
    w("Spalten: Termine (n) und mittlere Rendite je Termin (alle, auch überlappende Termine) in Basispunkten (1 bp = 0,01 %) der "
      "**beurteilten Reihe** (Long: Vorsprung vor dem Durchschnitt aller Werte; Short: "
      "tatsächlicher Gewinn des Shorts), jeweils nach Kosten. t = Median des einfachen t der nicht "
      "überlappenden Termine über alle Phasen (so wird beurteilt; |t| ≥ 2 gilt als auffällig), \"NW-t\" = "
      "Newey-West-t über alle Termine (nur zur Information, optimistisch), \"n ohne\" = Zahl der "
      "nicht überlappenden Termine im OOS-Teil.\n")
    for name in s["families"]:
        w(f"### {FAMILY_DE[name]}\n")
        rows = []
        for r in (x for x in res if x["family"] == name):
            ser = r["series"][r["judged"]]
            flag = ""
            if r.get("caveat"):
                flag = " (Achtung: hängt an " + ", ".join(r["robustness"]["removed"]) + ")"
            elif r.get("basket_too_small"):
                flag = " (Korb zu klein)"
            rows.append([f"{r['rule']} {s['rules'][r['rule']]['name_de']}", HORIZON_DE[r["horizon"]],
                         SIDE_DE[r["side"]], _f(r["baskets"].get("ALL"), 1),
                         str(ser["IS"]["n"]), _f(ser["IS"]["mean_bps"], 0),
                         _f(ser["IS"]["t_nonoverlap"]), str(ser["OOS"]["n"]),
                         _f(ser["OOS"]["mean_bps"], 0), _f(ser["OOS"]["t_nonoverlap"]),
                         str(ser["OOS"]["n_nonoverlap"]), _f(ser["OOS"]["nw_t"]),
                         VERDICT_DE[r["verdict"]] + (" (stark)" if r["strong"] else "") + flag])
        w(_table(["Regel", "Horizont", "Seite", "Ø Korb", "IS n", "IS bp", "IS t", "OOS n",
                  "OOS bp", "OOS t", "n ohne", "OOS NW-t", "Urteil"], rows))
        w("")
    w("## Robustheit: ohne die drei Gewinner\n")
    checked = [r for r in res if r.get("robustness")]
    if checked:
        w(f"Für jede Zelle, die belegt oder schwach belegt ist, und zur Information auch für jede, "
          "die es nach der früheren Regel (Newey-West-t) war, wurde die beurteilte Reihe "
          f"ohne die {th['robust_k']} Werte mit der höchsten Gesamtrendite neu berechnet "
          f"(Universum, Rangschwellen und Körbe auf der gekürzten Liste). Das Urteil bleibt nur bei "
          f"einem gepoolten t von mindestens {_f(th['robust_min_t'])}.\n")
        rows = []
        for r in checked:
            rb = r["robustness"]
            ser = r["series"][r["judged"]]
            rows.append([f"{FAMILY_DE[r['family']]}, {r['rule']}, {HORIZON_DE[r['horizon']]}, "
                         f"{SIDE_DE[r['side']]}", ", ".join(rb["removed"]),
                         f"{_f(ser['ALL']['mean_bps'], 0)} bp / t {_f(r['pooled_t'])} "
                         f"(Newey-West {_f(r.get('pooled_t_nw'))})",
                         f"{_f(rb['pooled_mean_bps'], 0)} bp / t {_f(rb['pooled_t'])} "
                         f"(Newey-West {_f(rb.get('pooled_t_nw'))})",
                         f"IS {_f(rb['IS']['t'])}, OOS {_f(rb['OOS']['t'])}",
                         VERDICT_DE[r.get("verdict_pre_robustness")
                                    or r.get("legacy_verdict") or r["verdict"]]
                         + (" (frühere Regel)" if rb.get("info_only") else ""),
                         VERDICT_DE[r["verdict"]]])
        w(_table(["Zelle", "Entfernt", "Gepoolt vorher", "Gepoolt ohne sie", "t ohne sie (IS / OOS)",
                  "Urteil vor Prüfung", "Urteil danach"], rows))
        if any(r["robustness"].get("info_only") for r in checked):
            w("\nDie Zellen mit dem Vermerk \"frühere Regel\" waren schon an der korrigierten "
              "Statistik gescheitert (getrenntes t im In-Sample- bzw. Out-of-Sample-Teil unter der "
              "Schwelle); die Robustheitsprüfung hätte sie für sich allein nur herabgesetzt, wenn "
              "das gepoolte t unter 2 fällt. Mit dem einfachen t bleibt es oberhalb (Spalte "
              "\"Gepoolt ohne sie\"), mit dem Newey-West-t der Prüfung darunter oder knapp "
              "darauf: das gepoolte t allein ist ein schwächeres Kriterium als die getrennte "
              "Prüfung beider Abschnitte. Ohne die drei Aktien bleibt im Auswahlzeitraum kaum "
              "ein Vorsprung (Spalte \"t ohne sie\").")
    else:
        w("Keine Zelle erreichte vor dieser Prüfung ein Urteil von \"schwach belegt\" oder besser; "
          "es gab daher nichts zu prüfen.")
    w("\n## Stop-Modus (warum die Ideen keinen Stop tragen)\n")
    w(f"Dieselben Regeln wurden zusätzlich mit einem Schutz-Stop gerechnet ({_f(s['stop_atr']['medium'])} "
      f"bzw. {_f(s['stop_atr']['long'])} x ATR(14), Lücken füllen zur Eröffnung). Gezeigt ist die "
      "beurteilte Reihe für alle Zellen, die ohne Robustheitsprüfung belegt oder schwach belegt "
      "gewesen wären:\n")
    prior = [r for r in res if r.get("robustness")]
    if prior:
        rows = []
        for r in prior:
            a, b = r["series"][r["judged"]], r["stop"][r["judged"]]
            rows.append([f"{FAMILY_DE[r['family']]}, {r['rule']}, {HORIZON_DE[r['horizon']]}, "
                         f"{SIDE_DE[r['side']]}",
                         f"{_f(a['IS']['mean_bps'], 0)} / {_f(a['OOS']['mean_bps'], 0)}",
                         f"{_f(a['IS']['t_nonoverlap'])} / {_f(a['OOS']['t_nonoverlap'])}",
                         f"{_f(b['IS']['mean_bps'], 0)} / {_f(b['OOS']['mean_bps'], 0)}",
                         f"{_f(b['IS']['t_nonoverlap'])} / {_f(b['OOS']['t_nonoverlap'])}"])
        w(_table(["Zelle", "ohne Stop: IS / OOS bp", "t", "mit Stop: IS / OOS bp", "t"], rows))
    else:
        w("Keine Zelle erreichte ein solches Urteil.")
    w("\n## Risikosicht (Long-Seite)\n")
    w("Nicht überlappende Termine, der Korb wird gehalten (Bargeld, wenn leer), gegen den "
      "Durchschnitt aller Werte und den Vergleichsindex über dieselben Fenster. Eine "
      "Trendregel muss nicht mehr verdienen, kann aber Verluste begrenzen; deshalb stehen "
      "Rendite und größter Rückgang nebeneinander (Gesamtzeitraum). Der größte Rückgang "
      "ist auf **täglichen** Kursen innerhalb der Haltefenster gemessen.\n")
    for name in s["families"]:
        w(f"### {FAMILY_DE[name]}\n")
        rows = []
        for r in (x for x in s["risk"] if x["family"] == name):
            a = r["ALL"]
            b, u, bm = a["basket"], a["universe"], a["benchmark"]
            rows.append([f"{r['rule']}", HORIZON_DE[r["horizon"]], _pct(a["in_market"], 0),
                         f"{_pct(b['cagr'])} / {_pct(u['cagr'])} / {_pct(bm['cagr'])}",
                         f"{_f(b['sharpe'], 2)} / {_f(u['sharpe'], 2)} / {_f(bm['sharpe'], 2)}",
                         f"{_pct(b['max_drawdown'], 0)} / {_pct(u['max_drawdown'], 0)} / "
                         f"{_pct(bm['max_drawdown'], 0)}"])
        w(_table(["Regel", "Horizont", "Anteil investiert", "Rendite p. a. (Korb / alle / Index)",
                  "Sharpe (Korb / alle / Index)",
                  "Max. Rückgang auf Tageskursen (Korb / alle / Index)"], rows))
        w("")
    w("## Einordnung der Befunde\n")
    w("Auch ein bestandener Test wäre kein Versprechen: Er stützt sich auf eine einzige "
      "Testperiode mit wenigen unabhängigen Beobachtungen (bei 6 Monaten Haltedauer im "
      "Testabschnitt nur etwa 7 bis 8 nicht überlappende Termine), auf heutige Listenmitglieder "
      "(Survivorship) und auf Regeln, die nach Sicht der In-Sample-Zahlen um Varianten ergänzt "
      "wurden. Bei Varianten gilt bei gleichem Urteil die vorab festgelegte Regel als "
      "maßgeblich.\n")
    w("Zur Beschriftung \"nur Marktrendite\": Dieses Etikett (Long-Seite besteht auf absoluter "
      "Rendite, hat aber keinen belegten Vorsprung vor dem Durchschnitt aller Werte außerhalb der "
      "Stichprobe, also Mittelwert ≤ 0 oder t < 1) wurde nach dem ersten Gesamtlauf präzisiert, "
      "damit Regeln, die außerhalb der Stichprobe doch vorn lagen, aber im Auswahlzeitraum "
      "durchfielen, nicht fälschlich als Beta erscheinen. Die Urteile \"belegt\", \"schwach "
      "belegt\" und \"nur relativ\" waren davon nicht betroffen.\n")
    w("## Varianten und Mehrfachtests\n")
    c = counts
    chance = chance_rates(res)
    w(f"Vorab festgelegt waren {c['rule_horizons_preregistered']} Regel-Horizont-Kombinationen je "
      f"Familie, dazu {c['variants']} Varianten ({c['rule_horizons_variants']} Kombinationen). "
      f"Das ergibt {c['evaluations_total']} Zellen (Regel, Horizont, Seite, Familie), aber nur "
      f"**{c['evaluations_distinct']} verschiedene Messungen**: {c['evaluations_duplicate']} Zellen "
      "(die Long-Seite von V3 ist die von T3) wiederholen eine andere und zählen nicht doppelt.\n")
    typ = chance["typical"]
    if typ:
        parts = "; ".join(
            f"{HORIZON_DE[h]}: {v['n_is']} bzw. {v['n_oos']} nicht überlappende Termine im In- bzw. "
            f"Out-of-Sample-Teil, Zufallstreffer je Teil {_pct(v['p_is'])} bzw. {_pct(v['p_oos'])}"
            for h, v in typ.items())
        w("Wer genug Regeln probiert, findet allein durch Zufall welche mit t ≥ 2. Mit dem "
          f"einfachen t auf nicht überlappenden Terminen (so wird jetzt beurteilt) gilt: {parts}. "
          "Aus den tatsächlichen Stichprobengrößen jeder verschiedenen Messung (Simulation mit "
          "festem Startwert, t-Verteilung, beide Abschnitte unabhängig, Kosten- und "
          f"Robustheitsbedingung nicht eingerechnet, also eine obere Grenze) folgen **{_f(chance['validated'], 1)} "
          f"zufällige \"belegt\"** und **{_f(chance['either'], 1)} zufällige \"belegt oder schwach "
          "belegt\"** bei keinem echten Effekt. Die erste Fassung hatte \"etwa 0,08\" genannt; "
          "das war um den Faktor 8 bis 30 zu niedrig: die unabhängige Prüfung kam für die damalige "
          "Newey-West-Regel auf etwa 0,6 zufällige \"belegt\" und 2,5 zufällige \"belegt oder "
          "schwach belegt\". Die Familien sind aber untereinander stark verwandt (derselbe "
          "Markt), die effektive Zahl unabhängiger Tests ist kleiner.\n")
    if s.get("variants"):
        rows = [[v["id"], v["name_de"], v.get("note_de", "")] for v in s["variants"]]
        w(_table(["Variante", "Regel", "Begründung (aus dem In-Sample-Teil)"], rows))
        w("\nV1 wurde mit genau der Zelle begründet, die später als einzige bestand "
          "(Momentum, US, lang): sie ist deshalb kein unabhängiger zweiter Test.\n")
    w("## Schwellen\n")
    w("Alle Schwellen sind vor dem Lauf festgelegt und wurden nicht angepasst:\n")
    w(_table(["Schwelle", "Wert", "Bedeutung"], [
        ["VALIDATED_T", _f(th["validated_t"]), "belegt: t in In-Sample und Out-of-Sample mindestens so hoch, "
         "und der Mittelwert bleibt bei doppelten Kosten positiv"],
        ["EXPERIMENTAL_T_IS", _f(th["experimental_t_is"]), "schwach belegt: t im In-Sample-Teil"],
        ["EXPERIMENTAL_T_OOS", _f(th["experimental_t_oos"]), "schwach belegt: t im Out-of-Sample-Teil "
         "bei positivem Mittelwert; zugleich die Grenze, unter der ein Vorsprung vor dem Markt als "
         "\"nicht vorhanden\" gilt (nur Marktrendite)"],
        ["STRONG_T", _f(th["strong_t"]), "\"stark\": gepooltes t mindestens so hoch; nie für Varianten"],
        ["VALIDATED_MIN_OOS_DATES", str(th["validated_min_oos_dates"]),
         "Mindestzahl Termine im Out-of-Sample-Teil für \"belegt\"; geprüft wird die Zahl aller "
         "Termine (auch überlappender), während das t aus den nicht überlappenden Terminen "
         "stammt; die Schwelle wurde nicht geändert"],
        ["EXPERIMENTAL_MIN_OOS_DATES", str(th["experimental_min_oos_dates"]),
         "Mindestzahl Termine im Out-of-Sample-Teil für \"schwach belegt\"; ein t ≥ 1 auf wenigen "
         "Terminen ist Rauschen; auch hier zählen alle (überlappenden) Termine, nicht die "
         "nicht überlappenden, aus denen das t stammt"],
        ["MIN_BASKET", str(th["min_basket"]), "durchschnittliche Korbgröße, ab der ein Urteil über "
         "\"nicht belegt\" hinaus möglich ist"],
        ["ROBUST_K / ROBUST_MIN_T", f"{th['robust_k']} / {_f(th['robust_min_t'])}",
         "Zahl der entfernten Gewinner und gepooltes t, das ohne sie bestehen muss"],
        ["IS_FRACTION", _f(s["is_fraction"], 2), "Anteil der Termine im In-Sample-Teil (mit Purge)"],
        ["Kosten", ", ".join(f"{FAMILY_DE[k]} {_f(v, 0)} bp" for k, v in s["costs_bps"].items()),
         "je Hin- und Rückweg und Position"],
        ["Stop-Modus", f"{_f(s['stop_atr']['medium'])} / {_f(s['stop_atr']['long'])} x ATR",
         "nur zum Vergleich gerechnet, nicht Teil des Urteils"],
        ["Veraltung", f"{STALE_AFTER_DAYS} Tage", "ältere Studien werden im Programm als veraltet markiert"],
    ]))
    w("")
    w("## Grenzen\n")
    for line in [
        "**Drei Werte tragen den früheren Befund.** Das Ergebnis der früheren \"belegten\" Zelle "
        "hing an NVDA, AVGO und AMD; es gibt keinen Grund anzunehmen, dass nicht auch andere "
        "Zellen an wenigen späteren Gewinnern hängen. Darum die Robustheitsprüfung; sie ist "
        "vorab auf k = 3 festgelegt, nicht auf das beste Ergebnis abgestimmt.",
        "**Der Out-of-Sample-Zeitraum war überall momentum-freundlich.** In jeder Familie "
        "schnitt Momentum im letzten Abschnitt besser ab als im Auswahlzeitraum (beim DAX bei "
        "einem In-Sample-t von etwa 0). Ein Bestehen dort sagt wenig über andere Marktphasen.",
        "**Der Stop-Modus war negativ.** Mit Stop verschwand der Vorteil der früheren US-Zelle; "
        "die Regel funktioniert, wenn überhaupt, nur als reine Zeitregel ohne Stop, und das ist "
        "dann ein Risiko für den Anleger (keine Verlustbegrenzung).",
        "**V1 entstand aus derselben Zelle**, die sie später bestätigen sollte; die Variante ist "
        "kein unabhängiger Beleg.",
        "**Das Newey-West-t ist optimistisch** und wird nur zur Information mitgeführt; "
        "beurteilt wird mit dem Median des einfachen t der nicht überlappenden Termine über alle "
        "Phasen. Dieses beruht auf "
        "wenigen Beobachtungen (beim Halbjahr etwa 7 bis 8 im Testabschnitt) und ist entsprechend "
        "unscharf.",
        "**Der größte Rückgang** der Risikosicht wird auf täglichen Kursen innerhalb der "
        "Haltefenster gemessen, die Rendite und der Sharpe-Wert aber an den Periodenenden; "
        "Tage zwischen zwei Haltefenstern (Bargeld) sind nicht abgebildet.",
        "**Survivorship-Bias**: Es werden die heutigen Mitglieder der Listen rückwirkend "
        "getestet. Firmen, die zwischenzeitlich schwach waren, ausschieden oder pleitegingen, "
        "fehlen. Das begünstigt vor allem Long-Regeln und verzerrt Short-Ergebnisse zum "
        "Schlechteren (Verlierer fehlen).",
        "**Keine Leihgebühren und Dividendenkosten** bei Shorts; reale Shorts sind teurer und "
        "teils nicht möglich. Kostenannahmen sind pauschal (10 bp bzw. 30 bp je Hin- und Rückweg), "
        "ohne Abwägung von Spreads in schwachen Märkten oder Steuern.",
        "**Eine einzige Out-of-Sample-Periode** (die letzten 30 % der Termine): sie ist ein "
        "einzelnes Marktumfeld. Ein Bestehen belegt nicht, dass die Regel in anderen Umfeldern "
        "funktioniert, ein Durchfallen nicht, dass sie nie funktionieren kann.",
        "**Gleichgewichtete Körbe** ohne Rücksicht auf Liquidität und Positionsgrößen; die "
        "Rendite eines echten Depots mit wenigen Positionen streut deutlich stärker.",
        "Kurse sind rückwirkend um Dividenden und Splits bereinigt (Yahoo), Krypto-Daten "
        "reichen je nach Coin nur wenige Jahre zurück.",
    ]:
        w(f"- {line}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")
    return Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tradingagents.scanner.horizon_research",
        description="Run the medium/long-horizon study and write evidence.json and the report.")
    parser.add_argument("--out", required=True, help="directory for evidence.json and the report")
    parser.add_argument("--families", default=None,
                        help="comma-separated subset of: " + ", ".join(family_specs()))
    parser.add_argument("--cache", default=None, help="directory caching the Yahoo downloads")
    parser.add_argument("--as-of", default=None, help="study date YYYY-MM-DD (default: today)")
    parser.add_argument("--before", default=None,
                        help="the snapshot from before a correction, for the before/after table "
                             "(default: evidence_before_review.json in --out, if present)")
    parser.add_argument("--update-package", action="store_true",
                        help="also overwrite the packaged evidence.json the live scan reads")
    parser.add_argument("--explore", action="store_true",
                        help="in-sample only: print the IS table, write nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    as_of = args.as_of or datetime.now(UTC).strftime("%Y-%m-%d")
    fams = args.families.split(",") if args.families else None
    study = run_study(as_of, args.cache, fams, include_oos=not args.explore,
                      boot=not args.explore)
    if args.explore:
        for e in study.entries:
            ser = e["series"][e["judged"]]["IS"]
            print(e["family"], e["rule"], e["horizon"], e["side"], ser["n"],
                  _f(ser["mean_bps"], 0), _f(ser["t_nonoverlap"]))
        return 0
    out = Path(args.out)
    snap_path = write_snapshot(study, out / SNAPSHOT_NAME)
    snapshot = json.loads(snap_path.read_text(encoding="utf-8"))
    before_path = Path(args.before) if args.before else out / "evidence_before_review.json"
    before = load_snapshot(before_path) if before_path.exists() else None
    report = write_report(snapshot, out / "BERICHT_Mittel_Lang.md", before)
    if args.update_package:
        write_snapshot(study, SNAPSHOT_PATH)
    print(f"snapshot: {snap_path}\nreport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
