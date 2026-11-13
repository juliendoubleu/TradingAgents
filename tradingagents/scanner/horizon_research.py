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
* Windows overlap (a one-month hold decided every five days), so a plain t-stat would be
  far too confident.  Significance is a Newey-West t (Bartlett kernel, lags = windows per
  hold - 1), cross-checked with a plain t on non-overlapping dates and a moving-block
  bootstrap interval.
* Verdicts use the pre-registered thresholds of ``shortterm.evaluate``; "validated" needs
  the edge on BOTH segments.  The number of evaluations is recorded because with enough
  tests something passes by chance.
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
SNAPSHOT_VERSION = 1
IS_FRACTION = 0.70
N_BOOT = 2000
SEED = 20261113
VALIDATED_MIN_OOS_DATES = 30
EXPERIMENTAL_MIN_OOS_DATES = 20       # added guard: a t >= 1 on a handful of dates is noise
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


def series_stats(values, hold: int, step: int, boot: bool = True) -> dict:
    """The statistics block of one series on one segment; values are fractions, output bps."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    k = math.ceil(hold / step)
    out = {"n": int(len(x)), "mean_bps": None, "median_bps": None, "pos_share": None,
           "nw_t": 0.0, "t_nonoverlap": 0.0, "ci_low_bps": None, "ci_high_bps": None}
    if len(x) == 0:
        return out
    out["mean_bps"] = float(x.mean() * 1e4)
    out["median_bps"] = float(np.median(x) * 1e4)
    out["pos_share"] = float((x > 0).mean())
    out["nw_t"] = nw_tstat(x, k - 1)
    out["t_nonoverlap"] = plain_tstat(x[::k])
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


def risk_stats(period_returns, periods_per_year: float) -> dict:
    """CAGR, annualised volatility, Sharpe (risk-free 0) and maximum drawdown of a series of
    non-overlapping period returns; the periods are compounded."""
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
    return {"n": int(n), "cagr": cagr, "vol": float(vol), "sharpe": sharpe,
            "max_drawdown": float(((equity / peak) - 1.0).min())}


# --------------------------------------------------------------------------- verdict


def _t(stats: dict) -> float:
    return float(stats.get("nw_t") or 0.0)


def _meets(level: str, is_s: dict, oos_s: dict, oos_mean_2x_bps: float | None) -> bool:
    oos_mean = oos_s.get("mean_bps")
    if oos_mean is None:
        return False
    if level == "validated":
        return (_t(is_s) >= VALIDATED_T and _t(oos_s) >= VALIDATED_T
                and oos_mean_2x_bps is not None and oos_mean_2x_bps > 0
                and oos_s["n"] >= VALIDATED_MIN_OOS_DATES)
    return (_t(is_s) >= EXPERIMENTAL_T_IS and oos_mean > 0 and _t(oos_s) >= EXPERIMENTAL_T_OOS
            and oos_s["n"] >= EXPERIMENTAL_MIN_OOS_DATES)


def _level(is_s: dict, oos_s: dict, oos_2x: float | None) -> str | None:
    for level in ("validated", "experimental"):
        if _meets(level, is_s, oos_s, oos_2x):
            return level
    return None


def judge_side(side: str, series: dict[str, dict], oos_2x: dict[str, float | None]) -> dict:
    """Pre-registered verdict of one side from its series statistics.

    Long is judged on ``long_excess`` (beating the universe, not riding it).  Short is
    judged on ``short_pnl`` (the price must really fall); failing that, ``short_excess``
    meeting the same thresholds gives ``relative`` (the name lags the market: a reason to
    avoid it, no evidence for an outright short).  A long that fails against the universe
    but passes on ``long_abs`` is ``beta``: it made money only because the market did.  The
    label needs the out-of-sample surplus over the universe to be absent (mean <= 0 or
    t < 1); a rule that did beat the universe out of sample but failed in-sample stays
    ``unproven``, because beta would misdescribe it.
    ``strong`` needs a pooled t >= 3 on the judged series and some verdict above unproven.
    """
    judged = JUDGED[side]
    s = series[judged]
    level = _level(s["IS"], s["OOS"], oos_2x.get(judged))
    verdict = level or "unproven"
    if level is None and side == "short":
        ex = series["short_excess"]
        if _level(ex["IS"], ex["OOS"], oos_2x.get("short_excess")):
            verdict = "relative"
    if level is None and side == "long":
        ab, ex = series["long_abs"], series["long_excess"]["OOS"]
        no_surplus = (ex.get("mean_bps") is None or ex["mean_bps"] <= 0
                      or _t(ex) < EXPERIMENTAL_T_OOS)
        if no_surplus and _level(ab["IS"], ab["OOS"], oos_2x.get("long_abs")):
            verdict = "beta"
    pooled = _t(s["ALL"])
    return {"verdict": verdict, "judged": judged, "pooled_t": pooled,
            "strong": bool(verdict != "unproven" and pooled >= STRONG_T)}


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


def _risk_view(grid: Grid, L: np.ndarray, nL: np.ndarray, cost: float, include_oos: bool,
               bars_per_year: int) -> dict:
    """The long basket held at non-overlapping dates (cash when empty) against the equal-
    weight universe and the benchmark held over the same windows."""
    k = math.ceil(grid.hold / grid.step)
    ppy = bars_per_year / (k * grid.step)
    strat = np.where(nL > 0, np.nan_to_num(L) - cost, 0.0)
    seg_idx = {"ALL": np.arange(len(grid.pos)), **({"OOS": grid.oos_idx} if include_oos else {})}
    out = {}
    for name, idx in seg_idx.items():
        sel = idx[::k]
        ok = np.isfinite(grid.U[sel]) & np.isfinite(grid.B[sel])
        sel = sel[ok]
        out[name] = {
            "basket": risk_stats(strat[sel], ppy),
            "universe": risk_stats(grid.U[sel], ppy),
            "benchmark": risk_stats(grid.B[sel], ppy),
            "in_market": float((nL[sel] > 0).mean()) if len(sel) else None,
        }
    return out


def evaluate_cell(grid: Grid, sides_pos: np.ndarray, cost_bps: float, bars_per_year: int, *,
                  include_oos: bool = True, boot: bool = True) -> dict:
    """Everything measured for one (family, rule, horizon): series statistics in both
    execution modes, the 2x-cost out-of-sample means, basket sizes and the risk view."""
    cost = cost_bps / 1e4
    L, S, nL, nS = grid.basket_means(sides_pos)
    ser = return_series(L, S, grid.U, cost)
    all_names = ("long_excess", "long_abs", "short_pnl", "short_excess", "spread")
    result = {"series": _stats_for(ser, grid, include_oos, boot, all_names)}
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
        seg: {"long": float(nL[idx].mean()) if len(idx) else None,
              "short": float(nS[idx].mean()) if len(idx) else None}
        for seg, idx in grid.segments(include_oos).items()}
    result["risk"] = _risk_view(grid, L, nL, cost, include_oos, bars_per_year)
    return result


def entries_for(family: str, rule_id: str, horizon: str, cell: dict, include_oos: bool) -> list[dict]:
    """The flat result entries (one per side) of a cell, with verdicts when OOS is in."""
    out = []
    for side in ("long", "short"):
        names = SIDE_SERIES[side]
        entry = {"family": family, "rule": rule_id, "horizon": horizon, "side": side,
                 "judged": JUDGED[side],
                 "series": {n: cell["series"][n] for n in names},
                 "stop": {n: cell["stop"][n] for n in names},
                 "baskets": {seg: v[side] for seg, v in cell["baskets"].items()}}
        if include_oos:
            entry.update(judge_side(side, cell["series"], cell["oos_mean_2x_bps"]))
            entry["oos_mean_2x_bps"] = {n: cell["oos_mean_2x_bps"][n] for n in names}
        out.append(entry)
    return out


# --------------------------------------------------------------------------- data


def _cached(cache_dir: Path | None, name: str, build):
    if cache_dir is not None:
        path = Path(cache_dir) / name
        if path.exists():
            return pickle.loads(path.read_bytes())
    value = build()
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / name).write_bytes(pickle.dumps(value))
    return value


@dataclass
class Family:
    name: str
    info: object
    frames: dict[str, pd.DataFrame]
    panel: Panel
    cost_bps: float
    params: SignalParams


def load_family(spec: FamilySpec, as_of: str, cache_dir: Path | None,
                now: datetime | None = None) -> Family:
    info = universe_info(spec.universe, list(spec.tickers) if spec.tickers else None)
    vix = ["^VIX"] if info.use_vix else []
    wanted = list(dict.fromkeys([*info.members, info.benchmark, *vix]))
    raw = _cached(cache_dir, f"{spec.name}_{as_of}.pkl",
                  lambda: load_bulk_ohlcv(wanted, as_of, HISTORY_DAYS))
    frames = settled(raw, now or datetime.now(UTC))
    params = SignalParams(info.month_bars, info.bars_per_year)
    members = [m for m in info.members if m in frames]
    panel = Panel(frames, members, info.benchmark, params)
    return Family(spec.name, info, frames, panel, COSTS_BPS[spec.name], params)


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
                entries.extend(entries_for(name, rid, h, cell, include_oos))
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


def count_evaluations(study: Study) -> dict:
    pre = [r for r in study.rule_ids if not RULES[r].variant]
    var = [r for r in study.rule_ids if RULES[r].variant]
    n_fam = len(study.families)

    def cells(ids):
        return sum(len(RULES[r].horizons) for r in ids)
    return {
        "families": n_fam,
        "rule_horizons_preregistered": cells(pre),
        "variants": len(var),
        "rule_horizons_variants": cells(var),
        "evaluations_preregistered": 2 * n_fam * cells(pre),
        "evaluations_variants": 2 * n_fam * cells(var),
        "evaluations_total": 2 * n_fam * (cells(pre) + cells(var)),
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
            "experimental_min_oos_dates": EXPERIMENTAL_MIN_OOS_DATES},
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
VERDICT_DE = {"validated": "belegt", "experimental": "schwach belegt", "relative": "nur relativ",
              "beta": "nur Markt-Beta", "unproven": "nicht belegt"}
HORIZON_DE = {"medium": "mittel (1 Monat)", "long": "lang (6 Monate)"}
SIDE_DE = {"long": "Long", "short": "Short"}


def _f(x, nd=1) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(x, nd=1) -> str:
    return "–" if x is None else _f(x * 100, nd) + " %"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def write_report(snapshot: dict, path: str | Path) -> Path:
    s = snapshot
    res = s["results"]
    counts = s["counts"]
    best = [r for r in res if r["verdict"] in ("validated", "experimental")]
    rel = [r for r in res if r["verdict"] == "relative"]
    beta = [r for r in res if r["verdict"] == "beta"]
    out: list[str] = []
    w = out.append
    w("# Mittel- und Langfrist: Was ist wirklich belegt?")
    w(f"\nStudie vom {s['study_date']}. Alle Zahlen stammen aus einem echten Testlauf auf "
      "Yahoo-Kursdaten. Verworfene Regeln und Varianten stehen in den Tabellen mit drin.\n")
    w("## Kurzfassung\n")
    if not best:
        w("**Keine der getesteten Regeln hat in beiden Testzeiträumen einen messbaren Vorteil "
          "gegenüber dem Durchschnitt aller Werte gezeigt.** Das ist das Ergebnis, kein Fehler: "
          "Die Anlageideen des Programms für Monate und Halbjahre sind deshalb als "
          "Beobachtungsliste gekennzeichnet, nicht als Empfehlung.")
    else:
        w(f"Von {counts['evaluations_total']} geprüften Regel-Seiten-Kombinationen bestehen "
          f"{len(best)} den vorab festgelegten Test (siehe Tabellen). Wegen der vielen Tests "
          "kann ein Teil davon Zufall sein; Details unten.")
        for r in best:
            w(f"- {FAMILY_DE[r['family']]}, {s['rules'][r['rule']]['name_de']}, "
              f"{HORIZON_DE[r['horizon']]}, {SIDE_DE[r['side']]}: {VERDICT_DE[r['verdict']]}"
              f"{' (stark)' if r['strong'] else ''}")
    if rel:
        w(f"\nNur **relativ** (der Wert hinkt dem Markt hinterher, fällt aber nicht belegbar "
          f"absolut; ein Grund zum Meiden/Untergewichten, kein Grund für ein Short): {len(rel)} "
          "Fälle.")
    if beta:
        w(f"\nNur **Markt-Beta** (Gewinn in absoluten Zahlen, aber nicht mehr als der "
          f"Durchschnitt aller Werte verdient hat): {len(beta)} Fälle.")
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
      "deren Haltefenster in den OOS-Zeitraum hineinreicht, werden entfernt (Purge).\n")
    w("## Ergebnisse je Familie\n")
    w("Spalten: Termine (n) und mittlere Rendite je Termin in Basispunkten (1 bp = 0,01 %) der "
      "**beurteilten Reihe** (Long: Vorsprung vor dem Durchschnitt aller Werte; Short: "
      "tatsächlicher Gewinn des Shorts), jeweils nach Kosten; t = Newey-West-t (berücksichtigt "
      "überlappende Fenster; |t| ≥ 2 gilt als auffällig).\n")
    for name in s["families"]:
        w(f"### {FAMILY_DE[name]}\n")
        rows = []
        for r in (x for x in res if x["family"] == name):
            ser = r["series"][r["judged"]]
            rows.append([f"{r['rule']} {s['rules'][r['rule']]['name_de']}", HORIZON_DE[r["horizon"]],
                         SIDE_DE[r["side"]], _f(r["baskets"].get("OOS"), 1),
                         str(ser["IS"]["n"]), _f(ser["IS"]["mean_bps"], 0),
                         _f(ser["IS"]["nw_t"]), str(ser["OOS"]["n"]), _f(ser["OOS"]["mean_bps"], 0),
                         _f(ser["OOS"]["nw_t"]), VERDICT_DE[r["verdict"]]
                         + (" (stark)" if r["strong"] else "")])
        w(_table(["Regel", "Horizont", "Seite", "Ø Korb", "IS n", "IS bp", "IS t", "OOS n", "OOS bp",
                  "OOS t", "Urteil"], rows))
        w("")
    w("## Risikosicht (Long-Seite)\n")
    w("Nicht überlappende Termine, der Korb wird gehalten (Bargeld, wenn leer), gegen den "
      "Durchschnitt aller Werte und den Vergleichsindex über dieselben Fenster. Eine "
      "Trendregel muss nicht mehr verdienen, kann aber Verluste begrenzen; deshalb stehen "
      "Rendite und größter Rückgang nebeneinander (Gesamtzeitraum).\n")
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
                  "Sharpe (Korb / alle / Index)", "Max. Rückgang (Korb / alle / Index)"], rows))
        w("")
    w("## Einordnung der Befunde\n")
    if best:
        w("Auch ein bestandener Test ist kein Versprechen: Er stützt sich auf eine einzige "
          "Testperiode mit wenigen unabhängigen Beobachtungen (bei 6 Monaten Haltedauer etwa 44 "
          "Termine, von denen sich benachbarte stark überlappen), auf heutige Listenmitglieder "
          "(Survivorship) und auf Regeln, die nach Sicht der In-Sample-Zahlen um Varianten "
          "ergänzt wurden. Bei Varianten gilt bei gleichem Urteil die vorab festgelegte Regel "
          "als maßgeblich.\n")
    w("Zur Beschriftung \"nur Markt-Beta\": Dieses Etikett (Long-Seite besteht auf absoluter "
      "Rendite, hat aber keinen belegten Vorsprung vor dem Durchschnitt aller Werte außerhalb der "
      "Stichprobe, also Mittelwert ≤ 0 oder t < 1) wurde nach dem ersten Gesamtlauf präzisiert, "
      "damit Regeln, die außerhalb der Stichprobe doch vorn lagen, aber im Auswahlzeitraum "
      "durchfielen, nicht fälschlich als Beta erscheinen. Die Urteile \"belegt\", \"schwach "
      "belegt\" und \"nur relativ\" waren davon nicht betroffen.\n")
    w("## Varianten und Mehrfachtests\n")
    c = counts
    w(f"Vorab festgelegt waren {c['rule_horizons_preregistered']} Regel-Horizont-Kombinationen je "
      f"Familie, dazu {c['variants']} Varianten ({c['rule_horizons_variants']} Kombinationen), "
      f"insgesamt **{c['evaluations_total']}** Prüfungen (Regel, Horizont, Seite, Familie). "
      "Wer genug Regeln probiert, findet allein durch Zufall welche mit t > 2: bei rund "
      f"{c['evaluations_total']} Prüfungen sind etwa {_f(c['evaluations_total'] * 0.023, 1)} "
      "einseitige Zufallstreffer allein im In-Sample-Teil zu erwarten, und rund "
      f"{_f(c['evaluations_total'] * 0.023 * 0.023, 2)}, die in beiden Teilen t ≥ 2 erreichen, "
      "wenn es keinen echten Effekt gibt und die Teile unabhängig wären. Die Familien sind "
      "aber untereinander stark verwandt (derselbe Markt), die effektive Zahl unabhängiger Tests "
      "ist kleiner, ein Treffer also weniger überraschend, als die Zahl vermuten lässt.\n")
    if s.get("variants"):
        rows = [[v["id"], v["name_de"], v.get("note_de", "")] for v in s["variants"]]
        w(_table(["Variante", "Regel", "Begründung (aus dem In-Sample-Teil)"], rows))
        w("")
    w("## Grenzen\n")
    for line in [
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
        "**Überlappende Fenster** verringern die Zahl unabhängiger Beobachtungen; die "
        "Newey-West-Korrektur ist eine Näherung (bei wenigen Terminen unsicher).",
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
                  _f(ser["mean_bps"], 0), _f(ser["nw_t"]))
        return 0
    out = Path(args.out)
    snap_path = write_snapshot(study, out / SNAPSHOT_NAME)
    snapshot = json.loads(snap_path.read_text(encoding="utf-8"))
    report = write_report(snapshot, out / "BERICHT_Mittel_Lang.md")
    if args.update_package:
        write_snapshot(study, SNAPSHOT_PATH)
    print(f"snapshot: {snap_path}\nreport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
