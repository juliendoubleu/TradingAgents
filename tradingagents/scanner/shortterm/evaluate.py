"""Honest evaluation of simulated trades: IS/OOS split, day-clustered significance,
skill-versus-drift and the pre-registered verdict rule.

Why the machinery: trades on the same day are strongly correlated (the whole market
moves together), so a t-statistic over trades would be wildly overconfident. Here
every statistic of "does it make money" is computed on ONE daily series, the
"judged series", and the confidence interval resamples whole days of it:

* day trades (no overlap): the mean net return of the trades entered that day;
* swing trades (positions last up to 10 days and overlap): the calendar-time
  portfolio, i.e. each calendar day the equal-weight mean of the day's P&L of every
  position that is open that day, the round-trip cost charged on the exit day. Day
  clustering by ENTRY day alone would ignore that one position contributes to many
  days.

The verdict's cost clauses and ``beats_drift`` use the mean of that same judged
series, never the per-trade mean (which is kept as information only: it weighs the
days with many signals more). The chronological 70/30 split is fixed before looking at
anything; parameters are chosen on the in-sample (IS) part only and the out-of-sample
(OOS) part is evaluated once per final candidate and reported whatever it shows.

A ``trades`` frame has one row per simulated trade with the columns:
``symbol, date, direction, raw, univ, window`` (plus, for multi-day trades, the
per-day legs ``path_days, path_ret, path_univ``) where ``date`` is the session day of
the entry (naive, midnight), ``raw`` the symbol's own return entry -> exit
(``exit/entry - 1``, before direction and costs), ``univ`` the equal-weight average
return of the whole research universe over the same entry-open -> exit-bar-close
window, and ``window`` the symbol's own return over that bar window with no stop
(used only by the baselines).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from tradingagents.scanner.shortterm import COST_MULTIPLES, IS_FRACTION

TRADING_DAYS = 252
N_BOOT = 2000
N_RANDOM = 200
SEED = 20260101

# Pre-registered verdict thresholds (fixed before any result was looked at).
VALIDATED_T = 2.0
VALIDATED_MIN_TRADES = 100
VALIDATED_MIN_DAYS = 20
EXPERIMENTAL_T_IS = 2.0
EXPERIMENTAL_T_OOS = 1.0
EXPERIMENTAL_MIN_TRADES = 50
STRONG_T = 3.0


def split_days(days, is_fraction: float = IS_FRACTION) -> tuple[list, list]:
    """Chronological split of the trading days: first ``is_fraction`` in-sample, rest OOS.

    Sorted, de-duplicated, never shuffled. The boundary is a count of days, so it
    is fixed by the dataset alone, before any strategy result exists.
    """
    ordered = sorted(set(pd.to_datetime(list(days))))
    cut = int(math.floor(len(ordered) * is_fraction))
    return ordered[:cut], ordered[cut:]


def with_net(trades: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Add direction-adjusted ``gross``, ``net`` and ``excess`` (net of the universe drift)."""
    t = trades.copy()
    t["gross"] = t["direction"] * t["raw"]
    t["net"] = t["gross"] - cost_bps / 10_000.0
    # long: ret - universe; short: universe - ret  ==  direction * (ret - universe)
    t["excess"] = t["direction"] * (t["raw"] - t["univ"]) - cost_bps / 10_000.0
    return t


def daily_series(t: pd.DataFrame, column: str = "net") -> pd.Series:
    """Mean of ``column`` per entry day over the days that had at least one trade."""
    if t.empty:
        return pd.Series(dtype=float)
    return t.groupby("date")[column].mean().sort_index()


def t_stat(daily: pd.Series) -> float:
    """Mean / (std / sqrt(n_days)) over the daily series; 0.0 when it is undefined."""
    n = len(daily)
    if n < 2:
        return 0.0
    sd = float(daily.std(ddof=1))
    if sd == 0.0 or math.isnan(sd):
        return 0.0
    return float(daily.mean()) / (sd / math.sqrt(n))


def bootstrap_ci(daily: pd.Series, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """95 % percentile interval of the mean daily value, resampling DAYS with replacement."""
    n = len(daily)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values = daily.to_numpy(dtype=float)
    means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- legs / judged series

_LEG_COLUMNS = ["day", "direction", "ret", "univ", "last"]


def explode_legs(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per (trade, calendar day the position is open): the unit of the judged series.

    Columns ``day, direction, ret, univ, last``: ``ret``/``univ`` are the symbol's and
    the universe's raw return over that day (entry day: open to close; later days:
    close to close; the exit day of a stopped trade ends at the exit price), ``last``
    marks the exit day (where the round-trip cost is charged). A trade without
    ``path_days`` (a day trade) is one leg on its entry day with its whole return.
    """
    if trades.empty:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in _LEG_COLUMNS})
    direction = trades["direction"].to_numpy(dtype=float)
    if "path_days" in trades.columns and trades["path_days"].notna().all():
        if "path_univ" not in trades.columns:
            raise ValueError("multi-day trades need path_univ to compute the excess series")
        lens = trades["path_days"].map(len).to_numpy()
        day = np.concatenate([np.asarray(x, dtype="datetime64[ns]") for x in trades["path_days"]])
        ret = np.concatenate([np.asarray(x, dtype=float) for x in trades["path_ret"]])
        univ = np.concatenate([np.asarray(x, dtype=float) for x in trades["path_univ"]])
        last = np.zeros(len(day), dtype=float)
        last[np.cumsum(lens) - 1] = 1.0
        return pd.DataFrame({"day": day, "direction": np.repeat(direction, lens),
                             "ret": ret, "univ": univ, "last": last})
    return pd.DataFrame({
        "day": pd.to_datetime(trades["date"]).to_numpy(dtype="datetime64[ns]"),
        "direction": direction, "ret": trades["raw"].to_numpy(dtype=float),
        "univ": trades["univ"].to_numpy(dtype=float), "last": np.ones(len(trades))})


def judged_series(legs: pd.DataFrame, cost_bps: float, days=None) -> dict[str, pd.Series]:
    """The judged daily series from the legs: ``gross``, ``net`` and ``excess``.

    Each calendar day with at least one open position is the equal-weight mean over
    the open positions of the direction-adjusted return (excess: minus the universe's
    return over the same day), with ``cost_bps`` charged on the exit day. ``days``
    restricts the series to those calendar days.
    """
    if legs.empty:
        empty = pd.Series(dtype=float)
        return {"gross": empty, "net": empty, "excess": empty}
    d = legs["direction"].to_numpy()
    r, u = legs["ret"].to_numpy(), legs["univ"].to_numpy()
    work = pd.DataFrame({"day": legs["day"].to_numpy(), "pnl": d * r, "xs": d * (r - u),
                         "last": legs["last"].to_numpy()})
    g = work.groupby("day").agg(n=("pnl", "size"), pnl=("pnl", "sum"), xs=("xs", "sum"),
                                n_exit=("last", "sum")).sort_index()
    if days is not None:
        g = g[g.index.isin(pd.DatetimeIndex(sorted(days)))]
    cost = cost_bps / 10_000.0
    return {"gross": g["pnl"] / g["n"],
            "net": (g["pnl"] - cost * g["n_exit"]) / g["n"],
            "excess": (g["xs"] - cost * g["n_exit"]) / g["n"]}


def _side_stats(t: pd.DataFrame, legs: pd.DataFrame, cost_bps: float, day_set) -> dict:
    """Per-side block: per-trade mean and win rate, t of the side's own judged series."""
    if t.empty:
        return {"n_trades": 0, "mean_net_bps": 0.0, "win_rate": 0.0, "t_stat": 0.0}
    return {
        "n_trades": int(len(t)),
        "mean_net_bps": float(t["net"].mean() * 1e4),
        "win_rate": float((t["net"] > 0).mean()),
        "t_stat": t_stat(judged_series(legs, cost_bps, day_set)["net"]),
    }


def _max_drawdown_pct(daily: pd.Series) -> float:
    """Largest peak-to-trough fall of the COMPOUNDED equity curve of the series, in % (<= 100)."""
    if daily.empty:
        return 0.0
    equity = (1.0 + daily.to_numpy(dtype=float)).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return float(((peak - equity) / peak).max() * 100)


def segment_stats(trades: pd.DataFrame, cost_bps: float, *, boot: bool = True,
                  days=None, legs: pd.DataFrame | None = None) -> dict:
    """The full statistics block for one set of trades at one cost level.

    Two kinds of numbers, never to be mixed up:

    * per-trade (``mean_net_bps``, ``win_rate``, ``median_net_bps``, ``profit_factor``,
      ``mean_gross_bps``, ``mean_excess_bps``): information, taken over the trades
      ENTERED on ``days``;
    * the judged series (``t_stat``, ``mean_daily_net_bps``, ``mean_daily_excess_bps``,
      ``excess_t_stat``, interval, Sharpe, drawdown): one daily series, see the module
      docstring. Every decision rule reads these.

    ``days`` restricts both to those calendar days; ``legs`` may be passed to avoid
    exploding the trades again (they must come from the same or a superset of ``trades``).
    """
    day_set = None if days is None else set(pd.to_datetime(list(days)))
    entered = trades if day_set is None else trades[trades["date"].isin(day_set)]
    t = with_net(entered, cost_bps)
    if legs is None:
        legs = explode_legs(trades)
    series = judged_series(legs, cost_bps, day_set)
    daily, ex, gross_daily = series["net"], series["excess"], series["gross"]
    n = len(t)
    stats: dict = {
        "n_trades": int(n),
        "n_days": int(len(daily)),
        "trades_per_day": float(n / len(daily)) if len(daily) else 0.0,
        "win_rate": float((t["net"] > 0).mean()) if n else 0.0,
        "mean_net_bps": float(t["net"].mean() * 1e4) if n else 0.0,
        "median_net_bps": float(t["net"].median() * 1e4) if n else 0.0,
        "mean_gross_bps": float(t["gross"].mean() * 1e4) if n else 0.0,
        "t_stat": t_stat(daily),
        "mean_daily_net_bps": float(daily.mean() * 1e4) if len(daily) else 0.0,
        "mean_daily_gross_bps": float(gross_daily.mean() * 1e4) if len(gross_daily) else 0.0,
        "judged_series": "calendar_time_portfolio" if (
            "path_days" in trades.columns and bool((trades["path_days"].map(len) > 1).any()))
        else "entry_day_mean",
    }
    wins, losses = t.loc[t["net"] > 0, "net"].sum(), -t.loc[t["net"] < 0, "net"].sum()
    stats["profit_factor"] = float(wins / losses) if losses > 0 else (
        float("inf") if wins > 0 else 0.0)
    if boot and len(daily) >= 2:
        lo, hi = bootstrap_ci(daily)
        stats["ci_low_bps"], stats["ci_high_bps"] = lo * 1e4, hi * 1e4
    else:
        stats["ci_low_bps"] = stats["ci_high_bps"] = float("nan")
    sd = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    # annualised on the days the strategy was invested (days without a position are absent)
    stats["sharpe"] = float(daily.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
    stats["max_drawdown_pct"] = _max_drawdown_pct(daily)
    stats["mean_excess_bps"] = float(t["excess"].mean() * 1e4) if n else 0.0
    stats["mean_daily_excess_bps"] = float(ex.mean() * 1e4) if len(ex) else 0.0
    stats["excess_t_stat"] = t_stat(ex)
    for name, sign in (("long", 1), ("short", -1)):
        mask_t, mask_l = t["direction"] * sign > 0, legs["direction"] * sign > 0
        stats[name] = _side_stats(t[mask_t], legs[mask_l], cost_bps, day_set)
    return stats


def in_days(trades: pd.DataFrame, days) -> pd.DataFrame:
    return trades[trades["date"].isin(set(pd.to_datetime(list(days))))]


def baselines(trades: pd.DataFrame, cost_bps: float, *, n_random: int = N_RANDOM,
              seed: int = SEED) -> dict:
    """What the strategy's own trade windows earn with no direction skill.

    The window of a trade runs from its entry open to the CLOSE of the bar it exited
    on, so the baselines are conditioned on the strategy's own exit day (a stopped
    trade's window ends on its stop day); they are not stop-free in that sense.
    ``always_long``: every trade of the strategy taken long over its window, no stop
    applied inside it. ``strategy_dir``: the strategy's own directions over the same
    windows, so the comparison with the random band isolates direction skill from the
    stop. ``random``: random directions, same trades, ``n_random`` repetitions, mean
    and 95 % band of the mean net bps per trade. All three are per-trade means.
    """
    if trades.empty:
        return {"n_trades": 0}
    cost = cost_bps / 10_000.0
    w = trades["window"].to_numpy(dtype=float)
    d = trades["direction"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_random, len(w)))
    rand_means = (signs * w).mean(axis=1) - cost
    always = trades.assign(raw=trades["window"], direction=1)
    return {
        "n_trades": int(len(trades)),
        "always_long_net_bps": float(w.mean() - cost) * 1e4,
        "always_long_t": t_stat(daily_series(with_net(always, cost_bps))),
        "strategy_dir_net_bps": float((d * w).mean() - cost) * 1e4,
        "random_mean_net_bps": float(rand_means.mean() * 1e4),
        "random_lo_bps": float(np.percentile(rand_means, 2.5) * 1e4),
        "random_hi_bps": float(np.percentile(rand_means, 97.5) * 1e4),
        "universe_mean_bps": float(trades["univ"].mean() * 1e4),
    }


def verdict(is_stats: dict, oos_stats: dict, oos_stats_2x: dict, all_stats: dict) -> dict:
    """The pre-registered verdict: ``{"verdict": ..., "strong": bool}``.

    validated   IS t >= 2 AND OOS t >= 2 (both positive, net of base costs) AND the OOS
                mean net > 0 at 2x costs AND >= 100 OOS trades over >= 20 OOS days.
    experimental IS t >= 2 AND the OOS mean net > 0 at base cost AND OOS t >= 1 AND
                >= 50 OOS trades.
    rejected    everything else.
    ``strong`` only when the pooled (ALL) t >= 3: with ~20 variants tried, one t of
    about 2 is expected by chance alone.

    One estimand: every "mean net" above is ``mean_daily_net_bps``, the mean of the
    judged series the t-statistic belongs to (not the per-trade mean).
    """
    is_t, oos_t = is_stats["t_stat"], oos_stats["t_stat"]
    if (is_t >= VALIDATED_T and oos_t >= VALIDATED_T
            and oos_stats_2x["mean_daily_net_bps"] > 0
            and oos_stats["n_trades"] >= VALIDATED_MIN_TRADES
            and oos_stats["n_days"] >= VALIDATED_MIN_DAYS):
        label = "validated"
    elif (is_t >= EXPERIMENTAL_T_IS and oos_stats["mean_daily_net_bps"] > 0
            and oos_t >= EXPERIMENTAL_T_OOS
            and oos_stats["n_trades"] >= EXPERIMENTAL_MIN_TRADES):
        label = "experimental"
    else:
        label = "rejected"
    return {"verdict": label, "strong": bool(all_stats["t_stat"] >= STRONG_T)}


def beats_drift(oos_stats: dict) -> bool:
    """Whether the OOS result is more than the market's own drift: excess t >= 2 and > 0.

    Both numbers belong to the judged series (``mean_daily_excess_bps`` and
    ``excess_t_stat``). Kept apart from ``verdict`` (whose thresholds were fixed
    beforehand): a long-only rule can pass it in a rising market while earning exactly
    what the universe earned.
    """
    mean = oos_stats.get("mean_daily_excess_bps", oos_stats.get("mean_excess_bps"))
    return bool(oos_stats["excess_t_stat"] >= VALIDATED_T and mean is not None and mean > 0)


def label_of(verdict_label: str | None, beats: bool | None) -> str:
    """The honest label derived from the pre-registered verdict and ``beats_drift``.

    ``validated``/``experimental`` only when the verdict says so AND the rule beats
    the drift; ``beta`` when the verdict passed but the result is no more than what
    the market itself earned in the same days; ``rejected`` otherwise.
    """
    if verdict_label in ("validated", "experimental"):
        return verdict_label if beats else "beta"
    return "rejected"


def evaluate_strategy(trades: pd.DataFrame, is_days, oos_days, base_cost_bps: float, *,
                      include_oos: bool = True) -> dict:
    """IS, OOS and ALL statistics at 1x/2x/4x costs, baselines and the verdict.

    ``include_oos=False`` computes the in-sample part only; it exists so the
    exploration phase cannot even look at out-of-sample numbers by accident.
    """
    legs = explode_legs(trades)
    is_t = in_days(trades, is_days)
    result: dict = {"IS": segment_stats(trades, base_cost_bps, days=is_days, legs=legs)}
    result["cost_sensitivity"] = {
        f"{m:g}x": {"IS": _cost_row(trades, base_cost_bps * m, is_days, legs)}
        for m in COST_MULTIPLES}
    result["baselines"] = {"IS": baselines(is_t, base_cost_bps)}
    if not include_oos:
        return result
    oos_t = in_days(trades, oos_days)
    all_days = [*is_days, *oos_days]
    result["OOS"] = segment_stats(trades, base_cost_bps, days=oos_days, legs=legs)
    result["ALL"] = segment_stats(trades, base_cost_bps, days=all_days, legs=legs)
    for m in COST_MULTIPLES:
        result["cost_sensitivity"][f"{m:g}x"]["OOS"] = _cost_row(
            trades, base_cost_bps * m, oos_days, legs)
    result["baselines"]["OOS"] = baselines(oos_t, base_cost_bps)
    oos_2x = segment_stats(trades, base_cost_bps * 2, boot=False, days=oos_days, legs=legs)
    result.update(verdict(result["IS"], result["OOS"], oos_2x, result["ALL"]))
    result["beats_drift"] = beats_drift(result["OOS"])
    result["label"] = label_of(result["verdict"], result["beats_drift"])
    return result


def _cost_row(trades: pd.DataFrame, cost_bps: float, days, legs: pd.DataFrame) -> dict:
    s = segment_stats(trades, cost_bps, boot=False, days=days, legs=legs)
    return {"cost_bps": cost_bps, "mean_net_bps": s["mean_net_bps"],
            "mean_daily_net_bps": s["mean_daily_net_bps"], "t_stat": s["t_stat"]}


def attach_universe(trades: pd.DataFrame, opens: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """Add ``univ``: equal-weight average return of every symbol in the panels over each
    trade's window (entry bar open -> exit bar close).

    ``opens``/``closes`` are bar-by-symbol panels. A symbol with no bar at either end of
    the window is left out of that average. This is the drift the strategy rides along
    with: a long rule in a rising market earns ``univ`` without any skill.
    """
    if trades.empty:
        return trades.assign(univ=pd.Series(dtype=float))
    r_in = opens.index.get_indexer(pd.DatetimeIndex(trades["entry_ts"]))
    r_out = closes.index.get_indexer(pd.DatetimeIndex(trades["exit_ts"]))
    if (r_in < 0).any() or (r_out < 0).any():
        raise ValueError("a trade's entry or exit bar is missing from the universe panel")
    o = opens.to_numpy(dtype=float)[r_in]
    c = closes.to_numpy(dtype=float)[r_out]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = c / o - 1.0
    ratio[~np.isfinite(ratio)] = np.nan
    valid = (~np.isnan(ratio)).sum(axis=1)
    univ = np.where(valid > 0, np.nansum(ratio, axis=1) / np.maximum(valid, 1), 0.0)
    return trades.assign(univ=univ)


def attach_universe_paths(trades: pd.DataFrame, opens: pd.DataFrame,
                          closes: pd.DataFrame) -> pd.DataFrame:
    """Add ``path_univ``: the universe's equal-weight return on each day a trade is open.

    Same convention as the per-day ``path_ret`` of the trade: the entry day runs from
    the open to the close, later days from close to close. Symbols without a bar on
    either end of a day are left out of that day's average.
    """
    if trades.empty:
        return trades.assign(path_univ=pd.Series(dtype=object))
    o, c = opens.to_numpy(dtype=float), closes.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        oc = c / o - 1.0
        cc = np.vstack([np.full((1, c.shape[1]), np.nan), c[1:] / c[:-1] - 1.0])
    oc[~np.isfinite(oc)] = np.nan
    cc[~np.isfinite(cc)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # a day with no valid symbol
        u_oc = np.nan_to_num(np.nanmean(oc, axis=1))
        u_cc = np.nan_to_num(np.nanmean(cc, axis=1))
    index = closes.index
    out = []
    for days in trades["path_days"]:
        pos = index.get_indexer(pd.DatetimeIndex(days))
        if (pos < 0).any():
            raise ValueError("a trade's day is missing from the universe panel")
        out.append([float(u_oc[pos[0]]), *map(float, u_cc[pos[1:]])])
    return trades.assign(path_univ=out)
