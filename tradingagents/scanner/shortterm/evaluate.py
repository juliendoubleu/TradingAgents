"""Honest evaluation of simulated trades: IS/OOS split, day-clustered significance,
skill-versus-drift and the pre-registered verdict rule.

Why the machinery: trades on the same day are strongly correlated (the whole market
moves together), so a t-statistic over trades would be wildly overconfident. Here
every statistic of "does it make money" is computed on DAILY aggregates (mean net
return of that day's trades) and the confidence interval resamples whole days. The
chronological 70/30 split is fixed before looking at anything; parameters are chosen
on the in-sample (IS) part only and the out-of-sample (OOS) part is evaluated once
per final candidate and reported whatever it shows.

A ``trades`` frame has one row per simulated trade with the columns:
``symbol, date, direction, raw, univ, window`` where ``date`` is the session day of
the entry (naive, midnight), ``raw`` the symbol's own return entry -> exit
(``exit/entry - 1``, before direction and costs), ``univ`` the equal-weight average
return of the whole research universe over the same entry-open -> exit-bar-close
window, and ``window`` the symbol's own return over that bar window with no stop
(used only by the baselines).
"""

from __future__ import annotations

import math

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


def _side_stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"n_trades": 0, "mean_net_bps": 0.0, "win_rate": 0.0, "t_stat": 0.0}
    return {
        "n_trades": int(len(t)),
        "mean_net_bps": float(t["net"].mean() * 1e4),
        "win_rate": float((t["net"] > 0).mean()),
        "t_stat": t_stat(daily_series(t)),
    }


def segment_stats(trades: pd.DataFrame, cost_bps: float, *, boot: bool = True) -> dict:
    """The full statistics block for one set of trades at one cost level."""
    t = with_net(trades, cost_bps)
    daily = daily_series(t)
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
    stats["sharpe"] = float(daily.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
    cum = daily.cumsum()
    stats["max_drawdown_pct"] = float((cum.cummax() - cum).max() * 100) if len(cum) else 0.0
    ex = daily_series(t, "excess")
    stats["mean_excess_bps"] = float(t["excess"].mean() * 1e4) if n else 0.0
    stats["excess_t_stat"] = t_stat(ex)
    stats["long"] = _side_stats(t[t["direction"] > 0])
    stats["short"] = _side_stats(t[t["direction"] < 0])
    return stats


def in_days(trades: pd.DataFrame, days) -> pd.DataFrame:
    return trades[trades["date"].isin(set(pd.to_datetime(list(days))))]


def baselines(trades: pd.DataFrame, cost_bps: float, *, n_random: int = N_RANDOM,
              seed: int = SEED) -> dict:
    """What the same trade windows earn with no direction skill.

    ``always_long``: every trade of the strategy taken long over its window (no
    stop). ``strategy_dir``: the strategy's own directions over the same windows,
    also without stops, so the comparison with the random band isolates direction
    skill from the stop. ``random``: random directions, same trades, ``n_random``
    repetitions, mean and 95 % band of the mean net bps per trade.
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

    validated   IS t >= 2 AND OOS t >= 2 (both positive, net of base costs) AND OOS
                mean net > 0 at 2x costs AND >= 100 OOS trades over >= 20 OOS days.
    experimental IS t >= 2 AND OOS mean net > 0 at base cost AND OOS t >= 1 AND
                >= 50 OOS trades.
    rejected    everything else.
    ``strong`` only when the pooled (ALL) t >= 3: with ~20 variants tried, one t of
    about 2 is expected by chance alone.
    """
    is_t, oos_t = is_stats["t_stat"], oos_stats["t_stat"]
    if (is_t >= VALIDATED_T and oos_t >= VALIDATED_T
            and oos_stats_2x["mean_net_bps"] > 0
            and oos_stats["n_trades"] >= VALIDATED_MIN_TRADES
            and oos_stats["n_days"] >= VALIDATED_MIN_DAYS):
        label = "validated"
    elif (is_t >= EXPERIMENTAL_T_IS and oos_stats["mean_net_bps"] > 0
            and oos_t >= EXPERIMENTAL_T_OOS
            and oos_stats["n_trades"] >= EXPERIMENTAL_MIN_TRADES):
        label = "experimental"
    else:
        label = "rejected"
    return {"verdict": label, "strong": bool(all_stats["t_stat"] >= STRONG_T)}


def evaluate_strategy(trades: pd.DataFrame, is_days, oos_days, base_cost_bps: float, *,
                      include_oos: bool = True) -> dict:
    """IS, OOS and ALL statistics at 1x/2x/4x costs, baselines and the verdict.

    ``include_oos=False`` computes the in-sample part only; it exists so the
    exploration phase cannot even look at out-of-sample numbers by accident.
    """
    is_t = in_days(trades, is_days)
    result: dict = {"IS": segment_stats(is_t, base_cost_bps)}
    result["cost_sensitivity"] = {
        f"{m:g}x": {"IS": _cost_row(is_t, base_cost_bps * m)} for m in COST_MULTIPLES}
    result["baselines"] = {"IS": baselines(is_t, base_cost_bps)}
    if not include_oos:
        return result
    oos_t = in_days(trades, oos_days)
    all_t = pd.concat([is_t, oos_t])
    result["OOS"] = segment_stats(oos_t, base_cost_bps)
    result["ALL"] = segment_stats(all_t, base_cost_bps)
    for m in COST_MULTIPLES:
        result["cost_sensitivity"][f"{m:g}x"]["OOS"] = _cost_row(oos_t, base_cost_bps * m)
    result["baselines"]["OOS"] = baselines(oos_t, base_cost_bps)
    oos_2x = segment_stats(oos_t, base_cost_bps * 2, boot=False)
    result.update(verdict(result["IS"], result["OOS"], oos_2x, result["ALL"]))
    return result


def _cost_row(t: pd.DataFrame, cost_bps: float) -> dict:
    s = segment_stats(t, cost_bps, boot=False)
    return {"cost_bps": cost_bps, "mean_net_bps": s["mean_net_bps"], "t_stat": s["t_stat"]}
