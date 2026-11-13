"""Evaluation: the IS/OOS boundary, day-clustered significance, excess arithmetic, verdicts."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.scanner.shortterm import evaluate as ev


def _trades(rows):
    """rows of (date, direction, raw, univ)."""
    df = pd.DataFrame(rows, columns=["date", "direction", "raw", "univ"])
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = "X"
    df["window"] = df["raw"]
    return df


@pytest.mark.unit
def test_split_is_chronological_70_30_and_never_shuffled():
    days = pd.bdate_range("2024-01-01", periods=10)
    is_days, oos_days = ev.split_days(list(days[::-1]))        # unsorted input
    assert len(is_days) == 7 and len(oos_days) == 3
    assert max(is_days) < min(oos_days)
    assert is_days == list(days[:7])


@pytest.mark.unit
def test_day_clustered_t_stat_on_a_hand_computed_case():
    # daily means: (0.01+0.03)/2 = 0.02, 0.00, 0.04  -> mean 0.02, sd 0.02, n 3
    t = _trades([("2024-01-02", 1, 0.01, 0), ("2024-01-02", 1, 0.03, 0),
                 ("2024-01-03", 1, 0.00, 0), ("2024-01-04", 1, 0.04, 0)])
    daily = ev.daily_series(ev.with_net(t, 0.0))
    assert daily.tolist() == pytest.approx([0.02, 0.0, 0.04])
    assert ev.t_stat(daily) == pytest.approx(np.sqrt(3))
    stats = ev.segment_stats(t, 0.0)
    assert stats["n_trades"] == 4 and stats["n_days"] == 3
    assert stats["t_stat"] == pytest.approx(np.sqrt(3))
    assert stats["mean_net_bps"] == pytest.approx(200.0)       # (1+3+0+4)/4 percent = 2 %
    assert stats["win_rate"] == pytest.approx(0.75)


@pytest.mark.unit
def test_costs_reduce_the_net_and_bootstrap_is_deterministic():
    t = _trades([(f"2024-01-{d:02d}", 1, 0.002, 0) for d in range(1, 21)])
    s1, s2 = ev.segment_stats(t, 5.0), ev.segment_stats(t, 5.0)
    assert s1["mean_net_bps"] == pytest.approx(15.0)
    assert (s1["ci_low_bps"], s1["ci_high_bps"]) == (s2["ci_low_bps"], s2["ci_high_bps"])


@pytest.mark.unit
def test_excess_is_return_minus_universe_and_mirrored_for_shorts():
    t = _trades([("2024-01-02", 1, 0.02, 0.005), ("2024-01-03", -1, -0.02, 0.005)])
    out = ev.with_net(t, 10.0)
    assert out["excess"].iloc[0] == pytest.approx(0.015 - 0.001)    # long: ret - universe
    assert out["excess"].iloc[1] == pytest.approx(0.025 - 0.001)    # short: universe - ret


@pytest.mark.unit
def test_attach_universe_averages_the_panel_over_the_trade_window():
    idx = pd.bdate_range("2024-01-01", periods=3)
    opens = pd.DataFrame({"A": [100, 100, 100], "B": [50, 50, 50]}, index=idx, dtype=float)
    closes = pd.DataFrame({"A": [101, 102, 110], "B": [50, 55, 50]}, index=idx, dtype=float)
    trades = pd.DataFrame({"entry_ts": [idx[0]], "exit_ts": [idx[1]]})
    out = ev.attach_universe(trades, opens, closes)
    assert out["univ"].iloc[0] == pytest.approx((0.02 + 0.10) / 2)


def _stats(t, n_trades=200, n_days=30, mean=10.0):
    # the verdict reads the mean of the judged daily series; the per-trade mean is a decoy
    return {"t_stat": t, "n_trades": n_trades, "n_days": n_days,
            "mean_daily_net_bps": mean, "mean_net_bps": 1e6 if mean < 0 else -1e6}


@pytest.mark.unit
@pytest.mark.parametrize("is_t, oos_t, oos2x, n, days, expected", [
    (2.5, 2.2, 3.0, 150, 25, "validated"),
    (2.5, 2.2, -1.0, 150, 25, "experimental"),       # dies at 2x costs
    (2.5, 2.2, 3.0, 150, 12, "experimental"),        # too few OOS days
    (2.5, 1.4, 3.0, 80, 30, "experimental"),
    (2.5, 0.8, 3.0, 150, 30, "rejected"),            # OOS t below 1
    (1.9, 3.0, 3.0, 150, 30, "rejected"),            # IS t below 2
    (2.5, 2.2, 3.0, 40, 30, "rejected"),             # fewer than 50 OOS trades
    (-3.0, -3.0, -3.0, 500, 100, "rejected"),        # significantly negative is not an edge
])
def test_verdict_rule(is_t, oos_t, oos2x, n, days, expected):
    oos_mean = 5.0 if oos_t > 0 else -5.0
    out = ev.verdict(_stats(is_t), _stats(oos_t, n, days, oos_mean),
                     _stats(0, mean=oos2x), _stats(2.0))
    assert out["verdict"] == expected


@pytest.mark.unit
def test_strong_needs_a_pooled_t_of_three():
    base = (_stats(2.5), _stats(2.5), _stats(0, mean=3.0))
    assert ev.verdict(*base, _stats(2.99))["strong"] is False
    assert ev.verdict(*base, _stats(3.0))["strong"] is True


@pytest.mark.unit
def test_in_sample_only_evaluation_never_touches_oos():
    days = pd.bdate_range("2024-01-01", periods=10)
    t = _trades([(d, 1, 0.01, 0.0) for d in days])
    is_days, oos_days = ev.split_days(days)
    res = ev.evaluate_strategy(t, is_days, oos_days, 5.0, include_oos=False)
    assert "OOS" not in res and "verdict" not in res
    full = ev.evaluate_strategy(t, is_days, oos_days, 5.0)
    assert full["IS"]["n_days"] == 7 and full["OOS"]["n_days"] == 3 and full["ALL"]["n_days"] == 10


@pytest.mark.unit
def test_random_baseline_band_contains_zero_gross_direction_skill():
    rng = np.random.default_rng(1)
    t = _trades([(f"2024-02-{d:02d}", 1, r, 0.0) for d, r in
                 enumerate(rng.normal(0, 0.01, 25), start=1)])
    b = ev.baselines(t, 0.0)
    assert b["random_lo_bps"] < 0 < b["random_hi_bps"]


@pytest.mark.unit
def test_beats_drift_needs_a_significant_positive_excess():
    assert ev.beats_drift({"excess_t_stat": 2.1, "mean_excess_bps": 5.0})
    assert not ev.beats_drift({"excess_t_stat": 2.1, "mean_excess_bps": -5.0})
    assert not ev.beats_drift({"excess_t_stat": 0.8, "mean_excess_bps": 5.0})


@pytest.mark.unit
def test_verdict_and_beats_drift_read_the_mean_of_the_judged_series_not_the_per_trade_mean():
    # per-trade mean positive, daily mean negative (sell-off days with many signals carry
    # the per-trade mean): the t-stat's own series says no edge, so no verdict clause passes
    oos = dict(_stats(2.5, 150, 30, -5.0), mean_net_bps=+50.0)
    oos_2x = dict(_stats(0, mean=-3.0), mean_net_bps=+40.0)
    assert ev.verdict(_stats(2.5), oos, oos_2x, _stats(2.0))["verdict"] == "rejected"
    # the cost clause too: daily mean at 2x costs negative -> not validated, however the
    # per-trade mean looks
    ok = _stats(2.5, 150, 30, 5.0)
    assert ev.verdict(_stats(2.5), ok, oos_2x, _stats(2.0))["verdict"] == "experimental"
    assert not ev.beats_drift({"excess_t_stat": 2.5, "mean_excess_bps": 9.0,
                               "mean_daily_excess_bps": -1.0})
    assert ev.beats_drift({"excess_t_stat": 2.5, "mean_excess_bps": -9.0,
                           "mean_daily_excess_bps": 1.0})


@pytest.mark.unit
@pytest.mark.parametrize("verdict, beats, expected", [
    ("validated", True, "validated"), ("validated", False, "beta"),
    ("experimental", True, "experimental"), ("experimental", False, "beta"),
    ("rejected", True, "rejected"), ("rejected", False, "rejected"), (None, None, "rejected"),
])
def test_label_is_the_verdict_only_when_the_rule_beats_the_drift(verdict, beats, expected):
    assert ev.label_of(verdict, beats) == expected


def _swing_trades():
    """Two overlapping long trades: A on days 1-3 (exits day 3), B on days 2-3 (exits day 3)."""
    d = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    rows = [
        {"symbol": "A", "date": d[0], "direction": 1, "raw": 0.02, "univ": 0.0, "window": 0.02,
         "path_days": list(d), "path_ret": [0.01, 0.02, -0.01], "path_univ": [0.0, 0.0, 0.0]},
        {"symbol": "B", "date": d[1], "direction": 1, "raw": 0.04, "univ": 0.0, "window": 0.04,
         "path_days": list(d[1:]), "path_ret": [0.03, 0.01], "path_univ": [0.0, 0.0]},
    ]
    return pd.DataFrame(rows), d


@pytest.mark.unit
def test_swing_trades_are_judged_on_the_calendar_time_portfolio_with_cost_on_the_exit_day():
    trades, d = _swing_trades()
    legs = ev.explode_legs(trades)
    assert len(legs) == 5 and legs["last"].sum() == 2
    ser = ev.judged_series(legs, 10.0)["net"]
    assert ser.index.tolist() == list(d)
    # day 1: A only; day 2: mean(0.02, 0.03); day 3 both exit: mean(-0.01-0.001, 0.01-0.001)
    assert ser.tolist() == pytest.approx([0.01, 0.025, -0.001])
    stats = ev.segment_stats(trades, 10.0, boot=False)
    assert stats["n_days"] == 3 and stats["judged_series"] == "calendar_time_portfolio"
    assert stats["mean_daily_net_bps"] == pytest.approx(np.mean([0.01, 0.025, -0.001]) * 1e4)
    assert stats["t_stat"] == pytest.approx(ev.t_stat(ser))
    # the entry-day estimand differs: it would see 2 "days" and ignore the 3-day overlap
    assert stats["t_stat"] != pytest.approx(ev.t_stat(ev.daily_series(ev.with_net(trades, 10.0))))
    # excess subtracts the universe's daily return on each open day
    trades["path_univ"] = [[0.005, 0.005, 0.005], [0.005, 0.005]]
    ex = ev.judged_series(ev.explode_legs(trades), 10.0)["excess"]
    assert ex.tolist() == pytest.approx([0.005, 0.020, -0.006])


@pytest.mark.unit
def test_sides_and_days_restrict_the_series():
    trades, d = _swing_trades()
    legs = ev.explode_legs(trades)
    only_late = ev.judged_series(legs, 0.0, days=d[1:])["net"]
    assert only_late.index.tolist() == list(d[1:])
    stats = ev.segment_stats(trades, 10.0, boot=False, days=d[:2], legs=legs)
    assert stats["n_trades"] == 2 and stats["n_days"] == 2      # both trades ENTERED on d1, d2
    assert stats["short"]["n_trades"] == 0 and stats["long"]["n_trades"] == 2


@pytest.mark.unit
def test_drawdown_is_on_the_compounded_curve_and_never_exceeds_100_percent():
    days = pd.bdate_range("2024-01-01", periods=4)
    t = _trades([(d, 1, r, 0.0) for d, r in zip(days, [-0.5, -0.5, -0.9, -0.9], strict=True)])
    stats = ev.segment_stats(t, 0.0, boot=False)
    assert stats["max_drawdown_pct"] == pytest.approx((1 - 0.5 * 0.5 * 0.1 * 0.1) * 100)
    assert stats["max_drawdown_pct"] <= 100.0
    assert ev._max_drawdown_pct(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(50.0)


@pytest.mark.unit
def test_day_trades_keep_the_entry_day_series():
    t = _trades([("2024-01-02", 1, 0.01, 0), ("2024-01-02", 1, 0.03, 0),
                 ("2024-01-03", 1, 0.00, 0), ("2024-01-04", 1, 0.04, 0)])
    stats = ev.segment_stats(t, 0.0)
    assert stats["judged_series"] == "entry_day_mean"
    assert stats["mean_daily_net_bps"] == pytest.approx(200.0)      # (0.02 + 0 + 0.04) / 3
    assert stats["mean_net_bps"] == pytest.approx(200.0)
