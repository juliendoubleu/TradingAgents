"""The medium/long study: statistics, split and purge, verdicts, costs, risk view, fills."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tests.scanner_helpers import frame, rising
from tradingagents.scanner import horizon_research as hr, rules
from tradingagents.scanner.signals import SignalParams

N = 700
NAMES = [f"S{i}" for i in range(10)]


def _frames(seed=3):
    rng = np.random.default_rng(seed)
    out = {}
    for i, name in enumerate(NAMES):
        steps = rng.normal(0.0012 * (i - 4.5) / 4.5, 0.006, N)
        closes = 100 * np.exp(np.cumsum(steps))
        f = frame(closes)
        f["Open"] = closes * (1 + rng.normal(0, 0.002, N))     # open differs from the close
        out[name] = f
    out["BM"] = rising(N, 0.0004)
    return out


def _panel(frames):
    return rules.Panel(frames, NAMES, "BM", SignalParams())


# ------------------------------------------------------------------ statistics


@pytest.mark.unit
def test_nw_tstat_matches_a_hand_computed_case():
    # x = 1..4: mean 2.5, gamma0 = 1.25, gamma1 = 0.3125, Bartlett weight 1/2
    # long-run variance = 1.25 + 2 * 0.5 * 0.3125 = 1.5625, se = sqrt(1.5625 / 4) = 0.625
    assert hr.nw_tstat([1, 2, 3, 4], 1) == pytest.approx(4.0)
    # no lags: the plain (population-variance) t
    assert hr.nw_tstat([1, 2, 3, 4], 0) == pytest.approx(2.5 / math.sqrt(1.25 / 4))


@pytest.mark.unit
def test_nw_tstat_edge_cases_and_overlap_lowers_confidence():
    assert hr.nw_tstat([1.0], 3) == 0.0
    assert hr.nw_tstat([2.0, 2.0, 2.0], 1) == 0.0                     # no variance
    assert hr.nw_tstat([1, 2, np.nan, 4], 1) == hr.nw_tstat([1, 2, 4], 1)
    rng = np.random.default_rng(0)
    smooth = np.convolve(rng.normal(0.1, 1, 400), np.ones(5) / 5, mode="valid")  # overlapped
    assert abs(hr.nw_tstat(smooth, 4)) < abs(hr.nw_tstat(smooth, 0))


@pytest.mark.unit
def test_block_bootstrap_is_deterministic_and_brackets_the_mean():
    x = np.random.default_rng(1).normal(0.01, 0.02, 120)
    a = hr.block_bootstrap_ci(x, 5)
    assert a == hr.block_bootstrap_ci(x, 5)
    assert a[0] < x.mean() < a[1]
    assert hr.block_bootstrap_ci(np.full(30, 0.02), 4) == (pytest.approx(0.02), pytest.approx(0.02))
    assert all(math.isnan(v) for v in hr.block_bootstrap_ci([0.1], 3))
    wide = hr.block_bootstrap_ci(np.repeat(x[:24], 5), 5)             # strongly dependent data
    narrow = hr.block_bootstrap_ci(np.repeat(x[:24], 5), 1)
    assert wide[1] - wide[0] > narrow[1] - narrow[0]


@pytest.mark.unit
def test_series_stats_block():
    x = np.random.default_rng(5).normal(0.008, 0.02, 50)
    s = hr.series_stats(np.concatenate([x, [np.nan]]), 21, 5)
    assert s["n"] == 50 and s["mean_bps"] == pytest.approx(x.mean() * 1e4)
    assert s["median_bps"] == pytest.approx(np.median(x) * 1e4)
    assert s["pos_share"] == pytest.approx((x > 0).mean())
    assert s["ci_low_bps"] < s["mean_bps"] < s["ci_high_bps"]
    empty = hr.series_stats(np.array([np.nan]), 21, 5)
    assert empty["n"] == 0 and empty["mean_bps"] is None


# ------------------------------------------------------------------ split, costs, risk


@pytest.mark.unit
def test_purge_drops_in_sample_dates_whose_window_reaches_the_oos_period():
    pos = np.arange(0, 500, 5)                    # 100 decision dates, hold 21 bars
    is_idx, oos_idx, purged = hr.split_positions(pos, 21, 0.7)
    assert len(oos_idx) == 30 and oos_idx[0] == 70
    # first OOS decision at bar 350: dates with pos + 21 >= 350 reach into it
    assert list(purged) == [66, 67, 68, 69]
    assert pos[is_idx[-1]] + 21 < pos[oos_idx[0]]
    assert len(is_idx) + len(purged) == 70
    long_is, _, long_purged = hr.split_positions(np.arange(0, 21 * 100, 21), 126, 0.7)
    assert len(long_purged) == 6                  # 126 / 21 windows overlap the boundary


@pytest.mark.unit
def test_costs_are_charged_once_per_position_and_twice_for_the_spread():
    L, S, U = np.array([0.05, np.nan]), np.array([0.02, 0.01]), np.array([0.03, 0.03])
    ser = hr.return_series(L, S, U, 0.001)
    assert ser["long_excess"][0] == pytest.approx(0.05 - 0.03 - 0.001)
    assert ser["long_abs"][0] == pytest.approx(0.049)
    assert ser["short_pnl"][1] == pytest.approx(-0.01 - 0.001)
    assert ser["short_excess"][0] == pytest.approx(0.03 - 0.02 - 0.001)
    assert ser["spread"][0] == pytest.approx(0.05 - 0.02 - 0.002)
    assert np.isnan(ser["long_excess"][1]) and np.isnan(ser["spread"][1])   # empty basket


@pytest.mark.unit
def test_risk_stats_arithmetic():
    r = hr.risk_stats([0.10, -0.10, 0.10, -0.10], 1)
    equity = np.array([1.1, 0.99, 1.089, 0.9801])
    assert r["cagr"] == pytest.approx(equity[-1] ** (1 / 4) - 1)
    assert r["max_drawdown"] == pytest.approx(0.9801 / 1.1 - 1)
    assert r["vol"] == pytest.approx(np.std([0.1, -0.1, 0.1, -0.1], ddof=1))
    assert r["sharpe"] == pytest.approx(0.0)
    assert hr.risk_stats([0.01], 12)["cagr"] is None


# ------------------------------------------------------------------ verdicts


def _stats(n, mean, t):
    return {"n": n, "mean_bps": mean, "nw_t": t}


def _series(is_t, oos_t, oos_mean=50.0, oos_n=40, all_t=2.0):
    return {"IS": _stats(100, 40.0, is_t), "OOS": _stats(oos_n, oos_mean, oos_t),
            "ALL": _stats(140, 45.0, all_t)}


def _neutral():
    return _series(0.0, 0.0, 0.0, all_t=0.0)


@pytest.mark.unit
def test_long_verdict_levels_and_the_strong_flag():
    ser = {"long_excess": _series(2.4, 2.3, all_t=3.2), "long_abs": _neutral(),
           "short_pnl": _neutral(), "short_excess": _neutral()}
    v = hr.judge_side("long", ser, {"long_excess": 20.0})
    assert v["verdict"] == "validated" and v["strong"] is True
    assert hr.judge_side("long", {**ser, "long_excess": _series(2.4, 2.3, all_t=2.5)},
                         {"long_excess": 20.0})["strong"] is False
    # fails at 2x costs -> experimental, not validated
    assert hr.judge_side("long", ser, {"long_excess": -1.0})["verdict"] == "experimental"
    # too few OOS dates for validated, enough for experimental
    few = {**ser, "long_excess": _series(2.4, 2.3, oos_n=25)}
    assert hr.judge_side("long", few, {"long_excess": 20.0})["verdict"] == "experimental"
    # OOS t between 1 and 2 -> experimental; below 1 or negative mean -> unproven
    assert hr.judge_side("long", {**ser, "long_excess": _series(2.4, 1.4)},
                         {"long_excess": 10.0})["verdict"] == "experimental"
    assert hr.judge_side("long", {**ser, "long_excess": _series(2.4, 0.8)},
                         {"long_excess": 10.0})["verdict"] == "unproven"
    assert hr.judge_side("long", {**ser, "long_excess": _series(1.5, 2.9)},
                         {"long_excess": 10.0})["verdict"] == "unproven"   # IS decides


@pytest.mark.unit
def test_short_side_relative_means_lagging_the_market_not_falling():
    ser = {"long_excess": _neutral(), "long_abs": _neutral(),
           "short_pnl": _series(-1.0, -2.0, oos_mean=-80.0), "short_excess": _series(2.5, 2.2)}
    v = hr.judge_side("short", ser, {"short_pnl": -90.0, "short_excess": 20.0})
    assert v["verdict"] == "relative" and v["judged"] == "short_pnl"
    real = {**ser, "short_pnl": _series(2.5, 2.2)}
    assert hr.judge_side("short", real, {"short_pnl": 20.0})["verdict"] == "validated"
    none = {**ser, "short_excess": _series(0.2, 0.1)}
    assert hr.judge_side("short", none, {"short_pnl": -90.0})["verdict"] == "unproven"


@pytest.mark.unit
def test_beta_is_a_long_that_only_rode_the_market():
    ser = {"long_excess": _series(0.5, 0.2, oos_mean=5.0), "long_abs": _series(3.0, 2.5),
           "short_pnl": _neutral(), "short_excess": _neutral()}
    assert hr.judge_side("long", ser, {"long_abs": 30.0, "long_excess": -5.0})["verdict"] == "beta"
    # it did beat the universe out of sample (but failed in-sample): not "beta"
    ser["long_excess"] = _series(0.5, 2.6, oos_mean=300.0)
    assert hr.judge_side("long", ser, {"long_abs": 30.0, "long_excess": 5.0})["verdict"] == "unproven"


# ------------------------------------------------------------------ fills and baskets


@pytest.mark.unit
def test_window_return_fills_at_the_next_open_and_exits_at_the_close_of_t_plus_hold():
    frames = _frames()
    panel = _panel(frames)
    grid = hr.Grid(panel, "medium")
    p = 7
    i = int(grid.pos[p])
    for s, name in enumerate(panel.members):
        f = frames[name]
        expect = f["Close"].iloc[i + grid.hold] / f["Open"].iloc[i + 1] - 1.0
        assert grid.R[p, s] == pytest.approx(expect)
    assert grid.U[p] == pytest.approx(np.nanmean(grid.R[p]))
    assert grid.hold == 21 and grid.step == 5
    long_grid = hr.Grid(panel, "long")
    assert long_grid.hold == 126 and long_grid.step == 21


@pytest.mark.unit
def test_an_instrument_without_the_entry_or_exit_bar_is_left_out_of_that_date():
    frames = _frames()
    base = hr.Grid(_panel(frames), "medium")
    p = 6
    i = int(base.pos[p])
    gap = frames["S3"].index[i + 1]                       # no bar to enter on
    frames["S3"] = frames["S3"].drop(gap)
    grid = hr.Grid(_panel(frames), "medium")
    col = grid_col = list(grid.panel.members).index("S3")
    assert np.isnan(grid.R[p, grid_col]) and col == 3
    others = [c for c in range(10) if c != 3]
    assert grid.U[p] == pytest.approx(np.nanmean(grid.R[p, others]))


@pytest.mark.unit
def test_basket_means_and_series_for_known_sides():
    grid = hr.Grid(_panel(_frames()), "medium")
    sides = np.zeros_like(grid.R, dtype=int)
    sides[:, 0] = 1
    sides[:, 1] = 1
    sides[:, 9] = -1
    L, S, nL, nS = grid.basket_means(sides)
    assert np.allclose(L, (grid.R[:, 0] + grid.R[:, 1]) / 2) and np.allclose(S, grid.R[:, 9])
    assert (nL == 2).all() and (nS == 1).all()


@pytest.mark.unit
def test_stop_mode_fills_a_gap_through_the_stop_at_the_open():
    frames = _frames()
    base = hr.Grid(_panel(frames), "medium")
    p = 10
    i = int(base.pos[p])
    j = i + 4
    f = frames["S9"].copy()
    idx = f.index[j]
    f.loc[idx, ["Open", "High", "Low", "Close"]] = f["Open"].iloc[i + 1] * 0.7   # gap down
    frames["S9"] = f
    grid = hr.Grid(_panel(frames), "medium")
    sides = np.zeros_like(grid.R, dtype=int)
    sides[p, 9] = 1
    out = grid.stop_returns(sides, 2.5)
    assert out[p, 9] == pytest.approx(0.7 - 1.0)                  # filled at the gap open
    assert np.isnan(out[p, :9]).all()                            # no position, no trade
    # a short in the same bar would not have been stopped by a fall
    sides[p, 9] = -1
    short = grid.stop_returns(sides, 2.5)
    assert short[p, 9] != pytest.approx(0.7 - 1.0)


# ------------------------------------------------------------------ study, snapshot, report


def _family(frames):
    panel = _panel(frames)
    info = SimpleNamespace(name="us", benchmark="BM", members=NAMES, use_vix=False)
    return hr.Family("stocks_us", info, frames, panel, 10.0, SignalParams())


@pytest.mark.unit
def test_study_snapshot_round_trip_and_report(tmp_path):
    fam = _family(_frames())
    study = hr.run_study("2026-01-01", None, ["stocks_us"], ["T1", "T2", "V1"],
                         loaded={"stocks_us": fam})
    assert {(e["rule"], e["horizon"], e["side"]) for e in study.entries} == {
        (r, h, s) for r in ("T1", "T2", "V1") for h in ("medium", "long")
        for s in ("long", "short")}
    path = hr.write_snapshot(study, tmp_path / "evidence.json")
    snap = json.loads(path.read_text(encoding="utf-8"))
    assert snap["version"] == 1 and snap["study_date"] == "2026-01-01"
    assert snap["counts"]["evaluations_total"] == 2 * 1 * (2 * 2 + 2)
    assert snap["counts"]["variants"] == 1
    assert hr.load_snapshot(path) == snap
    entry = snap["results"][0]
    assert {"family", "rule", "horizon", "side", "verdict", "strong", "judged", "series", "stop",
            "baskets", "pooled_t"} <= set(entry)
    assert set(entry["series"][entry["judged"]]) == {"IS", "OOS", "ALL"}
    assert snap["rules"]["T1"]["name_de"] and snap["rules"]["V1"]["variant"] is True
    assert snap["families"]["stocks_us"]["horizons"]["long"]["purged_n"] > 0
    assert snap["risk"] and set(snap["risk"][0]["ALL"]) == {"basket", "universe", "benchmark",
                                                           "in_market"}
    text = hr.write_report(snap, tmp_path / "bericht.md").read_text(encoding="utf-8")
    assert "Kurzfassung" in text and "Survivorship" in text and "Grenzen" in text


@pytest.mark.unit
def test_exploration_never_computes_out_of_sample_numbers():
    fam = _family(_frames())
    study = hr.run_study("2026-01-01", None, ["stocks_us"], ["T1"], include_oos=False,
                         boot=False, loaded={"stocks_us": fam})
    for e in study.entries:
        assert "verdict" not in e
        assert set(e["series"][e["judged"]]) == {"IS"}


@pytest.mark.unit
def test_missing_snapshot_is_none_not_an_error(tmp_path):
    assert hr.load_snapshot(tmp_path / "nope.json") is None
