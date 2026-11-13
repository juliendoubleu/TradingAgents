"""Candidate strategies on constructed series: fire when they should, stay quiet when
they should, and never look ahead."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.scanner.shortterm import StrategySpec, daily_features, daytrade as dt
from tradingagents.scanner.shortterm.swing import ALL_DAILY_SIGNALS, daily_signal

N = 330


def _frame(close, *, vol=None, gap=None):
    """Daily bars around a close path: ~0.5 % ranges, open = previous close (+ optional gap)."""
    close = np.asarray(close, dtype=float)
    open_ = np.r_[close[0], close[:-1]]
    if gap is not None:
        open_ = open_ * (1 + np.asarray(gap))
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    volume = np.full(len(close), 1_000_000.0) if vol is None else np.asarray(vol, dtype=float)
    idx = pd.bdate_range("2022-01-03", periods=len(close))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": volume}, index=idx)


def _uptrend(n=N, step=0.0015, wobble=0.002):
    base = 100 * np.exp(np.cumsum(np.full(n, step)))
    return base * (1 + wobble * np.sin(np.arange(n) * 0.9))


def _last(family, df, params=None):
    return int(daily_signal(family, daily_features(df), params or {}).iloc[-1])


@pytest.mark.unit
def test_d5_and_s1_fire_on_an_oversold_pullback_in_an_uptrend_only():
    close = _uptrend()
    close[-3:] = close[-4] * np.array([0.985, 0.97, 0.955])     # three hard down closes
    df = _frame(close)
    assert _last("D5", df) == 1 and _last("S1", df) == 1
    assert _last("D5", _frame(_uptrend())) == 0 and _last("S1", _frame(_uptrend())) == 0


@pytest.mark.unit
def test_d5_and_s1_short_mirror_needs_a_downtrend():
    close = _uptrend(step=-0.0015)
    close[-3:] = close[-4] * np.array([1.015, 1.03, 1.045])
    df = _frame(close)
    assert _last("D5", df) == -1 and _last("S1", df) == -1


@pytest.mark.unit
def test_d3_washout_fires_on_a_big_drop_closing_at_the_low():
    close = _uptrend()
    close[-1] = close[-2] * 0.93
    df = _frame(close)
    df.iloc[-1, df.columns.get_loc("Low")] = close[-1]           # closes at the low
    assert _last("D3", df) == 1
    assert _last("D3", _frame(_uptrend())) == 0


@pytest.mark.unit
def test_d4_trend_day_needs_close_at_the_high_on_heavy_volume():
    close = _uptrend()
    close[-1] = close[-2] * 1.02
    vol = np.full(N, 1e6)
    vol[-1] = 3e6
    df = _frame(close, vol=vol)
    df.iloc[-1, df.columns.get_loc("High")] = close[-1]
    assert _last("D4", df) == 1
    quiet = df.copy()
    quiet.iloc[-1, quiet.columns.get_loc("Volume")] = 1e6
    assert _last("D4", quiet) == 0


@pytest.mark.unit
def test_s2_reversal_fires_after_an_extreme_five_day_drop_in_an_uptrend():
    rng = np.random.default_rng(3)
    close = _uptrend(step=0.003) * np.exp(np.cumsum(rng.normal(0, 0.004, N)))
    close[-5:] = close[-6] * np.linspace(0.99, 0.92, 5)
    assert _last("S2", _frame(close)) == 1
    assert _last("S2", _frame(_uptrend())) == 0


@pytest.mark.unit
def test_s3_breakout_needs_a_new_20_day_high_on_volume():
    close = _uptrend(wobble=0.0)
    close[-1] = close[-2] * 1.03
    vol = np.full(N, 1e6)
    vol[-1] = 2.5e6
    assert _last("S3", _frame(close, vol=vol)) == 1
    vol[-1] = 1e6
    assert _last("S3", _frame(close, vol=vol)) == 0


@pytest.mark.unit
def test_s4_post_gap_drift_needs_gap_volume_and_a_confirming_close():
    close = _uptrend(wobble=0.0)
    gap = np.zeros(N)
    gap[-1] = 0.06
    close[-1] = close[-2] * 1.07                                # closes above the gapped open
    vol = np.full(N, 1e6)
    vol[-1] = 3e6
    assert _last("S4", _frame(close, vol=vol, gap=gap)) == 1
    close[-1] = close[-2] * 1.03                                # faded below the open
    assert _last("S4", _frame(close, vol=vol, gap=gap)) == 0


@pytest.mark.unit
def test_d1_fades_a_moderate_gap_against_the_trend_and_d2_follows_a_big_one():
    gap = np.zeros(N)
    gap[-1] = -0.008                                            # ~0.5 ATR gap down in an uptrend
    assert _last("D1", _frame(_uptrend(), gap=gap)) == 1
    assert _last("D2", _frame(_uptrend(), gap=gap)) == 0
    gap[-1] = 0.05
    assert _last("D2", _frame(_uptrend(), gap=gap)) == 1
    assert _last("D1", _frame(_uptrend(), gap=gap)) == 0
    assert _last("D1", _frame(_uptrend(), gap=np.zeros(N))) == 0


def _noisy(seed, n=700, gap_sd=0.012, vol_sigma=0.5, inject_gaps=False):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    gap = rng.normal(0, gap_sd, n)
    vol = rng.lognormal(13.8, vol_sigma, n)
    if inject_gaps:            # now and then a big, heavy-volume gap the day confirms
        for i in range(260, n - 1, 37):
            gap[i] = 0.08 * (1 if i % 2 else -1)
            close[i:] *= (1 + gap[i]) * (1 + 0.01 * np.sign(gap[i]))
            vol[i] = 6e6
    return _frame(close, vol=vol, gap=gap)


@pytest.mark.unit
@pytest.mark.parametrize("family", ["D3", "D4", "D5", "S1", "S2", "S3", "S4"])
def test_close_based_signals_do_not_look_ahead(family):
    df = _noisy(7, inject_gaps=True) if family == "S4" else _noisy(7)
    full = daily_signal(family, daily_features(df), {}).to_numpy()
    assert full.any(), "the test series must actually fire the signal"
    for cut in (320, 450, 600):
        prefix = daily_signal(family, daily_features(df.iloc[: cut + 1]), {}).to_numpy()
        assert (prefix == full[: cut + 1]).all()


@pytest.mark.unit
@pytest.mark.parametrize("family", ["D1", "D2"])
def test_open_based_signals_ignore_everything_but_the_open_of_the_entry_day(family):
    df = _noisy(11)
    full = daily_signal(family, daily_features(df), {}).to_numpy()
    assert full.any()
    k = 500
    altered = df.copy()
    altered.iloc[k:, altered.columns.get_loc("Close")] *= 1.7
    altered.iloc[k:, altered.columns.get_loc("High")] *= 1.9
    altered.iloc[k:, altered.columns.get_loc("Low")] *= 0.5
    altered.iloc[k:, altered.columns.get_loc("Volume")] *= 9
    # the open of day k (and earlier history) is unchanged, so the signal at k must be too
    again = daily_signal(family, daily_features(altered), {}).to_numpy()
    assert again[k] == full[k] and (again[:k] == full[:k]).all()


@pytest.mark.unit
def test_run_daily_enters_at_the_next_open_and_leaves_at_that_days_close():
    close = _uptrend()
    close[-3:] = close[-4] * np.array([0.985, 0.97, 0.955])
    close = np.r_[close, close[-1] * 1.01]                      # the entry day
    df = _frame(close)
    feats = {"AAA": daily_features(df)}
    spec = StrategySpec("D5", "D5", "day", "1d", "x", "x", {"lo": 5, "hi": 95, "stop_atr": 1.0, "hold": 1})
    rows = dt.run_daily(spec, feats, lambda f, p: daily_signal("D5", f, p),
                        first_day=df.index[0])
    row = [r for r in rows if r["entry_ts"] == df.index[-1]][0]
    assert row["direction"] == 1
    entry_open = df["Open"].iloc[-1]
    assert row["raw"] == pytest.approx(df["Close"].iloc[-1] / entry_open - 1, abs=0.02)
    assert row["exit_ts"] == df.index[-1]


# ----------------------------------------------------------------------- intraday setups


def _day60(prev_close=100.0, atr_pct=0.01):
    ctx = {"prev_close": prev_close, "atr": prev_close * atr_pct, "atr_pct": atr_pct,
           "up": True, "down": False}
    o = np.full(7, 100.0)
    return o.copy(), o + 0.3, o - 0.3, o.copy(), np.full(7, 1e6), np.arange(570, 990, 60), ctx


def _call(fn, o, h, low, c, v, mins, ctx, params=None):
    return fn(o, h, low, c, v, mins, ctx, params or {})


@pytest.mark.unit
def test_h1_follows_the_day_drift_into_the_last_hour_and_is_quiet_on_a_flat_day():
    o, h, low, c, v, m, ctx = _day60()
    c[5] = 101.0
    s = _call(dt.setup_h1, o, h, low, c, v, m, ctx)
    assert (s.entry_idx, s.direction) == (6, 1)
    o, h, low, c, v, m, ctx = _day60()
    assert _call(dt.setup_h1, o, h, low, c, v, m, ctx) is None


@pytest.mark.unit
def test_h2_breaks_out_of_the_first_hour_with_the_midpoint_stop():
    o, h, low, c, v, m, ctx = _day60()
    c[2] = 100.6                                                # closes above first-hour high 100.3
    o[3] = 100.6
    s = _call(dt.setup_h2, o, h, low, c, v, m, ctx)
    assert (s.entry_idx, s.direction) == (3, 1) and s.stop == pytest.approx(100.0)
    o, h, low, c, v, m, ctx = _day60()
    assert _call(dt.setup_h2, o, h, low, c, v, m, ctx) is None


@pytest.mark.unit
def test_h3_fades_a_big_first_hour_with_the_first_hour_extreme_as_stop():
    o, h, low, c, v, m, ctx = _day60()
    o[0], c[0], low[0] = 100.0, 98.5, 98.3                      # -1.5 % vs ATR% 1 %
    o[1] = 98.5
    s = _call(dt.setup_h3, o, h, low, c, v, m, ctx)
    assert (s.entry_idx, s.direction, s.stop) == (1, 1, 98.3)
    o, h, low, c, v, m, ctx = _day60()
    assert _call(dt.setup_h3, o, h, low, c, v, m, ctx) is None


def _day5(prev_close=100.0, atr_pct=0.01, up=True):
    n = 78
    ctx = {"prev_close": prev_close, "atr": prev_close * atr_pct, "atr_pct": atr_pct,
           "up": up, "down": not up}
    o = np.full(n, 100.0)
    return (o.copy(), o + 0.1, o - 0.1, o.copy(), np.full(n, 1e5),
            np.arange(570, 570 + 5 * n, 5), ctx)


@pytest.mark.unit
def test_m1_opening_range_breakout_has_midpoint_stop_and_range_width_target():
    o, h, low, c, v, m, ctx = _day5()
    h[:6], low[:6] = 101.0, 99.0                                # range 99..101
    k = 20                                                      # 11:10 bar
    c[k] = 101.5
    o[k + 1] = 101.5
    s = _call(dt.setup_m1, o, h, low, c, v, m, ctx)
    assert (s.entry_idx, s.direction, s.stop) == (k + 1, 1, 100.0)
    assert s.target(101.5) == pytest.approx(103.5)              # fill + range width 2
    o, h, low, c, v, m, ctx = _day5()
    h[:6], low[:6] = 101.0, 99.0
    assert _call(dt.setup_m1, o, h, low, c, v, m, ctx) is None


@pytest.mark.unit
def test_m2_gap_rules_enter_at_the_second_bar_after_a_gap():
    o, h, low, c, v, m, ctx = _day5()
    o[0] = 99.5                                                 # -0.5 % gap, ATR% 1 % -> 0.5 ATR
    s = _call(dt.setup_m2a, o, h, low, c, v, m, ctx)
    assert (s.entry_idx, s.direction) == (1, 1)
    assert _call(dt.setup_m2b, o, h, low, c, v, m, ctx) is None
    o[0] = 103.0
    s = _call(dt.setup_m2b, o, h, low, c, v, m, ctx)
    assert (s.entry_idx, s.direction) == (1, 1)
    o, h, low, c, v, m, ctx = _day5()
    assert _call(dt.setup_m2a, o, h, low, c, v, m, ctx) is None


@pytest.mark.unit
def test_m3_vwap_reversion_needs_a_two_sigma_drop_in_the_window_and_an_uptrend():
    o, h, low, c, v, m, ctx = _day5()
    k = 20                                                      # 11:10
    c[k] = 98.0
    low[k] = 97.9
    o[k + 1] = 98.0
    s = _call(dt.setup_m3, o, h, low, c, v, m, ctx)
    assert s is not None and s.direction == 1 and s.entry_idx == k + 1
    assert s.target > o[k + 1] > s.stop(o[k + 1])
    ctx_down = dict(ctx, up=False, down=False)
    assert _call(dt.setup_m3, o, h, low, c, v, m, ctx_down) is None
    o, h, low, c, v, m, ctx = _day5()
    assert _call(dt.setup_m3, o, h, low, c, v, m, ctx) is None


@pytest.mark.unit
@pytest.mark.parametrize("name, builder", [
    ("H1", "h1"), ("H2", "h2"), ("H3", "h3"),
])
def test_intraday_setups_ignore_bars_from_the_entry_bar_on(name, builder):
    rng = np.random.default_rng(5)
    fired = 0
    for _ in range(300):
        o, h, low, c, v, m, ctx = _day60()
        steps = rng.normal(0, 0.006, 7)
        c = 100 * np.cumprod(1 + steps)
        o = np.r_[100.0, c[:-1]]
        h, low = np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999
        fn = getattr(dt, f"setup_{builder}")
        s = _call(fn, o, h, low, c, v, m, ctx)
        if s is None:
            continue
        fired += 1
        o2, h2, l2, c2 = o.copy(), h.copy(), low.copy(), c.copy()
        e = s.entry_idx
        o2[e:] *= 1.05
        h2[e:] *= 1.09
        l2[e:] *= 0.9
        c2[e:] *= 0.93
        # H1/H2/H3 decide from bars before the entry bar; the entry bar's own open is the fill
        s2 = _call(fn, o2, h2, l2, c2, v, m, ctx)
        assert s2 is not None and (s2.entry_idx, s2.direction) == (s.entry_idx, s.direction)
    assert fired > 20


@pytest.mark.unit
def test_session_filter_accepts_only_complete_sessions_and_run_intraday_trades_them():
    full = np.arange(570, 990, 60)
    assert dt.full_session(full, "60m")
    assert not dt.full_session(full[:3], "60m")                 # half day
    assert not dt.full_session(np.delete(full, 2), "60m")       # a missing bar
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-03-05 09:30") + pd.Timedelta(minutes=60 * i) for i in range(7)]
        + [pd.Timestamp("2024-03-06 09:30") + pd.Timedelta(minutes=60 * i) for i in range(7)],
        tz="America/New_York")
    close = np.r_[np.full(5, 100.0), 101.0, 101.0, np.full(5, 100.0), 99.0, 99.0]
    df = pd.DataFrame({"Open": np.r_[100.0, close[:-1]], "High": close + 0.2, "Low": close - 0.2,
                       "Close": close, "Volume": 1e6}, index=idx)
    daily = _frame(_uptrend())
    daily.index = pd.bdate_range(end="2024-03-04", periods=N)
    feats = {"AAA": daily_features(daily)}
    spec = StrategySpec("H1", "H1", "day", "60m", "x", "x", {})
    rows = dt.run_intraday(spec, {"AAA": df}, feats)
    assert len(rows) == 2 and rows[0]["entry_ts"] == idx[6]     # last bar's open
    assert {r["direction"] for r in rows} <= {1, -1}


@pytest.mark.unit
def test_every_daily_family_is_registered_for_the_study_and_for_live():
    assert set(ALL_DAILY_SIGNALS) == {"D1", "D2", "D3", "D4", "D5", "S1", "S2", "S3", "S4"}
