"""Candidate strategies on constructed series: fire when they should, stay quiet when
they should, and never look ahead."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.scanner.shortterm import StrategySpec, daily_features, daytrade as dt, swing
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
    daily.index = pd.bdate_range(end="2024-03-05", periods=N)
    feats = {"AAA": daily_features(daily)}
    spec = StrategySpec("H1", "H1", "day", "60m", "x", "x", {})
    rows = dt.run_intraday(spec, {"AAA": df}, feats)
    # the first session has no previous intraday session (no prev_close) and is skipped
    assert len(rows) == 1 and rows[0]["entry_ts"] == idx[13]    # last bar's open
    assert rows[0]["direction"] == -1                           # 99 vs previous intraday close 101


@pytest.mark.unit
def test_every_daily_family_is_registered_for_the_study_and_for_live():
    assert set(ALL_DAILY_SIGNALS) == {"D1", "D2", "D3", "D4", "D5", "S1", "S2", "S3", "S4"}


# ----------------------------------------------------------------------- H-1: previous close


def _two_sessions_60m(first_close=100.0, second_first_close=99.8, *, truncate_first=False,
                      days=("2024-03-04", "2024-03-05")):
    idx_all, close = [], []
    for k, day in enumerate(days):
        n = 7
        times = [pd.Timestamp(f"{day} 09:30") + pd.Timedelta(minutes=60 * i) for i in range(n)]
        c = np.full(n, first_close if k == 0 else 100.0)
        if k == 1:
            c[0] = second_first_close
        if truncate_first and k == 0:
            times, c = times[:5], c[:5]                  # the session stops at 14:30: no close
        idx_all += times
        close += list(c)
    idx = pd.DatetimeIndex(idx_all, tz="America/New_York")
    close = np.asarray(close)
    return pd.DataFrame({"Open": np.r_[close[0], close[:-1]], "High": close + 0.2,
                         "Low": close - 0.2, "Close": close, "Volume": 1e6}, index=idx)


def _dividend_adjusted_daily(factor=0.90, end="2024-03-04"):
    """Daily bars the way auto-adjust serves them: every price scaled down by a dividend factor."""
    daily = _frame(_uptrend())
    scale = 100.0 * factor / daily["Close"].iloc[-1]
    daily[["Open", "High", "Low", "Close"]] *= scale
    daily.index = pd.bdate_range(end=end, periods=N)
    return daily


@pytest.mark.unit
def test_intraday_rules_take_the_previous_close_from_the_intraday_frame_not_the_daily_one():
    # intraday last close of day 1 is 100 (as traded); the ADJUSTED daily close is ~90.
    # Day 2's first hour closes at 99.8: below 100 = a down move, but "above" 90 if the daily
    # close were used. H1b follows that first hour, so its side shows which one was used.
    df = _two_sessions_60m()
    daily = _dividend_adjusted_daily()
    assert daily["Close"].iloc[-1] == pytest.approx(90.0)
    feats = {"AAA": daily_features(daily)}
    spec = StrategySpec("H1b", "H1b", "day", "60m", "x", "x", {})
    rows = dt.run_intraday(spec, {"AAA": df}, feats)
    assert [r["direction"] for r in rows] == [-1]
    ctx = dt.day_context(feats["AAA"], pd.Timestamp("2024-03-05"), 100.0)
    assert ctx["prev_close"] == 100.0
    # ATR in price units is scaled to the intraday price level too (atr% x intraday close)
    assert ctx["atr"] == pytest.approx(ctx["atr_pct"] * 100.0)


@pytest.mark.unit
def test_a_session_without_a_usable_previous_intraday_session_is_skipped():
    spec = StrategySpec("H1b", "H1b", "day", "60m", "x", "x", {})
    feats = {"AAA": daily_features(_dividend_adjusted_daily())}
    # the previous session was cut short (last bar 14:30): its last print is not the close
    assert dt.run_intraday(spec, {"AAA": _two_sessions_60m(truncate_first=True)}, feats) == []
    # a day missing from the intraday frame: the previous intraday session is not the
    # previous trading day of the daily frame
    gap = _two_sessions_60m(days=("2024-03-01", "2024-03-05"))
    assert dt.run_intraday(spec, {"AAA": gap}, feats) == []


@pytest.mark.unit
def test_an_early_close_session_counts_as_a_previous_session():
    half = np.arange(570, 570 + 4 * 60, 60)                 # 09:30 ... 12:30, closes 13:00
    assert dt.session_closed(half, "60m") and dt.session_closed(np.arange(570, 990, 60), "60m")
    assert not dt.session_closed(np.arange(570, 570 + 5 * 60, 60), "60m")
    assert dt.session_closed(np.arange(570, 570 + 78 * 5, 5), "5m")


# ----------------------------------------------------------------------- M-4: mutation killers


def _flat(n=N, level=100.0, wobble=0.004):
    """A flat series hugging its SMA200, so one big last day flips the trend flag."""
    return level * (1 + wobble * np.sin(np.arange(n) * 0.9))


def _open_family_frames(prev_above=True):
    """Two frames that differ ONLY in the entry day's own close/high/low/volume.

    Before the entry day the stock closes just above (or below) its SMA200; a moderate gap
    (against that trend) opens the entry day. In frame B the entry day's close then crosses
    the SMA200 and its range is huge (the ATR changes): everything a rule that is "known at
    the open" must not see.
    """
    close = _flat()
    sma = close[-201:-1].mean()
    close[-2] = close[-1] = sma * (1.004 if prev_above else 0.996)
    gap = np.zeros(N)
    gap[-1] = -0.008 if prev_above else 0.008
    a = _frame(close, gap=gap)
    b = a.copy()
    i = len(b) - 1
    b.iloc[i, b.columns.get_loc("Close")] = sma * (0.97 if prev_above else 1.03)
    b.iloc[i, b.columns.get_loc("High")] = b["High"].iloc[i] * 1.06
    b.iloc[i, b.columns.get_loc("Low")] = b["Low"].iloc[i] * 0.94
    b.iloc[i, b.columns.get_loc("Volume")] *= 9
    return a, b


@pytest.mark.unit
@pytest.mark.parametrize("prev_above", [True, False])
def test_d1_uses_the_trend_known_before_the_open_not_the_entry_days_close(prev_above):
    a, b = _open_family_frames(prev_above)
    expected = 1 if prev_above else -1                       # fade a gap against the trend
    assert _last("D1", a) == expected and _last("D1", b) == expected
    fa, fb = daily_features(a), daily_features(b)
    # the entry day's own close did flip the same-day flag; the *_prev flags did not move
    assert bool(fb["up"].iloc[-1]) != bool(fa["up"].iloc[-1]) or bool(fb["down"].iloc[-1]) != bool(fa["down"].iloc[-1])
    assert fa["up_prev"].iloc[-1] == fb["up_prev"].iloc[-1]


@pytest.mark.unit
def test_open_based_stop_uses_the_atr_known_before_the_entry_bar():
    a, b = _open_family_frames(prev_above=True)
    spec = StrategySpec("D1", "D1", "day", "1d", "x", "x", {"lo": 0.3, "hi": 1.5, "stop_atr": 1.0, "hold": 1})
    for frame in (a, b):
        feats = {"AAA": daily_features(frame)}
        rows = dt.run_daily(spec, feats, lambda f, p: daily_signal("D1", f, p),
                            first_day=frame.index[0])
        row = [r for r in rows if r["entry_ts"] == frame.index[-1]][0]
        f = feats["AAA"]
        entry = frame["Open"].iloc[-1]
        atr_prev = f["atr_prev"].iloc[-1]
        assert row["direction"] == 1
        if frame is b:
            # the huge entry-day range hits the stop; it must sit exactly 1 ATR (as known
            # BEFORE the entry day) below the fill, not 1 ATR of a range that includes the day
            assert row["reason"] == "stop"
            assert row["raw"] == pytest.approx(-atr_prev / entry)
            assert f["atr"].iloc[-1] > 1.3 * atr_prev        # the same-day ATR is far larger


@pytest.mark.unit
def test_the_volume_average_excludes_the_signal_day():
    # prior 20 days average 1.0e6; today 1.52e6 -> ratio 1.52 >= 1.5 only if today is NOT
    # in the average (including it gives 20 * 1.52 / 20.52 = 1.481 < 1.5)
    close = _uptrend(wobble=0.0)
    close[-1] = close[-2] * 1.03
    vol = np.full(N, 1e6)
    vol[-1] = 1.52e6
    df = _frame(close, vol=vol)
    feats = daily_features(df)
    assert feats["vol_ratio"].iloc[-1] == pytest.approx(1.52)
    assert _last("S3", df) == 1
    # and a past day's average never contains that day either
    assert feats["vol_ratio"].iloc[-30] == pytest.approx(1.0)


def _tiny_run(signal_idx, hold=5, n=40):
    close = 100 * np.exp(np.cumsum(np.full(n, 0.001)))
    df = _frame(close)
    df.index = pd.bdate_range("2024-01-01", periods=n)
    sig = pd.Series(0, index=df.index)
    sig.iloc[list(signal_idx)] = 1
    spec = StrategySpec("T", "S1", "swing", "1d", "x", "x", {"hold": hold})
    feats = {"AAA": daily_features(df)}
    # atr is NaN in the first 14 bars; stay beyond that
    return dt.run_daily(spec, feats, lambda f, p: sig, first_day=df.index[0]), df


@pytest.mark.unit
def test_a_symbol_in_a_trade_ignores_new_signals_until_it_is_out():
    # signals on bars 20, 21, 22 and 24: the first enters at 21 and holds 5 bars (to 25);
    # the others would enter at 22, 23, 25 inside that trade. No pyramiding.
    rows, df = _tiny_run([20, 21, 22, 24])
    assert len(rows) == 1
    assert rows[0]["entry_ts"] == df.index[21] and rows[0]["exit_ts"] == df.index[25]
    # a signal on the exit bar (25) fills at 26, after the position is out -> a second trade
    rows, df = _tiny_run([20, 25])
    assert [r["entry_ts"] for r in rows] == [df.index[21], df.index[26]]


@pytest.mark.unit
def test_s1_leaves_a_long_above_and_a_short_below_the_five_day_average():
    f = pd.DataFrame({"close": [10.0, 12.0, 8.0, 10.0], "sma5": [10.0, 10.0, 10.0, 10.0]})
    assert swing.exit_flags_s1(f, 1).tolist() == [False, True, False, False]
    assert swing.exit_flags_s1(f, -1).tolist() == [False, False, True, False]


@pytest.mark.unit
def test_run_daily_closes_s1_shorts_at_the_first_close_below_the_average():
    n = 60
    close = 100 * np.exp(np.cumsum(np.full(n, -0.001)))
    close[40:44] = close[39] * np.array([1.02, 1.03, 1.02, 0.97])   # bounce, then a drop
    df = _frame(close)
    df.index = pd.bdate_range("2024-01-01", periods=n)
    feats = {"AAA": daily_features(df)}
    sig = pd.Series(0, index=df.index)
    sig.iloc[40] = -1
    spec = StrategySpec("T", "S1", "swing", "1d", "x", "x", {"hold": 15})
    rows = dt.run_daily(spec, feats, lambda f, p: sig, first_day=df.index[0],
                        exit_flags=swing.EXIT_FLAGS["S1"])
    f = feats["AAA"]
    first_below = next(i for i in range(41, n) if f["close"].iloc[i] < f["sma5"].iloc[i])
    assert rows[0]["reason"] == "signal" and rows[0]["exit_ts"] == df.index[first_below]
