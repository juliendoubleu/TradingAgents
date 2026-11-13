"""Medium/long-horizon rules: they fire where they should, rank only names with data,
and never look past the decision date."""

import numpy as np
import pandas as pd
import pytest

from tests.scanner_helpers import frame, rising
from tradingagents.scanner import rules
from tradingagents.scanner.signals import SignalParams

N = 520
NAMES = [f"S{i}" for i in range(10)]


def _frames(n=N, seed=1, bench_trend=0.0004):
    rng = np.random.default_rng(seed)
    out = {}
    for i, name in enumerate(NAMES):
        drift = 0.0012 * (i - 4.5) / 4.5          # S0 falls hardest, S9 rises most
        steps = rng.normal(drift, 0.004, n)
        out[name] = frame(100 * np.exp(np.cumsum(steps)))
    out["BM"] = rising(n, bench_trend)
    return out


def _panel(frames=None, **kw):
    frames = frames or _frames()
    return rules.Panel(frames, NAMES, "BM", SignalParams(), **kw)


@pytest.mark.unit
def test_horizon_plan_follows_the_calendar():
    assert rules.horizon_plan("medium", 21) == (21, 5)
    assert rules.horizon_plan("long", 21) == (126, 21)
    assert rules.horizon_plan("medium", 30) == (30, 7)        # crypto
    assert rules.horizon_plan("long", 30) == (180, 30)
    with pytest.raises(ValueError):
        rules.horizon_plan("day", 21)


@pytest.mark.unit
def test_momentum_takes_the_top_and_bottom_fifth():
    panel = _panel()
    sides, scores = rules.rule_sides("T1", panel)
    last = sides.iloc[-1]
    assert (last == 1).sum() == 2 and (last == -1).sum() == 2        # 20 % of 10 names
    longs = set(last[last == 1].index)
    best = set(scores.iloc[-1].nlargest(2).index)
    assert longs == best
    assert set(last[last == -1].index) == set(scores.iloc[-1].nsmallest(2).index)


@pytest.mark.unit
def test_trend_state_fires_both_ways_and_stays_flat_in_between():
    # MIXED: price above its 200-day average (a spike on the last bars) while the 50-day
    # average is still below it -> neither the long nor the short condition holds
    mixed = np.concatenate([np.full(N - 62, 100.0), np.full(60, 80.0), [120.0, 120.0]])
    frames = {"UP": rising(N, 0.002), "DN": rising(N, -0.002), "CHOP": frame(mixed)}
    for i in range(3):
        frames[f"X{i}"] = rising(N, 0.001 * (i + 1))
    frames["BM"] = rising(N, 0.0004)
    panel = rules.Panel(frames, [k for k in frames if k != "BM"], "BM", SignalParams())
    sides, _ = rules.rule_sides("T2", panel)
    last = sides.iloc[-1]
    assert last["UP"] == 1 and last["DN"] == -1 and last["CHOP"] == 0


@pytest.mark.unit
def test_t3_short_needs_the_benchmark_below_its_200_day_average():
    rising_market = _panel()
    short_up = rules.rule_sides("T3", rising_market)[0].iloc[-1]
    assert (short_up == -1).sum() == 0                  # market filter blocks every short
    longs = rules.rule_sides("T3", rising_market)[0].iloc[-1]
    assert (longs == 1).sum() >= 1                      # ... but not the longs
    falling_market = _panel(_frames(bench_trend=-0.0008))
    sides, _ = rules.rule_sides("T3", falling_market)
    assert (sides.iloc[-1] == -1).sum() >= 1
    # the variant without the market filter shorts in the rising market as well
    assert (rules.rule_sides("V3", rising_market)[0].iloc[-1] == -1).sum() >= 1


@pytest.mark.unit
def test_t4_long_near_the_high_short_far_below_and_under_the_average():
    frames = {f"F{i}": rising(N, 0.0005) for i in range(4)}
    base = np.concatenate([np.linspace(100, 200, 300), np.linspace(200, 120, N - 300)])
    frames["CRASH"] = frame(base)                         # 40 % under its high, under SMA200
    frames["TOP"] = frame(np.linspace(100, 300, N))       # always at its high
    frames["BM"] = rising(N, 0.0004)
    panel = rules.Panel(frames, [k for k in frames if k != "BM"], "BM", SignalParams())
    last = rules.rule_sides("T4", panel)[0].iloc[-1]
    assert last["TOP"] == 1 and last["CRASH"] == -1 and last["F0"] == 1


@pytest.mark.unit
def test_t5_buys_the_weakest_month_above_the_trend_and_shorts_the_strongest_below():
    n = N
    frames = {}
    for i in range(6):                                    # uptrending names, one with a dip
        steps = np.full(n, 0.002)
        frames[f"U{i}"] = frame(100 * np.exp(np.cumsum(steps)))
    dip = np.full(n, 0.002)
    dip[-15:] = -0.004
    frames["DIP"] = frame(100 * np.exp(np.cumsum(dip)))
    for i in range(6):                                    # downtrending names, one bouncing
        frames[f"D{i}"] = frame(100 * np.exp(np.cumsum(np.full(n, -0.002))))
    bounce = np.full(n, -0.002)
    bounce[-15:] = 0.003
    frames["BOUNCE"] = frame(100 * np.exp(np.cumsum(bounce)))
    frames["BM"] = rising(n, 0.0004)
    panel = rules.Panel(frames, [k for k in frames if k != "BM"], "BM", SignalParams())
    last = rules.rule_sides("T5", panel)[0].iloc[-1]
    assert last["DIP"] == 1 and last["BOUNCE"] == -1
    assert (last == 1).sum() == 2 and (last == -1).sum() == 2      # 20 % of 7 names, rounded up


@pytest.mark.unit
def test_cross_sectional_ranks_ignore_names_without_a_value():
    values = pd.DataFrame({f"S{i}": [float(i)] * 2 for i in range(10)})
    values.loc[1, ["S9", "S8", "S0"]] = np.nan            # row 1: three names have no data
    top = rules._quantile_mask(values, 0.2, True)
    assert top.loc[0].sum() == 2 and set(top.columns[top.loc[0]]) == {"S9", "S8"}
    assert top.loc[1].sum() == 2 and set(top.columns[top.loc[1]]) == {"S7", "S6"}   # 7 names -> 2
    few = values.copy()
    few.loc[1, :] = np.nan
    few.loc[1, ["S1", "S2", "S3"]] = [1.0, 2.0, 3.0]
    assert rules._quantile_mask(few, 0.2, True).loc[1].sum() == 0    # < 5 names: no ranking


@pytest.mark.unit
def test_a_missing_bar_gives_no_signal_that_day():
    frames = _frames()
    gap_day = frames["S9"].index[-3]
    frames["S9"] = frames["S9"].drop(gap_day)             # S9 has a hole, the others do not
    panel = _panel(frames)
    sides, scores = rules.rule_sides("T1", panel)
    assert np.isnan(scores.loc[gap_day, "S9"]) and sides.loc[gap_day, "S9"] == 0
    assert scores.loc[panel.index[-1], "S9"] == scores.loc[panel.index[-1], "S9"]


@pytest.mark.unit
def test_no_look_ahead_changing_later_bars_leaves_every_earlier_side_unchanged():
    frames = _frames()
    base = _panel(frames)
    cut = len(base.index) - 60
    when = base.index[cut]
    rng = np.random.default_rng(7)
    altered = {}
    for name, f in frames.items():
        g = f.copy()
        later = g.index > when
        g.loc[later, ["Open", "High", "Low", "Close"]] = (
            g.loc[later, ["Open", "High", "Low", "Close"]].to_numpy()
            * rng.uniform(0.5, 1.5, (later.sum(), 1)))
        altered[name] = g
    other = _panel(altered)
    for rid in ("T1", "T2", "T3", "T4", "T5", "V1", "V3", "V5"):
        a, sa = rules.rule_sides(rid, base)
        b, sb = rules.rule_sides(rid, other)
        assert a.loc[:when].equals(b.loc[:when]), rid
        assert np.allclose(sa.loc[:when].to_numpy(float), sb.loc[:when].to_numpy(float),
                           equal_nan=True), rid
    assert not base.atr().loc[:when].equals(other.atr().loc[when:])      # the change is real
    assert np.allclose(base.atr().loc[:when].to_numpy(float),
                       other.atr().loc[:when].to_numpy(float), equal_nan=True)


@pytest.mark.unit
def test_composite_panel_is_the_scanners_own_analyse_at_that_date():
    from tradingagents.scanner.scan import analyse

    frames = {"SPY": rising(400, 0.0005), "^VIX": frame(np.full(400, 15.0))}
    for i in range(6):
        frames[f"M{i}"] = rising(400, 0.001 * (i - 2), wiggle=0.01)
    members = [f"M{i}" for i in range(6)]
    dates = frames["M0"].index[[300, 350, 399]]
    sides, scores = rules.composite_panel(frames, members, "SPY", dates, SignalParams())
    for d in dates:
        _, sigs, _ = analyse(members, frames, "SPY", at=d)
        for s in sigs:
            assert scores.loc[d, s.symbol] == pytest.approx(s.score)
            assert sides.loc[d, s.symbol] == {"Long": 1, "Short": -1, "Neutral": 0}[s.direction]


@pytest.mark.unit
def test_current_members_are_ordered_best_first_within_each_side():
    panel = _panel()
    got = rules.current_members("T1", panel)
    longs = [g for g in got if g[1] == 1]
    shorts = [g for g in got if g[1] == -1]
    assert [g[3] for g in longs] == list(range(1, len(longs) + 1))
    assert [g[2] for g in longs] == sorted((g[2] for g in longs), reverse=True)
    assert [g[2] for g in shorts] == sorted(g[2] for g in shorts)


@pytest.mark.unit
def test_variants_are_registered_with_a_motivation_and_a_cap_of_six():
    variants = [r for r in rules.RULES.values() if r.variant]
    assert 1 <= len(variants) <= 6
    assert all(r.note_en and r.note_de and r.base in rules._DEFAULTS for r in variants)
    assert rules.rule_params("V1")["q"] == 0.10 and rules.rule_params("T1")["q"] == 0.20
