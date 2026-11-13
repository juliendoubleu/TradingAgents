"""The walk-forward validation: no look-ahead, and aggregate arithmetic on a known case."""

import numpy as np
import pandas as pd
import pytest

from tests.scanner_helpers import END, falling, flat, frame, rising
from tradingagents.scanner import scan as sc, validation as va


def _market(n=520):
    return {
        "SPY": flat(n), "A": rising(n, daily=0.002), "B": falling(n, daily=0.002),
    }


@pytest.mark.unit
def test_changing_bars_after_a_rebalance_date_does_not_change_that_dates_signals():
    frames = _market()
    members = ["A", "B"]
    at = frames["SPY"].index[400]

    before = sc.analyse(members, frames, "SPY", at=at)

    tampered = {s: f.copy() for s, f in frames.items()}
    for f in tampered.values():
        f.loc[f.index > at, ["Open", "High", "Low", "Close", "Volume"]] = [1e6, 1e6, 1e-3, 777.0, 5.0]
    after = sc.analyse(members, tampered, "SPY", at=at)

    assert before[0].to_dict() == after[0].to_dict()
    assert [s.to_dict() for s in before[1]] == [s.to_dict() for s in after[1]]
    assert [s.symbol for s in before[1]] == members


@pytest.mark.unit
def test_validation_signals_equal_the_live_scans_on_truncated_bars():
    frames = _market()
    at = frames["SPY"].index[430]
    truncated = {s: f.loc[:at] for s, f in frames.items()}
    live = sc.analyse(["A", "B"], truncated, "SPY")              # the live scan, never told `at`
    walk = sc.analyse(["A", "B"], frames, "SPY", at=at)          # the validation's call
    assert [s.to_dict() for s in live[1]] == [s.to_dict() for s in walk[1]]
    assert live[0].to_dict() == walk[0].to_dict()


@pytest.mark.unit
def test_rebalance_positions_do_not_overlap_and_leave_room_for_the_outcome():
    pos = va.rebalance_positions(n_bars=300, horizon=21, window_bars=100)
    assert pos == sorted(pos) and all(b - a == 21 for a, b in zip(pos, pos[1:], strict=False))
    assert pos[-1] + 21 <= 299 and pos[-1] + 21 > 299 - 21
    assert va.rebalance_positions(10, 21, 100) == []


@pytest.mark.unit
def test_aggregates_on_a_constructed_case():
    frames = _market()
    res = va.validate_frames("us", END, ["A", "B"], "SPY", frames, months=6, horizon=21)
    up = 1.002 ** 21 - 1
    down = 0.998 ** 21 - 1
    assert res.rebalances >= 5 and res.horizon == 21
    long, short, neutral = res.stats["Long"], res.stats["Short"], res.stats["Neutral"]
    assert long.calls == short.calls == res.rebalances
    assert neutral.calls == 0 and neutral.avg_return is None and neutral.hit_rate is None
    assert long.avg_return == pytest.approx(up) and long.hit_rate == 1.0
    assert long.avg_excess == pytest.approx(up)                      # the benchmark is flat
    assert short.avg_return == pytest.approx(down) and short.hit_rate == 1.0
    assert short.avg_pnl == pytest.approx(-down) and short.avg_pnl_excess == pytest.approx(-down)
    assert res.spread == pytest.approx(up - down)
    assert res.benchmark_avg_return == pytest.approx(0.0, abs=1e-12)
    assert res.universe_avg_return == pytest.approx((up + down) / 2)
    assert res.instruments == 2
    assert res.first_rebalance <= res.last_rebalance <= END


@pytest.mark.unit
def test_a_losing_signal_is_counted_as_a_miss():
    # A looks like a strong riser until the final stretch, then falls: Long calls made
    # near the turn lose over the following window.
    n = 520
    path = np.r_[100 * 1.003 ** np.arange(n - 60), 100 * 1.003 ** (n - 61) * 0.99 ** np.arange(1, 61)]
    frames = {"SPY": flat(n), "A": frame(path)}
    res = va.validate_frames("us", END, ["A"], "SPY", frames, months=6, horizon=21)
    long = res.stats["Long"]
    assert long.calls >= 4 and 0 < long.hit_rate < 1
    assert long.avg_return < 1.003 ** 21 - 1


@pytest.mark.unit
def test_instruments_without_a_later_bar_have_no_outcome_and_are_not_counted():
    frames = _market()
    frames["GONE"] = rising(520, daily=0.002).iloc[:-80]            # data ends ~80 bars early
    frames["SPY"] = frames["SPY"]
    res = va.validate_frames("us", END, ["A", "B", "GONE"], "SPY", frames, months=6, horizon=21)
    long_calls = res.stats["Long"].calls
    # GONE contributes only for the windows that still had bars after the rebalance
    assert res.rebalances < long_calls < 2 * res.rebalances


@pytest.mark.unit
def test_validate_uses_the_loader_with_enough_history_and_states_caveats():
    seen = {}

    def loader(symbols, as_of, lookback):
        seen.update(symbols=symbols, as_of=as_of, lookback=lookback)
        return {s: f for s, f in _market().items() if s in symbols}

    res = va.validate("us", END, tickers=["A", "B"], months=6, loader=loader)
    assert seen["symbols"] == ["A", "B", "SPY", "^VIX"] and seen["lookback"] >= (6 + 14) * 30
    text = " ".join(res.caveats).lower()
    for word in ("survivorship", "costs", "future", "sample"):
        assert word in text
    assert res.to_dict()["caveats_de"]


@pytest.mark.unit
def test_too_little_history_yields_an_empty_result_not_a_crash():
    frames = {"SPY": flat(200), "A": rising(200)}
    res = va.validate_frames("us", END, ["A"], "SPY", frames, months=6, horizon=21)
    assert res.rebalances == 0 and res.spread is None
    assert all(s.calls == 0 for s in res.stats.values())


@pytest.mark.unit
def test_direction_stats_arithmetic():
    rows = [("Short", -0.10, 0.02), ("Short", 0.05, 0.01), ("Short", -0.02, 0.00)]
    st = va._stats("Short", rows)
    assert st.calls == 3
    assert st.avg_return == pytest.approx((-0.10 + 0.05 - 0.02) / 3)
    assert st.hit_rate == pytest.approx(2 / 3)
    assert st.avg_pnl == pytest.approx(-st.avg_return)
    assert st.avg_excess == pytest.approx(np.mean([-0.12, 0.04, -0.02]))
    assert st.avg_pnl_excess == pytest.approx(-st.avg_excess)
    assert pd.notna(st.avg_pnl)
