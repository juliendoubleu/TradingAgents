"""The execution simulator on hand-written bars: every rule of the honest fill model."""

import pytest

from tradingagents.scanner.shortterm.simulate import next_bar_entry, simulate_trade


def _bars(rows):
    """rows of (open, high, low, close) -> four lists."""
    o, h, low, c = zip(*rows, strict=True)
    return list(o), list(h), list(low), list(c)


@pytest.mark.unit
def test_a_close_signal_fills_at_the_next_bars_open_or_not_at_all():
    assert next_bar_entry(3, 10) == 4
    assert next_bar_entry(9, 10) is None           # no next bar in the session -> no trade


@pytest.mark.unit
def test_entry_is_the_open_of_the_entry_bar_not_its_close():
    o, h, low, c = _bars([(100, 101, 99, 100.5), (101, 103, 100.5, 102), (102, 102.5, 101, 101.5)])
    res = simulate_trade(o, h, low, c, entry_idx=1, direction=1, last_idx=1, cost_bps=0)
    assert res.entry_price == 101                  # the open of bar 1, not its close (102)
    assert res.exit_price == 102 and res.exit_reason == "time"


@pytest.mark.unit
def test_stop_wins_when_stop_and_target_are_inside_one_bar():
    o, h, low, c = _bars([(100, 101.5, 98.5, 100)])      # touches both 99 and 101
    res = simulate_trade(o, h, low, c, entry_idx=0, direction=1, last_idx=0, cost_bps=0,
                         stop=99.0, target=101.0)
    assert res.exit_reason == "stop" and res.exit_price == 99.0


@pytest.mark.unit
def test_a_gap_through_the_stop_fills_at_the_open_not_at_the_stop():
    o, h, low, c = _bars([(100, 100.5, 99.5, 100), (97, 98, 96, 97.5)])
    res = simulate_trade(o, h, low, c, entry_idx=0, direction=1, last_idx=1, cost_bps=0, stop=99.0)
    assert res.exit_idx == 1 and res.exit_price == 97 and res.exit_reason == "stop"


@pytest.mark.unit
def test_target_fills_at_the_target_or_at_a_better_gap_open():
    o, h, low, c = _bars([(100, 100.5, 99.5, 100), (101.5, 102, 101, 101.8)])
    res = simulate_trade(o, h, low, c, entry_idx=0, direction=1, last_idx=1, cost_bps=0,
                         target=101.0)
    assert res.exit_price == 101.5 and res.exit_reason == "target"      # opened above target
    o, h, low, c = _bars([(100, 100.5, 99.5, 100), (100.2, 101.4, 100, 101)])
    res = simulate_trade(o, h, low, c, entry_idx=0, direction=1, last_idx=1, cost_bps=0,
                         target=101.0)
    assert res.exit_price == 101.0


@pytest.mark.unit
def test_time_exit_is_the_close_of_the_last_allowed_bar():
    o, h, low, c = _bars([(100, 101, 99.5, 100.5), (100.5, 101, 100, 100.8), (100.8, 109, 90, 105)])
    res = simulate_trade(o, h, low, c, entry_idx=0, direction=1, last_idx=1, cost_bps=0)
    assert res.exit_idx == 1 and res.exit_price == 100.8       # bar 2 is beyond the last bar


@pytest.mark.unit
def test_costs_are_subtracted_in_basis_points():
    o, h, low, c = _bars([(100, 101, 99.5, 101)])
    res = simulate_trade(o, h, low, c, entry_idx=0, direction=1, last_idx=0, cost_bps=10)
    assert res.gross_return == pytest.approx(0.01)
    assert res.net_return == pytest.approx(0.01 - 0.001)


@pytest.mark.unit
def test_short_mirrors_stop_gap_and_target():
    # stop above: bar opens at 103 beyond the 101 stop -> fills at 103
    o, h, low, c = _bars([(100, 100.5, 99.5, 100), (103, 104, 102, 103.5)])
    res = simulate_trade(o, h, low, c, entry_idx=0, direction=-1, last_idx=1, cost_bps=0, stop=101.0)
    assert res.exit_price == 103 and res.exit_reason == "stop"
    assert res.gross_return == pytest.approx(-0.03)
    # target below
    o, h, low, c = _bars([(100, 100.5, 99.5, 100), (99.5, 99.8, 98.5, 99)])
    res = simulate_trade(o, h, low, c, entry_idx=0, direction=-1, last_idx=1, cost_bps=0, target=99.0)
    assert res.exit_price == 99.0 and res.gross_return == pytest.approx(0.01)
    # both in one bar: stop first
    o, h, low, c = _bars([(100, 101.5, 98.5, 100)])
    res = simulate_trade(o, h, low, c, entry_idx=0, direction=-1, last_idx=0, cost_bps=0,
                         stop=101.0, target=99.0)
    assert res.exit_reason == "stop"


@pytest.mark.unit
def test_stop_can_be_priced_from_the_fill_and_exit_on_close_flags():
    o, h, low, c = _bars([(100, 100.4, 99.7, 100.2), (100.2, 101, 100, 100.9), (101, 102, 100.5, 101.5)])
    res = simulate_trade(o, h, low, c, entry_idx=0, direction=1, last_idx=2, cost_bps=0,
                         stop=lambda entry: entry - 1.0, exit_on_close=[False, True, True])
    assert res.exit_idx == 1 and res.exit_reason == "signal" and res.exit_price == 100.9


@pytest.mark.unit
def test_a_stop_already_crossed_at_the_fill_exits_at_that_open():
    o, h, low, c = _bars([(100, 101, 99, 100)])
    res = simulate_trade(o, h, low, c, entry_idx=0, direction=1, last_idx=0, cost_bps=0, stop=101.0)
    assert res.exit_reason == "stop" and res.exit_price == 100.0     # min(stop, open)
    assert res.gross_return == 0.0


@pytest.mark.unit
def test_invalid_direction_is_refused():
    with pytest.raises(ValueError):
        simulate_trade([1], [1], [1], [1], entry_idx=0, direction=0, last_idx=0, cost_bps=0)
