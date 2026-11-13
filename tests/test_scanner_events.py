"""Event study, snapshot and renderers on constructed series with hand-computed returns."""

import json

import numpy as np
import pandas as pd
import pytest

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.scanner.events import (
    WINDOWS,
    event_study,
    performance_snapshot,
    render_event_study,
    render_implied_moves,
    render_snapshot,
)

AS_OF = "2026-06-30"


def _frame(closes, start="2026-01-05", high_pad=1.0):
    """Business-day frame; close of bar k is closes[k]."""
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(np.asarray(closes, dtype=float), index=idx)
    return pd.DataFrame({"Open": c, "High": c + high_pad, "Low": c - high_pad, "Close": c,
                         "Volume": 1000.0})


def _loader(**frames):
    def load(symbols, as_of_date, lookback_days):
        return {s: frames[s] for s in symbols if s in frames}
    return load


# stock: close = 100 + bar index; bench: close = 1000 + 2*bar index (same dates)
N = 120
STOCK = _frame(100 + np.arange(N))
BENCH = _frame(1000 + 2 * np.arange(N))
LOAD = _loader(AAA=STOCK, BBB=BENCH)


def _study(events, **kw):
    return event_study("AAA", events, benchmark="BBB", as_of_date=AS_OF, loader=LOAD, **kw)


@pytest.mark.unit
def test_windows_on_a_trading_day_are_exact():
    dates = STOCK.index
    ev = _study([(dates[80].strftime("%Y-%m-%d"), "e")]).events[0]
    assert ev.event_bar_date == dates[80].strftime("%Y-%m-%d") and not ev.upcoming
    w = ev.windows
    # bar k closes at 100+k; event bar 80 -> 180
    assert w["runup_60"].ret == pytest.approx(179 / 119 - 1)   # bars 19 -> 79
    assert w["runup_20"].ret == pytest.approx(179 / 159 - 1)   # bars 59 -> 79
    assert w["event_day"].ret == pytest.approx(180 / 179 - 1)
    assert w["after_5"].ret == pytest.approx(185 / 179 - 1)
    assert w["after_20"].ret == pytest.approx(200 / 179 - 1)
    # bench: 1000+2k over the same dates: bars 79 -> 85
    assert w["after_5"].bench_ret == pytest.approx(1170 / 1158 - 1)
    assert w["after_5"].excess_pts == pytest.approx((185 / 179 - 1 - (1170 / 1158 - 1)) * 100)
    assert w["after_60"].ret is None  # bar 140 does not exist (last bar is 119)


@pytest.mark.unit
def test_event_on_a_weekend_maps_to_the_next_session():
    sat = pd.Timestamp("2026-04-11")  # Saturday
    assert sat.dayofweek == 5
    ev = _study([("2026-04-11", "weekend")]).events[0]
    assert ev.event_bar_date == "2026-04-13"
    k = STOCK.index.get_loc(pd.Timestamp("2026-04-13"))
    assert ev.windows["event_day"].ret == pytest.approx((100 + k) / (100 + k - 1) - 1)


@pytest.mark.unit
def test_windows_without_bars_are_none_at_both_ends():
    early = _study([(STOCK.index[30].strftime("%Y-%m-%d"), "early")]).events[0]
    assert early.windows["runup_60"].ret is None  # needs bar -31
    assert early.windows["runup_20"].ret == pytest.approx(129 / 109 - 1)
    late = _study([(STOCK.index[115].strftime("%Y-%m-%d"), "late")]).events[0]
    assert late.windows["after_5"].ret is None  # needs bar 120, last is 119
    assert late.windows["after_5"].excess_pts is None
    assert late.windows["event_day"].ret is not None


@pytest.mark.unit
def test_upcoming_event_gets_runup_so_far_only():
    ev = _study([("2026-12-01", "future")]).events[0]
    assert ev.upcoming and ev.event_bar_date is None
    assert ev.windows["runup_60"].ret == pytest.approx(219 / 159 - 1)  # bars 59 -> 119
    assert ev.windows["runup_20"].ret == pytest.approx(219 / 199 - 1)
    assert ev.windows["runup_60"].end_date == STOCK.index[-1].strftime("%Y-%m-%d")
    for name in ("event_day", "after_5", "after_10", "after_20", "after_60"):
        assert ev.windows[name].ret is None


@pytest.mark.unit
def test_event_before_the_history_is_uncovered_not_guessed():
    study = _study([("2020-01-01", "ancient")])
    ev = study.events[0]
    assert not ev.covered and not ev.upcoming
    assert all(w.ret is None for w in ev.windows.values())
    assert study.summary["after_5"]["n"] == 0


@pytest.mark.unit
def test_missing_benchmark_gives_none_excess_not_an_error():
    study = event_study("AAA", [("2026-04-01", "e")], benchmark="ZZZ", as_of_date=AS_OF,
                        loader=_loader(AAA=STOCK))
    w = study.events[0].windows["after_5"]
    assert w.ret is not None and w.bench_ret is None and w.excess_pts is None
    assert not study.benchmark_available
    assert "No benchmark" in render_event_study(study, "en")


@pytest.mark.unit
def test_summary_arithmetic():
    # stock follows a custom path so the three events differ in sign
    closes = np.full(N, 100.0)
    closes[70:] = 110.0   # up at bar 70
    closes[90:] = 99.0    # down at bar 90
    stock = _frame(closes)
    load = _loader(AAA=stock, BBB=BENCH)
    d = [stock.index[i].strftime("%Y-%m-%d") for i in (70, 90, 100)]
    study = event_study("AAA", [(d[0], "up"), (d[1], "down"), (d[2], "flat")], benchmark="BBB",
                        as_of_date=AS_OF, loader=load)
    s = study.summary["event_day"]
    assert s["n"] == 3 and s["n_pos"] == 1 and s["n_neg"] == 1
    assert s["mean"] == pytest.approx((0.10 + (99 / 110 - 1) + 0.0) / 3)
    assert s["median"] == pytest.approx(0.0)
    assert study.summary["after_5"]["n"] == 3
    assert study.summary["after_20"]["n"] == 2  # the bar-100 event lacks bar 120
    assert study.summary["after_60"]["n"] == 0
    assert study.summary["after_60"]["mean"] is None


@pytest.mark.unit
def test_upcoming_events_stay_out_of_the_summary():
    study = _study([("2026-04-01", "past"), ("2026-12-01", "future")])
    assert study.summary["runup_20"]["n"] == 1


@pytest.mark.unit
def test_errors_for_bad_input():
    with pytest.raises(ValueError):
        _study([("13.11.2026", "bad")])
    with pytest.raises(ValueError):
        event_study("AAA", [], loader=LOAD, as_of_date=AS_OF)
    with pytest.raises(NoMarketDataError):
        event_study("NOPE", [("2026-04-01", "e")], loader=LOAD, as_of_date=AS_OF)


@pytest.mark.unit
def test_loader_gets_enough_history_in_one_call():
    seen = []

    def load(symbols, as_of_date, lookback_days):
        seen.append((list(symbols), as_of_date, lookback_days))
        return {"AAA": STOCK, "BBB": BENCH}

    event_study("AAA", [("2026-04-01", "e")], benchmark="BBB", as_of_date=AS_OF, loader=load)
    assert len(seen) == 1
    assert seen[0][0] == ["AAA", "BBB"]
    assert seen[0][2] >= (pd.Timestamp(AS_OF) - pd.Timestamp("2026-04-01")).days + 130


@pytest.mark.unit
def test_to_dict_is_json_serialisable():
    study = _study([("2026-04-01", "e"), ("2026-12-01", "f")])
    data = json.loads(json.dumps(study.to_dict()))
    assert data["events"][1]["upcoming"] is True
    assert data["events"][1]["windows"]["after_5"]["ret"] is None
    assert set(data["summary"]) == set(WINDOWS)


@pytest.mark.unit
def test_german_rendering_has_no_english_headers():
    study = _study([("2026-04-01", "Launch"), ("2026-12-01", "Zukunft")])
    de = render_event_study(study, "de")
    en = render_event_study(study, "en")
    for word in ("Event", "Window", "Mean", "Summary", "Caveats", "upcoming",
                 "vs index", "Returns", "pts", "up/down", "as of"):
        assert word not in de
    for word in ("Ereignis", "Fenster", "Mittel", "Zusammenfassung", "bevorstehend"):
        assert word in de
    assert "Event" in en and "upcoming" in en
    assert "," in de.split("Renditen")[1].split("\n")[4]  # decimal comma in the first data row


@pytest.mark.unit
def test_snapshot_values():
    closes = 100 + np.arange(300.0)
    snaps = performance_snapshot(["AAA"], as_of_date="2027-12-31",
                                 loader=_loader(AAA=_frame(closes)))
    s = snaps[0]
    last = 399.0
    assert s.last_close == last
    assert s.last_date == _frame(closes).index[-1].strftime("%Y-%m-%d")
    assert s.ret_5 == pytest.approx(last / 394 - 1)
    assert s.ret_21 == pytest.approx(last / 378 - 1)
    assert s.ret_252 == pytest.approx(last / 147 - 1)
    assert s.dist_high_252 == pytest.approx(0.0)
    assert s.dist_low_252 == pytest.approx(last / 148 - 1)  # lowest of the last 252 closes
    assert s.atr14_pct == pytest.approx(2.0 / last)  # constant true range of 2 -> ATR 2
    assert s.error is None


@pytest.mark.unit
def test_snapshot_short_history_and_missing_symbol():
    snaps = performance_snapshot(["AAA", "NOPE"], as_of_date="2027-12-31",
                                 loader=_loader(AAA=_frame(100 + np.arange(30.0))))
    a, nope = snaps
    assert a.ret_21 is not None and a.ret_63 is None and a.ret_252 is None
    assert a.dist_high_252 is None and a.dist_low_252 is None
    assert nope.symbol == "NOPE" and nope.error and nope.last_close is None
    json.dumps([s.to_dict() for s in snaps])


@pytest.mark.unit
def test_snapshot_rendering_both_languages():
    snaps = performance_snapshot(["AAA", "NOPE"], as_of_date="2027-12-31",
                                 loader=_loader(AAA=_frame(1000 + np.arange(300.0))))
    en, de = render_snapshot(snaps, "en"), render_snapshot(snaps, "de")
    assert "Snapshot" in en and "no data" in en and "1,299.00" in en
    assert "Momentaufnahme" in de and "keine Daten" in de and "1.299,00" in de
    assert "Snapshot" not in de and "no data" not in de and "to high" not in de


@pytest.mark.unit
def test_render_implied_moves_languages_and_stale_note():
    from tradingagents.dataflows.vendors.yahoo.options import ImpliedMove

    m = ImpliedMove("AAA", "2026-11-20", 7, 100.0, 100.0, 4.0, 5.0, 9.0, 0.09, "last", True)
    en, de = render_implied_moves([m], "en"), render_implied_moves([m], "de")
    assert "±9.0 %" in en and "last-trade" in en and "absolute move" in en
    assert "±9,0 %" in de and "letzte Handelspreise" in de and "20.11.2026" in de
    assert "Expiry" not in de
    fresh = ImpliedMove("AAA", "2026-11-20", 7, 100.0, 100.0, 4.0, 5.0, 9.0, 0.09, "mid", False)
    assert "last-trade" not in render_implied_moves([fresh], "en")
    assert "No listed options" in render_implied_moves([], "en")
    assert "Keine gelisteten" in render_implied_moves([], "de")
