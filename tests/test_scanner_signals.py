"""Signal scoring, direction thresholds, the trade plan and the market regime."""

import numpy as np
import pandas as pd
import pytest

from tests.scanner_helpers import falling, flat, frame, noisy, rising
from tradingagents.scanner import regime as rg, signals as sg


def _bench(n=400):
    return flat(n)["Close"]


@pytest.mark.unit
def test_a_steady_riser_is_long_with_a_positive_score():
    sig = sg.compute_signal("UP", rising(wiggle=0.004), _bench(), "neutral")
    assert sig.direction == "Long" and sig.score >= 35
    assert sig.components["trend"] == 1.0
    assert sig.components["momentum"] > 0.9
    assert sig.components["relative_strength"] > 0


@pytest.mark.unit
def test_a_steady_faller_is_short_with_a_negative_score():
    sig = sg.compute_signal("DN", falling(wiggle=0.004), _bench(), "neutral")
    assert sig.direction == "Short" and sig.score <= -35
    assert sig.components["trend"] == -1.0


@pytest.mark.unit
def test_flat_and_noisy_series_are_neutral_without_a_trade_plan():
    for df in (flat(), noisy()):
        sig = sg.compute_signal("X", df, _bench(), "neutral")
        assert sig.direction == "Neutral"
        assert sig.entry is None and sig.stop is None and sig.target is None
        assert np.isfinite(sig.score)


@pytest.mark.unit
def test_weights_sum_to_one_and_score_is_bounded():
    assert sum(sg.WEIGHTS.values()) == pytest.approx(1.0)
    sig = sg.compute_signal("UP", rising(daily=0.01), _bench(), "neutral")
    assert -100 <= sig.score <= 100


@pytest.mark.unit
def test_benchmark_has_zero_relative_strength():
    sig = sg.compute_signal("SPY", rising(), rising()["Close"], "neutral", benchmark="SPY")
    assert sig.components["relative_strength"] == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("score,regime,expected", [
    (36, "neutral", "Long"), (34.9, "neutral", "Neutral"), (-35, "neutral", "Short"),
    (25, "risk-on", "Long"), (-49.9, "risk-on", "Neutral"), (-50, "risk-on", "Short"),
    (49.9, "risk-off", "Neutral"), (50, "risk-off", "Long"), (-25, "risk-off", "Short"),
    (30, "risk-off", "Neutral"), (-30, "risk-on", "Neutral"), (0, "neutral", "Neutral"),
])
def test_thresholds_shift_with_the_regime(score, regime, expected):
    assert sg.direction_for(score, regime) == expected


@pytest.mark.unit
def test_the_same_series_can_flip_with_the_regime():
    df = rising(daily=0.0004, wiggle=0.01)
    base = sg.compute_signal("X", df, _bench(), "neutral")
    off = sg.compute_signal("X", df, _bench(), "risk-off")
    on = sg.compute_signal("X", df, _bench(), "risk-on")
    assert base.score == pytest.approx(on.score) == pytest.approx(off.score)
    assert sg.direction_for(base.score, "risk-on") == on.direction
    assert sg.direction_for(base.score, "risk-off") == off.direction


@pytest.mark.unit
@pytest.mark.parametrize("score,expected", [(65, "High"), (-80, "High"), (50, "Medium"),
                                            (-64.9, "Medium"), (49.9, "Low"), (0, "Low")])
def test_conviction_bands(score, expected):
    assert sg.conviction_for(score) == expected


@pytest.mark.unit
def test_trade_plan_arithmetic_for_a_long_and_a_short():
    long_sig = sg.Signal("L", "Long", 70, "High", {}, close=100.0, last_date="x",
                         regime="neutral", atr=3.0)
    sg._attach_plan(long_sig)
    assert (long_sig.entry, long_sig.stop, long_sig.target) == (100.0, 94.0, 109.0)
    # stop distance 6 % -> risking 1 % of capital allows 1/0.06 = 16.7 %, capped at 10 %
    assert long_sig.size_pct == 10.0
    assert pytest.approx(1.5) == (109 - 100) / (100 - 94)

    short_sig = sg.Signal("S", "Short", -70, "High", {}, close=100.0, last_date="x",
                          regime="neutral", atr=10.0)
    sg._attach_plan(short_sig)
    assert (short_sig.stop, short_sig.target) == (120.0, 70.0)
    assert short_sig.size_pct == pytest.approx(5.0)      # 1 % / 20 %


@pytest.mark.unit
def test_a_long_stop_never_goes_below_zero():
    sig = sg.Signal("L", "Long", 70, "High", {}, close=10.0, last_date="x",
                    regime="neutral", atr=8.0)
    sg._attach_plan(sig)
    assert sig.stop == 0.0 and sig.size_pct == pytest.approx(1 / 1.6)


@pytest.mark.unit
def test_live_signal_carries_a_consistent_plan():
    sig = sg.compute_signal("UP", rising(wiggle=0.004), _bench(), "neutral")
    assert sig.entry == sig.close
    assert sig.stop == pytest.approx(sig.close - 2 * sig.atr)
    assert sig.target == pytest.approx(sig.close + 3 * sig.atr)
    assert sig.size_pct == pytest.approx(min(10.0, 1.0 / (2 * sig.atr / sig.close)))
    short = sg.compute_signal("DN", falling(wiggle=0.004), _bench(), "neutral")
    assert short.stop > short.close > short.target


@pytest.mark.unit
def test_short_history_is_skipped_with_a_reason():
    with pytest.raises(sg.SkipSignal, match="only 150 bars"):
        sg.compute_signal("NEW", rising(n=150), _bench(), "neutral")
    with pytest.raises(sg.SkipSignal):
        sg.compute_signal("NONE", pd.DataFrame({"Open": []}), None, "neutral")


@pytest.mark.unit
def test_no_volume_and_gaps_do_not_produce_nan():
    df = rising(wiggle=0.004, volume=0.0)           # an index: volume reported as zero
    df.iloc[100, df.columns.get_loc("High")] = np.nan
    df.iloc[101, df.columns.get_loc("Close")] = np.nan
    sig = sg.compute_signal("^IDX", df, _bench(), "neutral")
    assert sig.components["volume"] == 0.0
    assert np.isfinite(sig.score) and np.isfinite(sig.atr)
    assert all(np.isfinite(v) for v in sig.components.values())


@pytest.mark.unit
def test_zero_volatility_series_stay_finite():
    sig = sg.compute_signal("FLAT", flat(), _bench(), "neutral")
    assert np.isfinite(sig.score)
    assert sig.direction == "Neutral"


@pytest.mark.unit
def test_a_series_with_no_atr_gets_no_trade_plan():
    # no range and no change: the ATR is 0, so no stop can be placed
    flat_df = frame(np.full(300, 50.0), spread=0.0)
    sig = sg.compute_signal("Z", flat_df, _bench(300), "neutral")
    assert sig.atr == 0 and sig.direction == "Neutral" and sig.stop is None


@pytest.mark.unit
def test_benchmark_bars_after_the_instruments_last_bar_are_ignored():
    df = rising(n=300, wiggle=0.004)
    bench = flat(400)["Close"]
    base = sg.compute_signal("X", df, bench, "neutral")
    # a benchmark that explodes after the instrument's last bar must not change the signal
    later = bench.copy()
    later.loc[later.index > df.index[-1]] = 1e6
    assert sg.compute_signal("X", df, later, "neutral").score == pytest.approx(base.score)


@pytest.mark.unit
def test_reasons_are_derived_from_the_numbers_in_both_languages():
    sig = sg.compute_signal("UP", rising(wiggle=0.004), _bench(), "neutral")
    en, de = sig.reasons_text("en"), sig.reasons_text("de")
    assert 3 <= len(en) <= 5 and len(en) == len(de)
    gap = (sig.close / sig.metrics["sma200"] - 1) * 100
    assert any(f"Price {gap:.1f} % above its 200-day average" == r for r in en)
    assert any("über dem 200-Tage-Schnitt" in r for r in de)
    assert any("Outperformed SPY by" in r for r in en)


@pytest.mark.unit
def test_overbought_rsi_shows_up_as_a_counter_signal_for_a_long():
    sig = sg.compute_signal("UP", rising(daily=0.01), _bench(), "neutral")
    assert sig.rsi > 70
    assert any(r.startswith("Counter-signal: RSI") and "overbought" in r for r in sig.reasons)


# --- regime ---

def _closes(kind, n=300):
    return (rising(n) if kind == "up" else falling(n))["Close"]


@pytest.mark.unit
def test_regime_risk_on_when_everything_points_up():
    r = rg.compute_regime("SPY", _closes("up"), breadth=0.7, vix_close=pd.Series([15.0]))
    assert r.points == 5 and r.label == "risk-on"
    assert [f.points for f in r.facts] == [1, 1, 1, 1, 1]
    assert any("VIX at 15.0" in line for line in r.describe("en"))


@pytest.mark.unit
def test_regime_risk_off_when_everything_points_down():
    r = rg.compute_regime("SPY", _closes("down"), breadth=0.2, vix_close=pd.Series([32.0]))
    assert r.points == -5 and r.label == "risk-off"


@pytest.mark.unit
def test_regime_points_and_boundaries():
    up = _closes("up")
    # trend facts +3, breadth in the 40-60 band 0, VIX 18..25 band 0 -> exactly the risk-on bar
    r = rg.compute_regime("SPY", up, breadth=0.5, vix_close=pd.Series([20.0]))
    assert r.points == 3 and r.label == "risk-on"
    r = rg.compute_regime("SPY", up, breadth=0.3, vix_close=pd.Series([20.0]))
    assert r.points == 2 and r.label == "neutral"
    down = _closes("down")
    r = rg.compute_regime("SPY", down, breadth=0.5, vix_close=pd.Series([20.0]))
    assert r.points == -3 and r.label == "risk-off"
    # -2 is already risk-off: two down facts with the rest neutral
    assert rg.classify(-2) == "risk-off" and rg.classify(-1) == "neutral"
    assert rg.classify(3) == "risk-on" and rg.classify(2) == "neutral"


@pytest.mark.unit
def test_regime_without_vix_or_breadth_adds_no_points_and_says_so():
    r = rg.compute_regime("SPY", _closes("up"), breadth=None, vix_close=None)
    assert r.points == 3 and r.label == "risk-on"
    assert "VIX" in r.missing and "breadth" in r.missing
    r = rg.compute_regime("SPY", None, None, None)
    assert r.points == 0 and r.label == "neutral" and "benchmark trend" in r.missing


@pytest.mark.unit
def test_breadth_counts_instruments_above_their_200_day_average():
    closes = {f"U{i}": rising()["Close"] for i in range(3)}
    closes |= {f"D{i}": falling()["Close"] for i in range(2)}
    assert rg.breadth_above_sma200(closes) == pytest.approx(0.6)
    assert rg.breadth_above_sma200({"A": rising()["Close"]}) is None     # too few to call it breadth
    short = {f"S{i}": rising(n=100)["Close"] for i in range(6)}
    assert rg.breadth_above_sma200(short) is None                        # none has 200 bars
