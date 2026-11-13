"""Golden and boundary tests: a wrong weight, lookback, threshold or sign must fail one of these.

The golden value is computed here by plain loops (no ``indicators`` helpers, no ``signals``
internals), so it is an independent statement of the formulas in the module docstrings.
"""

import math

import numpy as np
import pandas as pd
import pytest

from tests.scanner_helpers import END, falling, flat, frame, rising
from tradingagents.scanner import regime as rg, signals as sg, validation as va

# --- independent reference implementation -------------------------------------------------

def _ema(values, span):
    alpha, out = 2.0 / (span + 1), []
    for i, v in enumerate(values):
        out.append(v if i == 0 else alpha * v + (1 - alpha) * out[-1])
    return out


def _wilder(values, n):
    avg = sum(values[:n]) / n
    for v in values[n:]:
        avg = (avg * (n - 1) + v) / n
    return avg


def _sign(x):
    return float(np.sign(x))


def _hist(c):
    line = [a - b for a, b in zip(_ema(c, 12), _ema(c, 26), strict=True)]
    return [a - b for a, b in zip(line, _ema(line, 9), strict=True)]


def reference(df, bench, bars_per_month=21, bars_per_year=252, macd_back=5):
    c = df["Close"].to_numpy(float)
    h, lo, v = (df[k].to_numpy(float) for k in ("High", "Low", "Volume"))
    last = c[-1]
    sma50, sma200 = c[-50:].mean(), c[-200:].mean()
    sma50_20 = c[-70:-20].mean()
    trend = np.mean([_sign(last - sma50), _sign(last - sma200), _sign(sma50 - sma200),
                     _sign(sma50 - sma50_20)])
    m3, m6 = 3 * bars_per_month, 6 * bars_per_month
    r3, r6 = last / c[-1 - m3] - 1, last / c[-1 - m6] - 1
    r1 = last / c[-1 - bars_per_month] - 1
    logret = np.log(c[1:] / c[:-1])[-m3:]
    vol = max(float(np.std(logret, ddof=1) * math.sqrt(bars_per_year)), 0.01)
    momentum = (math.tanh(r6 / (vol * math.sqrt(m6 / bars_per_year)))
                + math.tanh(r3 / (vol * math.sqrt(m3 / bars_per_year)))) / 2
    b = bench["Close"].to_numpy(float)
    rel = math.tanh((r3 - (b[-1] / b[-1 - m3] - 1)) / 0.10)
    hist = _hist(c)
    macd = np.mean([_sign(hist[-1]), _sign(hist[-1] - hist[-1 - macd_back])])
    d = np.diff(c)
    rsi_avg_g, rsi_avg_l = _wilder(np.maximum(d, 0).tolist(), 14), _wilder(
        np.maximum(-d, 0).tolist(), 14)
    # _wilder above seeds at the first 14 changes, the same as the module's convention
    rsi = 100 - 100 / (1 + rsi_avg_g / rsi_avg_l)
    rsi_c = 0.0 if 30 <= rsi <= 70 else max(-1.0, min(1.0, (50 - rsi) / 30))
    vol_c = _sign(r1) * max(-1.0, min(1.0, v[-20:].mean() / v[-60:].mean() - 1))
    comps = {"trend": trend, "momentum": momentum, "relative_strength": rel, "macd": macd,
             "rsi_guard": rsi_c, "volume": vol_c}
    weights = {"trend": 0.30, "momentum": 0.25, "relative_strength": 0.20, "macd": 0.10,
               "rsi_guard": 0.10, "volume": 0.05}
    score = 100 * sum(weights[k] * comps[k] for k in comps)
    return comps, score, rsi


def _walk(seed, n=420, drift=0.0006, sigma=0.012, volume_seed=None):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, sigma, n)))
    out = frame(close, spread=0.0)
    wide = rng.uniform(0.004, 0.02, n)
    out["High"] = close * (1 + wide)
    out["Low"] = close * (1 - wide)
    out["Volume"] = rng.uniform(5e5, 2e6, n)
    return out


# --- golden --------------------------------------------------------------------------------

@pytest.mark.unit
def test_golden_components_and_score_match_an_independent_computation():
    df = _walk(seed=48)
    bench = _walk(seed=11, drift=0.0003)
    sig = sg.compute_signal("X", df, bench["Close"], "neutral")
    comps, score, _rsi = reference(df, bench)
    for key, value in comps.items():
        assert sig.components[key] == pytest.approx(value, abs=1e-9), key
    assert sig.score == pytest.approx(score, abs=1e-9)
    # non-trivial: the golden case exercises distinct, non-saturated values, so swapped
    # weights or a dropped sign cannot hide behind identical or zero components
    assert abs(comps["trend"] - comps["momentum"]) > 1e-3
    assert all(abs(v) > 0.05 for v in comps.values())


@pytest.mark.unit
def test_golden_for_a_crypto_calendar():
    df, bench = _walk(seed=3, n=480), _walk(seed=5, n=480, drift=0.0002)
    params = sg.SignalParams(month_bars=30, bars_per_year=365)
    sig = sg.compute_signal("X", df, bench["Close"], "neutral", params=params)
    comps, score, _ = reference(df, bench, 30, 365)
    for key, value in comps.items():
        assert sig.components[key] == pytest.approx(value, abs=1e-9), key
    assert sig.score == pytest.approx(score, abs=1e-9)
    assert sig.horizon_bars == 30


# --- lookbacks and signs --------------------------------------------------------------------

def _find(predicate, n=420, limit=4000):
    for seed in range(limit):
        df = _walk(seed, n=n, drift=0.0, sigma=0.01)
        if predicate(df["Close"].to_numpy(float)):
            return df
    raise AssertionError("no discriminating series found")


def _macd_differs(c):
    h = _hist(c)
    return _sign(h[-1]) != 0 and _sign(h[-1] - h[-2]) != _sign(h[-1] - h[-6]) \
        and _sign(h[-1] - h[-6]) != 0


def _slope_differs(c):
    s_now = c[-50:].mean()
    return _sign(s_now - c[-70:-20].mean()) != _sign(s_now - c[-55:-5].mean()) \
        and _sign(s_now - c[-70:-20].mean()) != 0


@pytest.mark.unit
def test_macd_compares_the_histogram_with_five_bars_ago_not_one():
    df = _find(_macd_differs)
    sig = sg.compute_signal("X", df, flat(420)["Close"], "neutral")
    h = _hist(df["Close"].to_numpy(float))
    expected = np.mean([_sign(h[-1]), _sign(h[-1] - h[-6])])
    assert sig.components["macd"] == pytest.approx(expected)
    assert sig.components["macd"] != pytest.approx(np.mean([_sign(h[-1]), _sign(h[-1] - h[-2])]))


@pytest.mark.unit
def test_sma50_slope_looks_20_bars_back_not_5():
    df = _find(_slope_differs)
    c = df["Close"].to_numpy(float)
    sig = sg.compute_signal("X", df, flat(420)["Close"], "neutral")
    expected = np.mean([_sign(c[-1] - c[-50:].mean()), _sign(c[-1] - c[-200:].mean()),
                        _sign(c[-50:].mean() - c[-200:].mean()),
                        _sign(c[-50:].mean() - c[-70:-20].mean())])
    assert sig.components["trend"] == pytest.approx(expected)


@pytest.mark.unit
def test_volume_component_takes_the_sign_of_the_one_month_return():
    for maker, sign in ((rising, 1), (falling, -1)):
        df = maker(420, daily=0.002, wiggle=0.004)
        df["Volume"] = np.r_[np.full(400, 1e6), np.full(20, 2e6)]        # 20d / 60d = 1.5
        ratio = df["Volume"].iloc[-20:].mean() / df["Volume"].iloc[-60:].mean()
        sig = sg.compute_signal("X", df, flat(420)["Close"], "neutral")
        assert sig.components["volume"] == pytest.approx(sign * min(1.0, ratio - 1.0))
        assert sign * sig.components["volume"] > 0.3


@pytest.mark.unit
def test_volume_component_is_zero_without_volume():
    sig = sg.compute_signal("X", rising(420, volume=0.0), flat(420)["Close"], "neutral")
    assert sig.components["volume"] == 0.0


@pytest.mark.unit
def test_weights_are_the_documented_ones():
    assert sg.WEIGHTS == {"trend": 0.30, "momentum": 0.25, "relative_strength": 0.20,
                          "macd": 0.10, "rsi_guard": 0.10, "volume": 0.05}


# --- MIN_BARS ---------------------------------------------------------------------------------

@pytest.mark.unit
def test_min_bars_is_exactly_220_for_stocks_and_crypto():
    bench = flat(400)["Close"]
    with pytest.raises(sg.SkipSignal, match="only 219 bars of history, need 220"):
        sg.compute_signal("X", rising(219), bench, "neutral")
    assert sg.compute_signal("X", rising(220), bench, "neutral").symbol == "X"
    assert sg.SignalParams().min_bars == 220
    assert sg.SignalParams(30, 365).min_bars == 220
    assert sg.SignalParams(60, 365).min_bars == 380          # 6 * 60 + 20 beats the floor


# --- regime inputs ----------------------------------------------------------------------------

def _up():
    return rising(300)["Close"]


def _pts(key, **kw):
    r = rg.compute_regime("SPY", _up(), **kw)
    return next(f.points for f in r.facts if f.key == key)


@pytest.mark.unit
@pytest.mark.parametrize("vix,points", [(17.99, 1), (18.0, 0), (21.5, 0), (25.0, 0),
                                       (25.01, -1), (19.0, 0), (29.0, -1)])
def test_vix_boundaries_18_and_25(vix, points):
    assert _pts("vix", breadth=0.5, vix_close=pd.Series([vix])) == points


@pytest.mark.unit
@pytest.mark.parametrize("breadth,points", [(0.61, 1), (0.6, 0), (0.55, 0), (0.5, 0),
                                           (0.45, 0), (0.4, 0), (0.39, -1)])
def test_breadth_boundaries_60_and_40_percent(breadth, points):
    assert _pts("breadth", breadth=breadth, vix_close=None) == points


@pytest.mark.unit
def test_regime_verdict_boundaries_end_to_end():
    up, down = _up(), falling(300)["Close"]
    assert rg.compute_regime("SPY", up, 0.5, pd.Series([20.0])).label == "risk-on"      # +3
    assert rg.compute_regime("SPY", up, 0.3, pd.Series([20.0])).label == "neutral"      # +2
    assert rg.compute_regime("SPY", down, 0.5, pd.Series([20.0])).label == "risk-off"   # -3
    mixed = pd.concat([falling(300)["Close"].iloc[:150], rising(300)["Close"].iloc[150:]])
    two_down = rg.compute_regime("SPY", mixed, None, None)
    assert (rg.RISK_ON_AT, rg.RISK_OFF_AT) == (3, -2)
    assert two_down.points in range(-3, 4)
    assert rg.classify(-2) == "risk-off" and rg.classify(-1) == "neutral"
    assert rg.classify(3) == "risk-on" and rg.classify(2) == "neutral"


# --- hit-rate definitions ---------------------------------------------------------------------

@pytest.mark.unit
def test_a_zero_return_is_no_hit_for_longs_and_for_shorts():
    rows = [("Long", 0.0, 0.0), ("Long", 0.02, 0.0), ("Long", -0.01, 0.0), ("Long", 0.0, 0.0)]
    assert va._stats("Long", rows).hit_rate == pytest.approx(0.25)
    rows = [("Short", 0.0, 0.0), ("Short", -0.02, 0.0), ("Short", 0.01, 0.0), ("Short", 0.0, 0.0)]
    assert va._stats("Short", rows).hit_rate == pytest.approx(0.25)


@pytest.mark.unit
def test_the_end_date_constant_is_a_business_day():
    assert pd.Timestamp(END).weekday() < 5
