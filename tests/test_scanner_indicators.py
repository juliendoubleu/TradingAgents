"""Indicators against values worked out by hand on tiny series."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.scanner import indicators as ind


@pytest.mark.unit
def test_sma_needs_a_full_window():
    out = ind.sma(pd.Series([1, 2, 3, 4, 5.0]), 3)
    assert np.isnan(out.iloc[:2]).all()
    assert out.iloc[2:].tolist() == [2.0, 3.0, 4.0]


@pytest.mark.unit
def test_ema_follows_the_recursion():
    # span 3 -> alpha 0.5: 1, then 0.5*2+0.5*1, then 0.5*3+0.5*1.5
    assert ind.ema(pd.Series([1, 2, 3.0]), 3).tolist() == [1.0, 1.5, 2.25]


@pytest.mark.unit
def test_rsi_wilder_by_hand():
    # n=2, changes +1 +1 -1 -1. Seed (bar 2): gain 1, loss 0 -> 100.
    # bar 3: gain (1*1+0)/2=.5, loss (0*1+1)/2=.5 -> RS 1 -> 50
    # bar 4: gain (.5*1+0)/2=.25, loss (.5*1+1)/2=.75 -> RS 1/3 -> 25
    out = ind.rsi(pd.Series([1, 2, 3, 2, 1.0]), 2)
    assert np.isnan(out.iloc[:2]).all()
    assert out.iloc[2:].tolist() == pytest.approx([100.0, 50.0, 25.0])


@pytest.mark.unit
def test_rsi_of_a_flat_series_is_neutral_not_nan():
    assert ind.rsi(pd.Series([5.0] * 10), 3).iloc[-1] == 50.0


@pytest.mark.unit
def test_rsi_of_a_falling_series_is_zero():
    assert ind.rsi(pd.Series(np.arange(20, 0, -1.0)), 5).iloc[-1] == pytest.approx(0.0)


@pytest.mark.unit
def test_atr_wilder_by_hand():
    high = pd.Series([10, 11, 12.0])
    low = pd.Series([9, 9, 10.0])
    close = pd.Series([9.5, 10, 11.0])
    # TR: 1 (no previous close), max(2, 1.5, .5)=2, max(2, 2, 0)=2
    # n=2: seed at bar 1 = (1+2)/2 = 1.5; bar 2 = (1.5*1 + 2)/2 = 1.75
    out = ind.atr(high, low, close, 2)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1:].tolist() == pytest.approx([1.5, 1.75])


@pytest.mark.unit
def test_true_range_uses_the_gap_from_the_previous_close():
    tr = ind.true_range(pd.Series([10.0, 20.0]), pd.Series([9.0, 19.0]), pd.Series([9.5, 19.5]))
    assert tr.tolist() == [1.0, 10.5]    # |20 - 9.5|


@pytest.mark.unit
def test_pct_return():
    out = ind.pct_return(pd.Series([100, 110, 121.0]), 2)
    assert np.isnan(out.iloc[:2]).all() and out.iloc[2] == pytest.approx(0.21)


@pytest.mark.unit
def test_realized_vol_of_constant_growth_is_zero_and_scales_with_sqrt_252():
    steady = pd.Series(100 * 1.01 ** np.arange(100))
    assert ind.realized_vol(steady, 63).iloc[-1] == pytest.approx(0.0, abs=1e-9)
    # alternating +1% / -1% log returns: sample std = 0.01*sqrt(n/(n-1)) with n = 63
    logs = np.array([0.01 if i % 2 else -0.01 for i in range(100)])
    price = pd.Series(100 * np.exp(np.cumsum(logs)))
    got = ind.realized_vol(price, 63).iloc[-1]
    window = logs[-63:]
    assert got == pytest.approx(np.std(window, ddof=1) * np.sqrt(252))


@pytest.mark.unit
def test_macd_histogram_of_a_constant_series_is_zero():
    assert ind.macd_hist(pd.Series([50.0] * 80)).abs().max() == pytest.approx(0.0)


@pytest.mark.unit
def test_macd_histogram_is_positive_when_a_series_accelerates_up():
    close = pd.Series(np.r_[np.full(60, 100.0), np.linspace(100, 130, 20)])
    assert ind.macd_hist(close).iloc[-1] > 0


@pytest.mark.unit
def test_distance_to_the_range_extremes():
    close = pd.Series(np.r_[np.linspace(10, 20, 70), [15.0]])
    assert ind.dist_from_high(close, 252).iloc[-1] == pytest.approx(15 / 20 - 1)
    assert ind.dist_from_low(close, 252).iloc[-1] == pytest.approx(15 / 10 - 1)


@pytest.mark.unit
def test_volume_ratio_and_missing_volume():
    volume = pd.Series([100.0] * 40 + [200.0] * 20)
    # last 20 average 200; last 60 average (40*100 + 20*200)/60
    assert ind.volume_ratio(volume).iloc[-1] == pytest.approx(200 / (8000 / 60))
    assert ind.volume_ratio(pd.Series([0.0] * 80)).isna().all()      # an index: no volume
    assert ind.volume_ratio(pd.Series([np.nan] * 80)).isna().all()
