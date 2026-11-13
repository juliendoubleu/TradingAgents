"""Implied moves from at-the-money straddles, with yfinance mocked."""

from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.vendors.yahoo import options as opt

TODAY = date.today()


def _iso(days):
    return (TODAY + timedelta(days=days)).isoformat()


def _side(rows):
    """rows: (strike, bid, ask, last[, oi, iv])"""
    cols = ["strike", "bid", "ask", "lastPrice", "openInterest", "impliedVolatility"]
    data = [list(r) + [10, 0.5][len(r) - 4:] if len(r) < 6 else list(r) for r in rows]
    return pd.DataFrame(data, columns=cols)


class FakeTicker:
    def __init__(self, expiries, chains, spot=100.0):
        self.options = tuple(expiries)
        self._chains = chains
        self.fast_info = {"lastPrice": spot}

    def option_chain(self, expiry):
        calls, puts = self._chains[expiry]
        return SimpleNamespace(calls=_side(calls), puts=_side(puts))

    def history(self, **kw):
        return pd.DataFrame({"Close": [99.0, 101.0]})


@pytest.fixture
def patch_ticker(monkeypatch):
    def install(ticker):
        monkeypatch.setattr(opt.yf, "Ticker", lambda symbol: ticker)
        monkeypatch.setattr("tradingagents.dataflows.vendors.yahoo.common.time.sleep",
                            lambda s: None)
    return install


@pytest.mark.unit
def test_nearest_common_strike_and_mid_prices(patch_ticker):
    e = _iso(7)
    calls = [(90, 11, 12, 11.4), (100, 4.0, 4.4, 4.1), (105, 2, 2.2, 2.1)]
    puts = [(100, 3.6, 4.0, 3.9), (105, 7, 7.4, 7.2)]  # no 90 put
    patch_ticker(FakeTicker([e], {e: (calls, puts)}, spot=101.0))
    (m,) = opt.load_implied_moves("TTWO")
    assert m.strike == 100.0  # 101 is nearest to 100 among strikes in both
    assert m.call_price == pytest.approx(4.2) and m.put_price == pytest.approx(3.8)
    assert m.straddle == pytest.approx(8.0)
    assert m.move_pct == pytest.approx(8.0 / 101.0)
    assert m.price_basis == "mid" and m.stale is False
    assert m.days_to_expiry == 7 and m.expiry == e
    assert m.call_open_interest == 10 and m.call_iv == 0.5


@pytest.mark.unit
def test_zero_bid_ask_falls_back_to_last_and_is_stale(patch_ticker):
    e = _iso(3)
    calls = [(100, 0, 0, 4.0)]
    puts = [(100, 0, 0, 3.0)]
    patch_ticker(FakeTicker([e], {e: (calls, puts)}))
    (m,) = opt.load_implied_moves("TTWO")
    assert m.straddle == 7.0 and m.price_basis == "last" and m.stale is True


@pytest.mark.unit
def test_mixed_basis_is_flagged(patch_ticker):
    e = _iso(3)
    patch_ticker(FakeTicker([e], {e: ([(100, 4, 4.2, 4.1)], [(100, 0, 0, 3.0)])}))
    (m,) = opt.load_implied_moves("TTWO")
    assert m.price_basis == "mixed" and m.stale is True
    assert m.straddle == pytest.approx(4.1 + 3.0)


@pytest.mark.unit
def test_expiry_with_an_unusable_leg_is_skipped(patch_ticker):
    bad, good = _iso(2), _iso(9)
    chains = {
        bad: ([(100, 0, 0, 0)], [(100, 3, 3.2, 3.1)]),       # call has no price at all
        good: ([(100, 4, 4.2, 4.1)], [(100, 3, 3.2, 3.1)]),
    }
    patch_ticker(FakeTicker([bad, good], chains))
    moves = opt.load_implied_moves("TTWO")
    assert [m.expiry for m in moves] == [good]


@pytest.mark.unit
def test_past_expiries_and_max_expiries(patch_ticker):
    exps = [_iso(-3), _iso(0), _iso(5), _iso(12), _iso(19)]
    one = ([(100, 4, 4.2, 4.1)], [(100, 3, 3.2, 3.1)])
    patch_ticker(FakeTicker(exps, {e: one for e in exps}))
    moves = opt.load_implied_moves("TTWO", max_expiries=2)
    assert [m.expiry for m in moves] == [_iso(0), _iso(5)]
    assert moves[0].days_to_expiry == 0


@pytest.mark.unit
def test_placeholder_iv_is_dropped(patch_ticker):
    e = _iso(3)
    calls = [(100, 4, 4.2, 4.1, 0, 0.00001)]
    puts = [(100, 3, 3.2, 3.1, 5, 0.42)]
    patch_ticker(FakeTicker([e], {e: (calls, puts)}))
    (m,) = opt.load_implied_moves("TTWO")
    assert m.call_iv is None and m.put_iv == 0.42
    assert m.call_open_interest == 0 and m.put_open_interest == 5


@pytest.mark.unit
def test_no_listed_options_is_an_empty_answer(patch_ticker):
    patch_ticker(FakeTicker([], {}))
    assert opt.load_implied_moves("SAP.DE") == []


@pytest.mark.unit
def test_vendor_failure_is_reported_as_unavailable(monkeypatch, patch_ticker):
    class Boom:
        @property
        def options(self):
            raise RuntimeError("connection reset")

    patch_ticker(Boom())
    with pytest.raises(VendorUnavailableError):
        opt.load_implied_moves("TTWO")


@pytest.mark.unit
def test_spot_falls_back_to_history(patch_ticker):
    e = _iso(3)
    t = FakeTicker([e], {e: ([(100, 4, 4.2, 4.1)], [(100, 3, 3.2, 3.1)])})
    t.fast_info = {}
    patch_ticker(t)
    (m,) = opt.load_implied_moves("TTWO")
    assert m.spot == 101.0


@pytest.mark.unit
def test_to_dict_roundtrips(patch_ticker):
    e = _iso(3)
    patch_ticker(FakeTicker([e], {e: ([(100, 4, 4.2, 4.1)], [(100, 3, 3.2, 3.1)])}))
    (m,) = opt.load_implied_moves("TTWO")
    d = m.to_dict()
    assert d["symbol"] == "TTWO" and d["stale"] is False
