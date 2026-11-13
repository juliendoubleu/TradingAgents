"""Scan ranking, skips, dates, universes and the Yahoo bulk loader's point-in-time cut."""

import numpy as np
import pandas as pd
import pytest

from tests.scanner_helpers import END, falling, flat, rising
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.vendors.yahoo import bulk
from tradingagents.scanner import scan as sc, universe as un


def _loader(frames):
    calls = []

    def load(symbols, as_of, lookback):
        calls.append((list(symbols), as_of, lookback))
        return {s: frames[s] for s in symbols if s in frames}
    load.calls = calls
    return load


def _market():
    frames = {"SPY": flat(), "^VIX": flat(start=15.0)}
    for i, daily in enumerate((0.004, 0.003, 0.002, 0.001)):
        frames[f"UP{i}"] = rising(daily=daily, wiggle=0.004)
    for i, daily in enumerate((0.004, 0.003, 0.002)):
        frames[f"DN{i}"] = falling(daily=daily, wiggle=0.004)
    frames["FLAT"] = flat()
    return frames


@pytest.mark.unit
def test_ranking_order_and_top_n():
    frames = _market()
    members = [s for s in frames if s not in ("SPY", "^VIX")]
    res = sc.run_scan("us", END, tickers=members, top=2, loader=_loader(frames))
    assert [s.symbol for s in res.longs] == [s.symbol for s in sorted(
        (x for x in res.signals if x.direction == "Long"), key=lambda s: -s.score)][:2]
    assert len(res.longs) == 2 and len(res.shorts) == 2
    assert all(a.score >= b.score for a, b in zip(res.longs, res.longs[1:], strict=False))
    assert all(a.score <= b.score for a, b in zip(res.shorts, res.shorts[1:], strict=False))
    assert all(s.direction == "Long" for s in res.longs)
    assert all(s.direction == "Short" for s in res.shorts)
    assert "FLAT" not in {s.symbol for s in res.longs + res.shorts}
    assert res.signals[0].score >= res.signals[-1].score


@pytest.mark.unit
def test_scan_fetches_members_benchmark_and_vix_and_reports_the_last_bar():
    frames = _market()
    load = _loader(frames)
    res = sc.run_scan("us", END, tickers=["UP0", "DN0"], loader=load)
    symbols, as_of, lookback = load.calls[0]
    assert symbols == ["UP0", "DN0", "SPY", "^VIX"] and as_of == END and lookback >= 400
    assert res.benchmark == "SPY" and res.as_of_date == END and res.custom is True
    assert res.regime.benchmark == "SPY"


@pytest.mark.unit
def test_skipped_symbols_carry_reasons():
    frames = _market()
    frames["SHORT"] = rising(n=120)
    frames["OLD"] = rising(n=400, end="2024-01-31")     # data stops far before the others
    res = sc.run_scan("us", END, tickers=["UP0", "SHORT", "OLD", "GHOST"], loader=_loader(frames))
    reasons = dict(res.skipped)
    assert "only 120 bars" in reasons["SHORT"]
    assert "stale" in reasons["OLD"]
    assert "no price data" in reasons["GHOST"]
    assert [s.symbol for s in res.signals] == ["UP0"]
    assert any("skipped" in w for w in res.warnings)


@pytest.mark.unit
def test_as_of_date_is_the_last_bar_used_and_old_data_is_flagged():
    frames = {s: f.iloc[:-9] for s, f in _market().items()}        # newest bar ~9 days earlier
    res = sc.run_scan("us", END, tickers=["UP0", "DN0"], loader=_loader(frames))
    assert res.as_of_date == frames["UP0"].index[-1].strftime("%Y-%m-%d") != END
    assert any("days before the requested" in w for w in res.warnings)


@pytest.mark.unit
def test_missing_benchmark_and_vix_are_reported_not_fatal():
    frames = {k: v for k, v in _market().items() if k not in ("SPY", "^VIX")}
    res = sc.run_scan("us", END, tickers=["UP0", "DN0"], loader=_loader(frames))
    assert any("Benchmark SPY had no data" in w for w in res.warnings)
    assert res.signals and all(s.components["relative_strength"] == 0 for s in res.signals)


@pytest.mark.unit
def test_parse_date_rejects_bad_and_future_dates():
    assert sc.parse_date("2025-01-31") == "2025-01-31"
    for bad in ("June 1", "2025-13-01", "2025-1-5", "2999-01-01"):
        with pytest.raises(ValueError):
            sc.parse_date(bad)
    assert len(sc.parse_date(None)) == 10


@pytest.mark.unit
def test_to_dict_is_json_serialisable():
    import json
    res = sc.run_scan("us", END, tickers=["UP0", "DN0"], loader=_loader(_market()))
    json.dumps(res.to_dict())


# --- universes ---

@pytest.mark.unit
def test_universes_have_no_duplicates_and_a_benchmark():
    for name, (description, benchmark, members) in un.UNIVERSES.items():
        assert description and benchmark, name
        assert len(members) == len(set(members)), name
        assert members, name
    assert un.UNIVERSES["dax"][1] == "^GDAXI" and len(un.UNIVERSES["dax"][2]) == 40
    assert all(t.endswith(".DE") for t in un.UNIVERSES["dax"][2])
    assert "November 2026" in un.UNIVERSES["dax"][0]
    assert len(un.UNIVERSES["sectors"][2]) == 11
    assert un.UNIVERSES["crypto"][1] == "BTC-USD"
    assert set(un.UNIVERSES["all"][2]) == set(
        un.UNIVERSES["us"][2] + un.UNIVERSES["sectors"][2] + un.UNIVERSES["global"][2])


@pytest.mark.unit
def test_resolve_universe_custom_tickers_and_errors():
    desc, bench, members = un.resolve_universe("dax", [" sap.de", "SIE.DE", "sap.de"])
    assert members == ["SAP.DE", "SIE.DE"] and bench == "^GDAXI" and "Custom" in desc
    with pytest.raises(ValueError, match="us, sectors, global, dax, crypto, all"):
        un.resolve_universe("mars")


# --- the Yahoo bulk loader (yf.download mocked) ---

def _download_frame(symbols, days=8, tz=None):
    index = pd.bdate_range(end="2025-06-30", periods=days)
    if tz:
        index = index.tz_localize(tz)
    cols = pd.MultiIndex.from_product([symbols, ["Open", "High", "Low", "Close", "Volume"]],
                                      names=["Ticker", "Price"])
    data = np.tile(np.arange(1, days + 1, dtype=float)[:, None], (1, len(cols)))
    return pd.DataFrame(data, index=index, columns=cols)


@pytest.mark.unit
def test_bulk_loader_cuts_at_the_as_of_date_and_drops_rows_without_a_close(monkeypatch):
    raw = _download_frame(["AAA", "BBB"], days=8, tz="America/New_York")
    raw[("AAA", "Close")] = raw[("AAA", "Close")].astype(float)
    raw.iloc[-1, raw.columns.get_loc(("AAA", "Close"))] = np.nan       # a row with no close
    seen = {}

    def fake(symbols, **kw):
        seen.update(kw, symbols=symbols)
        return raw
    monkeypatch.setattr(bulk.yf, "download", fake)

    # as-of two bars before the end: later rows must never appear
    as_of = raw.index[-3].strftime("%Y-%m-%d")
    out = bulk.load_bulk_ohlcv(["AAA", "BBB", "MISSING"], as_of, 30)
    assert set(out) == {"AAA", "BBB"}
    for frame_ in out.values():
        assert frame_.index.max() <= pd.Timestamp(as_of)
        assert frame_.index.tz is None and frame_.index.is_monotonic_increasing
        assert list(frame_.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(out["AAA"]) == 6
    # yfinance's end is exclusive, so it was asked for the day after as-of
    assert seen["end"] == (pd.Timestamp(as_of) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert seen["auto_adjust"] is True


@pytest.mark.unit
def test_bulk_loader_keeps_a_running_session_bar_for_the_caller_to_cut(monkeypatch):
    """Yahoo returns today's bar with the live price as its Close, so it has a close and
    stays; deciding that it is unfinished is the scanner's job (``session.settled``)."""
    raw = _download_frame(["AAA"], days=6)
    live = raw.index[-1]
    monkeypatch.setattr(bulk.yf, "download", lambda symbols, **kw: raw)
    out = bulk.load_bulk_ohlcv(["AAA"], live.strftime("%Y-%m-%d"), 30)
    assert out["AAA"].index[-1] == live and out["AAA"]["Close"].iloc[-1] == 6.0


@pytest.fixture(autouse=True)
def _no_retry_pause(monkeypatch):
    monkeypatch.setattr(bulk, "RETRY_PAUSE_S", 0.0)


@pytest.mark.unit
def test_bulk_loader_retries_a_symbol_that_came_back_empty_once_on_its_own(monkeypatch):
    calls = []

    def fake(symbols, **kw):
        calls.append(list(symbols))
        if len(symbols) > 1:                       # the throttled batch loses BBB silently
            return _download_frame([s for s in symbols if s != "BBB"])
        return _download_frame(symbols)
    monkeypatch.setattr(bulk.yf, "download", fake)
    out = bulk.load_bulk_ohlcv(["AAA", "BBB", "CCC"], "2025-06-30", 30)
    assert set(out) == {"AAA", "BBB", "CCC"}
    assert calls == [["AAA", "BBB", "CCC"], ["BBB"]]


@pytest.mark.unit
def test_bulk_loader_retries_once_only_and_a_failed_retry_is_not_fatal(monkeypatch):
    calls = []

    def fake(symbols, **kw):
        calls.append(list(symbols))
        if symbols == ["BBB"]:
            raise RuntimeError("boom")             # becomes VendorUnavailableError in yf_retry
        return _download_frame(["AAA"])
    monkeypatch.setattr(bulk.yf, "download", fake)
    out = bulk.load_bulk_ohlcv(["AAA", "BBB"], "2025-06-30", 30)
    assert set(out) == {"AAA"} and calls.count(["BBB"]) == 1

    monkeypatch.setattr(bulk.yf, "download", lambda symbols, **kw: pd.DataFrame())
    with pytest.raises(VendorUnavailableError):
        bulk.load_bulk_ohlcv(["AAA"], "2025-06-30", 30)


@pytest.mark.unit
def test_bulk_loader_drops_nan_close_rows_without_a_cut(monkeypatch):
    raw = _download_frame(["AAA"], days=6)
    raw.iloc[-1, :] = np.nan
    monkeypatch.setattr(bulk.yf, "download", lambda symbols, **kw: raw)
    out = bulk.load_bulk_ohlcv(["AAA"], "2025-06-30", 30)
    assert len(out["AAA"]) == 5 and out["AAA"]["Close"].notna().all()


@pytest.mark.unit
def test_bulk_loader_batches_and_raises_when_nothing_comes_back(monkeypatch):
    batches = []

    def fake(symbols, **kw):
        batches.append(len(symbols))
        return _download_frame(symbols)
    monkeypatch.setattr(bulk.yf, "download", fake)
    names = [f"S{i}" for i in range(120)]
    assert len(bulk.load_bulk_ohlcv(names, "2025-06-30", 30)) == 120
    assert batches == [50, 50, 20]

    monkeypatch.setattr(bulk.yf, "download", lambda symbols, **kw: pd.DataFrame())
    with pytest.raises(VendorUnavailableError):
        bulk.load_bulk_ohlcv(["AAA"], "2025-06-30", 30)


@pytest.mark.unit
def test_bulk_loader_handles_a_single_symbol_with_flat_columns(monkeypatch):
    raw = _download_frame(["AAA"])["AAA"]          # a frame without the symbol level
    monkeypatch.setattr(bulk.yf, "download", lambda symbols, **kw: raw)
    assert len(bulk.load_bulk_ohlcv(["AAA"], "2025-06-30", 30)["AAA"]) == 8

