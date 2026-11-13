"""The intraday loader with ``yf.download`` mocked: time zones, de-duplication, emptiness."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.vendors.yahoo import intraday


def _download(symbols, index, with_nan_row=True):
    cols = pd.MultiIndex.from_product([symbols, ["Open", "High", "Low", "Close", "Volume"]])
    data = np.tile([10.0, 11.0, 9.0, 10.5, 1000.0], (len(index), len(symbols)))
    df = pd.DataFrame(data, index=index, columns=cols)
    if with_nan_row:
        df.loc[index[-1], (symbols[0], "Close")] = np.nan
    return df


@pytest.mark.unit
def test_bars_are_converted_to_exchange_time_sorted_and_deduplicated(monkeypatch):
    # 14:30 UTC in winter is 09:30 in New York; the duplicate and the NaN close must go
    idx = pd.DatetimeIndex(["2024-01-03 15:30", "2024-01-03 14:30", "2024-01-03 14:30",
                            "2024-01-03 16:30"], tz="UTC")
    calls = []

    def fake_download(batch, **kwargs):
        calls.append((list(batch), kwargs))
        return _download(["AAA", "BBB"], idx)

    monkeypatch.setattr(intraday.yf, "download", fake_download)
    out = intraday.load_intraday_ohlcv(["AAA", "BBB", "MISSING"], "60m", 5000, pause=0)
    assert set(out) == {"AAA", "BBB"}                      # an absent symbol is not in the dict
    a = out["AAA"]
    assert str(a.index.tz) == "America/New_York"
    assert [t.strftime("%H:%M") for t in a.index] == ["09:30", "10:30"]   # NaN-close row dropped
    assert a.index.is_monotonic_increasing and not a.index.has_duplicates
    assert list(a.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(out["BBB"]) == 3
    kwargs = calls[0][1]
    assert kwargs["prepost"] is False and kwargs["interval"] == "60m"


@pytest.mark.unit
def test_requested_window_is_clamped_inside_yahoos_limits(monkeypatch):
    seen = {}

    def fake_download(batch, **kwargs):
        seen.update(kwargs)
        return _download(["AAA"], pd.DatetimeIndex(["2024-01-03 14:30"], tz="UTC"), False)

    monkeypatch.setattr(intraday.yf, "download", fake_download)
    intraday.load_intraday_ohlcv(["AAA"], "5m", 500, pause=0)
    span = (pd.Timestamp(seen["end"]) - pd.Timestamp(seen["start"])).days
    assert span <= 59 + 1


@pytest.mark.unit
def test_batches_of_25_and_an_empty_answer_is_an_error(monkeypatch):
    sizes = []

    def fake_download(batch, **kwargs):
        sizes.append(len(batch))
        return pd.DataFrame()

    monkeypatch.setattr(intraday.yf, "download", fake_download)
    with pytest.raises(VendorUnavailableError):
        intraday.load_intraday_ohlcv([f"S{i}" for i in range(60)], "60m", 30, pause=0)
    assert sizes == [25, 25, 10]


@pytest.mark.unit
def test_unknown_interval_is_refused():
    with pytest.raises(ValueError):
        intraday.load_intraday_ohlcv(["AAA"], "1m", 5)


@pytest.mark.unit
def test_a_failing_batch_keeps_what_was_fetched_and_logs_the_failed_symbols(monkeypatch, caplog):
    idx = pd.DatetimeIndex(["2024-01-03 14:30", "2024-01-03 15:30"], tz="UTC")

    def fake_download(batch, **kwargs):
        if "S25" in batch:
            raise RuntimeError("boom")
        return _download(list(batch), idx, with_nan_row=False)

    monkeypatch.setattr(intraday.yf, "download", fake_download)
    with caplog.at_level("WARNING"):
        out = intraday.load_intraday_ohlcv([f"S{i}" for i in range(30)], "60m", 30, pause=0)
    assert set(out) == {f"S{i}" for i in range(25)}          # batch 1 survived, batch 2 is out
    assert "S25" in caplog.text and "S29" in caplog.text


@pytest.mark.unit
def test_the_load_raises_only_when_every_batch_failed(monkeypatch):
    def fake_download(batch, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(intraday.yf, "download", fake_download)
    with pytest.raises(VendorUnavailableError):
        intraday.load_intraday_ohlcv(["AAA", "BBB"], "60m", 30, pause=0)


@pytest.mark.unit
def test_the_window_is_anchored_on_an_explicit_end_date_not_the_wall_clock(monkeypatch):
    seen = {}

    def fake_download(batch, **kwargs):
        seen.update(kwargs)
        return _download(["AAA"], pd.DatetimeIndex(["2024-06-14 14:30"], tz="UTC"), False)

    monkeypatch.setattr(intraday.yf, "download", fake_download)
    intraday.load_intraday_ohlcv(["AAA"], "60m", 30, pause=0, end="2024-06-14")
    assert seen["start"] == "2024-05-15" and seen["end"] == "2024-06-15"   # end is exclusive
