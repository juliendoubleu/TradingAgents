"""Session settlement, universe metadata, crypto calendar, VIX use, partial universes,
translated notices, and the validation window / spreads."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import cli.main as m
from tests.scanner_helpers import END, falling, flat, frame, rising
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.scanner import (
    notices as nt,
    scan as sc,
    session as ss,
    signals as sg,
    universe as un,
    validation as va,
)


def utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


# --- session settlement -----------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("symbol,day,before,at", [
    # US: 16:00 + 20 min New York; EST in November (UTC-5), EDT in July (UTC-4)
    ("AAPL", "2026-11-13", "2026-11-13T21:19", "2026-11-13T21:20"),
    ("AAPL", "2026-07-15", "2026-07-15T20:19", "2026-07-15T20:20"),
    ("^GSPC", "2026-11-13", "2026-11-13T21:19", "2026-11-13T21:20"),
    ("^VIX", "2026-11-13", "2026-11-13T21:34", "2026-11-13T21:35"),      # 16:15 + 20
    # Xetra 17:45 + 20 min; CET before and CEST after the last Sunday of March 2026 (29th)
    ("SAP.DE", "2026-03-27", "2026-03-27T17:04", "2026-03-27T17:05"),
    ("SAP.DE", "2026-03-30", "2026-03-30T16:04", "2026-03-30T16:05"),
    ("^GDAXI", "2026-11-13", "2026-11-13T17:04", "2026-11-13T17:05"),
    ("BP.L", "2026-07-15", "2026-07-15T15:54", "2026-07-15T15:55"),     # 16:35 BST
    ("AIR.PA", "2026-11-13", "2026-11-13T16:54", "2026-11-13T16:55"),    # 17:35 CET
    ("7203.T", "2026-11-13", "2026-11-13T06:49", "2026-11-13T06:50"),    # 15:30 JST
    ("0700.HK", "2026-11-13", "2026-11-13T08:29", "2026-11-13T08:30"),   # 16:10 HKT
    ("RY.TO", "2026-11-13", "2026-11-13T21:19", "2026-11-13T21:20"),
    ("BHP.AX", "2026-11-13", "2026-11-13T05:31", "2026-11-13T05:32"),    # 16:12 AEDT (+11)
    ("TCS.NS", "2026-11-13", "2026-11-13T10:19", "2026-11-13T10:20"),    # 15:30 IST (+5:30)
    ("600519.SS", "2026-11-13", "2026-11-13T07:19", "2026-11-13T07:20"), # 15:00 CST
    ("PKN.WA", "2026-11-13", "2026-11-13T16:24", "2026-11-13T16:25"),    # 17:05 CET
])
def test_a_bar_is_settled_20_minutes_after_its_exchange_session_ends(symbol, day, before, at):
    assert not ss.is_settled(symbol, day, utc(before))
    assert ss.is_settled(symbol, day, utc(at))
    assert ss.is_settled(symbol, day, utc(at).replace(year=2027))     # any later time


@pytest.mark.unit
@pytest.mark.parametrize("symbol", ["BTC-USD", "GC=F", "EURUSD=X", "XYZ.QQ", "^FOO"])
def test_crypto_futures_and_unknown_symbols_settle_once_the_utc_day_is_over(symbol):
    assert not ss.is_settled(symbol, "2026-11-13", utc("2026-11-13T23:59"))
    assert ss.is_settled(symbol, "2026-11-13", utc("2026-11-14T00:00"))


@pytest.mark.unit
def test_a_naive_now_is_read_as_utc_and_the_previous_day_is_always_settled():
    assert ss.is_settled("AAPL", "2026-11-12", datetime(2026, 11, 13, 0, 0))
    assert not ss.is_settled("AAPL", "2026-11-13", datetime(2026, 11, 13, 12, 0))


@pytest.mark.unit
def test_settled_drops_only_running_trailing_bars_and_counts_them():
    idx = pd.to_datetime(["2026-11-11", "2026-11-12", "2026-11-13"])
    f = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx)
    now = utc("2026-11-13T10:00")      # Berlin is mid-session, New York not yet open
    out, trimmed = ss.drop_unsettled({"SAP.DE": f, "AAPL": f, "ONE.DE": f.iloc[2:]}, now)
    assert len(out["SAP.DE"]) == 2 and len(out["AAPL"]) == 2
    assert "ONE.DE" not in out                                   # nothing left: no data
    assert sorted(trimmed) == ["AAPL", "ONE.DE", "SAP.DE"]
    assert list(ss.settled({"SAP.DE": f}, utc("2026-11-13T17:04"))["SAP.DE"].index)[-1] == idx[1]
    assert len(ss.settled({"SAP.DE": f}, utc("2026-11-13T18:10"))["SAP.DE"]) == 3
    assert len(f) == 3                                           # the input is untouched


# --- universe metadata ------------------------------------------------------------------------

@pytest.mark.unit
def test_universe_info_for_every_universe():
    expect = {"us": ("stocks_us", 252, 21, True), "dax": ("stocks_dax", 252, 21, False),
              "sectors": ("etfs", 252, 21, True), "global": ("etfs", 252, 21, True),
              "crypto": ("crypto", 365, 30, False), "all": ("mixed", 252, 21, True)}
    for name, (family, per_year, month, vix) in expect.items():
        info = un.universe_info(name)
        desc, bench, members = un.resolve_universe(name)
        assert (info.family, info.bars_per_year, info.month_bars, info.use_vix) == (
            family, per_year, month, vix), name
        assert info.name == name and info.benchmark == bench and info.members == members
        assert info.description["en"] == desc and info.description["de"]
    assert un.universe_info("dax").description["en"].startswith("DAX 40 members as of November 2026")
    custom = un.universe_info("crypto", ["btc-usd"])
    assert custom.members == ["BTC-USD"] and custom.bars_per_year == 365
    assert "Custom" in custom.description["en"] and "Eigene" in custom.description["de"]


@pytest.mark.unit
def test_dax_has_the_current_40_members():
    members = un.UNIVERSES["dax"][2]
    assert len(members) == len(set(members)) == 40
    assert {"G24.DE", "G1A.DE", "HOT.DE"} <= set(members)
    assert not {"P911.DE", "SRT3.DE", "PAH3.DE"} & set(members)


# --- SignalParams / crypto calendar ------------------------------------------------------------

@pytest.mark.unit
def test_default_params_are_todays_stock_numbers_and_crypto_params_scale_the_windows():
    d, c = sg.SignalParams(), sg.SignalParams(30, 365)
    assert (d.ret_1m, d.ret_3m, d.ret_6m, d.vol_window, d.horizon_bars, d.min_bars) == (
        21, 63, 126, 63, 21, 220)
    assert (c.ret_1m, c.ret_3m, c.ret_6m, c.vol_window, c.horizon_bars) == (30, 90, 180, 90, 30)
    df, bench = rising(420, wiggle=0.01), flat(420)["Close"]
    base = sg.compute_signal("X", df, bench, "neutral")
    explicit = sg.compute_signal("X", df, bench, "neutral", params=d)
    assert base.to_dict() == explicit.to_dict() and base.horizon_bars == 21


@pytest.mark.unit
def test_crypto_params_change_the_windows_and_annualisation():
    df, bench = rising(420, wiggle=0.01), flat(420)["Close"]
    crypto = sg.compute_signal("X", df, bench, "neutral", params=sg.SignalParams(30, 365))
    close = df["Close"].to_numpy()
    assert crypto.metrics["ret63"] == pytest.approx(close[-1] / close[-91] - 1)
    assert crypto.metrics["ret126"] == pytest.approx(close[-1] / close[-181] - 1)
    assert crypto.metrics["ret21"] == pytest.approx(close[-1] / close[-31] - 1)
    logret = np.diff(np.log(close))[-90:]
    assert crypto.metrics["vol"] == pytest.approx(logret.std(ddof=1) * np.sqrt(365))
    assert crypto.horizon_bars == 30
    stock = sg.compute_signal("X", df, bench, "neutral")
    assert crypto.components["momentum"] != pytest.approx(stock.components["momentum"])


# --- scan: VIX use, running bars, partial universes ---------------------------------------------

def _loader(frames):
    calls = []

    def load(symbols, as_of, lookback):
        calls.append(list(symbols))
        return {s: frames[s] for s in symbols if s in frames}
    load.calls = calls
    return load


def _market(n=420):
    frames = {"SPY": flat(n), "^VIX": flat(n, start=15.0), "BTC-USD": flat(n), "^GDAXI": flat(n)}
    for i in range(3):
        frames[f"UP{i}"] = rising(n, daily=0.002 + 0.001 * i, wiggle=0.004)
        frames[f"DN{i}"] = falling(n, daily=0.002 + 0.001 * i, wiggle=0.004)
    return frames


@pytest.mark.unit
def test_universes_without_vix_do_not_download_it_nor_score_it():
    frames = _market()
    frames["^VIX"] = flat(420, start=40.0)           # would be -1 if it were used
    load = _loader(frames)
    res = sc.run_scan("crypto", END, tickers=["UP0", "UP1", "DN0"], loader=load)
    assert "^VIX" not in load.calls[0]
    assert res.regime.vix_used is False and not any(f.key == "vix" for f in res.regime.facts)
    assert any("VIX not used" in line for line in res.regime.describe("en"))
    assert any("VIX wird" in line for line in res.regime.describe("de"))
    assert "VIX" not in res.regime.missing
    assert not any(w.code == "vix_missing" for w in res.warnings)
    assert res.signals[0].horizon_bars == 30

    load = _loader(frames)
    res = sc.run_scan("us", END, tickers=["UP0", "UP1", "DN0"], loader=load)
    assert "^VIX" in load.calls[0]
    assert next(f for f in res.regime.facts if f.key == "vix").points == -1


@pytest.mark.unit
def test_a_running_bar_is_removed_from_members_benchmark_and_vix_before_computing():
    frames = _market()
    frames["^VIX"] = frames["^VIX"].copy()
    frames["^VIX"].iloc[-1, frames["^VIX"].columns.get_loc("Close")] = 40.0   # live, spiking
    day = pd.Timestamp(END)
    assert frames["UP0"].index[-1] == day
    before_close = utc(f"{END}T15:00")                 # 11:00 in New York
    res = sc.run_scan("us", END, tickers=["UP0", "DN0"], loader=_loader(frames), now=before_close)
    assert res.as_of_date == frames["UP0"].index[-2].strftime("%Y-%m-%d") != END
    assert res.running_bars_removed == 2
    assert next(f for f in res.regime.facts if f.key == "vix").value == pytest.approx(15.0)
    assert any(w.code == "running_bar_removed" and w.params["n"] == 2 for w in res.warnings)
    assert "running_bars_removed" in res.to_dict()

    after_close = utc(f"{END}T21:40")                  # 17:40 in New York: VIX settled too
    res = sc.run_scan("us", END, tickers=["UP0", "DN0"], loader=_loader(frames), now=after_close)
    assert res.as_of_date == END and res.running_bars_removed == 0
    assert next(f for f in res.regime.facts if f.key == "vix").value == pytest.approx(40.0)
    assert not any(w.code == "running_bar_removed" for w in res.warnings)


@pytest.mark.unit
def test_a_past_scan_date_warns_about_back_adjusted_prices():
    frames = _market()
    res = sc.run_scan("us", END, tickers=["UP0"], loader=_loader(frames), now=utc("2026-11-13T12:00"))
    note = next(w for w in res.warnings if w.code == "past_date")
    assert "back-adjusted" in str(note) and "rückwirkend" in note.text("de") and END in str(note)
    same_day = sc.run_scan("us", END, tickers=["UP0"], loader=_loader(frames),
                           now=utc(f"{END}T22:00"))
    assert not any(w.code == "past_date" for w in same_day.warnings)


def _sector_frames(have):
    return {s: rising(420, wiggle=0.004) for s in [*have, "SPY", "^VIX"]}


@pytest.mark.unit
def test_a_universe_with_more_than_20_percent_missing_is_refused():
    members = un.UNIVERSES["sectors"][2]                       # 11 members
    with pytest.raises(VendorUnavailableError, match="3 of 11"):
        sc.run_scan("sectors", END, loader=_loader(_sector_frames(members[:8])))
    ok = sc.run_scan("sectors", END, loader=_loader(_sector_frames(members[:9])))   # 2/11 = 18 %
    assert len(ok.skipped) == 2
    # exactly 20 % is still a scan: ten members, two missing
    ten = [f"S{i}" for i in range(10)]
    res = sc.run_scan("us", END, tickers=ten, loader=_loader(_sector_frames(ten[:8])))
    assert len(res.skipped) == 2


@pytest.mark.unit
def test_with_custom_tickers_missing_symbols_are_reported_not_fatal():
    frames = _sector_frames(["AAA"])
    res = sc.run_scan("us", END, tickers=["AAA", "B", "C", "D"], loader=_loader(frames))
    assert [s for s, _ in res.skipped] == ["B", "C", "D"]
    assert all(r.code == "no_data" for _, r in res.skipped)


# --- translatable notices ----------------------------------------------------------------------

@pytest.mark.unit
def test_notices_are_strings_with_a_code_and_german_text():
    n = nt.Notice("short_history", n=120, need=220)
    assert isinstance(n, str) and "only 120 bars" in n and n.code == "short_history"
    assert n.params == {"n": 120, "need": 220}
    assert n.text("de") == "nur 120 Kurse Historie, benötigt werden 220"
    assert n.text("fr") == n.text("en")                     # unknown language falls back
    assert nt.Notice("regime_input_missing", item="breadth").text("de").endswith("Marktbreite.")
    assert n.to_dict()["text_de"] == n.text("de")


@pytest.mark.unit
def test_scan_warnings_and_skips_translate_and_serialise_in_both_languages():
    frames = _market()
    frames["SHORT"] = rising(120)
    res = sc.run_scan("us", END, tickers=["UP0", "SHORT", "GHOST"], loader=_loader(frames))
    reasons = dict(res.skipped)
    assert reasons["SHORT"].code == "short_history" and "benötigt" in reasons["SHORT"].text("de")
    assert reasons["GHOST"].text("de") == "keine Kursdaten von Yahoo"
    skipped_note = next(w for w in res.warnings if w.code == "skipped_summary")
    assert "übersprungen" in skipped_note.text("de") and "skipped" in skipped_note
    data = res.to_dict()
    assert len(data["warnings"]) == len(data["warnings_de"]) == len(data["warning_items"])
    assert {s["symbol"]: s["reason_de"] for s in data["skipped"]}["GHOST"].startswith("keine")
    with pytest.raises(sg.SkipSignal) as exc:
        sg.compute_signal("N", rising(100), None, "neutral")
    assert exc.value.notice.code == "short_history" and "only 100 bars" in str(exc.value)


# --- validation ----------------------------------------------------------------------------------

@pytest.mark.unit
def test_only_rebalance_dates_inside_the_requested_window_count():
    n = 600
    frames = {"SPY": flat(n), "A": rising(n, daily=0.002), "B": falling(n, daily=0.002)}
    res = va.validate_frames("us", END, ["A", "B"], "SPY", frames, months=6, horizon=21)
    window_start = pd.Timestamp(END) - pd.DateOffset(months=6)
    assert pd.Timestamp(res.first_rebalance) >= window_start
    assert res.rebalances >= 5
    # the whole grid (one date before the window under the old counting) is not shortened
    # when the window is a multiple of the horizon
    longer = va.validate_frames("us", END, ["A", "B"], "SPY", frames, months=12, horizon=21)
    assert pd.Timestamp(longer.first_rebalance) >= pd.Timestamp(END) - pd.DateOffset(months=12)
    assert longer.rebalances > res.rebalances


def _fake_sig(symbol, direction):
    return sg.Signal(symbol, direction, 50.0 if direction == "Long" else -50.0, "Medium", {},
                     close=100.0, last_date=END, regime="neutral")


@pytest.mark.unit
def test_pooled_and_per_date_spread_can_have_opposite_signs(monkeypatch):
    n = 120
    spy = flat(n)
    calendar = spy.index
    pos_a, pos_b = 98, 77                      # the two rebalance dates the fake analyse serves
    end_a, end_b = 119, 98

    def jump(at_end, ret):
        closes = np.full(n, 100.0)
        closes[at_end:] = 100.0 * (1 + ret)
        return frame(closes)

    frames = {"SPY": spy, "A_L": jump(end_a, 0.05), "A_S": jump(end_a, 0.0)}
    longs_b, shorts_b = ["B_L"], [f"B_S{i}" for i in range(9)]
    frames["B_L"] = jump(end_b, 0.05)
    for s in shorts_b:
        frames[s] = jump(end_b, 0.06)

    def fake(members, frames_, benchmark, at=None, params=None, use_vix=True):
        if at == calendar[pos_a]:
            sigs = [_fake_sig("A_L", "Long"), _fake_sig("A_S", "Short")]
        elif at == calendar[pos_b]:
            sigs = [_fake_sig(s, "Long") for s in longs_b] + [_fake_sig(s, "Short") for s in shorts_b]
        else:
            sigs = []
        return None, sigs, []
    monkeypatch.setattr(va, "analyse", fake)
    members = [s for s in frames if s != "SPY"]
    res = va.validate_frames("us", END, members, "SPY", frames, months=6, horizon=21)
    assert res.rebalances == 2 and res.spread_dates == 2
    assert res.spread == pytest.approx((0.05 + (0.05 - 0.06)) / 2)            # +2.0 %
    assert res.spread_pooled == pytest.approx(0.05 - (0.0 + 9 * 0.06) / 10)   # -0.4 %
    assert res.spread > 0 > res.spread_pooled
    assert res.to_dict()["spread_pooled"] == res.spread_pooled


@pytest.mark.unit
def test_validate_applies_settlement_and_universe_params(monkeypatch):
    seen = {}
    frames = {"BTC-USD": flat(600, start=100.0)}
    frames["AAA"] = rising(600)
    frames["BBB"] = falling(600)
    # the last bar of every series is "today", before the US/UTC day is over
    now = utc(f"{END}T10:00")

    def loader(symbols, as_of, lookback):
        seen["symbols"] = list(symbols)
        return {s: frames[s] for s in symbols if s in frames}
    real = va.validate_frames
    cut = {}

    def capture(universe, as_of, members, benchmark, frames_, *a, **kw):
        cut.update({s: f.index[-1] for s, f in frames_.items()})
        return real(universe, as_of, members, benchmark, frames_, *a, **kw)
    monkeypatch.setattr(va, "validate_frames", capture)
    res = va.validate("crypto", END, tickers=["AAA", "BBB"], months=6, loader=loader, now=now)
    assert all(last < pd.Timestamp(END) for last in cut.values())     # the running bar is gone
    assert res.horizon == 30                                      # the crypto month
    assert "^VIX" not in seen["symbols"]
    assert pd.Timestamp(res.last_rebalance) < pd.Timestamp(END)
    res21 = va.validate("us", END, tickers=["AAA", "BBB"], months=6, loader=loader, now=now)
    assert res21.horizon == 21 and "^VIX" in seen["symbols"]


# --- CLI / report -------------------------------------------------------------------------------

def _cli_frames(n=520):
    frames = {"SPY": flat(n), "^VIX": flat(n, start=15.0)}
    for i in range(3):
        frames[f"UP{i}"] = rising(n, daily=0.002 + 0.001 * i, wiggle=0.004)
        frames[f"DN{i}"] = falling(n, daily=0.002 + 0.001 * i, wiggle=0.004)
    return frames


@pytest.fixture
def runner(monkeypatch):
    def load(symbols, as_of, lookback):
        return {s: f for s, f in _cli_frames().items() if s in symbols}
    monkeypatch.setattr("tradingagents.scanner.scan.load_bulk_ohlcv", load)
    monkeypatch.setattr("tradingagents.scanner.validation.load_bulk_ohlcv", load)
    return CliRunner()


TICKERS = "UP0,UP1,UP2,DN0,DN1,DN2"


@pytest.mark.unit
def test_cli_prints_both_spreads_with_labels(runner):
    out = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--date", END, "--validate",
                                "--months", "6", "--no-save", "--lang", "en"]).output
    assert "spread (per date)" in out and "spread (pooled)" in out
    out = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--date", END, "--validate",
                                "--months", "6", "--no-save", "--lang", "de"]).output
    assert "(je Termin)" in out and "(gepoolt)" in out


@pytest.mark.unit
def test_cli_says_in_one_line_when_the_validation_had_no_rebalances(runner, monkeypatch):
    short = {s: f.iloc[-200:] for s, f in _cli_frames().items()}     # < 220 bars: nothing scorable
    monkeypatch.setattr("tradingagents.scanner.validation.load_bulk_ohlcv",
                        lambda symbols, as_of, lookback: {s: f for s, f in short.items()
                                                          if s in symbols})
    out = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--date", END, "--validate",
                                "--months", "6", "--no-save"])
    assert out.exit_code == 0
    assert "0 rebalances" in out.output and "nothing to show" in out.output


@pytest.mark.unit
def test_report_has_both_spreads_in_both_languages_and_an_explicit_empty_validation(runner):
    from tradingagents.scanner import report as rp
    members = [s for s in _cli_frames() if s not in ("SPY", "^VIX")]

    def load(symbols, as_of, lookback):
        return {s: f for s, f in _cli_frames().items() if s in symbols}
    result = sc.run_scan("us", END, tickers=members, loader=load)
    val = va.validate("us", END, tickers=members, months=6, loader=load)
    en = rp.render_markdown(result, val, None, "en")
    de = rp.render_markdown(result, val, None, "de")
    assert "per date" in en and "pooled" in en
    assert "je Termin" in de and "gepoolt" in de
    empty = va.validate("us", END, tickers=members, months=6, loader=lambda s, a, lb: {
        k: f.iloc[-200:] for k, f in load(s, a, lb).items()})
    assert empty.rebalances == 0
    assert "0 rebalances" in rp.render_markdown(result, empty, None, "en")
    assert "0 Rebalancing-Termine" in rp.render_markdown(result, empty, None, "de")
    assert "UP0" in de and "Das Scan-Datum" in de      # translated warning in the notes


@pytest.mark.unit
def test_deep_gets_the_date_of_the_newest_bar_used(runner, monkeypatch):
    seen = {}
    monkeypatch.setattr("tradingagents.scanner.deep_dive.check_llm_credentials", lambda c: None)

    def fake_deep(picks, as_of, config):
        seen["as_of"] = as_of
        from tradingagents.scanner.deep_dive import DeepDiveResult
        return DeepDiveResult()
    monkeypatch.setattr("tradingagents.scanner.deep_dive.deep_dive", fake_deep)
    def cut(symbols, as_of, lookback):
        return {s: f.loc[:as_of] for s, f in _cli_frames().items() if s in symbols}
    monkeypatch.setattr("tradingagents.scanner.scan.load_bulk_ohlcv", cut)
    saturday = "2025-06-28"          # the newest bar is Friday the 27th
    res = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--date", saturday, "--deep", "1",
                                "--no-save"])
    assert res.exit_code == 0, res.output
    assert seen["as_of"] == "2025-06-27"
