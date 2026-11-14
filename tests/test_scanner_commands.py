"""The question commands on a local typer app (they are mounted into the main CLI later)."""

import json

import numpy as np
import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from cli import scanner_commands
from tests.scanner_helpers import END
from tests.test_scanner_assess import fake_scan, idea, signal
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.vendors.yahoo import options as yahoo_options
from tradingagents.scanner import assess as A, events


def _app():
    app = typer.Typer()
    scanner_commands.register(app)

    @app.command()
    def other():          # a Typer app with one command collapses to it; keep two
        """placeholder"""
    return app


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def app():
    return _app()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(A, "run_scan", fake_scan(
        [signal("NVDA", -18.0), signal("SAP.DE", 62.0)],
        [idea("SAP.DE", "long", label="validated", rule="T1")]))
    monkeypatch.setattr(A, "performance_snapshot", lambda syms, d, loader=None, now=None: [])


# --------------------------------------------------------------------------- assess


@pytest.mark.unit
def test_register_adds_the_four_commands(app):
    names = {c.name or c.callback.__name__ for c in app.registered_commands}
    assert {"assess", "ask", "event-study", "implied-move"} <= names


@pytest.mark.unit
def test_assess_prints_markdown_in_german(runner, app):
    res = runner.invoke(app, ["assess", "NVDA", "SAP.DE", "--date", END, "--lang", "de"])
    assert res.exit_code == 0, res.output
    for needed in ("## Einschätzung", "Das Wichtigste", "Nvidia (NVDA)", "SAP (SAP.DE)",
                   "Keine Anlageberatung"):
        assert needed in res.output, needed


@pytest.mark.unit
def test_assess_accepts_commas_names_and_english(runner, app):
    res = runner.invoke(app, ["assess", "nvidia,SAP.DE", "--date", END, "--lang", "en"])
    assert res.exit_code == 0, res.output
    assert "Assessment" in res.output and "Nvidia (NVDA)" in res.output


@pytest.mark.unit
def test_assess_json_prints_only_json(runner, app):
    res = runner.invoke(app, ["assess", "NVDA", "--date", END, "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert data["requested"] == ["NVDA"] and data["tickers"][0]["symbol"] == "NVDA"


@pytest.mark.unit
@pytest.mark.parametrize("args,message", [
    (["assess", "NVDA", "--lang", "fr"], "--lang must be en or de"),
    (["assess", "NVDA", "--date", "2999-01-01"], "future"),
    (["assess", "NVDA", "--date", "not-a-date"], "date"),
    (["assess", ",", ","], "no ticker given"),
])
def test_assess_bad_input_is_a_red_message_and_exit_1(runner, app, args, message):
    res = runner.invoke(app, args)
    assert res.exit_code == 1 and message in res.output
    assert "Traceback" not in res.output and isinstance(res.exception, SystemExit)


@pytest.mark.unit
def test_assess_vendor_outage_is_not_a_traceback(runner, app, monkeypatch):
    def down(*a, **k):
        raise VendorUnavailableError("Yahoo throttled")
    monkeypatch.setattr(A, "run_scan", down)
    res = runner.invoke(app, ["assess", "NVDA", "--date", END])
    assert res.exit_code == 1 and "Could not get price data: Yahoo throttled" in res.output
    assert isinstance(res.exception, SystemExit)


@pytest.mark.unit
def test_assess_options_flag_reaches_the_assessment(runner, app, monkeypatch):
    seen = {}
    real = A.assess

    def spy(tickers, **kw):
        seen.update(kw)
        return real(tickers, **kw)
    monkeypatch.setattr(A, "assess", spy)
    monkeypatch.setattr(yahoo_options, "load_implied_moves", lambda s, max_expiries=4: [])
    res = runner.invoke(app, ["assess", "NVDA", "--date", END, "--options", "--lang", "de"])
    assert res.exit_code == 0, res.output
    assert seen["with_options"] is True and "Keine börsengehandelten Optionen" in res.output


# --------------------------------------------------------------------------- ask


@pytest.mark.unit
def test_ask_answers_a_question_about_instruments(runner, app):
    res = runner.invoke(app, ["ask", "Soll ich Nvidia oder SAP auf 3 Monate kaufen?"])
    assert res.exit_code == 0, res.output
    assert "Kurzantwort" in res.output and "Mittelfristig" in res.output
    assert "Keine Anlageberatung" in res.output


@pytest.mark.unit
def test_ask_event_question_without_instrument_is_the_honest_fallback(runner, app):
    res = runner.invoke(app, ["ask", "Was bedeutet der GTA-Release für den Markt?", "--lang", "de"])
    assert res.exit_code == 0, res.output
    assert "So kann ich das nicht beantworten" in res.output and "Stell diese Frage direkt im Claude-Chat" in res.output


@pytest.mark.unit
def test_ask_json(runner, app):
    res = runner.invoke(app, ["ask", "NVDA?", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert data["parsed"]["tickers"] == ["NVDA"] and "Antwort" in data["text"] + "Antwort"


@pytest.mark.unit
def test_ask_has_no_llm_option(runner, app):
    assert runner.invoke(app, ["ask", "NVDA?", "--llm"]).exit_code == 2


@pytest.mark.unit
def test_ask_without_question_and_without_terminal_is_an_error(runner, app, monkeypatch):
    monkeypatch.setattr(scanner_commands, "_is_terminal", lambda: False)
    res = runner.invoke(app, ["ask"])
    assert res.exit_code == 1 and "No question given" in res.output
    assert isinstance(res.exception, SystemExit)


@pytest.mark.unit
def test_ask_loop_on_a_terminal_ends_on_empty_line_and_survives_an_error(runner, app, monkeypatch):
    monkeypatch.setattr(scanner_commands, "_is_terminal", lambda: True)
    res = runner.invoke(app, ["ask", "--lang", "de"], input="Nvidia kaufen?\n\n")
    assert res.exit_code == 0, res.output
    assert res.output.count("Kurzantwort") == 1

    def down(*a, **k):
        raise VendorUnavailableError("offline")
    monkeypatch.setattr(A, "run_scan", down)
    res = runner.invoke(app, ["ask", "--lang", "de"], input="Nvidia kaufen?\nexit\n")
    assert res.exit_code == 0 and "Kursdaten konnten nicht geladen werden" in res.output


@pytest.mark.unit
def test_ask_bad_lang(runner, app):
    res = runner.invoke(app, ["ask", "NVDA?", "--lang", "xx"])
    assert res.exit_code == 1 and "--lang must be en or de" in res.output


# --------------------------------------------------------------------------- event-study


def _bars(start, n, slope):
    idx = pd.bdate_range(start, periods=n)
    c = pd.Series(100 + slope * np.arange(n), index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1000.0})


@pytest.fixture
def bars(monkeypatch):
    frames = {"TTWO": _bars("2018-01-01", 2000, 0.05), "^GSPC": _bars("2018-01-01", 2000, 0.02)}
    monkeypatch.setattr(events, "_default_loader",
                        lambda symbols, as_of, lookback: {s: frames[s] for s in symbols
                                                          if s in frames})


@pytest.mark.unit
def test_event_study_prints_the_table(runner, app, bars):
    res = runner.invoke(app, ["event-study", "TTWO", "--dates",
                              "2018-10-26:Red Dead Redemption 2,2019-06-03:E3", "--lang", "de"])
    assert res.exit_code == 0, res.output
    assert "Red Dead Redemption 2" in res.output and "E3" in res.output


@pytest.mark.unit
def test_event_study_resolves_names_and_prints_json(runner, app, bars):
    res = runner.invoke(app, ["event-study", "take-two", "--dates", "2018-10-26", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert data["symbol"] == "TTWO" and data["events"][0]["label"] == "2018-10-26"
    assert data["benchmark"] == "^GSPC"


@pytest.mark.unit
@pytest.mark.parametrize("dates,message", [
    ("garbage", "date"), (",", "no event given"), ("2018-13-45:x", "date"),
])
def test_event_study_bad_dates(runner, app, bars, dates, message):
    res = runner.invoke(app, ["event-study", "TTWO", "--dates", dates])
    assert res.exit_code == 1 and message in res.output
    assert isinstance(res.exception, SystemExit)


@pytest.mark.unit
def test_event_study_unknown_symbol_is_a_red_message(runner, app, bars):
    res = runner.invoke(app, ["event-study", "NOPE", "--dates", "2018-10-26"])
    assert res.exit_code == 1 and "No price data found for NOPE" in res.output
    assert isinstance(res.exception, SystemExit)


@pytest.mark.unit
def test_event_study_requires_dates(runner, app, bars):
    assert runner.invoke(app, ["event-study", "TTWO"]).exit_code == 2


# --------------------------------------------------------------------------- implied-move


def _move(symbol="TTWO"):
    return yahoo_options.ImpliedMove(
        symbol=symbol, expiry="2026-12-18", days_to_expiry=35, spot=250.0, strike=250.0,
        call_price=10.0, put_price=9.0, straddle=19.0, move_pct=0.076, price_basis="mid",
        stale=False)


@pytest.mark.unit
def test_implied_move_prints_and_passes_the_expiry_count(runner, app, monkeypatch):
    seen = {}

    def fake(symbol, max_expiries=4):
        seen.update(symbol=symbol, n=max_expiries)
        return [_move()]
    monkeypatch.setattr(yahoo_options, "load_implied_moves", fake)
    res = runner.invoke(app, ["implied-move", "take-two", "--expiries", "2", "--lang", "de"])
    assert res.exit_code == 0, res.output
    assert seen == {"symbol": "TTWO", "n": 2} and "7,6 %" in res.output


@pytest.mark.unit
def test_implied_move_json_and_empty(runner, app, monkeypatch):
    monkeypatch.setattr(yahoo_options, "load_implied_moves", lambda s, max_expiries=4: [_move()])
    res = runner.invoke(app, ["implied-move", "TTWO", "--json"])
    assert json.loads(res.stdout)[0]["move_pct"] == 0.076
    monkeypatch.setattr(yahoo_options, "load_implied_moves", lambda s, max_expiries=4: [])
    res = runner.invoke(app, ["implied-move", "SAP.DE", "--json"])
    assert res.exit_code == 0 and json.loads(res.stdout) == []


@pytest.mark.unit
def test_implied_move_errors(runner, app, monkeypatch):
    assert runner.invoke(app, ["implied-move", "TTWO", "--expiries", "0"]).exit_code == 1

    def down(symbol, max_expiries=4):
        raise VendorUnavailableError("down")
    monkeypatch.setattr(yahoo_options, "load_implied_moves", down)
    res = runner.invoke(app, ["implied-move", "TTWO"])
    assert res.exit_code == 1 and "Could not get option data: down" in res.output
    assert isinstance(res.exception, SystemExit)
