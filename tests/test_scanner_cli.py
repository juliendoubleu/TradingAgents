"""The `scan` command: it prints the tables, saves, rejects bad input and leaves `analyze` alone."""

import json

import pytest
from typer.testing import CliRunner

import cli.main as m
from tests.scanner_helpers import END, falling, flat, rising

calls: list = []


def _frames(n=520):
    frames = {"SPY": flat(n), "^VIX": flat(n, start=15.0)}
    for i in range(3):
        frames[f"UP{i}"] = rising(n, daily=0.002 + 0.001 * i, wiggle=0.004)
        frames[f"DN{i}"] = falling(n, daily=0.002 + 0.001 * i, wiggle=0.004)
    return frames


def _loader(symbols, as_of, lookback):
    return {s: f for s, f in _frames().items() if s in symbols}


@pytest.fixture
def runner(monkeypatch):
    calls.clear()
    monkeypatch.setattr(m, "run_analysis", lambda **kw: calls.append(kw))
    monkeypatch.setattr("tradingagents.scanner.scan.load_bulk_ohlcv", _loader)
    monkeypatch.setattr("tradingagents.scanner.validation.load_bulk_ohlcv", _loader)
    return CliRunner()


TICKERS = "UP0,UP1,UP2,DN0,DN1,DN2"


@pytest.mark.unit
def test_scan_prints_regime_and_both_tables_and_never_runs_the_interactive_flow(runner, tmp_path):
    res = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--date", END, "--top", "2",
                                "--output", str(tmp_path), "--lang", "en"])
    assert res.exit_code == 0, res.output
    assert calls == []
    for text in ("Market regime", "Top Long", "Top Short", "UP2", "DN2", "Saved:", "financial advice"):
        assert text in res.output, text
    out = tmp_path / "scans" / f"us-custom_{END}"
    assert (out / "scan.md").exists() and (out / "scan.html").exists()
    data = json.loads((out / "scan.json").read_text(encoding="utf-8"))
    assert data["as_of_date"] == END and len(data["longs"]) == 2


@pytest.mark.unit
def test_scan_with_validation_and_german_output(runner, tmp_path):
    res = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--date", END, "--validate",
                                "--months", "6", "--lang", "de", "--output", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "Validierung" in res.output and "Marktregime" in res.output
    assert "Long - Short spread" in res.output
    assert (tmp_path / "scans" / f"us-custom_{END}" / "scan.md").read_text(
        encoding="utf-8").count("Haftungsausschluss") == 1


@pytest.mark.unit
def test_no_save_writes_nothing(runner, tmp_path):
    res = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--date", END, "--no-save",
                                "--output", str(tmp_path)])
    assert res.exit_code == 0 and "Saved:" not in res.output
    assert not (tmp_path / "scans").exists()


@pytest.mark.unit
@pytest.mark.parametrize("args,needle", [
    (["--universe", "mars"], "unknown universe"),
    (["--date", "2999-01-01"], "future"),
    (["--date", "June"], "YYYY-MM-DD"),
    (["--lang", "fr"], "--lang"),
    (["--top", "0"], "--top"),
])
def test_bad_input_is_a_red_message_and_exit_1(runner, args, needle):
    res = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--no-save", *args])
    assert res.exit_code == 1
    assert needle in res.output and "Traceback" not in res.output
    assert calls == []


@pytest.mark.unit
def test_unknown_universe_lists_the_valid_names(runner):
    res = runner.invoke(m.app, ["scan", "--universe", "mars", "--no-save"])
    assert res.exit_code == 1 and "us, sectors, global, dax, crypto, all" in res.output


@pytest.mark.unit
def test_deep_without_an_llm_key_fails_cleanly(runner, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setitem(m.DEFAULT_CONFIG, "llm_provider", "openai")
    monkeypatch.setitem(m.DEFAULT_CONFIG, "deep_think_provider", None)
    monkeypatch.setitem(m.DEFAULT_CONFIG, "quick_think_provider", None)
    res = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--date", END, "--deep", "2",
                                "--no-save"])
    assert res.exit_code == 1
    assert "OPENAI_API_KEY" in res.output and "Traceback" not in res.output


@pytest.mark.unit
def test_vendor_outage_is_reported_not_raised(runner, monkeypatch):
    from tradingagents.dataflows.errors import VendorUnavailableError

    def down(*a, **k):
        raise VendorUnavailableError("Yahoo Finance is unreachable")
    monkeypatch.setattr("tradingagents.scanner.scan.load_bulk_ohlcv", down)
    res = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--date", END, "--no-save"])
    assert res.exit_code == 1 and "Could not get price data" in res.output


@pytest.mark.unit
def test_bare_invocation_still_runs_the_analysis(runner):
    assert runner.invoke(m.app, []).exit_code == 0
    assert len(calls) == 1


@pytest.mark.unit
def test_language_defaults_to_the_configured_output_language(runner, monkeypatch):
    monkeypatch.setitem(m.DEFAULT_CONFIG, "output_language", "German")
    res = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--date", END, "--no-save"])
    assert "Marktregime" in res.output
    monkeypatch.setitem(m.DEFAULT_CONFIG, "output_language", "English")
    res = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--date", END, "--no-save"])
    assert "Market regime" in res.output


@pytest.mark.unit
def test_deep_runs_the_agents_on_the_strongest_ideas(runner, monkeypatch, tmp_path):
    from tradingagents.scanner.deep_dive import DeepDiveResult
    seen = {}

    def fake_deep(signals, as_of, config):
        seen["symbols"] = [s.symbol for s in signals]
        return DeepDiveResult(ratings={signals[0].symbol: "Buy"}, errors={signals[1].symbol: "boom"})
    monkeypatch.setattr("tradingagents.scanner.deep_dive.check_llm_credentials", lambda c: None)
    monkeypatch.setattr("tradingagents.scanner.deep_dive.deep_dive", fake_deep)
    res = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--date", END, "--deep", "2",
                                "--output", str(tmp_path), "--lang", "en"])
    assert res.exit_code == 0, res.output
    assert len(seen["symbols"]) == 2 and ": Buy" in res.output and "boom" in res.output
    saved = json.loads((tmp_path / "scans" / f"us-custom_{END}" / "scan.json").read_text("utf-8"))
    assert saved["deep_dive"]["errors"]
