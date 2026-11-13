"""The `scan` command: it prints the tables, saves, rejects bad input and leaves `analyze` alone."""

import json
import re

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


ENGLISH_LEFTOVERS = (
    "Market regime", "Recommendations", "Watch list", "Direction", "Reference price", "Close",
    "Target", "Conv.", "Hit rate", "Avg ", "Saved", "Running the LLM", "Downloading",
    "High", "Medium", "Long - Short spread", "How solid", "Verdict", "Caveats", "Research tool",
    "not investment advice", "Survivorship bias", "Not proven", "Validated", "Horizon",
)
GERMAN_DECIMAL_POINT = re.compile(r"(?<![\d.,])\d+\.\d+(?![\d.,])")


@pytest.mark.unit
def test_scan_prints_regime_horizons_watch_list_and_never_runs_the_interactive_flow(runner, tmp_path):
    res = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--date", END, "--top", "2",
                                "--output", str(tmp_path), "--lang", "en"])
    assert res.exit_code == 0, res.output
    assert calls == []
    for text in ("Market regime", "Recommendations by investment horizon", "Long term",
                 "Day trade", "Technical picture (not evidenced)", "UP2", "DN2", "How solid is this?",
                 "Saved:", "not investment advice", "do not add up"):
        assert text in res.output, text
    out = tmp_path / "scans" / f"us-custom_{END}"
    assert (out / "scan.md").exists() and (out / "scan.html").exists()
    assert (out / "digest.txt").exists()
    data = json.loads((out / "scan.json").read_text(encoding="utf-8"))
    assert data["as_of_date"] == END and len(data["longs"]) == 2 and data["ideas"] is not None


@pytest.mark.unit
def test_scan_with_validation_prints_the_verdict_and_all_caveats_in_german(runner, tmp_path):
    from tradingagents.scanner.validation import CAVEATS
    res = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--date", END, "--validate",
                                "--months", "6", "--lang", "de", "--output", str(tmp_path)])
    assert res.exit_code == 0, res.output
    out = " ".join(res.output.split())
    for text in ("Validierung", "Marktregime", "Urteil:", "Long-minus-Short-Spread (je Termin)",
                 "Long-minus-Short-Spread (gepoolt)", "Gespeichert:", "Beobachtungsliste",
                 "Empfehlungen nach Anlagehorizont", "Wie belastbar ist das?", "Anlageberatung"):
        assert text in out, text
    for caveat in CAVEATS["de"]:
        assert " ".join(caveat.split()) in out
    for word in ENGLISH_LEFTOVERS:
        assert not re.search(rf"\b{re.escape(word)}\b", out), word
    assert not GERMAN_DECIMAL_POINT.findall(out), GERMAN_DECIMAL_POINT.findall(out)
    assert (tmp_path / "scans" / f"us-custom_{END}" / "scan.md").read_text(
        encoding="utf-8").count("Haftungsausschluss") == 1


@pytest.mark.unit
def test_validation_verdict_and_all_caveats_are_printed_in_english_too(runner):
    from tradingagents.scanner.validation import CAVEATS
    res = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--date", END, "--validate",
                                "--months", "6", "--no-save", "--lang", "en"])
    out = " ".join(res.output.split())
    assert "Verdict: Over the last 6 months" in out
    for caveat in CAVEATS["en"]:
        assert " ".join(caveat.split()) in out


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
    res = runner.invoke(m.app, ["scan", "--universe", "us,mars", "--no-save"])
    assert res.exit_code == 1 and "unknown universe 'mars'" in res.output


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
    assert "Marktregime" in res.output and "Beobachtungsliste" in res.output
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


# --- several universes, --horizon, --digest -------------------------------------------------------

def _fake_scan(failing=()):
    """run_scan stand-in: a synthetic result per universe name, VendorUnavailableError for some."""
    from tests.scanner_idea_helpers import empty_result, evidenced_result
    from tradingagents.dataflows.errors import VendorUnavailableError

    def run(universe, as_of, tickers=None, top=10, **kw):
        if universe in failing:
            raise VendorUnavailableError(f"{universe} is down")
        result = (evidenced_result if universe == "us" else empty_result)(universe)
        result.universe, result.custom = universe, False
        result.description = f"{universe} market"
        return result
    return run


@pytest.fixture
def fake(monkeypatch):
    def install(failing=()):
        monkeypatch.setattr("tradingagents.scanner.scan.run_scan", _fake_scan(failing))
    return install


@pytest.mark.unit
def test_several_universes_are_scanned_in_turn(runner, fake, tmp_path):
    fake()
    res = runner.invoke(m.app, ["scan", "--universe", "us,dax,crypto", "--date", END, "--lang", "de",
                                "--output", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert res.output.index("(us)") < res.output.index("(dax)") < res.output.index("(crypto)")
    for name in ("us", "dax", "crypto"):
        assert (tmp_path / "scans" / f"{name}_{END}" / "digest.txt").exists()


@pytest.mark.unit
def test_one_failing_market_does_not_stop_the_others(runner, fake):
    fake(failing=("dax",))
    res = runner.invoke(m.app, ["scan", "--universe", "us,dax,crypto", "--date", END, "--no-save"])
    assert res.exit_code == 0
    assert "dax failed: Could not get price data: dax is down" in res.output
    assert "(us)" in res.output and "(crypto)" in res.output
    de = runner.invoke(m.app, ["scan", "--universe", "us,dax", "--date", END, "--no-save",
                               "--lang", "de"])
    assert "dax fehlgeschlagen: Kursdaten konnten nicht geladen werden" in de.output


@pytest.mark.unit
def test_exit_1_only_when_every_market_failed(runner, fake):
    fake(failing=("us", "dax"))
    res = runner.invoke(m.app, ["scan", "--universe", "us,dax", "--date", END, "--no-save"])
    assert res.exit_code == 1 and "No market could be scanned." in res.output
    assert "Traceback" not in res.output


@pytest.mark.unit
def test_horizon_filters_what_is_shown_and_rejects_unknown_names(runner, fake):
    fake()
    res = runner.invoke(m.app, ["scan", "--universe", "us", "--date", END, "--no-save",
                                "--horizon", "day,long"])
    assert res.exit_code == 0
    assert "Long term" in res.output and "Day trade" in res.output
    assert "Medium term" not in res.output and "Short term (swing)" not in res.output
    bad = runner.invoke(m.app, ["scan", "--universe", "us", "--no-save", "--horizon", "week"])
    assert bad.exit_code == 1 and "unknown horizon 'week'" in bad.output
    assert "Traceback" not in bad.output


@pytest.mark.unit
def test_tickers_with_several_universes_is_rejected(runner):
    res = runner.invoke(m.app, ["scan", "--universe", "us,dax", "--tickers", TICKERS, "--no-save"])
    assert res.exit_code == 1 and "--tickers works with one universe only" in res.output


@pytest.mark.unit
def test_digest_prints_only_the_plain_text_digest(runner, fake):
    fake()
    res = runner.invoke(m.app, ["scan", "--universe", "us,dax", "--date", END, "--digest",
                                "--no-save", "--lang", "de"])
    assert res.exit_code == 0, res.output
    headline, blank, *rest = res.output.split("\n")
    assert headline.startswith("Belegt:") and len(headline) <= 180 and blank == ""
    body = "\n".join(rest)
    assert "LONG UP2 · Einstieg nächste Eröffnung" in body and "Keine Anlageberatung." in body
    for noise in ("┃", "┏", "[", "Lade Kurse", "Gespeichert", "Datenstand", "Marktregime:"):
        assert noise not in res.output, noise
    assert "|" not in res.output and "#" not in res.output


@pytest.mark.unit
def test_digest_still_saves_when_asked_and_names_markets_that_failed(runner, fake, tmp_path):
    fake(failing=("dax",))
    res = runner.invoke(m.app, ["scan", "--universe", "us,dax", "--date", END, "--digest",
                                "--save", "--output", str(tmp_path), "--lang", "de"])
    assert res.exit_code == 0
    assert (tmp_path / "scans" / f"us_{END}" / "digest.txt").exists()
    assert (tmp_path / "scans" / f"us_{END}" / "scan.md").exists()
    assert "Nicht geladen: dax." in res.output and "Gespeichert" not in res.output
    assert not (tmp_path / "scans" / f"dax_{END}").exists()


@pytest.mark.unit
def test_digest_exit_code_is_1_when_all_markets_fail(runner, fake):
    fake(failing=("us",))
    res = runner.invoke(m.app, ["scan", "--universe", "us", "--digest", "--no-save"])
    assert res.exit_code == 1


@pytest.mark.unit
def test_digest_runs_end_to_end_on_synthetic_bars(runner):
    res = runner.invoke(m.app, ["scan", "--tickers", TICKERS, "--date", END, "--digest",
                                "--no-save", "--lang", "en"])
    assert res.exit_code == 0, res.output
    assert res.output.startswith("No evidenced recommendation") or res.output.startswith("Evidenced")
    assert "Not investment advice." in res.output
