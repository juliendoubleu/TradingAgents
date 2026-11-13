"""Report rendering (en/de, escaping, disclaimer, saving) and the optional deep dive."""

import json

import pytest

from tests.scanner_helpers import END, falling, flat, rising
from tradingagents.scanner import deep_dive as dd, report as rp, scan as sc, validation as va


def _frames(n=520):
    frames = {"SPY": flat(n), "^VIX": flat(n, start=15.0)}
    for i in range(3):
        frames[f"UP{i}"] = rising(n, daily=0.002 + 0.001 * i, wiggle=0.004)
        frames[f"DN{i}"] = falling(n, daily=0.002 + 0.001 * i, wiggle=0.004)
    return frames


def _loader(symbols, as_of, lookback):
    return {s: f for s, f in _frames().items() if s in symbols}


@pytest.fixture
def result():
    members = [s for s in _frames() if s not in ("SPY", "^VIX")]
    return sc.run_scan("us", END, tickers=members, top=3, loader=_loader)


@pytest.fixture
def validation():
    members = [s for s in _frames() if s not in ("SPY", "^VIX")]
    return va.validate("us", END, tickers=members, months=6, loader=_loader)


@pytest.mark.unit
def test_markdown_has_every_section_in_english(result, validation):
    deep = dd.DeepDiveResult(ratings={"UP2": "Buy"}, errors={"DN2": "RuntimeError: no key"})
    md = rp.render_markdown(result, validation, deep, "en")
    for heading in ("# Market scan: US", "## Market regime", "## Top Long ideas",
                    "## Top Short ideas", "## Walk-forward validation", "Caveats:",
                    "## Deep-dive ratings", "## Method", "## Disclaimer"):
        assert heading in md, heading
    assert "not financial advice" in md and "unlimited loss risk" in md
    assert "UP2" in md and "Buy" in md and "failed: RuntimeError" in md
    assert "| # | Ticker | Score | Conviction | Close | Entry | Stop | Target | Size % | Why |" in md
    assert "Survivorship" in md


@pytest.mark.unit
def test_german_report_translates_labels_and_reasons(result, validation):
    md = rp.render_markdown(result, validation, None, "de")
    for text in ("Marktregime", "Top Long-Ideen", "Top Short-Ideen", "Walk-forward-Validierung",
                 "Haftungsausschluss", "keine Anlageberatung", "Überzeugung", "Einstieg",
                 "über dem 200-Tage-Schnitt", "Methode"):
        assert text in md, text
    assert "Long" in md and "Short" in md
    assert "Market regime" not in md and "Top Long ideas" not in md
    assert "above its 200-day average" not in md


@pytest.mark.unit
def test_report_without_validation_or_deep_dive_omits_those_sections(result):
    md = rp.render_markdown(result)
    assert "Walk-forward validation" not in md and "Deep-dive" not in md


@pytest.mark.unit
def test_html_is_self_contained_escaped_and_themed(result, validation):
    result.skipped.append(("<script>alert(1)</script>", "no price data & <b>bad</b>"))
    result.warnings.append("<img src=x onerror=alert(1)>")
    html = rp.render_html(result, validation, None, "en")
    assert html.startswith("<!doctype html>")
    assert "<script" not in html.lower().replace("&lt;script", "")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<img" not in html and "&lt;img src=x" in html
    assert "Content-Security-Policy" in html and "default-src 'none'" in html
    assert "prefers-color-scheme: dark" in html
    assert "http://" not in html and "https://" not in html
    assert 'class="long"' in html and 'class="short"' in html
    assert "not financial advice" in html
    de = rp.render_html(result, validation, None, "de")
    assert '<html lang="de">' in de and "Haftungsausschluss" in de


@pytest.mark.unit
def test_empty_idea_lists_say_so(result):
    result.longs, result.shorts = [], []
    md = rp.render_markdown(result, lang="en")
    assert "No Long signal" in md and "No Short signal" in md


@pytest.mark.unit
def test_save_scan_writes_three_files_under_the_results_dir(result, validation, tmp_path):
    paths = rp.save_scan(result, validation, None, "en", tmp_path)
    folder = tmp_path / "scans" / f"us-custom_{END}"
    assert {p.parent for p in paths.values()} == {folder}
    assert sorted(p.name for p in paths.values()) == ["scan.html", "scan.json", "scan.md"]
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["regime"]["label"] == result.regime.label
    assert data["validation"]["rebalances"] == validation.rebalances
    assert len(data["longs"]) == len(result.longs)
    assert paths["md"].read_text(encoding="utf-8") == rp.render_markdown(result, validation, None, "en")


# --- deep dive ---

class _Graph:
    def __init__(self, analysts):
        self.analysts, self.calls = analysts, []

    def propagate(self, ticker, date, asset_type="stock"):
        self.calls.append((ticker, date, asset_type))
        if ticker == "BAD":
            raise RuntimeError("vendor down")
        return {"final_rating": "Overweight"}, "Overweight"


def _sig(symbol):
    from tradingagents.scanner.signals import Signal
    return Signal(symbol, "Long", 60, "Medium", {}, 1.0, END, "neutral")


@pytest.mark.unit
def test_deep_dive_collects_ratings_and_errors_per_ticker():
    graphs = []

    def factory(analysts, config):
        graphs.append(_Graph(analysts))
        return graphs[-1]

    out = dd.deep_dive([_sig("AAPL"), _sig("BAD"), _sig("BTC-USD")], END, {}, factory)
    assert out.ratings == {"AAPL": "Overweight", "BTC-USD": "Overweight"}
    assert "vendor down" in out.errors["BAD"]
    # stocks and crypto get their own graph; a coin has no fundamentals analyst
    assert [g.analysts for g in graphs] == [dd.STOCK_ANALYSTS, dd.CRYPTO_ANALYSTS]
    assert graphs[0].calls == [("AAPL", END, "stock"), ("BAD", END, "stock")]
    assert graphs[1].calls == [("BTC-USD", END, "crypto")]


@pytest.mark.unit
def test_deep_dive_records_a_graph_that_cannot_be_built():
    def factory(analysts, config):
        raise ValueError("no model")
    out = dd.deep_dive([_sig("AAPL")], END, {}, factory)
    assert out.ratings == {} and "no model" in out.errors["AAPL"]


@pytest.mark.unit
def test_missing_llm_key_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        dd.check_llm_credentials({"llm_provider": "openai"})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    dd.check_llm_credentials({"llm_provider": "openai"})
    dd.check_llm_credentials({"llm_provider": "ollama"})          # a local runtime needs no key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        dd.check_llm_credentials({"llm_provider": "openai", "deep_think_provider": "anthropic"})
