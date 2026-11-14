"""Report rendering: horizons first, honest evidence labels, German without English leftovers,
escaping, the computed validation verdict, saving; plus the optional deep dive."""

import json
import re

import pytest

from tests.scanner_helpers import END
from tests.scanner_idea_helpers import (
    base_result,
    day_note,
    empty_result,
    evidenced_result,
    frames as _frames,
    idea,
    loader as _loader,
    result_with,
    rules_note,
    validation as make_validation,
)
from tradingagents.scanner import deep_dive as dd, report as rp, scan as sc, validation as va

ENGLISH_LEFTOVERS = (
    "Market scan", "Market regime", "Universe", "Data as of", "Requested date",
    "Generated", "Recommendations", "Direction", "Reference price", "Entry", "Holding period",
    "Evidence", "Watch list", "technical picture", "Technically", "Last close", "Hit rate",
    "Avg ", "Close", "Target", "Conviction", "High", "Medium", "Saved", "scored", "Disclaimer",
    "Validation", "Caveats", "Method", "Size %", "Why", "Long - Short spread", "not investment advice",
    "Skipped", "Notes", "Count", "Rating", "How solid",
)
GERMAN_DECIMAL_POINT = re.compile(r"(?<![\d.,])\d+\.\d+(?![\d.,])")


@pytest.fixture
def result():
    return base_result()


@pytest.fixture
def validation():
    members = [s for s in _frames() if s not in ("SPY", "^VIX")]
    return va.validate("us", END, tickers=members, months=6, loader=_loader)


def _tables(blocks):
    return [b for b in blocks if b[0] == "table"]


def _rec_tables(blocks, lang="en"):
    """Tables of recommendations: the ones with an entry-rule column."""
    entry = rp.tr("entry", lang)
    return [t for t in _tables(blocks) if entry in t[1]]


@pytest.mark.unit
def test_markdown_sections_come_in_the_planned_order_in_english(result, validation):
    deep = dd.DeepDiveResult(ratings={"UP2": "Buy"}, errors={"DN2": "RuntimeError: no key"})
    md = rp.render_markdown(result, validation, deep, "en")
    order = ["# Market scan: US", "## Market regime", "## Recommendations by investment horizon",
             "## Technical picture (not evidenced) – watch list, not a recommendation", "## How solid is this?",
             "## Validation of the technical picture", "## Deep-dive ratings", "## Method",
             "## Disclaimer"]
    positions = [md.index(h) for h in order]
    assert positions == sorted(positions)
    assert "UP2" in md and "Buy" in md and "failed: RuntimeError" in md
    assert "Survivorship" in md and "not investment advice" in md and "without limit" in md
    # the composite never shows conviction, target or a size: it is a watch list
    watch = md[md.index("## Technical picture"):md.index("## How solid")]
    for word in ("Conviction", "Target", "Size"):
        assert word not in watch


@pytest.mark.unit
def test_every_horizon_has_a_section_and_no_empty_table(result):
    result = empty_result()
    blocks = rp.build_blocks(result, lang="en")
    for block in _tables(blocks):
        assert block[2], "an empty table was rendered"
    md = rp.render_markdown(result, lang="en")
    for heading in ("### Long term", "### Medium term", "### Short term (swing)", "### Day trade"):
        assert heading in md
    assert rp.tr("no_rec_any", "en") in md
    # the why-not notes carry the numbers
    assert "15 rules" in md or "24 rules" in md
    assert "none with a proven edge out of sample" in md
    assert rp.render_markdown(result, lang="de").count("Keine belegte Empfehlung. Das technische Bild unten") == 1


@pytest.mark.unit
def test_a_watch_only_idea_never_appears_in_a_recommendation_table():
    ideas = [
        idea("REC1", "Long", "long", "validated"),
        idea("EXP1", "Long", "medium", "experimental"),
        idea("WATCHBETA", "Long", "swing", "beta"),
        idea("AVOIDME", "Short", "medium", "relative", stance="avoid"),
        idea("UNPROVEN", "Short", "medium", "unproven"),
    ]
    result = result_with(ideas, [day_note()])
    for lang in ("en", "de"):
        blocks = rp.build_blocks(result, lang=lang)
        rec_symbols = [row[1] for t in _rec_tables(blocks, lang) for row in t[2]]
        assert sorted(rec_symbols) == ["EXP1", "REC1"]
        for table in _rec_tables(blocks, lang):
            for row in table[2]:
                cell = row[-1]
                idea_obj = next(i for i in ideas if i.symbol == row[1])
                assert cell.text.startswith(idea_obj.evidence.text(lang))      # verbatim
    md = rp.render_markdown(result, lang="en")
    watch = md[md.index("## Technical picture"):md.index("## How solid")]
    assert "WATCHBETA" in watch and "AVOIDME" in watch and "avoid / underweight" in watch
    assert "UNPROVEN" not in md.split("## Technical picture")[0]


@pytest.mark.unit
def test_a_recommended_long_that_is_technically_weak_says_so():
    result = result_with([idea("DN0", "Long", "long", "validated")], [day_note()])
    assert any(s.symbol == "DN0" and s.direction == "Short" for s in result.signals)
    md = rp.render_markdown(result, lang="de")
    row = next(line for line in md.splitlines() if "| Long | DN0 |" in line)
    assert "Technisch aktuell schwach" in row


@pytest.mark.unit
def test_german_report_has_no_english_leftovers_and_decimal_commas(validation):
    result = evidenced_result()
    result.warnings.append(sc.Notice("past_date", date="2026-01-02"))
    result.skipped.append(("XYZ", sc.Notice("stale", last="2026-01-01")))
    for text in (rp.render_markdown(result, validation, None, "de"),
                 rp.render_markdown(empty_result(), None, None, "de")):
        for word in ENGLISH_LEFTOVERS:
            assert not re.search(rf"\b{re.escape(word)}\b", text), word
        assert not GERMAN_DECIMAL_POINT.findall(text), GERMAN_DECIMAL_POINT.findall(text)
        assert "Eigene Liste (6 Ticker)" in text and "6 bewertet" in text
    de = rp.render_markdown(result, validation, None, "de")
    for needle in ("Marktregime", "Empfehlungen nach Anlagehorizont",
                   "Technisches Bild (nicht belegt) – Beobachtungsliste, keine Empfehlung",
                   "Wie belastbar ist das?", "Validierung des technischen Bildes", "Urteil:",
                   "Haftungsausschluss", "Das Scan-Datum", "letzter Kurs vom", "5,97 %",
                   "über dem 200-Tage-Schnitt", "Belegt", "Nicht belegt"):
        assert needle in de, needle
    assert "Market regime" not in de and "above its 200-day average" not in de


@pytest.mark.unit
def test_german_prices_use_decimal_comma_and_thousands_dot():
    result = result_with([idea("BTC-USD", "Long", "long", "validated", price=69271.61,
                               stop=60000.0)], [])
    md = rp.render_markdown(result, lang="de")
    assert "69.271,61" in md and "60.000,00" in md
    assert "69,271.61" in rp.render_markdown(result, lang="en")


@pytest.mark.unit
def test_the_sizes_do_not_add_up_line_is_in_both_languages_and_the_disclaimer():
    result = evidenced_result()
    en, de = rp.render_markdown(result, lang="en"), rp.render_markdown(result, lang="de")
    assert "do NOT add up" in en and "cap the total exposure yourself" in en
    assert "addieren sich NICHT" in de and "Gesamtengagement" in de
    assert "can lose more than the stake" in en and "mehr als den Einsatz" in de
    assert "not investment advice" in en and "keine Anlageberatung" in de


@pytest.mark.unit
def test_how_solid_table_and_computed_paragraph():
    result = evidenced_result()
    en = rp.render_markdown(result, lang="en")
    section = en[en.index("## How solid is this?"):en.index("## Method")]
    assert "| Long term | Long | Momentum 12-1 (T1) | Validated |" in section
    assert "44" in section and "2022–2026" in section and "2.2" in section
    assert "Passed the out-of-sample test: Long term Long (Momentum 12-1)" in section
    assert "Short ideas are not evidenced anywhere" in section
    assert "Tested: 9 swing and 15 day-trade rules (24); the only one that passed earns no more than the market" in section
    assert "Tested: 24 day-trade rules; none passed after costs" in rp.render_markdown(
        empty_result(), lang="en")
    for label in ("Validated", "Weakly supported", "Relatively weaker", "Market return only",
                  "Not proven", "Not studied"):
        assert label in section
    de = rp.render_markdown(result, lang="de")
    for label in ("Belegt", "Schwach belegt", "Nur relativ schwächer", "Nur Marktrendite",
                  "Nicht belegt", "Nicht untersucht"):
        assert label in de
    assert "Short-Ideen sind nirgends belegt" in de
    # nothing evidenced: the paragraph says so
    assert "Nothing is evidenced" in rp.render_markdown(empty_result(), lang="en")


@pytest.mark.unit
def test_horizon_filter_limits_the_sections():
    result = evidenced_result()
    md = rp.render_markdown(result, lang="en", horizons=["day"])
    assert "### Day trade" in md and "### Long term" not in md and "UP2" in md.split("## Technical picture")[1]
    assert "| Long | UP2 |" not in md


@pytest.mark.unit
def test_report_without_an_idea_layer_says_so_and_still_renders(result):
    result.ideas = None
    md = rp.render_markdown(result, lang="en")
    assert "idea layer was not available" in md and "## Technical picture" in md
    assert "UP2" in md and "nothing can be rated" in md


@pytest.mark.unit
@pytest.mark.parametrize("long_ret,short_pnl,uni,en_end,de_end", [
    (0.0087, -0.0315, 0.0132, "no sign of an edge", "kein Hinweis auf einen Vorteil"),
    (0.0300, 0.0200, 0.0100, "a hint of an edge, but not proof", "ein Hinweis auf einen Vorteil, aber noch kein Beleg"),
    (0.0300, -0.0200, 0.0100, "a mixed picture", "gemischtes Bild"),
])
def test_validation_verdict_is_computed_from_the_numbers(long_ret, short_pnl, uni, en_end, de_end):
    val = make_validation(long_ret, short_pnl, uni)
    en, de = rp.validation_verdict(val, "en"), rp.validation_verdict(val, "de")
    assert en_end in en and de_end in de
    assert en.startswith("Over the last 24 months") and de.startswith("In den letzten 24 Monaten lagen")


@pytest.mark.unit
def test_validation_verdict_matches_the_example_sentence_and_the_report_carries_it():
    val = make_validation(0.0087, -0.0315, 0.0132)
    assert rp.validation_verdict(val, "de") == (
        "In den letzten 24 Monaten lagen Long-Signale 0,45 Prozentpunkte unter dem Durchschnitt "
        "aller Werte, und Short-Signale verloren 3,15 %: kein Hinweis auf einen Vorteil.")
    result = evidenced_result()
    md = rp.render_markdown(result, val, None, "de")
    assert "Urteil: In den letzten 24 Monaten" in md
    assert all(c in md for c in rp.CAVEATS["de"])           # all four caveats
    assert "-0,84 %" not in md                               # spread is the constructed -0.8 %
    assert "-0,80 %" in md and "je Termin" in md and "gepoolt" in md and "-0,60" in md


@pytest.mark.unit
def test_validation_with_zero_rebalances_is_one_clear_sentence_without_a_table():
    val = make_validation(None, None, None, rebalances=0)
    val.first_rebalance = val.last_rebalance = ""
    result = evidenced_result()
    en = rp.render_markdown(result, val, None, "en")
    section = en[en.index("## Validation"):en.index("## Method")]
    assert "0 rebalances" in section and "nothing to show" in section
    assert "|" not in section and "0 rebalances,  - " not in section
    assert "0 Rebalancing-Termine" in rp.render_markdown(result, val, None, "de")


@pytest.mark.unit
def test_html_is_self_contained_escaped_badged_and_themed(validation):
    evil = "<script>alert(1)</script>"
    result = result_with([idea(evil, "Long", "long", "validated"),
                          idea("OK", "Short", "swing", "beta")], [day_note()])
    result.skipped.append(("<b>x</b>", "no price data & <i>bad</i>"))
    result.warnings.append("<img src=x onerror=alert(1)>")
    html = rp.render_html(result, validation, None, "en")
    assert html.startswith("<!doctype html>")
    assert "<script" not in html.lower().replace("&lt;script", "")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<img" not in html and "&lt;img src=x" in html
    assert "<b>" not in html and "<i>" not in html
    assert "Content-Security-Policy" in html and "default-src 'none'" in html
    assert "prefers-color-scheme: dark" in html
    assert "http://" not in html and "https://" not in html
    assert 'class="long"' in html and 'class="badge b-validated">Validated</span>' in html
    assert 'class="badge b-beta">Market return only</span>' in html
    assert "width=device-width" in html
    assert "not investment advice" in html
    de = rp.render_html(result, validation, None, "de")
    assert '<html lang="de">' in de and "Haftungsausschluss" in de
    assert 'class="badge b-validated">Belegt</span>' in de


@pytest.mark.unit
def test_save_scan_writes_four_files_and_the_json_carries_the_ideas(validation, tmp_path):
    result = evidenced_result()
    paths = rp.save_scan(result, validation, None, "de", tmp_path)
    folder = tmp_path / "scans" / f"us-custom_{END}"
    assert {p.parent for p in paths.values()} == {folder}
    assert sorted(p.name for p in paths.values()) == ["digest.txt", "scan.html", "scan.json",
                                                      "scan.md"]
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["regime"]["label"] == result.regime.label
    assert data["validation"]["rebalances"] == validation.rebalances
    assert "verdict" in data["validation"] and "verdict_de" in data["validation"]
    assert [i["symbol"] for i in data["ideas"]["ideas"]] == ["UP2", "UP1", "UP0"]
    assert data["ideas"]["ideas"][0]["watch_only"] is False
    assert paths["md"].read_text(encoding="utf-8") == rp.render_markdown(
        result, validation, None, "de")
    digest = paths["digest"].read_text(encoding="utf-8")
    assert digest.startswith("Belegt:") and "Keine Anlageberatung." in digest


@pytest.mark.unit
def test_rules_note_helper_text_is_the_engines_own_sentence():
    # the report prints the engine's notes verbatim in the chosen language
    result = result_with([], [rules_note("medium", "long")])
    assert rules_note("medium", "long").text("de") in rp.render_markdown(result, lang="de")


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
