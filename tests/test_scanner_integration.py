"""The scanner wired into the app, and the shared plumbing: numbers, notices, digest and
question output after the review of the medium/long study."""

import copy
import pickle
import re
from datetime import datetime

import pytest
from typer.testing import CliRunner

from tests.scanner_helpers import END
from tests.scanner_idea_helpers import (
    base_result,
    empty_result,
    evidenced_result,
    idea,
    result_with,
)
from tests.test_scanner_assess import (
    NOW as ASSESS_NOW,
    fake_scan,
    idea as assess_idea,
    loader,
    signal,
)
from tradingagents.scanner import (
    ask as K,
    assess as A,
    digest as dg,
    fmt,
    horizons as hz,
    notices,
    report as rp,
)
from tradingagents.scanner.notices import Notice

NOW = datetime(2026, 11, 13, 16, 5)
ENGLISH_NUMBER = re.compile(r"(?<![\d.,])\d+\.\d+(?![\d.,])")
INTERNAL_IDS = re.compile(r"\b(S3l|S\d[a-z]?|D\d[a-z]?|H\dx?|T[0-5]|V[1-5])\b|\(Basis\)|\(baseline\)")


# ------------------------------------------------------------------ wired into the app


@pytest.mark.unit
def test_the_main_app_lists_the_question_commands_and_keeps_its_default_command():
    import typer.main

    from cli.main import analyze, app

    names = set(typer.main.get_command(app).commands)
    assert {"scan", "assess", "ask", "event-study", "implied-move", "backtest"} <= names
    assert app.registered_callback.callback is analyze          # no subcommand: the analysis flow
    out = CliRunner().invoke(app, ["--help"]).output
    for command in ("scan", "assess", "ask", "event-study", "implied-move"):
        assert command in out


# ------------------------------------------------------------------ notices


@pytest.mark.unit
def test_a_notice_survives_pickle_copy_and_deepcopy():
    n = Notice("stale", last="2026-11-01")
    for clone in (pickle.loads(pickle.dumps(n)), copy.copy(n), copy.deepcopy(n)):
        assert isinstance(clone, Notice) and clone == n and clone.code == "stale"
        assert clone.params == {"last": "2026-11-01"} and clone.text("de") == n.text("de")
    regime = Notice("regime_input_missing", item="breadth")
    assert pickle.loads(pickle.dumps(regime)).text("de") == regime.text("de")


@pytest.mark.unit
def test_modules_register_their_own_codes_through_one_public_function():
    notices.register({"test_code": {"en": "{n:.1f} apples", "de": "{n:.1f} Äpfel"}})
    n = Notice("test_code", n=1234.5)
    assert str(n) == "1234.5 apples" and n.text("de") == "1234,5 Äpfel"
    notices.register({"test_code": {"en": "{n:.1f} apples", "de": "{n:.1f} Äpfel"}})   # same: fine
    with pytest.raises(ValueError):
        notices.register({"test_code": {"en": "different"}})
    with pytest.raises(ValueError):
        notices.register({"no_english": {"de": "nur deutsch"}})
    assert pickle.loads(pickle.dumps(n)).text("en") == "1234.5 apples"
    # the assessment's summary sentences are ordinary notices now
    assert not hasattr(A, "Statement") and "assess_summary" in notices._TEXT


# ------------------------------------------------------------------ numbers at the source


@pytest.mark.unit
def test_numbers_are_formatted_in_the_readers_language_at_the_source():
    assert fmt.format_text("{x:+.1f} % und {n:.0f} Tage {name}", "de", x=18.24, n=3, name="a.b") \
        == "+18,2 % und 3 Tage a.b"
    assert fmt.format_text("{x:+.1f} %", "en", x=18.24) == "+18.2 %"
    assert fmt.format_text("{x:,.2f}", "de", x=1234.5) == "1.234,50"
    assert fmt.format_text("{ok}", "de", ok=True) == "True"          # a bool is not a number


@pytest.mark.unit
def test_no_german_output_contains_an_english_formatted_number():
    result = base_result()
    texts = []
    for s in result.signals:
        texts += s.reasons_text("de")
    texts += result.regime.describe("de")
    texts += [w.text("de") for w in result.warnings]
    texts += [n.text("de") for n in result.ideas.notes]
    for i in result.ideas.ideas:
        texts += i.reasons_text("de")
        texts.append(i.evidence.text("de"))
    assert texts and any("," in t for t in texts)
    for text in texts:
        assert not ENGLISH_NUMBER.findall(text), text
    # the regex repairs are gone: nothing rewrites a sentence after the fact
    assert not hasattr(rp, "localize_text") and not hasattr(rp, "_DECIMAL_POINT")
    notes = [Notice("stale_data", asof="2026-11-10", lag=3, requested="2026-11-13"),
             hz._note("idea_relative_only", "medium", "short", "stocks_us", rule_en="x",
                      rule_de="y", mean=0.45, t=1.2, period_en="month", period_de="Monat",
                      date="2026-11-13")]
    assert "0,45 %" in notes[1].text("de") and "0.45 %" in notes[1].text("en")
    assert "1,2" in notes[1].text("de")


# ------------------------------------------------------------------ plain language


@pytest.mark.unit
def test_no_internal_rule_id_appears_in_a_sentence():
    result = base_result()
    sentences = []
    for lang in ("de", "en"):
        sentences += [n.text(lang) for n in result.ideas.notes]
        sentences += [i.evidence.text(lang) for i in result.ideas.ideas]
        sentences += [i.rule_name.get(lang, "") for i in result.ideas.ideas]
    assert sentences
    for text in sentences:
        assert not INTERNAL_IDS.search(text), text
    assert hz.rules.RULES["T0"].name_de == "Technisches Gesamtbild"
    for lang in ("de", "en"):
        note = hz._note("idea_beta_only", "swing", None, "stocks_us", n=9, date="2026-11-12",
                        rules_en="Volume breakout, long side only",
                        rules_de="Ausbruch mit Volumen, nur Long")
        assert "S3l" not in note.text(lang)
        assert "Ausbruch mit Volumen" in note.text("de")


@pytest.mark.unit
def test_a_note_never_points_at_a_list_the_output_does_not_show():
    notes = [
        hz._note("idea_no_rec_rules", "medium", "long", "stocks_us", n=11, date="2026-11-13"),
        hz._note("idea_not_studied_watch", "medium", "long", "custom"),
        Notice("idea_no_snapshot"),
    ]
    for note in notes:
        for lang in ("de", "en"):
            text = note.text(lang).lower()
            assert "unten" not in text and "below" not in text, text


# ------------------------------------------------------------------ the digest


@pytest.mark.unit
def test_the_digest_title_shows_the_range_when_the_markets_prices_differ():
    a, b = evidenced_result("us"), evidenced_result("dax")
    a.universe, b.universe = "us", "dax"
    a.as_of_date, b.as_of_date = "2026-11-12", "2026-11-13"
    title = dg.build_digest([a, b], "de", now=NOW).body.split("\n")[0]
    assert title.endswith("Kurse bis 12.11.2026–13.11.2026")
    b.as_of_date = "2026-11-12"
    assert dg.build_digest([a, b], "de", now=NOW).body.split("\n")[0].endswith("Kurse bis 12.11.2026")
    assert dg.build_digest([a, b], "en", now=NOW).body.split("\n")[0].endswith("prices up to 2026-11-12")


@pytest.mark.unit
def test_a_market_with_nothing_evidenced_says_so_in_one_sentence_then_labels_the_picture():
    result = empty_result()
    result.universe = "us"
    body = dg.build_digest([result], "de", now=NOW).body
    assert "Keine belegte Empfehlung. Technisches Bild (nicht belegt): stark UP2" in body
    assert "schwach DN2" in body
    # without room for the picture the sentence alone remains
    tight = dg.build_digest([result], "de", max_per_side=1, now=NOW).body
    assert "Keine belegte Empfehlung." in tight
    en = dg.build_digest([result], "en", now=NOW).body
    assert "No evidenced recommendation. Technical picture (not evidenced): strong UP2" in en


@pytest.mark.unit
def test_a_time_exit_recommendation_shows_equal_weight_and_no_stop_in_the_digest():
    rec = idea("UP2", "Long", "long", "validated", stop=None, size=7.7)
    result = result_with([rec], [])
    line = next(ln for ln in dg.build_digest([result], "de", now=NOW).body.split("\n")
                if ln.startswith("LONG UP2"))
    assert "7,7 % gleichgewichtet, kein Stop" in line and "Stop " not in line.replace("kein Stop", "")
    en = next(ln for ln in dg.build_digest([result], "en", now=NOW).body.split("\n")
              if ln.startswith("LONG UP2"))
    assert "7.7 % equal weight, no stop" in en


@pytest.mark.unit
def test_the_digest_line_carries_the_caveat_and_the_conflict_of_the_idea():
    ev_kw = {"caveat": {"de": "Ergebnis hängt an 3 Aktien (A, B, C); ohne sie t = 1,2",
                        "en": "depends on 3 stocks (A, B, C)", "short_de": "hängt an A, B, C",
                        "short_en": "depends on A, B, C"}}
    rec = idea("DN0", "Long", "long", "experimental", stop=None, size=10.0, **ev_kw)
    result = result_with([rec], [])
    line = next(ln for ln in dg.build_digest([result], "de", now=NOW).body.split("\n")
                if ln.startswith("LONG DN0"))
    assert "hängt an A, B, C" in line and "schwach belegt" in line.lower()


# ------------------------------------------------------------------ the report and the engine


@pytest.mark.unit
def test_report_and_digest_render_time_exit_ideas_watch_items_conflicts_and_stale_flags():
    rec = idea("UP2", "Long", "long", "validated", stop=None, size=10.0)
    rec.size_note = {"en": "Sized equal-weight ... The tested rule has no stop.",
                     "de": "Gleichgewichtet ... Die getestete Regel hat keinen Stop."}
    rec.conflict = {"en": "Technically weak right now: ...", "de": "Technisch aktuell schwach: ..."}
    rec.evidence = hz.Evidence(**{**rec.evidence.__dict__, "stale": True, "age_days": 140})
    watch = idea("DN2", "Short", "medium", "unproven", stop=None, size=None)
    watch.entry = {}
    result = result_with([rec, watch], [])
    result.ideas.stale, result.ideas.study_age_days = True, 140
    result.ideas.notes.append(Notice("idea_stale", days=140, date="2026-06-26",
                                     date_de="26.06.2026", limit=120))
    for lang in ("de", "en"):
        md = rp.render_markdown(result, lang=lang)
        html = rp.render_html(result, lang=lang)
        assert ("kein Stop" if lang == "de" else "no stop") in md
        assert ("Gleichgewichtet ..." if lang == "de" else "Sized equal-weight ...") in md
        assert ("140 Tage" if lang == "de" else "140 days") in md and "<table" in html
    d = dg.build_digest([result], "de", now=NOW)
    assert "LONG UP2" in d.body and "veraltet" in rp.render_markdown(result, lang="de")


@pytest.mark.unit
def test_a_watch_item_row_has_no_entry_stop_or_size_column():
    result = result_with([idea("AVO", "Short", "medium", "relative", stance="avoid",
                               stop=None, size=None)], [])
    for lang in ("de", "en"):
        blocks = rp.build_blocks(result, lang=lang)
        tables = [b for b in blocks if b[0] == "table"]
        watch = next(t for t in tables if any(r[1] == "AVO" for r in t[2]))
        header = [h.lower() for h in watch[1]]
        assert not any(word in " ".join(header) for word in ("stop", "größe", "size", "einstieg",
                                                              "entry"))


# ------------------------------------------------------------------ the question commands


@pytest.mark.unit
def test_an_answer_is_short_one_block_per_instrument_one_legend_one_disclaimer(monkeypatch):
    nvda = assess_idea("NVDA", "medium", "Short")
    sap = assess_idea("SAP.DE", "medium", "Long")
    monkeypatch.setattr(A, "run_scan", fake_scan([signal("NVDA", -58.0), signal("SAP.DE", 62.0)],
                                                 [nvda, sap]))
    answer = K.answer_question("Soll ich Nvidia oder SAP auf 3 Monate kaufen?", lang="de",
                               as_of_date=END, loader=loader, now=ASSESS_NOW)
    text = answer.text
    assert text.index("Kurzantwort") < text.index("**Nvidia")
    assert text.count("**Nvidia") == 1 and text.count("**SAP") == 1
    assert text.count("Belegt =") == 1 and text.count("Keine Anlageberatung") == 1
    assert "Gegensignal" not in text            # no reasons list in the compact answer
    assert "unten" not in text.lower()
    assert len(text.splitlines()) < 30 and len(text) < 2600
