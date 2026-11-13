"""Parsing a free-text question and answering it deterministically."""

import pytest

from tests.scanner_helpers import END
from tests.test_scanner_assess import NOW, fake_scan, idea, loader, note, signal
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.scanner import aliases
from tradingagents.scanner import assess as A
from tradingagents.scanner import ask as K
from tradingagents.scanner.notices import Notice
from tradingagents.scanner.universe import UNIVERSES

GTA = ("Ich weiß das GTA6 nächste Woche rauskommt. Ist dadurch am Markt etwas zu erwarten unter "
       "Berücksichtigung das viele der menschheit in Amerika und Deutschland zb wahrscheinlich "
       "über mehrere Tage zocken werden und sich in Deutschland wahrscheinlich viele Freitags "
       "krankmelden")


# --------------------------------------------------------------------------- aliases


@pytest.mark.unit
def test_every_alias_target_has_a_valid_symbol_shape_and_a_name():
    import re
    shape = re.compile(r"^\^?[A-Z0-9]{1,8}([.=-][A-Z]{1,3})?$")
    assert all(shape.match(s) for s in aliases.ALIASES.values())
    assert set(aliases.ALIASES.values()) <= set(aliases.NAMES)
    assert all(a == a.lower() for a in aliases.ALIASES)


@pytest.mark.unit
def test_every_universe_member_has_a_name_and_the_asked_for_names_resolve():
    members = {s for n, (_d, _b, m) in UNIVERSES.items() if n != "all" for s in m}
    assert members <= set(aliases.NAMES)
    expect = {"apple": "AAPL", "nvidia": "NVDA", "tesla": "TSLA", "alphabet": "GOOGL",
              "google": "GOOGL", "sap": "SAP.DE", "siemens": "SIE.DE", "rheinmetall": "RHM.DE",
              "deutsche telekom": "DTE.DE", "telekom": "DTE.DE", "bitcoin": "BTC-USD",
              "ethereum": "ETH-USD", "dax": "^GDAXI", "s&p 500": "SPY", "s&p": "SPY",
              "nasdaq": "QQQ", "gold": "GLD", "silber": "SLV", "silver": "SLV", "öl": "USO",
              "oil": "USO", "anleihen": "TLT", "bonds": "TLT", "take-two": "TTWO",
              "sony": "SONY", "gamestop": "GME"}
    for name, symbol in expect.items():
        assert aliases.ALIASES[name] == symbol, name


@pytest.mark.unit
def test_resolve_and_display_name():
    assert aliases.resolve(" Nvidia ") == "NVDA" and aliases.resolve("sap.de") == "SAP.DE"
    assert aliases.resolve("xyz") == "XYZ"
    assert aliases.display_name("NVDA") == "Nvidia (NVDA)"
    assert aliases.display_name("QQQ") == "Nasdaq 100 (QQQ)"
    assert aliases.display_name("UNKNOWN") == "UNKNOWN"


# --------------------------------------------------------------------------- parser


@pytest.mark.unit
def test_names_symbols_horizon_intent_and_language_of_the_example_question():
    p = K.parse_question("Soll ich Nvidia oder SAP auf 3 Monate kaufen?")
    assert p.tickers == ["NVDA", "SAP.DE"] and p.horizons == ["medium"]
    assert p.intent == "buy" and p.compare and p.lang == "de" and not p.event_like


@pytest.mark.unit
def test_short_question_is_a_short_intent():
    p = K.parse_question("Lohnt sich ein Short auf Tesla?")
    assert p.tickers == ["TSLA"] and p.intent == "short" and not p.compare


@pytest.mark.unit
@pytest.mark.parametrize("text,expected", [
    ("Should I buy NVDA or SAP.DE?", ["NVDA", "SAP.DE"]),
    ("BTC-USD und btc-usd", ["BTC-USD"]),
    ("Wie steht NVDA.", ["NVDA"]),
    ("Was ist mit Take-Two und Sony, Teslas Zahlen?", ["TTWO", "SONY", "TSLA"]),
    ("Siemens Energy vs Siemens", ["ENR.DE", "SIE.DE"]),          # longest name first
    ("Gold, Silber und Öl", ["GLD", "SLV", "USO"]),
    ("Lohnt Deutsche Telekom oder die Telekom-Aktie", ["DTE.DE"]),
    ("^GSPC und BRK-B", ["^GSPC", "BRK-B"]),
    ("Kauf XLK statt SPY", ["XLK", "SPY"]),
])
def test_tickers_found(text, expected):
    assert K.parse_question(text).tickers == expected


@pytest.mark.unit
def test_upper_case_words_and_countries_do_not_become_tickers():
    p = K.parse_question("ICH denke die USA und ein ETF mit KI sind spannend, Deutschland auch")
    assert p.tickers == [] and p.unverified == []
    assert K.parse_question("Sap ist ein Wort, ich bin ok").tickers == ["SAP.DE"]   # a name, by design


@pytest.mark.unit
def test_dax_is_understood_as_the_index_by_name():
    assert K.parse_question("Wie steht der DAX?").tickers == ["^GDAXI"]


@pytest.mark.unit
def test_unknown_symbol_like_word_is_listed_unverified():
    p = K.parse_question("Was hältst du von XYZW?")
    assert p.tickers == ["XYZW"] and p.unverified == ["XYZW"]
    assert K.parse_question("Was hältst du von Xyzw?").tickers == []      # not upper case


@pytest.mark.unit
def test_all_caps_text_does_not_turn_words_into_symbols():
    assert K.parse_question("SOLL ICH KAUFEN").tickers == []


@pytest.mark.unit
@pytest.mark.parametrize("text,expected", [
    ("heute NVDA", ["day"]), ("Daytrade auf NVDA", ["day"]),
    ("in ein paar Tagen", ["swing"]), ("nächste Woche", ["swing"]),
    ("kurzfristig", ["swing"]), ("mittelfristig", ["medium"]), ("langfristig", ["long"]),
    ("auf 3 Monate", ["medium"]), ("12 Monate", ["long"]), ("6 Monate", ["long"]),
    ("in 2 Jahren", ["long"]), ("10 Tage", ["swing"]), ("3 Wochen", ["medium"]),
    ("heute und langfristig", ["day", "long"]), ("NVDA", []),
    ("for 12 months", ["long"]), ("over the next week", ["swing"]),
])
def test_horizons(text, expected):
    assert K.parse_question(text).horizons == expected


@pytest.mark.unit
@pytest.mark.parametrize("text,intent", [
    ("NVDA kaufen?", "buy"), ("Soll ich einsteigen", "buy"), ("Should I buy it", "buy"),
    ("NVDA verkaufen?", "sell"), ("Short auf SAP", "short"), ("Werden die Kurse fallen?", "short"),
    ("Soll ich Tesla halten?", "hold"), ("Wie steht Tesla?", "none"),
    ("Verkaufen oder kaufen?", "sell"),
])
def test_intent(text, intent):
    assert K.parse_question(text).intent == intent


@pytest.mark.unit
@pytest.mark.parametrize("text,lang", [
    ("Soll ich Nvidia kaufen?", "de"), ("Würde Gold steigen", "de"),
    ("Should I buy Nvidia or SAP?", "en"), ("Is gold a good buy for the next year?", "en"),
    ("NVDA", "en"),
])
def test_language(text, lang):
    assert K.parse_question(text).lang == lang


@pytest.mark.unit
def test_the_gta_question_names_no_instrument_but_reads_like_an_event():
    p = K.parse_question(GTA)
    assert p.tickers == [] and p.event_like and p.lang == "de"


# --------------------------------------------------------------------------- answers


def _scan(*args, **kw):
    return fake_scan(*args, **kw)


@pytest.mark.unit
def test_answer_for_two_instruments_leads_with_the_asked_horizon(monkeypatch):
    monkeypatch.setattr(A, "run_scan", _scan([signal("NVDA", -18.0), signal("SAP.DE", 62.0)], []))
    ans = K.answer_question("Soll ich Nvidia oder SAP auf 3 Monate kaufen?", as_of_date=END,
                            loader=loader, now=NOW)
    text = ans.text
    assert ans.assessment is not None and ans.parsed.horizons == ["medium"]
    short = text.split("**Kurzantwort**")[1].split("\n\n")[0]
    assert short.count("- ") == 1 and "Mittelfristig: Für keinen der beiden Werte" in short
    for needed in ("Verstanden:", "Nvidia (NVDA)", "SAP (SAP.DE)", "Horizont: Mittelfristig",
                   "Absicht: kaufen / einsteigen", "Vergleich", "Technisches Bild",
                   "Keine Anlageberatung"):
        assert needed in text, needed
    assert "Zur Ereignisfrage" not in text


@pytest.mark.unit
def test_without_a_horizon_medium_and_long_are_answered(monkeypatch):
    monkeypatch.setattr(A, "run_scan", _scan([signal("TSLA", 10.0)], []))
    text = K.answer_question("Wie steht Tesla?", as_of_date=END, loader=loader, now=NOW).text
    short = text.split("**Kurzantwort**")[1].split("\n\n")[0]
    assert "Mittelfristig" in short and "Langfristig" in short and "Daytrade" not in short


@pytest.mark.unit
def test_short_question_shows_the_short_evidence_and_the_risk(monkeypatch):
    shorts = [idea("TSLA", "medium", "Short")]
    monkeypatch.setattr(A, "run_scan", _scan(
        [signal("TSLA", -60.0)], shorts, notes=[]))
    ans = K.answer_question("Lohnt sich ein Short auf Tesla?", as_of_date=END, loader=loader,
                            now=NOW)
    assert "**Zur Short-Seite**" in ans.text
    assert shorts[0].evidence.text("de") in ans.text.split("**Zur Short-Seite**")[1]
    assert "nur Short-Beobachtung" in ans.text and "Verlust nicht auf den Einsatz begrenzt" in ans.text
    assert "Short-Empfehlung" not in ans.text


@pytest.mark.unit
def test_short_question_without_a_short_signal_quotes_the_horizon_note(monkeypatch):
    n = Notice("no_data")
    n.params = {"horizon": "medium", "side": "short"}
    monkeypatch.setattr(A, "run_scan", _scan([signal("TSLA", 10.0)], [], notes=[n]))
    text = K.answer_question("Lohnt sich ein Short auf Tesla?", as_of_date=END, loader=loader,
                             now=NOW).text
    assert "Mittelfristig, Tesla (TSLA): keine Kursdaten von Yahoo" in text   # the note, verbatim
    assert "kein Short-Signal." in text                                        # long has no note


@pytest.mark.unit
def test_sell_question_does_not_pretend_to_know_the_users_situation(monkeypatch):
    monkeypatch.setattr(A, "run_scan", _scan([signal("TSLA", 10.0)], []))
    text = K.answer_question("Soll ich Tesla verkaufen?", as_of_date=END, loader=loader,
                             now=NOW).text
    assert "hängt von deiner Lage ab" in text


@pytest.mark.unit
def test_instruments_in_an_event_question_get_the_event_note(monkeypatch):
    monkeypatch.setattr(A, "run_scan", _scan([signal("TTWO", 10.0)], []))
    ans = K.answer_question("GTA 6 erscheint nächste Woche, was passiert mit Take-Two?",
                            as_of_date=END, loader=loader, now=NOW)
    assert ans.assessment is not None and ans.parsed.event_like
    assert "Zur Ereignisfrage" in ans.text and "Claude-Chat-Sitzung" in ans.text
    assert "event-study" in ans.text and "implied-move" in ans.text


@pytest.mark.unit
def test_open_event_question_gets_the_honest_fallback_not_an_assessment(monkeypatch):
    def never(*a, **k):
        raise AssertionError("no instrument: nothing to scan")
    monkeypatch.setattr(A, "run_scan", never)
    ans = K.answer_question(GTA)
    text = ans.text
    assert ans.assessment is None and ans.parsed.tickers == []
    for needed in ("kein Wertpapier erkannt", "So kann ich das nicht beantworten",
                   "erfinde deshalb keine Einschätzung", "Claude-Chat-Sitzung",
                   "eigene Überlegungen", "event-study", "implied-move", "assess",
                   "Keine Anlageberatung"):
        assert needed in text, needed
    assert "Score" not in text and "Empfehlung Long" not in text


@pytest.mark.unit
def test_fallback_in_english():
    text = K.answer_question("What does the GTA VI release mean for the market?", lang="en").text
    assert "I cannot answer this as asked" in text and "Claude chat session" in text
    assert "Nicht" not in text and "Keine" not in text


@pytest.mark.unit
def test_unrecognised_symbol_alone_falls_back_and_names_it(monkeypatch):
    monkeypatch.setattr(A, "run_scan", _scan([], [], skipped=[("XYZW", Notice("no_data"))]))
    ans = K.answer_question("Was hältst du von XYZW?", as_of_date=END, loader=loader, now=NOW)
    assert "XYZW" in ans.text and "Nicht als Wertpapier erkannt" in ans.text
    assert ans.assessment is not None and "So kann ich das nicht beantworten" in ans.text


@pytest.mark.unit
def test_partly_unknown_names_the_failed_one_and_answers_the_rest(monkeypatch):
    monkeypatch.setattr(A, "run_scan", _scan(
        [signal("NVDA", 20.0)], [], skipped=[("XYZW", Notice("no_data"))]))
    text = K.answer_question("NVDA oder XYZW?", as_of_date=END, loader=loader, now=NOW).text
    assert "Keine Kursdaten gefunden für: XYZW" in text and "Kurzantwort" in text


@pytest.mark.unit
def test_english_question_gets_an_english_answer_and_lang_overrides(monkeypatch):
    monkeypatch.setattr(A, "run_scan", _scan([signal("NVDA", 20.0)], []))
    en = K.answer_question("Should I buy NVDA for 12 months?", as_of_date=END, loader=loader,
                           now=NOW).text
    assert "Short answer" in en and "Not investment advice" in en and "Keine" not in en
    de = K.answer_question("Should I buy NVDA?", lang="de", as_of_date=END, loader=loader,
                           now=NOW).text
    assert "Kurzantwort" in de


@pytest.mark.unit
def test_empty_question_and_vendor_outage(monkeypatch):
    with pytest.raises(ValueError):
        K.answer_question("   ")

    def down(*a, **k):
        raise VendorUnavailableError("no yahoo")
    monkeypatch.setattr(A, "run_scan", down)
    with pytest.raises(VendorUnavailableError):
        K.answer_question("NVDA?", as_of_date=END, loader=loader, now=NOW)


@pytest.mark.unit
def test_answer_to_dict_is_json_safe(monkeypatch):
    import json
    monkeypatch.setattr(A, "run_scan", _scan([signal("NVDA", 20.0)], [], notes=[note("day", None)]))
    ans = K.answer_question("NVDA?", as_of_date=END, loader=loader, now=NOW)
    data = json.loads(json.dumps(ans.to_dict()))
    assert data["parsed"]["tickers"] == ["NVDA"] and data["assessment"]["tickers"]
