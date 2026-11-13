"""Assessment of asked instruments: families, statuses per horizon, comparison, rendering.

Most tests replace ``run_scan`` with a scan whose ideas are built by hand, so the status of
every ticker is known exactly and no evidence snapshot can change the outcome; one test
runs the real scan on synthetic bars to prove the pieces fit.
"""

import json
import re
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from tests.scanner_helpers import END, frame
from tradingagents.scanner import assess as A
from tradingagents.scanner.horizons import Evidence, Idea, IdeaSet
from tradingagents.scanner.notices import Notice
from tradingagents.scanner.signals import Signal
from tradingagents.scanner.universe import UNIVERSES

NOW = datetime(2025, 7, 1, 12, tzinfo=UTC)

# --------------------------------------------------------------------------- builders


def evidence(label="unproven", horizon="medium", side="long", mean=-120.0, t=-0.8, rule="T0"):
    return Evidence(label=label, rule_id=rule, rule_name={"en": "Momentum 12-1", "de": "Momentum 12-1"},
                    family="stocks_us", horizon=horizon, side=side, study_date="2026-11-12",
                    judged="long_excess", oos_n=44, oos_mean_bps=mean, oos_t=t,
                    window=("2022-01-31", "2026-06-30"), n_rules_tested=15)


def idea(symbol, horizon, direction="Long", label="unproven", rule="T0"):
    rec = label in ("validated", "experimental")
    return Idea(
        symbol=symbol, direction=direction, horizon=horizon, rule_id=rule, rank_score=1.0, rank=1,
        reference_price=123.45, reference_date="2025-06-30",
        entry={"en": "Buy at the next open", "de": "Kauf zur nächsten Eröffnung"},
        stop=110.0, target=None, size_pct=4.2,
        holding={"en": "Long term: about 6 months", "de": "Langfristig – etwa 6 Monate"},
        reason_items=[], evidence=evidence(label, horizon, direction.lower(), 597.0 if rec else -120.0,
                                           2.2 if rec else -0.8, rule),
        watch_only=not rec, stance="recommended" if rec else "watch",
        rule_name={"en": "Momentum 12-1", "de": "Momentum 12-1"})


def signal(symbol, score, direction=None):
    direction = direction or ("Long" if score >= 35 else "Short" if score <= -35 else "Neutral")
    return Signal(symbol=symbol, direction=direction, score=score, conviction="Medium",
                  components={}, close=100.0, last_date="2025-06-30", regime="neutral",
                  reason_items=[("trend_up", {"pct": 5.3})])


def note(horizon, side):
    n = Notice("no_data")
    n.params = {"horizon": horizon, "side": side}
    return n


def fake_scan(signals, ideas, notes=(), skipped=()):
    calls = []

    def run_scan(universe, as_of_date, tickers=None, top=10, loader=None, now=None):
        calls.append({"universe": universe, "tickers": list(tickers), "top": top})
        return SimpleNamespace(
            regime=SimpleNamespace(label="risk-on", points=3), benchmark="SPY", warnings=[],
            signals=list(signals), skipped=list(skipped),
            ideas=IdeaSet(family="stocks_us", study_date="2026-11-12", evidence_available=True,
                          ideas=list(ideas), notes=list(notes)))
    run_scan.calls = calls
    return run_scan


def loader(symbols, as_of, lookback):
    return {s: frame(100 * np.exp(0.001 * np.arange(300))) for s in symbols}


@pytest.fixture
def patch_scan(monkeypatch):
    def apply(scan):
        monkeypatch.setattr(A, "run_scan", scan)
        return scan
    return apply


# --------------------------------------------------------------------------- classification


@pytest.mark.unit
@pytest.mark.parametrize("symbol,expected", [
    ("NVDA", ("us", "stocks_us", True)), ("SAP.DE", ("dax", "stocks_dax", True)),
    ("XLK", ("sectors", "etfs", True)), ("GLD", ("global", "etfs", True)),
    ("BTC-USD", ("crypto", "crypto", True)), ("FOO.DE", ("dax", "stocks_dax", False)),
    ("PEPE-USD", ("crypto", "crypto", False)), ("^GDAXI", ("dax", "stocks_dax", False)),
    ("^N225", ("global", "etfs", False)), ("TTWO", ("us", "stocks_us", False)),
])
def test_classify_uses_universe_membership_then_shape(symbol, expected):
    assert A.classify(symbol) == expected


@pytest.mark.unit
def test_every_universe_member_is_classified_into_its_own_universe():
    for name in ("us", "dax", "crypto"):
        assert all(A.classify(s)[0] == name for s in UNIVERSES[name][2])


# --------------------------------------------------------------------------- statuses


@pytest.mark.unit
def test_statuses_comparison_order_and_conflict(patch_scan):
    ideas = [
        idea("AAA", "long", label="validated", rule="T1"),       # evidenced Long ...
        idea("BBB", "medium", label="unproven"),                  # watch only
        idea("DDD", "medium", "Short", label="unproven"),         # on the weak list
    ]
    scan = patch_scan(fake_scan(
        [signal("AAA", -40.0), signal("BBB", 70.0), signal("CCC", 10.0), signal("DDD", -60.0)],
        ideas))
    result = A.assess(["DDD", "CCC", "BBB", "AAA"], as_of_date=END, loader=loader, now=NOW)

    assert scan.calls[0]["universe"] == "us"
    assert scan.calls[0]["top"] >= 1000                  # every basket member must be visible
    assert {"AAA", "BBB", "CCC", "DDD"} <= set(scan.calls[0]["tickers"])
    assert "NVDA" in scan.calls[0]["tickers"]            # the asked names join their peers
    status = {t.symbol: {h: v.status for h, v in t.views.items()} for t in result.tickers}
    assert status["AAA"]["long"] == "recommended" and status["AAA"]["medium"] == "none"
    assert status["BBB"]["medium"] == "watch" and status["CCC"]["medium"] == "none"
    assert status["DDD"]["medium"] == "weak"
    assert result.get("AAA").views["long"].conflict is True      # recommended yet technically weak
    assert result.get("BBB").views["medium"].conflict is False
    assert result.comparison["long"] == ["AAA", "BBB", "CCC", "DDD"]
    # watch first, then no signal (stronger technical picture first), the weak name last
    assert result.comparison["medium"] == ["BBB", "CCC", "AAA", "DDD"]


@pytest.mark.unit
def test_summary_names_the_stronger_name_and_says_no_recommendation(patch_scan):
    patch_scan(fake_scan([signal("NVDA", -18.0), signal("SAP.DE", 62.0)], []))
    # two universes: one scan each
    result = A.assess(["NVDA", "SAP.DE"], as_of_date=END, loader=loader, now=NOW)
    medium = next(s for s in result.summary_items if s.params["horizon"] == "medium")
    text = medium.text("de")
    assert text.startswith("Mittelfristig: Für keinen der beiden Werte gibt es eine belegte "
                           "Empfehlung; technisch ist SAP (SAP.DE) stärker (Score +62 gegen -18)")
    assert "no evidenced recommendation for either of the two names" in str(medium)


@pytest.mark.unit
def test_summary_partial_recommendation_and_conflict_wording(patch_scan):
    patch_scan(fake_scan(
        [signal("AAA", -50.0), signal("BBB", 40.0)],
        [idea("AAA", "long", label="validated", rule="T1")]))
    result = A.assess(["AAA", "BBB"], as_of_date=END, loader=loader, now=NOW)
    text = next(s for s in result.summary_items if s.params["horizon"] == "long").text("de")
    assert "Eine belegte Long-Empfehlung gibt es nur für AAA, nicht für BBB" in text
    assert "technisch aktuell aber schwach" in text


@pytest.mark.unit
def test_experimental_is_called_weakly_supported_never_validated(patch_scan):
    patch_scan(fake_scan([signal("AAA", 50.0)], [idea("AAA", "long", label="experimental")]))
    result = A.assess(["AAA"], as_of_date=END, loader=loader, now=NOW)
    text = result.summary_items[-1].text("de")
    assert "schwach belegt" in text and "(belegt)" not in text


@pytest.mark.unit
def test_unknown_and_empty_tickers_are_reported_per_ticker(patch_scan):
    patch_scan(fake_scan([signal("AAA", 50.0)], [], skipped=[("ZZZZ", Notice("no_data"))]))
    result = A.assess(["AAA", "ZZZZ", " "], as_of_date=END, loader=loader, now=NOW)
    errors = {t.symbol: t.error for t in result.tickers if t.error is not None}
    assert set(errors) == {"ZZZZ", "?"}
    assert result.get("AAA").error is None
    assert "ZZZZ" not in result.comparison.get("medium", [])
    text = A.render_assessment(result, "de")
    assert "Keine Einschätzung möglich" in text and "keine Kursdaten" in text


@pytest.mark.unit
def test_duplicates_are_asked_once_and_symbols_normalised(patch_scan):
    scan = patch_scan(fake_scan([signal("AAA", 50.0)], []))
    result = A.assess(["aaa", "AAA", "aaa"], as_of_date=END, loader=loader, now=NOW)
    assert result.requested == ["AAA"] and scan.calls[0]["tickers"].count("AAA") == 1


@pytest.mark.unit
def test_each_peer_group_is_scanned_once(patch_scan):
    scan = patch_scan(fake_scan([], []))
    A.assess(["NVDA", "AAPL", "SAP.DE", "BTC-USD", "XLK", "GLD"], as_of_date=END,
             loader=loader, now=NOW)
    assert sorted(c["universe"] for c in scan.calls) == ["crypto", "dax", "global", "sectors", "us"]


@pytest.mark.unit
def test_future_date_is_a_value_error():
    with pytest.raises(ValueError):
        A.assess(["NVDA"], as_of_date="2999-01-01", loader=loader)


# --------------------------------------------------------------------------- rendering


def _german_text(patch_scan):
    patch_scan(fake_scan(
        [signal("NVDA", -18.0), signal("SAP.DE", 62.0)],
        [idea("SAP.DE", "long", label="validated", rule="T1"), idea("SAP.DE", "medium"),
         idea("NVDA", "medium", "Short")],
        notes=[note("day", None)]))
    result = A.assess(["NVDA", "SAP.DE"], as_of_date=END, loader=loader, now=NOW)
    return result, A.render_assessment(result, "de")


@pytest.mark.unit
def test_german_rendering_is_german_with_decimal_commas(patch_scan):
    _, text = _german_text(patch_scan)
    for needed in ("Einschätzung", "Das Wichtigste", "Technisches Bild", "Keine Anlageberatung",
                   "Empfehlung Long", "nur Beobachtung", "Kein Signal", "Langfristig",
                   "Belegt: im Test"):
        assert needed in text, needed
    for english in ("Recommendation", "Entry", "No signal", "watch only", "Not investment",
                    "Assessment", "In short", "technical picture"):
        assert english not in text, english
    body = re.sub(r"\d{2}\.\d{2}\.\d{4}", "", text)       # dates are dd.mm.yyyy
    assert not re.search(r"\d\.\d", body), re.search(r".{20}\d\.\d.{10}", body).group(0)
    assert "5,3 %" in text and "597" not in text           # reasons localised; bps become percent


@pytest.mark.unit
def test_evidence_text_is_printed_verbatim_and_watch_is_not_a_recommendation(patch_scan):
    result, text = _german_text(patch_scan)
    for t in result.tickers:
        for view in t.views.values():
            for i in view.ideas:
                assert i.evidence.text("de") in text
    medium_sap = [line for line in text.splitlines() if "Mittelfristig: nur Beobachtung" in line]
    assert medium_sap and all("Empfehlung" not in line.replace("keine belegte Empfehlung", "")
                              for line in medium_sap)
    assert "Plan:" in text and "Stop: 110,00" in text and "4,2 %" in text   # only the recommendation


@pytest.mark.unit
def test_watch_only_idea_gets_no_plan_line(patch_scan):
    patch_scan(fake_scan([signal("AAA", 50.0)], [idea("AAA", "medium")]))
    text = A.render_assessment(A.assess(["AAA"], as_of_date=END, loader=loader, now=NOW), "de")
    assert "Plan:" not in text and "Stop:" not in text


@pytest.mark.unit
def test_conflict_is_shown_on_the_recommendation(patch_scan):
    patch_scan(fake_scan([signal("AAA", -50.0)], [idea("AAA", "long", label="validated")]))
    text = A.render_assessment(A.assess(["AAA"], as_of_date=END, loader=loader, now=NOW), "de")
    assert "aktuell technisch schwach" in text
    en = A.render_assessment(A.assess(["AAA"], as_of_date=END, loader=loader, now=NOW), "en")
    assert "currently technically weak" in en


@pytest.mark.unit
def test_english_rendering_has_no_german_leftovers(patch_scan):
    result, _ = _german_text(patch_scan)
    text = A.render_assessment(result, "en")
    for needed in ("Assessment", "In short", "Technical picture", "Not investment advice",
                   "Recommendation Long", "watch only", "No signal"):
        assert needed in text, needed
    for german in ("Einschätzung", "Keine Anlageberatung", "Mittelfristig", "Empfehlung",
                   "Technisches", "Überzeugung"):
        assert german not in text, german
    assert "5.3 %" in text


@pytest.mark.unit
def test_not_in_universe_is_flagged_in_the_block(patch_scan):
    patch_scan(fake_scan([signal("TTWO", 20.0)], []))
    text = A.render_assessment(A.assess(["TTWO"], as_of_date=END, loader=loader, now=NOW), "de")
    assert "Steht in keiner Liste des Programms" in text


@pytest.mark.unit
def test_to_dict_is_json_safe_and_complete(patch_scan):
    result, _ = _german_text(patch_scan)
    data = json.loads(json.dumps(result.to_dict()))
    assert data["requested"] == ["NVDA", "SAP.DE"] and data["comparison"]["long"]
    sap = next(t for t in data["tickers"] if t["symbol"] == "SAP.DE")
    assert sap["views"]["long"]["status"] == "recommended"
    assert sap["views"]["long"]["ideas"][0]["evidence"]["text_de"].startswith("Belegt")
    assert data["summary"][0]["text_de"] and data["summary"][0]["text"]


@pytest.mark.unit
def test_options_only_for_us_stocks_and_failures_leave_a_gap(patch_scan, monkeypatch):
    from tradingagents.dataflows.errors import VendorUnavailableError
    from tradingagents.dataflows.vendors.yahoo import options

    patch_scan(fake_scan([signal("NVDA", 50.0), signal("SAP.DE", 50.0)], []))

    def boom(symbol, max_expiries=4):
        raise VendorUnavailableError("down")
    monkeypatch.setattr(options, "load_implied_moves", boom)
    result = A.assess(["NVDA", "SAP.DE"], as_of_date=END, loader=loader, now=NOW,
                      with_options=True)
    assert result.get("SAP.DE").implied == [] and result.get("SAP.DE").implied_error is None
    assert result.get("NVDA").implied == [] and "down" in result.get("NVDA").implied_error
    assert "nicht verfügbar" in A.render_assessment(result, "de")


# --------------------------------------------------------------------------- the real scan


@pytest.mark.unit
def test_pipeline_on_synthetic_bars_with_the_real_scan():
    members = {s: frame(100 * np.exp((0.0005 + 0.00002 * i) * np.arange(520) +
                                     0.01 * np.sin(np.arange(520) * (0.5 + i / 50))))
               for i, s in enumerate([*UNIVERSES["us"][2], "SPY", "TTWO"])}
    members["^VIX"] = frame(np.full(520, 15.0))

    def load(symbols, as_of, lookback):
        return {s: members[s] for s in symbols if s in members}

    result = A.assess(["NVDA", "TTWO", "NOPE"], as_of_date=END, loader=load, now=NOW)
    nvda, ttwo, nope = result.tickers
    assert nvda.signal is not None and ttwo.in_universe is False and nope.error is not None
    assert set(nvda.views) == set(A.HORIZON_ORDER)
    assert nvda.snapshot is not None and nvda.snapshot.last_close
    text = A.render_assessment(result, "de")
    assert "Keine Anlageberatung" in text and "NOPE" in text
    json.dumps(result.to_dict())
