"""The phone digest: honest recommendation lines, one-line "why not", limits and determinism."""

import re
from datetime import datetime

import pytest

from tests.scanner_idea_helpers import (
    day_note,
    empty_result,
    evidenced_result,
    idea,
    result_with,
    rules_note,
)
from tradingagents.scanner import digest as dg

NOW = datetime(2026, 11, 13, 16, 5)


def _four(evidenced=True):
    make = evidenced_result if evidenced else empty_result
    out = []
    for name in ("us", "dax", "global", "crypto"):
        result = make(name)
        result.universe = name
        out.append(result)
    return out


@pytest.mark.unit
def test_signature_and_dataclass_shape():
    import inspect
    sig = inspect.signature(dg.build_digest)
    assert list(sig.parameters) == ["results", "lang", "max_per_side", "now", "horizons"]
    assert sig.parameters["lang"].default == "de" and sig.parameters["max_per_side"].default == 3
    d = dg.build_digest([evidenced_result()], now=NOW)
    assert d.text == f"{d.headline}\n\n{d.body}"


@pytest.mark.unit
@pytest.mark.parametrize("lang", ["de", "en"])
@pytest.mark.parametrize("evidenced", [True, False])
def test_headline_fits_a_push_notification_and_body_has_no_markdown(lang, evidenced):
    d = dg.build_digest(_four(evidenced), lang, now=NOW)
    assert 0 < len(d.headline) <= dg.HEADLINE_MAX and "\n" not in d.headline
    assert len(d.body) <= dg.BODY_MAX
    assert "|" not in d.body and "#" not in d.body and "**" not in d.body
    assert all(line == line.rstrip() for line in d.body.split("\n"))


@pytest.mark.unit
def test_german_recommendation_lines_follow_the_plan_format():
    d = dg.build_digest([evidenced_result()], "de", now=NOW)
    assert ("LONG UP2 · Einstieg nächste Eröffnung (Ref. 354,88) · Stop 321,34 · max. 8,5 % "
            "· Belegt") in d.body
    assert d.body.startswith("Fr 13.11.2026 16:05 · Kurse bis 30.06.2025")
    assert d.body.rstrip().endswith("Keine Anlageberatung.")
    assert "addieren sich nicht" in d.body
    # a merged line never leaves it open which side a horizon name stands for
    assert ("Langfristig Short, Mittelfristig Long+Short: keine Empfehlung – Regeln getestet, "
            "kein belegter Vorteil") in d.body
    assert d.headline.startswith("Belegt: Eigene Liste Langfristig Long: UP2, UP1, UP0")


@pytest.mark.unit
def test_nothing_evidenced_is_said_plainly_with_the_reasons():
    d = dg.build_digest([empty_result()], "de", now=NOW)
    assert d.headline.startswith("Keine belegte Empfehlung")
    assert "LONG " not in d.body and "SHORT " not in d.body
    assert "Daytrade: keine Empfehlung – 24 Regeln getestet, nach Kosten kein Vorteil" in d.body
    # the two identical medium/long reasons are merged into one line
    assert ("Langfristig, Mittelfristig: keine Empfehlung – Regeln getestet, kein belegter "
            "Vorteil") in d.body
    assert "Kurzfristig (Swing): keine Empfehlung – nicht untersucht" in d.body
    en = dg.build_digest([empty_result()], "en", now=NOW)
    assert en.headline.startswith("No evidenced recommendation")
    assert "Day trade: no recommendation – 24 rules tested, no edge after costs" in en.body


@pytest.mark.unit
def test_watch_names_are_labelled_not_proven_and_avoid_is_not_a_short():
    result = result_with(
        [idea("AVO", "Short", "medium", "relative", stance="avoid")], [day_note()])
    d = dg.build_digest([result], "de", now=NOW)
    assert ("Keine belegte Empfehlung. Technisches Bild (nicht belegt): stark UP2" in d.body
            and "schwach DN2" in d.body)
    assert "Meiden / untergewichten (relativ belegt): AVO" in d.body
    assert "SHORT AVO" not in d.body and d.headline.startswith("Keine belegte Empfehlung")


@pytest.mark.unit
def test_a_short_side_without_evidence_gets_its_own_line_and_extra_ideas_are_counted():
    ideas = [idea(f"L{i}", rank=i) for i in range(1, 6)]
    result = result_with(ideas, [rules_note("long", "short", 7), day_note()])
    d = dg.build_digest([result], "de", now=NOW)
    assert d.body.count("LONG L") == 3 and "(+2 weitere im Bericht)" in d.body
    assert "Langfristig Short: keine Empfehlung" in d.body
    assert d.headline.startswith("Belegt: Eigene Liste Langfristig Long: L1, L2, L3 +2")


@pytest.mark.unit
def test_conflict_with_the_technical_picture_is_shown_on_the_line():
    result = result_with([idea("DN0", "Long", "long", "validated")], [day_note()])
    d = dg.build_digest([result], "de", now=NOW)
    assert "LONG DN0" in d.body and "aktuell technisch schwach" in d.body


@pytest.mark.unit
def test_body_limit_trims_the_watch_lines_first_then_shrinks_the_lists():
    results = []
    for name in ("us", "dax", "global", "crypto"):
        result = result_with([idea(f"{name.upper()}{i}", rank=i, price=1234.5) for i in range(1, 9)],
                             [day_note(), rules_note("medium", "long"), rules_note("medium", "short"),
                              rules_note("long", "short", 7)], name)
        results.append(result)
    roomy = dg.build_digest(results, "de", max_per_side=2, now=NOW)
    assert "Beobachten (nicht belegt)" in roomy.body and len(roomy.body) <= dg.BODY_MAX
    tight = dg.build_digest(results, "de", max_per_side=8, now=NOW)
    assert len(tight.body) <= dg.BODY_MAX
    assert "Beobachten (nicht belegt)" not in tight.body          # watch goes first
    assert "LONG US1" in tight.body and "Keine Anlageberatung." in tight.body
    # even a hopelessly long input stays within the limit and still ends in valid text
    many = dg.build_digest(results * 3, "de", max_per_side=8, now=NOW)
    assert len(many.body) <= dg.BODY_MAX


@pytest.mark.unit
def test_deterministic_for_the_same_input_and_now():
    a = dg.build_digest(_four(), "de", now=NOW)
    b = dg.build_digest(_four(), "de", now=NOW)
    assert a == b
    assert dg.build_digest(_four(), "de", now=datetime(2026, 11, 14, 9, 0)).body != a.body
    assert dg.build_digest(_four(), "en", now=NOW).body != a.body


@pytest.mark.unit
def test_english_digest_has_no_german_and_german_has_no_english():
    en = dg.build_digest(_four(), "en", now=NOW)
    assert en.body.startswith("Fri 2026-11-13 16:05 · prices up to 2025-06-30")
    for word in ("Einstieg", "Keine Anlageberatung", "Beobachten", "Regeln", "Empfehlung"):
        assert word not in en.body + en.headline
    de = dg.build_digest(_four(), "de", now=NOW)
    for word in ("Entry", "no recommendation", "Watch", "rules tested", "Not investment advice",
                 "Validated", "prices up to"):
        assert word not in de.body + de.headline
    assert not re.search(r"(?<![\d.,])\d+\.\d+(?![\d.,])", de.body)


@pytest.mark.unit
def test_horizon_filter_and_missing_idea_layer():
    d = dg.build_digest([evidenced_result()], "de", now=NOW, horizons=["day"])
    assert "LONG UP2" not in d.body and "Daytrade: keine Empfehlung" in d.body
    assert d.headline.startswith("Keine belegte Empfehlung")
    result = evidenced_result()
    result.ideas = None
    d = dg.build_digest([result], "de", now=NOW)
    assert "Ideen nicht verfügbar, keine Empfehlung" in d.body
    assert dg.build_digest([], "de", now=NOW).headline == "Kein Scan-Ergebnis"
