"""Fixes of the final acceptance review (F2): one test per finding, all offline."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import cli.main as m
from tests.scanner_helpers import END, flat, rising
from tests.scanner_idea_helpers import empty_result, evidenced_result
from tests.test_scanner_assess import NOW, fake_scan, idea, loader, signal
from tradingagents.scanner import (
    assess as A,
    digest as dg,
    horizon_research as hr,
    horizons as hz,
    notices,
    report as rp,
    scan as sc,
)

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- M1: rule counts


@pytest.mark.unit
def test_rule_counts_name_both_day_and_swing():
    assert rp._rule_counts({"swing": 9, "day": 15}, "de") == "9 Swing- und 15 Daytrade-Regeln (24)"
    assert rp._rule_counts({"swing": 9, "day": 15}, "en") == "9 swing and 15 day-trade rules (24)"
    assert rp._rule_counts({"day": 24}, "de") == "24 Daytrade-Regeln"


# --------------------------------------------------------------------------- one label name


@pytest.mark.unit
def test_label_names_come_from_one_source():
    expected = {"validated": "Belegt", "experimental": "Schwach belegt",
                "relative": "Nur relativ schwächer", "beta": "Nur Marktrendite",
                "unproven": "Nicht belegt", "not_studied": "Nicht untersucht"}
    for label, de in expected.items():
        assert notices.label_name(label, "de") == de
        assert rp.label_text(label, "de") == de
        assert A.label_text(label, "de") == de.lower()
        assert notices.label_name(label, "en") == rp.label_text(label, "en")
    for label, verdict in hr.VERDICT_DE.items():
        assert verdict == expected[label][0].lower() + expected[label][1:]
    # the Evidence sentence opens with the same name
    ev = hz.Evidence(label="beta", judged="long_abs", horizon="medium", oos_n=10,
                     oos_mean_bps=40.0, oos_t=1.0, excess_mean_bps=1.0, excess_t=0.1)
    assert ev.text("de").startswith("Nur Marktrendite:")
    for doc in ("README.md", "CLAUDE.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        assert "Markt-Beta" not in text, doc


# --------------------------------------------------------------------------- M4: watch wording


@pytest.mark.unit
def test_watch_direction_is_never_a_call_word():
    assert hz.watch_direction("Long", "de") == "technisch stark"
    assert hz.watch_direction("Short", "de") == "technisch schwach"
    assert hz.watch_direction("Short", "en") == "technically weak"


@pytest.mark.unit
def test_a_significantly_negative_result_carries_the_warning():
    base = {"rule_id": "T0", "rule_name": {"en": "Technical", "de": "Technisches Gesamtbild"},
            "family": "stocks_us", "horizon": "medium", "side": "short", "judged": "short_pnl",
            "study_date": "2026-11-13", "oos_n": 60, "oos_mean_bps": -211.0, "is_t": -1.0,
            "window": ("2018-01-31", "2026-06-30")}
    bad = hz.Evidence(label="unproven", oos_t=-2.57, **base)
    assert "frühere Signale dieser Art verloren im Test deutlich" in bad.text("de")
    assert "earlier signals of this kind lost money in the test" in bad.text("en")
    assert "Short" not in bad.text("de")
    mild = hz.Evidence(label="unproven", **{**base, "oos_mean_bps": -20.0}, oos_t=-1.9)
    assert "frühere Signale" not in mild.text("de")


@pytest.mark.unit
def test_assess_and_ask_show_no_call_words_on_a_watch_item(monkeypatch):
    from tradingagents.scanner import ask as K

    shorts = [idea("TSLA", "medium", "Short"), idea("TSLA", "medium", "Long")]
    monkeypatch.setattr(A, "run_scan", lambda *a, **k: fake_scan(
        [signal("TSLA", -60.0)], shorts)(*a, **k))
    a = A.assess(["TSLA"], as_of_date=END, loader=loader, now=NOW)
    text = A.render_assessment(a, "de")
    assert "technisch schwach" in text or "technisch stark" in text
    assert "nur Beobachtung, Short" not in text and "nur Beobachtung, Long" not in text
    assert "Short-Beobachtungsliste" not in text
    ans = K.answer_question("Lohnt sich ein Short auf Tesla?", as_of_date=END, loader=loader,
                            now=NOW).text
    assert "Short-Beobachtung" not in ans and "Short watch" not in ans


# --------------------------------------------------------------------------- M5: median phase


@pytest.mark.unit
def test_snapshot_keeps_every_phase_value_and_the_mean_of_the_judged_phase():
    import numpy as np

    rng = np.random.default_rng(3)
    x = rng.normal(0.004, 0.02, 40)
    s = hr.series_stats(x, 6, 1, boot=False)
    assert len(s["phase_t"]) == 6 == s["phases"]
    # the mean next to the judged t belongs to the same (median) phase(s)
    order = np.argsort([hr.plain_tstat(x[o::6]) for o in range(6)], kind="stable")
    mid = order[2:4]
    assert s["mean_nonoverlap_bps"] == pytest.approx(
        float(np.mean([x[o::6].mean() for o in mid])) * 1e4)
    ev = hz._evidence_from_entry(
        {"families": {"stocks_dax": {"horizons": {"medium": {}}}}},
        {"family": "stocks_dax", "horizon": "medium", "rule": "T4", "side": "long",
         "judged": "long_excess", "verdict": "unproven",
         "series": {"long_excess": {"OOS": {**s, "n": 40}, "IS": s}}})
    assert ev.oos_mean_bps == pytest.approx(s["mean_nonoverlap_bps"])      # not the overlapping mean
    assert ev.oos_t == pytest.approx(s["t_nonoverlap"])


@pytest.mark.unit
def test_packaged_snapshot_has_phase_values_and_the_report_names_them():
    snap = json.loads((ROOT / "tradingagents/scanner/evidence.json").read_text(encoding="utf-8"))
    assert snap["version"] == hr.SNAPSHOT_VERSION
    r = next(x for x in snap["results"] if x["family"] == "stocks_us" and x["rule"] == "T1"
             and x["horizon"] == "long" and x["side"] == "long")
    ser = r["series"][r["judged"]]
    assert len(ser["IS"]["phase_t"]) == 6 and len(r["phase_verdicts"]) == 6
    assert r["verdict"] == "unproven"


# --------------------------------------------------------------------------- M7: digest failure


@pytest.mark.unit
def test_total_failure_prints_one_line_to_stdout_and_exits_1(monkeypatch):
    from tradingagents.dataflows.errors import VendorUnavailableError

    def down(*a, **k):
        raise VendorUnavailableError("Yahoo Finance returned no price data for any of 63 symbols")
    monkeypatch.setattr("tradingagents.scanner.scan.load_bulk_ohlcv", down)
    runner = CliRunner(mix_stderr=False) if "mix_stderr" in CliRunner.__init__.__code__.co_varnames \
        else CliRunner()
    res = runner.invoke(m.app, ["scan", "--universe", "us,dax", "--digest", "--no-save",
                                "--lang", "de"])
    assert res.exit_code == 1
    assert res.stdout.strip() == "Markt-Scan fehlgeschlagen: Kursdaten nicht abrufbar – kein Digest"
    assert "returned no" not in res.output and "No price data" not in res.output
    assert "Yahoo Finance lieferte für keines der 63 Symbole Kursdaten" in res.output
    en = runner.invoke(m.app, ["scan", "--universe", "us", "--digest", "--no-save", "--lang", "en"])
    assert en.exit_code == 1 and "Market scan failed: price data unavailable" in en.stdout


# --------------------------------------------------------------------------- digest wording


def _watch_results(names=("us", "dax", "global", "crypto"), symbols=None):
    out = []
    for name in names:
        syms = (symbols or {}).get(name, ("UP2", "UP1", "UP0"))
        r = empty_result(name, "stocks_us")
        r.universe, r.custom = name, False
        from tests.scanner_idea_helpers import base_result
        base = base_result(name)
        r.longs = base.longs[:len(syms)]
        for sig, sym in zip(r.longs, syms, strict=False):
            sig.symbol = sym
        out.append(r)
    return out


@pytest.mark.unit
def test_headline_says_technically_strong_with_short_names_and_fits():
    results = _watch_results(symbols={"us": ("AAPL", "JNJ", "XOM"), "dax": ("DB1.DE", "DHL.DE"),
                                      "crypto": ("BTC-USD", "ETH-USD")})
    d = dg.build_digest(results, "de", now=datetime(2026, 11, 13, 16, 5))
    assert d.headline.startswith(
        "Keine belegte Empfehlung · technisch stark (nicht belegt): US AAPL, JNJ · DAX DB1, DHL")
    assert ".DE" not in d.headline and "-USD" not in d.headline and "Krypto BTC, ETH" in d.headline
    assert len(d.headline) <= dg.HEADLINE_MAX
    en = dg.build_digest(results, "en", now=datetime(2026, 11, 13, 16, 5))
    assert "technically strong (not proven): US AAPL, JNJ" in en.headline


@pytest.mark.unit
def test_headline_is_shortened_to_180_characters():
    names = ("us", "dax", "global", "crypto")
    results = _watch_results(names, dict.fromkeys(names, ("VERYLONGSYMBOL1", "VERYLONGSYMBOL2", "X")))
    d = dg.build_digest(results, "de", now=datetime(2026, 11, 13, 16, 5))
    assert len(d.headline) <= 180 and d.headline.startswith("Keine belegte Empfehlung")


@pytest.mark.unit
def test_sizes_are_explained_only_when_a_size_is_shown():
    now = datetime(2026, 11, 13, 16, 5)
    sized = dg.build_digest([evidenced_result()], "de", now=now)
    assert "addieren sich nicht" in sized.body
    plain = dg.build_digest([empty_result()], "de", now=now)
    assert "addieren sich nicht" not in plain.body
    assert plain.body.rstrip().endswith("Keine Anlageberatung.")
    md = rp.render_markdown(empty_result(), lang="de")
    assert "Die Größen gelten je Trade" not in md
    assert "Die Größen gelten je Trade" in rp.render_markdown(evidenced_result(), lang="de")


@pytest.mark.unit
def test_weekend_title_on_saturday_and_sunday_but_not_for_crypto_only():
    sat = datetime(2026, 11, 14, 10, 0)
    d = dg.build_digest(_watch_results(("us", "crypto")), "de", now=sat)
    assert "Sa 14.11.2026 11:00 MEZ · Wochenende – Börsen geschlossen" in d.body.split("\n")[0]
    sun = dg.build_digest(_watch_results(("us",)), "de", now=datetime(2026, 11, 15, 9, 0))
    assert "Wochenende – Börsen geschlossen" in sun.body.split("\n")[0]
    crypto = dg.build_digest(_watch_results(("crypto",)), "de", now=sat)
    assert "Wochenende" not in crypto.body.split("\n")[0]
    fri = dg.build_digest(_watch_results(("us",)), "de", now=datetime(2026, 11, 13, 10, 0))
    assert "Wochenende" not in fri.body.split("\n")[0]
    # a timezone-aware clock is read in Berlin time: Friday 23:30 UTC is already Saturday there
    aware = dg.build_digest(_watch_results(("us",)), "de",
                            now=datetime(2026, 11, 13, 23, 30, tzinfo=UTC))
    assert "Wochenende" in aware.body.split("\n")[0]


# --------------------------------------------------------------------------- scan fixes


@pytest.mark.unit
def test_run_scan_with_no_date_means_today():
    frames = {"SPY": flat(), "^VIX": flat(start=15.0), "UP0": rising(daily=0.003, wiggle=0.004)}
    res = sc.run_scan("us", None, tickers=["UP0"], loader=lambda s, d, n: {
        k: v for k, v in frames.items() if k in s},
        now=datetime(2025, 7, 1, 21, tzinfo=UTC))
    assert res.requested_date == datetime.now().strftime("%Y-%m-%d")


@pytest.mark.unit
def test_benchmark_and_vix_are_cut_at_the_members_newest_bar():
    members = {"UP0": rising(daily=0.003, wiggle=0.004), "UP1": rising(daily=0.002, wiggle=0.004)}
    one_day = pd.bdate_range(end=END, periods=2)[-1] + pd.offsets.BDay(1)
    bench = flat()
    extra = pd.DataFrame(bench.iloc[[-1]].to_numpy(), columns=bench.columns, index=[one_day])
    bench_long = pd.concat([bench, extra])
    bench_long.iloc[-1, bench_long.columns.get_loc("Close")] = 50.0       # a crash a day later
    vix_long = pd.concat([flat(start=15.0), pd.DataFrame(
        flat(start=15.0).iloc[[-1]].to_numpy(), columns=bench.columns, index=[one_day])])
    vix_long.iloc[-1, vix_long.columns.get_loc("Close")] = 80.0
    frames = {**members, "SPY": bench_long, "^VIX": vix_long}
    regime, _, _ = sc.analyse(list(members), frames, "SPY")
    clean, _, _ = sc.analyse(list(members), {**members, "SPY": flat(), "^VIX": flat(start=15.0)},
                             "SPY")
    assert regime.points == clean.points and regime.label == clean.label


@pytest.mark.unit
def test_unsettled_notice_wording():
    n = notices.Notice("running_bar_removed", n=3, total=10)
    assert "Börsenschluss plus 20 Minuten Puffer" in n.text("de")
    assert "Börsensitzung war noch nicht beendet" not in n.text("de")


@pytest.mark.unit
def test_parse_date_errors_are_german_under_de():
    with pytest.raises(ValueError, match="JJJJ-MM-TT"):
        sc.parse_date("Juni", lang="de")
    with pytest.raises(ValueError, match="liegt in der Zukunft"):
        sc.parse_date("2999-01-01", lang="de")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        sc.parse_date("June")


# --------------------------------------------------------------------------- CLI messages


@pytest.fixture
def app():
    import typer

    from cli.scanner_commands import register

    a = typer.Typer()
    register(a)
    return a


@pytest.mark.unit
def test_scan_bad_date_is_german(monkeypatch):
    res = CliRunner().invoke(m.app, ["scan", "--tickers", "AAA", "--date", "Juni", "--no-save",
                                     "--lang", "de"])
    assert res.exit_code == 1 and "JJJJ-MM-TT" in res.output and "YYYY" not in res.output


@pytest.mark.unit
def test_ask_event_study_and_implied_move_errors_are_german(app, monkeypatch):
    runner = CliRunner()
    res = runner.invoke(app, ["ask", "", "--lang", "de"])
    assert res.exit_code == 1 and "Keine Frage angegeben" in res.output
    res = runner.invoke(app, ["event-study", "TTWO", "--dates", "Juni:Test", "--lang", "de"])
    assert res.exit_code == 1 and "Ungültiges Datum" in res.output and "bad date" not in res.output
    res = runner.invoke(app, ["implied-move", "TTWO", "--expiries", "0", "--lang", "de"])
    assert res.exit_code == 1 and "mindestens 1" in res.output


@pytest.mark.unit
def test_unknown_symbols_exit_1_in_assess_and_implied_move(app, monkeypatch):
    from tradingagents.dataflows.errors import NoMarketDataError

    # assess: a ticker without data in every scan -> nothing could be assessed
    monkeypatch.setattr(A, "run_scan", fake_scan([], []))
    monkeypatch.setattr(A, "performance_snapshot", lambda *a, **k: [])
    res = CliRunner().invoke(app, ["assess", "XXXXQQ", "--lang", "de"])
    assert res.exit_code == 1 and "Keines der Wertpapiere" in res.output
    # implied-move: Yahoo does not know the symbol -> an error, not "no listed options"
    import tradingagents.dataflows.vendors.yahoo.options as opt

    def unknown(symbol, max_expiries=4, now=None):
        raise NoMarketDataError(symbol, symbol, "no price history")
    monkeypatch.setattr(opt, "load_implied_moves", unknown)
    res = CliRunner().invoke(app, ["implied-move", "XXXXQQ", "--lang", "de"])
    assert res.exit_code == 1 and "keine Kursdaten gefunden" in res.output


# --------------------------------------------------------------------------- events / options


@pytest.mark.unit
def test_event_study_table_has_the_same_columns_as_the_summary():
    from tests.test_scanner_events import _study
    from tradingagents.scanner.events import render_event_study

    study = _study([(pd.bdate_range("2026-01-05", periods=120)[80].strftime("%Y-%m-%d"), "e")])
    text = render_event_study(study, "de")
    header = next(line for line in text.split("\n") if line.startswith("| Ereignis"))
    assert "+10T" in header
    summary_rows = [line for line in text.split("\n") if line.startswith("| +")]
    assert any(line.startswith("| +10T") for line in summary_rows)


@pytest.mark.unit
def test_implied_move_marks_an_expiry_that_runs_out_today():
    from tradingagents.dataflows.vendors.yahoo.options import ImpliedMove
    from tradingagents.scanner.events import render_implied_moves

    mv = ImpliedMove("AAA", "2026-11-13", 0, 100.0, 100.0, 4.0, 5.0, 9.0, 0.09, "mid", False,
                     expires_today=True)
    assert "läuft heute aus" in render_implied_moves([mv], "de")
    assert "expires today" in render_implied_moves([mv], "en")


# --------------------------------------------------------------------------- assess summary


@pytest.mark.unit
def test_summary_collapses_equal_statements_and_ranks_only_horizons_with_ideas(monkeypatch):
    monkeypatch.setattr(A, "run_scan", fake_scan(
        [signal("NVDA", -18.0), signal("SAP.DE", 62.0)],
        [idea("NVDA", "long"), idea("SAP.DE", "long")]))
    a = A.assess(["NVDA", "SAP.DE"], as_of_date=END, loader=loader, now=NOW)
    lines = A.collapse_summary(a.summary_items, "de")
    # day, swing and medium have no idea for either name: one line, no ranking
    joined = next(line for line in lines if "Mittelfristig" in line)
    assert joined.startswith("Daytrade, Kurzfristig (Swing), Mittelfristig:") or \
        "Mittelfristig" in joined.split(":")[0]
    assert "stärker" not in joined
    assert len([x for x in lines if "Für keinen der beiden Werte" in x]) <= 2


# --------------------------------------------------------------------------- short-term report


@pytest.mark.unit
def test_shortterm_report_can_be_rebuilt_from_the_snapshot_and_matches_the_program():
    from tradingagents.scanner.shortterm import research as sr

    snap = json.loads((ROOT / "tradingagents/scanner/shortterm/validated.json").read_text(
        encoding="utf-8"))
    study = sr.study_from_snapshot(snap)
    assert study.as_of == snap["study_date"] and len(study.specs) == len(snap["strategies"])
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = sr.write_report(study, Path(tmp) / "r.md")
        text = out.read_text(encoding="utf-8")
    assert "Nur auf ausdrückliche Anfrage" not in text and "erzeugen **keine** Ideen" not in text
    assert "gekennzeichneten Beobachtungspunkt" in text or "keine Kurzfrist-Empfehlung" in text
    # the report never claims the program hides the market-return rules
    assert "standardmäßig keine" not in text
