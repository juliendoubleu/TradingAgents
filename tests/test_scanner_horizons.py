"""Ideas per horizon with an honest evidence label: selection, fallbacks, notes, wording."""

import json

import numpy as np
import pytest

from tests.scanner_helpers import END, frame, rising
from tradingagents.scanner import horizons as hz, rules, scan as sc
from tradingagents.scanner.scan import analyse
from tradingagents.scanner.shortterm import live
from tradingagents.scanner.signals import SignalParams
from tradingagents.scanner.universe import universe_info

N = 520
NAMES = [f"S{i}" for i in range(10)]


def _frames(seed=2):
    rng = np.random.default_rng(seed)
    out = {}
    for i, name in enumerate(NAMES):
        steps = rng.normal(0.0014 * (i - 4.5) / 4.5, 0.005, N)
        out[name] = frame(100 * np.exp(np.cumsum(steps)))
    out["SPY"] = rising(N, 0.0004)
    out["^VIX"] = frame(np.full(N, 15.0))
    return out


def _setup(universe="us"):
    frames = _frames()
    info = universe_info(universe, NAMES)
    regime, signals, _ = analyse(NAMES, frames, "SPY", params=SignalParams(), use_vix=True)
    return info, frames, signals, regime


def _entry(rule, horizon, side, verdict, pooled_t=2.0, oos_t=2.3, oos_mean=60.0, family="stocks_us"):
    judged = "long_excess" if side == "long" else "short_pnl"
    block = {"IS": {"n": 90, "mean_bps": 40.0, "nw_t": 2.1},
             "OOS": {"n": 44, "mean_bps": oos_mean, "nw_t": oos_t, "pos_share": 0.6,
                      "ci_low_bps": 5.0, "ci_high_bps": 110.0},
             "ALL": {"n": 130, "mean_bps": 45.0, "nw_t": pooled_t}}
    return {"family": family, "rule": rule, "horizon": horizon, "side": side, "verdict": verdict,
            "strong": False, "judged": judged, "pooled_t": pooled_t,
            "series": {judged: block, "short_excess": block, "long_excess": block,
                       "long_abs": block}}


def _snap(entries, family="stocks_us"):
    return {
        "study_date": "2026-11-13",
        "rules": {r: {"name_en": f"Rule {r}", "name_de": f"Regel {r}",
                      "variant": r.startswith("V")} for r in ("T0", "T1", "T2", "T3", "V1")},
        "families": {family: {"horizons": {h: {"oos_first": "2023-01-17", "oos_last": "2026-10-08"}
                                           for h in ("medium", "long")}}},
        "results": entries,
    }


QUIET_SHORT = {"study_date": "2026-11-12", "splits": {"1d": {"oos_first": "2023-11-10",
                                                              "oos_last": "2026-11-12"}},
               "strategies": [{"id": "D1", "horizon": "day", "timeframe": "60m"},
                              {"id": "D2", "horizon": "day", "timeframe": "5m"},
                              {"id": "S1", "horizon": "swing", "timeframe": "1d",
                               "verdict": "rejected"}]}


def _build(entries, **kw):
    info, frames, signals, regime = _setup(kw.pop("universe", "us"))
    return build_ideas_for(info, frames, signals, regime, entries, **kw)


def build_ideas_for(info, frames, signals, regime, entries, short=QUIET_SHORT, top=5,
                    family="stocks_us", snap=None):
    return hz.build_ideas(info, frames, signals, regime,
                          evidence_snapshot=snap if snap is not None else _snap(entries, family),
                          shortterm_snapshot=short, top=top)


@pytest.mark.unit
def test_picks_the_best_evidenced_rule_per_side_and_horizon():
    entries = [
        _entry("T1", "medium", "long", "experimental", pooled_t=3.4),
        _entry("T2", "medium", "long", "validated", pooled_t=2.2),
        _entry("V1", "medium", "long", "validated", pooled_t=3.9),   # a variant: after T2
        _entry("T3", "medium", "long", "unproven", pooled_t=5.0),
    ]
    snap = _snap(entries)
    assert hz.best_entry(snap, "stocks_us", "medium", "long")["rule"] == "T2"
    assert hz.best_entry(snap, "stocks_us", "medium", "short") is None
    snap["results"] = [_entry("T1", "long", "long", "experimental", 2.0),
                       _entry("T2", "long", "long", "experimental", 2.8)]
    assert hz.best_entry(snap, "stocks_us", "long", "long")["rule"] == "T2"   # ties: pooled t
    snap["results"] = [_entry("T1", "long", "short", "relative", 1.0),
                       _entry("T2", "long", "short", "experimental", 0.5)]
    assert hz.best_entry(snap, "stocks_us", "long", "short")["rule"] == "T2"


@pytest.mark.unit
def test_validated_rule_members_come_from_the_same_rule_code_as_the_study():
    info, frames, signals, regime = _setup()
    out = build_ideas_for(info, frames, signals, regime,
                          [_entry("T1", "medium", "long", "validated")], top=3)
    ideas = out.select("medium", "Long")
    panel = rules.Panel(frames, NAMES, "SPY", SignalParams())
    expect = [m[0] for m in rules.current_members("T1", panel) if m[1] == 1][:3]
    assert [i.symbol for i in ideas] == expect and len(ideas) == 2      # 20 % of 10 names
    first = ideas[0]
    assert first.rule_id == "T1" and first.evidence.label == "validated"
    assert first.watch_only is False and first.stance == "recommended"
    assert first.reference_date == str(panel.index[-1].date())
    assert first.reference_price == pytest.approx(float(panel.close[first.symbol].iloc[-1]))
    atr = float(panel.atr()[first.symbol].iloc[-1])
    assert first.stop == pytest.approx(first.reference_price - 2.5 * atr)      # medium stop
    assert first.target is None and first.entry["de"] == "zur nächsten Eröffnung"
    assert first.size_pct == pytest.approx(min(10.0, 1.0 / (2.5 * atr / first.reference_price)))
    assert first.holding["de"].startswith("Mittelfristig")
    assert any(code == "rule_momentum" for code, _ in first.reason_items)
    assert first.reasons_text("de") and first.reasons_text("en")


@pytest.mark.unit
def test_long_horizon_uses_the_wider_stop_and_the_half_year_holding():
    info, frames, signals, regime = _setup()
    out = build_ideas_for(info, frames, signals, regime,
                          [_entry("T1", "long", "long", "validated")], top=2)
    idea = out.select("long", "Long")[0]
    panel = rules.Panel(frames, NAMES, "SPY", SignalParams())
    atr = float(panel.atr()[idea.symbol].iloc[-1])
    assert idea.stop == pytest.approx(idea.reference_price - 4.0 * atr)
    assert "6 Monate" in idea.holding["de"]


@pytest.mark.unit
def test_without_evidence_the_composite_becomes_an_unproven_watch_list():
    info, frames, signals, regime = _setup()
    out = build_ideas_for(info, frames, signals, regime,
                          [_entry("T0", "medium", "long", "unproven", oos_t=-0.4, oos_mean=-5.0)])
    for horizon in ("medium", "long"):
        for ideas in (out.select(horizon, "Long"), out.select(horizon, "Short")):
            assert ideas, (horizon, "composite names exist in this market")
            assert all(i.watch_only and i.evidence.label == "unproven" and i.rule_id == "T0"
                       and i.stance == "watch" for i in ideas)
    medium_long = out.select("medium", "Long")[0]
    assert medium_long.evidence.oos_t == pytest.approx(-0.4)          # the composite's own numbers
    assert "Nicht belegt" in medium_long.evidence.text("de")
    assert out.recommended() == []
    codes = [(n.code, n.params["horizon"], n.params["side"]) for n in out.notes]
    assert ("idea_no_rec_rules", "medium", "long") in codes
    assert ("idea_no_rec_rules", "long", "short") in codes
    composite = [s.symbol for s in sorted(
        (s for s in signals if s.direction == "Long"), key=lambda s: -s.score)][:5]
    assert [i.symbol for i in out.select("medium", "Long")] == composite


@pytest.mark.unit
def test_a_relative_short_is_shown_as_avoid_and_never_as_a_recommendation():
    info, frames, signals, regime = _setup()
    out = build_ideas_for(info, frames, signals, regime,
                          [_entry("T1", "medium", "short", "relative", oos_mean=45.0)])
    shorts = out.select("medium", "Short")
    assert shorts and all(i.watch_only and i.stance == "avoid"
                          and i.evidence.label == "relative" for i in shorts)
    note = next(n for n in out.notes if n.code == "idea_relative_only")
    assert "meiden" in note.text("de") and "avoid" in note.text("en")
    assert "meiden" in shorts[0].evidence.text("de")


@pytest.mark.unit
def test_swing_idea_that_only_earns_the_drift_is_labelled_beta(monkeypatch):
    idea = live.ShortTermIdea(
        symbol="S3", direction="Long", horizon="swing", strategy_id="S3l",
        strategy_name_en="Volume breakout, long only", strategy_name_de="Ausbruch, nur Long",
        entry_rule_en="at the next open", entry_rule_de="zur nächsten Eröffnung",
        reference_price=100.0, stop=95.0, target=None, holding_en="up to 10 trading days",
        holding_de="bis zu 10 Handelstage", oos_win_rate=0.52, oos_mean_net_bps=67.0,
        oos_n_trades=655, oos_t_stat=2.9, verdict="validated", study_date="2026-11-12",
        beats_drift=False, oos_excess_bps=-1.8, oos_excess_t_stat=0.8, label="beta")
    monkeypatch.setattr(live, "shortterm_ideas", lambda *a, **k: [idea])
    short = {**QUIET_SHORT, "strategies": [
        {"id": "S3l", "horizon": "swing", "timeframe": "1d", "verdict": "validated",
         "beats_drift": False}]}
    info, frames, signals, regime = _setup()
    out = build_ideas_for(info, frames, signals, regime, [], short=short)
    swing = out.select("swing")
    assert len(swing) == 1 and swing[0].evidence.label == "beta" and swing[0].watch_only
    assert swing[0].size_pct == pytest.approx(10.0 if 1.0 / 0.05 > 10 else 1.0 / 0.05)
    text = swing[0].evidence.text("de")
    assert "Markt-Beta" in text and "655" in text
    note = next(n for n in out.notes if n.code == "idea_beta_only")
    assert "S3l" in note.text("de") and note.params["horizon"] == "swing"


@pytest.mark.unit
def test_day_and_swing_notes_state_the_numbers_of_the_study():
    info, frames, signals, regime = _setup()
    out = build_ideas_for(info, frames, signals, regime, [])
    day = next(n for n in out.notes if n.code == "idea_no_rec_short_term"
               and n.params["horizon"] == "day")
    assert day.params["n"] == 2 and day.params["date"] == "2026-11-12"
    assert "Studie vom 12.11.2026" in day.text("de") and "keine Empfehlung" in day.text("de")
    assert "60-Minuten- und 5-Minuten-Kursen" in day.text("de")
    assert out.select("day") == [] and out.select("swing") == []


@pytest.mark.unit
def test_other_families_are_not_studied_for_short_term_and_the_note_says_so():
    info, frames, signals, regime = _setup("dax")
    out = build_ideas_for(info, frames, signals, regime, [], family="stocks_dax")
    assert out.family == "stocks_dax"
    codes = {(n.code, n.params["horizon"]) for n in out.notes}
    assert ("idea_not_studied", "day") in codes and ("idea_not_studied", "swing") in codes
    assert out.select("day") == [] and out.select("swing") == []


@pytest.mark.unit
def test_missing_snapshot_never_raises_and_everything_is_not_studied(tmp_path, monkeypatch):
    info, frames, signals, regime = _setup()
    out = hz.build_ideas(info, frames, signals, regime, evidence_snapshot={},
                         shortterm_snapshot=QUIET_SHORT)
    assert out.evidence_available is False and out.recommended() == []
    medium = out.select("medium")
    assert medium and all(i.evidence.label == "not_studied" and i.watch_only for i in medium)
    assert any(n.code == "idea_no_snapshot" for n in out.notes)
    assert "Nicht untersucht" in medium[0].evidence.text("de")
    monkeypatch.setattr(hz, "SNAPSHOT_PATH", tmp_path / "absent.json")
    again = hz.build_ideas(info, frames, signals, regime, shortterm_snapshot=QUIET_SHORT)
    assert again.evidence_available is False
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert hz.load_snapshot(tmp_path / "bad.json") is None
    json.dumps(out.to_dict())


@pytest.mark.unit
def test_a_mixed_universe_is_not_studied():
    info, frames, signals, regime = _setup("all")
    out = hz.build_ideas(info, frames, signals, regime, evidence_snapshot=_snap([]),
                         shortterm_snapshot=QUIET_SHORT)
    assert out.family == "mixed"
    assert all(i.evidence.label == "not_studied" for i in out.select("medium"))


@pytest.mark.unit
def test_a_validated_rule_that_selects_nobody_gets_a_note_not_filler():
    info, frames, signals, regime = _setup()
    # T3's short side needs the (rising) benchmark below its average: no member today
    out = build_ideas_for(info, frames, signals, regime,
                          [_entry("T3", "medium", "short", "validated")])
    assert out.select("medium", "Short") == []
    note = next(n for n in out.notes if n.code == "idea_no_members")
    assert note.params["rule"] == "T3" and "wählt aktuell kein" in note.text("de")


@pytest.mark.unit
def test_evidence_wording_matches_each_label():
    base = {"rule_id": "T1", "rule_name": {"en": "Momentum 12-1", "de": "Momentum 12-1"},
            "family": "stocks_us", "horizon": "medium", "study_date": "2026-11-13",
            "oos_n": 612, "oos_mean_bps": 42.0, "oos_t": 2.6, "is_t": 2.3,
            "window": ("2014-03-01", "2026-10-08")}
    v = hz.Evidence(label="validated", judged="long_excess", **base)
    assert v.text("de") == ("Belegt: im Test (2014–2026, 612 Termine) lag diese Regel 0,42 % pro "
                            "Monat vor dem Durchschnitt, t = 2,6")
    assert v.text("en").startswith("Validated: in the test (2014–2026, 612 dates)")
    assert "0.42 %" in v.text("en")
    short = hz.Evidence(label="validated", judged="short_pnl", **base)
    assert "Short-Korb" in short.text("de")
    exp = hz.Evidence(label="experimental", judged="long_excess", **base)
    assert "Schwach belegt" in exp.text("de") and "Zufall" in exp.text("de")
    rel = hz.Evidence(label="relative", judged="short_excess", **base)
    assert "meiden" in rel.text("de") and "avoid" in rel.text("en")
    beta = hz.Evidence(label="beta", judged="long_abs", excess_mean_bps=-2.0, excess_t=0.3, **base)
    assert "Markt-Beta" in beta.text("de") and "-0,02 %" in beta.text("de")
    flat = hz.Evidence(label="unproven", judged="long_excess",
                       **{**base, "oos_t": 0.3, "oos_mean_bps": 4.0})
    assert flat.text("de").startswith("Nicht belegt: im Test (2014–2026, 612 Termine) kein messbarer")
    assert "t = 0,3" in flat.text("de")
    luck = hz.Evidence(label="unproven", judged="long_excess", **{**base, "is_t": 0.4})
    assert "Auswahlzeitraum" in luck.text("de") and "nicht belegt" in luck.text("de")
    none = hz.Evidence(label="unproven", family="stocks_us", horizon="long", n_rules_tested=7)
    assert "7 Regeln geprüft" in none.text("de") and "7 rules tested" in none.text("en")
    assert hz.Evidence(label="not_studied").text("en").startswith("Not studied")
    long_h = hz.Evidence(label="validated", judged="long_excess", **{**base, "horizon": "long"})
    assert "pro Halbjahr" in long_h.text("de")
    assert json.dumps(v.to_dict()) and v.to_dict()["text_de"] == v.text("de")


@pytest.mark.unit
def test_crypto_holdings_are_stated_in_calendar_days():
    assert "30 Kalendertage" in hz.holding_text("medium", "crypto", "de")
    assert "180 calendar days" in hz.holding_text("long", "crypto", "en")
    assert hz.holding_text("day", "stocks_us", "de") == "Daytrade – Glattstellung bis Handelsschluss"
    assert hz.HORIZONS["swing"].holding["de"] == "Kurzfristig (Swing) – 2 bis 10 Handelstage"
    assert set(hz.HORIZONS) == {"day", "swing", "medium", "long"}


@pytest.mark.unit
def test_idea_set_serialises_to_json_with_both_languages():
    info, frames, signals, regime = _setup()
    out = build_ideas_for(info, frames, signals, regime,
                          [_entry("T1", "medium", "long", "validated")])
    d = json.loads(json.dumps(out.to_dict()))
    assert set(d["horizons"]) == {"day", "swing", "medium", "long"}
    idea = next(i for i in d["ideas"] if i["horizon"] == "medium" and i["direction"] == "Long")
    assert idea["evidence"]["label"] == "validated" and idea["evidence"]["text_de"]
    assert idea["reasons"] and idea["reasons_de"] and idea["watch_only"] is False
    assert all({"code", "text", "text_de", "horizon", "side"} <= set(n) for n in d["notes"])


@pytest.mark.unit
def test_run_scan_carries_the_ideas_and_survives_a_failing_idea_layer(monkeypatch):
    frames = _frames()

    def load(symbols, as_of, lookback):
        return {s: frames[s] for s in symbols if s in frames}
    res = sc.run_scan("us", END, tickers=NAMES, loader=load)
    assert res.ideas is not None and res.ideas.family == "stocks_us"
    assert res.to_dict()["ideas"]["ideas"] is not None and json.dumps(res.to_dict())
    assert res.longs is not None                                     # composite stays

    def boom(*a, **k):
        raise RuntimeError("idea layer down")
    monkeypatch.setattr(sc, "build_ideas", boom)
    res = sc.run_scan("us", END, tickers=NAMES, loader=load)
    assert res.ideas is None and res.to_dict()["ideas"] is None and res.signals
