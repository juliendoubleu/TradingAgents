"""Live ideas: only validated strategies speak, with the study's own signal functions."""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from tradingagents.scanner.shortterm import daytrade as dt, evaluate as ev, live, research, swing


def _pullback_frame(end=None):
    n = 330
    close = 100 * np.exp(np.cumsum(np.full(n, 0.0015))) * (1 + 0.002 * np.sin(np.arange(n) * 0.9))
    close[-3:] = close[-4] * np.array([0.985, 0.97, 0.955])
    open_ = np.r_[close[0], close[:-1]]
    idx = pd.bdate_range("2022-01-03", periods=n) if end is None \
        else pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.003,
                         "Low": np.minimum(open_, close) * 0.997, "Close": close,
                         "Volume": 1e6}, index=idx)


def _snapshot(verdict, beats_drift=True, study_date="2026-11-12"):
    return {
        "study_date": study_date,
        "strategies": [{
            "id": "S1", "family": "S1", "horizon": "swing", "timeframe": "1d",
            "name_en": "RSI(2) pullback", "name_de": "RSI(2)-Rücksetzer",
            "params": {"lo": 10, "hi": 90, "hold": 5, "stop_atr": 2.5},
            "verdict": verdict, "strong": False, "beats_drift": beats_drift,
            "OOS": {"win_rate": 0.6, "mean_net_bps": 12.0, "n_trades": 300, "t_stat": 2.4},
        }],
    }


@pytest.mark.unit
def test_validated_strategy_produces_a_labelled_idea():
    df = _pullback_frame()
    ideas = live.shortterm_ideas({"AAA": df}, str(df.index[-1].date()), _snapshot("validated"))
    assert len(ideas) == 1
    idea = ideas[0]
    assert (idea.symbol, idea.direction, idea.horizon, idea.strategy_id) == ("AAA", "Long", "swing", "S1")
    assert idea.entry_rule_en == "at the next open" and idea.entry_rule_de == "zur nächsten Eröffnung"
    assert idea.reference_price == pytest.approx(df["Close"].iloc[-1])
    assert idea.stop < idea.reference_price and idea.target is None
    assert "5 trading days" in idea.holding_en and "5 Handelstage" in idea.holding_de
    assert (idea.oos_win_rate, idea.oos_mean_net_bps, idea.oos_n_trades, idea.oos_t_stat) == (
        0.6, 12.0, 300, 2.4)
    assert idea.verdict == "validated" and idea.study_date == "2026-11-12"
    assert idea.label == "validated" and idea.beats_drift is True
    assert idea.caveat_en is None and idea.caveat_de is None
    assert idea.snapshot_stale is False and idea.snapshot_age_days is not None


@pytest.mark.unit
def test_nothing_validated_means_no_ideas_and_experimental_only_on_request():
    df = _pullback_frame()
    as_of = str(df.index[-1].date())
    assert live.shortterm_ideas({"AAA": df}, as_of, _snapshot("rejected")) == []
    assert live.shortterm_ideas({"AAA": df}, as_of, _snapshot("experimental")) == []
    flagged = live.shortterm_ideas({"AAA": df}, as_of, _snapshot("experimental"),
                                   include_experimental=True)
    assert [i.verdict for i in flagged] == ["experimental"]
    assert [i.label for i in flagged] == ["experimental"]


@pytest.mark.unit
def test_a_passed_verdict_without_beating_the_drift_is_beta_and_silent_by_default():
    df = _pullback_frame()
    as_of = str(df.index[-1].date())
    beta = _snapshot("validated", beats_drift=False)
    assert live.shortterm_ideas({"AAA": df}, as_of, beta) == []
    assert live.shortterm_ideas({"AAA": df}, as_of, beta, include_experimental=True) == []
    ideas = live.shortterm_ideas({"AAA": df}, as_of, beta, include_beta=True)
    assert len(ideas) == 1
    idea = ideas[0]
    assert idea.label == "beta" and idea.verdict == "validated"      # the verdict stays as recorded
    assert idea.beats_drift is False
    assert "no measured skill" in idea.caveat_en
    assert "kein belegter Vorteil" in idea.caveat_de
    # an experimental verdict that does not beat the drift is beta as well
    beta_x = _snapshot("experimental", beats_drift=False)
    assert live.shortterm_ideas({"AAA": df}, as_of, beta_x, include_experimental=True) == []
    assert [i.label for i in live.shortterm_ideas(
        {"AAA": df}, as_of, beta_x, include_beta=True)] == ["beta"]


@pytest.mark.unit
def test_an_explicit_label_in_the_snapshot_wins_and_a_missing_flag_is_not_validated():
    df = _pullback_frame()
    as_of = str(df.index[-1].date())
    snap = _snapshot("validated", beats_drift=True)
    snap["strategies"][0]["label"] = "beta"
    assert live.shortterm_ideas({"AAA": df}, as_of, snap) == []
    legacy = _snapshot("validated")
    del legacy["strategies"][0]["beats_drift"]
    assert live.shortterm_ideas({"AAA": df}, as_of, legacy) == []


@pytest.mark.unit
def test_an_old_snapshot_marks_its_ideas_stale_with_a_caveat():
    df = _pullback_frame()
    as_of = str(df.index[-1].date())
    old = _snapshot("validated", study_date="2020-01-01")
    idea = live.shortterm_ideas({"AAA": df}, as_of, old)[0]
    assert idea.snapshot_stale is True and idea.snapshot_age_days > live.SNAPSHOT_STALE_DAYS
    assert "re-run" in idea.caveat_en and "neu rechnen" in idea.caveat_de
    fresh = _snapshot("validated", study_date=as_of)
    idea = live.shortterm_ideas({"AAA": df}, as_of, fresh)[0]
    assert idea.snapshot_stale is False and idea.snapshot_age_days == 0


@pytest.mark.unit
def test_the_signal_must_come_from_the_newest_settled_session():
    df = _pullback_frame(end="2026-11-12")                       # a Thursday
    snap = _snapshot("validated")
    after_close = datetime(2026, 11, 13, 9, 0, tzinfo=UTC)
    assert len(live.shortterm_ideas({"AAA": df}, "2026-11-12", snap, now=after_close)) == 1
    # Friday's bar does not exist yet but is not yet due either: Thursday's bar is current
    assert len(live.shortterm_ideas({"AAA": df}, "2026-11-13", snap, now=after_close)) == 1
    # two sessions ago is not "the next open": Friday 11-13 was a session nobody has a bar for
    assert live.shortterm_ideas({"AAA": df}, "2026-11-16", snap, now=after_close) == []
    assert live.shortterm_ideas({"AAA": df}, "2026-11-14", snap, now=after_close) == []
    assert live.shortterm_ideas({"AAA": df}, "2026-11-30", snap, now=after_close) == []
    # a bar of a session that is still running is not a signal
    running = datetime(2026, 11, 12, 18, 0, tzinfo=UTC)          # 13:00 New York
    ideas = live.shortterm_ideas({"AAA": df}, "2026-11-12", snap, now=running)
    assert not any(i.reference_price == pytest.approx(df["Close"].iloc[-1]) for i in ideas)
    # a long weekend does not hide a current bar (Friday bar, Monday holiday, Tuesday as_of)
    fri = _pullback_frame(end="2026-09-04")                      # Friday before Labor Day
    late = datetime(2026, 9, 9, 9, 0, tzinfo=UTC)
    assert len(live.shortterm_ideas({"AAA": fri}, "2026-09-08", snap, now=late)) == 1


@pytest.mark.unit
def test_a_quiet_symbol_and_a_stale_bar_give_no_idea():
    df = _pullback_frame()
    quiet = df.copy()
    quiet.iloc[-3:] = quiet.iloc[-4].to_numpy()          # no pullback
    as_of = str(df.index[-1].date())
    assert live.shortterm_ideas({"AAA": quiet}, as_of, _snapshot("validated")) == []
    assert live.shortterm_ideas({"AAA": df}, "2030-01-01", _snapshot("validated")) == []
    assert live.MAX_STALE_DAYS == 4


@pytest.mark.unit
def test_no_snapshot_file_means_no_ideas(monkeypatch, tmp_path):
    monkeypatch.setattr(live, "SNAPSHOT_PATH", tmp_path / "missing.json")
    assert live.load_snapshot() is None
    df = _pullback_frame()
    assert live.shortterm_ideas({"AAA": df}, str(df.index[-1].date())) == []


@pytest.mark.unit
def test_snapshot_is_read_from_disk(tmp_path):
    path = tmp_path / "validated.json"
    path.write_text(json.dumps(_snapshot("validated")), encoding="utf-8")
    assert live.load_snapshot(path)["study_date"] == "2026-11-12"


@pytest.mark.unit
def test_live_and_the_study_share_one_signal_implementation():
    assert live.daily_signal is swing.daily_signal
    assert research.daily_signal is swing.daily_signal
    assert live.ALL_DAILY_SIGNALS["D3"] is dt.sig_d3


@pytest.mark.unit
def test_the_packaged_snapshot_is_consistent():
    snap = live.load_snapshot()
    assert snap is not None and snap["simulation_only"] is True
    ids = {s["id"] for s in snap["strategies"]}
    assert {s.id for s in research.all_specs()} == ids
    assert snap["variants_tried"] == len(ids)
    assert all(s["verdict"] in {"validated", "experimental", "rejected"} for s in snap["strategies"])
    # every strategy carries the derived label and it follows verdict AND beats_drift
    for s in snap["strategies"]:
        assert s["label"] == ev.label_of(s["verdict"], s["beats_drift"])
        assert s["label"] in {"validated", "experimental", "beta", "rejected"}
