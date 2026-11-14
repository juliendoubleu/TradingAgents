"""Investment ideas with a horizon and an honest evidence label.

The scan's composite score has no measured edge, and the study behind ``evidence.json``
(``horizon_research``) says which rules, if any, beat the market per instrument family and
horizon.  This module is where the two meet: every idea carries

* the horizon it is meant for (day, swing, medium, long) with its holding period, and
* an ``Evidence`` label saying what the measurements support: ``validated`` and
  ``experimental`` (a rule passed the pre-registered test), ``relative`` (the names lag the
  market: a reason to avoid them, not to short them), ``beta`` (made money, but no more than
  the universe), ``unproven`` (tested, no edge) or ``not_studied``.

Only ``validated`` and ``experimental`` ideas are recommendations.  Everything else is
``watch_only``: a watch list, shown so the reader sees the technical picture, never
presented as advice.  When nothing validated the idea set says exactly that, with the
numbers, instead of filling the list.

Medium and long ideas come from the SAME rule code as the study (``rules``), evaluated at
the latest settled bar; swing and day ideas are the translation of ``shortterm.live``.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.scanner import notices, rules
from tradingagents.scanner.fmt import fmt_date, fmt_num, format_text
from tradingagents.scanner.notices import Notice
from tradingagents.scanner.regime import MarketRegime
from tradingagents.scanner.signals import (
    _REASONS,
    MAX_POSITION_PCT,
    RISK_PER_TRADE_PCT,
    Signal,
    SignalParams,
)
from tradingagents.scanner.universe import UniverseInfo

logger = logging.getLogger(__name__)

SNAPSHOT_PATH = Path(__file__).with_name("evidence.json")
DAY, SWING, MEDIUM, LONG = "day", "swing", "medium", "long"
STALE_AFTER_DAYS = 120       # a study older than this is marked stale; it never updates itself
TIME_EXIT = (MEDIUM, LONG)   # rules that were measured with a time exit and NO stop
RECOMMENDED = ("validated", "experimental")
_RANK = {"validated": 3, "experimental": 2, "relative": 1}
LABELS = ("validated", "experimental", "relative", "beta", "unproven", "not_studied")


@dataclass(frozen=True)
class Horizon:
    key: str
    label: dict[str, str]
    holding: dict[str, str]
    holding_crypto: dict[str, str]


HORIZONS: dict[str, Horizon] = {h.key: h for h in [
    Horizon(DAY,
            {"en": "Day trade", "de": "Daytrade"},
            {"en": "Day trade: close out by the end of the session",
             "de": "Daytrade – Glattstellung bis Handelsschluss"},
            {"en": "Day trade: close out by the end of the day",
             "de": "Daytrade – Glattstellung bis Tagesende"}),
    Horizon(SWING,
            {"en": "Short term (swing)", "de": "Kurzfristig (Swing)"},
            {"en": "Short term (swing): 2 to 10 trading days",
             "de": "Kurzfristig (Swing) – 2 bis 10 Handelstage"},
            {"en": "Short term (swing): 2 to 10 days",
             "de": "Kurzfristig (Swing) – 2 bis 10 Tage"}),
    Horizon(MEDIUM,
            {"en": "Medium term", "de": "Mittelfristig"},
            {"en": "Medium term: about 1 month", "de": "Mittelfristig – etwa 1 Monat"},
            {"en": "Medium term: about 30 calendar days",
             "de": "Mittelfristig – etwa 30 Kalendertage"}),
    Horizon(LONG,
            {"en": "Long term", "de": "Langfristig"},
            {"en": "Long term: about 6 months", "de": "Langfristig – etwa 6 Monate"},
            {"en": "Long term: about 180 calendar days",
             "de": "Langfristig – etwa 180 Kalendertage"}),
]}

_FAMILY_NAME = {
    "stocks_us": {"en": "US stocks", "de": "US-Aktien"},
    "stocks_dax": {"en": "DAX stocks", "de": "DAX-Aktien"},
    "etfs": {"en": "ETFs", "de": "ETFs"},
    "crypto": {"en": "crypto", "de": "Krypto"},
    "mixed": {"en": "mixed universe", "de": "gemischtes Universum"},
    "custom": {"en": "this list", "de": "diese Liste"},
}
_PERIOD = {MEDIUM: {"en": "month", "de": "Monat"}, LONG: {"en": "half-year", "de": "Halbjahr"}}
_SIDE = {"long": {"en": "Long", "de": "Long"}, "short": {"en": "Short", "de": "Short"}}
_TF = {"1d": {"en": "daily", "de": "Tages-"}, "60m": {"en": "60-minute", "de": "60-Minuten-"},
       "5m": {"en": "5-minute", "de": "5-Minuten-"}}


_WATCH_DIR = {"Long": {"en": "technically strong", "de": "technisch stark"},
              "Short": {"en": "technically weak", "de": "technisch schwach"}}


def watch_direction(direction: str, lang: str = "en") -> str:
    """How a watch-list item names its side: never as a call (Long/Short) but as the
    technical picture ("technisch stark" / "technisch schwach")."""
    return _WATCH_DIR.get(direction, {"en": direction, "de": direction})["de" if lang == "de" else "en"]


WATCH_LOSS_WARNING = {"en": "earlier signals of this kind lost money in the test",
                      "de": "frühere Signale dieser Art verloren im Test deutlich"}


def holding_text(horizon: str, family: str, lang: str = "en") -> str:
    h = HORIZONS[horizon]
    src = h.holding_crypto if family == "crypto" else h.holding
    return src.get(lang, src["en"])


# ---------------------------------------------------------------------------- notes
# The notes are ``Notice`` objects like every other warning of the scan, so each output can
# print them in English or German.  Their texts are registered here, next to the code that
# raises them.

def _de_date(iso: str) -> str:
    return fmt_date(iso, "de")


notices.register({
    "idea_no_snapshot": {
        "en": "No evidence file found: no idea can be labelled with measured evidence, so "
              "nothing is recommended.",
        "de": "Keine Evidenzdatei gefunden: keine Idee kann mit gemessener Evidenz belegt "
              "werden, es wird nichts empfohlen."},
    "idea_not_studied": {
        "en": "{horizon_en}: not studied for {family_en}; no recommendation.",
        "de": "{horizon_de}: für {family_de} nicht untersucht; keine Empfehlung."},
    "idea_not_studied_watch": {
        "en": "{horizon_en} {side_en}: not studied for {family_en}; no recommendation, the "
              "technical picture is only something to watch.",
        "de": "{horizon_de} {side_de}: für {family_de} nicht untersucht; keine Empfehlung, "
              "das technische Bild ist nur eine Beobachtung."},
    "idea_no_rec_rules": {
        "en": "{horizon_en} {side_en}: no recommendation. {n} rules tested for {family_en}, none "
              "with a proven edge out of sample (study of {date}).",
        "de": "{horizon_de} {side_de}: keine Empfehlung. {n} Regeln für {family_de} getestet, "
              "keine mit belegtem Vorteil außerhalb der Stichprobe (Studie vom {date_de})."},
    "idea_relative_only": {
        "en": "{horizon_en} {side_en}: only a relative finding. The rule '{rule_en}' found that "
              "these names lagged the average by {mean:.2f} % per {period_en} out of sample "
              "(t = {t:.1f}), but they did not fall in absolute terms: avoid or underweight, no "
              "short recommendation (study of {date}).",
        "de": "{horizon_de} {side_de}: nur ein relativer Befund. Die Regel „{rule_de}“ fand, dass "
              "diese Werte außerhalb der Stichprobe {mean:.2f} % pro {period_de} hinter dem "
              "Durchschnitt lagen (t = {t:.1f}), aber nicht absolut fielen: meiden oder "
              "untergewichten, keine Short-Empfehlung (Studie vom {date_de})."},
    "idea_no_members": {
        "en": "{horizon_en} {side_en}: the rule '{rule_en}' currently selects no instrument.",
        "de": "{horizon_de} {side_de}: die Regel „{rule_de}“ wählt aktuell kein Instrument "
              "aus."},
    "idea_no_rec_short_term": {
        "en": "{horizon_en}: no recommendation. {n} rules tested on {tf_en} prices, none with an "
              "edge after costs (study of {date}).",
        "de": "{horizon_de}: keine Empfehlung. {n} Regeln auf {tf_de}Kursen getestet, nach Kosten "
              "keine mit Vorteil (Studie vom {date_de})."},
    "idea_beta_only": {
        "en": "{horizon_en}: no recommendation. Of {n} rules tested, only '{rules_en}' passed the "
              "test on absolute return, and it earns no more than the market's own drift: "
              "shown as a watch item (study of {date}).",
        "de": "{horizon_de}: keine Empfehlung. Von {n} getesteten Regeln bestand nur "
              "„{rules_de}“ den Test auf absolute Rendite, und sie verdient nicht mehr als der "
              "Markt selbst: als Beobachtung gezeigt (Studie vom {date_de})."},
    "idea_robust_failed": {
        "en": "{horizon_en} {side_en}: the rule '{rule_en}' passed the test but does not count: "
              "{caveat_en}.",
        "de": "{horizon_de} {side_de}: die Regel „{rule_de}“ bestand den Test, zählt aber nicht: "
              "{caveat_de}."},
    "idea_stale": {
        "en": "The evidence study is {days} days old (study of {date}; stale after {limit} days): "
              "every label is marked stale, the study does not update itself.",
        "de": "Die Evidenzstudie ist {days} Tage alt (Studie vom {date_de}; veraltet nach "
              "{limit} Tagen): jede Einstufung ist als veraltet markiert, die Studie "
              "aktualisiert sich nicht selbst."},
})


def _note(code: str, horizon: str, side: str | None, family: str, **params) -> Notice:
    h = HORIZONS[horizon].label
    fam = _FAMILY_NAME.get(family, {"en": family, "de": family})
    base = {"horizon": horizon, "side": side, "family": family, "horizon_en": h["en"],
            "horizon_de": h["de"], "family_en": fam["en"], "family_de": fam["de"]}
    if side:
        base.update(side_en=_SIDE[side]["en"], side_de=_SIDE[side]["de"])
    if "date" in params:
        params["date_de"] = _de_date(params["date"])
    return Notice(code, **base, **params)


# ---------------------------------------------------------------------------- evidence


def _s(x, nd=1, lang="en") -> str:
    return fmt_num(x, nd, lang)


@dataclass(frozen=True)
class Evidence:
    """What the measurements say about the rule behind an idea, in numbers.

    ``oos_*`` describe the judged series on the out-of-sample dates (trades for the short-
    term strategies); ``oos_t`` and ``is_t`` are the plain t of the non-overlapping dates,
    which is what the verdict used, ``oos_nw_t`` the optimistic Newey-West t (information
    only).  ``mean_bps`` is per decision date, i.e. per holding period.  ``caveat`` is the
    mandatory note of a result that hangs on a few names; ``robust`` says whether the
    result survived their removal (None: not checked).  ``stop_*`` is what the same rule
    earned with a protective stop.  ``text(lang)`` turns it all into ONE sentence that
    never claims more than the label.
    """

    label: str
    strong: bool = False
    rule_id: str | None = None
    rule_name: dict = field(default_factory=dict)
    family: str = ""
    horizon: str = ""
    side: str = ""
    study_date: str = ""
    judged: str | None = None            # long_excess | short_pnl | short_excess | long_abs | trades
    oos_n: int | None = None
    oos_mean_bps: float | None = None
    oos_t: float | None = None
    oos_pos_share: float | None = None
    oos_ci_low_bps: float | None = None
    oos_ci_high_bps: float | None = None
    is_t: float | None = None
    window: tuple[str, str] | None = None  # out-of-sample first and last decision date
    n_rules_tested: int | None = None
    excess_mean_bps: float | None = None   # beta: the surplus over the universe
    excess_t: float | None = None
    oos_nw_t: float | None = None
    caveat: dict | None = None             # {"en": ..., "de": ..., "short_en": ..., "short_de": ...}
    robust: bool | None = None
    stop_oos_bps: float | None = None      # the same rule with a stop (study's stop mode)
    stop_atr: float | None = None
    basket_avg: float | None = None        # average measured basket size
    stale: bool = False
    age_days: int | None = None

    def text(self, lang: str = "en") -> str:
        """The evidence as one sentence, caveat and staleness included."""
        lang = "de" if lang == "de" else "en"
        body = self._core(lang)
        if self.caveat and self.label in RECOMMENDED:
            body += (" – Achtung: " if lang == "de" else " - Caution: ") + self.caveat[lang]
        if self.stale and self.age_days is not None and self.label != "not_studied":
            body += (f" (Studie {self.age_days} Tage alt: veraltet)" if lang == "de"
                     else f" (study {self.age_days} days old: stale)")
        return body

    def _core(self, lang: str) -> str:
        de = lang == "de"
        if self.label == "not_studied":
            return (f"{notices.label_name('not_studied', 'de')}: für diesen Bereich liegt keine Messung vor."
                    if de else "Not studied: there is no measurement for this area.")
        per = _PERIOD.get(self.horizon, {"en": "trade", "de": "Trade"})[lang]
        if self.oos_mean_bps is None or self.oos_t is None:
            tested = (f" ({self.n_rules_tested} Regeln geprüft)" if de else
                      f" ({self.n_rules_tested} rules tested)") if self.n_rules_tested else ""
            return (f"{notices.label_name('unproven', 'de')}: im Test kein messbarer Vorteil{tested}." if de
                    else f"Not proven: no measurable edge in the test{tested}.")
        win = ""
        if self.window:
            y1, y2 = self.window[0][:4], self.window[1][:4]
            span = y1 if y1 == y2 else f"{y1}–{y2}"
            unit_de, unit_en = ("Trades", "trades") if self.judged == "trades" else (
                "Termine", "dates")
            win = (f"im Test ({span}, {self.oos_n} {unit_de})" if de
                   else f"in the test ({span}, {self.oos_n} {unit_en})")
        elif self.oos_n:
            win = f"im Test ({self.oos_n} Trades)" if de else f"in the test ({self.oos_n} trades)"
        else:
            win = "im Test" if de else "in the test"
        pct = self.oos_mean_bps / 100.0
        t = _s(self.oos_t, 1, lang)
        p = _s(pct, 2, lang)
        strong = (" (stark: gepoolt t ≥ 3)" if de else " (strong: pooled t >= 3)") if self.strong else ""
        if self.label in RECOMMENDED:
            head = notices.label_name(self.label, "de")
            head_en = notices.label_name(self.label, "en")
            if self.judged == "short_pnl":
                core_de = f"brachte ein Short-Korb {p} % pro {per}"
                core_en = f"a short basket made {p} % per {per}"
            elif self.judged == "trades":
                core_de = f"{p} % je Trade nach Kosten"
                core_en = f"{p} % per trade after costs"
            else:
                core_de = f"lag diese Regel {p} % pro {per} vor dem Durchschnitt"
                core_en = f"this rule beat the average by {p} % per {per}"
            prelim_de = " – vorläufig, kann Zufall sein" if self.label == "experimental" else ""
            prelim_en = " - preliminary, may be chance" if self.label == "experimental" else ""
            return (f"{head}: {win} {core_de}, t = {t}{prelim_de}{strong}" if de
                    else f"{head_en}: {win} {core_en}, t = {t}{prelim_en}{strong}")
        if self.label == "relative":
            return (f"{notices.label_name('relative', 'de')}: {win} blieben diese Werte {p} % pro {per} hinter dem "
                    f"Durchschnitt zurück (t = {t}), fielen aber nicht belegbar absolut: eher "
                    "meiden oder untergewichten als shorten" if de else
                    f"{notices.label_name('relative', 'en')}: {win} these names lagged the average by {p} % per "
                    f"{per} (t = {t}) but did not demonstrably fall in absolute terms: avoid or "
                    "underweight rather than short")
        if self.label == "beta":
            ex = "" if self.excess_mean_bps is None else (
                f", der Vorsprung vor dem Durchschnitt aller Werte betrug nur "
                f"{_s(self.excess_mean_bps / 100.0, 2, lang)} %"
                + (f" (t = {_s(self.excess_t, 1, lang)})" if self.excess_t is not None else "")
                if de else
                f", but the edge over the average of all names was only "
                f"{_s(self.excess_mean_bps / 100.0, 2, lang)} %"
                + (f" (t = {_s(self.excess_t, 1, lang)})" if self.excess_t is not None else ""))
            return (f"{notices.label_name('beta', 'de')}: {win} {p} % pro {per}{ex}: verdient nur, was der Markt "
                    "ohnehin verdiente" if de else
                    f"{notices.label_name('beta', 'en')}: {win} {p} % per {per}{ex}: earns what the market "
                    "earned anyway")
        # unproven
        if self.oos_t >= 2.0 and self.is_t is not None and self.is_t < 2.0:
            return (f"{notices.label_name('unproven', 'de')}: {win} lag die Regel zwar {p} % pro {per} vor dem Durchschnitt "
                    f"(t = {t}), im Auswahlzeitraum zeigte sich aber kein Vorteil (t = "
                    f"{_s(self.is_t, 1, lang)}): nach der vorab festgelegten Regel nicht belegt"
                    if de else
                    f"Not proven: {win} the rule beat the average by {p} % per {per} (t = {t}), "
                    f"but showed no edge in the selection period (t = {_s(self.is_t, 1, lang)}): "
                    "not proven under the pre-registered rule")
        what = {"long_excess": ("Vorsprung vor dem Durchschnitt", "edge over the average"),
                "short_pnl": ("Ergebnis einer Wette auf fallende Kurse", "result of a bet on falling prices")}.get(
            self.judged or "", ("Ergebnis", "result"))
        worse_de = "sogar im Minus, " if self.oos_t <= -2.0 else ""
        worse_en = "even negative, " if self.oos_t <= -2.0 else ""
        warn = (f"; {WATCH_LOSS_WARNING['de' if de else 'en']}" if self.oos_t <= -2.0 else "")
        return (f"{notices.label_name('unproven', 'de')}: {win} kein messbarer Vorteil ({worse_de}{what[0]} {p} % pro "
                f"{per}, t = {t}){warn}" if de else
                f"Not proven: {win} no measurable edge ({worse_en}{what[1]} {p} % per {per}, "
                f"t = {t}){warn}")

    def to_dict(self) -> dict:
        def num(x, nd=4):
            return None if x is None or not math.isfinite(x) else round(float(x), nd)
        return {
            "label": self.label, "strong": self.strong, "rule_id": self.rule_id,
            "rule_name": self.rule_name.get("en"), "rule_name_de": self.rule_name.get("de"),
            "family": self.family, "horizon": self.horizon, "side": self.side,
            "study_date": self.study_date, "judged": self.judged, "oos_n": self.oos_n,
            "oos_mean_bps": num(self.oos_mean_bps, 2), "oos_t": num(self.oos_t, 2),
            "oos_pos_share": num(self.oos_pos_share), "oos_ci_low_bps": num(self.oos_ci_low_bps, 2),
            "oos_ci_high_bps": num(self.oos_ci_high_bps, 2), "is_t": num(self.is_t, 2),
            "oos_nw_t": num(self.oos_nw_t, 2),
            "window": list(self.window) if self.window else None,
            "n_rules_tested": self.n_rules_tested, "excess_mean_bps": num(self.excess_mean_bps, 2),
            "excess_t": num(self.excess_t, 2), "caveat": self.caveat, "robust": self.robust,
            "stop_oos_bps": num(self.stop_oos_bps, 2), "basket_avg": num(self.basket_avg, 1),
            "stale": self.stale, "age_days": self.age_days,
            "text": self.text("en"), "text_de": self.text("de"),
        }


def _rule_name(snap: dict, rule_id: str) -> dict:
    meta = (snap.get("rules") or {}).get(rule_id) or {}
    return {"en": meta.get("name_en", rule_id), "de": meta.get("name_de", rule_id)}


def _t_of(stats: dict) -> float | None:
    """The judged t of a series block: the plain non-overlapping t (``nw_t`` only for a
    snapshot from before the correction)."""
    return stats.get("t_nonoverlap", stats.get("nw_t"))


def _mean_of(stats: dict) -> float | None:
    """The mean (bp) that belongs to the judged t: that of the non-overlapping subsample
    (``mean_nonoverlap_bps``); the overlapping mean only for an older snapshot."""
    m = stats.get("mean_nonoverlap_bps")
    return m if m is not None else stats.get("mean_bps")


def _evidence_from_entry(snap: dict, entry: dict, label: str | None = None) -> Evidence:
    judged = entry.get("judged")
    series = (entry.get("series") or {}).get(judged) or {}
    oos, is_ = series.get("OOS") or {}, series.get("IS") or {}
    fam = (snap.get("families") or {}).get(entry["family"], {})
    hz = (fam.get("horizons") or {}).get(entry["horizon"], {})
    window = (hz["oos_first"], hz["oos_last"]) if hz.get("oos_first") and hz.get("oos_last") else None
    label = label or entry.get("verdict", "unproven")
    ev_series = judged
    if label == "relative":
        ev_series = "short_excess"
        oos = ((entry.get("series") or {}).get("short_excess") or {}).get("OOS") or oos
        is_ = ((entry.get("series") or {}).get("short_excess") or {}).get("IS") or is_
    ex = ((entry.get("series") or {}).get("long_excess") or {}).get("OOS") or {}
    if label == "beta":
        ev_series = "long_abs"
        oos = ((entry.get("series") or {}).get("long_abs") or {}).get("OOS") or oos
        is_ = ((entry.get("series") or {}).get("long_abs") or {}).get("IS") or is_
    stop_oos = (((entry.get("stop") or {}).get(judged) or {}).get("OOS") or {}).get("mean_bps")
    caveat = entry.get("caveat") if label in RECOMMENDED else None
    return Evidence(
        label=label, strong=bool(entry.get("strong")), rule_id=entry["rule"],
        rule_name=_rule_name(snap, entry["rule"]), family=entry["family"],
        horizon=entry["horizon"], side=entry["side"], study_date=snap.get("study_date", ""),
        judged=ev_series, oos_n=oos.get("n_nonoverlap") or oos.get("n"), oos_mean_bps=_mean_of(oos),
        oos_t=_t_of(oos), oos_pos_share=oos.get("pos_share"),
        oos_ci_low_bps=oos.get("ci_low_bps"), oos_ci_high_bps=oos.get("ci_high_bps"),
        is_t=_t_of(is_), window=window,
        excess_mean_bps=_mean_of(ex) if label == "beta" else None,
        excess_t=_t_of(ex) if label == "beta" else None, oos_nw_t=oos.get("nw_t"),
        caveat=caveat, robust=entry.get("robust"), stop_oos_bps=stop_oos,
        stop_atr=(snap.get("stop_atr") or {}).get(entry["horizon"]),
        basket_avg=(entry.get("baskets") or {}).get("ALL"))


def best_entry(snap: dict, family: str, horizon: str, side: str) -> dict | None:
    """The best-evidenced rule for a (family, horizon, side): validated > experimental >
    relative; then pre-registered rules before variants; then the higher pooled t of the
    judged series.  None when no rule has evidence."""
    pool = [e for e in snap.get("results", [])
            if e.get("family") == family and e.get("horizon") == horizon
            and e.get("side") == side and e.get("verdict") in _RANK]
    if not pool:
        return None
    rules_meta = snap.get("rules") or {}

    def key(e):
        variant = bool((rules_meta.get(e["rule"]) or {}).get("variant"))
        # same verdict: a pre-registered rule before a variant (a variant has had more
        # chances to look good), then the higher pooled t
        return (_RANK[e["verdict"]], not variant, e.get("pooled_t") or 0.0)
    return max(pool, key=key)


def _tested_rules(snap: dict, family: str, horizon: str) -> int:
    return len({e["rule"] for e in snap.get("results", [])
                if e.get("family") == family and e.get("horizon") == horizon})


def _baseline_entry(snap: dict, family: str, horizon: str, side: str) -> dict | None:
    """The composite's own result (``T0``) when it was studied at this horizon."""
    for e in snap.get("results", []):
        if (e.get("family"), e.get("horizon"), e.get("side"), e.get("rule")) == (
                family, horizon, side, "T0"):
            return e
    return None


def _failed_robustness(snap: dict, family: str, horizon: str, side: str) -> dict | None:
    """The best cell that passed its test but not the removal of the top winners."""
    pool = [e for e in snap.get("results", [])
            if e.get("family") == family and e.get("horizon") == horizon
            and e.get("side") == side and e.get("robust") is False and e.get("caveat")]
    return max(pool, key=lambda e: e.get("pooled_t") or 0.0) if pool else None


# ---------------------------------------------------------------------------- ideas

_RULE_REASONS = {
    "rule_momentum": {
        "en": "12-1 momentum {v:+.1f} % (rank {rank} of {n} names)",
        "de": "12-1-Momentum {v:+.1f} % (Rang {rank} von {n} Werten)"},
    "rule_trend": {
        "en": "Price {gap:+.1f} % vs its 200-day average, 50-day average {cross:+.1f} % vs 200-day",
        "de": "Kurs {gap:+.1f} % gegenüber dem 200-Tage-Schnitt, 50-Tage-Schnitt {cross:+.1f} % "
              "gegenüber dem 200-Tage-Schnitt"},
    "rule_high": {
        "en": "Close at {pct:.0f} % of its 52-week closing high",
        "de": "Schlusskurs bei {pct:.0f} % des 52-Wochen-Schlusshochs"},
    "rule_pullback": {
        "en": "1-month return {v:+.1f} % while above/below its 200-day average (rank {rank} of {n})",
        "de": "1-Monats-Rendite {v:+.1f} % bei Lage zum 200-Tage-Schnitt (Rang {rank} von {n})"},
}


@dataclass
class Idea:
    """One instrument on one horizon.

    A recommendation of a time-exit rule has ``stop=None`` (the study measured it without
    one and the stop destroyed the edge) and ``size_pct`` = equal weight in the measured
    basket, explained in ``size_note``.  A watch-only idea carries no entry, stop or size:
    it is something to look at, not a plan.  ``conflict`` is set when the scanner's
    technical picture points the other way; ``outside`` when the instrument was not part
    of the studied universe.
    """

    symbol: str
    direction: str                         # "Long" | "Short"
    horizon: str
    rule_id: str
    rank_score: float | None
    reference_price: float
    reference_date: str
    entry: dict                            # {"en": ..., "de": ...}; empty for a watch item
    stop: float | None
    target: float | None
    size_pct: float | None
    holding: dict
    reason_items: list[tuple[str, dict]]
    evidence: Evidence
    watch_only: bool
    stance: str = "watch"                  # recommended | experimental | avoid | watch
    rank: int | None = None
    rule_name: dict = field(default_factory=dict)
    size_note: dict | None = None          # {"en": ..., "de": ...} how the size and the missing stop arise
    conflict: dict | None = None           # {"en": ..., "de": ...} the technical picture says the opposite
    outside: dict | None = None            # {"en": ..., "de": ...} not part of the studied universe
    basket_size: int | None = None         # names in the basket the rule selects today

    def reasons_text(self, lang: str = "en") -> list[str]:
        lang = lang if lang in ("en", "de") else "en"
        prefix = {"en": "Counter-signal: ", "de": "Gegensignal: "}[lang]
        out = []
        for code, params in self.reason_items:
            table = _RULE_REASONS.get(code) or _REASONS[code]
            out.append(("" if not params.get("against") else prefix)
                       + format_text(table.get(lang, table["en"]), lang, **params))
        return out

    def to_dict(self) -> dict:
        def num(x, nd=4):
            return None if x is None or not math.isfinite(x) else round(float(x), nd)
        return {
            "symbol": self.symbol, "direction": self.direction, "horizon": self.horizon,
            "rule_id": self.rule_id, "rule_name": self.rule_name.get("en"),
            "rule_name_de": self.rule_name.get("de"), "rank": self.rank,
            "rank_score": num(self.rank_score, 3), "reference_price": num(self.reference_price),
            "reference_date": self.reference_date, "entry": self.entry.get("en"),
            "entry_de": self.entry.get("de"), "stop": num(self.stop), "target": num(self.target),
            "size_pct": num(self.size_pct, 2), "holding": self.holding.get("en"),
            "holding_de": self.holding.get("de"), "reasons": self.reasons_text("en"),
            "reasons_de": self.reasons_text("de"), "evidence": self.evidence.to_dict(),
            "watch_only": self.watch_only, "stance": self.stance,
            "size_note": self.size_note, "conflict": self.conflict, "outside": self.outside,
            "basket_size": self.basket_size,
        }


@dataclass
class IdeaSet:
    """Ideas per horizon and side, plus the notes that say why a list is empty or only a
    watch list.  ``ideas`` keeps the horizon order day, swing, medium, long."""

    family: str = ""
    study_date: str = ""
    evidence_available: bool = False
    ideas: list[Idea] = field(default_factory=list)
    notes: list[Notice] = field(default_factory=list)
    study_age_days: int | None = None
    stale: bool = False

    def select(self, horizon: str | None = None, direction: str | None = None,
               recommended_only: bool = False) -> list[Idea]:
        return [i for i in self.ideas
                if (horizon is None or i.horizon == horizon)
                and (direction is None or i.direction == direction)
                and (not recommended_only or not i.watch_only)]

    def recommended(self) -> list[Idea]:
        return [i for i in self.ideas if not i.watch_only]

    def to_dict(self) -> dict:
        return {
            "family": self.family, "study_date": self.study_date,
            "study_age_days": self.study_age_days, "stale": self.stale,
            "evidence_available": self.evidence_available,
            "horizons": {k: {"label": h.label["en"], "label_de": h.label["de"],
                             "holding": holding_text(k, self.family, "en"),
                             "holding_de": holding_text(k, self.family, "de")}
                         for k, h in HORIZONS.items()},
            "ideas": [i.to_dict() for i in self.ideas],
            "notes": [{**n.to_dict(), "horizon": n.params.get("horizon"),
                       "side": n.params.get("side")} for n in self.notes],
        }


def load_snapshot(path: str | Path | None = None) -> dict | None:
    """The persisted medium/long study, or None when there is none (never an error)."""
    try:
        return json.loads(Path(path or SNAPSHOT_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("evidence snapshot unavailable: %s", exc)
        return None


def _equal_weight_pct(n_names: int) -> float | None:
    """Equal weight inside the basket that was measured: 100 % / names, capped."""
    return min(MAX_POSITION_PCT, 100.0 / n_names) if n_names > 0 else None


def _size_pct(price: float, stop: float | None) -> float | None:
    """Risk-based size for the short-term rules, which were measured WITH a stop."""
    if stop is None or price <= 0:
        return None
    frac = abs(price - stop) / price
    return min(MAX_POSITION_PCT, RISK_PER_TRADE_PCT / frac) if frac > 0 else None


def _size_note(evidence: Evidence, n_names: int, size: float | None) -> dict:
    """Why a time-exit idea has no stop and how its size arises, in both languages."""
    lang_text = {}
    for lang in ("en", "de"):
        part = {
            "en": "Sized equal-weight within the measured basket ({n} names): {size:.1f} % each "
                  "if you hold the whole basket. The tested rule has no stop",
            "de": "Gleichgewichtet im gemessenen Korb ({n} Werte): je {size:.1f} %, wenn du den "
                  "ganzen Korb hältst. Die getestete Regel hat keinen Stop"}[lang]
        text = format_text(part, lang, n=n_names, size=size or 0.0)
        if evidence.stop_oos_bps is not None and evidence.oos_mean_bps is not None:
            more = {
                "en": "; in the test with a stop ({atr:.1f} x ATR) the edge out of sample was "
                      "{a:.0f} instead of {b:.0f} basis points per period, so no stop is "
                      "suggested.",
                "de": "; im Test mit Stop ({atr:.1f} x ATR) lag der Vorteil außerhalb der "
                      "Stichprobe bei {a:.0f} statt {b:.0f} Basispunkten je Periode, deshalb "
                      "wird kein Stop vorgeschlagen."}[lang]
            text += format_text(more, lang, atr=evidence.stop_atr or 0.0,
                                a=evidence.stop_oos_bps, b=evidence.oos_mean_bps)
        else:
            text += {"en": ", so none is suggested.", "de": ", deshalb wird keiner vorgeschlagen."}[lang]
        lang_text[lang] = text
    return lang_text


def _rule_reason(panel: rules.Panel, rule_id: str, sym: str, rank: int, n: int) -> list:
    spec = rules.RULES[rule_id]
    p = rules.rule_params(rule_id)
    close = float(panel.close[sym].iloc[-1])
    if spec.base in ("T1", "T3"):
        v = panel.momentum(p["lookback"], p["skip"])[sym].iloc[-1]
        return [("rule_momentum", {"v": float(v) * 100, "rank": rank, "n": n})] \
            if np.isfinite(v) else []
    if spec.base == "T2":
        s50, s200 = panel.sma(p["fast"])[sym].iloc[-1], panel.sma(p["slow"])[sym].iloc[-1]
        return [("rule_trend", {"gap": (close / s200 - 1) * 100, "cross": (s50 / s200 - 1) * 100})]
    if spec.base == "T4":
        hi = panel.high_n(panel.params.bars_per_year)[sym].iloc[-1]
        return [("rule_high", {"pct": close / hi * 100})]
    if spec.base == "T5":
        v = panel.ret_months(1)[sym].iloc[-1]
        return [("rule_pullback", {"v": float(v) * 100, "rank": rank, "n": n})]
    return []


def _stance(label: str, side: str) -> str:
    if label == "validated":
        return "recommended"
    if label == "experimental":
        return "experimental"
    if label == "relative" and side == "short":
        return "avoid"
    return "watch"


def _make_idea(panel: rules.Panel | None, sym: str, side: int, horizon: str, rule_id: str,
               score: float | None, rank: int | None, evidence: Evidence, family: str,
               rule_name: dict, signals: dict[str, Signal], reference: tuple[float, str, float | None],
               n_names: int, basket: int | None = None) -> Idea:
    """An idea of a time-exit rule: no stop; a recommendation is sized equal-weight in its
    basket; a watch item has no entry, stop or size at all."""
    price, date_, _atr = reference
    items: list[tuple[str, dict]] = []
    if panel is not None and rule_id != "T0" and rank is not None:
        items += _rule_reason(panel, rule_id, sym, rank, n_names)
    sig = signals.get(sym)
    if sig is not None:
        items += list(sig.reason_items)
    watch = evidence.label not in RECOMMENDED
    direction = "Long" if side > 0 else "Short"
    size = None if watch else _equal_weight_pct(basket or 0)
    return Idea(
        symbol=sym, direction=direction, horizon=horizon, rule_id=rule_id, rank_score=score,
        reference_price=price, reference_date=date_,
        entry={} if watch else {"en": "at the next open", "de": "zur nächsten Eröffnung"},
        stop=None, target=None, size_pct=size,
        holding={"en": holding_text(horizon, family, "en"),
                 "de": holding_text(horizon, family, "de")},
        reason_items=items, evidence=evidence, watch_only=watch,
        stance=_stance(evidence.label, direction.lower()), rank=rank, rule_name=rule_name,
        size_note=None if watch else _size_note(evidence, basket or 0, size),
        basket_size=basket)


def _reference(panel: rules.Panel, sym: str) -> tuple[float, str, float | None]:
    atr = panel.atr()[sym].iloc[-1]
    return (float(panel.close[sym].iloc[-1]), str(panel.index[-1].date()),
            float(atr) if np.isfinite(atr) else None)


def _composite_members(signals: list[Signal], side: int, top: int | None) -> list[Signal]:
    want = "Long" if side > 0 else "Short"
    pool = [s for s in signals if s.direction == want]
    pool.sort(key=lambda s: (-s.score if side > 0 else s.score, s.symbol))
    return pool if top is None else pool[:top]


def _medium_long_ideas(
    info: UniverseInfo, panel: rules.Panel | None, signals: list[Signal],
    snap: dict | None, top: int, out: IdeaSet,
) -> None:
    family = info.family
    sig_by = {s.symbol: s for s in signals}
    studied = bool(snap) and family in (snap.get("families") or {})
    for horizon in (MEDIUM, LONG):
        for side_name, side in (("long", 1), ("short", -1)):
            entry = best_entry(snap, family, horizon, side_name) if studied else None
            if entry is None:
                _watch_list(info, panel, signals, sig_by, snap, studied, horizon, side_name,
                            side, top, out)
                continue
            rule_id = entry["rule"]
            evidence = _evidence_from_entry(snap, entry)
            rname = _rule_name(snap, rule_id)
            if entry["verdict"] == "relative":
                out.notes.append(_note(
                    "idea_relative_only", horizon, side_name, family, rule_en=rname["en"],
                    rule_de=rname["de"], mean=(evidence.oos_mean_bps or 0.0) / 100.0,
                    t=evidence.oos_t or 0.0, period_en=_PERIOD[horizon]["en"],
                    period_de=_PERIOD[horizon]["de"], date=snap.get("study_date", "")))
            if rule_id not in rules.RULES:
                logger.warning("evidence names unknown rule %s; skipped", rule_id)
                continue
            # The whole measured basket, not its first ten: the size and the evidence are
            # about the basket as a whole.
            picks: list[tuple[str, float | None, int | None, tuple]] = []
            if rules.RULES[rule_id].base == "T0":
                for i, s in enumerate(_composite_members(signals, side, None), 1):
                    picks.append((s.symbol, s.score, i, (s.close, s.last_date, s.atr)))
                n_names = len(signals)
            elif panel is not None:
                members = [m for m in rules.current_members(rule_id, panel) if m[1] == side]
                n_names = int(rules.rule_sides(rule_id, panel)[1].iloc[-1].notna().sum())
                for sym, _, score, rank in members:
                    picks.append((sym, score, rank, _reference(panel, sym)))
            else:
                n_names = 0
            if not picks:
                out.notes.append(_note(
                    "idea_no_members", horizon, side_name, family, rule_en=rname["en"],
                    rule_de=rname["de"]))
                continue
            for sym, score, rank, ref in picks:
                out.ideas.append(_make_idea(panel, sym, side, horizon, rule_id, score, rank,
                                            evidence, family, rname, sig_by, ref, n_names,
                                            basket=len(picks)))


def _watch_list(info, panel, signals, sig_by, snap, studied, horizon, side_name, side, top,
                out: IdeaSet) -> None:
    """No rule has evidence for this side.

    Medium horizon: the composite's names as a labelled watch list (the composite was
    tested there).  Long horizon: only the note, because the composite was never tested at
    that horizon and a list would borrow evidence it does not have.
    """
    family = info.family
    if not snap or not studied:
        evidence = Evidence(label="not_studied", family=family, horizon=horizon,
                            side=side_name, study_date=(snap or {}).get("study_date", ""))
        out.notes.append(_note("idea_not_studied_watch", horizon, side_name, family))
    else:
        base = _baseline_entry(snap, family, horizon, side_name)
        if base is not None:
            evidence = _evidence_from_entry(snap, base, label="unproven")
        else:
            evidence = Evidence(
                label="unproven", rule_id="T0", rule_name=_rule_name(snap, "T0"), family=family,
                horizon=horizon, side=side_name, study_date=snap.get("study_date", ""),
                n_rules_tested=_tested_rules(snap, family, horizon))
        out.notes.append(_note(
            "idea_no_rec_rules", horizon, side_name, family,
            n=_tested_rules(snap, family, horizon), date=snap.get("study_date", "")))
        failed = _failed_robustness(snap, family, horizon, side_name)
        if failed is not None:
            name = _rule_name(snap, failed["rule"])
            out.notes.append(_note(
                "idea_robust_failed", horizon, side_name, family, rule_en=name["en"],
                rule_de=name["de"], caveat_en=failed["caveat"]["en"],
                caveat_de=failed["caveat"]["de"]))
    if horizon != MEDIUM:
        return
    for i, s in enumerate(_composite_members(signals, side, top), 1):
        out.ideas.append(_make_idea(
            panel, s.symbol, side, horizon, "T0", s.score, i, evidence, family,
            {"en": rules.RULES["T0"].name_en, "de": rules.RULES["T0"].name_de}, sig_by,
            (s.close, s.last_date, s.atr), len(signals)))


def _short_term_ideas(info: UniverseInfo, frames: dict[str, pd.DataFrame], as_of: str,
                      shortterm_snapshot: dict | None, top: int, out: IdeaSet) -> None:
    from tradingagents.scanner.shortterm import live

    family = info.family
    if family != "stocks_us":
        for h in (DAY, SWING):
            out.notes.append(_note("idea_not_studied", h, None, family))
        return
    snap = shortterm_snapshot if shortterm_snapshot is not None else live.load_snapshot()
    if not snap:
        for h in (DAY, SWING):
            out.notes.append(_note("idea_not_studied", h, None, family))
        return
    date_ = snap.get("study_date", "")
    members = {s: f for s, f in frames.items() if s in set(info.members)}
    try:
        raw = live.shortterm_ideas(members, as_of, snapshot=snap, include_experimental=True,
                                   include_beta=True) if members else []
    except Exception as exc:     # a broken short-term snapshot must not take the scan down
        logger.warning("short-term ideas failed: %s", exc)
        raw = []
    strategies = snap.get("strategies", [])
    for horizon in (DAY, SWING):
        strats = [s for s in strategies if s.get("horizon") == horizon]
        mine = [r for r in raw if r.horizon == horizon]
        recommendable = False
        for r in mine[:max(top * 4, top)]:
            label = getattr(r, "label", None) or r.verdict
            if r.beats_drift is False and label == "validated":
                label = "beta"        # passed the verdict but earns only the market's drift
            ev = Evidence(
                label=label, strong=False, rule_id=r.strategy_id,
                rule_name={"en": r.strategy_name_en, "de": r.strategy_name_de}, family=family,
                horizon=horizon, side=r.direction.lower(), study_date=r.study_date,
                judged="trades", oos_n=r.oos_n_trades, oos_mean_bps=r.oos_mean_net_bps,
                oos_t=r.oos_t_stat, oos_pos_share=r.oos_win_rate,
                window=_oos_window(snap), excess_mean_bps=r.oos_excess_bps,
                excess_t=r.oos_excess_t_stat)
            recommendable |= label in RECOMMENDED
            watch = label not in RECOMMENDED
            out.ideas.append(Idea(
                symbol=r.symbol, direction=r.direction, horizon=horizon, rule_id=r.strategy_id,
                rank_score=None, reference_price=r.reference_price,
                reference_date=_last_date(members.get(r.symbol)),
                entry={} if watch else {"en": r.entry_rule_en, "de": r.entry_rule_de},
                stop=None if watch else r.stop, target=None,
                size_pct=None if watch else _size_pct(r.reference_price, r.stop),
                holding={"en": f"{HORIZONS[horizon].label['en']}: {r.holding_en}",
                         "de": f"{HORIZONS[horizon].label['de']}: {r.holding_de}"},
                reason_items=[], evidence=ev, watch_only=watch,
                stance=_stance(label, r.direction.lower()),
                rule_name={"en": r.strategy_name_en, "de": r.strategy_name_de}))
        if recommendable:
            continue
        if not strats:
            out.notes.append(_note("idea_not_studied", horizon, None, family))
            continue
        passed = [s for s in strats if s.get("verdict") in ("validated", "experimental")]
        if passed and all(s.get("beats_drift") is False for s in passed):
            out.notes.append(_note(
                "idea_beta_only", horizon, None, family, n=len(strats), date=date_,
                rules_en=", ".join(s.get("name_en") or s["id"] for s in passed),
                rules_de=", ".join(s.get("name_de") or s["id"] for s in passed)))
        else:
            tfs = sorted({s.get("timeframe", "1d") for s in strats},
                         key=["1d", "60m", "5m"].index)
            out.notes.append(_note(
                "idea_no_rec_short_term", horizon, None, family, n=len(strats), date=date_,
                tf_en=_join([_TF[t]["en"] for t in tfs], "and"),
                tf_de=_join([_TF[t]["de"].rstrip("-") + "-" for t in tfs], "und")))


def _join(parts: list[str], word: str) -> str:
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} {word} {parts[-1]}"


def _oos_window(snap: dict) -> tuple[str, str] | None:
    sp = (snap.get("splits") or {}).get("1d") or {}
    return (sp["oos_first"], sp["oos_last"]) if sp.get("oos_first") else None


def _last_date(frame) -> str:
    return "" if frame is None or frame.empty else str(frame.index[-1].date())


def _flag_conflicts(ideas: list[Idea], signals: list[Signal]) -> None:
    """Where the composite's direction opposes an idea, say so on the idea itself: a name can
    be a long-term Long and on the technical Short list at once, and the reader must see both."""
    seen = {s.symbol: s for s in signals}
    for idea in ideas:
        sig = seen.get(idea.symbol)
        if sig is None or sig.direction not in ("Long", "Short") or sig.direction == idea.direction:
            continue
        weak = idea.direction == "Long"
        score = {"en": fmt_num(sig.score, 0, "en"), "de": fmt_num(sig.score, 0, "de")}
        idea.conflict = {
            "en": (f"Technically weak right now: the scanner's technical picture is weak "
                   f"(score {score['en']})." if weak else
                   f"Technically strong right now: the scanner's technical picture is strong "
                   f"(score {score['en']})."),
            "de": (f"Technisch aktuell schwach: das technische Bild des Scanners ist schwach "
                   f"(Score {score['de']})." if weak else
                   f"Technisch aktuell stark: das technische Bild des Scanners ist stark "
                   f"(Score {score['de']}).")}


def _flag_outside(ideas: list[Idea], info: UniverseInfo) -> None:
    """An instrument the studies never saw can be ranked, but its idea says so."""
    for idea in ideas:
        if info.is_extra(idea.symbol):
            idea.outside = {
                "en": f"{idea.symbol} was not part of the studied universe: the evidence is the "
                      "group's, not this instrument's.",
                "de": f"{idea.symbol} gehörte nicht zum untersuchten Universum: der Beleg gilt "
                      "für die Gruppe, nicht für dieses Papier."}


def _mark_stale(out: IdeaSet, today: date) -> None:
    """Flag every label whose study is older than ``STALE_AFTER_DAYS`` (a note says so once)."""
    def age(iso: str) -> int | None:
        try:
            return (today - date.fromisoformat(str(iso)[:10])).days
        except ValueError:
            return None
    out.study_age_days = age(out.study_date) if out.study_date else None
    worst = out.study_age_days
    for idea in out.ideas:
        days = age(idea.evidence.study_date) if idea.evidence.study_date else None
        if days is not None and days > STALE_AFTER_DAYS:
            idea.evidence = replace(idea.evidence, stale=True, age_days=days)
            worst = days if worst is None else max(worst, days)
    if out.study_age_days is not None and out.study_age_days > STALE_AFTER_DAYS:
        out.stale = True
        out.notes.append(Notice("idea_stale", days=out.study_age_days, date=out.study_date,
                                date_de=fmt_date(out.study_date, "de"), limit=STALE_AFTER_DAYS))


def build_ideas(
    info: UniverseInfo,
    frames: dict[str, pd.DataFrame],
    signals: list[Signal],
    regime: MarketRegime | None = None,
    evidence_snapshot: dict | None = None,
    shortterm_snapshot: dict | None = None,
    top: int = 10,
    today: date | None = None,
) -> IdeaSet:
    """Ideas per horizon and side for one scan, each with its evidence label.

    ``frames`` are the settled daily bars of the scan (members and benchmark),
    ``signals`` its composite signals.  ``evidence_snapshot=None`` reads the packaged
    ``evidence.json``; a missing or unreadable file is not an error: everything is then
    ``not_studied`` and a note says so.  ``top`` limits the watch lists only; an evidenced
    basket is shown whole.  ``today`` (default the current date) decides staleness.
    ``regime`` is accepted for a stable signature (the composite signals already carry it).
    """
    snap = load_snapshot() if evidence_snapshot is None else evidence_snapshot
    snap = snap or None
    out = IdeaSet(family=info.family, study_date=(snap or {}).get("study_date", ""),
                  evidence_available=bool(snap))
    if snap is None:
        out.notes.append(Notice("idea_no_snapshot"))
    panel = None
    members = [m for m in info.members if m in frames]
    if members:
        try:
            panel = rules.Panel(frames, members, info.benchmark,
                                SignalParams(info.month_bars, info.bars_per_year))
        except ValueError:
            panel = None
    as_of = str(panel.index[-1].date()) if panel is not None else max(
        (s.last_date for s in signals), default="")
    if as_of:
        _short_term_ideas(info, frames, as_of, shortterm_snapshot, top, out)
    else:
        for h in (DAY, SWING):
            out.notes.append(_note("idea_not_studied", h, None, info.family))
    _medium_long_ideas(info, panel, signals, snap, top, out)
    _flag_conflicts(out.ideas, signals)
    _flag_outside(out.ideas, info)
    _mark_stale(out, today or date.today())
    order = {DAY: 0, SWING: 1, MEDIUM: 2, LONG: 3}
    out.ideas.sort(key=lambda i: (order[i.horizon], i.direction != "Long", i.rank or 0))
    return out


__all__ = ["HORIZONS", "Evidence", "Idea", "IdeaSet", "build_ideas", "best_entry",
           "holding_text", "load_snapshot"]
