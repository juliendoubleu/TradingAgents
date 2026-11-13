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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tradingagents.scanner import notices, rules
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
STOP_ATR = {MEDIUM: 2.5, LONG: 4.0}
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
}
_PERIOD = {MEDIUM: {"en": "month", "de": "Monat"}, LONG: {"en": "half-year", "de": "Halbjahr"}}
_SIDE = {"long": {"en": "Long", "de": "Long"}, "short": {"en": "Short", "de": "Short"}}
_TF = {"1d": {"en": "daily", "de": "Tages-"}, "60m": {"en": "60-minute", "de": "60-Minuten-"},
       "5m": {"en": "5-minute", "de": "5-Minuten-"}}


def holding_text(horizon: str, family: str, lang: str = "en") -> str:
    h = HORIZONS[horizon]
    src = h.holding_crypto if family == "crypto" else h.holding
    return src.get(lang, src["en"])


# ---------------------------------------------------------------------------- notes
# The notes are ``Notice`` objects like every other warning of the scan, so each output can
# print them in English or German.  Their texts are registered here, next to the code that
# raises them.

def _de_date(iso: str) -> str:
    try:
        return pd.Timestamp(iso).strftime("%d.%m.%Y")
    except (ValueError, TypeError):
        return str(iso)


notices._TEXT.update({
    "idea_no_snapshot": {
        "en": "No evidence file found: no idea can be labelled with measured evidence, every "
              "list is a watch list.",
        "de": "Keine Evidenzdatei gefunden: keine Idee kann mit gemessener Evidenz belegt "
              "werden, jede Liste ist eine Beobachtungsliste."},
    "idea_not_studied": {
        "en": "{horizon_en}: not studied for {family_en}; no recommendation.",
        "de": "{horizon_de}: für {family_de} nicht untersucht; keine Empfehlung."},
    "idea_not_studied_watch": {
        "en": "{horizon_en} {side_en}: not studied for {family_en}; the technical picture below is "
              "a watch list, not a recommendation.",
        "de": "{horizon_de} {side_de}: für {family_de} nicht untersucht; das technische Bild "
              "unten ist eine Beobachtungsliste, keine Empfehlung."},
    "idea_no_rec_rules": {
        "en": "{horizon_en} {side_en}: no recommendation. {n} rules tested for {family_en}, none "
              "with a proven edge out of sample (study of {date}); the list below is the "
              "technical picture as a watch list.",
        "de": "{horizon_de} {side_de}: keine Empfehlung. {n} Regeln für {family_de} getestet, "
              "keine mit belegtem Vorteil außerhalb der Stichprobe (Studie vom {date_de}); die "
              "Liste unten ist das technische Bild als Beobachtungsliste."},
    "idea_relative_only": {
        "en": "{horizon_en} {side_en}: only a relative finding. Rule {rule} found that these "
              "names lag the average by {mean:.2f} % per {period_en} out of sample (t = {t:.1f}), "
              "but they did not fall in absolute terms: avoid or underweight, no short "
              "recommendation (study of {date}).",
        "de": "{horizon_de} {side_de}: nur ein relativer Befund. Regel {rule} fand, dass diese "
              "Werte außerhalb der Stichprobe {mean_de} % pro {period_de} hinter dem Durchschnitt "
              "lagen (t = {t_de}), aber nicht absolut fielen: meiden oder untergewichten, keine "
              "Short-Empfehlung (Studie vom {date_de})."},
    "idea_no_members": {
        "en": "{horizon_en} {side_en}: rule {rule} ({evidence_en}) currently selects no "
              "instrument.",
        "de": "{horizon_de} {side_de}: Regel {rule} ({evidence_de}) wählt aktuell kein "
              "Instrument aus."},
    "idea_no_rec_short_term": {
        "en": "{horizon_en}: no recommendation. {n} rules tested on {tf_en} prices, none with an "
              "edge after costs (study of {date}).",
        "de": "{horizon_de}: keine Empfehlung. {n} Regeln auf {tf_de}Kursen getestet, nach Kosten "
              "keine mit Vorteil (Studie vom {date_de})."},
    "idea_beta_only": {
        "en": "{horizon_en}: no recommendation. Of {n} rules tested, only {rules} passed the "
              "test on absolute return, and it earns no more than the market's own drift: "
              "shown as a watch item (study of {date}).",
        "de": "{horizon_de}: keine Empfehlung. Von {n} getesteten Regeln bestand nur {rules} den "
              "Test auf absolute Rendite, und sie verdient nicht mehr als der Markt selbst: "
              "als Beobachtung gezeigt (Studie vom {date_de})."},
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


def _de(x, nd=1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def _s(x, nd=1, lang="en") -> str:
    return _de(x, nd) if lang == "de" else f"{x:.{nd}f}"


@dataclass(frozen=True)
class Evidence:
    """What the measurements say about the rule behind an idea, in numbers.

    ``oos_*`` describe the judged series on the out-of-sample dates (trades for the short-
    term strategies), ``is_t`` the in-sample t.  ``mean_bps`` is per decision date, i.e. per
    holding period.  ``text(lang)`` turns it into ONE sentence that never claims more than
    the label.
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

    def text(self, lang: str = "en") -> str:
        de = lang == "de"
        if self.label == "not_studied":
            return ("Nicht untersucht: für diesen Bereich liegt keine Messung vor."
                    if de else "Not studied: there is no measurement for this area.")
        per = _PERIOD.get(self.horizon, {"en": "trade", "de": "Trade"})["de" if de else "en"]
        if self.oos_mean_bps is None or self.oos_t is None:
            tested = (f" ({self.n_rules_tested} Regeln geprüft)" if de else
                      f" ({self.n_rules_tested} rules tested)") if self.n_rules_tested else ""
            return (f"Nicht belegt: im Test kein messbarer Vorteil{tested}." if de
                    else f"Not proven: no measurable edge in the test{tested}.")
        win = ""
        if self.window:
            y1, y2 = self.window[0][:4], self.window[1][:4]
            span = y1 if y1 == y2 else f"{y1}–{y2}"
            win = (f"im Test ({span}, {self.oos_n} Termine)" if de
                   else f"in the test ({span}, {self.oos_n} dates)")
        elif self.oos_n:
            win = f"im Test ({self.oos_n} Trades)" if de else f"in the test ({self.oos_n} trades)"
        else:
            win = "im Test" if de else "in the test"
        pct = self.oos_mean_bps / 100.0
        t = _s(self.oos_t, 1, lang)
        p = _s(pct, 2, lang)
        strong = (" (stark: gepoolt t ≥ 3)" if de else " (strong: pooled t >= 3)") if self.strong else ""
        if self.label in RECOMMENDED:
            head = "Belegt" if self.label == "validated" else "Schwach belegt (experimentell)"
            head_en = "Validated" if self.label == "validated" else "Weakly supported (experimental)"
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
            return (f"Nur relativ belegt: {win} blieben diese Werte {p} % pro {per} hinter dem "
                    f"Durchschnitt zurück (t = {t}), fielen aber nicht belegbar absolut: eher "
                    "meiden oder untergewichten als shorten" if de else
                    f"Relative evidence only: {win} these names lagged the average by {p} % per "
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
            return (f"Nur Markt-Beta: {win} {p} % pro {per}{ex}: verdient nur, was der Markt "
                    "ohnehin verdiente" if de else
                    f"Market beta only: {win} {p} % per {per}{ex}: earns what the market "
                    "earned anyway")
        # unproven
        if self.oos_t >= 2.0 and self.is_t is not None and self.is_t < 2.0:
            return (f"Nicht belegt: {win} lag die Regel zwar {p} % pro {per} vor dem Durchschnitt "
                    f"(t = {t}), im Auswahlzeitraum zeigte sich aber kein Vorteil (t = "
                    f"{_s(self.is_t, 1, lang)}): nach der vorab festgelegten Regel nicht belegt"
                    if de else
                    f"Not proven: {win} the rule beat the average by {p} % per {per} (t = {t}), "
                    f"but showed no edge in the selection period (t = {_s(self.is_t, 1, lang)}): "
                    "not proven under the pre-registered rule")
        what = {"long_excess": ("Vorsprung vor dem Durchschnitt", "edge over the average"),
                "short_pnl": ("Short-Ergebnis", "short result")}.get(
            self.judged or "", ("Ergebnis", "result"))
        return (f"Nicht belegt: {win} kein messbarer Vorteil ({what[0]} {p} % pro {per}, "
                f"t = {t})" if de else
                f"Not proven: {win} no measurable edge ({what[1]} {p} % per {per}, t = {t})")

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
            "window": list(self.window) if self.window else None,
            "n_rules_tested": self.n_rules_tested, "excess_mean_bps": num(self.excess_mean_bps, 2),
            "excess_t": num(self.excess_t, 2),
            "text": self.text("en"), "text_de": self.text("de"),
        }


def _rule_name(snap: dict, rule_id: str) -> dict:
    meta = (snap.get("rules") or {}).get(rule_id) or {}
    return {"en": meta.get("name_en", rule_id), "de": meta.get("name_de", rule_id)}


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
    return Evidence(
        label=label, strong=bool(entry.get("strong")), rule_id=entry["rule"],
        rule_name=_rule_name(snap, entry["rule"]), family=entry["family"],
        horizon=entry["horizon"], side=entry["side"], study_date=snap.get("study_date", ""),
        judged=ev_series, oos_n=oos.get("n"), oos_mean_bps=oos.get("mean_bps"),
        oos_t=oos.get("nw_t"), oos_pos_share=oos.get("pos_share"),
        oos_ci_low_bps=oos.get("ci_low_bps"), oos_ci_high_bps=oos.get("ci_high_bps"),
        is_t=is_.get("nw_t"), window=window,
        excess_mean_bps=ex.get("mean_bps") if label == "beta" else None,
        excess_t=ex.get("nw_t") if label == "beta" else None)


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
    symbol: str
    direction: str                         # "Long" | "Short"
    horizon: str
    rule_id: str
    rank_score: float | None
    reference_price: float
    reference_date: str
    entry: dict                            # {"en": ..., "de": ...}
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

    def reasons_text(self, lang: str = "en") -> list[str]:
        prefix = {"en": "Counter-signal: ", "de": "Gegensignal: "}[lang if lang in ("en", "de") else "en"]
        out = []
        for code, params in self.reason_items:
            table = _RULE_REASONS.get(code) or _REASONS[code]
            out.append(("" if not params.get("against") else prefix)
                       + table.get(lang, table["en"]).format(**params))
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


def _size_pct(price: float, stop: float | None) -> float | None:
    if stop is None or price <= 0:
        return None
    frac = abs(price - stop) / price
    return min(MAX_POSITION_PCT, RISK_PER_TRADE_PCT / frac) if frac > 0 else None


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
               n_names: int) -> Idea:
    price, date, atr = reference
    mult = STOP_ATR[horizon]
    stop = None
    if atr is not None and np.isfinite(atr) and atr > 0:
        stop = max(price - side * mult * atr, 0.0)
    items: list[tuple[str, dict]] = []
    if panel is not None and rule_id != "T0" and rank is not None:
        items += _rule_reason(panel, rule_id, sym, rank, n_names)
    sig = signals.get(sym)
    if sig is not None:
        items += list(sig.reason_items)
    watch = evidence.label not in RECOMMENDED
    direction = "Long" if side > 0 else "Short"
    return Idea(
        symbol=sym, direction=direction, horizon=horizon, rule_id=rule_id, rank_score=score,
        reference_price=price, reference_date=date,
        entry={"en": "at the next open", "de": "zur nächsten Eröffnung"}, stop=stop,
        target=None, size_pct=_size_pct(price, stop),
        holding={"en": holding_text(horizon, family, "en"),
                 "de": holding_text(horizon, family, "de")},
        reason_items=items, evidence=evidence, watch_only=watch,
        stance=_stance(evidence.label, direction.lower()), rank=rank, rule_name=rule_name)


def _reference(panel: rules.Panel, sym: str) -> tuple[float, str, float | None]:
    atr = panel.atr()[sym].iloc[-1]
    return (float(panel.close[sym].iloc[-1]), str(panel.index[-1].date()),
            float(atr) if np.isfinite(atr) else None)


def _composite_members(signals: list[Signal], side: int, top: int) -> list[Signal]:
    want = "Long" if side > 0 else "Short"
    pool = [s for s in signals if s.direction == want]
    pool.sort(key=lambda s: (-s.score if side > 0 else s.score, s.symbol))
    return pool[:top]


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
                    "idea_relative_only", horizon, side_name, family, rule=rule_id,
                    mean=(evidence.oos_mean_bps or 0.0) / 100.0, mean_de=_de(
                        (evidence.oos_mean_bps or 0.0) / 100.0, 2),
                    t=evidence.oos_t or 0.0, t_de=_de(evidence.oos_t or 0.0, 1),
                    period_en=_PERIOD[horizon]["en"], period_de=_PERIOD[horizon]["de"],
                    date=snap.get("study_date", "")))
            if rule_id not in rules.RULES:
                logger.warning("evidence names unknown rule %s; skipped", rule_id)
                continue
            picks: list[tuple[str, float | None, int | None, tuple]] = []
            if rules.RULES[rule_id].base == "T0":
                for i, s in enumerate(_composite_members(signals, side, top), 1):
                    picks.append((s.symbol, s.score, i, (
                        s.close, s.last_date, s.atr)))
                n_names = len(signals)
            elif panel is not None:
                members = [m for m in rules.current_members(rule_id, panel) if m[1] == side]
                n_names = int(rules.rule_sides(rule_id, panel)[1].iloc[-1].notna().sum())
                for sym, _, score, rank in members[:top]:
                    picks.append((sym, score, rank, _reference(panel, sym)))
            else:
                n_names = 0
            if not picks:
                out.notes.append(_note(
                    "idea_no_members", horizon, side_name, family, rule=rule_id,
                    evidence_en=evidence.text("en"), evidence_de=evidence.text("de")))
                continue
            for sym, score, rank, ref in picks:
                out.ideas.append(_make_idea(panel, sym, side, horizon, rule_id, score, rank,
                                            evidence, family, rname, sig_by, ref, n_names))


def _watch_list(info, panel, signals, sig_by, snap, studied, horizon, side_name, side, top,
                out: IdeaSet) -> None:
    """No rule has evidence for this side: the composite's names as a labelled watch list."""
    family = info.family
    members = _composite_members(signals, side, top)
    if not snap or not studied:
        label = "not_studied"
        evidence = Evidence(label=label, family=family, horizon=horizon, side=side_name,
                            study_date=(snap or {}).get("study_date", ""))
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
    for i, s in enumerate(members, 1):
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
    date = snap.get("study_date", "")
    members = {s: f for s, f in frames.items() if s in set(info.members)}
    try:
        raw = live.shortterm_ideas(members, as_of, snapshot=snap, include_experimental=True) \
            if members else []
    except Exception as exc:     # a broken short-term snapshot must not take the scan down
        logger.warning("short-term ideas failed: %s", exc)
        raw = []
    strategies = snap.get("strategies", [])
    for horizon in (DAY, SWING):
        strats = [s for s in strategies if s.get("horizon") == horizon]
        mine = [r for r in raw if r.horizon == horizon]
        recommendable = False
        for r in mine[:max(top * 4, top)]:
            beta = r.beats_drift is False
            label = "beta" if beta else r.verdict
            ev = Evidence(
                label=label, strong=False, rule_id=r.strategy_id,
                rule_name={"en": r.strategy_name_en, "de": r.strategy_name_de}, family=family,
                horizon=horizon, side=r.direction.lower(), study_date=r.study_date,
                judged="trades", oos_n=r.oos_n_trades, oos_mean_bps=r.oos_mean_net_bps,
                oos_t=r.oos_t_stat, oos_pos_share=r.oos_win_rate,
                window=_oos_window(snap), excess_mean_bps=r.oos_excess_bps,
                excess_t=r.oos_excess_t_stat)
            recommendable |= label in RECOMMENDED
            out.ideas.append(Idea(
                symbol=r.symbol, direction=r.direction, horizon=horizon, rule_id=r.strategy_id,
                rank_score=None, reference_price=r.reference_price,
                reference_date=_last_date(members.get(r.symbol)),
                entry={"en": r.entry_rule_en, "de": r.entry_rule_de}, stop=r.stop, target=None,
                size_pct=_size_pct(r.reference_price, r.stop),
                holding={"en": f"{HORIZONS[horizon].label['en']}: {r.holding_en}",
                         "de": f"{HORIZONS[horizon].label['de']}: {r.holding_de}"},
                reason_items=[], evidence=ev, watch_only=label not in RECOMMENDED,
                stance=_stance(label, r.direction.lower()),
                rule_name={"en": r.strategy_name_en, "de": r.strategy_name_de}))
        if recommendable:
            continue
        passed = [s for s in strats if s.get("verdict") in ("validated", "experimental")]
        if passed and all(s.get("beats_drift") is False for s in passed):
            out.notes.append(_note(
                "idea_beta_only", horizon, None, family, n=len(strats), date=date,
                rules=", ".join(s["id"] for s in passed)))
        else:
            tfs = sorted({s.get("timeframe", "1d") for s in strats},
                         key=["1d", "60m", "5m"].index)
            out.notes.append(_note(
                "idea_no_rec_short_term", horizon, None, family, n=len(strats), date=date,
                tf_en=" and ".join(_TF[t]["en"] for t in tfs),
                tf_de=" und ".join(_TF[t]["de"].rstrip("-") for t in tfs) + "-"))


def _oos_window(snap: dict) -> tuple[str, str] | None:
    sp = (snap.get("splits") or {}).get("1d") or {}
    return (sp["oos_first"], sp["oos_last"]) if sp.get("oos_first") else None


def _last_date(frame) -> str:
    return "" if frame is None or frame.empty else str(frame.index[-1].date())


def build_ideas(
    info: UniverseInfo,
    frames: dict[str, pd.DataFrame],
    signals: list[Signal],
    regime: MarketRegime | None = None,
    evidence_snapshot: dict | None = None,
    shortterm_snapshot: dict | None = None,
    top: int = 10,
) -> IdeaSet:
    """Ideas per horizon and side for one scan, each with its evidence label.

    ``frames`` are the settled daily bars of the scan (members and benchmark),
    ``signals`` its composite signals.  ``evidence_snapshot=None`` reads the packaged
    ``evidence.json``; a missing or unreadable file is not an error: everything is then
    ``not_studied`` and a note says so.  ``regime`` is accepted for a stable signature
    (the composite signals already carry it).
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
    order = {DAY: 0, SWING: 1, MEDIUM: 2, LONG: 3}
    out.ideas.sort(key=lambda i: (order[i.horizon], i.direction != "Long", i.rank or 0))
    return out


__all__ = ["HORIZONS", "Evidence", "Idea", "IdeaSet", "build_ideas", "best_entry",
           "holding_text", "load_snapshot"]
