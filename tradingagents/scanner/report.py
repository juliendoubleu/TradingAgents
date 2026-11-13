"""The scan as a Markdown report, one self-contained HTML page and a JSON file.

The report is organised around what a reader can act on: first the recommendations per
investment horizon (only ideas whose rule has measured evidence), then a watch list that
is explicitly *not* a recommendation, then how solid the evidence is. The composite score
of the scanner has no measured edge; the report therefore never lets a watch-only idea
look like a call. ``Evidence.text(lang)`` is printed verbatim, never paraphrased.

The report is first built as a list of neutral blocks (headings, paragraphs, lists,
tables) and then rendered twice, so the Markdown and the HTML cannot drift apart. The
helpers that decide *what* is said (``horizon_views``, ``evidence_cells``,
``validation_verdict``, ``sizing_note`` ...) are public so the phone digest and the CLI
say the same thing. Every string exists in English and German, every number goes
through ``fmt.py``. The HTML loads nothing and runs nothing, like
``tradingagents.report_html``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.scanner.deep_dive import DeepDiveResult
from tradingagents.scanner.fmt import fmt_date, fmt_num, fmt_pct, fmt_price
from tradingagents.scanner.horizons import HORIZONS, Idea, IdeaSet, holding_text
from tradingagents.scanner.notices import Notice
from tradingagents.scanner.regime import LABELS as REGIME_LABELS, RISK_OFF_AT, RISK_ON_AT
from tradingagents.scanner.scan import ScanResult
from tradingagents.scanner.signals import (
    CONVICTION_HIGH,
    CONVICTION_MEDIUM,
    MAX_POSITION_PCT,
    RISK_PER_TRADE_PCT,
    STOP_ATR,
    TARGET_ATR,
    THRESHOLDS,
    WEIGHTS,
)
from tradingagents.scanner.universe import universe_info
from tradingagents.scanner.validation import CAVEATS, ValidationResult

# Shown order: the longest horizon first, because that is the only one with evidence today.
HORIZON_ORDER = ("long", "medium", "swing", "day")
LABEL_ORDER = ("validated", "experimental", "relative", "beta", "unproven", "not_studied")
SIDES = ("Long", "Short")

_LABEL = {
    "validated": ("Validated", "Belegt"),
    "experimental": ("Weakly supported", "Schwach belegt"),
    "relative": ("Relatively weaker", "Relativ schwächer"),
    "beta": ("Market return only", "Nur Marktrendite"),
    "unproven": ("Not proven", "Nicht belegt"),
    "not_studied": ("Not studied", "Nicht untersucht"),
}
_LABEL_HELP = {
    "validated": ("passed the pre-registered out-of-sample test",
                  "hat den vorab festgelegten Test außerhalb der Stichprobe bestanden"),
    "experimental": ("hints at an edge, preliminary, may be chance",
                     "deutet einen Vorteil an, vorläufig, kann Zufall sein"),
    "relative": ("these names lagged the average, but did not fall in absolute terms: avoid, "
                 "do not short",
                 "diese Werte blieben hinter dem Durchschnitt, fielen aber nicht absolut: meiden, "
                 "nicht shorten"),
    "beta": ("earns only what the market earned anyway",
             "verdient nur, was der Markt ohnehin verdiente"),
    "unproven": ("tested, no measurable edge", "getestet, kein messbarer Vorteil"),
    "not_studied": ("no measurement exists for this area", "für diesen Bereich liegt keine Messung vor"),
}

_T = {
    "title": ("Market scan", "Marktscan"),
    "universe": ("Universe", "Universum"),
    "scored": ("scored", "bewertet"),
    "benchmark": ("Benchmark", "Benchmark"),
    "asof": ("Data as of (newest bar used)", "Datenstand (neuester verwendeter Kurs)"),
    "requested": ("Requested date", "Angefragtes Datum"),
    "generated": ("Generated", "Erstellt"),
    "regime": ("Market regime", "Marktregime"),
    "regime_points": ("points", "Punkte"),
    "recs": ("Recommendations by investment horizon", "Empfehlungen nach Anlagehorizont"),
    "recs_intro": (
        "Only ideas whose rule passed a measured out-of-sample test appear here. Everything "
        "else is on the watch list below and is not a recommendation.",
        "Hier stehen nur Ideen, deren Regel einen gemessenen Test außerhalb der Stichprobe "
        "bestanden hat. Alles andere steht in der Beobachtungsliste weiter unten und ist "
        "keine Empfehlung."),
    "no_rec_any": (
        "No evidenced recommendation. The technical picture below is not evidenced: it is "
        "shown only so you can see what looks strong or weak right now.",
        "Keine belegte Empfehlung. Das technische Bild unten ist nicht belegt: es zeigt nur, "
        "was aktuell stark oder schwach aussieht."),
    "no_rec": ("No recommendation for this horizon.", "Keine Empfehlung für diesen Horizont."),
    "ideas_missing": (
        "The idea layer was not available for this scan: there are no recommendations.",
        "Die Ideen-Ebene war für diesen Scan nicht verfügbar: es gibt keine Empfehlungen."),
    "dir": ("Direction", "Richtung"), "ticker": ("Ticker", "Ticker"),
    "ref": ("Reference price (date)", "Referenzkurs (Datum)"),
    "entry": ("Entry", "Einstieg"), "stop": ("Stop", "Stop"),
    "no_stop": ("no stop", "kein Stop"),
    "size": ("Size (max.)", "Größe (max.)"), "holding": ("Holding period", "Haltedauer"),
    "evidence": ("Evidence", "Beleg"),
    "tech_weak_now": ("Technically weak right now: the scanner's technical picture says Short.",
                      "Technisch aktuell schwach: das technische Bild des Scanners sagt Short."),
    "tech_strong_now": ("Technically strong right now: the scanner's technical picture says Long.",
                        "Technisch aktuell stark: das technische Bild des Scanners sagt Long."),
    "sizing": (
        "Sizes are per trade and do NOT add up: several ideas together can exceed 100 % of "
        "your capital, so cap the total exposure yourself. The medium- and long-term rules "
        "were measured with a time exit and without a stop (a stop destroyed the measured "
        "edge), so they carry no stop and no price target; their size is equal weight in the "
        "measured basket (100 % divided by the number of names, at most {cap} %). For day-trade "
        "and swing rules the size is the share of capital that loses {risk} % if the stop is "
        "hit, capped at {cap} %. Entry is the next open after the reference close.",
        "Die Größen gelten je Trade und addieren sich NICHT: mehrere Ideen zusammen können über "
        "100 % deines Kapitals liegen, begrenze das Gesamtengagement daher selbst. Die mittel- "
        "und langfristigen Regeln wurden mit Zeit-Ausstieg und ohne Stop gemessen (ein Stop "
        "zerstörte den gemessenen Vorteil), sie tragen deshalb weder Stop noch Kursziel; ihre "
        "Größe ist gleichgewichtet im gemessenen Korb (100 % geteilt durch die Zahl der Werte, "
        "höchstens {cap} %). Bei Daytrade- und Swing-Regeln ist die Größe der Kapitalanteil, bei "
        "dem ein ausgelöster Stop {risk} % des Kapitals kostet, höchstens {cap} %. Einstieg ist "
        "die nächste Eröffnung nach dem Referenzkurs."),
    "watch": ("Technical picture (not evidenced) – watch list, not a recommendation",
              "Technisches Bild (nicht belegt) – Beobachtungsliste, keine Empfehlung"),
    "watch_intro": (
        "These names are NOT recommendations. The scanner's technical picture (trend, momentum, "
        "relative strength, MACD, RSI, volume) showed no measured edge in the tests; the lists "
        "only show what looks strong or weak right now.",
        "Diese Werte sind KEINE Empfehlungen. Das technische Bild des Scanners (Trend, Momentum, "
        "relative Stärke, MACD, RSI, Volumen) hatte in den Tests keinen gemessenen Vorteil; die "
        "Listen zeigen nur, was aktuell stark oder schwach aussieht."),
    "watch_specific": ("Specific to {h}", "Speziell für {h}"),
    "avoid": ("avoid / underweight", "meiden / untergewichten"),
    "strong": ("Technically strong right now", "Aktuell technisch stark"),
    "weak": ("Technically weak right now", "Aktuell technisch schwach"),
    "none_long": ("No name above the Long threshold.", "Kein Wert über der Long-Schwelle."),
    "none_short": ("No name below the Short threshold.", "Kein Wert unter der Short-Schwelle."),
    "rank": ("#", "#"), "score": ("Score", "Score"), "close": ("Last close", "Letzter Schlusskurs"),
    "why": ("Why", "Begründung"),
    "solid": ("How solid is this?", "Wie belastbar ist das?"),
    "area": ("Horizon", "Horizont"), "label": ("Rating", "Einstufung"),
    "rule": ("Rule", "Regel"),
    "mean": ("Mean per period", "Ø je Periode"), "tstat": ("t", "t"), "n": ("n", "n"),
    "window": ("Test window", "Testzeitraum"), "study": ("Study of", "Studie vom"),
    "labels_h": ("What the ratings mean", "Was die Einstufungen bedeuten"),
    "solid_none": (
        "No evidence file was available: nothing can be rated.",
        "Es lag keine Evidenzdatei vor: nichts kann eingestuft werden."),
    "solid_validated": (
        "Passed the out-of-sample test: {items}.",
        "Außerhalb der Stichprobe erfolgreich getestet: {items}."),
    "solid_experimental": ("Hints, preliminary: {items}.", "Hinweise, vorläufig: {items}."),
    "solid_nothing": (
        "Nothing is evidenced in this universe: no rule showed a measurable edge in the tests.",
        "In diesem Universum ist nichts belegt: keine Regel zeigte in den Tests einen "
        "messbaren Vorteil."),
    "solid_rest": ("Everything else is not evidenced: {items}.",
                   "Alles andere ist nicht belegt: {items}."),
    "solid_no_short": (
        "Short ideas are not evidenced anywhere: no Short call has a measured edge.",
        "Short-Ideen sind nirgends belegt: für keinen Short-Call gibt es einen gemessenen "
        "Vorteil."),
    "solid_short_term": (
        "Day-trade and swing rules were tested ({n} rules) and none passed after costs.",
        "Daytrade- und Swing-Regeln wurden getestet ({n} Regeln), keine bestand nach Kosten."),
    "solid_short_term_beta": (
        "Day-trade and swing rules were tested ({n} rules); the only one that passed earns no "
        "more than the market itself (market return).",
        "Daytrade- und Swing-Regeln wurden getestet ({n} Regeln); die einzige, die bestand, "
        "verdient nicht mehr als der Markt selbst (Marktrendite)."),
    "solid_date": ("Study of {date}; it is a snapshot and does not update itself.",
                   "Studie vom {date}; sie ist eine Momentaufnahme und aktualisiert sich nicht "
                   "selbst."),
    "validation": ("Validation of the technical picture", "Validierung des technischen Bildes"),
    "verdict_h": ("Verdict", "Urteil"),
    "val_intro": (
        "Signals recomputed at {n} past rebalance dates ({first} to {last}) using only bars up to "
        "each date, then measured over the next {h} trading days. {inst} instruments, "
        "universe {uni}, benchmark {bench}.",
        "Signale an {n} vergangenen Rebalancing-Terminen ({first} bis {last}), jeweils nur mit "
        "Kursen bis zu diesem Tag berechnet und danach über {h} Handelstage gemessen. "
        "{inst} Instrumente, Universum {uni}, Benchmark {bench}."),
    "val_none": (
        "Validation ran 0 rebalances: there was not enough history for a rebalance date with "
        "a following {h}-day outcome inside the requested window, so there is nothing to show.",
        "Die Validierung hatte 0 Rebalancing-Termine: für den gewählten Zeitraum gab es nicht "
        "genug Historie für einen Termin mit anschließenden {h} Handelstagen Ergebnis, daher "
        "gibt es nichts anzuzeigen."),
    "direction": ("Signal", "Signal"), "calls": ("Count", "Anzahl"),
    "avg_ret": ("Avg fwd return", "Ø Folgerendite"),
    "avg_exc": ("Avg excess vs benchmark", "Ø Überrendite vs. Benchmark"),
    "hit": ("Hit rate", "Trefferquote"), "pnl": ("Avg P&L of the position", "Ø P&L der Position"),
    "spread": ("Long minus Short spread, per date (average of the per-date spreads)",
               "Long-minus-Short-Spread je Termin (Mittel der Spreads je Termin)"),
    "spread_pooled": ("Long minus Short spread, pooled (average of all Long calls minus "
                      "average of all Short calls)",
                      "Long-minus-Short-Spread gepoolt (Mittel aller Long-Signale minus "
                      "Mittel aller Short-Signale)"),
    "tstat_val": ("t-statistic of the per-date spread ({n} dates)",
                  "t-Statistik des Spreads je Termin ({n} Termine)"),
    "baseline": ("Equal-weight average of all calls", "Gleichgewichteter Durchschnitt aller Signale"),
    "bench_avg": ("Benchmark over the same windows", "Benchmark über dieselben Fenster"),
    "caveats": ("Caveats", "Einschränkungen"),
    "deep": ("Deep-dive ratings (LLM agents)", "Deep-Dive-Ratings (LLM-Agenten)"),
    "scanner": ("Scanner", "Scanner"), "agents": ("Agents' rating", "Rating der Agenten"),
    "failed": ("failed", "fehlgeschlagen"),
    "notes": ("Notes", "Hinweise"), "skipped": ("Skipped instruments", "Übersprungene Instrumente"),
    "method": ("Method", "Methode"),
    "method_evidence": (
        "Where the evidence comes from: the medium and long-term study "
        "(tradingagents/scanner/evidence.json) and the day-trade/swing study "
        "(tradingagents/scanner/shortterm/validated.json). Both are fixed snapshots. Refresh them "
        "with `python -m tradingagents.scanner.horizon_research --out DIR` and "
        "`python -m tradingagents.scanner.shortterm.research --out DIR`.",
        "Woher die Belege stammen: die Studie für mittel- und langfristig "
        "(tradingagents/scanner/evidence.json) und die Studie für Daytrade/Swing "
        "(tradingagents/scanner/shortterm/validated.json). Beide sind feste Momentaufnahmen. "
        "Aktualisieren mit `python -m tradingagents.scanner.horizon_research --out DIR` und "
        "`python -m tradingagents.scanner.shortterm.research --out DIR`."),
    "method_score": (
        "Technical picture (watch list): score = 100 x weighted sum of six components, each "
        "between -1 and +1: {weights}.",
        "Technisches Bild (Beobachtungsliste): Score = 100 x gewichtete Summe aus sechs "
        "Komponenten zwischen -1 und +1: {weights}."),
    "method_dir": (
        "Strong / weak thresholds by regime (strong at or above / weak at or below): {thr}. "
        "Regime: risk-on at {on:+d} points or more, risk-off at {off:+d} or less. "
        "Strength of the signal: high from |score| {hi}, medium from {mid}. The scanner's own "
        "stop and target multiples are {stop} and {target} x ATR(14); the horizon rules use "
        "their own stops.",
        "Schwellen für stark / schwach je Regime (stark ab / schwach bis): {thr}. "
        "Regime: Risk-on ab {on:+d} Punkten, Risk-off bis {off:+d}. "
        "Stärke des Signals: hoch ab |Score| {hi}, mittel ab {mid}. Die eigenen Stop- und "
        "Ziel-Vielfachen des Scanners sind {stop} bzw. {target} x ATR(14); die Horizont-Regeln "
        "nutzen eigene Stops."),
    "comp": {
        "trend": ("trend", "Trend"), "momentum": ("momentum", "Momentum"),
        "relative_strength": ("relative strength", "relative Stärke"), "macd": ("MACD", "MACD"),
        "rsi_guard": ("RSI guard", "RSI-Filter"), "volume": ("volume", "Volumen"),
    },
    "disclaimer_h": ("Disclaimer", "Haftungsausschluss"),
    "disclaimer": (
        "This is a research tool, not investment advice. The signals are rule-based and "
        "derived from past prices only; they do not know about news, earnings or events. "
        "Position sizes are per trade and do not add up: cap your total exposure yourself. "
        "Short positions can lose more than the stake, in theory without limit. Stops can be "
        "gapped through. Check any idea yourself before acting on it.",
        "Dies ist ein Recherchewerkzeug und keine Anlageberatung. Die Signale sind regelbasiert "
        "und nur aus vergangenen Kursen abgeleitet; Nachrichten, Quartalszahlen oder Ereignisse "
        "kennt das Verfahren nicht. Positionsgrößen gelten je Trade und addieren sich nicht: "
        "begrenze dein Gesamtengagement selbst. Short-Positionen können mehr als den Einsatz "
        "kosten, theoretisch ohne Grenze. Stops können durch Kurslücken übersprungen werden. "
        "Prüfe jede Idee selbst, bevor du handelst."),
}

# What each "not a recommendation" note says, short (phone digest, CLI one-liners).
_SHORT_WHY = {
    "idea_no_snapshot": ("no evidence data", "keine Evidenzdaten"),
    "idea_not_studied": ("not studied", "nicht untersucht"),
    "idea_not_studied_watch": ("not studied", "nicht untersucht"),
    "idea_no_rec_rules": ("rules tested, no proven edge", "Regeln getestet, kein belegter Vorteil"),
    "idea_relative_only": ("only relative weakness proven: avoid, do not short",
                           "nur relative Schwäche belegt: meiden statt shorten"),
    "idea_no_members": ("the rule currently selects nothing", "die Regel wählt aktuell nichts aus"),
    "idea_robust_failed": ("passed the test but hangs on a few names, so it does not count",
                           "bestand den Test, hängt aber an wenigen Werten und zählt deshalb nicht"),
    "idea_no_rec_short_term": ("{n} rules tested, no edge after costs",
                               "{n} Regeln getestet, nach Kosten kein Vorteil"),
    "idea_beta_only": ("{n} rules tested, only market return", "{n} Regeln getestet, nur Marktrendite"),
    "idea_stale": ("the study is stale", "die Studie ist veraltet"),
}


def _i(lang: str) -> int:
    return 1 if lang == "de" else 0


def _t(key: str, lang: str):
    return _T[key][_i(lang)]


def tr(key: str, lang: str = "en") -> str:
    """A report string by key in ``lang``; the CLI uses it so both outputs share one wording."""
    return _t(key, _norm_lang(lang))


def label_text(label: str, lang: str = "en") -> str:
    """The reader-facing name of an evidence label (de: Belegt / Schwach belegt / ...)."""
    return _LABEL.get(label, (label, label))[_i(lang)]


def _norm_lang(lang: str | None) -> str:
    return "de" if lang == "de" else "en"


def _notice(item, lang: str) -> str:
    """A Notice in ``lang``; a plain string (older callers) as it is."""
    return item.text(lang) if hasattr(item, "text") else str(item)


def format_price(x: float | None) -> str:
    """Kept for older imports; ``fmt.fmt_price`` is the real implementation."""
    return fmt_price(x)


def _pct(x: float | None, digits: int = 1, signed: bool = True, lang: str = "en") -> str:
    return fmt_pct(x, digits, signed, lang)


# --- shared facts -----------------------------------------------------------------------------

_MARKET_NAME = {
    "us": ("US stocks", "US-Aktien"), "dax": ("DAX", "DAX"), "global": ("Global ETFs", "Global-ETFs"),
    "crypto": ("Crypto", "Krypto"), "sectors": ("US sectors", "US-Sektoren"),
    "all": ("All markets", "Alle Märkte"),
}


def market_name(result: ScanResult, lang: str = "en") -> str:
    """Short name of the market for headlines and digests."""
    if result.custom:
        return "Eigene Liste" if lang == "de" else "Custom list"
    pair = _MARKET_NAME.get(result.universe)
    return pair[_i(lang)] if pair else result.universe.upper()


def universe_description(result: ScanResult, lang: str = "en") -> str:
    """The universe description in ``lang``; the scan result carries both languages."""
    if result.descriptions:
        return result.describe(lang)
    if lang != "de":      # a hand-built result without descriptions: look the name up
        return result.description
    try:
        return universe_info(result.universe).description["de"]
    except (ValueError, KeyError):
        return result.description


def _regime_line(result: ScanResult, lang: str) -> str:
    regime = result.regime
    return (f"{REGIME_LABELS[regime.label][lang]} "
            f"({regime.points:+d} {_t('regime_points', lang)})")


def horizon_keys(horizons: Iterable[str] | None = None) -> tuple[str, ...]:
    """The wanted horizons in report order (all four when ``horizons`` is empty)."""
    if not horizons:
        return HORIZON_ORDER
    wanted = {h.strip().lower() for h in horizons}
    return tuple(h for h in HORIZON_ORDER if h in wanted)


def horizon_label(key: str, lang: str = "en") -> str:
    return HORIZONS[key].label[lang]


def sizing_note(lang: str = "en") -> str:
    """The line that says sizes are per trade and do not add up (review item L3)."""
    return _t("sizing", lang).format(risk=fmt_num(RISK_PER_TRADE_PCT, 0, lang),
                                     cap=fmt_num(MAX_POSITION_PCT, 0, lang))


def short_reason(note: Notice, lang: str = "en") -> str:
    """A note reduced to a clause for one-liners; the full sentence is ``note.text(lang)``."""
    template = _SHORT_WHY.get(note.code)
    if template is None:
        return note.text(lang)
    return template[_i(lang)].format(n=note.params.get("n", "?"))


@dataclass
class HorizonView:
    """One horizon of one scan: its evidenced ideas and, per empty side, why there are none."""

    key: str
    recs: list[Idea] = field(default_factory=list)
    why: dict[str, list[Notice]] = field(default_factory=lambda: {"Long": [], "Short": []})

    def side(self, direction: str) -> list[Idea]:
        return [i for i in self.recs if i.direction == direction]


def _by_rank(ideas: list[Idea]) -> list[Idea]:
    return sorted(ideas, key=lambda i: (i.rank if i.rank is not None else 10**6))


def horizon_views(result: ScanResult, horizons: Iterable[str] | None = None) -> list[HorizonView]:
    """Per horizon: the ideas that are NOT watch-only, Long then Short, and the notes that
    explain each side without one. A note without a side covers both sides."""
    ideas: IdeaSet | None = result.ideas
    views = []
    for key in horizon_keys(horizons):
        view = HorizonView(key)
        if ideas is not None:
            for side in SIDES:
                view.recs += _by_rank(ideas.select(key, side, recommended_only=True))
            for note in ideas.notes:
                if note.params.get("horizon") != key:
                    continue
                for side in SIDES:
                    if note.params.get("side") in (None, side.lower()) and not view.side(side):
                        view.why[side].append(note)
        views.append(view)
    return views


def composite_conflicts(result: ScanResult) -> dict[str, str]:
    """Recommended names that the scanner's technical picture sees the other way round.

    ``"weak"`` for a Long idea the composite calls Short, ``"strong"`` for the opposite.
    The idea itself carries the bilingual note (``Idea.conflict``); a hand-built idea
    without one is compared with the scan's signals.
    """
    out: dict[str, str] = {}
    if result.ideas is None:
        return out
    seen = {s.symbol: s.direction for s in result.signals}
    for idea in result.ideas.recommended():
        direction = seen.get(idea.symbol)
        if idea.conflict is None and direction not in ("Long", "Short"):
            continue
        if idea.direction == "Long" and (idea.conflict or direction == "Short"):
            out[idea.symbol] = "weak"
        elif idea.direction == "Short" and (idea.conflict or direction == "Long"):
            out[idea.symbol] = "strong"
    return out


def conflict_text(idea: Idea, flag: str | None, lang: str) -> str:
    """The note that the technical picture disagrees with an idea ('' when it does not)."""
    if idea.conflict:
        return idea.conflict.get(lang) or idea.conflict.get("en", "")
    if flag:
        return _t("tech_weak_now" if flag == "weak" else "tech_strong_now", lang)
    return ""


def general_notes(result: ScanResult, lang: str = "en") -> list[str]:
    """Notes about the whole idea set rather than one horizon (a stale study)."""
    ideas = result.ideas
    if ideas is None:
        return []
    return [n.text(lang) for n in ideas.notes if n.code == "idea_stale"]


def evidence_cells(result: ScanResult, horizons: Iterable[str] | None = None) -> list[dict]:
    """One row per horizon x side: the best label there and its judged numbers.

    The evidence comes from the ideas; where a cell has no idea (no member today, or day
    trades that were tested and failed) the label follows the note's code.
    """
    ideas = result.ideas
    rows = []
    for key in horizon_keys(horizons):
        for side in SIDES:
            found = ideas.select(key, side) if ideas is not None else []
            best = None
            for idea in found:
                if best is None or LABEL_ORDER.index(idea.evidence.label) < LABEL_ORDER.index(
                        best.label):
                    best = idea.evidence
            note = None
            if ideas is not None:
                note = next((n for n in ideas.notes if n.params.get("horizon") == key
                             and n.params.get("side") in (None, side.lower())), None)
            if best is not None:
                label = best.label
            elif note is not None and note.code in ("idea_not_studied", "idea_not_studied_watch"):
                label = "not_studied"
            elif note is not None:
                label = "unproven"
            else:
                label = "not_studied"
            rows.append({"horizon": key, "side": side, "label": label, "evidence": best,
                         "note": note})
    return rows


def _window_text(ev) -> str:
    if not ev or not ev.window:
        return "-"
    y1, y2 = ev.window[0][:4], ev.window[1][:4]
    return y1 if y1 == y2 else f"{y1}–{y2}"


def solid_paragraph(result: ScanResult, lang: str, horizons: Iterable[str] | None = None) -> str:
    """The plain-language 'what is evidenced and what is not', computed from the idea set."""
    if result.ideas is None or not result.ideas.evidence_available:
        return _t("solid_none", lang)
    cells = evidence_cells(result, horizons)
    parts = []

    def name(c):
        return f"{horizon_label(c['horizon'], lang)} {c['side']}"

    for label in ("validated", "experimental"):
        hit = [c for c in cells if c["label"] == label]
        if hit:
            items = "; ".join(
                f"{name(c)} ({c['evidence'].rule_name.get(lang) or c['evidence'].rule_id}) – "
                f"{c['evidence'].text(lang)}" if c["evidence"] else name(c) for c in hit)
            parts.append(_t(f"solid_{label}", lang).format(items=items))
    rest = [c for c in cells if c["label"] not in ("validated", "experimental")]
    if not parts:
        parts.append(_t("solid_nothing", lang))
    elif rest:
        parts.append(_t("solid_rest", lang).format(items=", ".join(
            f"{name(c)} ({label_text(c['label'], lang).lower()})" for c in rest)))
    if not any(c["side"] == "Short" and c["label"] in ("validated", "experimental")
               for c in cells):
        parts.append(_t("solid_no_short", lang))
    short_notes = [c["note"] for c in cells if c["horizon"] in ("day", "swing") and c["note"]
                   and c["note"].code in ("idea_no_rec_short_term", "idea_beta_only")]
    if short_notes:
        n = max(int(x.params.get("n", 0) or 0) for x in short_notes)
        beta = any(x.code == "idea_beta_only" for x in short_notes)
        parts.append(_t("solid_short_term_beta" if beta else "solid_short_term", lang).format(n=n))
    if result.ideas.study_date:
        parts.append(_t("solid_date", lang).format(date=fmt_date(result.ideas.study_date, lang)))
    parts.extend(general_notes(result, lang))
    return " ".join(parts)


def validation_verdict(v: ValidationResult, lang: str = "en") -> str:
    """One sentence computed from the validation numbers, saying plainly whether they show an edge.

    Long calls are compared with the equal-weight average of all calls (the no-skill
    baseline), Short calls by what the position made (a Short loses when its names rose).
    """
    lang = _norm_lang(lang)
    if v.rebalances == 0:
        return _t("val_none", lang).format(h=v.horizon)
    de = lang == "de"
    uni = v.universe_avg_return
    long_s, short_s = v.stats.get("Long"), v.stats.get("Short")
    long_diff = (long_s.avg_return - uni) if (long_s and long_s.avg_return is not None
                                              and uni is not None) else None
    short_pnl = short_s.avg_pnl if short_s else None
    pieces, oks = [], []
    if long_diff is not None:
        pts = fmt_num(abs(long_diff) * 100, 2, lang)
        above = long_diff > 0
        if de:
            pieces.append(f"Long-Signale {pts} Prozentpunkte {'über' if above else 'unter'} dem "
                          "Durchschnitt aller Werte")
        else:
            pieces.append(f"Long signals were {pts} points "
                          f"{'above' if above else 'below'} the average of all instruments")
        oks.append(above)
    if short_pnl is not None:
        amount = fmt_pct(abs(short_pnl), 2, False, lang)
        gained = short_pnl > 0
        beat = gained and (uni is None or short_s.avg_return is None
                           or short_s.avg_return < uni)
        if de:
            pieces.append(f"Short-Signale {'gewannen' if gained else 'verloren'} {amount}")
        else:
            pieces.append(f"Short signals {'gained' if gained else 'lost'} {amount}")
        oks.append(beat)
    if not pieces:
        return (f"Über die letzten {v.months} Monate lässt sich kein Urteil berechnen: zu wenige "
                "Signale." if de else
                f"Over the last {v.months} months no verdict can be computed: too few signals.")
    if de:
        # German verb-second order: "In den letzten N Monaten lagen Long-Signale ..., und ..."
        if long_diff is not None:
            pieces[0] = pieces[0].replace("Long-Signale", "lagen Long-Signale", 1)
        else:
            pieces[0] = pieces[0].replace("Short-Signale ", "", 1)
            pieces[0] = pieces[0].split(" ", 1)[0] + " Short-Signale " + pieces[0].split(" ", 1)[1]
    lead = (f"In den letzten {v.months} Monaten " if de else f"Over the last {v.months} months, ")
    body = pieces[0] + (f", und {pieces[1]}" if de and len(pieces) > 1
                        else f", and {pieces[1]}" if len(pieces) > 1 else "")
    if all(oks):
        end = ("ein Hinweis auf einen Vorteil, aber noch kein Beleg: die Stichprobe ist klein und "
               "Kosten sind nicht abgezogen" if de else
               "a hint of an edge, but not proof: the sample is small and costs are not deducted")
    elif not any(oks):
        end = "kein Hinweis auf einen Vorteil" if de else "no sign of an edge"
    else:
        end = ("gemischtes Bild, kein belastbarer Vorteil" if de else
               "a mixed picture, no reliable edge")
    return f"{lead}{body}: {end}."


# --- the report as neutral blocks -------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """A table cell with an optional badge (an evidence label); Markdown prints text, else badge."""

    text: str = ""
    badge: str | None = None
    kind: str = ""


Block = tuple  # ("h1"|"h2"|"h3", text) | ("p", text) | ("ul", [text]) | ("table", header, rows, tones)


def _idea_table(ideas: list[Idea], lang: str, conflicts: dict[str, str]) -> Block:
    header = [_t(k, lang) for k in ("dir", "ticker", "ref", "entry", "stop", "size", "holding",
                                    "evidence")]
    rows, tones = [], []
    for idea in ideas:
        sentence = idea.evidence.text(lang)
        flag = conflict_text(idea, conflicts.get(idea.symbol), lang)
        if flag:
            sentence += " " + flag
        if idea.outside:
            sentence += " " + idea.outside.get(lang, idea.outside["en"])
        rows.append([
            idea.direction, idea.symbol,
            f"{fmt_price(idea.reference_price, lang)} ({fmt_date(idea.reference_date, lang)})",
            idea.entry.get(lang) or idea.entry.get("en", "-"),
            fmt_price(idea.stop, lang) if idea.stop is not None else _t("no_stop", lang),
            "-" if idea.size_pct is None else f"{fmt_num(idea.size_pct, 1, lang)} %",
            idea.holding.get(lang) or idea.holding.get("en", "-"),
            Cell(sentence, label_text(idea.evidence.label, lang), idea.evidence.label),
        ])
        tones.append(idea.direction.lower())
    return ("table", header, rows, tones)


def size_notes(ideas: list[Idea], lang: str) -> list[str]:
    """The distinct size / no-stop explanations of ``ideas`` (one per basket, not per name)."""
    return list(dict.fromkeys(i.size_note.get(lang) or i.size_note["en"]
                              for i in ideas if i.size_note))


def _signal_table(signals, lang: str, tone: str) -> Block:
    header = [_t(k, lang) for k in ("rank", "ticker", "score", "close", "why")]
    rows = []
    for i, s in enumerate(signals, 1):
        rows.append([str(i), s.symbol, ("+" if s.score > 0 else "") + fmt_num(s.score, 1, lang),
                     fmt_price(s.close, lang),
                     "; ".join(s.reasons_text(lang))])
    return ("table", header, rows, [tone] * len(rows))


def _specific_watch(result: ScanResult, keys: Sequence[str]) -> list[Idea]:
    """Watch-only ideas that carry their own finding (market return only, relatively weaker,
    short-term). The composite's names that medium and long share are listed once, separately."""
    if result.ideas is None:
        return []
    return [i for i in result.ideas.ideas
            if i.watch_only and i.horizon in keys
            and (i.evidence.label in ("beta", "relative") or i.horizon in ("day", "swing"))]


def _watch_table(ideas: list[Idea], lang: str) -> Block:
    """Watch items carry no entry, stop or size: they are not a plan."""
    header = [_t(k, lang) for k in ("dir", "ticker", "ref", "evidence")]
    rows, tones = [], []
    for idea in ideas:
        avoid = idea.stance == "avoid"
        rows.append([
            _t("avoid", lang) if avoid else idea.direction, idea.symbol,
            f"{fmt_price(idea.reference_price, lang)} ({fmt_date(idea.reference_date, lang)})",
            Cell(idea.evidence.text(lang), label_text(idea.evidence.label, lang),
                 idea.evidence.label)])
        tones.append("avoid" if avoid else "")
    return ("table", header, rows, tones)


def _rule_cell(ev, lang: str) -> str:
    """The rule's plain name; its id only in parentheses, here in the evidence table."""
    if ev is None or not ev.rule_id:
        return "-"
    name = ev.rule_name.get(lang) or ev.rule_name.get("en") or ev.rule_id
    return f"{name} ({ev.rule_id})"


def _solid_blocks(result: ScanResult, lang: str, keys: Sequence[str]) -> list[Block]:
    blocks: list[Block] = [("h2", _t("solid", lang))]
    if result.ideas is None or not result.ideas.evidence_available:
        blocks.append(("p", _t("solid_none", lang)))
        return blocks
    header = [_t(k, lang) for k in ("area", "dir", "rule", "label", "mean", "tstat", "n", "window",
                                    "study")]
    rows = []
    for c in evidence_cells(result, keys):
        ev = c["evidence"]
        has = ev is not None and ev.oos_mean_bps is not None
        rows.append([
            holding_text(c["horizon"], result.ideas.family, lang).split(" – ")[0].split(":")[0],
            c["side"], _rule_cell(ev, lang), Cell("", label_text(c["label"], lang), c["label"]),
            fmt_pct(ev.oos_mean_bps / 10000.0, 2, True, lang) if has else "-",
            fmt_num(ev.oos_t, 1, lang) if has and ev.oos_t is not None else "-",
            str(ev.oos_n) if has and ev.oos_n else "-",
            _window_text(ev) if has else "-",
            fmt_date(ev.study_date or result.ideas.study_date, lang) if ev is not None
            else fmt_date(result.ideas.study_date, lang),
        ])
    blocks.append(("table", header, rows, [""] * len(rows)))
    blocks.append(("p", solid_paragraph(result, lang, keys)))
    blocks.append(("p", f"{_t('labels_h', lang)}:"))
    blocks.append(("ul", [f"{label_text(k, lang)}: {_LABEL_HELP[k][_i(lang)]}" for k in LABEL_ORDER]))
    return blocks


def build_blocks(
    result: ScanResult,
    validation: ValidationResult | None = None,
    deep: DeepDiveResult | None = None,
    lang: str = "en",
    horizons: Iterable[str] | None = None,
) -> list[Block]:
    lang = _norm_lang(lang)
    keys = horizon_keys(horizons)
    blocks: list[Block] = [("h1", f"{_t('title', lang)}: {result.universe.upper()}")]
    blocks.append(("ul", [
        f"{_t('universe', lang)}: {universe_description(result, lang)} ({result.universe}, "
        f"{len(result.signals)} {_t('scored', lang)})",
        f"{_t('benchmark', lang)}: {result.benchmark}",
        f"{_t('asof', lang)}: {fmt_date(result.as_of_date, lang) if result.as_of_date else '-'}",
        f"{_t('requested', lang)}: {fmt_date(result.requested_date, lang)}",
        f"{_t('generated', lang)}: {_generated(result.generated, lang)}",
    ]))

    blocks.append(("h2", _t("regime", lang)))
    blocks.append(("p", _regime_line(result, lang)))
    blocks.append(("ul", result.regime.describe(lang)))

    blocks += _recommendation_blocks(result, keys, lang)
    blocks += _watch_blocks(result, keys, lang)
    blocks += _solid_blocks(result, lang, keys)

    if validation is not None:
        blocks += _validation_blocks(validation, lang)
    if deep is not None:
        blocks += _deep_blocks(result, deep, lang)

    if result.warnings or result.skipped:
        blocks.append(("h2", _t("notes", lang)))
        if result.warnings:
            blocks.append(("ul", [_notice(w, lang) for w in result.warnings]))
        if result.skipped:
            blocks.append(("p", f"{_t('skipped', lang)}:"))
            blocks.append(("ul", [f"{sym}: {_notice(why, lang)}" for sym, why in result.skipped]))

    blocks.append(("h2", _t("method", lang)))
    comp = _T["comp"]
    weights = ", ".join(f"{comp[k][_i(lang)]} {fmt_num(w * 100, 0, lang)} %"
                        for k, w in WEIGHTS.items())
    thr = "; ".join(f"{name} {fmt_num(up, 0, lang)} / {fmt_num(down, 0, lang)}"
                    for name, (up, down) in THRESHOLDS.items())
    blocks.append(("p", _t("method_evidence", lang)))
    blocks.append(("p", _t("method_score", lang).format(weights=weights)))
    blocks.append(("p", _t("method_dir", lang).format(
        thr=thr, on=RISK_ON_AT, off=RISK_OFF_AT, hi=fmt_num(CONVICTION_HIGH, 0, lang),
        mid=fmt_num(CONVICTION_MEDIUM, 0, lang), stop=fmt_num(STOP_ATR, 0, lang),
        target=fmt_num(TARGET_ATR, 0, lang))))
    blocks.append(("h2", _t("disclaimer_h", lang)))
    blocks.append(("p", _t("disclaimer", lang)))
    return blocks


def _generated(text: str, lang: str) -> str:
    """'2026-11-13 15:52' as '13.11.2026 15:52' for German."""
    if not text:
        return "-"
    day, _, clock = text.partition(" ")
    return f"{fmt_date(day, lang)} {clock}".strip()


def _recommendation_blocks(result: ScanResult, keys: Sequence[str], lang: str) -> list[Block]:
    blocks: list[Block] = [("h2", _t("recs", lang)), ("p", _t("recs_intro", lang))]
    if result.ideas is None:
        blocks.append(("p", _t("ideas_missing", lang)))
        return blocks
    views = horizon_views(result, keys)
    if not any(v.recs for v in views):
        blocks.append(("p", _t("no_rec_any", lang)))
    blocks.extend(("p", text) for text in general_notes(result, lang))
    conflicts = composite_conflicts(result)
    family = result.ideas.family
    for view in views:
        blocks.append(("h3", holding_text(view.key, family, lang)))
        if view.recs:
            blocks.append(_idea_table(view.recs, lang, conflicts))
            notes = size_notes(view.recs, lang)
            if notes:
                blocks.append(("ul", notes))
        why = []
        for side in SIDES:
            if view.side(side):
                continue
            for note in view.why[side]:
                text = note.text(lang)
                if text not in why:
                    why.append(text)
        if not view.recs and not why:
            why.append(_t("no_rec", lang))
        if why:
            blocks.append(("ul", why))
    blocks.append(("p", sizing_note(lang)))
    return blocks


def _watch_blocks(result: ScanResult, keys: Sequence[str], lang: str) -> list[Block]:
    blocks: list[Block] = [
        ("h2", _t("watch", lang)),
        ("p", _t("watch_intro", lang))]
    family = result.ideas.family if result.ideas is not None else ""
    specific = _specific_watch(result, keys)
    for key in keys:
        group = [i for i in specific if i.horizon == key]
        if group:
            blocks.append(("h3", _t("watch_specific", lang).format(
                h=holding_text(key, family, lang))))
            blocks.append(_watch_table(group, lang))
    blocks.append(("h3", _t("strong", lang)))
    blocks.append(_signal_table(result.longs, lang, "") if result.longs
                  else ("p", _t("none_long", lang)))
    blocks.append(("h3", _t("weak", lang)))
    blocks.append(_signal_table(result.shorts, lang, "") if result.shorts
                  else ("p", _t("none_short", lang)))
    return blocks


def _validation_blocks(v: ValidationResult, lang: str) -> list[Block]:
    blocks: list[Block] = [("h2", _t("validation", lang)),
                           ("p", f"{_t('verdict_h', lang)}: {validation_verdict(v, lang)}")]
    if v.rebalances:
        blocks.append(("p", _t("val_intro", lang).format(
            n=v.rebalances, first=fmt_date(v.first_rebalance, lang),
            last=fmt_date(v.last_rebalance, lang), h=v.horizon,
            inst=v.instruments, uni=v.universe, bench=v.benchmark)))
        header = [_t(k, lang) for k in ("direction", "calls", "avg_ret", "avg_exc", "hit", "pnl")]
        rows, tones = [], []
        for name, tone in (("Long", "long"), ("Short", "short"), ("Neutral", "")):
            s = v.stats[name]
            rows.append([name, str(s.calls), _pct(s.avg_return, 2, lang=lang),
                         _pct(s.avg_excess, 2, lang=lang),
                         _pct(s.hit_rate, 0, signed=False, lang=lang),
                         _pct(s.avg_pnl, 2, lang=lang)])
            tones.append(tone)
        blocks.append(("table", header, rows, tones))
        facts = [f"{_t('spread', lang)}: {_pct(v.spread, 2, lang=lang)}",
                 f"{_t('spread_pooled', lang)}: {_pct(v.spread_pooled, 2, lang=lang)}"]
        if v.spread_tstat is not None:
            facts.append(f"{_t('tstat_val', lang).format(n=v.spread_dates)}: "
                         f"{fmt_num(v.spread_tstat, 2, lang)}")
        facts.append(f"{_t('baseline', lang)}: {_pct(v.universe_avg_return, 2, lang=lang)}")
        facts.append(f"{_t('bench_avg', lang)}: {_pct(v.benchmark_avg_return, 2, lang=lang)}")
        blocks.append(("ul", facts))
    blocks.append(("p", f"{_t('caveats', lang)}:"))
    blocks.append(("ul", list(CAVEATS[lang])))
    return blocks


def _deep_blocks(result: ScanResult, deep: DeepDiveResult, lang: str) -> list[Block]:
    by_symbol = {s.symbol: s for s in [*result.longs, *result.shorts]}
    rows, tones = [], []
    for symbol, rating in deep.ratings.items():
        sig = by_symbol.get(symbol)
        rows.append([symbol, sig.direction if sig else "-", rating])
        tones.append(sig.direction.lower() if sig else "")
    for symbol, error in deep.errors.items():
        sig = by_symbol.get(symbol)
        rows.append([symbol, sig.direction if sig else "-", f"{_t('failed', lang)}: {error}"])
        tones.append("")
    return [("h2", _t("deep", lang)),
            ("table", [_t("ticker", lang), _t("scanner", lang), _t("agents", lang)], rows, tones)]


# --- Markdown -----------------------------------------------------------------------------------

def _md_cell(cell) -> str:
    text = (cell.text or cell.badge or "") if isinstance(cell, Cell) else str(cell)
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(
    result: ScanResult,
    validation: ValidationResult | None = None,
    deep: DeepDiveResult | None = None,
    lang: str = "en",
    horizons: Iterable[str] | None = None,
) -> str:
    out: list[str] = []
    for block in build_blocks(result, validation, deep, lang, horizons):
        kind = block[0]
        if kind in ("h1", "h2", "h3"):
            out.append(f"{'#' * int(kind[1])} {block[1]}\n")
        elif kind == "p":
            out.append(f"{block[1]}\n")
        elif kind == "ul":
            out.append("\n".join(f"- {item}" for item in block[1]) + "\n")
        elif kind == "table":
            _, header, rows, _tones = block
            lines = ["| " + " | ".join(_md_cell(h) for h in header) + " |",
                     "|" + "|".join("---" for _ in header) + "|"]
            lines += ["| " + " | ".join(_md_cell(c) for c in row) + " |" for row in rows]
            out.append("\n".join(lines) + "\n")
    return "\n".join(out)


# --- HTML ---------------------------------------------------------------------------------------

_CSS = """
:root { --paper:#fbfbfa; --ink:#1c2126; --muted:#5d6670; --rule:#e2e5e8; --wash:#f2f4f5;
  --long:#087a59; --short:#b4372f; --avoid:#9a5b00; --amber:#8a6100; color-scheme: light dark;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
@media (prefers-color-scheme: dark) {
  :root { --paper:#131a1f; --ink:#e4e8eb; --muted:#9aa6af; --rule:#263239; --wash:#1a242a;
    --long:#14c290; --short:#ef6f63; --avoid:#e0a040; --amber:#e0b04a; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font:16px/1.55 var(--sans); }
main { max-width: 72rem; margin: 0 auto; padding: 1.5rem 1rem 4rem; }
h1 { font-size: 1.8rem; letter-spacing: -0.02em; margin: 0 0 1rem; }
h2 { font-size: 1.3rem; margin: 2.2rem 0 0.75rem; padding-top: 1rem; border-top: 2px solid var(--ink); }
h3 { font-size: 1.05rem; margin: 1.4rem 0 0.5rem; }
ul { padding-left: 1.2rem; } li { margin: 0.2rem 0; }
.table { overflow-x: auto; margin: 0 0 1rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; font-variant-numeric: tabular-nums; }
th, td { padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; border-bottom: 1px solid var(--rule); }
th { font-weight: 600; border-bottom-color: var(--muted); }
tr.long td:first-child { border-left: 4px solid var(--long); color: var(--long); font-weight: 600; }
tr.short td:first-child { border-left: 4px solid var(--short); color: var(--short); font-weight: 600; }
tr.avoid td:first-child { border-left: 4px solid var(--avoid); color: var(--avoid); font-weight: 600; }
.badge { display: inline-block; padding: 0 0.5rem; margin-right: 0.3rem; border-radius: 0.7rem;
  border: 1px solid var(--muted); color: var(--muted); font-size: 0.75rem; font-weight: 600;
  white-space: nowrap; }
.b-validated { border-color: var(--long); color: var(--long); }
.b-experimental { border-color: var(--amber); color: var(--amber); }
.b-relative { border-color: var(--avoid); color: var(--avoid); }
.disclaimer { padding: 0.8rem 1rem; background: var(--wash); border-left: 3px solid var(--short); }
@media print { body { font-size: 11pt; } main { padding: 0; } }
"""


def _html_cell(cell) -> str:
    if isinstance(cell, Cell):
        badge = (f'<span class="badge b-{escape(cell.kind)}">{escape(cell.badge)}</span>'
                 if cell.badge else "")
        return badge + escape(cell.text)
    return escape(str(cell))


def render_html(
    result: ScanResult,
    validation: ValidationResult | None = None,
    deep: DeepDiveResult | None = None,
    lang: str = "en",
    horizons: Iterable[str] | None = None,
) -> str:
    lang = _norm_lang(lang)
    body: list[str] = []
    in_disclaimer = False
    for block in build_blocks(result, validation, deep, lang, horizons):
        kind = block[0]
        if kind == "h1":
            body.append(f"<h1>{escape(block[1])}</h1>")
        elif kind in ("h2", "h3"):
            if kind == "h2":
                in_disclaimer = block[1] == _t("disclaimer_h", lang)
            body.append(f"<{kind}>{escape(block[1])}</{kind}>")
        elif kind == "p":
            cls = ' class="disclaimer"' if in_disclaimer else ""
            body.append(f"<p{cls}>{escape(block[1])}</p>")
        elif kind == "ul":
            body.append("<ul>" + "".join(f"<li>{escape(i)}</li>" for i in block[1]) + "</ul>")
        elif kind == "table":
            _, header, rows, tones = block
            head = "".join(f"<th>{escape(h)}</th>" for h in header)
            trs = "".join(
                f'<tr class="{escape(tone)}">' + "".join(f"<td>{_html_cell(c)}</td>" for c in row)
                + "</tr>" for row, tone in zip(rows, tones, strict=True))
            body.append(f'<div class="table"><table><thead><tr>{head}</tr></thead>'
                        f"<tbody>{trs}</tbody></table></div>")
    title = f"{_t('title', lang)}: {result.universe.upper()} {fmt_date(result.requested_date, lang)}"
    return f"""<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>{escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
{chr(10).join(body)}
</main>
</body>
</html>
"""


# --- saving -------------------------------------------------------------------------------------

def scan_directory(result: ScanResult, results_dir: str | Path) -> Path:
    name = f"{result.universe}{'-custom' if result.custom else ''}_{result.requested_date}"
    return Path(results_dir) / "scans" / safe_ticker_component(name, max_len=64)


def save_scan(
    result: ScanResult,
    validation: ValidationResult | None = None,
    deep: DeepDiveResult | None = None,
    lang: str = "en",
    results_dir: str | Path = ".",
    horizons: Iterable[str] | None = None,
) -> dict[str, Path]:
    """Write ``scan.md``, ``scan.html``, ``scan.json`` and ``digest.txt`` under
    ``<results_dir>/scans/<name>/``. The JSON carries the ideas with their evidence."""
    from tradingagents.scanner.digest import build_digest  # digest imports this module

    lang = _norm_lang(lang)
    folder = scan_directory(result, results_dir)
    folder.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    if validation is not None:
        payload["validation"] = validation.to_dict()
        payload["validation"]["verdict"] = validation_verdict(validation, "en")
        payload["validation"]["verdict_de"] = validation_verdict(validation, "de")
    if deep is not None:
        payload["deep_dive"] = {"ratings": deep.ratings, "errors": deep.errors}
    paths = {"md": folder / "scan.md", "html": folder / "scan.html", "json": folder / "scan.json",
             "digest": folder / "digest.txt"}
    paths["md"].write_text(render_markdown(result, validation, deep, lang, horizons),
                           encoding="utf-8")
    paths["html"].write_text(render_html(result, validation, deep, lang, horizons),
                             encoding="utf-8")
    paths["json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    digest = build_digest([result], lang, horizons=horizons)
    paths["digest"].write_text(digest.text + "\n", encoding="utf-8")
    return paths
