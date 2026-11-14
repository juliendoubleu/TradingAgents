"""Factual assessment and comparison of the instruments the owner asks about.

The scanner already knows, per investment horizon, which names a rule with measured
evidence puts in its Long or Short basket and which names only sit on the composite's
watch list.  This module answers the narrower question "what do the data say about THESE
instruments?": it scans each asked ticker inside its peer group (so the cross-sectional
rules rank it among real peers and breadth/regime are real), then reports per ticker and
horizon one of four states and compares the tickers by them.

Honesty rules the code enforces rather than hopes for:

* ``Evidence.text(lang)`` is printed verbatim; no sentence here restates or upgrades it.
* A watch-list name is never worded as a recommendation.  Only a rule whose evidence
  label is ``validated`` or ``experimental`` (``Idea.watch_only`` False) counts as one.
* A ticker can be an evidenced Long and technically weak at once; the conflict is shown.
* Nothing is promised and an unknown ticker is reported, not guessed.

The summary sentences are ``notices.Notice`` objects (code ``assess_summary``, registered
below): they keep a code and parameters and render in German or English on demand.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from tradingagents.dataflows.errors import VendorError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.scanner import notices
from tradingagents.scanner.aliases import display_name
from tradingagents.scanner.events import Snapshot, performance_snapshot, render_implied_moves
from tradingagents.scanner.fmt import fmt_date, fmt_num, fmt_pct, fmt_price
from tradingagents.scanner.horizons import HORIZONS, Idea, watch_direction
from tradingagents.scanner.notices import Notice, label_name
from tradingagents.scanner.regime import LABELS as REGIME_LABELS
from tradingagents.scanner.scan import parse_date, run_scan
from tradingagents.scanner.signals import Signal
from tradingagents.scanner.universe import UNIVERSES

logger = logging.getLogger(__name__)

Loader = Callable[[list[str], str, int], dict[str, pd.DataFrame]]

HORIZON_ORDER = ("day", "swing", "medium", "long")
# Every basket member must be visible, so ask the scan for "all of them".
_ALL = 10_000
# Statuses, best first.  "short_rec" (an evidenced Short) sorts with the weak names: for
# a reader choosing between names to hold, a name that is only a Short candidate is last.
STATUS_ORDER = {"recommended": 0, "watch": 1, "none": 2, "weak": 3, "short_rec": 3}

_FAMILY_LABEL = {"stocks_us": ("US-Aktien", "US stocks"), "stocks_dax": ("DAX-Aktien", "DAX stocks"),
                 "etfs": ("ETFs", "ETFs"), "crypto": ("Krypto", "crypto")}
_UNIVERSE_FOR_FAMILY = {"stocks_us": "us", "stocks_dax": "dax", "crypto": "crypto"}
_SHOWN_WARNINGS = ("stale_data", "benchmark_missing", "vix_missing", "regime_input_missing",
                   "past_date")


def label_text(label: str, lang: str = "en") -> str:
    return label_name(label, lang).lower()


def _t(de: str, en: str, lang: str) -> str:
    return de if lang == "de" else en


def _join(items: list[str], lang: str) -> str:
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {_t('und', 'and', lang)} {items[-1]}"


def _score(x: float, lang: str) -> str:
    return ("+" if x >= 0 else "") + fmt_num(x, 0, lang)


# --------------------------------------------------------------------------- statements
# The summary sentences are ``Notice`` objects like every other message of the scanner; this
# module registers the one code it owns (its wording needs the entries, not a template).

def _tech_clause(entries: list[dict], lang: str) -> str:
    """The technical-picture comparison; empty when fewer than two scores or all equal."""
    scored = sorted((e for e in entries if e.get("score") is not None),
                    key=lambda e: -e["score"])
    if len(scored) < 2 or scored[0]["score"] == scored[-1]["score"]:
        return ""
    top, bottom = scored[0], scored[-1]
    if len(scored) == 2:
        return _t(f"technisch ist {top['name']} stärker (Score {_score(top['score'], lang)} "
                  f"gegen {_score(bottom['score'], lang)})",
                  f"technically {top['name']} is stronger (score {_score(top['score'], lang)} "
                  f"against {_score(bottom['score'], lang)})", lang)
    return _t(f"technisch am stärksten: {top['name']}, Score {_score(top['score'], lang)}, "
              f"am schwächsten: {bottom['name']}, Score {_score(bottom['score'], lang)}",
              f"technically strongest: {top['name']}, score {_score(top['score'], lang)}, "
              f"weakest: {bottom['name']}, score {_score(bottom['score'], lang)}", lang)


def _single_text(h: str, e: dict, lang: str) -> str:
    lab = HORIZONS[h].label[lang if lang in ("en", "de") else "en"]
    name, status, label = e["name"], e["status"], label_text(e["label"], lang)
    if status == "recommended":
        weak = _t(" – technisch ist der Wert aktuell aber schwach",
                  " - technically the name is currently weak though", lang) if e["conflict"] else ""
        return _t(f"{lab}: Für {name} gibt es eine Long-Empfehlung ({label}){weak}.",
                  f"{lab}: {name} has a Long recommendation ({label}){weak}.", lang)
    if status == "watch":
        return _t(f"{lab}: Für {name} gibt es keine belegte Empfehlung, nur einen "
                  f"Beobachtungswert ({watch_direction(e['direction'], lang)}, {label}).",
                  f"{lab}: there is no evidenced recommendation for {name}, only a watch item "
                  f"({watch_direction(e['direction'], lang)}, {label}).", lang)
    if status == "short_rec":
        return _t(f"{lab}: {name} ist ein Short-Kandidat mit Beleg ({label}).",
                  f"{lab}: {name} is a Short candidate with evidence ({label}).", lang)
    if status == "weak":
        return _t(f"{lab}: {name} steht auf der Beobachtungsliste, technisch schwach ({label}); "
                  "keine Empfehlung.",
                  f"{lab}: {name} is on the watch list, technically weak ({label}); "
                  "no recommendation.", lang)
    return _t(f"{lab}: Für {name} gibt es hier kein Signal.",
              f"{lab}: there is no signal for {name} here.", lang)


def _summary_text(p: dict, lang: str) -> str:
    h, ents = p["horizon"], p["entries"]
    if len(ents) == 1:
        return _single_text(h, ents[0], lang)
    lab = HORIZONS[h].label[lang if lang in ("en", "de") else "en"]
    rec = [e for e in ents if e["status"] == "recommended"]
    rest = [e for e in ents if e["status"] != "recommended"]
    n = len(ents)
    # no ranking for a horizon where a name has no idea ("Keine Idee für" in its own block)
    tech = (_tech_clause([e for e in ents if e["status"] != "none"], lang)
            if h in ("medium", "long") else "")
    adjective_de = "schwach belegte" if any(e["label"] == "experimental" for e in rec) else "belegte"
    adjective_en = "weakly supported" if any(e["label"] == "experimental" for e in rec) else "evidenced"
    if not rec:
        head = _t("Für keinen der beiden Werte gibt es eine belegte Empfehlung" if n == 2
                  else f"Für keinen der {n} Werte gibt es eine belegte Empfehlung",
                  "There is no evidenced recommendation for either of the two names" if n == 2
                  else f"There is no evidenced recommendation for any of the {n} names", lang)
    elif not rest:
        head = _t(f"Für {'beide Werte' if n == 2 else f'alle {n} Werte'} gibt es eine "
                  f"{adjective_de} Long-Empfehlung",
                  f"{'Both names have' if n == 2 else f'All {n} names have'} an "
                  f"{adjective_en} Long recommendation", lang)
    else:
        names = _join([e["name"] for e in rec], lang)
        others = _join([e["name"] for e in rest], lang)
        head = _t(f"Eine {adjective_de} Long-Empfehlung gibt es nur für {names}, nicht für {others}",
                  f"An {adjective_en} Long recommendation exists only for {names}, not for "
                  f"{others}", lang)
    weak = [e["name"] for e in rec if e["conflict"]]
    if weak:
        head += _t(f" ({_join(weak, lang)} technisch aktuell aber schwach)",
                   f" ({_join(weak, lang)} technically weak right now though)", lang)
    return f"{lab}: {head}{'; ' + tech if tech else ''}."


def collapse_summary(items: list[Notice], lang: str) -> list[str]:
    """One line per distinct statement: horizons that say the same thing are joined
    ("Daytrade, Kurzfristig (Swing): Für keinen der 2 Werte gibt es ...")."""
    groups: dict[str, list[str]] = {}
    for item in items:
        h = item.params["horizon"]
        lab = HORIZONS[h].label[lang if lang in ("en", "de") else "en"]
        text = item.text(lang)
        body = text[len(lab) + 2:] if text.startswith(f"{lab}: ") else text
        groups.setdefault(body, []).append(lab)
    return [f"{', '.join(labels)}: {body}" for body, labels in groups.items()]


notices.register({"assess_summary": {"en": lambda p: _summary_text(p, "en"),
                                      "de": lambda p: _summary_text(p, "de")}})


# --------------------------------------------------------------------------- data classes

@dataclass
class HorizonView:
    """What one horizon says about one ticker: the status and the ideas behind it."""

    horizon: str
    status: str = "none"                 # recommended | watch | none | weak | short_rec
    ideas: list[Idea] = field(default_factory=list)
    notes: list[Notice] = field(default_factory=list)
    conflict: bool = False               # recommended, yet technically weak right now

    @property
    def lead(self) -> Idea | None:
        """The idea that decides the status (a recommendation first)."""
        if not self.ideas:
            return None
        order = {"recommended": 0, "watch": 1, "weak": 1, "short_rec": 0}
        return sorted(self.ideas, key=lambda i: (order.get(_idea_status(i), 2),
                                                 i.direction != "Long"))[0]

    def to_dict(self) -> dict:
        return {"horizon": self.horizon, "status": self.status, "conflict": self.conflict,
                "ideas": [i.to_dict() for i in self.ideas],
                "notes": [{**n.to_dict(), "side": n.params.get("side")} for n in self.notes]}


@dataclass
class TickerAssessment:
    symbol: str
    family: str = ""
    universe: str = ""
    in_universe: bool = True
    error: Notice | None = None          # no data / empty input: the other fields stay empty
    signal: Signal | None = None
    views: dict[str, HorizonView] = field(default_factory=dict)
    snapshot: Snapshot | None = None
    implied: list | None = None          # None: not requested; []: none available
    implied_error: str | None = None
    study_date: str = ""

    @property
    def name(self) -> str:
        return display_name(self.symbol)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "name": self.name, "family": self.family,
            "universe": self.universe, "in_universe": self.in_universe,
            "error": None if self.error is None else str(self.error),
            "error_de": None if self.error is None else self.error.text("de"),
            "signal": None if self.signal is None else self.signal.to_dict(),
            "views": {h: v.to_dict() for h, v in self.views.items()},
            "snapshot": None if self.snapshot is None else self.snapshot.to_dict(),
            "implied_moves": None if self.implied is None else [m.to_dict() for m in self.implied],
            "implied_error": self.implied_error, "study_date": self.study_date,
        }


@dataclass
class Assessment:
    requested: list[str]
    as_of_date: str
    tickers: list[TickerAssessment] = field(default_factory=list)
    regimes: dict[str, dict] = field(default_factory=dict)       # universe -> label/points
    comparison: dict[str, list[str]] = field(default_factory=dict)   # horizon -> symbols, best first
    summary_items: list[Notice] = field(default_factory=list)
    warnings: list[Notice] = field(default_factory=list)
    with_options: bool = False

    def get(self, symbol: str) -> TickerAssessment | None:
        return next((t for t in self.tickers if t.symbol == symbol), None)

    def to_dict(self) -> dict:
        return {
            "requested": self.requested, "as_of_date": self.as_of_date,
            "tickers": [t.to_dict() for t in self.tickers], "regimes": self.regimes,
            "comparison": self.comparison,
            "summary": [{"code": s.code, "horizon": s.params.get("horizon"),
                         "text": s.text("en"), "text_de": s.text("de")}
                        for s in self.summary_items],
            "warnings": [{"code": w.code, "text": str(w), "text_de": w.text("de")}
                         for w in self.warnings],
            "with_options": self.with_options,
        }


# --------------------------------------------------------------------------- classification

def classify(symbol: str) -> tuple[str, str, bool]:
    """``(universe, family, in_universe)`` for a symbol.

    A member of a scanner universe keeps that universe.  Anything else is placed by its
    shape (``*-USD`` crypto, ``.DE`` and ``^GDAXI`` DAX, other ``^``/``=`` symbols ETF-like,
    the rest US stocks) and flagged ``in_universe=False``, because its evidence label is
    then only what the family's study says, not a study of this very instrument.
    """
    for name in ("us", "dax", "sectors", "global", "crypto"):
        if symbol in UNIVERSES[name][2]:
            family = {"us": "stocks_us", "dax": "stocks_dax", "crypto": "crypto"}.get(name, "etfs")
            return name, family, True
    if symbol.endswith("-USD"):
        return "crypto", "crypto", False
    if symbol.endswith(".DE") or symbol == "^GDAXI":
        return "dax", "stocks_dax", False
    if symbol.startswith("^") or "=" in symbol:
        return "global", "etfs", False
    return "us", "stocks_us", False


def _idea_status(idea: Idea) -> str:
    if not idea.watch_only:
        return "recommended" if idea.direction == "Long" else "short_rec"
    return "watch" if idea.direction == "Long" else "weak"


def _view(horizon: str, ideas: list[Idea], notes: list[Notice],
          signal: Signal | None) -> HorizonView:
    mine = [i for i in ideas if i.horizon == horizon]
    view = HorizonView(horizon=horizon, ideas=mine,
                       notes=[n for n in notes if n.params.get("horizon") == horizon])
    statuses = {_idea_status(i) for i in mine}
    for status in ("recommended", "watch", "short_rec", "weak"):
        if status in statuses:
            view.status = status
            break
    view.conflict = view.status == "recommended" and signal is not None \
        and signal.direction == "Short"
    return view


def _entry(horizon: str, t: TickerAssessment) -> dict | None:
    view = t.views.get(horizon)
    if t.error is not None or view is None:
        return None
    lead = view.lead
    return {"symbol": t.symbol, "name": t.name, "status": view.status,
            "conflict": view.conflict, "score": None if t.signal is None else t.signal.score,
            "label": lead.evidence.label if lead else "not_studied",
            "direction": lead.direction if lead else ""}


def _compare(tickers: list[TickerAssessment]) -> tuple[dict, list[Notice]]:
    comparison: dict[str, list[str]] = {}
    items: list[Notice] = []
    for h in HORIZON_ORDER:
        entries = [e for t in tickers if (e := _entry(h, t)) is not None]
        if not entries:
            continue
        entries.sort(key=lambda e: (STATUS_ORDER[e["status"]],
                                    -(e["score"] if e["score"] is not None else -1e9),
                                    e["symbol"]))
        comparison[h] = [e["symbol"] for e in entries]
        items.append(Notice("assess_summary", horizon=h, entries=entries))
    return comparison, items


# --------------------------------------------------------------------------- the assessment

def assess(
    tickers: list[str],
    as_of_date: str | None = None,
    loader: Loader | None = None,
    now: datetime | None = None,
    with_options: bool = False,
) -> Assessment:
    """Assess the asked instruments per horizon and compare them.

    One scan per peer group involved (see ``classify``), with the asked tickers added to
    the group's members.  ``loader`` and ``now`` go to ``run_scan``/``performance_snapshot``
    so tests run offline.  A ticker without data is reported on its own entry
    (``error``), an empty entry too; Yahoo being unreachable raises ``VendorError``.
    ``with_options`` adds straddle-implied moves for US stocks (an expiry-less answer or a
    vendor failure there only leaves that part empty).
    """
    as_of = parse_date(as_of_date)
    now = now or datetime.now(UTC)
    asked: list[str] = []
    out = Assessment(requested=[], as_of_date=as_of, with_options=with_options)
    results: dict[str, TickerAssessment] = {}
    for raw in tickers:
        text = (raw or "").strip()
        if not text:
            out.tickers.append(TickerAssessment(symbol="?", error=Notice("no_price_data")))
            continue
        symbol = normalize_symbol(text).upper()
        if symbol in results:
            continue
        universe, family, known = classify(symbol)
        results[symbol] = TickerAssessment(symbol=symbol, family=family, universe=universe,
                                           in_universe=known)
        asked.append(symbol)
    out.requested = asked

    by_universe: dict[str, list[str]] = {}
    for symbol in asked:
        by_universe.setdefault(results[symbol].universe, []).append(symbol)

    warnings: list[Notice] = []
    for universe, extras in by_universe.items():
        members = list(dict.fromkeys([*UNIVERSES[universe][2], *extras]))
        scan = run_scan(universe, as_of, tickers=members, top=_ALL, loader=loader, now=now)
        out.regimes[universe] = {"label": scan.regime.label, "points": scan.regime.points,
                                 "benchmark": scan.benchmark}
        warnings.extend(w for w in scan.warnings if w.code in _SHOWN_WARNINGS)
        sig_by = {s.symbol: s for s in scan.signals}
        skipped = dict(scan.skipped)
        ideas = [] if scan.ideas is None else scan.ideas.ideas
        notes = [] if scan.ideas is None else scan.ideas.notes
        for symbol in extras:
            t = results[symbol]
            t.signal = sig_by.get(symbol)
            if t.signal is None:
                t.error = skipped.get(symbol) or Notice("no_data")
                continue
            t.study_date = "" if scan.ideas is None else scan.ideas.study_date
            mine = [i for i in ideas if i.symbol == symbol]
            t.views = {h: _view(h, mine, notes, t.signal) for h in HORIZON_ORDER}
    seen: set[str] = set()
    for w in warnings:
        if w.text("en") not in seen:
            seen.add(w.text("en"))
            out.warnings.append(w)

    valid = [s for s in asked if results[s].error is None]
    if valid:
        snaps = {s.symbol: s for s in performance_snapshot(valid, as_of, loader, now)}
        for symbol in valid:
            results[symbol].snapshot = snaps.get(symbol)
    if with_options:
        _add_options(results, valid)

    # keep the user's order; "?" placeholders for empty input come first as they were given
    out.tickers = [*out.tickers, *(results[s] for s in asked)]
    out.comparison, out.summary_items = _compare([t for t in out.tickers if t.error is None])
    return out


def _add_options(results: dict[str, TickerAssessment], symbols: list[str]) -> None:
    from tradingagents.dataflows.vendors.yahoo.options import load_implied_moves

    for symbol in symbols:
        t = results[symbol]
        if t.family != "stocks_us":
            t.implied = []          # no listed options for DAX stocks, coins and indices here
            continue
        try:
            t.implied = load_implied_moves(symbol)
        except VendorError as exc:
            t.implied, t.implied_error = [], str(exc)


# --------------------------------------------------------------------------- rendering

def _plan_line(idea: Idea, lang: str) -> str:
    """Entry, size, holding period and reference of a recommendation.  A time-exit rule has
    no stop: its size is equal weight in the measured basket and the note says why."""
    parts = [f"{_t('Einstieg', 'Entry', lang)}: {idea.entry.get(lang, idea.entry.get('en'))}"]
    if idea.stop is not None:
        parts.append(f"Stop: {fmt_price(idea.stop, lang)}")
        if idea.size_pct is not None:
            parts.append(_t(f"Größe: bis {fmt_num(idea.size_pct, 1, lang)} % des Kapitals "
                            "(Risiko je Trade 1 %)",
                            f"size: up to {fmt_num(idea.size_pct, 1, lang)} % of capital "
                            "(1 % risked per trade)", lang))
    elif idea.size_pct is not None:
        parts.append(_t(f"Größe: {fmt_num(idea.size_pct, 1, lang)} % gleichgewichtet im Korb, "
                        "kein Stop",
                        f"size: {fmt_num(idea.size_pct, 1, lang)} % equal weight in the basket, "
                        "no stop", lang))
    parts.append(idea.holding.get(lang, idea.holding.get("en", "")))
    ref = (f"{_t('Referenzkurs', 'Reference price', lang)} {fmt_price(idea.reference_price, lang)} "
           f"({fmt_date(idea.reference_date, lang)})")
    return "; ".join([*parts, ref])


def _horizon_lines(t: TickerAssessment, lang: str,
                   horizons: tuple[str, ...] = HORIZON_ORDER) -> list[str]:
    lines: list[str] = []
    quiet: list[str] = []
    for h in horizons:
        view = t.views.get(h)
        if view is None:
            continue
        lab = HORIZONS[h].label[lang]
        if view.status == "none":
            quiet.append(lab)
            continue
        for idea in sorted(view.ideas, key=lambda i: (i.watch_only, i.direction != "Long")):
            status = _idea_status(idea)
            kind = {"recommended": _t("Empfehlung Long", "Recommendation Long", lang),
                    "short_rec": _t("Empfehlung Short", "Recommendation Short", lang),
                    "watch": _t("nur Beobachtung, technisch stark", "watch only, technically strong", lang),
                    "weak": _t("nur Beobachtung, technisch schwach",
                               "watch only, technically weak", lang)}[status]
            rule = idea.rule_name.get(lang) or idea.rule_name.get("en") or idea.rule_id
            evidence = idea.evidence.text(lang)       # verbatim; only a full stop is appended
            line = (f"- **{lab}: {kind}** ({_t('Regel', 'rule', lang)} {rule}): {evidence}"
                    + ("" if evidence.endswith((".", "!", "?")) else "."))
            if status in ("recommended", "short_rec"):
                line += f"\n  {_t('Plan', 'Plan', lang)}: {_plan_line(idea, lang)}."
                if idea.size_note:
                    line += "\n  " + idea.size_note.get(lang, idea.size_note["en"])
                if idea.conflict:
                    line += "\n  " + idea.conflict.get(lang, idea.conflict["en"])
                elif view.conflict:
                    line += "\n  " + _t("Aber: aktuell technisch schwach, beides steht "
                                        "nebeneinander.",
                                        "But: currently technically weak, both stand side by "
                                        "side.", lang)
            if idea.outside:
                line += "\n  " + idea.outside.get(lang, idea.outside["en"])
            lines.append(line)
    if quiet:
        lines.append(f"- {_t('Keine Idee für', 'No idea for', lang)}: {', '.join(quiet)}.")
    return lines


def _snapshot_line(s: Snapshot | None, lang: str) -> str:
    if s is None or s.error:
        return ""
    cells = [(_t("1 Monat", "1 month", lang), s.ret_21), (_t("3 Monate", "3 months", lang), s.ret_63),
             (_t("6 Monate", "6 months", lang), s.ret_126), (_t("12 Monate", "12 months", lang),
                                                              s.ret_252)]
    got = [f"{name} {fmt_pct(v, 1, True, lang)}" for name, v in cells if v is not None]
    parts = [f"{_t('Kurs', 'Close', lang)} {fmt_price(s.last_close, lang)} "
             f"({fmt_date(s.last_date, lang)})"]
    if got:
        parts.append(" · ".join(got))
    if s.dist_high_252 is not None:
        parts.append(_t(f"{fmt_pct(s.dist_high_252, 1, True, lang)} zum 52-Wochen-Hoch",
                        f"{fmt_pct(s.dist_high_252, 1, True, lang)} from the 52-week high", lang))
    if s.atr14_pct is not None:
        parts.append(_t(f"übliche Tagesschwankung {fmt_pct(s.atr14_pct, 1, False, lang)}",
                        f"typical daily range {fmt_pct(s.atr14_pct, 1, False, lang)}", lang))
    return " | ".join(parts)


def _snapshot_short(s: Snapshot | None, lang: str) -> str:
    """Price and the 1- and 3-month return: the compact answer needs no more."""
    if s is None or s.error:
        return ""
    got = [f"{name} {fmt_pct(v, 1, True, lang)}"
           for name, v in ((_t("1 Monat", "1 month", lang), s.ret_21),
                           (_t("3 Monate", "3 months", lang), s.ret_63)) if v is not None]
    return (f"{_t('Kurs', 'Close', lang)} {fmt_price(s.last_close, lang)} "
            f"({fmt_date(s.last_date, lang)})" + (f", {', '.join(got)}" if got else ""))


def _compact_block(t: TickerAssessment, lang: str, horizons: tuple[str, ...]) -> str:
    """One short block for a phone: the price line, the technical picture in one line, one
    line per horizon.  No reasons list, no options, no notes: ``assess`` has those."""
    if t.error is not None:
        return _ticker_block(t, lang, horizons)
    snap = _snapshot_short(t.snapshot, lang)
    out = [f"**{t.name}**" + (f" – {snap}" if snap else "")]
    if not t.in_universe:
        fam = _FAMILY_LABEL.get(t.family, (t.family, t.family))[0 if lang == "de" else 1]
        out.append(_t(f"_In keiner Liste des Programms; als {fam} eingestuft._",
                      f"_In none of the program's lists; treated as {fam}._", lang))
    s = t.signal
    if s is not None:
        conv = {"High": _t("hoch", "high", lang), "Medium": _t("mittel", "medium", lang),
                "Low": _t("niedrig", "low", lang)}.get(s.conviction, s.conviction)
        out.append(_t(f"Technisches Bild (ohne gemessenen Vorteil): {watch_direction(s.direction, 'de')}, Score "
                      f"{_score(s.score, lang)}, Überzeugung {conv}.",
                      f"Technical picture (no measured edge): {watch_direction(s.direction, 'en')}, score "
                      f"{_score(s.score, lang)}, conviction {conv}.", lang))
    out.append("\n".join(_horizon_lines(t, lang, horizons)))
    return "\n".join(part for part in out if part)


def _ticker_block(t: TickerAssessment, lang: str,
                  horizons: tuple[str, ...] = HORIZON_ORDER) -> str:
    out = [f"### {t.name}"]
    if t.error is not None:
        out.append(_t(f"Keine Einschätzung möglich: {t.error.text('de')}. "
                      "Symbol prüfen (Yahoo-Schreibweise, z. B. SAP.DE, BTC-USD).",
                      f"No assessment possible: {t.error.text('en')}. Check the symbol "
                      "(Yahoo spelling, e.g. SAP.DE, BTC-USD).", lang))
        return "\n".join(out)
    fam = _FAMILY_LABEL.get(t.family, (t.family, t.family))[0 if lang == "de" else 1]
    if not t.in_universe:
        out.append(_t(f"_Steht in keiner Liste des Programms; als {fam} eingestuft. Ein Beleg "
                      "gilt hier höchstens so, wie die Studie der Gruppe es sagt, nicht für "
                      "dieses Papier selbst._",
                      f"_Not in any list of the program; treated as {fam}. Any evidence holds at "
                      "most as the group's study says, not for this very instrument._", lang))
    s = t.signal
    if s is not None:
        direction = {"Long": watch_direction("Long", lang), "Short": watch_direction("Short", lang),
                     "Neutral": _t("Neutral", "Neutral", lang)}.get(s.direction, s.direction)
        conv = {"High": _t("hoch", "high", lang), "Medium": _t("mittel", "medium", lang),
                "Low": _t("niedrig", "low", lang)}.get(s.conviction, s.conviction)
        reasons = "; ".join(s.reasons_text(lang))
        out.append(_t(f"**Technisches Bild (ohne gemessenen Vorteil):** {direction}, Score "
                      f"{_score(s.score, lang)} von ±100, Überzeugung {conv}. {reasons}.",
                      f"**Technical picture (no measured edge):** {direction}, score "
                      f"{_score(s.score, lang)} of ±100, conviction {conv}. {reasons}.", lang))
    out.append("\n".join(_horizon_lines(t, lang, horizons)))
    line = _snapshot_line(t.snapshot, lang)
    if line:
        out.append(line)
    if t.implied is not None:
        if t.implied:
            out.append(render_implied_moves(t.implied, lang).strip())
        elif t.implied_error:
            out.append(_t(f"Optionsdaten nicht verfügbar: {t.implied_error}",
                          f"Options data unavailable: {t.implied_error}", lang))
        else:
            out.append(_t("Keine börsengehandelten Optionen gefunden (kein implizierter "
                          "Ausschlag).", "No listed options found (no implied move).", lang))
    return "\n\n".join(out)


def _regime_line(a: Assessment, lang: str) -> str:
    names = {"us": ("US-Aktien", "US stocks"), "dax": ("DAX", "DAX"), "sectors": ("ETFs", "ETFs"),
             "global": ("ETFs", "ETFs"), "crypto": ("Krypto", "crypto")}
    parts = [f"{names[u][0 if lang == 'de' else 1]}: {REGIME_LABELS[r['label']][lang]} "
             f"({r['points']:+d})" for u, r in a.regimes.items()]
    return (f"{_t('Datenstand', 'Data as of', lang)} {fmt_date(a.as_of_date, lang)}"
            + (f" · {_t('Marktlage', 'Market regime', lang)} " + "; ".join(parts) if parts else ""))


def legend(lang: str = "en") -> str:
    return _t(
        "Belegt = im Test nach Kosten ein messbarer Vorteil · Schwach belegt = vorläufig, kann "
        "Zufall sein · Nur relativ schwächer / Nur Marktrendite = kein eigener Vorteil · Nicht belegt "
        "= kein messbarer Vorteil · Nicht untersucht = keine Messung. Der Score ist ein "
        "technisches Bild, kein belegter Vorteil.",
        "Validated = a measurable edge after costs in the test · Weakly supported = "
        "preliminary, may be chance · Relatively weaker only / Market return only = no edge of its "
        "own · Not proven = no measurable edge · Not studied = no measurement. The score is a "
        "technical picture, not an evidenced edge.", lang)


def disclaimer(lang: str = "en") -> str:
    return _t("Keine Anlageberatung. Nichts hier ist eine Zusage: Auswertungen der Vergangenheit "
              "sagen nicht voraus, was kommt.",
              "Not investment advice. Nothing here is a promise: results measured on the past do "
              "not predict what comes next.", lang)


def render_body(a: Assessment, lang: str = "en",
                horizons: tuple[str, ...] | list[str] | None = None) -> str:
    """The per-ticker blocks, warnings, legend and disclaimer (no title, no summary).

    ``horizons`` limits the per-horizon lines (and the notes on the short horizons) to the
    ones asked about; the default shows all four.
    """
    shown = tuple(h for h in HORIZON_ORDER if horizons is None or h in horizons)
    parts = [_regime_line(a, lang)]
    parts.extend(w.text(lang) for w in a.warnings)
    parts.extend(_ticker_block(t, lang, shown) for t in a.tickers)
    notes: list[str] = []
    for t in a.tickers:
        for h in (h for h in ("day", "swing") if h in shown):
            view = t.views.get(h)
            if view is not None and view.status == "none":
                notes.extend(n.text(lang) for n in view.notes if n.params.get("side") is None)
    notes = list(dict.fromkeys(notes))
    if notes:
        parts.append(f"**{_t('Zu den kurzen Horizonten', 'On the short horizons', lang)}:** "
                     + " ".join(notes))
    parts.append(f"_{legend(lang)}_")
    parts.append(f"**{disclaimer(lang)}**")
    return "\n\n".join(parts)


def legend_short(lang: str = "en") -> str:
    return _t(
        "Belegt = messbarer Vorteil im Test nach Kosten · Schwach belegt = vorläufig, kann Zufall "
        "sein · Nicht belegt = kein messbarer Vorteil · Nicht untersucht = keine Messung. Der "
        "Score ist ein technisches Bild, kein Beleg.",
        "Validated = measurable edge in the test after costs · Weakly supported = preliminary, "
        "may be chance · Not proven = no measurable edge · Not studied = no measurement. The "
        "score is a technical picture, not evidence.", lang)


def render_compact(a: Assessment, lang: str = "en",
                   horizons: tuple[str, ...] | list[str] | None = None) -> str:
    """The phone version of ``render_body``: one compact block per instrument, legend and
    disclaimer in one line each."""
    shown = tuple(h for h in HORIZON_ORDER if horizons is None or h in horizons)
    parts = [_regime_line(a, lang)]
    parts.extend(w.text(lang) for w in a.warnings)
    parts.extend(_compact_block(t, lang, shown) for t in a.tickers)
    parts.append(f"_{legend_short(lang)}_")
    parts.append(f"**{disclaimer(lang)}**")
    return "\n\n".join(parts)


def render_assessment(assessment: Assessment, lang: str = "en") -> str:
    """Phone-friendly Markdown: summary first, then one compact block per ticker."""
    lang = lang if lang in ("en", "de") else "en"
    names = ", ".join(t.name for t in assessment.tickers)
    head = f"## {_t('Einschätzung', 'Assessment', lang)}: {names}"
    summary = [f"- {line}" for line in collapse_summary(assessment.summary_items, lang)]
    parts = [head]
    if summary:
        parts.append(f"**{_t('Das Wichtigste', 'In short', lang)}**\n" + "\n".join(summary))
    parts.append(render_body(assessment, lang))
    return "\n\n".join(parts) + "\n"
