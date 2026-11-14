"""The phone digest: what a scheduled scan pushes as a notification or message.

A headline that stands alone as a push notification and a plain-text body (no Markdown,
no tables) that reads on a phone. Honesty rules, same as the report: only ideas whose rule
has measured evidence are listed as recommendations; where there is none the digest says
so in one line and why; the composite's strong/weak names appear only as a short "watch,
not proven" line, which is the first thing dropped when the body gets too long.

The text is deterministic for a given input and ``now``. German is the default because the
digest is read by a German-speaking user; English exists for completeness.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from tradingagents.scanner.fmt import fmt_date, fmt_num, fmt_price
from tradingagents.scanner.horizons import holding_text
from tradingagents.scanner.notices import label_name
from tradingagents.scanner.regime import LABELS as REGIME_LABELS
from tradingagents.scanner.report import (
    SIDES,
    HorizonView,
    composite_conflicts,
    horizon_label,
    horizon_views,
    label_text,
    market_name,
    short_reason,
)
from tradingagents.scanner.scan import ScanResult

HEADLINE_MAX = 180
BODY_MAX = 2400

_WEEKDAY = {
    "de": ("Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"),
    "en": ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"),
}

_X = {
    "title_prices": ("prices up to", "Kurse bis"),
    "prices": ("prices", "Kurse"),
    "entry": ("Entry", "Einstieg"),
    "ref": ("ref.", "Ref."),
    "stop": ("Stop", "Stop"),
    "max": ("max.", "max."),
    "more": ("(+{n} more in the report)", "(+{n} weitere im Bericht)"),
    "no_rec": ("no recommendation", "keine Empfehlung"),
    "none_market": ("No evidenced recommendation.", "Keine belegte Empfehlung."),
    "tech_picture": ("Technical picture (not evidenced)", "Technisches Bild (nicht belegt)"),
    "equal": ("equal weight", "gleichgewichtet"),
    "no_stop": ("no stop", "kein Stop"),
    "ideas_missing": ("ideas unavailable, no recommendation",
                      "Ideen nicht verfügbar, keine Empfehlung"),
    "watch": ("Watch (not proven)", "Beobachten (nicht belegt)"),
    "strong": ("strong", "stark"), "weak": ("weak", "schwach"),
    "avoid": (f"Avoid / underweight ({label_name('relative', 'en')})",
              f"Meiden / untergewichten ({label_name('relative', 'de')})"),
    "weak_now": ("currently technically weak", "aktuell technisch schwach"),
    "strong_now": ("currently technically strong", "aktuell technisch stark"),
    "foot_labels": (
        "Validated = passed an out-of-sample test; technical picture = prices only, no proven "
        "edge.",
        "Belegt = in einem Test außerhalb der Stichprobe bestanden; Technisches Bild = nur "
        "Kursbild, kein belegter Vorteil."),
    "foot_sizes": ("Sizes are per trade and do not add up.",
                   "Größen gelten je Trade und addieren sich nicht."),
    "weekend": ("weekend - exchanges closed", "Wochenende – Börsen geschlossen"),
    "tech_strong_np": ("technically strong (not proven)", "technisch stark (nicht belegt)"),
    "foot_advice": ("Not investment advice.", "Keine Anlageberatung."),
    "no_scan": ("No scan result", "Kein Scan-Ergebnis"),
    "none_proven": ("No evidenced recommendation", "Keine belegte Empfehlung"),
    "proven": ("Evidenced", "Belegt"),
    "watch_only": ("Watch only (not proven)", "Nur Beobachtung (nicht belegt)"),
}

# short market names for the headline (a push notification has no room for "US-Aktien")
_SHORT_MARKET = {"us": "US", "dax": "DAX", "global": "Global", "crypto": ("Crypto", "Krypto"),
                 "sectors": ("Sectors", "Sektoren")}


def _x(key: str, lang: str):
    return _X[key][1 if lang == "de" else 0]


@dataclass(frozen=True)
class Digest:
    """``headline``: one line, at most 180 characters. ``body``: plain text, at most 2,400."""

    headline: str
    body: str

    @property
    def text(self) -> str:
        """Headline, a blank line, the body: what a script can send as one message."""
        return f"{self.headline}\n\n{self.body}"


# --- the pieces ---------------------------------------------------------------------------------


def _side_word(direction: str) -> str:
    return direction.upper()


def _idea_line(idea, lang: str, conflict: str | None) -> str:
    entry = idea.entry.get(lang) or idea.entry.get("en", "")
    parts = [
        f"{_side_word(idea.direction)} {idea.symbol}",
        f"{_x('entry', lang)} {_short_entry(entry, lang)} "
        f"({_x('ref', lang)} {fmt_price(idea.reference_price, lang)})",
    ]
    if idea.stop is not None:
        parts.append(f"{_x('stop', lang)} {fmt_price(idea.stop, lang)}")
        if idea.size_pct is not None:
            parts.append(f"{_x('max', lang)} {fmt_num(idea.size_pct, 1, lang)} %")
    elif idea.size_pct is not None:     # a time-exit rule: no stop, equal weight in its basket
        parts.append(f"{fmt_num(idea.size_pct, 1, lang)} % {_x('equal', lang)}, "
                     f"{_x('no_stop', lang)}")
    parts.append(label_text(idea.evidence.label, lang))
    if idea.evidence.caveat:
        parts.append(idea.evidence.caveat.get(f"short_{lang}") or idea.evidence.caveat[lang])
    if conflict:
        parts.append(_x("weak_now" if conflict == "weak" else "strong_now", lang))
    return " · ".join(parts)


def _short_entry(entry: str, lang: str) -> str:
    """'zur nächsten Eröffnung' -> 'nächste Eröffnung' (the plan's phone wording)."""
    if lang == "de" and entry.startswith("zur "):
        return "nächste Eröffnung" if entry == "zur nächsten Eröffnung" else entry[4:]
    if lang != "de" and entry.startswith("at the "):
        return entry[7:]
    return entry


def _why_text(notes, lang: str) -> str:
    clause = short_reason(notes[0], lang) if notes else ""
    return f"{_x('no_rec', lang)} – {clause}" if clause else _x("no_rec", lang)


def _nothing_entries(view: HorizonView, lang: str) -> list[tuple[str, str | None, str]]:
    """``(horizon, side, text)`` for each side without an evidenced idea; both sides with the
    same reason collapse into the horizon alone (``side`` None)."""
    name = horizon_label(view.key, lang)
    missing = [s for s in SIDES if not view.side(s)]
    if not missing:
        return []
    texts = {s: _why_text(view.why[s], lang) for s in missing}
    if len(missing) == 2 and texts["Long"] == texts["Short"]:
        return [(name, None, texts["Long"])]
    return [(name, s, texts[s]) for s in missing]


def _avoid_symbols(result: ScanResult, horizons) -> list[str]:
    if result.ideas is None:
        return []
    keys = {v.key for v in horizon_views(result, horizons)}
    return list(dict.fromkeys(i.symbol for i in result.ideas.ideas
                              if i.stance == "avoid" and i.horizon in keys))


def _watch_line(result: ScanResult, lang: str, max_per_side: int, horizons) -> list[str]:
    """The composite's strong and weak names, and the names to avoid (relative evidence)."""
    lines = []
    parts = []
    strong = [s.symbol for s in result.longs[:max_per_side]]
    weak = [s.symbol for s in result.shorts[:max_per_side]]
    if strong:
        parts.append(f"{_x('strong', lang)} {', '.join(strong)}")
    if weak:
        parts.append(f"{_x('weak', lang)} {', '.join(weak)}")
    if parts:
        lines.append(" · ".join(parts))
    avoid = _avoid_symbols(result, horizons)[:max_per_side]
    if avoid:
        lines.append(f"{_x('avoid', lang)}: {', '.join(avoid)}")
    return lines


def _market_lines(result: ScanResult, lang: str, max_per_side: int, watch: bool,
                  newest: str, horizons) -> list[str]:
    head = f"{market_name(result, lang)} · {REGIME_LABELS[result.regime.label][lang]}"
    if result.as_of_date and result.as_of_date != newest:
        head += f" ({_x('prices', lang)} {fmt_date(result.as_of_date, lang)})"
    lines = [head]
    if result.ideas is None:
        lines.append(_x("ideas_missing", lang))
        return lines
    conflicts = composite_conflicts(result)
    merged: dict[str, list[tuple[str, str | None]]] = {}
    views = horizon_views(result, horizons)
    evidenced = any(v.recs for v in views)
    for view in views:
        if view.recs:
            lines.append(f"{holding_text(view.key, result.ideas.family, lang)}:")
            for side in SIDES:
                ideas = view.side(side)
                lines += [_idea_line(i, lang, conflicts.get(i.symbol))
                          for i in ideas[:max_per_side]]
                if len(ideas) > max_per_side:
                    lines.append(_x("more", lang).format(n=len(ideas) - max_per_side))
        for name, side, text in _nothing_entries(view, lang):
            merged.setdefault(text, []).append((name, side))
    picture = _watch_line(result, lang, max_per_side, horizons) if watch else []
    if not evidenced:
        # nothing is evidenced here: say so in one plain sentence, then label the picture
        sentence = _x("none_market", lang)
        if picture:
            sentence += f" {_x('tech_picture', lang)}: {' · '.join(picture)}"
        lines.append(sentence)
    for text, entries in merged.items():
        # a line that mixes "both sides" and "one side" must not leave the reader guessing
        mixed = any(side for _, side in entries)
        labels = [f"{name} {side or 'Long+Short'}" if mixed else name for name, side in entries]
        lines.append(f"{', '.join(labels)}: {text}")
    if evidenced and picture:
        lines.append(f"{_x('watch', lang)}: {' · '.join(picture)}")
    return lines


DEFAULT_TZ = "Europe/Berlin"
_ZONE_DE = {"CET": "MEZ", "CEST": "MESZ"}


def _local(now: datetime, tz: str) -> datetime:
    """``now`` in the zone ``tz`` (a naive ``now`` is taken as UTC)."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(ZoneInfo(tz))


def _is_weekend(local: datetime) -> bool:
    """Saturday or Sunday on the already localised clock."""
    return local.weekday() >= 5


def _title(results: list[ScanResult], lang: str, now: datetime, newest: str,
           tz: str = DEFAULT_TZ) -> str:
    """``Fr 13.11.2026 17:52 MEZ · Kurse bis 12.11.2026`` (the clock in zone ``tz``); when the markets' newest bars differ
    (a closed exchange, a holiday) the range ``Kurse bis 11.11.2026–12.11.2026``."""
    local = _local(now, tz)
    abbr = local.strftime("%Z")
    if lang == "de":
        abbr = _ZONE_DE.get(abbr, abbr)
    weekday = _WEEKDAY[lang][local.weekday()]
    stamp = f"{weekday} {fmt_date(local.date(), lang)} {local.strftime('%H:%M')} {abbr}"
    if _is_weekend(local) and any(r.universe != "crypto" for r in results):
        stamp += f" · {_x('weekend', lang)}"
    dates = sorted({r.as_of_date for r in results if r.as_of_date})
    if len(dates) > 1:
        stamp += (f" · {_x('title_prices', lang)} {fmt_date(dates[0], lang)}–"
                  f"{fmt_date(dates[-1], lang)}")
    elif newest:
        stamp += f" · {_x('title_prices', lang)} {fmt_date(newest, lang)}"
    return stamp


def _any_size(results: list[ScanResult], max_per_side: int, horizons) -> bool:
    """True when at least one displayed idea carries a size (only then the footer explains it)."""
    for result in results:
        if result.ideas is None:
            continue
        for view in horizon_views(result, horizons):
            for side in SIDES:
                if any(i.size_pct is not None for i in view.side(side)[:max_per_side]):
                    return True
    return False


def _body(results: list[ScanResult], lang: str, max_per_side: int, watch: bool, now: datetime,
          horizons, tz: str = DEFAULT_TZ) -> str:
    newest = max((r.as_of_date for r in results if r.as_of_date), default="")
    blocks = [_title(results, lang, now, newest, tz)]
    for result in results:
        blocks.append("\n".join(_market_lines(result, lang, max_per_side, watch, newest,
                                              horizons)))
    foot = [_x("foot_labels", lang)]
    if _any_size(results, max_per_side, horizons):
        foot.append(_x("foot_sizes", lang))
    foot.append(_x("foot_advice", lang))
    blocks.append(" ".join(foot))
    return "\n\n".join(blocks)


def _evidenced_groups(results: list[ScanResult], lang: str, horizons) -> list[tuple[str, list[str]]]:
    """``("US Langfristig Long", [tickers])`` for every horizon and side with evidenced ideas."""
    groups = []
    for result in results:
        for view in horizon_views(result, horizons):
            for side in SIDES:
                ideas = view.side(side)
                if ideas:
                    groups.append((f"{market_name(result, lang)} {horizon_label(view.key, lang)} "
                                   f"{side}", [i.symbol for i in ideas]))
    return groups


def _short_market(result: ScanResult, lang: str) -> str:
    if result.custom:
        return market_name(result, lang)
    name = _SHORT_MARKET.get(result.universe)
    if name is None:
        return result.universe.upper()
    return name if isinstance(name, str) else name[1 if lang == "de" else 0]


def _bare(symbol: str) -> str:
    """A ticker without its Yahoo suffix (.DE, -USD) for the headline."""
    for suffix in (".DE", "-USD"):
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def _headline(results: list[ScanResult], lang: str, horizons) -> str:
    if not results:
        return _x("no_scan", lang)
    regimes = ", ".join(f"{market_name(r, lang)} {REGIME_LABELS[r.regime.label][lang]}"
                        for r in results)
    groups = _evidenced_groups(results, lang, horizons)

    def names(symbols: list[str], limit: int) -> str:
        text = ", ".join(symbols[:limit])
        return text + (f" +{len(symbols) - limit}" if len(symbols) > limit else "")

    candidates = []
    if groups:
        for limit in (3, 2, 1):
            shown = "; ".join(f"{label}: {names(syms, limit)}" for label, syms in groups)
            candidates.append(f"{_x('proven', lang)}: {shown} · {regimes}")
        candidates.append(f"{_x('proven', lang)}: {groups[0][0]}: {names(groups[0][1], 2)}"
                          + (f" (+{len(groups) - 1})" if len(groups) > 1 else ""))
    else:
        for limit in (2, 1):
            watch = " · ".join(
                f"{_short_market(r, lang)} {', '.join(_bare(s.symbol) for s in r.longs[:limit])}"
                for r in results if r.longs)
            tail = f" · {_x('tech_strong_np', lang)}: {watch}" if watch else ""
            candidates.append(f"{_x('none_proven', lang)}{tail} · {regimes}")
            candidates.append(f"{_x('none_proven', lang)}{tail}")
        candidates.append(_x("none_proven", lang))
    for text in candidates:
        if len(text) <= HEADLINE_MAX:
            return text
    return candidates[-1][: HEADLINE_MAX - 1].rstrip() + "…"


def build_digest(
    results: list[ScanResult],
    lang: str = "de",
    max_per_side: int = 3,
    now: datetime | None = None,
    horizons: Iterable[str] | None = None,
    tz: str = DEFAULT_TZ,
) -> Digest:
    """Headline and body for ``results`` (one per market).

    ``now`` is the clock shown in the title (aware, or naive meaning UTC; default: the current
    time), shown in the IANA zone ``tz`` (default Europe/Berlin, which also decides the weekend
    flag), so the same input, ``now`` and ``tz`` always give the same text. The body stays within ``BODY_MAX``
    characters: first the watch lines go, then the number of ideas per side shrinks to one,
    and only as a last resort the text is cut. ``horizons`` filters like ``scan --horizon``.
    """
    lang = "de" if lang == "de" else "en"
    now = now or datetime.now(UTC)
    ZoneInfo(tz)                      # an unknown zone fails here, not halfway through
    horizons = tuple(horizons) if horizons else None
    body = ""
    attempts = [(max_per_side, True), (max_per_side, False)]
    attempts += [(n, False) for n in range(max_per_side - 1, 0, -1)]
    for count, watch in attempts:
        body = _body(results, lang, max(1, count), watch, now, horizons, tz)
        if len(body) <= BODY_MAX:
            break
    else:
        cut = body[: BODY_MAX - 1]
        body = cut[: cut.rfind("\n")] if "\n" in cut else cut
        body = body.rstrip() + "\n…"
    body = "\n".join(line.rstrip() for line in body.split("\n"))
    return Digest(_headline(results, lang, horizons), body)
