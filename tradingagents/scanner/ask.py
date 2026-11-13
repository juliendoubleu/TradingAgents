"""From a free-text question to an answer, deterministically.

``parse_question`` finds the instruments, horizons and intent in a German or English
sentence; ``answer_question`` hands the instruments to ``assess`` and words the result.
There is no language model in this path on purpose: every sentence either comes from the
data (``Evidence.text``, the summary statements, the plan) or is a fixed template, so
nothing can be invented.

What the program cannot do is say what an open event ("GTA VI erscheint, wirkt das auf den
Markt?") means for the market: that needs reasoning about which companies are affected
and which past events are comparable.  It then says so, and points to the Claude chat
session on this repository (which combines its own reasoning with the measured facts of
``event-study``, ``implied-move`` and ``assess``) and to those commands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from tradingagents.scanner.aliases import ALIASES, KNOWN_SYMBOLS, display_name
from tradingagents.scanner.assess import (
    HORIZON_ORDER,
    Assessment,
    Loader,
    assess,
    label_text,
    render_body,
)
from tradingagents.scanner.horizons import HORIZONS

# --------------------------------------------------------------------------- parsing

# Upper-case words that look like tickers but are not (also words of event questions).
_NOT_TICKERS = frozenset(["ICH", "DAX", "USA", "ETF", "ETFS", "KI", "AI", "EU", "USD", "EUR", "GBP", "CEO", "CFO", "IPO", "KGV", "EPS", "GDP", "BIP", "CPI", "EZB", "FED", "SEC", "FDA", "NATO", "OPEC", "NFL", "NBA", "UEFA", "FIFA", "WM", "EM", "GTA", "RDR", "UND", "ODER", "VS", "AM", "IM", "ZU", "AN", "AB", "DIE", "DER", "DAS", "EIN", "OK", "USW", "ZB", "ZUM", "ZUR", "BIS", "MIT", "NUR", "AUF", "AUS", "FUR", "WIE", "WAS", "WER", "WIR", "SIE", "ER", "ES", "MAN", "GMBH", "AG", "SE", "NEWS", "BUY", "SELL", "HOLD", "LONG", "SHORT", "CAD", "CHF", "JPY", "YOY", "QOQ", "ATH", "ATR", "RSI", "MACD"])

_ALIAS_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(a) for a in sorted(ALIASES, key=lambda x: (-len(x), x)))
    + r")(?:s|'s)?(?!\w)", re.IGNORECASE)
# distinctive shapes that identify a symbol even in lower case: SAP.DE, btc-usd, ^GDAXI, BRK-B
_SHAPED_RE = re.compile(r"(?<![\w.^-])(\^?[A-Za-z][A-Za-z0-9]{0,5}[.-][A-Za-z]{1,3}|\^[A-Za-z0-9]{2,8})"
                        r"(?![\w])")
_UPPER_RE = re.compile(r"(?<![\w.^-])([A-Z][A-Z0-9]{1,5})(?!\w)(?![.-][A-Za-z])")

_NUMBER_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*[- ]?\s*(tag\w*|day\w*|woche\w*|week\w*|monat\w*|month\w*|jahr\w*|year\w*)",
    re.IGNORECASE)
_HORIZON_WORDS = {
    "day": re.compile(r"\b(heute|daytrad\w*|day[- ]?trad\w*|intraday|tageshandel|today)\b", re.I),
    "swing": re.compile(r"\b(tage\w*|wochen?|kurzfristig\w*|swing\w*|short[- ]term|week\w*|days?)\b",
                        re.I),
    "medium": re.compile(r"\b(monat\w*|mittelfristig\w*|medium[- ]term|month\w*|quartal\w*)\b", re.I),
    "long": re.compile(r"\b(jahr\w*|langfristig\w*|long[- ]term|year\w*|halbjahr\w*)\b", re.I),
}
_INTENTS = [   # first match wins: a short question is not a buy question
    ("short", re.compile(r"\b(short\w*|leerverk\w*|fallen\w*|sinken\w*|abstürz\w*)\b", re.I)),
    ("sell", re.compile(r"\b(verkauf\w*|verkaufen|abstoßen|sell\w*)\b", re.I)),
    ("hold", re.compile(r"\b(halten|behalten|hold\w*|aussitzen)\b", re.I)),
    ("buy", re.compile(r"\b(kauf\w*|einsteigen|einstieg|long|buy\w*|investier\w*|anlegen)\b", re.I)),
]
_EVENT_RE = re.compile(
    r"rauskomm\w*|erscheint|erscheinen|veröffentlich\w*|release\w*|launch\w*|\bevent\b|wahl\w*|"
    r"election|zinsentscheid\w*|\bezb\b|\bfed\b|earnings|quartalszahl\w*|zu erwarten|erwarten|"
    r"auswirkung\w*|\bimpact\b|was passiert|what happens|krankmeld\w*|nächste woche|next week|"
    r"ankündig\w*|olympia|weltmeisterschaft|streik\w*|zölle?|tariff\w*|\bgta\b|\bgta ?\d|\bgta vi",
    re.IGNORECASE)
_DE_WORDS = frozenset(["ich", "ist", "und", "oder", "soll", "sollte", "auf", "nicht", "ein", "eine", "der", "die", "das", "wie", "was", "lohnt", "sich", "kaufen", "verkaufen", "mit", "für", "bei", "zu", "mein", "meine", "welche", "wann", "wird", "darf", "mir", "wäre", "besser", "lieber", "monate", "monaten", "jahr", "jahre", "woche", "wochen", "tage", "kurzfristig", "langfristig", "mittelfristig"])
_EN_WORDS = frozenset(["the", "should", "buy", "sell", "or", "what", "is", "will", "my", "on", "for", "of", "with", "which", "when", "would", "better", "worth", "month", "months", "year", "years", "week", "weeks", "days", "short", "long", "term"])


@dataclass
class ParsedQuestion:
    """What the sentence names: instruments (by name or symbol), horizons, intent, language."""

    text: str
    lang: str
    tickers: list[str] = field(default_factory=list)       # Yahoo symbols, in order of mention
    names: list[tuple[str, str]] = field(default_factory=list)   # (words used, symbol)
    unverified: list[str] = field(default_factory=list)    # symbol-shaped, in no list of ours
    horizons: list[str] = field(default_factory=list)
    intent: str = "none"                                   # buy | short | sell | hold | none
    compare: bool = False
    event_like: bool = False

    def to_dict(self) -> dict:
        return {"text": self.text, "lang": self.lang, "tickers": self.tickers,
                "names": [list(n) for n in self.names], "unverified": self.unverified,
                "horizons": self.horizons, "intent": self.intent, "compare": self.compare,
                "event_like": self.event_like}


def _horizon_for(n: float, unit: str) -> str:
    u = unit.lower()
    if u.startswith(("tag", "day")):
        return "swing" if n <= 14 else "medium" if n <= 120 else "long"
    if u.startswith(("woche", "week")):
        return "swing" if n <= 2 else "medium" if n <= 17 else "long"
    if u.startswith(("monat", "month")):
        return "medium" if n <= 4 else "long"
    return "long"


def _horizons(text: str) -> list[str]:
    found: set[str] = set()

    def numbered(m: re.Match) -> str:
        found.add(_horizon_for(float(m.group(1).replace(",", ".")), m.group(2)))
        return " "

    rest = _NUMBER_RE.sub(numbered, text)       # "12 Monate" must not also count as "Monate"
    found.update(h for h, rx in _HORIZON_WORDS.items() if rx.search(rest))
    return [h for h in HORIZON_ORDER if h in found]


def _language(text: str) -> str:
    words = re.findall(r"[a-zäöüß]+", text.lower())
    de = sum(w in _DE_WORDS for w in words) + (2 if re.search(r"[äöüß]", text.lower()) else 0)
    en = sum(w in _EN_WORDS for w in words)
    return "de" if de > en or (de and de == en) else "en"


def parse_question(text: str) -> ParsedQuestion:
    """Instruments, horizons, intent and language of a question.

    Instruments come from the alias table (word boundaries, longest name first) and from
    explicit symbols: a symbol of a scanner universe or the alias table written in upper
    case, a distinctively shaped one (``SAP.DE``, ``BTC-USD``, ``^GSPC``), or any other
    3-5 letter upper-case word that is not a known non-ticker ("ICH", "USA", "ETF", "KI"):
    the last kind is listed as ``unverified`` and left to Yahoo to confirm.
    """
    original = (text or "").strip()
    all_caps = original == original.upper()
    hits: list[tuple[int, str]] = []
    names: list[tuple[str, str]] = []
    unverified: list[str] = []
    taken: list[tuple[int, int]] = []

    def free(span: tuple[int, int]) -> bool:
        return not any(a < span[1] and span[0] < b for a, b in taken)

    for m in _SHAPED_RE.finditer(original):
        token = m.group(1)
        symbol = token.upper()
        if not free(m.span()) or (symbol not in KNOWN_SYMBOLS and not token.isupper()):
            continue
        # shaped words such as "e.on" or "u.s" are not symbols unless known or upper case
        hits.append((m.start(), symbol))
        taken.append(m.span())
        if symbol not in KNOWN_SYMBOLS:
            unverified.append(symbol)
    for m in _ALIAS_RE.finditer(original):
        if not free(m.span()):
            continue
        symbol = ALIASES[m.group(1).lower()]
        hits.append((m.start(), symbol))
        names.append((m.group(0), symbol))
        taken.append(m.span())

    for m in _UPPER_RE.finditer(original):
        token = m.group(1)
        if not free(m.span()) or token in _NOT_TICKERS or token.lower() in ALIASES:
            continue
        if token in KNOWN_SYMBOLS and (len(token) >= 3 or not all_caps):
            hits.append((m.start(), token))
        elif 3 <= len(token) <= 5 and token.isalpha() and not all_caps:
            hits.append((m.start(), token))
            unverified.append(token)
    hits.sort()
    tickers = list(dict.fromkeys(s for _, s in hits))
    intent = next((name for name, rx in _INTENTS if rx.search(original)), "none")
    return ParsedQuestion(
        text=original, lang=_language(original), tickers=tickers, names=names,
        unverified=[u for u in dict.fromkeys(unverified) if u in tickers],
        horizons=_horizons(original), intent=intent, compare=len(tickers) >= 2,
        event_like=bool(_EVENT_RE.search(original)))


# --------------------------------------------------------------------------- answering

@dataclass
class Answer:
    text: str
    parsed: ParsedQuestion
    assessment: Assessment | None = None

    def to_dict(self) -> dict:
        return {"text": self.text, "parsed": self.parsed.to_dict(),
                "assessment": None if self.assessment is None else self.assessment.to_dict()}


def _t(de: str, en: str, lang: str) -> str:
    return de if lang == "de" else en


_INTENT_WORD = {
    "buy": ("kaufen / einsteigen", "buy / enter"), "short": ("Short gehen", "go short"),
    "sell": ("verkaufen", "sell"), "hold": ("halten", "hold"),
}
CMD = "python -m cli.main"


def _understood(p: ParsedQuestion, lang: str) -> str:
    parts = []
    if p.tickers:
        parts.append(_t("Instrumente: ", "Instruments: ", lang)
                     + ", ".join(display_name(s) for s in p.tickers))
    else:
        parts.append(_t("kein Wertpapier erkannt", "no instrument recognised", lang))
    if p.horizons:
        parts.append(_t("Horizont: ", "Horizon: ", lang)
                     + ", ".join(HORIZONS[h].label[lang] for h in p.horizons))
    if p.intent in _INTENT_WORD:
        parts.append(_t("Absicht: ", "Intent: ", lang) + _INTENT_WORD[p.intent][lang != "de"])
    if p.compare:
        parts.append(_t("Vergleich", "comparison", lang))
    return _t("Verstanden: ", "Understood: ", lang) + " · ".join(parts) + "."


def _event_note(lang: str) -> str:
    return _t(
        "**Zur Ereignisfrage:** Was ein Ereignis (Spiel-Release, Wahl, Zinsentscheid ...) für "
        "den Markt bedeutet, beantwortet dieses Programm nicht selbst: dazu gehört Nachdenken "
        "darüber, welche Unternehmen betroffen sind und welche früheren Ereignisse vergleichbar "
        "sind. Das macht eine Claude-Chat-Sitzung zu diesem Repository: sie verbindet eigene "
        "Überlegungen mit den gemessenen Fakten dieses Programms. Messbare Bausteine zum "
        f"selbst Ausführen: `{CMD} event-study SYMBOL --dates \"JJJJ-MM-TT:Name\"` "
        f"(Kursverhalten rund um frühere Termine) und `{CMD} implied-move SYMBOL` "
        "(was der Optionsmarkt für die nächsten Wochen einpreist).",
        "**On the event question:** what an event (a game release, an election, a rate "
        "decision ...) means for the market is not answered by this program itself: it takes "
        "reasoning about which companies are affected and which past events are comparable. "
        "A Claude chat session on this repository does that: it combines its own reasoning "
        "with this program's measured facts. Measurable building blocks you can run yourself: "
        f"`{CMD} event-study SYMBOL --dates \"YYYY-MM-DD:name\"` (price behaviour around past "
        f"dates) and `{CMD} implied-move SYMBOL` (what the options market prices in for the "
        "coming weeks).", lang)


def _fallback(p: ParsedQuestion, lang: str, failed: list[str]) -> str:
    lines = ["## " + _t("Frage", "Question", lang), _understood(p, lang)]
    if failed:
        lines.append(_t(f"Nicht als Wertpapier erkannt (keine Kursdaten): {', '.join(failed)}.",
                        f"Not recognised as an instrument (no price data): {', '.join(failed)}.",
                        lang))
    lines.append(_t(
        "**So kann ich das nicht beantworten.** Ich erkenne darin kein Wertpapier, das ich "
        "einschätzen könnte, und erfinde deshalb keine Einschätzung.",
        "**I cannot answer this as asked.** I recognise no instrument in it that I could "
        "assess, so I will not invent an assessment.", lang))
    lines.append(_event_note(lang))
    lines.append(_t(
        "**Bausteine:**\n"
        f"- `{CMD} assess NVDA SAP.DE --lang de`: Einschätzung und Vergleich je Anlagehorizont\n"
        f"- `{CMD} event-study TTWO --dates \"2018-10-26:Red Dead Redemption 2\" --lang de`\n"
        f"- `{CMD} implied-move TTWO --lang de`\n"
        f"- `{CMD} scan --universe us --lang de`: die Lage im ganzen Markt\n"
        "Nenne Instrumente beim Namen oder Symbol (z. B. „Nvidia\", „SAP\", „Bitcoin\", "
        "„Gold\", „TTWO\"), dann antworte ich mit den Daten.",
        "**Building blocks:**\n"
        f"- `{CMD} assess NVDA SAP.DE`: assessment and comparison per horizon\n"
        f"- `{CMD} event-study TTWO --dates \"2018-10-26:Red Dead Redemption 2\"`\n"
        f"- `{CMD} implied-move TTWO`\n"
        f"- `{CMD} scan --universe us`: the picture of the whole market\n"
        "Name instruments by name or symbol (e.g. \"Nvidia\", \"SAP\", \"Bitcoin\", \"gold\", "
        "\"TTWO\") and I answer from the data.", lang))
    lines.append(f"**{_t('Keine Anlageberatung.', 'Not investment advice.', lang)}**")
    return "\n\n".join(lines) + "\n"


def _short_lines(a: Assessment, horizons: list[str], lang: str) -> list[str]:
    """What the data say about the Short side, per ticker and horizon, verbatim."""
    out: list[str] = []
    for t in a.tickers:
        if t.error is not None:
            continue
        for h in horizons:
            view = t.views.get(h)
            if view is None:
                continue
            lab = HORIZONS[h].label[lang]
            shorts = [i for i in view.ideas if i.direction == "Short"]
            if shorts:
                idea = shorts[0]
                kind = _t("Short-Empfehlung", "Short recommendation", lang) \
                    if not idea.watch_only else _t("nur Short-Beobachtung", "Short watch only", lang)
                out.append(f"- {lab}, {t.name}: {kind} ({label_text(idea.evidence.label, lang)}): "
                           f"{idea.evidence.text(lang)}")
                continue
            notes = [n.text(lang) for n in view.notes if n.params.get("side") == "short"]
            out.append(f"- {lab}, {t.name}: " + (" ".join(notes[:1]) if notes else
                       _t("kein Short-Signal.", "no Short signal.", lang)))
    return out


def answer_question(
    question: str,
    lang: str | None = None,
    as_of_date: str | None = None,
    loader: Loader | None = None,
    now: datetime | None = None,
) -> Answer:
    """Answer a question from ``assess``, or say plainly that it cannot be answered.

    Instruments found: a lead sentence per asked horizon (medium and long when none was
    named), the comparison, the evidence, plan and snapshot blocks, and the caveat; a
    Short or sell question also gets what the data say about the Short side.  Instruments
    found in a question that reads like an event question also get the event note.  No
    instrument: the honest fallback.  ``lang=None`` follows the question's language.
    Raises ``ValueError`` for an empty question and ``VendorError`` if Yahoo is down.
    """
    if not (question or "").strip():
        raise ValueError("empty question")
    parsed = parse_question(question)
    lang = lang if lang in ("en", "de") else parsed.lang
    if not parsed.tickers:
        return Answer(_fallback(parsed, lang, []), parsed)
    a = assess(parsed.tickers, as_of_date=as_of_date, loader=loader, now=now)
    failed = [t.symbol for t in a.tickers if t.error is not None]
    if len(failed) == len(a.tickers):
        return Answer(_fallback(parsed, lang, failed), parsed, a)

    horizons = parsed.horizons or ["medium", "long"]
    leads = [s for s in a.summary_items if s.params["horizon"] in horizons]
    parts = ["## " + _t("Antwort", "Answer", lang), _understood(parsed, lang)]
    if failed:
        parts.append(_t(f"Keine Kursdaten gefunden für: {', '.join(failed)}.",
                        f"No price data found for: {', '.join(failed)}.", lang))
    head = {"sell": _t("Ob du verkaufst, hängt von deiner Lage ab (Einstand, Steuern, Ziel). "
                       "Das sagen die Daten:", "Whether to sell depends on your situation "
                       "(cost basis, taxes, goal). This is what the data say:", lang),
            "hold": _t("Ob du hältst, hängt von deiner Lage ab. Das sagen die Daten:",
                       "Whether to hold depends on your situation. This is what the data say:",
                       lang)}.get(parsed.intent, "")
    parts.append(f"**{_t('Kurzantwort', 'Short answer', lang)}**" + (f"\n{head}" if head else "")
                 + "\n" + "\n".join(f"- {s.text(lang)}" for s in leads))
    if parsed.intent in ("short", "sell"):
        lines = _short_lines(a, horizons, lang)
        if lines:
            parts.append(f"**{_t('Zur Short-Seite', 'On the Short side', lang)}**\n"
                         + "\n".join(lines) + "\n"
                         + _t("Bei einem Short ist der Verlust nicht auf den Einsatz begrenzt.",
                              "With a short the loss is not limited to the amount staked.", lang))
    parts.append(render_body(a, lang))
    if parsed.event_like:
        parts.append(_event_note(lang))
    return Answer("\n\n".join(parts) + "\n", parsed, a)
