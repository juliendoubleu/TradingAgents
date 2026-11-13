"""Warnings and skip reasons as a code plus parameters, rendered in English or German.

Same idea as ``Signal.reason_items``: the scan stores *what* happened, the report picks
the language. ``Notice`` is a ``str`` holding the English text, so older code that
prints, searches or serialises a warning or reason as a plain string keeps working.
"""

from __future__ import annotations

from collections.abc import Callable

from tradingagents.scanner.fmt import format_text

# code -> language -> a template for ``fmt.format_text`` or a function of the parameters
_TEXT: dict[str, dict[str, str | Callable[[dict], str]]] = {
    "no_data": {"en": "no price data from Yahoo", "de": "keine Kursdaten von Yahoo"},
    "no_price_data": {"en": "no price data", "de": "keine Kursdaten"},
    "stale": {"en": "last bar {last} is stale", "de": "letzter Kurs vom {last} ist veraltet"},
    "short_history": {
        "en": "only {n} bars of history, need {need}",
        "de": "nur {n} Kurse Historie, benötigt werden {need}",
    },
    "no_last_close": {"en": "no valid last close", "de": "kein gültiger letzter Schlusskurs"},
    "benchmark_missing": {
        "en": "Benchmark {bench} had no data: relative strength is 0 for all and the regime "
              "has no benchmark facts.",
        "de": "Für den Benchmark {bench} gab es keine Daten: die relative Stärke ist überall 0 "
              "und das Regime hat keine Benchmark-Fakten.",
    },
    "vix_missing": {"en": "VIX had no data: it adds no points to the regime.",
                    "de": "Für den VIX gab es keine Daten: er gibt dem Regime keine Punkte."},
    "regime_input_missing": {"en": "Regime input not available: {item}.",
                             "de": "Regime-Eingabe nicht verfügbar: {item_de}."},
    "skipped_summary": {
        "en": "{n} of {total} instruments were skipped (see list).",
        "de": "{n} von {total} Instrumenten wurden übersprungen (siehe Liste).",
    },
    "stale_data": {
        "en": "The newest price bar is {asof}, {lag} days before the requested {requested}.",
        "de": "Der neueste Kurs ist vom {asof}, {lag} Tage vor dem angefragten {requested}.",
    },
    "running_bar_removed": {
        "en": "{n} of {total} instruments had a still-running daily bar removed (their exchange "
              "session was not over yet); the figures use the last finished session.",
        "de": "Bei {n} von {total} Instrumenten wurde ein noch laufender Tageskurs entfernt "
              "(die Börsensitzung war noch nicht beendet); die Zahlen nutzen die letzte "
              "abgeschlossene Sitzung.",
    },
    "past_date": {
        "en": "Scan date {date} is in the past: prices are back-adjusted for later dividends "
              "and splits, so price levels differ from what actually traded then.",
        "de": "Das Scan-Datum {date} liegt in der Vergangenheit: Kurse sind rückwirkend um "
              "spätere Dividenden und Splits bereinigt, die Kursniveaus weichen daher von den "
              "damals gehandelten ab.",
    },
}

_ITEM_DE = {
    "benchmark trend": "Benchmark-Trend",
    "benchmark 3-month return": "Benchmark-3-Monats-Rendite",
    "breadth": "Marktbreite",
    "VIX": "VIX",
}


def register(texts: dict[str, dict]) -> None:
    """Add message codes, so a module owns the wording of the notes it raises.

    ``texts`` maps a code to ``{"en": ..., "de": ...}``; each text is a template for
    ``fmt.format_text`` (numbers come out in the language of the reader) or a function of
    the parameters dict for sentences that need more than a template.  A code may be
    registered again with identical texts (a module imported twice); a different text for
    an existing code is an error, because two modules would then disagree (functions are
    replaced, so a reloaded module registers itself again).
    """
    for code, table in texts.items():
        if "en" not in table:
            raise ValueError(f"notice {code!r} needs an English text")
        callable_text = any(callable(v) for v in table.values())
        if code in _TEXT and _TEXT[code] != table and not callable_text:
            raise ValueError(f"notice code {code!r} is already registered with another text")
        _TEXT[code] = table


def _render(code: str, lang: str, params: dict) -> str:
    table = _TEXT[code]
    entry = table.get(lang, table["en"])
    if callable(entry):
        return entry(params)
    return format_text(entry, lang, **params)


def _rebuild(code: str, params: dict) -> Notice:
    return Notice(code, **params)


class Notice(str):
    """A translatable message: ``code`` + ``params``; ``str(notice)`` is the English text."""

    code: str
    params: dict

    def __new__(cls, code: str, **params) -> Notice:
        if code == "regime_input_missing":
            params.setdefault("item_de", _ITEM_DE.get(params.get("item", ""), params.get("item")))
        obj = super().__new__(cls, _render(code, "en", params))
        obj.code = code
        obj.params = params
        return obj

    def __reduce__(self):
        """Pickle and copy by code + parameters: ``__new__`` takes a code, which the default
        ``str`` protocol would not pass."""
        return (_rebuild, (self.code, dict(self.params)))

    def text(self, lang: str = "en") -> str:
        return _render(self.code, lang, self.params)

    def to_dict(self) -> dict:
        return {"code": self.code, "params": dict(self.params),
                "text": self.text("en"), "text_de": self.text("de")}
