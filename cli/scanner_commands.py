"""The question-answering commands of the scanner: ``assess``, ``ask``, ``event-study``,
``implied-move``.

They live here, behind ``register(app)``, so the main CLI can mount them with one line
and tests can run them on a bare ``typer.Typer()``.  Results go to stdout as plain
Markdown (or only JSON with ``--json``) so a person on a phone and a Claude session
reading the output see the same thing; errors are one red line on stderr and exit code 1,
never a traceback.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Annotated

import typer

from tradingagents.default_config import DEFAULT_CONFIG

# yfinance prints "Failed download" noise for an unknown symbol; the commands report it themselves
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

_M = {
    "no_ticker": ("no ticker given; pass them space or comma separated, e.g. NVDA SAP.DE",
                  "Kein Wertpapier angegeben; bitte mit Leerzeichen oder Komma getrennt, "
                  "z. B. NVDA SAP.DE"),
    "nothing_assessed": ("None of the instruments could be assessed: {names} (no price data "
                         "found; check the symbol)",
                         "Keines der Wertpapiere konnte eingeschätzt werden: {names} (keine "
                         "Kursdaten gefunden; bitte das Symbol prüfen)"),
    "empty_question": ("No question given. Pass it as an argument, e.g. ask \"Soll ich Nvidia "
                       "oder SAP auf 3 Monate kaufen?\"",
                       "Keine Frage angegeben. Bitte als Argument übergeben, z. B. ask "
                       "\"Soll ich Nvidia oder SAP auf 3 Monate kaufen?\""),
    "no_event": ('no event given; use --dates "YYYY-MM-DD:label,..."',
                 'Kein Ereignis angegeben; bitte --dates "JJJJ-MM-TT:Name,..." verwenden'),
    "bad_event_date": ("bad date {d!r}: expected YYYY-MM-DD",
                       "Ungültiges Datum {d!r}: erwartet wird JJJJ-MM-TT"),
    "expiries": ("--expiries must be at least 1", "--expiries muss mindestens 1 sein"),
    "no_prices": ("Could not get price data: {err}",
                  "Kursdaten konnten nicht geladen werden: {err}"),
    "no_options": ("Could not get option data: {err}",
                   "Optionsdaten konnten nicht geladen werden: {err}"),
    "unknown_symbol": ("No price data found for {symbol}; check the symbol",
                       "Für {symbol} wurden keine Kursdaten gefunden; bitte das Symbol prüfen"),
}


def _m(key: str, lang: str, **kw) -> str:
    return _M[key][1 if lang == "de" else 0].format(**kw)


def vendor_message(exc: Exception, lang: str) -> str:
    """The vendor's error in the reader's language: its English text is translated, never
    shown next to a German sentence."""
    from tradingagents.dataflows.errors import NoMarketDataError

    text = str(exc)
    if lang != "de":
        return text
    if isinstance(exc, NoMarketDataError):
        return _m("unknown_symbol", lang, symbol=exc.symbol)
    m = re.search(r"returned no price data for any of (\d+) symbols", text)
    if m:
        return f"Yahoo Finance lieferte für keines der {m.group(1)} Symbole Kursdaten"
    m = re.search(r"No price data for (\d+) of (\d+) (\w+) symbols", text)
    if m:
        return (f"Für {m.group(1)} von {m.group(2)} Symbolen ({m.group(3).upper()}) gab es keine "
                "Kursdaten; Yahoo drosselt vermutlich, und ein Teilmarkt ist kein Marktüberblick. "
                "Bitte in einigen Minuten erneut versuchen")
    m = re.search(r"returned no (\w+) bars", text)
    if m:
        return "Yahoo Finance lieferte keine Kursdaten für den gewünschten Zeitraum"
    return "Yahoo Finance konnte die Anfrage nicht beantworten"


def data_error_text(exc: Exception, lang: str, key: str = "no_prices") -> str:
    """One sentence for a vendor error: an unknown symbol is named as such, anything else gets
    the translated vendor message after ``key``'s sentence."""
    from tradingagents.dataflows.errors import NoMarketDataError

    if isinstance(exc, NoMarketDataError):
        return _m("unknown_symbol", lang, symbol=exc.symbol)
    return _m(key, lang, err=vendor_message(exc, lang))


def _default_lang() -> str:
    language = str(DEFAULT_CONFIG.get("output_language", "English")).strip().lower()
    return "de" if language in ("german", "deutsch", "de") else "en"


def _lang(value: str | None) -> str:
    lang = value if value is not None else _default_lang()
    if lang not in ("en", "de"):
        raise ValueError("--lang must be en or de")
    return lang


def _fail(message: str) -> typer.Exit:
    typer.secho(message, fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


def _json(data) -> None:
    typer.echo(json.dumps(data, ensure_ascii=False, indent=2))


def _is_terminal() -> bool:
    return sys.stdin.isatty()


def _split(items: list[str]) -> list[str]:
    return [p.strip() for item in items for p in item.split(",") if p.strip()]


def register(app: typer.Typer) -> None:
    """Add ``assess``, ``ask``, ``event-study`` and ``implied-move`` to ``app``."""

    @app.command()
    def assess(
        tickers: Annotated[list[str], typer.Argument(
            help="Tickers or names, space or comma separated: NVDA SAP.DE BTC-USD")],
        date: str = typer.Option(None, "--date", help="As-of date, YYYY-MM-DD (default: today)"),
        options: bool = typer.Option(False, "--options", help="Add option-implied moves (US stocks)"),
        lang: str = typer.Option(None, "--lang", help="Output language, en or de"),
        as_json: bool = typer.Option(False, "--json", help="Print only JSON"),
    ):
        """Assess instruments per investment horizon, with the evidence behind each statement."""
        from tradingagents.dataflows.errors import VendorError
        from tradingagents.scanner.aliases import resolve
        from tradingagents.scanner.assess import assess as run_assess, render_assessment
        from tradingagents.scanner.scan import parse_date

        language = "de" if lang == "de" else "en"
        try:
            language = _lang(lang)
            names = [resolve(t) for t in _split(tickers)]
            if not names:
                raise ValueError(_m("no_ticker", language))
            if date:
                parse_date(date, lang=language)
            result = run_assess(names, as_of_date=date, with_options=options)
        except ValueError as exc:
            raise _fail(str(exc)) from None
        except VendorError as exc:
            raise _fail(data_error_text(exc, language)) from None
        if result.tickers and all(t.error is not None for t in result.tickers):
            raise _fail(_m("nothing_assessed", language, names=", ".join(
                t.symbol for t in result.tickers)))
        if as_json:
            _json(result.to_dict())
        else:
            typer.echo(render_assessment(result, language))

    @app.command()
    def ask(
        question: str = typer.Argument(None, help="Your question, in German or English"),
        lang: str = typer.Option(None, "--lang", help="Answer language, en or de "
                                 "(default: the question's language)"),
        as_json: bool = typer.Option(False, "--json", help="Print only JSON"),
    ):
        """Ask about instruments ("Nvidia oder SAP auf 3 Monate?") and get a data-based answer.

        Deterministic: every sentence comes from the scanner's data or is a fixed template.
        Open event questions it cannot answer say so and show the commands to run.
        """
        from tradingagents.dataflows.errors import VendorError
        from tradingagents.scanner.ask import answer_question

        try:
            language = None if lang is None else _lang(lang)
        except ValueError as exc:
            raise _fail(str(exc)) from None
        shown = language or _default_lang()

        def one(text: str) -> None:
            try:
                answer = answer_question(text, lang=language)
            except ValueError as exc:
                raise _fail(_m("empty_question", shown) if str(exc) == "empty question"
                            else str(exc)) from None
            except VendorError as exc:
                raise _fail(data_error_text(exc, language or shown)) from None
            if as_json:
                _json(answer.to_dict())
            else:
                typer.echo(answer.text)

        if question and question.strip():
            one(question)
            return
        if not _is_terminal():
            raise _fail(_m("empty_question", shown))
        while True:
            line = typer.prompt("Frage", default="", show_default=False).strip()
            if not line or line.lower() in ("exit", "quit", "ende"):
                return
            try:
                one(line)
            except typer.Exit:
                continue       # the error is printed; the next question may work

    @app.command("event-study")
    def event_study(
        symbol: str = typer.Argument(..., help="Ticker or name, e.g. TTWO"),
        dates: str = typer.Option(
            ..., "--dates", help='Comma-separated "YYYY-MM-DD:label" events, '
                                 'e.g. "2018-10-26:Red Dead Redemption 2"'),
        benchmark: str = typer.Option("^GSPC", "--benchmark", help="Benchmark symbol"),
        lang: str = typer.Option(None, "--lang", help="Output language, en or de"),
        as_json: bool = typer.Option(False, "--json", help="Print only JSON"),
    ):
        """How a symbol behaved around dated events, against a benchmark (an upcoming date
        gets only the run-up so far)."""
        from tradingagents.dataflows.errors import VendorError
        from tradingagents.scanner.aliases import resolve
        from tradingagents.scanner.events import event_study as run_study, render_event_study

        language = "de" if lang == "de" else "en"
        try:
            language = _lang(lang)
            events = []
            for part in dates.split(","):
                if not part.strip():
                    continue
                day, _, label = part.partition(":")
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day.strip()):
                    raise ValueError(_m("bad_event_date", language, d=day.strip()))
                events.append((day.strip(), label.strip() or day.strip()))
            if not events:
                raise ValueError(_m("no_event", language))
            study = run_study(resolve(symbol), events, benchmark=benchmark or None)
        except ValueError as exc:
            msg = str(exc)
            if msg.startswith("bad date ") and language == "de":
                msg = _m("bad_event_date", language, d=msg.split("'")[1] if "'" in msg else msg)
            raise _fail(msg) from None
        except VendorError as exc:
            raise _fail(data_error_text(exc, language)) from None
        if as_json:
            _json(study.to_dict())
        else:
            typer.echo(render_event_study(study, language))

    @app.command("implied-move")
    def implied_move(
        symbol: str = typer.Argument(..., help="Ticker or name, e.g. TTWO"),
        expiries: int = typer.Option(4, "--expiries", help="How many upcoming expiries to read"),
        lang: str = typer.Option(None, "--lang", help="Output language, en or de"),
        as_json: bool = typer.Option(False, "--json", help="Print only JSON"),
    ):
        """What the options market prices in: the at-the-money straddle's implied move."""
        from tradingagents.dataflows.errors import VendorError
        from tradingagents.dataflows.vendors.yahoo.options import load_implied_moves
        from tradingagents.scanner.aliases import resolve
        from tradingagents.scanner.events import render_implied_moves

        language = "de" if lang == "de" else "en"
        try:
            language = _lang(lang)
            if expiries < 1:
                raise ValueError(_m("expiries", language))
            moves = load_implied_moves(resolve(symbol), max_expiries=expiries)
        except ValueError as exc:
            raise _fail(str(exc)) from None
        except VendorError as exc:
            raise _fail(data_error_text(exc, language, "no_options")) from None
        if as_json:
            _json([m.to_dict() for m in moves])
        else:
            typer.echo(render_implied_moves(moves, language))
