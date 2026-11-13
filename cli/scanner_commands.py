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
import sys
from typing import Annotated

import typer

from tradingagents.default_config import DEFAULT_CONFIG


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

        try:
            language = _lang(lang)
            names = [resolve(t) for t in _split(tickers)]
            if not names:
                raise ValueError("no ticker given; pass them space or comma separated, "
                                 "e.g. NVDA SAP.DE")
            result = run_assess(names, as_of_date=date, with_options=options)
        except ValueError as exc:
            raise _fail(str(exc)) from None
        except VendorError as exc:
            raise _fail(f"Could not get price data: {exc}") from None
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

        def one(text: str) -> None:
            try:
                answer = answer_question(text, lang=language)
            except ValueError as exc:
                raise _fail(str(exc)) from None
            except VendorError as exc:
                raise _fail(f"Could not get price data: {exc}") from None
            if as_json:
                _json(answer.to_dict())
            else:
                typer.echo(answer.text)

        if question and question.strip():
            one(question)
            return
        if not _is_terminal():
            raise _fail("No question given. Pass it as an argument, "
                        'e.g. ask "Soll ich Nvidia oder SAP auf 3 Monate kaufen?"')
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

        try:
            language = _lang(lang)
            events = []
            for part in dates.split(","):
                if not part.strip():
                    continue
                day, _, label = part.partition(":")
                events.append((day.strip(), label.strip() or day.strip()))
            if not events:
                raise ValueError('no event given; use --dates "YYYY-MM-DD:label,..."')
            study = run_study(resolve(symbol), events, benchmark=benchmark or None)
        except ValueError as exc:
            raise _fail(str(exc)) from None
        except VendorError as exc:
            raise _fail(f"Could not get price data: {exc}") from None
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

        try:
            language = _lang(lang)
            if expiries < 1:
                raise ValueError("--expiries must be at least 1")
            moves = load_implied_moves(resolve(symbol), max_expiries=expiries)
        except ValueError as exc:
            raise _fail(str(exc)) from None
        except VendorError as exc:
            raise _fail(f"Could not get option data: {exc}") from None
        if as_json:
            _json([m.to_dict() for m in moves])
        else:
            typer.echo(render_implied_moves(moves, language))
