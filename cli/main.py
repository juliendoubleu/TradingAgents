import sys

import typer

from cli.display import console
from cli.models import AnalystType, AssetType
from cli.prompts import filter_analysts_for_asset_type, parse_analysts
from cli.run import run_analysis
from cli.scanner_commands import (
    register as register_scanner_commands,
    vendor_message as _vendor_message,
)
from tradingagents.backtest import iter_grid, run_backtest, summarize
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.portfolio import load_portfolio

# prompt_toolkit's win32 output module is importable only on Windows (it asserts
# the platform at import time), so gate on the platform rather than catching the
# failure — that way a genuinely broken prompt_toolkit on Windows still surfaces
# instead of silently disabling the handler below. Off Windows this stays an
# empty tuple, which `except` accepts and never matches (#1138).
if sys.platform == "win32":  # pragma: no cover - platform dependent
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError

    _NO_CONSOLE_ERRORS: tuple[type[BaseException], ...] = (NoConsoleScreenBufferError,)
else:
    _NO_CONSOLE_ERRORS = ()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


@app.callback(invoke_without_command=True)
def analyze(
    ctx: typer.Context,
    checkpoint: bool | None = typer.Option(
        None,
        "--checkpoint/--no-checkpoint",
        help="Enable/disable checkpoint-resume (save state after each node so a "
        "crashed run can resume). Omit to honor TRADINGAGENTS_CHECKPOINT_ENABLED.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
    portfolio: str = typer.Option(
        None,
        "--portfolio",
        help="JSON file with current holdings and cash, so the trader, risk and "
        "portfolio agents size against your actual position.",
    ),
    ticker: str = typer.Option(None, "--ticker", help="Ticker to analyze, e.g. NVDA or 0700.HK; skips the prompt"),
    date: str = typer.Option(None, "--date", help="Analysis date, YYYY-MM-DD; skips the prompt"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts, e.g. market,news; skips the prompt"
    ),
    save: bool | None = typer.Option(
        None, "--save/--no-save", help="Save the report under results_dir without asking"
    ),
    show: bool | None = typer.Option(
        None, "--show/--no-show", help="Show the full report at the end without asking"
    ),
    html: bool | None = typer.Option(
        None, "--html/--no-html",
        help="Save the report as one HTML page too, complete_report.html, without asking (default: yes)",
    ),
):
    """Run an analysis. This is what a bare `tradingagents` does.

    Flags answer their questions; with provider, models, depth and language also
    set through TRADINGAGENTS_* variables, the run asks nothing.
    """
    if ctx.invoked_subcommand is not None:
        return
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    portfolio_context = None
    if portfolio:
        try:
            portfolio_context = load_portfolio(portfolio)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None

    try:
        flags = {"ticker": ticker, "date": date, "analysts": analysts, "save": save, "show": show, "html": html}
        run_analysis(checkpoint=checkpoint, portfolio=portfolio_context, flags=flags)
    except _NO_CONSOLE_ERRORS:
        # A terminal with no console buffer cannot host the interactive prompts.
        # Emit one actionable line on stderr instead of a prompt_toolkit
        # traceback; plain text, since rich may not render here either (#1138).
        typer.echo(
            "Error: no Windows console available. The interactive CLI needs a real "
            "console buffer — run it from Windows Terminal, PowerShell, or cmd.exe "
            "rather than a piped or embedded terminal.",
            err=True,
        )
        raise typer.Exit(code=1) from None


@app.command()
def backtest(
    tickers: str = typer.Argument(..., help="Comma-separated tickers, e.g. NVDA,AAPL"),
    start: str = typer.Option(..., "--start", help="First analysis date, YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="Last analysis date, YYYY-MM-DD"),
    every: int = typer.Option(7, "--every", help="Days between analysis dates"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts to run: market, sentiment, news, fundamentals; omit for all the asset type allows"
    ),
    asset_type: str = typer.Option("stock", "--asset-type", help="stock or crypto"),
    portfolio: str = typer.Option(
        None, "--portfolio", help="JSON file with holdings and cash, held constant across the grid"
    ),
    run_id: str = typer.Option(
        None, "--run-id", help="Continue an earlier sweep: its cells are skipped and its log reused"
    ),
):
    """Score past decisions over a grid of tickers and dates."""

    try:
        dates = iter_grid(start, end, every)
        book = load_portfolio(portfolio) if portfolio else None
        kind = AssetType(asset_type.strip().lower())
        # The analysts are named and checked as for an analysis; without a
        # choice, every analyst the asset type allows runs.
        chosen = (parse_analysts(analysts, kind) if analysts
                  else filter_analysts_for_asset_type(list(AnalystType), kind))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    names = [t.strip() for t in tickers.split(",") if t.strip()]
    if not names:
        console.print("[red]No ticker to analyze; pass them comma-separated, e.g. NVDA,AAPL[/red]")
        raise typer.Exit(code=1)

    def show_progress(done, total, ticker, date):
        console.print(f"[dim][{done}/{total}] {ticker} {date}[/dim]")

    kwargs = {"asset_type": kind.value, "portfolio": book, "run_id": run_id, "progress": show_progress,
              "selected_analysts": [a.value for a in chosen]}

    try:
        result = run_backtest(names, dates, DEFAULT_CONFIG, **kwargs)
    except Exception as exc:  # a missing key or an unknown analyst is a setup error
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(summarize(result).render())
    console.print(f"\nRan {result.cells_run} cells, skipped {result.skipped}. Log: {result.log_path}")
    console.print(f"Continue or settle this sweep: --run-id {result.run_id}")
    for ticker, date, reason in result.failures:
        console.print(f"[yellow]failed:[/yellow] {ticker} {date}: {reason}")
    for ticker, reason in result.settlement_failures:
        console.print(f"[yellow]unsettled:[/yellow] {ticker}: {reason}")


_CLI = {
    "status_scan": ("Downloading prices and scanning...", "Lade Kurse und scanne ..."),
    "status_validate": ("Validating the signals over {months} months...",
                        "Prüfe die Signale über {months} Monate ..."),
    "no_vendor": ("Could not get price data: {err}", "Kursdaten konnten nicht geladen werden: {err}"),
    "failed": ("{market} failed: {err}", "{market} fehlgeschlagen: {err}"),
    "failed_digest": ("Not loaded: {markets}.", "Nicht geladen: {markets}."),
    "all_failed": ("No market could be scanned.", "Kein Markt konnte gescannt werden."),
    "digest_failed": ("Market scan failed: price data unavailable - no digest",
                      "Markt-Scan fehlgeschlagen: Kursdaten nicht abrufbar – kein Digest"),
    "unexpected": ("Unexpected error", "Unerwarteter Fehler"),
    "data_as_of": ("data as of", "Datenstand"),
    "regime": ("Market regime", "Marktregime"),
    "val_title": ("Walk-forward validation of the technical picture",
                  "Walk-forward-Validierung des technischen Bildes"),
    "verdict": ("Verdict", "Urteil"),
    "val_head": ("{n} rebalances, {first} - {last}, {h}d",
                 "{n} Termine, {first} - {last}, {h} Tage"),
    "per_date": ("per date", "je Termin"), "pooled": ("pooled", "gepoolt"),
    "spread": ("Long - Short spread", "Long-minus-Short-Spread"),
    "dates": ("{n} dates", "{n} Termine"),
    "agents": ("Running the LLM agents on {n} ideas: {names} ...",
               "Starte die LLM-Agenten für {n} Ideen: {names} ..."),
    "agent_failed": ("{sym} failed: {err}", "{sym} fehlgeschlagen: {err}"),
    "saved": ("Saved:", "Gespeichert:"),
    "disclaimer": (
        "Research tool, not investment advice. Shorts can lose more than the stake.",
        "Recherchewerkzeug, keine Anlageberatung. Shorts können mehr als den Einsatz kosten."),
    "disclaimer_sizes": ("Sizes are per trade and do not add up.",
                         "Größen gelten je Trade und addieren sich nicht."),
    "err_tz": ("unknown time zone {tz!r}; use an IANA name such as Europe/Berlin",
               "unbekannte Zeitzone {tz!r}; nutze einen IANA-Namen wie Europe/Berlin"),
    "err_horizon": ("unknown horizon {h!r}; choose from: {valid}",
                    "unbekannter Horizont {h!r}; wähle aus: {valid}"),
    "err_universe": ("unknown universe {u!r}; choose one of: {valid}",
                     "unbekanntes Universum {u!r}; wähle aus: {valid}"),
    "err_tickers_multi": ("--tickers works with one universe only",
                          "--tickers geht nur mit einem Universum"),
    "err_numbers": ("--top and --months must be at least 1, and --deep at least 0",
                    "--top und --months müssen mindestens 1 sein, --deep mindestens 0"),
    "err_no_ticker": ("no ticker given; pass them comma-separated, e.g. AAPL,MSFT",
                      "kein Ticker angegeben; kommagetrennt übergeben, z. B. AAPL,MSFT"),
    "err_lang": ("--lang must be en or de", "--lang muss en oder de sein"),
}


def _c(key: str, lang: str, **kw) -> str:
    text = _CLI[key][1 if lang == "de" else 0]
    return text.format(**kw) if kw else text


def _print_recommendations(result, lang: str, horizons) -> None:
    """The per-horizon recommendation tables, or the plain reason there is none."""
    from rich.markup import escape
    from rich.table import Table

    from tradingagents.scanner.fmt import fmt_date, fmt_num, fmt_price
    from tradingagents.scanner.horizons import holding_text
    from tradingagents.scanner.report import (
        SIDES,
        any_size,
        composite_conflicts,
        conflict_text,
        general_notes,
        horizon_views,
        size_notes,
        sizing_note,
        tr,
    )

    if result.ideas is None:
        console.print(f"\n[yellow]{tr('ideas_missing', lang)}[/yellow]")
        return
    views = horizon_views(result, horizons)
    conflicts = composite_conflicts(result)
    console.print(f"\n[bold]{tr('recs', lang)}[/bold]")
    if not any(v.recs for v in views):
        console.print(f"[bold yellow]{escape(tr('no_rec_any', lang))}[/bold yellow]",
                      soft_wrap=True)
    for line in general_notes(result, lang):
        console.print(f"[yellow]{escape(line)}[/yellow]", soft_wrap=True)
    for view in views:
        console.print(f"\n[bold]{escape(holding_text(view.key, result.ideas.family, lang))}[/bold]")
        if view.recs:
            table = Table()
            for col in ("dir", "ticker", "ref", "stop", "size"):
                table.add_column(tr(col, lang), no_wrap=True)
            for i in view.recs:
                colour = "green" if i.direction == "Long" else "red"
                table.add_row(
                    f"[{colour}]{i.direction}[/{colour}]", escape(i.symbol),
                    f"{fmt_price(i.reference_price, lang)} ({fmt_date(i.reference_date, lang)})",
                    fmt_price(i.stop, lang) if i.stop is not None else tr("no_stop", lang),
                    "-" if i.size_pct is None else f"{fmt_num(i.size_pct, 1, lang)} %")
            console.print(table)
            shown = list(dict.fromkeys(
                (i.direction, i.evidence.text(lang)) for i in view.recs))
            for direction, sentence in shown:
                console.print(f"  {direction}: {escape(sentence)}", soft_wrap=True)
            for line in size_notes(view.recs, lang):
                console.print(f"  [dim]{escape(line)}[/dim]", soft_wrap=True)
            for i in view.recs:
                flag = conflict_text(i, conflicts.get(i.symbol), lang)
                if flag:
                    console.print(f"  [yellow]{escape(i.symbol)}: {escape(flag)}[/yellow]",
                                  soft_wrap=True)
                if i.outside:
                    console.print(f"  [yellow]{escape(i.outside.get(lang, i.outside['en']))}"
                                  "[/yellow]", soft_wrap=True)
        why = []
        for side in SIDES:
            if not view.side(side):
                why += [n.text(lang) for n in view.why[side] if n.text(lang) not in why]
        if not view.recs and not why:
            why = [tr("no_rec", lang)]
        for line in why:
            console.print(f"  [yellow]{escape(line)}[/yellow]", soft_wrap=True)
    if any_size(views):
        console.print(f"\n[dim]{escape(sizing_note(lang))}[/dim]", soft_wrap=True)


def _print_watch(result, lang: str, horizons) -> None:
    """The watch list: explicitly not a recommendation; the composite names appear once."""
    from rich.markup import escape

    from tradingagents.scanner.fmt import fmt_num
    from tradingagents.scanner.horizons import holding_text, watch_direction
    from tradingagents.scanner.report import _specific_watch, horizon_keys, tr

    keys = horizon_keys(horizons)
    console.print(f"\n[bold]{tr('watch', lang)}[/bold]")
    console.print(f"[dim]{escape(tr('watch_intro', lang))}[/dim]", soft_wrap=True)
    family = result.ideas.family if result.ideas is not None else ""
    groups: dict[tuple, list[str]] = {}
    for idea in _specific_watch(result, keys):
        avoid = idea.stance == "avoid"
        key = (idea.horizon, tr("avoid", lang) if avoid else watch_direction(idea.direction, lang),
               idea.evidence.text(lang))
        groups.setdefault(key, []).append(idea.symbol)
    for (horizon, side, sentence), symbols in groups.items():
        console.print(f"  [bold]{escape(holding_text(horizon, family, lang).split(' – ')[0])} "
                      f"{escape(side)}[/bold]: {escape(', '.join(symbols))}", soft_wrap=True)
        console.print(f"    {escape(sentence)}", soft_wrap=True)
    for title, signals, colour in ((tr("strong", lang), result.longs, "green"),
                                   (tr("weak", lang), result.shorts, "red")):
        console.print(f"\n[{colour}]{title}[/{colour}]")
        if not signals:
            console.print("  -")
        for s in signals:
            sign = "+" if s.score > 0 else ""
            reasons = "; ".join(s.reasons_text(lang))
            console.print(f"  [bold]{escape(s.symbol)}[/bold] ({sign}{fmt_num(s.score, 1, lang)}): "
                          f"{escape(reasons)}", soft_wrap=True)


def _print_solid(result, lang: str, horizons) -> None:
    from rich.markup import escape
    from rich.table import Table

    from tradingagents.scanner.report import (
        evidence_cells,
        horizon_keys,
        horizon_label,
        label_text,
        solid_paragraph,
        tr,
    )

    console.print(f"\n[bold]{tr('solid', lang)}[/bold]")
    keys = horizon_keys(horizons)
    if result.ideas is not None and result.ideas.evidence_available:
        table = Table()
        for col in ("area", "dir", "label"):
            table.add_column(tr(col, lang), no_wrap=True)
        for c in evidence_cells(result, keys):
            table.add_row(horizon_label(c["horizon"], lang), c["side"], label_text(c["label"], lang))
        console.print(table)
    console.print(escape(solid_paragraph(result, lang, keys)), soft_wrap=True)


def _print_validation(validation, lang: str) -> None:
    from rich.markup import escape
    from rich.table import Table

    from tradingagents.scanner.fmt import fmt_date, fmt_num, fmt_pct
    from tradingagents.scanner.report import tr, validation_verdict
    from tradingagents.scanner.validation import CAVEATS

    console.print(f"\n[bold]{_c('val_title', lang)}[/bold]")
    console.print(f"[bold]{_c('verdict', lang)}:[/bold] {escape(validation_verdict(validation, lang))}",
                  soft_wrap=True)
    if validation.rebalances:
        console.print(_c("val_head", lang, n=validation.rebalances,
                         first=fmt_date(validation.first_rebalance, lang),
                         last=fmt_date(validation.last_rebalance, lang), h=validation.horizon))
        vt = Table()
        for col in ("direction", "calls", "avg_ret", "avg_exc", "hit", "pnl"):
            vt.add_column(tr(col, lang))
        for name in ("Long", "Short", "Neutral"):
            st = validation.stats[name]
            vt.add_row(name, str(st.calls), fmt_pct(st.avg_return, 2, True, lang),
                       fmt_pct(st.avg_excess, 2, True, lang), fmt_pct(st.hit_rate, 0, False, lang),
                       fmt_pct(st.avg_pnl, 2, True, lang))
        console.print(vt)
        t = (f" (t={fmt_num(validation.spread_tstat, 2, lang)}, "
             f"{_c('dates', lang, n=validation.spread_dates)})"
             if validation.spread_tstat is not None else "")
        console.print(f"{_c('spread', lang)} ({_c('per_date', lang)}): "
                      f"{fmt_pct(validation.spread, 2, True, lang)}{t}")
        console.print(f"{_c('spread', lang)} ({_c('pooled', lang)}): "
                      f"{fmt_pct(validation.spread_pooled, 2, True, lang)}")
    console.print(f"[dim]{tr('caveats', lang)}:[/dim]")
    for caveat in CAVEATS[lang]:
        console.print(f"[dim]- {escape(caveat)}[/dim]", soft_wrap=True)


@app.command()
def scan(
    universe: str = typer.Option(
        "us", "--universe",
        help="us, sectors, global, dax, crypto or all; several comma-separated, e.g. us,dax,crypto"),
    tickers: str = typer.Option(
        None, "--tickers", help="Comma-separated tickers that replace the universe's members"
    ),
    date: str = typer.Option(None, "--date", help="Scan date, YYYY-MM-DD (default: today)"),
    top: int = typer.Option(10, "--top", help="How many ideas per list to show"),
    horizon: str = typer.Option(
        None, "--horizon",
        help="Only these investment horizons, comma-separated: day, swing, medium, long (default: all)"),
    digest: bool = typer.Option(
        False, "--digest/--no-digest",
        help="Print only the short phone digest (headline, blank line, body) as plain text"),
    validate: bool = typer.Option(
        False, "--validate/--no-validate",
        help="Also check the technical picture walk-forward against what happened next",
    ),
    months: int = typer.Option(24, "--months", help="History the validation covers, in months"),
    deep: int = typer.Option(
        0, "--deep", help="Run the LLM agents on the N best technical Long and Short names (needs an LLM key)"
    ),
    save: bool = typer.Option(True, "--save/--no-save",
                              help="Write scan.md, scan.html, scan.json, digest.txt"),
    lang: str = typer.Option(None, "--lang", help="Report language, en or de"),
    output: str = typer.Option(None, "--output", help="Directory to save under (default: results_dir)"),
    tz: str = typer.Option("Europe/Berlin", "--tz",
                           help="IANA time zone of the digest clock and its weekend flag"),
):
    """Scan markets and show ideas by investment horizon, with no LLM needed.

    Recommendations appear only where a rule passed a measured out-of-sample test; the
    scanner's technical picture (trend, momentum, relative strength, MACD, RSI, volume)
    is shown as a watch list because it has no measured edge. A research tool, not
    investment advice.
    """
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    from rich.markup import escape

    from tradingagents.dataflows.errors import VendorError
    from tradingagents.scanner.deep_dive import check_llm_credentials, deep_dive
    from tradingagents.scanner.digest import build_digest
    from tradingagents.scanner.fmt import fmt_date
    from tradingagents.scanner.regime import LABELS as REGIME_LABELS
    from tradingagents.scanner.report import (
        HORIZON_ORDER,
        save_scan,
        universe_description,
    )
    from tradingagents.scanner.scan import parse_date, run_scan
    from tradingagents.scanner.universe import universe_names
    from tradingagents.scanner.validation import validate as run_validation

    if lang is None:
        language = str(DEFAULT_CONFIG.get("output_language", "English")).strip().lower()
        lang_ok = "de" if language in ("german", "deutsch", "de") else "en"
    else:
        lang_ok = lang
    msg_lang = lang_ok if lang_ok in ("en", "de") else "en"
    names = [t.strip() for t in tickers.split(",") if t.strip()] if tickers else None
    wanted = list(dict.fromkeys(u.strip().lower() for u in universe.split(",") if u.strip()))
    try:
        as_of = parse_date(date, lang=msg_lang)
        if tickers is not None and not names:
            raise ValueError(_c("err_no_ticker", msg_lang))
        if top < 1 or months < 1 or deep < 0:
            raise ValueError(_c("err_numbers", msg_lang))
        if lang_ok not in ("en", "de"):
            raise ValueError(_c("err_lang", "en"))
        lang = lang_ok
        try:
            ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
            raise ValueError(_c("err_tz", lang, tz=tz)) from exc
        for u in wanted or ["us"]:
            if u not in universe_names():
                raise ValueError(_c("err_universe", lang, u=u, valid=", ".join(universe_names())))
        wanted = wanted or ["us"]
        if names and len(wanted) > 1:
            raise ValueError(_c("err_tickers_multi", lang))
        horizons = None
        if horizon:
            horizons = list(dict.fromkeys(h.strip().lower() for h in horizon.split(",") if h.strip()))
            for h in horizons:
                if h not in HORIZON_ORDER:
                    raise ValueError(_c("err_horizon", lang, h=h, valid=", ".join(HORIZON_ORDER[::-1])))
        if deep:
            check_llm_credentials(DEFAULT_CONFIG)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from None

    results, failures = [], []
    for name in wanted:
        try:
            if digest:
                result = run_scan(name, as_of, tickers=names, top=top)
                validation = (run_validation(name, as_of, tickers=names, months=months)
                              if validate else None)
            else:
                with console.status(_c("status_scan", lang)):
                    result = run_scan(name, as_of, tickers=names, top=top)
                validation = None
                if validate:
                    with console.status(_c("status_validate", lang, months=months)):
                        validation = run_validation(name, as_of, tickers=names, months=months)
        except VendorError as exc:
            failures.append((name, _c("no_vendor", lang, err=_vendor_message(exc, lang))))
            continue
        except Exception as exc:  # one broken market must not stop the others
            failures.append((name, f"{_c('unexpected', lang)}: {type(exc).__name__}" if lang == "de"
                             else f"{type(exc).__name__}: {exc}"))
            continue
        results.append((result, validation))

    for name, err in failures:
        if len(wanted) == 1:
            typer.echo(err, err=True) if digest else console.print(f"[red]{err}[/red]", soft_wrap=True)
        else:
            line = _c("failed", lang, market=name, err=err)
            typer.echo(line, err=True) if digest else console.print(f"[red]{line}[/red]", soft_wrap=True)
    if not results:
        if len(wanted) > 1:
            typer.echo(_c("all_failed", lang), err=True)
        if digest:
            # a scheduled job pushes stdout: say in one line that there is no digest
            typer.echo(_c("digest_failed", lang))
        raise typer.Exit(code=1)

    saved = []
    if digest:
        text = build_digest([r for r, _ in results], lang, horizons=horizons, tz=tz).text
        if failures:
            text += "\n\n" + _c("failed_digest", lang, markets=", ".join(n for n, _ in failures))
        typer.echo(text)
        # The digest run stays silent (no agents, no tables); the files are written when asked.
        saved = [(r, v, None) for r, v in results]
    else:
        for result, validation in results:
            console.print(f"\n[bold]{escape(universe_description(result, lang))}[/bold] "
                          f"({result.universe}), {_c('data_as_of', lang)} "
                          f"{fmt_date(result.as_of_date, lang) if result.as_of_date else '-'}")
            tone = {"risk-on": "green", "risk-off": "red", "neutral": "yellow"}[result.regime.label]
            console.print(f"{_c('regime', lang)}: [{tone}]{REGIME_LABELS[result.regime.label][lang]}"
                          f"[/{tone}] ({result.regime.points:+d})")
            for line in result.regime.describe(lang):
                console.print(f"  - {escape(line)}", soft_wrap=True)
            for warning in result.warnings:
                text = warning.text(lang) if hasattr(warning, "text") else warning
                console.print(f"[yellow]{escape(text)}[/yellow]", soft_wrap=True)
            _print_recommendations(result, lang, horizons)
            _print_watch(result, lang, horizons)
            _print_solid(result, lang, horizons)
            if validation is not None:
                _print_validation(validation, lang)
            deep_result = None
            if deep:
                picks = sorted([*result.longs, *result.shorts], key=lambda s: -abs(s.score))[:deep]
                console.print(f"\n{_c('agents', lang, n=len(picks), names=', '.join(s.symbol for s in picks))}")
                deep_result = deep_dive(picks, result.as_of_date or result.requested_date, DEFAULT_CONFIG)
                for symbol, rating in deep_result.ratings.items():
                    console.print(f"  {symbol}: {rating}")
                for symbol, error in deep_result.errors.items():
                    console.print(f"  [yellow]{escape(_c('agent_failed', lang, sym=symbol, err=error))}[/yellow]")
            saved.append((result, validation, deep_result))
    if save:
        for result, validation, deep_result in saved:
            paths = save_scan(result, validation, deep_result, lang,
                              output or DEFAULT_CONFIG["results_dir"], horizons, tz=tz)
            if not digest:
                console.print(f"\n{_c('saved', lang)} {paths['md']}\n       {paths['html']}\n"
                              f"       {paths['json']}\n       {paths['digest']}", soft_wrap=True)
    if not digest:
        from tradingagents.scanner.report import any_size, horizon_views
        text = _c("disclaimer", lang)
        if any(r.ideas is not None and any_size(horizon_views(r, horizons))
               for r, _v, _d in saved):
            text += " " + _c("disclaimer_sizes", lang)
        console.print(f"\n[dim]{text}[/dim]", soft_wrap=True)


register_scanner_commands(app)


if __name__ == "__main__":
    app()
