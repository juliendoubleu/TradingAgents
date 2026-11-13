import sys

import typer

from cli.display import console
from cli.models import AnalystType, AssetType
from cli.prompts import filter_analysts_for_asset_type, parse_analysts
from cli.run import run_analysis
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


@app.command()
def scan(
    universe: str = typer.Option("us", "--universe", help="us, sectors, global, dax, crypto or all"),
    tickers: str = typer.Option(
        None, "--tickers", help="Comma-separated tickers that replace the universe's members"
    ),
    date: str = typer.Option(None, "--date", help="Scan date, YYYY-MM-DD (default: today)"),
    top: int = typer.Option(10, "--top", help="How many Long and Short ideas to list"),
    validate: bool = typer.Option(
        False, "--validate/--no-validate",
        help="Also check the signals walk-forward against what happened next",
    ),
    months: int = typer.Option(24, "--months", help="History the validation covers, in months"),
    deep: int = typer.Option(
        0, "--deep", help="Run the LLM agents on the N best Long and Short ideas (needs an LLM key)"
    ),
    save: bool = typer.Option(True, "--save/--no-save", help="Write scan.md, scan.html, scan.json"),
    lang: str = typer.Option(None, "--lang", help="Report language, en or de"),
    output: str = typer.Option(None, "--output", help="Directory to save under (default: results_dir)"),
):
    """Scan a market for Long and Short ideas, with no LLM needed.

    Ranks every instrument by a rule-based score (trend, momentum, relative strength,
    MACD, RSI, volume), reads the market regime, and gives each idea an entry, stop
    and target. A research tool, not financial advice.
    """
    from rich.table import Table

    from tradingagents.dataflows.errors import VendorError
    from tradingagents.scanner.deep_dive import check_llm_credentials, deep_dive
    from tradingagents.scanner.regime import LABELS as REGIME_LABELS
    from tradingagents.scanner.report import format_price, save_scan
    from tradingagents.scanner.scan import parse_date, run_scan
    from tradingagents.scanner.validation import CAVEATS, validate as run_validation

    names = [t.strip() for t in tickers.split(",") if t.strip()] if tickers else None
    try:
        as_of = parse_date(date)
        if tickers is not None and not names:
            raise ValueError("no ticker given; pass them comma-separated, e.g. AAPL,MSFT")
        if top < 1 or months < 1 or deep < 0:
            raise ValueError("--top and --months must be at least 1, and --deep at least 0")
        if lang is None:
            language = str(DEFAULT_CONFIG.get("output_language", "English")).strip().lower()
            lang = "de" if language in ("german", "deutsch", "de") else "en"
        if lang not in ("en", "de"):
            raise ValueError("--lang must be en or de")
        if deep:
            check_llm_credentials(DEFAULT_CONFIG)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    try:
        with console.status("Downloading prices and scanning..."):
            result = run_scan(universe, as_of, tickers=names, top=top)
        validation = None
        if validate:
            with console.status(f"Validating the signals over {months} months..."):
                validation = run_validation(universe, as_of, tickers=names, months=months)
    except ValueError as exc:  # an unknown universe
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    except VendorError as exc:
        console.print(f"[red]Could not get price data: {exc}[/red]")
        raise typer.Exit(code=1) from None

    de = lang == "de"
    console.print(f"\n[bold]{result.description}[/bold] ({result.universe}), "
                  f"{'Datenstand' if de else 'data as of'} {result.as_of_date or '-'}")
    tone = {"risk-on": "green", "risk-off": "red", "neutral": "yellow"}[result.regime.label]
    console.print(f"{'Marktregime' if de else 'Market regime'}: "
                  f"[{tone}]{REGIME_LABELS[result.regime.label][lang]}[/{tone}] "
                  f"({result.regime.points:+d})")
    for line in result.regime.describe(lang):
        console.print(f"  - {line}")
    for warning in result.warnings:
        console.print(f"[yellow]{warning}[/yellow]")

    def ideas_table(title: str, signals, colour: str) -> None:
        if not signals:
            console.print(f"\n[bold {colour}]{title}[/bold {colour}]: -")
            return
        table = Table(title=title, title_style=f"bold {colour}")
        for col in ("#", "Ticker", "Score", "Conv.", "Close", "Stop", "Target", "Size %"):
            table.add_column(col, no_wrap=True)
        for i, s in enumerate(signals, 1):
            table.add_row(str(i), s.symbol, f"{s.score:+.1f}", s.conviction, format_price(s.close),
                          format_price(s.stop), format_price(s.target), f"{s.size_pct:.1f}")
        console.print()
        console.print(table)
        for s in signals:
            console.print(f"  [bold]{s.symbol}[/bold]: {'; '.join(s.reasons_text(lang))}")

    ideas_table("Top Long" if not de else "Top Long-Ideen", result.longs, "green")
    ideas_table("Top Short" if not de else "Top Short-Ideen", result.shorts, "red")

    if validation is not None:
        console.print(f"\n[bold]{'Validierung' if de else 'Walk-forward validation'}[/bold]: "
                      f"{validation.rebalances} {'Termine' if de else 'rebalances'}, "
                      f"{validation.first_rebalance} - {validation.last_rebalance}, "
                      f"{validation.horizon}d")
        vt = Table()
        for col in ("Call", "n", "Avg ret", "Avg excess", "Hit rate", "Avg P&L"):
            vt.add_column(col)
        pct = lambda x, d=2: "-" if x is None else f"{x * 100:+.{d}f} %"  # noqa: E731
        for name in ("Long", "Short", "Neutral"):
            st = validation.stats[name]
            vt.add_row(name, str(st.calls), pct(st.avg_return), pct(st.avg_excess),
                       "-" if st.hit_rate is None else f"{st.hit_rate * 100:.0f} %", pct(st.avg_pnl))
        console.print(vt)
        console.print(f"Long - Short spread: {pct(validation.spread)}"
                      + (f" (t={validation.spread_tstat:+.2f}, {validation.spread_dates} dates)"
                         if validation.spread_tstat is not None else ""))
        console.print(f"[dim]{CAVEATS[lang][0]}[/dim]")

    deep_result = None
    if deep:
        picks = sorted([*result.longs, *result.shorts], key=lambda s: -abs(s.score))[:deep]
        console.print(f"\nRunning the LLM agents on {len(picks)} ideas: "
                      f"{', '.join(s.symbol for s in picks)} ...")
        deep_result = deep_dive(picks, result.requested_date, DEFAULT_CONFIG)
        for symbol, rating in deep_result.ratings.items():
            console.print(f"  {symbol}: {rating}")
        for symbol, error in deep_result.errors.items():
            console.print(f"  [yellow]{symbol} failed: {error}[/yellow]")

    if save:
        paths = save_scan(result, validation, deep_result, lang,
                          output or DEFAULT_CONFIG["results_dir"])
        console.print(f"\nSaved: {paths['md']}\n       {paths['html']}\n       {paths['json']}",
                      soft_wrap=True)
    console.print("[dim]Research tool, not financial advice. Shorts carry unlimited loss risk.[/dim]")


if __name__ == "__main__":
    app()
