"""The scan as a Markdown report, one self-contained HTML page and a JSON file.

The report is first built as a list of neutral blocks (headings, paragraphs, lists,
tables) and then rendered twice, so the Markdown and the HTML cannot drift apart.
Every label and reason exists in English and German; "Long" and "Short" stay as they are.
The HTML loads nothing and runs nothing, like ``tradingagents.report_html``.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.scanner.deep_dive import DeepDiveResult
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
    Signal,
)
from tradingagents.scanner.validation import CAVEATS, ValidationResult

_T = {
    "title": ("Market scan", "Marktscan"),
    "universe": ("Universe", "Universum"),
    "benchmark": ("Benchmark", "Benchmark"),
    "asof": ("Data as of (newest bar used)", "Datenstand (neuester verwendeter Kurs)"),
    "requested": ("Requested date", "Angefragtes Datum"),
    "generated": ("Generated", "Erstellt"),
    "regime": ("Market regime", "Marktregime"),
    "regime_points": ("points", "Punkte"),
    "longs": ("Top Long ideas", "Top Long-Ideen"),
    "shorts": ("Top Short ideas", "Top Short-Ideen"),
    "none_long": ("No Long signal under the current thresholds.",
                  "Kein Long-Signal bei den aktuellen Schwellen."),
    "none_short": ("No Short signal under the current thresholds.",
                   "Kein Short-Signal bei den aktuellen Schwellen."),
    "rank": ("#", "#"), "ticker": ("Ticker", "Ticker"), "score": ("Score", "Score"),
    "conviction": ("Conviction", "Überzeugung"), "close": ("Close", "Schlusskurs"),
    "entry": ("Entry", "Einstieg"), "stop": ("Stop", "Stop"), "target": ("Target", "Ziel"),
    "size": ("Size %", "Größe %"), "reasons": ("Why", "Begründung"),
    "High": ("High", "Hoch"), "Medium": ("Medium", "Mittel"), "Low": ("Low", "Niedrig"),
    "plan_note": (
        "Entry is the last close; stop and target are {stop:g} and {target:g} x ATR(14) away; "
        "size is the share of capital that loses {risk:g} % if the stop is hit, capped at "
        "{cap:g} %. Horizon: about 1 month (21 trading days).",
        "Einstieg ist der letzte Schlusskurs; Stop und Ziel liegen {stop:g} bzw. {target:g} x "
        "ATR(14) entfernt; die Größe ist der Kapitalanteil, bei dem ein ausgelöster Stop "
        "{risk:g} % des Kapitals kostet, begrenzt auf {cap:g} %. Horizont: etwa 1 Monat "
        "(21 Handelstage).",
    ),
    "validation": ("Walk-forward validation", "Walk-forward-Validierung"),
    "val_intro": (
        "Signals recomputed at {n} past rebalance dates ({first} to {last}) using only bars up to "
        "each date, then measured over the next {h} trading days. {inst} instruments, "
        "universe {uni}, benchmark {bench}.",
        "Signale an {n} vergangenen Rebalancing-Terminen ({first} bis {last}), jeweils nur mit "
        "Kursen bis zu diesem Tag berechnet und danach über {h} Handelstage gemessen. "
        "{inst} Instrumente, Universum {uni}, Benchmark {bench}.",
    ),
    "val_none": (
        "Validation ran 0 rebalances: there was not enough history for a rebalance date with "
        "a following {h}-day outcome inside the requested window, so there is nothing to show.",
        "Die Validierung hatte 0 Rebalancing-Termine: für den gewählten Zeitraum gab es nicht "
        "genug Historie für einen Termin mit anschließenden {h} Handelstagen Ergebnis, daher "
        "gibt es nichts anzuzeigen."),
    "direction": ("Call", "Signal"), "calls": ("Calls", "Anzahl"),
    "avg_ret": ("Avg fwd return", "Ø Folgerendite"),
    "avg_exc": ("Avg excess vs benchmark", "Ø Überrendite vs. Benchmark"),
    "hit": ("Hit rate", "Trefferquote"), "pnl": ("Avg P&L of the position", "Ø P&L der Position"),
    "spread": ("Long minus Short spread, per date (average of the per-date spreads)",
               "Long-minus-Short-Spread je Termin (Mittel der Spreads je Termin)"),
    "spread_pooled": ("Long minus Short spread, pooled (average of all Long calls minus "
                      "average of all Short calls)",
                      "Long-minus-Short-Spread gepoolt (Mittel aller Long-Signale minus "
                      "Mittel aller Short-Signale)"),
    "tstat": ("t-statistic of the per-date spread ({n} dates)",
              "t-Statistik des Spreads je Termin ({n} Termine)"),
    "baseline": ("Equal-weight average of all calls", "Gleichgewichteter Durchschnitt aller Signale"),
    "bench_avg": ("Benchmark over the same windows", "Benchmark über dieselben Fenster"),
    "caveats": ("Caveats", "Einschränkungen"),
    "deep": ("Deep-dive ratings (LLM agents)", "Deep-Dive-Ratings (LLM-Agenten)"),
    "scanner": ("Scanner", "Scanner"), "agents": ("Agents' rating", "Rating der Agenten"),
    "failed": ("failed", "fehlgeschlagen"),
    "notes": ("Notes", "Hinweise"), "skipped": ("Skipped instruments", "Übersprungene Instrumente"),
    "method": ("Method", "Methode"),
    "method_score": (
        "Score = 100 x weighted sum of six components, each between -1 and +1: {weights}.",
        "Score = 100 x gewichtete Summe aus sechs Komponenten zwischen -1 und +1: {weights}.",
    ),
    "method_dir": (
        "Long / Short thresholds by regime (Long at or above / Short at or below): {thr}. "
        "Regime: risk-on at {on:+d} points or more, risk-off at {off:+d} or less. "
        "Conviction: High from |score| {hi:g}, Medium from {mid:g}.",
        "Long-/Short-Schwellen je Regime (Long ab / Short bis): {thr}. "
        "Regime: Risk-on ab {on:+d} Punkten, Risk-off bis {off:+d}. "
        "Überzeugung: Hoch ab |Score| {hi:g}, Mittel ab {mid:g}.",
    ),
    "comp": {
        "trend": ("trend", "Trend"), "momentum": ("momentum", "Momentum"),
        "relative_strength": ("relative strength", "relative Stärke"), "macd": ("MACD", "MACD"),
        "rsi_guard": ("RSI guard", "RSI-Filter"), "volume": ("volume", "Volumen"),
    },
    "disclaimer_h": ("Disclaimer", "Haftungsausschluss"),
    "disclaimer": (
        "This is a research tool, not financial advice. The signals are rule-based and derived "
        "from past prices only; they do not know about news, earnings or events. Short positions "
        "carry theoretically unlimited loss risk. Stops can be gapped through. Check any idea "
        "yourself before acting on it.",
        "Dies ist ein Recherchewerkzeug und keine Anlageberatung. Die Signale sind regelbasiert "
        "und nur aus vergangenen Kursen abgeleitet; Nachrichten, Quartalszahlen oder Ereignisse "
        "kennt das Verfahren nicht. Short-Positionen bergen ein theoretisch unbegrenztes "
        "Verlustrisiko. Stops können durch Kurslücken übersprungen werden. Prüfe jede Idee "
        "selbst, bevor du handelst.",
    ),
}


def _t(key: str, lang: str):
    pair = _T[key]
    return pair[1] if lang == "de" else pair[0]


def _notice(item, lang: str) -> str:
    """A Notice in ``lang``; a plain string (older callers) as it is."""
    return item.text(lang) if hasattr(item, "text") else str(item)


def format_price(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:,.2f}" if abs(x) >= 1 else f"{x:.4f}"


def _pct(x: float | None, digits: int = 1, signed: bool = True) -> str:
    if x is None:
        return "-"
    return f"{x * 100:+.{digits}f} %" if signed else f"{x * 100:.{digits}f} %"


# --- the report as neutral blocks -------------------------------------------------------------

Block = tuple  # ("h1"|"h2", text) | ("p", text) | ("ul", [text]) | ("table", header, rows, tones)


def _signal_table(signals: list[Signal], lang: str, tone: str) -> Block:
    header = [_t(k, lang) for k in ("rank", "ticker", "score", "conviction", "close", "entry",
                                    "stop", "target", "size", "reasons")]
    rows = []
    for i, s in enumerate(signals, 1):
        rows.append([
            str(i), s.symbol, f"{s.score:+.1f}", _t(s.conviction, lang), format_price(s.close),
            format_price(s.entry), format_price(s.stop), format_price(s.target),
            "-" if s.size_pct is None else f"{s.size_pct:.1f}",
            "; ".join(s.reasons_text(lang)),
        ])
    return ("table", header, rows, [tone] * len(rows))


def build_blocks(
    result: ScanResult,
    validation: ValidationResult | None = None,
    deep: DeepDiveResult | None = None,
    lang: str = "en",
) -> list[Block]:
    lang = "de" if lang == "de" else "en"
    blocks: list[Block] = [("h1", f"{_t('title', lang)}: {result.universe.upper()}")]
    blocks.append(("ul", [
        f"{_t('universe', lang)}: {result.description} ({result.universe}, "
        f"{len(result.signals)} scored)",
        f"{_t('benchmark', lang)}: {result.benchmark}",
        f"{_t('asof', lang)}: {result.as_of_date or '-'}",
        f"{_t('requested', lang)}: {result.requested_date}",
        f"{_t('generated', lang)}: {result.generated}",
    ]))

    regime = result.regime
    label = REGIME_LABELS[regime.label][lang]
    blocks.append(("h2", _t("regime", lang)))
    blocks.append(("p", f"{label} ({regime.points:+d} {_t('regime_points', lang)})"))
    blocks.append(("ul", regime.describe(lang)))

    blocks.append(("h2", _t("longs", lang)))
    if result.longs:
        blocks.append(_signal_table(result.longs, lang, "long"))
    else:
        blocks.append(("p", _t("none_long", lang)))
    blocks.append(("h2", _t("shorts", lang)))
    if result.shorts:
        blocks.append(_signal_table(result.shorts, lang, "short"))
    else:
        blocks.append(("p", _t("none_short", lang)))
    blocks.append(("p", _t("plan_note", lang).format(
        stop=STOP_ATR, target=TARGET_ATR, risk=RISK_PER_TRADE_PCT, cap=MAX_POSITION_PCT)))

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
    weights = ", ".join(f"{comp[k][1 if lang == 'de' else 0]} {int(w * 100)} %"
                        for k, w in WEIGHTS.items())
    thr = "; ".join(f"{name} {up:+g} / {down:+g}" for name, (up, down) in THRESHOLDS.items())
    blocks.append(("p", _t("method_score", lang).format(weights=weights)))
    blocks.append(("p", _t("method_dir", lang).format(
        thr=thr, on=RISK_ON_AT, off=RISK_OFF_AT, hi=CONVICTION_HIGH, mid=CONVICTION_MEDIUM)))
    blocks.append(("h2", _t("disclaimer_h", lang)))
    blocks.append(("p", _t("disclaimer", lang)))
    return blocks


def _validation_blocks(v: ValidationResult, lang: str) -> list[Block]:
    blocks: list[Block] = [("h2", _t("validation", lang))]
    if v.rebalances == 0:
        blocks.append(("p", _t("val_none", lang).format(h=v.horizon)))
    else:
        blocks.append(("p", _t("val_intro", lang).format(
            n=v.rebalances, first=v.first_rebalance, last=v.last_rebalance, h=v.horizon,
            inst=v.instruments, uni=v.universe, bench=v.benchmark)))
        header = [_t(k, lang) for k in ("direction", "calls", "avg_ret", "avg_exc", "hit", "pnl")]
        rows, tones = [], []
        for name, tone in (("Long", "long"), ("Short", "short"), ("Neutral", "")):
            s = v.stats[name]
            rows.append([name, str(s.calls), _pct(s.avg_return, 2), _pct(s.avg_excess, 2),
                         _pct(s.hit_rate, 0, signed=False), _pct(s.avg_pnl, 2)])
            tones.append(tone)
        blocks.append(("table", header, rows, tones))
        facts = [f"{_t('spread', lang)}: {_pct(v.spread, 2)}",
                 f"{_t('spread_pooled', lang)}: {_pct(v.spread_pooled, 2)}"]
        if v.spread_tstat is not None:
            facts.append(f"{_t('tstat', lang).format(n=v.spread_dates)}: {v.spread_tstat:+.2f}")
        facts.append(f"{_t('baseline', lang)}: {_pct(v.universe_avg_return, 2)}")
        facts.append(f"{_t('bench_avg', lang)}: {_pct(v.benchmark_avg_return, 2)}")
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

def _md_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(
    result: ScanResult,
    validation: ValidationResult | None = None,
    deep: DeepDiveResult | None = None,
    lang: str = "en",
) -> str:
    out: list[str] = []
    for block in build_blocks(result, validation, deep, lang):
        kind = block[0]
        if kind == "h1":
            out.append(f"# {block[1]}\n")
        elif kind == "h2":
            out.append(f"## {block[1]}\n")
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
  --long:#087a59; --short:#b4372f; color-scheme: light dark;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
@media (prefers-color-scheme: dark) {
  :root { --paper:#131a1f; --ink:#e4e8eb; --muted:#9aa6af; --rule:#263239; --wash:#1a242a;
    --long:#14c290; --short:#ef6f63; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font:16px/1.55 var(--sans); }
main { max-width: 72rem; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
h1 { font-size: 2rem; letter-spacing: -0.02em; margin: 0 0 1rem; }
h2 { font-size: 1.3rem; margin: 2.5rem 0 0.75rem; padding-top: 1rem; border-top: 2px solid var(--ink); }
ul { padding-left: 1.2rem; } li { margin: 0.2rem 0; }
.table { overflow-x: auto; margin: 0 0 1rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; font-variant-numeric: tabular-nums; }
th, td { padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; border-bottom: 1px solid var(--rule); }
th { font-weight: 600; border-bottom-color: var(--muted); }
tr.long td:first-child { border-left: 4px solid var(--long); color: var(--long); font-weight: 600; }
tr.short td:first-child { border-left: 4px solid var(--short); color: var(--short); font-weight: 600; }
.disclaimer { padding: 0.8rem 1rem; background: var(--wash); border-left: 3px solid var(--short); }
@media print { body { font-size: 11pt; } main { padding: 0; } }
"""


def render_html(
    result: ScanResult,
    validation: ValidationResult | None = None,
    deep: DeepDiveResult | None = None,
    lang: str = "en",
) -> str:
    lang = "de" if lang == "de" else "en"
    body: list[str] = []
    in_disclaimer = False
    for block in build_blocks(result, validation, deep, lang):
        kind = block[0]
        if kind == "h1":
            body.append(f"<h1>{escape(block[1])}</h1>")
        elif kind == "h2":
            in_disclaimer = block[1] == _t("disclaimer_h", lang)
            body.append(f"<h2>{escape(block[1])}</h2>")
        elif kind == "p":
            cls = ' class="disclaimer"' if in_disclaimer else ""
            body.append(f"<p{cls}>{escape(block[1])}</p>")
        elif kind == "ul":
            body.append("<ul>" + "".join(f"<li>{escape(i)}</li>" for i in block[1]) + "</ul>")
        elif kind == "table":
            _, header, rows, tones = block
            head = "".join(f"<th>{escape(h)}</th>" for h in header)
            trs = "".join(
                f'<tr class="{escape(tone)}">' + "".join(f"<td>{escape(str(c))}</td>" for c in row)
                + "</tr>" for row, tone in zip(rows, tones, strict=True))
            body.append(f'<div class="table"><table><thead><tr>{head}</tr></thead>'
                        f"<tbody>{trs}</tbody></table></div>")
    title = f"{_t('title', lang)}: {result.universe.upper()} {result.requested_date}"
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
) -> dict[str, Path]:
    """Write ``scan.md``, ``scan.html`` and ``scan.json`` under ``<results_dir>/scans/<name>/``."""
    folder = scan_directory(result, results_dir)
    folder.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    if validation is not None:
        payload["validation"] = validation.to_dict()
    if deep is not None:
        payload["deep_dive"] = {"ratings": deep.ratings, "errors": deep.errors}
    paths = {"md": folder / "scan.md", "html": folder / "scan.html", "json": folder / "scan.json"}
    paths["md"].write_text(render_markdown(result, validation, deep, lang), encoding="utf-8")
    paths["html"].write_text(render_html(result, validation, deep, lang), encoding="utf-8")
    paths["json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths
