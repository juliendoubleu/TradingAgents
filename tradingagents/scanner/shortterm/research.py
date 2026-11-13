"""The study driver: load data, simulate every candidate, evaluate, persist, report.

Run it as ``python -m tradingagents.scanner.shortterm.research --out <dir>``. It
downloads (and optionally caches) Yahoo bars, runs the pre-registered candidates and
the logged variants, writes ``validated.json`` (the snapshot ``live`` reads) and the
German report. Nothing is executed at a broker: everything is simulation.

Discipline built into the flow: ``run_study(..., include_oos=False)`` evaluates the
in-sample part only; the out-of-sample part is computed once, in the final run.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.vendors.yahoo.bulk import load_bulk_ohlcv
from tradingagents.dataflows.vendors.yahoo.intraday import load_intraday_ohlcv
from tradingagents.scanner.shortterm import (
    COSTS_BPS,
    IS_FRACTION,
    StrategySpec,
    daily_features,
    evaluate as ev,
)
from tradingagents.scanner.shortterm.daytrade import (
    DAY_SPECS,
    run_daily,
    run_intraday,
    session_days,
)
from tradingagents.scanner.shortterm.swing import EXIT_FLAGS, SWING_SPECS, daily_signal
from tradingagents.scanner.universe import resolve_universe

logger = logging.getLogger(__name__)

EXTRA_SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
DAILY_YEARS = 10
DAILY_WARMUP_DAYS = 430          # calendar days before the window: SMA200 + 252-day z-score
SNAPSHOT_NAME = "validated.json"

PRE_REGISTERED: list[StrategySpec] = DAY_SPECS + SWING_SPECS
# Variants tried after the first in-sample results (at most 8; every one is logged,
# the failures included). Filled in below, each with its one-sentence motivation.
VARIANTS: list[StrategySpec] = [
    StrategySpec("S4s", "S4", "swing", "1d", "Post-gap drift, short side only",
                 "Drift nach Kurslücke, nur Short",
                 {"gap": 1.5, "vol": 2.0, "hold": 5, "stop_atr": 2.0, "sides": "short"},
                 "In-sample, all of S4's per-trade edge sat on the short side (heavy-volume "
                 "gap downs keep falling).",
                 note_de="Im IS lag der gesamte Vorteil von S4 auf der Short-Seite (Kurslücken nach unten mit hohem Volumen laufen weiter)."),
    StrategySpec("S2s", "S2", "swing", "1d", "5-day reversal, short side only",
                 "5-Tage-Umkehr, nur Short",
                 {"z": 2.0, "hold": 5, "stop_atr": 2.5, "sides": "short"},
                 "In-sample, S2 earned on the short side and lost on the long side.",
                 note_de="Im IS verdiente S2 auf der Short-Seite und verlor auf der Long-Seite."),
    StrategySpec("S3l", "S3", "swing", "1d", "Volume breakout, long side only",
                 "Ausbruch mit Volumen, nur Long",
                 {"vol": 1.5, "hold": 10, "stop_atr": 2.0, "sides": "long"},
                 "In-sample, breakouts paid while breakdown shorts were squeezed.",
                 note_de="Im IS zahlten sich Ausbrüche aus, während Short-Positionen bei Ausbrüchen nach unten ausgequetscht wurden."),
    StrategySpec("S1x", "S1", "swing", "1d", "RSI(2) pullback, extreme threshold",
                 "RSI(2)-Rücksetzer, extremere Schwelle",
                 {"lo": 5, "hi": 95, "hold": 5, "stop_atr": 2.5},
                 "A deeper pullback should bounce further relative to the fixed round-trip cost.",
                 note_de="Ein tieferer Rücksetzer sollte im Verhältnis zu den festen Kosten weiter zurückspringen."),
    StrategySpec("S1v", "S1", "swing", "1d", "RSI(2) pullback, volatile names only",
                 "RSI(2)-Rücksetzer, nur volatile Titel",
                 {"lo": 10, "hi": 90, "hold": 5, "stop_atr": 2.5, "atr_pct_min": 0.02},
                 "Costs are fixed in bps while the bounce scales with volatility, so trade "
                 "only when ATR% >= 2 %.",
                 note_de="Die Kosten sind in bp fest, der Rücksprung wächst mit der Schwankung: nur Titel mit ATR% ≥ 2 %."),
    StrategySpec("D5v", "D5", "day", "1d", "RSI(2) extreme day trade, volatile names only",
                 "RSI(2)-Extrem Daytrade, nur volatile Titel",
                 {"lo": 5, "hi": 95, "stop_atr": 1.0, "hold": 1, "atr_pct_min": 0.02},
                 "D5 was gross-positive but about equal to its 5 bps cost; larger daily "
                 "ranges (ATR% >= 2 %) dilute the cost.",
                 note_de="D5 war brutto positiv, aber etwa so groß wie die 5 bp Kosten; größere Tagesspannen (ATR% ≥ 2 %) verdünnen die Kosten."),
    StrategySpec("H3x", "H3", "day", "60m", "Morning fade, only extreme first hours",
                 "Morgen-Fade, nur extreme erste Stunde",
                 {"mult": 2.0},
                 "A 1-ATR first hour is too common to fade; require 2 ATR%.",
                 note_de="Eine erste Stunde von 1 ATR ist zu häufig zum Faden; verlangt werden 2 ATR%."),
]


def all_specs() -> list[StrategySpec]:
    return PRE_REGISTERED + VARIANTS


# --------------------------------------------------------------------------- data


@dataclass
class Data:
    as_of: pd.Timestamp
    eval_start: pd.Timestamp
    daily: dict[str, pd.DataFrame]
    feats: dict[str, pd.DataFrame]
    intraday: dict[str, dict[str, pd.DataFrame]]            # "60m"/"5m" -> symbol -> frame
    days: dict[str, list] = field(default_factory=dict)      # timeframe -> trading days
    panels: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = field(default_factory=dict)
    inventory: dict[str, dict] = field(default_factory=dict)


def research_symbols() -> list[str]:
    _, _, members = resolve_universe("us")
    return list(dict.fromkeys([*members, *EXTRA_SYMBOLS]))


def _cached(cache_dir: Path | None, name: str, loader):
    """Load through a pickle cache so iterating on the rules never re-downloads."""
    if cache_dir is not None:
        path = Path(cache_dir) / name
        if path.exists():
            return pickle.loads(path.read_bytes())
    value = loader()
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / name).write_bytes(pickle.dumps(value))
    return value


def _cut(frames: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Drop every bar after the end of ``as_of`` (the study date): no look-ahead."""
    limit = as_of + pd.Timedelta(days=1)
    out = {}
    for sym, df in frames.items():
        idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
        kept = df[idx < limit]
        if not kept.empty:
            out[sym] = kept
    return out


def _panel(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.concat({s: f["Open"] for s, f in frames.items()}, axis=1),
            pd.concat({s: f["Close"] for s, f in frames.items()}, axis=1))


def load_data(as_of_date: str, cache_dir: Path | None = None) -> Data:
    as_of = pd.Timestamp(as_of_date).normalize()
    symbols = research_symbols()
    eval_start = as_of - pd.DateOffset(years=DAILY_YEARS)
    lookback = (as_of - eval_start).days + DAILY_WARMUP_DAYS
    daily = _cached(cache_dir, f"daily_{as_of.date()}.pkl",
                    lambda: load_bulk_ohlcv(symbols, as_of.strftime("%Y-%m-%d"), lookback))
    intraday = {
        "60m": _cached(cache_dir, f"i60m_{as_of.date()}.pkl",
                       lambda: load_intraday_ohlcv(symbols, "60m", 729)),
        "5m": _cached(cache_dir, f"i5m_{as_of.date()}.pkl",
                      lambda: load_intraday_ohlcv(symbols, "5m", 59)),
    }
    daily = _cut(daily, as_of)
    intraday = {k: _cut(v, as_of) for k, v in intraday.items()}
    data = Data(as_of=as_of, eval_start=eval_start, daily=daily,
                feats={s: daily_features(df) for s, df in daily.items()}, intraday=intraday)
    # trading days of each dataset (the unit of the IS/OOS split)
    day_set = sorted({d for f in data.feats.values() for d in f.index if d >= eval_start})
    data.days["1d"] = day_set
    data.panels["1d"] = _panel(daily)
    for tf, frames in intraday.items():
        seen, used = set(), set()
        n_sessions = n_full = 0
        for df in frames.values():
            all_dates = pd.DatetimeIndex(df.index.tz_localize(None)).normalize()
            n_sessions += len(set(all_dates))
            for day, _ in session_days(df, tf):
                used.add(day)
                n_full += 1
            seen |= set(all_dates)
        data.days[tf] = sorted(used)
        data.panels[tf] = _panel(frames)
        data.inventory[tf] = {
            "symbols": len(frames), "calendar_days": len(seen), "complete_days": len(used),
            "symbol_sessions": n_sessions, "complete_symbol_sessions": n_full,
            "first_day": str(min(used).date()) if used else None,
            "last_day": str(max(used).date()) if used else None,
        }
    data.inventory["1d"] = {
        "symbols": len(daily), "first_day": str(day_set[0].date()), "last_day": str(day_set[-1].date()),
        "calendar_days": len(day_set),
    }
    return data


# --------------------------------------------------------------------------- simulation


def build_trades(spec: StrategySpec, data: Data) -> pd.DataFrame:
    """All simulated trades of one candidate, with the universe drift attached."""
    if spec.timeframe == "1d":
        records = run_daily(
            spec, data.feats, lambda f, p, fam=spec.family: daily_signal(fam, f, p),
            first_day=data.eval_start, exit_flags=EXIT_FLAGS.get(spec.family),
        )
    else:
        records = run_intraday(spec, data.intraday[spec.timeframe], data.feats)
    columns = ["symbol", "date", "entry_ts", "exit_ts", "direction", "raw", "window", "reason"]
    trades = pd.DataFrame(records, columns=columns)
    opens, closes = data.panels[spec.timeframe]
    return ev.attach_universe(trades, opens, closes)


def drift_reference(data: Data) -> dict:
    """What simply holding the whole universe earned, per dataset and segment.

    Open-to-close of every symbol-day and the 5-day hold (open of day+1 to close of
    day+5), equal-weight, as a yardstick for the long-only rules.
    """
    opens, closes = data.panels["1d"]
    days = pd.DatetimeIndex(data.days["1d"])
    oc = (closes / opens - 1.0).mean(axis=1).reindex(days)
    five = (closes.shift(-4) / opens - 1.0).mean(axis=1).reindex(days)
    out = {}
    for tf, tf_days in data.days.items():
        is_days, oos_days = ev.split_days(tf_days)
        seg = {}
        for name, ds in (("IS", is_days), ("OOS", oos_days)):
            idx = pd.DatetimeIndex(ds)
            seg[name] = {
                "open_close_bps": float(oc.reindex(idx).mean() * 1e4),
                "open_close_t": ev.t_stat(oc.reindex(idx).dropna()),
                "hold5_bps": float(five.reindex(idx).mean() * 1e4),
                "hold5_t": ev.t_stat(five.reindex(idx).dropna()),
            }
        out[tf] = seg
    return out


@dataclass
class Study:
    as_of: str
    specs: list[StrategySpec]
    results: dict[str, dict]
    variants_tried: int
    data_windows: dict
    splits: dict
    drift: dict
    include_oos: bool
    n_trades_total: int = 0


def run_study(as_of_date: str, cache_dir: str | Path | None = None, *,
              include_oos: bool = True, specs: list[StrategySpec] | None = None,
              data: Data | None = None) -> Study:
    """Run every candidate. ``include_oos=False`` evaluates in-sample only."""
    cache = Path(cache_dir) if cache_dir else None
    data = data or load_data(as_of_date, cache)
    specs = specs or all_specs()
    splits = {tf: ev.split_days(days) for tf, days in data.days.items()}
    results: dict[str, dict] = {}
    total = 0
    for spec in specs:
        trades = build_trades(spec, data)
        total += len(trades)
        is_days, oos_days = splits[spec.timeframe]
        res = ev.evaluate_strategy(trades, is_days, oos_days, COSTS_BPS[spec.horizon],
                                   include_oos=include_oos)
        res["exit_reasons"] = trades["reason"].value_counts().to_dict() if len(trades) else {}
        results[spec.id] = res
    window = {tf: {"first_day": str(d[0].date()), "last_day": str(d[-1].date()), "n_days": len(d)}
              for tf, d in data.days.items() if d}
    split_info = {tf: {"is_first": str(i[0].date()), "is_last": str(i[-1].date()),
                       "oos_first": str(o[0].date()), "oos_last": str(o[-1].date()),
                       "n_is_days": len(i), "n_oos_days": len(o)}
                  for tf, (i, o) in splits.items() if i and o}
    return Study(as_of=str(data.as_of.date()), specs=specs, results=results,
                 variants_tried=len(PRE_REGISTERED) + len(VARIANTS),
                 data_windows={"windows": window, "inventory": data.inventory},
                 splits=split_info, drift=drift_reference(data), include_oos=include_oos,
                 n_trades_total=total)


# --------------------------------------------------------------------------- persistence


def _clean(value):
    """JSON-safe copy: NaN/inf become None, numpy scalars become Python numbers."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def snapshot_dict(study: Study) -> dict:
    strategies = []
    for spec in study.specs:
        res = study.results[spec.id]
        strategies.append({
            "id": spec.id, "family": spec.family, "horizon": spec.horizon,
            "timeframe": spec.timeframe, "name_en": spec.name_en, "name_de": spec.name_de,
            "params": spec.params, "note": spec.note,
            "verdict": res.get("verdict"), "strong": res.get("strong"),
            "beats_drift": res.get("beats_drift"),
            "IS": res["IS"], "OOS": res.get("OOS"), "ALL": res.get("ALL"),
            "cost_sensitivity": res["cost_sensitivity"], "baselines": res["baselines"],
        })
    return _clean({
        "study_date": study.as_of,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "is_fraction": IS_FRACTION,
        "data": study.data_windows, "splits": study.splits,
        "cost_assumptions_bps": COSTS_BPS,
        "variants_tried": study.variants_tried,
        "drift_reference": study.drift,
        "simulation_only": True,
        "strategies": strategies,
    })


def write_snapshot(study: Study, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot_dict(study), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


# --------------------------------------------------------------------------- report (German)


def _n(x, nd=1) -> str:
    """German number format (decimal comma); '-' for missing."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(x) -> str:
    return "–" if x is None else f"{_n(100 * x, 1)} %"


VERDICT_DE = {"validated": "belegt", "experimental": "experimentell", "rejected": "verworfen"}
TF_DE = {"1d": "Tagesbalken (10 J.)", "60m": "60-Min. (≈2 J.)", "5m": "5-Min. (≈2 Mon.)"}


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _assessment(study: Study) -> list[str]:
    res = study.results
    ids = [s.id for s in study.specs]
    verdicts = {i: res[i].get("verdict") for i in ids}
    validated = [i for i in ids if verdicts[i] == "validated"]
    experimental = [i for i in ids if verdicts[i] == "experimental"]
    is_hits = [i for i in ids if res[i]["IS"]["t_stat"] >= 2.0]
    oos_pos = [i for i in ids if (res[i]["OOS"]["mean_net_bps"] or 0) > 0]
    skill = [i for i in ids if res[i]["OOS"]["excess_t_stat"] >= 2.0]
    n = study.variants_tried
    out = []
    if validated:
        out.append(
            f"Nach den vorab festgelegten Regeln gelten {len(validated)} von {n} getesteten "
            f"Varianten als belegt: {', '.join(validated)}. Das heißt: im Zeitraum vor UND nach "
            "der Trennlinie ein positiver, statistisch erkennbarer Netto-Ertrag nach Kosten.")
    else:
        out.append(
            f"Keine der {n} getesteten Varianten erfüllt die vorab festgelegten Kriterien für "
            "„belegt“. Nach Kosten ist in diesen Daten kein verlässlicher Vorteil nachgewiesen.")
    if experimental:
        out.append(f"Als „experimentell“ (Hinweis, kein Beleg) gelten: {', '.join(experimental)}.")
    out.append(
        f"{len(is_hits)} Variante(n) erreichten im Lernzeitraum (IS) einen t-Wert von mindestens 2 "
        f"({', '.join(is_hits) or 'keine'}). Bei {n} Versuchen sind einzelne t-Werte um 2 allein "
        "durch Zufall zu erwarten. Maßgeblich ist der unberührte Testzeitraum (OOS): dort hatten "
        f"{len(oos_pos)} Variante(n) einen positiven Netto-Durchschnitt "
        f"({', '.join(oos_pos) or 'keine'}).")
    for i in validated + experimental:
        o, b = res[i]["OOS"], res[i]["baselines"]["OOS"]
        if not res[i]["beats_drift"]:
            out.append(
                f"**Wichtig zu {i}:** Das Urteil „{VERDICT_DE[verdicts[i]]}“ besagt nur, dass der "
                f"Netto-Ertrag nach Kosten positiv war. Gegenüber dem Durchschnitt des Universums "
                f"im gleichen Zeitfenster beträgt der Excess im OOS {_n(o['mean_excess_bps'])} bp "
                f"(t = {_n(o['excess_t_stat'], 2)}), und die Basislinie „immer Long“ über dieselben "
                f"Fenster erzielt {_n(b['always_long_net_bps'])} bp gegenüber "
                f"{_n(b['strategy_dir_net_bps'])} bp bei der eigenen Richtung. Der Ertrag stammt "
                "also aus der Marktdrift (ein steigender Aktienmarkt, plus Survivorship-Bias der "
                "heutigen Large Caps), nicht aus einem Können der Regel. Wer Long-Aktien hält, "
                "verdient dasselbe, ohne die Regel zu brauchen.")
    if skill:
        out.append(
            "Über den Marktdurchschnitt hinaus (Excess, t ≥ 2 im OOS) zeigten sich Hinweise bei: "
            f"{', '.join(skill)}.")
    else:
        out.append(
            "Über die allgemeine Marktdrift hinaus (Excess gegenüber dem Durchschnitt des "
            "Universums) ließ sich im OOS bei keiner Variante ein Können mit t ≥ 2 belegen. "
            "Long-Regeln verdienen in einem steigenden Markt ohnehin etwas; das ist keine Fertigkeit.")
    return out


def write_report(study: Study, path: str | Path) -> Path:
    """The German research report for a non-technical reader."""
    if not study.include_oos:
        raise ValueError("the report needs the out-of-sample results (include_oos=True)")
    res, specs = study.results, study.specs
    inv = study.data_windows["inventory"]
    L: list[str] = []
    L.append("# Kurzfrist-Handel: Simulation und ehrliche Bewertung")
    L.append(f"\nStudienstand: {study.as_of}. Alles ist eine **Simulation auf historischen "
             "Yahoo-Kursen**. Es wird nichts bei einem Broker ausgeführt, und eine Simulation ist "
             "keine echte Ausführung.\n")
    L.append("## 1. Was getestet wurde")
    L.append(
        f"- **Universum:** {inv['1d']['symbols']} liquide US-Werte (Large Caps aller Sektoren) "
        "plus die ETFs SPY, QQQ, IWM, DIA.\n"
        f"- **Tagesbalken:** {study.splits['1d']['is_first']} bis {study.splits['1d']['oos_last']} "
        f"(≈10 Jahre, {study.splits['1d']['n_is_days'] + study.splits['1d']['n_oos_days']} Handelstage).\n"
        f"- **60-Minuten-Balken:** {study.splits['60m']['is_first']} bis "
        f"{study.splits['60m']['oos_last']} (Yahoo liefert höchstens ≈2 Jahre; "
        f"{inv['60m']['complete_days']} vollständige Handelstage).\n"
        f"- **5-Minuten-Balken:** {study.splits['5m']['is_first']} bis "
        f"{study.splits['5m']['oos_last']} (Yahoo liefert höchstens 60 Tage; nur "
        f"{inv['5m']['complete_days']} Handelstage, **kleine Stichprobe**, nur ein Realismus-Check).\n"
        f"- **Kosten (Hin- und Rückweg):** {_n(COSTS_BPS['day'], 0)} Basispunkte (bp) für Daytrades, "
        f"{_n(COSTS_BPS['swing'], 0)} bp für Swing-Trades; zusätzlich 2× und 4× als Empfindlichkeit. "
        "Für deutsche Privatanleger (Ordergebühr, größere Spreads bei Tradegate/L&S, CFDs oder "
        "Knock-outs für Short-Positionen) sind eher 10–20 bp realistisch. 1 bp = 0,01 %.\n"
        "- **Ausführung streng:** Ein Signal aus dem Schlusskurs eines Balkens wird erst zur "
        "Eröffnung des nächsten Balkens gehandelt. Stop und Ziel im selben Balken: der Stop gilt "
        "zuerst. Eröffnet ein Balken jenseits des Stops, wird zum schlechteren Eröffnungskurs "
        "ausgestoppt.\n"
        "- **Lern- und Testzeitraum:** Die ersten 70 % der Handelstage jedes Datensatzes sind der "
        "Lernzeitraum (IS), die letzten 30 % der Testzeitraum (OOS). Parameter wurden nur im IS "
        "gewählt; der OOS wurde je Kandidat einmal ausgewertet und wird hier unverändert berichtet.\n"
        "- **Signifikanz:** Trades desselben Tages sind stark verwandt. Der t-Wert wird deshalb "
        "über Tagesmittelwerte gerechnet; das 95 %-Intervall zieht ganze Tage neu (2000 Ziehungen). "
        "Als grobe Regel: |t| unter 2 ist von Zufall nicht zu unterscheiden.")
    L.append(f"\n**Anzahl getesteter Varianten insgesamt: {study.variants_tried}** "
             f"({len(PRE_REGISTERED)} vorab festgelegte, {len(VARIANTS)} nachträgliche).\n")

    L.append("## 2. Ergebnisse je Strategie (netto nach Kosten)")
    L.append("Ø bp = durchschnittlicher Netto-Ertrag je Trade in Basispunkten; t = Tages-geclusterter "
             "t-Wert (gehört zum Durchschnitt der Tageserträge; der Trade-Durchschnitt kann an "
             "Tagen mit vielen Trades davon abweichen, auch im Vorzeichen); Intervall = 95 %-Intervall des durchschnittlichen **Tagesertrags** (bp) im OOS.\n")
    rows = []
    for s in specs:
        r = res[s.id]
        i, o = r["IS"], r["OOS"]
        rows.append([
            s.id, s.name_de, TF_DE[s.timeframe],
            str(i["n_trades"]), _pct(i["win_rate"]), _n(i["mean_net_bps"]), _n(i["t_stat"], 2),
            str(o["n_trades"]), _pct(o["win_rate"]), _n(o["mean_net_bps"]), _n(o["t_stat"], 2),
            f"[{_n(o['ci_low_bps'])}; {_n(o['ci_high_bps'])}]",
            ("**" if r["verdict"] != "rejected" else "") + VERDICT_DE[r["verdict"]]
            + (" (stark)" if r["strong"] else "") + ("**" if r["verdict"] != "rejected" else ""),
        ])
    L.append(_table(["ID", "Strategie", "Daten", "IS Trades", "IS Treffer", "IS Ø bp", "IS t",
                     "OOS Trades", "OOS Treffer", "OOS Ø bp", "OOS t", "OOS 95 %-Intervall",
                     "Urteil"], rows))
    L.append("\nUrteilsregeln (vorab festgelegt): *belegt* = IS-t ≥ 2 und OOS-t ≥ 2 (beide positiv), "
             "OOS-Durchschnitt auch bei doppelten Kosten > 0, mindestens 100 OOS-Trades an ≥ 20 "
             "OOS-Tagen. *experimentell* = IS-t ≥ 2, OOS-Durchschnitt > 0, OOS-t ≥ 1, ≥ 50 OOS-Trades. "
             "Alles andere *verworfen*. „stark“ nur bei gepooltem t ≥ 3. Die 5-Minuten-Strategien haben "
             f"nur {study.splits['5m']['n_oos_days']} OOS-Tage und können die Regel für „belegt“ konstruktionsbedingt nicht erfüllen.")

    L.append("\n### Brutto, Excess und Long/Short (OOS)")
    L.append("Excess = Ertrag abzüglich des Durchschnitts des gesamten Universums im gleichen "
             "Zeitfenster (Short: umgekehrt). Nur der Excess ist Können; der Rest ist Marktdrift.\n")
    rows = []
    for s in specs:
        o = res[s.id]["OOS"]
        rows.append([s.id, _n(o["mean_gross_bps"]), _n(o["median_net_bps"]),
                     _n(o["profit_factor"], 2), _n(o["mean_excess_bps"]), _n(o["excess_t_stat"], 2),
                     f"{o['long']['n_trades']} / {_n(o['long']['mean_net_bps'])}",
                     f"{o['short']['n_trades']} / {_n(o['short']['mean_net_bps'])}",
                     _n(o["sharpe"], 2), _n(o["max_drawdown_pct"], 1) + " %"])
    L.append(_table(["ID", "Ø brutto bp", "Median netto bp", "Profit-Faktor", "Ø Excess bp",
                     "Excess t", "Long: Trades / Ø bp", "Short: Trades / Ø bp", "Sharpe (Tage)",
                     "Max. Drawdown"], rows))

    L.append("\n## 3. Vergleich mit Baselines (OOS)")
    L.append("Gleiche Trades und Zeitfenster, aber ohne Stop: **immer Long**, die **eigenen "
             "Richtungen** der Strategie und **Zufallsrichtung** (200 Wiederholungen, Ø und "
             "95 %-Band). Liegt die eigene Richtung nicht über dem Zufallsband, steckt in der "
             "Richtungswahl kein Können.\n")
    rows = []
    for s in specs:
        b = res[s.id]["baselines"]["OOS"]
        if not b.get("n_trades"):
            continue
        rows.append([s.id, _n(b["always_long_net_bps"]), _n(b["strategy_dir_net_bps"]),
                     _n(b["random_mean_net_bps"]),
                     f"[{_n(b['random_lo_bps'])}; {_n(b['random_hi_bps'])}]",
                     _n(b["universe_mean_bps"])])
    L.append(_table(["ID", "Immer Long Ø bp", "Eigene Richtung Ø bp", "Zufall Ø bp",
                     "Zufall 95 %-Band", "Universum Ø bp"], rows))
    d = study.drift["1d"]
    L.append(f"\nZum Vergleich die reine Marktdrift (gleichgewichtetes Universum, Tagesbalken): "
             f"Eröffnung bis Schluss {_n(d['IS']['open_close_bps'])} bp "
             f"(IS) / {_n(d['OOS']['open_close_bps'])} bp (OOS) je Tag; 5-Tage-Halten "
             f"{_n(d['IS']['hold5_bps'])} bp (IS) / {_n(d['OOS']['hold5_bps'])} bp (OOS). "
             "Der Aktienmarkt stieg im Testzeitraum; jede Long-Regel profitiert davon.")

    L.append("\n## 4. Kostenempfindlichkeit (Ø Netto bp je Trade, OOS)")
    rows = []
    for s in specs:
        cs = res[s.id]["cost_sensitivity"]
        rows.append([s.id] + [f"{_n(cs[k]['OOS']['mean_net_bps'])} (t {_n(cs[k]['OOS']['t_stat'], 2)})"
                              for k in ("1x", "2x", "4x")])
    L.append(_table(["ID", "Basiskosten", "2× Kosten", "4× Kosten"], rows))

    L.append("\n## 5. Was verbessert wurde (und was nicht)")
    L.append(f"Nach den ersten IS-Ergebnissen wurden {len(VARIANTS)} zusätzliche Varianten "
             "ausprobiert (erlaubt waren höchstens 8), jede mit einer Begründung, **alle "
             "protokolliert, auch die Fehlschläge**, und nur im IS gewählt. Danach wurde der OOS "
             "einmal für alle ausgewertet; es wurde nicht zurückgegangen und nachjustiert.\n")
    rows = []
    for s in specs:
        if s.id in {v.id for v in VARIANTS}:
            r = res[s.id]
            rows.append([s.id, s.name_de, s.note_de, _n(r["IS"]["mean_net_bps"]),
                         _n(r["IS"]["t_stat"], 2), _n(r["OOS"]["mean_net_bps"]),
                         _n(r["OOS"]["t_stat"], 2), VERDICT_DE[r["verdict"]]])
    L.append(_table(["ID", "Variante", "Begründung (nur IS-Evidenz)", "IS Ø bp", "IS t",
                     "OOS Ø bp", "OOS t", "Urteil"], rows))

    L.append("\n## 6. Ehrliche Gesamteinschätzung")
    for para in _assessment(study):
        L.append(f"\n{para}")

    L.append("\n## 7. Grenzen")
    L.append(
        "- **Survivorship-Bias:** Das Universum sind die heutigen Large Caps, rückwirkend "
        "angewandt; Verlierer von gestern fehlen. Das begünstigt vor allem Long-Regeln.\n"
        "- **Yahoo-Datenqualität intraday:** fehlende Balken, Halbtage und gelegentliche "
        "Ausreißer; unvollständige Handelstage wurden ausgelassen "
        f"(60 Min.: {inv['60m']['complete_symbol_sessions']} von {inv['60m']['symbol_sessions']} "
        f"Symbol-Tagen vollständig; 5 Min.: {inv['5m']['complete_symbol_sessions']} von "
        f"{inv['5m']['symbol_sessions']}).\n"
        "- **Keine Leerverkaufskosten:** Leihgebühren, Verfügbarkeit und Short-Squeezes sind nicht "
        "modelliert; Short-Ergebnisse sind deshalb eher zu gut als zu schlecht.\n"
        "- **Kleine 5-Minuten-Stichprobe:** nur etwa 43 Handelstage; Ergebnisse sind Anekdoten, "
        "kein Beweis.\n"
        "- **Gap-Strategien auf Tagesbalken** (D1/D2) setzen voraus, dass man zum "
        "Eröffnungskurs handeln kann, den man erst kennt, wenn er entsteht; M2a/M2b prüfen das "
        "realistischer.\n"
        "- **Swing-Trades überlappen:** Tagescluster berücksichtigen Überlappung mehrtägiger "
        "Positionen nur teilweise; t-Werte der Swing-Strategien sind eher zu optimistisch. "
        "S1 verlässt die Position zum Schlusskurs, den man erst am Ende des Tages kennt.\n"
        "- **Mehrfachtests:** Bei vielen Varianten entstehen Treffer zufällig. Der OOS ist nur "
        "ein einziger Zeitraum und kann selbst Zufall sein.\n"
        "- **Simulation ≠ echte Ausführung:** Teilausführungen, Slippage in schnellen Märkten, "
        "Steuern und Verhalten sind nicht abgebildet.")

    L.append("\n## 8. Was daraus für die täglichen Empfehlungen folgt")
    validated = [s for s in specs if res[s.id]["verdict"] == "validated"]
    experimental = [s for s in specs if res[s.id]["verdict"] == "experimental"]
    if validated:
        L.append("Nur belegte Strategien (" + ", ".join(s.id for s in validated) + ") erzeugen "
                 "gekennzeichnete Kurzfrist-Ideen (Horizont Tag/Swing, Einstieg zur nächsten "
                 "Eröffnung, Stop, Haltedauer). Sie bleiben Simulationsergebnisse mit "
                 "Unsicherheit, keine Anlageberatung.")
    else:
        L.append("Da keine Strategie belegt ist, erzeugt die tägliche Liste **keine** "
                 "Kurzfrist-Ideen. „Kein Vorteil gefunden“ ist ein gültiges Ergebnis; "
                 "Anlagehorizonte der übrigen Empfehlungen bleiben die der Analyse.")
    if experimental:
        L.append("Experimentelle Strategien (" + ", ".join(s.id for s in experimental) + ") "
                 "können auf Wunsch gesondert als „experimentell“ markiert angezeigt werden.")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tradingagents.scanner.shortterm.research",
        description="Run the short-horizon study (simulation only) and write the report.")
    parser.add_argument("--out", required=True, help="directory for validated.json and the report")
    parser.add_argument("--as-of", default=None, help="study date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--cache", default=None, help="directory caching the Yahoo downloads")
    parser.add_argument("--update-package", action="store_true",
                        help="also overwrite the packaged validated.json that live.py reads")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    as_of = args.as_of or str(
        (pd.Timestamp.now(tz="America/New_York").normalize() - pd.Timedelta(days=1)).date())
    study = run_study(as_of, args.cache, include_oos=True)
    out = Path(args.out)
    write_snapshot(study, out / SNAPSHOT_NAME)
    report = write_report(study, out / "BERICHT_Kurzfrist.md")
    if args.update_package:
        from tradingagents.scanner.shortterm.live import SNAPSHOT_PATH
        write_snapshot(study, SNAPSHOT_PATH)
    print(f"snapshot: {out / SNAPSHOT_NAME}\nreport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
