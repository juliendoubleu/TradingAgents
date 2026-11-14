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
                 "In-sample, S4's short side had by far the higher per-trade mean (+123 bps "
                 "vs +19 bps long), so the short side alone was tried.",
                 note_de="Im IS hatte die Short-Seite von S4 den deutlich höheren Trade-Durchschnitt (+123 bp gegenüber +19 bp Long); deshalb wurde die Short-Seite allein probiert."),
    StrategySpec("S2s", "S2", "swing", "1d", "5-day reversal, short side only",
                 "5-Tage-Umkehr, nur Short",
                 {"z": 2.0, "hold": 5, "stop_atr": 2.5, "sides": "short"},
                 "In-sample, S2's short side had a positive per-trade mean (+38 bps; its "
                 "daily-mean t was -0.45) while its long side was negative.",
                 note_de="Im IS war der Trade-Durchschnitt der Short-Seite von S2 positiv (+38 bp; der t-Wert der Tagesmittel lag bei −0,45), der der Long-Seite negativ."),
    StrategySpec("S3l", "S3", "swing", "1d", "Volume breakout, long side only",
                 "Ausbruch mit Volumen, nur Long",
                 {"vol": 1.5, "hold": 10, "stop_atr": 2.0, "sides": "long"},
                 "In-sample, the long side's gross result was positive while breakdown shorts "
                 "lost; chosen on that gross result although its in-sample excess over the "
                 "market was already -0.4 bps (t -0.09).",
                 note_de="Im IS war das Brutto-Ergebnis der Long-Seite positiv, Short-Positionen bei Ausbrüchen nach unten verloren; gewählt wurde auf dieses Brutto-Ergebnis, obwohl der IS-Excess gegenüber dem Markt schon damals bei −0,4 bp lag (t = −0,09)."),
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
                       lambda: load_intraday_ohlcv(symbols, "60m", 729, end=as_of)),
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
    if spec.timeframe == "1d":
        columns += ["path_days", "path_ret"]      # the position's P&L day by day (swing overlap)
    trades = pd.DataFrame(records, columns=columns)
    opens, closes = data.panels[spec.timeframe]
    trades = ev.attach_universe(trades, opens, closes)
    if spec.timeframe == "1d":
        trades = ev.attach_universe_paths(trades, opens, closes)
    return trades


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
            "beats_drift": res.get("beats_drift"), "label": res.get("label"),
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
        # estimand: every decision (verdict clauses, beats_drift, label) reads the judged
        # series (day trades: entry-day mean; swing: calendar-time portfolio), see evaluate
        "corrected_after_review": True,
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
LABEL_DE = {"validated": "belegt", "experimental": "experimentell",
            "beta": "formal bestanden, aber nur Marktrendite", "rejected": "verworfen"}
TF_DE = {"1d": "Tagesbalken (10 J.)", "60m": "60-Min. (≈2 J.)", "5m": "5-Min. (≈2 Mon.)"}
DAILY_HEADER = "Ø je Handelstag (Basis des t-Werts)"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _label(res: dict) -> str:
    return res.get("label") or ev.label_of(res.get("verdict"), res.get("beats_drift"))


def _assessment(study: Study) -> list[str]:
    res = study.results
    ids = [s.id for s in study.specs]
    labels = {i: _label(res[i]) for i in ids}
    validated = [i for i in ids if labels[i] == "validated"]
    experimental = [i for i in ids if labels[i] == "experimental"]
    beta = [i for i in ids if labels[i] == "beta"]
    is_hits = [i for i in ids if res[i]["IS"]["t_stat"] >= 2.0]
    oos_pos = [i for i in ids if (res[i]["OOS"]["mean_daily_net_bps"] or 0) > 0]
    skill = [i for i in ids if res[i]["beats_drift"]]
    negative = [i for i in ids
                if res[i]["IS"]["t_stat"] <= -2.0 and res[i]["OOS"]["t_stat"] <= -2.0]
    n = study.variants_tried
    out = []
    if validated:
        out.append(
            f"Nach den vorab festgelegten Regeln UND gegenüber der Marktdrift gelten {len(validated)} "
            f"von {n} getesteten Varianten als belegt: {', '.join(validated)}.")
    else:
        out.append(
            f"Keine der {n} getesteten Varianten ist belegt: Keine erfüllt die vorab festgelegten "
            "Kriterien für „belegt“ UND schlägt zugleich die Marktdrift. Nach Kosten ist in diesen "
            "Daten kein verlässlicher Vorteil nachgewiesen.")
    if experimental:
        out.append(f"Als „experimentell“ (Hinweis, kein Beleg; Kriterien erfüllt und Marktdrift "
                   f"geschlagen) gelten: {', '.join(experimental)}.")
    for i in beta:
        o, b = res[i]["OOS"], res[i]["baselines"]["OOS"]
        out.append(
            f"**Wichtig zu {i}:** Die vorab festgelegten Kriterien für „{VERDICT_DE[res[i]['verdict']]}“ "
            f"sind formal bestanden (der Netto-Ertrag nach Kosten war positiv), aber der Ertrag ist nur "
            f"die Marktrendite. Gegenüber dem Durchschnitt des Universums im gleichen Zeitfenster "
            f"beträgt der Excess im OOS {_n(o['mean_daily_excess_bps'])} bp je Handelstag "
            f"(t = {_n(o['excess_t_stat'], 2)}), und die Basislinie „immer Long“ über dieselben "
            f"Fenster erzielt {_n(b['always_long_net_bps'])} bp je Trade gegenüber "
            f"{_n(b['strategy_dir_net_bps'])} bp bei der eigenen Richtung. Der Ertrag stammt "
            "also aus der Marktdrift (ein steigender Aktienmarkt, plus Survivorship-Bias der "
            "heutigen Large Caps), nicht aus einem Können der Regel: kein gemessener Vorteil. "
            "Wer Long-Aktien hält, verdient dasselbe, ohne die Regel zu brauchen. Deshalb heißt "
            f"{i} „nur Marktrendite“: keine Empfehlung, nur ein gekennzeichneter Beobachtungspunkt.")
    out.append(
        f"{len(is_hits)} Variante(n) erreichten im Lernzeitraum (IS) einen t-Wert von mindestens 2 "
        f"({', '.join(is_hits) or 'keine'}). Bei {n} Versuchen sind einzelne t-Werte um 2 allein "
        "durch Zufall zu erwarten. Maßgeblich ist der unberührte Testzeitraum (OOS): dort hatten "
        f"{len(oos_pos)} Variante(n) einen positiven Netto-Durchschnitt je Handelstag "
        f"({', '.join(oos_pos) or 'keine'}).")
    if skill:
        out.append(
            "Über den Marktdurchschnitt hinaus (Excess, t ≥ 2 und > 0 im OOS) zeigten sich Hinweise "
            f"bei: {', '.join(skill)}.")
    else:
        out.append(
            "Über die allgemeine Marktdrift hinaus (Excess gegenüber dem Durchschnitt des "
            "Universums) ließ sich im OOS bei keiner Variante ein Können mit t ≥ 2 belegen. "
            "Long-Regeln verdienen in einem steigenden Markt ohnehin etwas; das ist keine Fertigkeit.")
    if negative:
        parts = []
        for i in negative:
            r = res[i]
            parts.append(
                f"{i} (t IS {_n(r['IS']['t_stat'], 2)}, OOS {_n(r['OOS']['t_stat'], 2)}, gesamt "
                f"{_n(r['ALL']['t_stat'], 2)}; Ø brutto je Trade IS {_n(r['IS']['mean_gross_bps'])} bp, "
                f"OOS {_n(r['OOS']['mean_gross_bps'])} bp)")
        gross = [r[k]["mean_gross_bps"] for i in negative for r in (res[i],) for k in ("IS", "OOS")]
        out.append(
            "Signifikant **negativ** (IS-t ≤ −2 und OOS-t ≤ −2) sind: " + "; ".join(parts) + ". "
            "Das ist überwiegend die Kostenwirkung und keine nachgewiesene Gegenrichtung: vor "
            f"Kosten liegt der Ø brutto je Trade bei diesen Regeln nahe null (zwischen {_n(min(gross))} "
            f"und {_n(max(gross))} bp), die Kosten ({_n(COSTS_BPS['day'], 0)} bp je Daytrade, "
            f"{_n(COSTS_BPS['swing'], 0)} bp je Swing-Trade) machen das Ergebnis erst signifikant "
            "negativ. Die Umkehrung der Signale wäre deshalb kein Gewinn: auch die Gegenrichtung "
            "zahlt die Kosten.")
    return out


def _arrow(before, after, nd=1, pct=False) -> str:
    fmt = (lambda x: _pct(x)) if pct else (lambda x: _n(x, nd))
    if before is None:
        return fmt(after)
    return fmt(after) if fmt(before) == fmt(after) else f"{fmt(before)} → {fmt(after)}"


def _before_after(study: Study, before: dict) -> list[str]:
    """The before/after table of every strategy and the rows that changed."""
    old = {s["id"]: s for s in before.get("strategies", [])}
    rows, changed = [], []
    for spec in study.specs:
        new = study.results[spec.id]
        o = old.get(spec.id)
        if o is None:
            continue
        o_oos, n_oos = o["OOS"], new["OOS"]
        o_label = o.get("label") or ev.label_of(o.get("verdict"), o.get("beats_drift"))
        n_label = _label(new)
        is_changed = (o["verdict"] != new["verdict"] or o_label != n_label
                      or round(o_oos["t_stat"], 2) != round(n_oos["t_stat"], 2)
                      or round(o["IS"]["t_stat"], 2) != round(new["IS"]["t_stat"], 2)
                      or o["IS"]["n_trades"] != new["IS"]["n_trades"]
                      or o_oos["n_trades"] != n_oos["n_trades"])
        if is_changed:
            changed.append(spec.id)
        d_old = o_oos.get("mean_daily_net_bps")
        rows.append([
            f"**{spec.id}**" if is_changed else spec.id,
            _arrow(o_oos["n_trades"], n_oos["n_trades"], 0),
            _arrow(o_oos["mean_net_bps"], n_oos["mean_net_bps"]),
            _arrow(d_old, n_oos["mean_daily_net_bps"]),
            _arrow(o["IS"]["t_stat"], new["IS"]["t_stat"], 2),
            _arrow(o_oos["t_stat"], n_oos["t_stat"], 2),
            VERDICT_DE[o["verdict"]] if o["verdict"] == new["verdict"]
            else f"{VERDICT_DE[o['verdict']]} → {VERDICT_DE[new['verdict']]}",
            LABEL_DE[o_label] if o_label == n_label else f"{LABEL_DE[o_label]} → {LABEL_DE[n_label]}",
        ])
    out = [_table(["ID", "OOS Trades", "OOS Ø je Trade (bp)", "OOS " + DAILY_HEADER + " (bp)",
                   "IS t", "OOS t", "Urteil (vorab festgelegt)", "Etikett"], rows)]
    out.append("\nFett = Zeile hat sich geändert (Trade-Zahl, IS-t, OOS-t, Urteil oder Etikett). "
               "„Vorher“ ist der Stand vor der Korrektur (Datei `validated_before_review.json`). "
               "Bei „vorher“ war „Ø je Handelstag“ der Mittelwert der Tages-Cluster nach "
               "Einstiegstag, „nachher“ derselbe Wert (Tagestrader) bzw. der des "
               "Kalenderzeit-Portfolios (Swing).")
    out.append(f"\nGeänderte Zeilen: {', '.join(changed) or 'keine'}.")
    return out


def write_report(study: Study, path: str | Path, before: dict | None = None) -> Path:
    """The German research report for a non-technical reader.

    ``before``: the snapshot of the study before the review's corrections, for the
    before/after table.
    """
    if not study.include_oos:
        raise ValueError("the report needs the out-of-sample results (include_oos=True)")
    res, specs = study.results, study.specs
    inv = study.data_windows["inventory"]
    n_etf = len(EXTRA_SYMBOLS)
    L: list[str] = []
    L.append("# Kurzfrist-Handel: Simulation und ehrliche Bewertung")
    L.append(f"\nStudienstand: {study.as_of}. Alles ist eine **Simulation auf historischen "
             "Yahoo-Kursen**. Es wird nichts bei einem Broker ausgeführt, und eine Simulation ist "
             "keine echte Ausführung. Dies ist die Fassung **nach einer unabhängigen Prüfung**; "
             "Abschnitt 9 sagt, was korrigiert wurde und was sich dadurch geändert hat.\n")
    L.append("## 1. Was getestet wurde")
    L.append(
        f"- **Universum:** {inv['1d']['symbols'] - n_etf} liquide US-Einzelwerte (Large Caps aller "
        f"Sektoren) plus die {n_etf} ETFs SPY, QQQ, IWM, DIA = {inv['1d']['symbols']} Symbole.\n"
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
        "ausgestoppt. Bei M3 (VWAP-Rückkehr) liegt der Stop 1 σ vom tatsächlichen Einstiegskurs "
        "entfernt (nicht 1 σ jenseits der 2-σ-Auslöseschwelle, wie ursprünglich geplant); das "
        "Ziel ist das VWAP.\n"
        "- **Vorheriger Schlusskurs der Intraday-Regeln:** Er stammt aus dem letzten Balken der "
        "vorherigen Intraday-Sitzung, nicht aus den Tagesbalken (siehe Abschnitt 7, Datenfehler "
        "H-1).\n"
        "- **Lern- und Testzeitraum:** Die ersten 70 % der Handelstage jedes Datensatzes sind der "
        "Lernzeitraum (IS), die letzten 30 % der Testzeitraum (OOS). Parameter wurden nur im IS "
        "gewählt; der OOS wurde je Kandidat einmal ausgewertet und wird hier unverändert berichtet.\n"
        "- **Signifikanz, eine Größe für alle Entscheidungen:** Jede Strategie hat **eine** "
        "Tagesreihe, die „Basis des t-Werts“. Bei Daytrades ist es der Mittelwert der Trades des "
        "Einstiegstags. Bei Swing-Trades, deren Positionen bis zu 10 Tage offen sind und sich "
        "überlappen, ist es das **Kalenderzeit-Portfolio**: jeden Kalendertag der Durchschnitt der "
        "Tagesergebnisse aller an dem Tag offenen Positionen, Kosten am Ausstiegstag abgezogen. "
        "t-Wert, 95 %-Intervall (zieht ganze Tage neu, 2000 Ziehungen), die Kostenbedingung des "
        "Urteils (OOS-Ergebnis bei doppelten Kosten > 0) und die Frage „schlägt die Marktdrift?“ "
        "verwenden alle den Mittelwert **dieser** Reihe. Der Durchschnitt je Trade steht "
        "daneben nur zur Information (er gewichtet Tage mit vielen Signalen stärker). Als grobe "
        "Regel: |t| unter 2 ist von Zufall nicht zu unterscheiden.")
    L.append(f"\n**Anzahl getesteter Varianten insgesamt: {study.variants_tried}** "
             f"({len(PRE_REGISTERED)} vorab festgelegte, {len(VARIANTS)} nachträgliche).\n")

    L.append("## 2. Ergebnisse je Strategie (netto nach Kosten)")
    L.append("„Ø je Trade“ = durchschnittlicher Netto-Ertrag je Trade in Basispunkten (zur "
             "Information; kann an Tagen mit vielen Trades vom Tageswert abweichen, auch im "
             f"Vorzeichen). „{DAILY_HEADER}“ = Mittelwert der Tagesreihe, zu der der t-Wert gehört "
             "(Swing: Kalenderzeit-Portfolio, siehe Abschnitt 1). t = t-Wert dieser Reihe; "
             "Intervall = 95 %-Intervall von „Ø je Handelstag“ im OOS (bp). Treffer = Anteil der "
             "Trades mit Gewinn.\n")
    rows = []
    for s in specs:
        r = res[s.id]
        i, o = r["IS"], r["OOS"]
        lab = _label(r)
        mark = "**" if lab != "rejected" else ""
        rows.append([
            s.id, s.name_de, TF_DE[s.timeframe],
            str(i["n_trades"]), _pct(i["win_rate"]), _n(i["mean_net_bps"]),
            _n(i["mean_daily_net_bps"]), _n(i["t_stat"], 2),
            str(o["n_trades"]), _pct(o["win_rate"]), _n(o["mean_net_bps"]),
            _n(o["mean_daily_net_bps"]), _n(o["t_stat"], 2),
            f"[{_n(o['ci_low_bps'])}; {_n(o['ci_high_bps'])}]",
            mark + LABEL_DE[lab] + (" (stark)" if r["strong"] and lab == "validated" else "") + mark,
        ])
    L.append(_table(["ID", "Strategie", "Daten", "IS Trades", "IS Treffer", "IS Ø je Trade",
                     "IS " + DAILY_HEADER, "IS t", "OOS Trades", "OOS Treffer", "OOS Ø je Trade",
                     "OOS " + DAILY_HEADER, "OOS t", "OOS 95 %-Intervall", "Urteil"], rows))
    L.append("\nUrteilsregeln (vorab festgelegt, alle Mittelwerte = „Ø je Handelstag“): *belegt* = "
             "IS-t ≥ 2 und OOS-t ≥ 2 (beide positiv), OOS-Mittel auch bei doppelten Kosten > 0, "
             "mindestens 100 OOS-Trades an ≥ 20 OOS-Tagen. *experimentell* = IS-t ≥ 2, OOS-Mittel > 0, "
             "OOS-t ≥ 1, ≥ 50 OOS-Trades. Alles andere *verworfen*. „stark“ nur bei gepooltem t ≥ 3 "
             "**und** geschlagener Marktdrift. **Etikett** (abgeleitet, nach dem Urteil): *belegt* / "
             "*experimentell* nur, wenn das Urteil so lautet UND die Regel die Marktdrift schlägt "
             "(Excess-t ≥ 2 und Excess > 0 im OOS); besteht sie das Urteil, schlägt aber die "
             "Marktdrift nicht, heißt sie *formal bestanden, aber nur Marktrendite* (in den Daten "
             "„beta“). Die 5-Minuten-Strategien haben "
             f"nur {study.splits['5m']['n_oos_days']} OOS-Tage und können die Regel für „belegt“ "
             "konstruktionsbedingt nicht erfüllen.")

    L.append("\n### Brutto, Excess und Long/Short (OOS)")
    L.append("Excess = Ertrag abzüglich des Durchschnitts des gesamten Universums im gleichen "
             "Zeitfenster (Short: umgekehrt). Nur der Excess ist Können; der Rest ist Marktdrift. "
             "Sharpe wird nur über die Handelstage mit Position gerechnet (Tage ohne Trade zählen "
             "nicht) und ist deshalb höher als bei einem ganzjährig investierten Portfolio. "
             "Max. Drawdown = größter Rückgang der **zusammengesetzten** Kapitalkurve der "
             "Tagesreihe in % (höchstens 100 %).\n")
    rows = []
    for s in specs:
        o = res[s.id]["OOS"]
        rows.append([s.id, _n(o["mean_gross_bps"]), _n(o["median_net_bps"]),
                     _n(o["profit_factor"], 2), _n(o["mean_excess_bps"]),
                     _n(o["mean_daily_excess_bps"]), _n(o["excess_t_stat"], 2),
                     f"{o['long']['n_trades']} / {_n(o['long']['mean_net_bps'])}",
                     f"{o['short']['n_trades']} / {_n(o['short']['mean_net_bps'])}",
                     _n(o["sharpe"], 2), _n(o["max_drawdown_pct"], 1) + " %"])
    L.append(_table(["ID", "Ø brutto je Trade (bp)", "Median netto je Trade (bp)", "Profit-Faktor",
                     "Ø Excess je Trade (bp)", "Ø Excess je Handelstag (bp, Basis des Excess-t)",
                     "Excess t", "Long: Trades / Ø bp je Trade", "Short: Trades / Ø bp je Trade",
                     "Sharpe (nur Handelstage mit Position)", "Max. Drawdown (zusammengesetzt)"], rows))

    L.append("\n## 3. Vergleich mit Baselines (OOS)")
    L.append("Auf den **eigenen Trade-Fenstern der Strategie (einschließlich ihres Ausstiegstags)**: "
             "Jedes Fenster läuft von der Einstiegseröffnung bis zum Schluss des Balkens, an dem die "
             "Strategie ausgestiegen ist; ein ausgestoppter Trade endet also am Stopptag. Die "
             "Baselines sind damit nicht „ohne Stop“ im strengen Sinn, sondern an den Ausstieg der "
             "Strategie gebunden. Verglichen werden **immer Long**, die **eigenen Richtungen** der "
             "Strategie (jeweils ohne Stop innerhalb des Fensters) und **Zufallsrichtung** "
             "(200 Wiederholungen, Ø und 95 %-Band). Liegt die eigene Richtung nicht über dem "
             "Zufallsband, steckt in der Richtungswahl kein Können. Alle Werte sind Ø je Trade.\n")
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

    L.append("\n## 4. Kostenempfindlichkeit (OOS, " + DAILY_HEADER + " in bp, t-Wert der Reihe)")
    rows = []
    for s in specs:
        cs = res[s.id]["cost_sensitivity"]
        rows.append([s.id] + [
            f"{_n(cs[k]['OOS']['mean_daily_net_bps'])} (t {_n(cs[k]['OOS']['t_stat'], 2)})"
            for k in ("1x", "2x", "4x")])
    L.append(_table(["ID", "Basiskosten", "2× Kosten", "4× Kosten"], rows))

    L.append("\n## 5. Was verbessert wurde (und was nicht), und was offenzulegen ist")
    L.append(f"Nach den ersten IS-Ergebnissen wurden {len(VARIANTS)} zusätzliche Varianten "
             "ausprobiert (erlaubt waren höchstens 8), jede mit einer Begründung, **alle "
             "protokolliert, auch die Fehlschläge**, und nur im IS gewählt. Danach wurde der OOS "
             "einmal für alle ausgewertet; es wurde nichts an den Regeln oder Parametern "
             "nachjustiert. **Zwei Dinge sind dabei offenzulegen:**\n\n"
             "1. **S3l (Ausbruch mit Volumen, nur Long)** wurde im IS wegen des positiven "
             "Brutto-Ergebnisses der Long-Seite gewählt, obwohl sein IS-Excess gegenüber dem "
             "Marktdurchschnitt schon damals bei −0,4 bp lag (t = −0,09). Dass sich im OOS "
             "dasselbe Bild zeigt (Ertrag = Marktrendite), ist deshalb keine Überraschung.\n"
             "2. **Das Kriterium „schlägt die Marktdrift“ (`beats_drift`) wurde erst nach dem "
             "ersten vollständigen Lauf ergänzt**, nachdem die OOS-Ergebnisse bereits bekannt "
             "waren (der erste Lauf, `validated_run1.json`, enthält diese Kennzahl nicht). "
             "Alle Zahlen des ersten Laufs waren identisch zu denen des zweiten, es wurde "
             "nichts neu eingestellt; die Kennzahl ist aber ein nachträglich hinzugefügter "
             "Filter und kein vorab festgelegter. Sie ändert nur, wie Ergebnisse benannt werden, "
             "nie, ob eine Regel Geld verdient hat.\n\n"
             "Außerdem wurden **die Statistiken nach einer unabhängigen Prüfung korrigiert** "
             "(Datenfehler H-1, Überlappung der Swing-Trades M-1, einheitliche Größe M-2), "
             "die Regeln selbst aber nicht verändert. Die Vorher/Nachher-Tabelle steht in "
             "Abschnitt 9.\n")
    rows = []
    for s in specs:
        if s.id in {v.id for v in VARIANTS}:
            r = res[s.id]
            rows.append([s.id, s.name_de, s.note_de, _n(r["IS"]["mean_net_bps"]),
                         _n(r["IS"]["mean_daily_net_bps"]), _n(r["IS"]["t_stat"], 2),
                         _n(r["OOS"]["mean_net_bps"]), _n(r["OOS"]["mean_daily_net_bps"]),
                         _n(r["OOS"]["t_stat"], 2), LABEL_DE[_label(r)]])
    L.append(_table(["ID", "Variante", "Begründung (nur IS-Evidenz)", "IS Ø je Trade",
                     "IS " + DAILY_HEADER, "IS t", "OOS Ø je Trade", "OOS " + DAILY_HEADER,
                     "OOS t", "Etikett"], rows))

    L.append("\n## 6. Ehrliche Gesamteinschätzung")
    for para in _assessment(study):
        L.append(f"\n{para}")

    L.append("\n## 7. Grenzen")
    L.append(
        "- **Survivorship-Bias:** Das Universum sind die heutigen Large Caps, rückwirkend "
        "angewandt; Verlierer von gestern fehlen. Das begünstigt vor allem Long-Regeln.\n"
        "- **Dividendenbereinigung (Datenfehler H-1, korrigiert):** Yahoo bereinigt Tagesbalken um "
        "Dividenden, Intraday-Balken nicht. Die erste Fassung verglich den (bereinigten) "
        "Tagesschlusskurs mit unbereinigten Intraday-Kursen; der „vorherige Schlusskurs“ lag zu "
        "Beginn des 60-Minuten-Fensters um bis zu 12 % zu niedrig (Median 370 bp in den ersten "
        "20 Tagen), was H1/H1b fast immer auf „Long“ stellte und M2a/M2b verfälschte. Jetzt kommt "
        "der vorherige Schlusskurs aus dem letzten Intraday-Balken der Vorsitzung (gleiche Quelle, "
        "gleiche Bereinigung); Symbol-Tage ohne vorherige Intraday-Sitzung entfallen, ebenso solche, "
        "deren Vorsitzung nicht mit dem Schlussbalken endet (fehlender letzter Balken, 325 "
        "Symbol-Tage im 60-Minuten-Datensatz), weil deren letzter Kurs nicht der Schlusskurs ist. "
        "Restrisiko: Die Intraday-Daten enthalten Dividenden-Abschläge als echte Kurslücken "
        "(wie im realen Handel); Trendflags und ATR% stammen weiter aus bereinigten Tagesbalken "
        "(Verhältniszahlen, davon kaum berührt).\n"
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
        "- **Swing-Trades überlappen:** Die Auswertung läuft deshalb auf dem Kalenderzeit-Portfolio "
        "(Abschnitt 1) statt auf Einstiegstagen. Auch die Tageswerte dieses Portfolios sind "
        "untereinander noch verwandt (Positionen laufen mehrere Tage); die t-Werte bleiben eher "
        "optimistisch. S1 verlässt die Position zum Schlusskurs, den man erst am Ende des Tages "
        "kennt.\n"
        "- **Mehrfachtests:** Bei vielen Varianten entstehen Treffer zufällig. Der OOS ist nur "
        "ein einziger Zeitraum und kann selbst Zufall sein.\n"
        "- **Simulation ≠ echte Ausführung:** Teilausführungen, Slippage in schnellen Märkten, "
        "Steuern und Verhalten sind nicht abgebildet.")

    L.append("\n## 8. Was daraus für die täglichen Empfehlungen folgt")
    labels = {s.id: _label(res[s.id]) for s in specs}
    validated = [i for i, v in labels.items() if v == "validated"]
    experimental = [i for i, v in labels.items() if v == "experimental"]
    beta = [i for i, v in labels.items() if v == "beta"]
    if validated:
        L.append("Nur als „belegt“ etikettierte Strategien (" + ", ".join(validated) + ") erzeugen "
                 "standardmäßig gekennzeichnete Kurzfrist-Ideen (Horizont Tag/Swing, Einstieg zur "
                 "nächsten Eröffnung, Stop, Haltedauer). Sie bleiben Simulationsergebnisse mit "
                 "Unsicherheit, keine Anlageberatung.")
    else:
        L.append("Da keine Strategie belegt ist (Abschnitt 6), gibt es **keine** Kurzfrist-"
                 "Empfehlung. Die tägliche Liste zeigt die Regeln nur als technisches Bild mit "
                 "ihrem Etikett (Beobachtungsliste, nie als Empfehlung). „Kein Vorteil gefunden“ "
                 "ist ein gültiges Ergebnis; Anlagehorizonte der übrigen Empfehlungen bleiben die "
                 "der Analyse.")
    if experimental:
        L.append("Experimentelle Strategien (" + ", ".join(experimental) + ") "
                 "können auf Wunsch gesondert als „experimentell“ markiert angezeigt werden.")
    if beta:
        L.append("Formal bestandene Strategien ohne Vorteil gegenüber der Marktdrift (" + ", ".join(beta)
                 + ", Etikett „nur Marktrendite“) sind **keine** Empfehlung. Das Programm zeigt sie "
                 "nur als gekennzeichneten Beobachtungspunkt mit dem "
                 "Pflichthinweis: „verdient nur, was der Markt in denselben Tagen ohnehin brachte; "
                 "kein belegter Vorteil“.")

    L.append("\n## 9. Korrekturen nach der unabhängigen Prüfung")
    L.append(
        "Eine unabhängige Prüfung hat die Zahlen der ersten Fassung nachgerechnet, die "
        "Schlussfolgerung bestätigt (nach Kosten zeigt keine Regel Können, nichts schlägt die "
        "Marktdrift, kein Look-ahead, der Simulator ist korrekt) und folgende Mängel gefunden, "
        "die jetzt behoben sind. Es wurde keine Strategie verändert oder hinzugefügt:\n\n"
        "- **H-1 Datenfehler:** vorheriger Schlusskurs der Intraday-Regeln aus der vorherigen "
        "Intraday-Sitzung statt aus (dividendenbereinigten) Tagesbalken (Abschnitt 7).\n"
        "- **H-2 Beschriftung:** Das vorab festgelegte Urteil bleibt unverändert, die Auszeichnung "
        "„belegt“ gilt aber nur noch, wenn die Regel auch die Marktdrift schlägt; sonst "
        "„formal bestanden, aber nur Marktrendite“. Die Kurzfrist-Ideen der Tagesliste folgen "
        "dem Etikett.\n"
        "- **M-1 Überlappung:** t-Werte der Swing-Strategien jetzt auf dem Kalenderzeit-"
        "Portfolio statt nach Einstiegstag.\n"
        "- **M-2 eine Größe:** Kostenbedingung und Marktdrift-Kriterium verwenden den Mittelwert "
        "derselben Reihe wie der t-Wert; „Ø je Trade“ bleibt als Information.\n"
        "- Kleinere Punkte: Beschreibung der Baselines, Drawdown auf zusammengesetzter Kurve, "
        "Zählung des Universums, Wortlaut zu den signifikant negativen Regeln und zu den "
        "Variantenbegründungen, Alterskontrollen der Live-Ideen.\n")
    if before:
        L.extend(_before_after(study, before))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def study_from_snapshot(snap: dict) -> Study:
    """Rebuild a ``Study`` from a written ``validated.json`` so the report can be regenerated
    without running the study again."""
    by_id = {sp.id: sp for sp in all_specs()}
    rows = [r for r in snap["strategies"] if r["id"] in by_id]
    results = {r["id"]: {k: r.get(k) for k in (
        "verdict", "strong", "beats_drift", "label", "IS", "OOS", "ALL", "cost_sensitivity",
        "baselines")} for r in rows}
    return Study(
        as_of=snap["study_date"], specs=[by_id[r["id"]] for r in rows], results=results,
        variants_tried=snap.get("variants_tried", 0), data_windows=snap["data"],
        splits=snap["splits"], drift=snap.get("drift_reference") or {}, include_oos=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tradingagents.scanner.shortterm.research",
        description="Run the short-horizon study (simulation only) and write the report.")
    parser.add_argument("--out", required=True, help="directory for validated.json and the report")
    parser.add_argument("--as-of", default=None, help="study date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--cache", default=None, help="directory caching the Yahoo downloads")
    parser.add_argument("--before", default=None,
                        help="snapshot of the study before the review corrections, for the "
                             "before/after table (default: <out>/validated_before_review.json)")
    parser.add_argument("--update-package", action="store_true",
                        help="also overwrite the packaged validated.json that live.py reads")
    parser.add_argument("--report-only", action="store_true",
                        help="regenerate the report from <out>/validated.json without running "
                             "the study (nothing is downloaded or measured)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.report_only:
        out = Path(args.out)
        snap = json.loads((out / SNAPSHOT_NAME).read_text(encoding="utf-8"))
        before_path = Path(args.before) if args.before else out / "validated_before_review.json"
        before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.exists() else None
        report = write_report(study_from_snapshot(snap), out / "BERICHT_Kurzfrist.md", before=before)
        print(f"report: {report}")
        return 0
    as_of = args.as_of or str(
        (pd.Timestamp.now(tz="America/New_York").normalize() - pd.Timedelta(days=1)).date())
    study = run_study(as_of, args.cache, include_oos=True)
    out = Path(args.out)
    before_path = Path(args.before) if args.before else out / "validated_before_review.json"
    before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.exists() else None
    write_snapshot(study, out / SNAPSHOT_NAME)
    report = write_report(study, out / "BERICHT_Kurzfrist.md", before=before)
    if args.update_package:
        from tradingagents.scanner.shortterm.live import SNAPSHOT_PATH
        write_snapshot(study, SNAPSHOT_PATH)
    print(f"snapshot: {out / SNAPSHOT_NAME}\nreport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
