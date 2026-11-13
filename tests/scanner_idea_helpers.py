"""Hand-built ideas, evidence and notes for the report, digest and CLI tests.

The real idea layer needs the study snapshots and live bars; these helpers build the same
objects directly so a test can state exactly which idea is recommended, watch-only or
"avoid" and what the notes say. Everything is offline and deterministic.
"""

from tests.scanner_helpers import END, falling, flat, rising
from tradingagents.scanner import horizons as hz
from tradingagents.scanner.scan import run_scan as _real_run_scan
from tradingagents.scanner.validation import DirectionStats, ValidationResult

STUDY = "2026-11-13"


def frames(n=520):
    out = {"SPY": flat(n), "^VIX": flat(n, start=15.0)}
    for i in range(3):
        out[f"UP{i}"] = rising(n, daily=0.002 + 0.001 * i, wiggle=0.004)
        out[f"DN{i}"] = falling(n, daily=0.002 + 0.001 * i, wiggle=0.004)
    return out


def loader(symbols, as_of, lookback):
    return {s: f for s, f in frames().items() if s in symbols}


def base_result(universe="us", top=3):
    members = [s for s in frames() if s not in ("SPY", "^VIX")]
    # the real function, bound at import: a test may patch scan.run_scan afterwards
    return _real_run_scan(universe, END, tickers=members, top=top, loader=loader)


def evidence(label="validated", horizon="long", side="long", **kw):
    fields = {
        "label": label, "rule_id": "T1", "rule_name": {"en": "Momentum 12-1", "de": "Momentum 12-1"},
        "family": "stocks_us", "horizon": horizon, "side": side, "study_date": STUDY,
        "judged": "long_excess", "oos_n": 44, "oos_mean_bps": 597.0, "oos_t": 2.2, "is_t": 2.5,
        "window": ("2022-01-03", "2026-06-30"), "n_rules_tested": 7}
    fields.update(kw)
    return hz.Evidence(**fields)


def idea(symbol, direction="Long", horizon="long", label="validated", watch_only=None,
         stance=None, price=354.88, stop=321.34, size=8.5, rank=1, **kw):
    if watch_only is None:
        watch_only = label not in hz.RECOMMENDED
    if stance is None:
        stance = {"validated": "recommended", "experimental": "experimental"}.get(label, "watch")
    return hz.Idea(
        symbol=symbol, direction=direction, horizon=horizon, rule_id="T1", rank_score=1.0,
        reference_price=price, reference_date="2026-11-12",
        entry={"en": "at the next open", "de": "zur nächsten Eröffnung"}, stop=stop, target=None,
        size_pct=size, holding={"en": hz.holding_text(horizon, "stocks_us", "en"),
                                "de": hz.holding_text(horizon, "stocks_us", "de")},
        reason_items=[], evidence=evidence(label, horizon, direction.lower(), **kw),
        watch_only=watch_only, stance=stance, rank=rank,
        rule_name={"en": "Momentum 12-1", "de": "Momentum 12-1"})


def note(code, horizon, side=None, family="stocks_us", **params):
    return hz._note(code, horizon, side, family, **params)


def day_note(n=24):
    return note("idea_no_rec_short_term", "day", None, n=n, date="2026-11-12",
                tf_en="daily", tf_de="Tages-")


def rules_note(horizon, side, n=11):
    return note("idea_no_rec_rules", horizon, side, n=n, date=STUDY)


def idea_set(ideas, notes, family="stocks_us"):
    return hz.IdeaSet(family=family, study_date=STUDY, evidence_available=True, ideas=ideas,
                      notes=notes)


def result_with(ideas, notes, universe="us", family="stocks_us"):
    result = base_result(universe)
    result.ideas = idea_set(ideas, notes, family)
    return result


def empty_result(universe="us", family="stocks_us"):
    """Nothing evidenced anywhere: every horizon has its note, like DAX or crypto today."""
    notes = [day_note(), rules_note("medium", "long"), rules_note("medium", "short"),
             rules_note("long", "long", 7), rules_note("long", "short", 7),
             note("idea_not_studied", "swing", None, family)]
    return result_with([], notes, universe, family)


def evidenced_result(universe="us", symbols=("UP2", "UP1", "UP0")):
    """One evidenced long-term Long list plus notes for everything else, like US stocks today."""
    ideas = [idea(s, rank=i) for i, s in enumerate(symbols, 1)]
    notes = [day_note(15), rules_note("medium", "long"), rules_note("medium", "short"),
             rules_note("long", "short", 7),
             note("idea_beta_only", "swing", None, n=9, rules_en="Volume breakout, long side only", rules_de="Ausbruch mit Volumen, nur Long", date="2026-11-12")]
    return result_with(ideas, notes, universe)


def validation(long_ret, short_pnl, uni, months=24, rebalances=23):
    """A ValidationResult from constructed numbers (fractions, not percent)."""
    return ValidationResult(
        universe="us", benchmark="SPY", as_of_date=END, months=months, horizon=21,
        first_rebalance="2024-12-09", last_rebalance="2026-10-14", rebalances=rebalances,
        instruments=60,
        stats={"Long": DirectionStats("Long", 100, long_ret, 0.0, 0.5, long_ret),
               "Short": DirectionStats("Short", 50, None if short_pnl is None else -short_pnl, 0.0,
                                       0.4, short_pnl),
               "Neutral": DirectionStats("Neutral", 80, uni)},
        spread=-0.008, spread_tstat=-0.6, spread_dates=23, universe_avg_return=uni,
        benchmark_avg_return=0.01, spread_pooled=-0.024)
