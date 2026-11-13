"""Today's labelled short-horizon ideas from the persisted study, nothing more.

An idea exists only when the strategy behind it carries the label ``validated`` (or,
when asked for, ``experimental`` or ``beta``, and then it is flagged as such). The label
is derived from the pre-registered ``verdict`` AND ``beats_drift`` (``evaluate.label_of``):
a rule that passes the verdict but earns no more than the market itself in the same
days is ``beta`` and is not emitted by default; when it is, it carries a mandatory
caveat. If no strategy qualifies, there are no ideas: an honest "no edge found" is a
valid, expected outcome. The signals come from the same functions the study used
(``swing.daily_signal`` on ``daily_features``), never from a second implementation.

Only signals known from daily bars BEFORE the next open are produced here (D3-D5,
S1-S4 and their variants). D1/D2 need the opening price itself, and the 60/5-minute
strategies need today's intraday bars; the optional ``intraday_frames`` argument is
accepted so callers have a stable signature, but intraday strategies are not
evaluated live (see ``INTRADAY_LIVE_SUPPORTED``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from tradingagents.scanner import session
from tradingagents.scanner.shortterm import daily_features, evaluate as ev
from tradingagents.scanner.shortterm.daytrade import SAME_OPEN_FAMILIES
from tradingagents.scanner.shortterm.swing import ALL_DAILY_SIGNALS, daily_signal

logger = logging.getLogger(__name__)

SNAPSHOT_PATH = Path(__file__).with_name("validated.json")
INTRADAY_LIVE_SUPPORTED = False
MAX_STALE_DAYS = 4          # calendar days allowed between the last settled bar and as_of_date
SNAPSHOT_STALE_DAYS = 120   # a study older than this marks its ideas ``snapshot_stale``
BETA_CAVEAT_EN = ("earns what the market itself earned over the same days; no measured skill")
BETA_CAVEAT_DE = ("verdient nur, was der Markt in denselben Tagen ohnehin brachte; "
                  "kein belegter Vorteil")


@dataclass(frozen=True)
class ShortTermIdea:
    symbol: str
    direction: str            # "Long" | "Short"
    horizon: str              # "day" | "swing"
    strategy_id: str
    strategy_name_en: str
    strategy_name_de: str
    entry_rule_en: str
    entry_rule_de: str
    reference_price: float    # last close; the real fill is the next open
    stop: float | None        # indicative, measured from the last close
    target: float | None
    holding_en: str
    holding_de: str
    oos_win_rate: float | None
    oos_mean_net_bps: float | None
    oos_n_trades: int | None
    oos_t_stat: float | None
    verdict: str              # the pre-registered verdict as recorded: "validated" | "experimental"
    study_date: str
    simulation_only: bool = True
    # False when the OOS gain is no more than the universe's own drift (a long rule in a
    # rising market): the idea passed the pre-registered rule but is not "skill"
    beats_drift: bool | None = None
    oos_excess_bps: float | None = None       # mean of the judged daily series (see evaluate)
    oos_excess_t_stat: float | None = None
    # derived from verdict AND beats_drift: "validated" | "experimental" | "beta"
    label: str = "validated"
    # human-readable warning; mandatory (never None) for beta ideas and stale snapshots
    caveat_en: str | None = None
    caveat_de: str | None = None
    snapshot_age_days: int | None = None      # as_of_date minus the study date
    snapshot_stale: bool = False              # the study is older than SNAPSHOT_STALE_DAYS
    oos_mean_daily_net_bps: float | None = None   # mean of the judged series (the t's basis)


def load_snapshot(path: str | Path | None = None) -> dict | None:
    """The persisted study, or None when there is none (or it cannot be read)."""
    target = Path(path) if path else SNAPSHOT_PATH
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("short-term snapshot unavailable (%s): %s", target, exc)
        return None


def _holding(horizon: str, params: dict, family: str) -> tuple[str, str]:
    if horizon == "day":
        return "until that day's close", "bis Handelsschluss des Einstiegstags"
    n = int(params.get("hold", 5))
    en = f"up to {n} trading days"
    de = f"bis zu {n} Handelstage"
    if family == "S1":
        return (f"{en}, earlier at the first close beyond the 5-day average",
                f"{de}, früher beim ersten Schluss jenseits des 5-Tage-Durchschnitts")
    return en, de


_HOLIDAYS = None


def _market_holidays(start: pd.Timestamp, end: pd.Timestamp) -> set:
    """Approximate US market holidays (federal holidays plus Good Friday) in a range."""
    global _HOLIDAYS
    if _HOLIDAYS is None:
        from pandas.tseries.holiday import GoodFriday, USFederalHolidayCalendar
        cal = USFederalHolidayCalendar()
        rng = ("2000-01-01", "2100-12-31")
        _HOLIDAYS = set(cal.holidays(*rng)) | set(GoodFriday.dates(*rng))
    return {d for d in _HOLIDAYS if start <= d <= end}


def _is_current(last_bar: pd.Timestamp, as_of: pd.Timestamp) -> bool:
    """True when ``last_bar`` is the newest settled session as of ``as_of``.

    At most ``MAX_STALE_DAYS`` calendar days old, AND no weekday between the two dates
    that could have been a session (a Thursday bar is not "yesterday" for a Monday).
    Holidays are approximated; a bar from before a holiday weekend still counts.
    """
    if (as_of - last_bar).days > MAX_STALE_DAYS:
        return False
    between = pd.date_range(last_bar + pd.Timedelta(days=1), as_of - pd.Timedelta(days=1))
    holidays = _market_holidays(last_bar, as_of)
    return all(d.weekday() >= 5 or d in holidays for d in between)


def shortterm_ideas(
    daily_frames: dict[str, pd.DataFrame],
    as_of_date: str,
    snapshot: dict | None = None,
    include_experimental: bool = False,
    intraday_frames: dict[str, dict[str, pd.DataFrame]] | None = None,
    include_beta: bool = False,
    now: datetime | None = None,
) -> list[ShortTermIdea]:
    """Ideas for the next session from the strategies the snapshot says have an edge.

    ``daily_frames`` are daily OHLCV frames (as ``load_bulk_ohlcv`` returns), cut at
    ``as_of_date``. By default only strategies labelled ``validated`` speak
    (``experimental`` too with ``include_experimental``; ``beta`` ones, i.e. verdict passed
    but no better than the market's own drift, only with ``include_beta`` and always with
    a caveat). No snapshot or nothing qualifying means no ideas.

    The signal must come from the newest settled session: a bar that is still running
    at ``now`` (default: the wall clock) is dropped (``scanner.session.settled``), and
    the last bar must be at most ``MAX_STALE_DAYS`` calendar days before ``as_of_date``
    with no missed session in between. ``snapshot_age_days`` and ``snapshot_stale``
    tell how old the study is.
    """
    snap = snapshot if snapshot is not None else load_snapshot()
    if not snap:
        return []
    wanted = {"validated"} | ({"experimental"} if include_experimental else set()) \
        | ({"beta"} if include_beta else set())
    as_of = pd.Timestamp(as_of_date).normalize()
    study_date = snap.get("study_date", "")
    try:
        age = (as_of - pd.Timestamp(study_date).normalize()).days
    except (ValueError, TypeError):
        age = None
    stale = age is None or age > SNAPSHOT_STALE_DAYS
    cut = {sym: df[df.index <= as_of] for sym, df in daily_frames.items()}
    cut = session.settled({sym: df for sym, df in cut.items() if len(df)}, now)
    ideas: list[ShortTermIdea] = []
    for strat in snap.get("strategies", []):
        family = strat.get("family")
        label = strat.get("label") or ev.label_of(strat.get("verdict"), strat.get("beats_drift"))
        if label not in wanted:
            continue
        if strat.get("timeframe") != "1d" or family in SAME_OPEN_FAMILIES \
                or family not in ALL_DAILY_SIGNALS:
            continue          # needs the opening print or intraday bars (see module docstring)
        params = strat.get("params") or {}
        stop_atr = params.get("stop_atr")
        oos = strat.get("OOS") or {}
        hold_en, hold_de = _holding(strat["horizon"], params, family)
        caveat_en = BETA_CAVEAT_EN if label == "beta" else None
        caveat_de = BETA_CAVEAT_DE if label == "beta" else None
        if stale:
            note_en = (f"the study snapshot is {age} days old (limit {SNAPSHOT_STALE_DAYS}); "
                       "re-run the study" if age is not None
                       else "the study snapshot has no valid date; re-run the study")
            note_de = (f"die Studie ist {age} Tage alt (Grenze {SNAPSHOT_STALE_DAYS}); bitte neu "
                       "rechnen" if age is not None
                       else "die Studie hat kein gültiges Datum; bitte neu rechnen")
            caveat_en = f"{caveat_en}; {note_en}" if caveat_en else note_en
            caveat_de = f"{caveat_de}; {note_de}" if caveat_de else note_de
        for symbol, df in cut.items():
            if len(df) < 2 or not _is_current(df.index[-1].normalize(), as_of):
                continue
            feats = daily_features(df)
            side = int(daily_signal(family, feats, params).iloc[-1])
            if side == 0:
                continue
            last = feats.iloc[-1]
            close, atr = float(last["close"]), float(last["atr"])
            stop = close - side * float(stop_atr) * atr if stop_atr and atr == atr else None
            ideas.append(ShortTermIdea(
                symbol=symbol, direction="Long" if side > 0 else "Short",
                horizon=strat["horizon"], strategy_id=strat["id"],
                strategy_name_en=strat["name_en"], strategy_name_de=strat["name_de"],
                entry_rule_en="at the next open", entry_rule_de="zur nächsten Eröffnung",
                reference_price=close, stop=stop, target=None,
                holding_en=hold_en, holding_de=hold_de,
                oos_win_rate=oos.get("win_rate"), oos_mean_net_bps=oos.get("mean_net_bps"),
                oos_n_trades=oos.get("n_trades"), oos_t_stat=oos.get("t_stat"),
                verdict=strat["verdict"], study_date=study_date,
                beats_drift=strat.get("beats_drift"),
                oos_excess_bps=oos.get("mean_daily_excess_bps", oos.get("mean_excess_bps")),
                oos_excess_t_stat=oos.get("excess_t_stat"),
                label=label, caveat_en=caveat_en, caveat_de=caveat_de,
                snapshot_age_days=age, snapshot_stale=stale,
                oos_mean_daily_net_bps=oos.get("mean_daily_net_bps"),
            ))
    return ideas
