"""Today's labelled short-horizon ideas from the persisted study, nothing more.

An idea exists only when the strategy behind it was ``validated`` (or, when asked
for, ``experimental``, and then it is flagged as such) by the pre-registered rule in
``evaluate``. If no strategy passed, there are no ideas: an honest "no edge found" is
a valid, expected outcome. The signals come from the same functions the study used
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
from pathlib import Path

import pandas as pd

from tradingagents.scanner.shortterm import daily_features
from tradingagents.scanner.shortterm.daytrade import SAME_OPEN_FAMILIES
from tradingagents.scanner.shortterm.swing import ALL_DAILY_SIGNALS, daily_signal

logger = logging.getLogger(__name__)

SNAPSHOT_PATH = Path(__file__).with_name("validated.json")
INTRADAY_LIVE_SUPPORTED = False
MAX_STALE_DAYS = 5          # a last bar older than this is not "today"


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
    verdict: str              # "validated" | "experimental"
    study_date: str
    simulation_only: bool = True
    # False when the OOS gain is no more than the universe's own drift (a long rule in a
    # rising market): the idea is validated by the pre-registered rule but not "skill"
    beats_drift: bool | None = None
    oos_excess_bps: float | None = None
    oos_excess_t_stat: float | None = None


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


def shortterm_ideas(
    daily_frames: dict[str, pd.DataFrame],
    as_of_date: str,
    snapshot: dict | None = None,
    include_experimental: bool = False,
    intraday_frames: dict[str, dict[str, pd.DataFrame]] | None = None,
) -> list[ShortTermIdea]:
    """Ideas for the next session from the strategies the snapshot says have an edge.

    ``daily_frames`` are daily OHLCV frames (as ``load_bulk_ohlcv`` returns), cut at
    ``as_of_date``. No snapshot or nothing validated means no ideas.
    """
    snap = snapshot if snapshot is not None else load_snapshot()
    if not snap:
        return []
    wanted = {"validated"} | ({"experimental"} if include_experimental else set())
    as_of = pd.Timestamp(as_of_date).normalize()
    ideas: list[ShortTermIdea] = []
    for strat in snap.get("strategies", []):
        family = strat.get("family")
        if strat.get("verdict") not in wanted:
            continue
        if strat.get("timeframe") != "1d" or family in SAME_OPEN_FAMILIES \
                or family not in ALL_DAILY_SIGNALS:
            continue          # needs the opening print or intraday bars (see module docstring)
        params = strat.get("params") or {}
        stop_atr = params.get("stop_atr")
        oos = strat.get("OOS") or {}
        hold_en, hold_de = _holding(strat["horizon"], params, family)
        for symbol, df in daily_frames.items():
            df = df[df.index <= as_of]
            if len(df) < 2 or (as_of - df.index[-1]).days > MAX_STALE_DAYS:
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
                verdict=strat["verdict"], study_date=snap.get("study_date", ""),
                beats_drift=strat.get("beats_drift"),
                oos_excess_bps=oos.get("mean_excess_bps"),
                oos_excess_t_stat=oos.get("excess_t_stat"),
            ))
    return ideas
