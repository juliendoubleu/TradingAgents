"""One instrument's score, direction and trade plan, from its daily bars alone.

Six components, each in [-1, +1], are weighted into a score in [-100, +100]. The
direction comes from that score and the market regime (a long is easier in a risk-on
market, a short in a risk-off one). A Long or Short also carries a volatility-based
trade plan. Everything is a function of the bars it is given, so the live scan and the
walk-forward validation call the very same code, the latter on bars cut at each
rebalance date.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradingagents.scanner import indicators as ind
from tradingagents.scanner.fmt import format_text
from tradingagents.scanner.notices import Notice

MIN_BARS = 220

WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.25,
    "relative_strength": 0.20,
    "macd": 0.10,
    "rsi_guard": 0.10,
    "volume": 0.05,
}

# regime -> (score at or above which a Long is called, score at or below which a Short is)
THRESHOLDS = {
    "neutral": (35.0, -35.0),
    "risk-on": (25.0, -50.0),
    "risk-off": (50.0, -25.0),
}
CONVICTION_HIGH, CONVICTION_MEDIUM = 65.0, 50.0

STOP_ATR, TARGET_ATR = 2.0, 3.0
RISK_PER_TRADE_PCT = 1.0     # percent of capital risked if the stop is hit
MAX_POSITION_PCT = 10.0      # cap on the suggested position, percent of capital
HORIZON_BARS = 21            # also the validation horizon
VOL_FLOOR = 0.01             # annualised; keeps momentum finite on an almost flat series
RS_SCALE = 0.10              # a 10-point 3-month edge over the benchmark saturates tanh at ~0.76


@dataclass(frozen=True)
class SignalParams:
    """Bar counts that depend on the market's calendar; the defaults are the stock market's.

    Stocks trade ~21 days a month and ~252 a year; crypto trades every day (30 / 365).
    Every window that the text calls a month, 3 months or a year derives from these, so
    "3-month return" means three calendar months for a coin too. SMA50/200, RSI14, ATR14
    and MACD stay in bars: they are conventions on bars, not on calendar time.
    """

    month_bars: int = 21
    bars_per_year: int = 252

    @property
    def ret_1m(self) -> int:
        return self.month_bars

    @property
    def ret_3m(self) -> int:
        return 3 * self.month_bars

    @property
    def ret_6m(self) -> int:
        return 6 * self.month_bars

    @property
    def vol_window(self) -> int:
        return 3 * self.month_bars

    @property
    def horizon_bars(self) -> int:
        return self.month_bars

    @property
    def min_bars(self) -> int:
        return max(MIN_BARS, 6 * self.month_bars + 20)


class SkipSignal(Exception):
    """The instrument cannot be scored; ``notice`` is the translatable reason, ``str`` its English text."""

    def __init__(self, code: str, **params):
        self.notice = Notice(code, **params)
        super().__init__(str(self.notice))


@dataclass
class Signal:
    symbol: str
    direction: str                    # "Long" | "Short" | "Neutral"
    score: float                      # -100 .. +100
    conviction: str                   # "High" | "Medium" | "Low"
    components: dict[str, float]
    close: float
    last_date: str
    regime: str
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    size_pct: float | None = None
    atr: float | None = None
    rsi: float | None = None
    horizon_bars: int = HORIZON_BARS
    reason_items: list[tuple[str, dict]] = field(default_factory=list)
    metrics: dict[str, float | None] = field(default_factory=dict)

    @property
    def reasons(self) -> list[str]:
        return self.reasons_text("en")

    def reasons_text(self, lang: str = "en") -> list[str]:
        lang = lang if lang in ("en", "de") else "en"
        prefix = {"en": "Counter-signal: ", "de": "Gegensignal: "}[lang]
        return [("" if not params.get("against") else prefix)
                + format_text(_REASONS[code].get(lang, _REASONS[code]["en"]), lang, **params)
                for code, params in self.reason_items]

    def to_dict(self) -> dict:
        def num(x, digits=4):
            return None if x is None or not math.isfinite(x) else round(float(x), digits)
        return {
            "symbol": self.symbol, "direction": self.direction, "score": num(self.score, 2),
            "conviction": self.conviction, "close": num(self.close), "last_date": self.last_date,
            "regime": self.regime, "entry": num(self.entry), "stop": num(self.stop),
            "target": num(self.target), "size_pct": num(self.size_pct, 2),
            "atr": num(self.atr), "rsi": num(self.rsi, 2), "horizon_bars": self.horizon_bars,
            "components": {k: num(v) for k, v in self.components.items()},
            "reasons": self.reasons, "reasons_de": self.reasons_text("de"),
        }


_REASONS = {
    "trend_up": {"en": "Price {pct:.1f} % above its 200-day average",
                 "de": "Kurs {pct:.1f} % über dem 200-Tage-Schnitt"},
    "trend_down": {"en": "Price {pct:.1f} % below its 200-day average",
                   "de": "Kurs {pct:.1f} % unter dem 200-Tage-Schnitt"},
    "momentum": {"en": "Return {r63:+.1f} % over 3 months and {r126:+.1f} % over 6 months",
                 "de": "Rendite {r63:+.1f} % über 3 Monate und {r126:+.1f} % über 6 Monate"},
    "rs_out": {"en": "Outperformed {bench} by {pts:.1f} pts over 3 months",
               "de": "{bench} über 3 Monate um {pts:.1f} Punkte übertroffen"},
    "rs_under": {"en": "Underperformed {bench} by {pts:.1f} pts over 3 months",
                 "de": "{bench} über 3 Monate um {pts:.1f} Punkte unterboten"},
    "macd_pos_rising": {"en": "MACD histogram positive and rising",
                        "de": "MACD-Histogramm positiv und steigend"},
    "macd_pos_falling": {"en": "MACD histogram positive but fading",
                         "de": "MACD-Histogramm positiv, aber abnehmend"},
    "macd_neg_rising": {"en": "MACD histogram negative but recovering",
                        "de": "MACD-Histogramm negativ, aber erholend"},
    "macd_neg_falling": {"en": "MACD histogram negative and falling",
                         "de": "MACD-Histogramm negativ und fallend"},
    "rsi_over": {"en": "RSI {rsi:.0f}: overbought", "de": "RSI {rsi:.0f}: überkauft"},
    "rsi_under": {"en": "RSI {rsi:.0f}: oversold", "de": "RSI {rsi:.0f}: überverkauft"},
    "rsi_mid": {"en": "RSI {rsi:.0f}: neutral zone", "de": "RSI {rsi:.0f}: neutrale Zone"},
    "volume_up": {"en": "20-day volume at {pct:.0f} % of its 60-day average, on a rising price",
                  "de": "20-Tage-Volumen bei {pct:.0f} % des 60-Tage-Schnitts, bei steigendem Kurs"},
    "volume_down": {"en": "20-day volume at {pct:.0f} % of its 60-day average, on a falling price",
                    "de": "20-Tage-Volumen bei {pct:.0f} % des 60-Tage-Schnitts, bei fallendem Kurs"},
    "near_high": {"en": "{pct:.1f} % below its 52-week high",
                  "de": "{pct:.1f} % unter dem 52-Wochen-Hoch"},
    "near_low": {"en": "{pct:.1f} % above its 52-week low",
                 "de": "{pct:.1f} % über dem 52-Wochen-Tief"},
}


def _finite(x) -> bool:
    return x is not None and bool(np.isfinite(x))


def _sgn(x) -> float:
    """Sign of ``x`` as -1/0/+1; NaN and None count as 0 (no evidence either way)."""
    return float(np.sign(x)) if _finite(x) else 0.0


def _tanh(x) -> float:
    return float(np.tanh(x)) if _finite(x) else 0.0


def _last(series: pd.Series, back: int = 0) -> float | None:
    """Value ``back`` bars before the last one, None if absent or not finite."""
    if len(series) <= back:
        return None
    v = series.iloc[-1 - back]
    return float(v) if np.isfinite(v) else None


def direction_for(score: float, regime: str = "neutral") -> str:
    """Long / Short / Neutral for a score under a regime's thresholds."""
    long_at, short_at = THRESHOLDS.get(regime, THRESHOLDS["neutral"])
    if score >= long_at:
        return "Long"
    if score <= short_at:
        return "Short"
    return "Neutral"


def conviction_for(score: float) -> str:
    strength = abs(score)
    if strength >= CONVICTION_HIGH:
        return "High"
    if strength >= CONVICTION_MEDIUM:
        return "Medium"
    return "Low"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or "Close" not in df.columns:
        raise SkipSignal("no_price_data")
    frame = df.copy()
    frame = frame[frame["Close"].notna()]
    frame = frame[frame["Close"] > 0]
    # A missing high/low falls back to the close, so the ATR is then close-to-close.
    for col in ("High", "Low"):
        frame[col] = frame[col].fillna(frame["Close"]) if col in frame.columns else frame["Close"]
    if "Volume" not in frame.columns:
        frame["Volume"] = np.nan
    return frame


def compute_signal(
    symbol: str,
    df: pd.DataFrame,
    benchmark_close: pd.Series | None = None,
    regime: str = "neutral",
    benchmark: str = "SPY",
    params: SignalParams | None = None,
) -> Signal:
    """Score ``df`` (daily bars, ascending, already cut at the decision date).

    ``benchmark_close`` is the benchmark's closes; they are cut here at the
    instrument's last bar, so no later benchmark bar can enter. Passing the
    benchmark's own symbol as ``symbol`` gives it a relative strength of 0. Raises
    ``SkipSignal`` with the reason when the instrument cannot be scored.
    """
    params = params or SignalParams()
    frame = _clean(df)
    if len(frame) < params.min_bars:
        raise SkipSignal("short_history", n=len(frame), need=params.min_bars)
    close, high, low, volume = frame["Close"], frame["High"], frame["Low"], frame["Volume"]
    last_ts = frame.index[-1]
    last_close = float(close.iloc[-1])
    if not np.isfinite(last_close) or last_close <= 0:
        raise SkipSignal("no_last_close")

    sma50, sma200 = ind.sma(close, 50), ind.sma(close, 200)
    ret21, ret63, ret126 = (
        _last(ind.pct_return(close, n)) for n in (params.ret_1m, params.ret_3m, params.ret_6m))
    vol = _last(ind.realized_vol(close, params.vol_window, params.bars_per_year))
    vol = max(vol, VOL_FLOOR) if vol is not None else None
    rsi_now = _last(ind.rsi(close, 14))
    atr_now = _last(ind.atr(high, low, close, 14))
    hist = ind.macd_hist(close)
    hist_now, hist_prev = _last(hist), _last(hist, 5)
    vratio = _last(ind.volume_ratio(volume))
    d_high = _last(ind.dist_from_high(close, params.bars_per_year))
    d_low = _last(ind.dist_from_low(close, params.bars_per_year))
    s50, s200, s50_prev = _last(sma50), _last(sma200), _last(sma50, 20)

    # Benchmark 3-month return as of this instrument's last bar (no later bar enters).
    bench_ret63 = None
    if benchmark_close is not None and symbol != benchmark:
        bclose = benchmark_close[benchmark_close.index <= last_ts].dropna()
        if len(bclose) >= params.ret_3m + 1:
            bench_ret63 = _last(ind.pct_return(bclose, params.ret_3m))

    # --- components, each in [-1, +1] ---
    trend = float(np.mean([
        _sgn(last_close - s50) if s50 is not None else 0.0,
        _sgn(last_close - s200) if s200 is not None else 0.0,
        _sgn(s50 - s200) if s50 is not None and s200 is not None else 0.0,
        _sgn(s50 - s50_prev) if s50 is not None and s50_prev is not None else 0.0,
    ]))

    def mom_term(ret, bars):
        if ret is None or vol is None:
            return 0.0
        return _tanh(ret / (vol * math.sqrt(bars / params.bars_per_year)))

    momentum = (mom_term(ret126, params.ret_6m) + mom_term(ret63, params.ret_3m)) / 2

    if ret63 is not None and bench_ret63 is not None:
        rel = _tanh((ret63 - bench_ret63) / RS_SCALE)
    else:
        rel = 0.0

    macd = float(np.mean([
        _sgn(hist_now),
        _sgn(hist_now - hist_prev) if hist_now is not None and hist_prev is not None else 0.0,
    ]))

    if rsi_now is None or 30.0 <= rsi_now <= 70.0:
        rsi_guard = 0.0
    else:
        rsi_guard = float(np.clip((50.0 - rsi_now) / 30.0, -1.0, 1.0))

    if vratio is None or ret21 is None:
        volume_c = 0.0
    else:
        volume_c = _sgn(ret21) * float(np.clip(vratio - 1.0, -1.0, 1.0))

    components = {
        "trend": trend, "momentum": momentum, "relative_strength": rel,
        "macd": macd, "rsi_guard": rsi_guard, "volume": volume_c,
    }
    score = 100.0 * sum(WEIGHTS[k] * v for k, v in components.items())
    score = float(np.clip(score, -100.0, 100.0))

    direction = direction_for(score, regime)
    # A trade needs a stop, and a stop needs a usable ATR.
    if direction != "Neutral" and (atr_now is None or atr_now <= 0):
        direction = "Neutral"

    signal = Signal(
        symbol=symbol, direction=direction, score=score, conviction=conviction_for(score),
        components=components, close=last_close, last_date=last_ts.strftime("%Y-%m-%d"),
        regime=regime, atr=atr_now, rsi=rsi_now, horizon_bars=params.horizon_bars,
        metrics={"sma50": s50, "sma200": s200, "ret21": ret21, "ret63": ret63, "ret126": ret126,
                 "vol": vol, "volume_ratio": vratio, "dist_high": d_high, "dist_low": d_low,
                 "bench_ret63": bench_ret63, "macd_hist": hist_now},
    )
    if direction != "Neutral":
        _attach_plan(signal)
    signal.reason_items = _reasons(signal, benchmark, hist_now, hist_prev)
    return signal


def _attach_plan(sig: Signal) -> None:
    """Entry at the last close, stop 2 ATR against, target 3 ATR for (reward:risk 1.5).

    Size is the share of capital that loses ``RISK_PER_TRADE_PCT`` if the stop is hit
    (risk % / stop distance %), capped at ``MAX_POSITION_PCT``. A long's stop is floored
    at zero: a price cannot fall below it, and an ATR that large says the instrument is
    too volatile for this plan anyway.
    """
    side = 1.0 if sig.direction == "Long" else -1.0
    entry, atr = sig.close, float(sig.atr)
    sig.entry = entry
    sig.stop = max(entry - side * STOP_ATR * atr, 0.0)
    sig.target = max(entry + side * TARGET_ATR * atr, 0.0)
    stop_frac = STOP_ATR * atr / entry
    sig.size_pct = min(MAX_POSITION_PCT, RISK_PER_TRADE_PCT / stop_frac) if stop_frac > 0 else 0.0


def _pct(x):
    return None if x is None else x * 100


def _reasons(sig: Signal, benchmark: str, hist_now, hist_prev) -> list[tuple[str, dict]]:
    """3-5 statements backed by the computed numbers, strongest contributors first."""
    m = sig.metrics
    candidates: list[tuple[float, str, dict]] = []   # signed contribution to the score

    def add(component: str, code: str, params: dict) -> None:
        contribution = WEIGHTS[component] * sig.components[component]
        if contribution != 0:
            candidates.append((contribution, code, params))

    if m["sma200"]:
        gap = (sig.close / m["sma200"] - 1) * 100
        add("trend", "trend_up" if gap >= 0 else "trend_down", {"pct": abs(gap)})
    if m["ret63"] is not None and m["ret126"] is not None:
        add("momentum", "momentum", {"r63": _pct(m["ret63"]), "r126": _pct(m["ret126"])})
    if m["ret63"] is not None and m["bench_ret63"] is not None:
        diff = (m["ret63"] - m["bench_ret63"]) * 100
        add("relative_strength", "rs_out" if diff >= 0 else "rs_under",
            {"bench": benchmark, "pts": abs(diff)})
    if hist_now is not None and hist_prev is not None:
        code = ("macd_pos" if hist_now > 0 else "macd_neg") \
            + ("_rising" if hist_now > hist_prev else "_falling")
        add("macd", code, {})
    if sig.rsi is not None and sig.rsi > 70:
        add("rsi_guard", "rsi_over", {"rsi": sig.rsi})
    elif sig.rsi is not None and sig.rsi < 30:
        add("rsi_guard", "rsi_under", {"rsi": sig.rsi})
    if m["volume_ratio"] is not None and m["ret21"] is not None:
        add("volume", "volume_up" if m["ret21"] > 0 else "volume_down",
            {"pct": m["volume_ratio"] * 100})

    # Evidence for the score's side first, strongest first; then the one strongest
    # piece of evidence against it, flagged, so a call never hides what contradicts it.
    side = 1.0 if sig.score >= 0 else -1.0
    for_side = sorted((c for c in candidates if c[0] * side > 0), key=lambda c: -abs(c[0]))
    against = sorted((c for c in candidates if c[0] * side < 0), key=lambda c: -abs(c[0]))
    chosen = [(code, params) for _, code, params in for_side[:4]]
    # Pad to at least three with plain facts, so every call states its position in the range.
    fillers = []
    if m["dist_high"] is not None:
        fillers.append(("near_high", {"pct": abs(m["dist_high"]) * 100}))
    if m["dist_low"] is not None:
        fillers.append(("near_low", {"pct": m["dist_low"] * 100}))
    if sig.rsi is not None:
        fillers.append(("rsi_mid", {"rsi": sig.rsi}))
    for code, params in fillers:
        if len(chosen) >= 3:
            break
        if all(code != c for c, _ in chosen) and not (code == "rsi_mid" and any(
                c.startswith("rsi_") for c, _ in chosen)):
            chosen.append((code, params))
    chosen = chosen[:4]
    if against:
        chosen.append((against[0][1], {**against[0][2], "against": True}))
    return chosen
