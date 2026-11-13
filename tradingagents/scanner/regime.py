"""The market backdrop a call is made in: risk-on, neutral or risk-off.

A long is easier to justify when the market itself is rising and broad, and a short
when it is not, so the regime moves the score thresholds (see ``signals``). It is
built from the benchmark's trend, how many of the scanned instruments sit above their
200-day average (breadth) and, when available, the VIX. Each fact scores +1 / 0 / -1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradingagents.scanner.fmt import format_text
from tradingagents.scanner.indicators import pct_return, sma

RISK_ON_AT = 3
RISK_OFF_AT = -2
MIN_BREADTH_MEMBERS = 5     # a share of fewer names than this is noise, not breadth
VIX_CALM, VIX_STRESS = 18.0, 25.0
BREADTH_STRONG, BREADTH_WEAK = 0.60, 0.40

# Facts as sentences: key -> language -> template. ``{v}`` is the fact's number.
_TEXT = {
    "bench_vs_sma200": {
        "en": "{bench} close is {v:+.1f} % vs its 200-day average",
        "de": "{bench}-Schlusskurs liegt {v:+.1f} % gegenüber dem 200-Tage-Schnitt",
    },
    "sma50_vs_sma200": {
        "en": "{bench} 50-day average is {v:+.1f} % vs its 200-day average",
        "de": "{bench}-50-Tage-Schnitt liegt {v:+.1f} % gegenüber dem 200-Tage-Schnitt",
    },
    "bench_ret63": {
        "en": "{bench} 3-month return: {v:+.1f} %",
        "de": "{bench} 3-Monats-Rendite: {v:+.1f} %",
    },
    "breadth": {
        "en": "{v:.0f} % of the scanned instruments trade above their 200-day average",
        "de": "{v:.0f} % der gescannten Werte liegen über ihrem 200-Tage-Schnitt",
    },
    "vix": {
        "en": "VIX at {v:.1f}",
        "de": "VIX bei {v:.1f}",
    },
}
_VIX_UNUSED = {
    "en": "VIX not used for this universe (it measures US stock-market fear): no points",
    "de": "VIX wird für dieses Universum nicht verwendet (er misst US-Aktienmarkt-Angst): "
          "keine Punkte",
}
LABELS = {
    "risk-on": {"en": "Risk-on", "de": "Risk-on"},
    "neutral": {"en": "Neutral", "de": "Neutral"},
    "risk-off": {"en": "Risk-off", "de": "Risk-off"},
}


@dataclass
class RegimeFact:
    key: str            # one of _TEXT
    value: float        # the number shown (percent, or VIX level)
    points: int         # -1, 0 or +1


@dataclass
class MarketRegime:
    label: str                      # "risk-on" | "neutral" | "risk-off"
    points: int
    benchmark: str
    facts: list[RegimeFact] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)   # inputs that were not available
    vix_used: bool = True       # False for universes the US VIX does not describe

    def describe(self, lang: str = "en") -> list[str]:
        """Each fact as a sentence with its points, e.g. ``... (+1)``."""
        return [
            format_text(_TEXT[f.key].get(lang, _TEXT[f.key]["en"]), lang, v=f.value,
                        bench=self.benchmark) + f" ({f.points:+d})"
            for f in self.facts
        ] + ([] if self.vix_used else [_VIX_UNUSED.get(lang, _VIX_UNUSED["en"])])

    def to_dict(self) -> dict:
        return {
            "label": self.label, "points": self.points, "benchmark": self.benchmark,
            "facts": [{"key": f.key, "value": round(f.value, 4), "points": f.points}
                      for f in self.facts],
            "missing": list(self.missing), "vix_used": self.vix_used,
        }


def breadth_above_sma200(closes: dict[str, pd.Series]) -> float | None:
    """Share (0..1) of instruments whose last close is above their 200-day average.

    Instruments with fewer than 200 bars are left out. None when fewer than
    ``MIN_BREADTH_MEMBERS`` can be judged.
    """
    flags = []
    for close in closes.values():
        if len(close) < 200:
            continue
        avg = sma(close, 200).iloc[-1]
        last = close.iloc[-1]
        if np.isfinite(avg) and np.isfinite(last):
            flags.append(last > avg)
    if len(flags) < MIN_BREADTH_MEMBERS:
        return None
    return float(np.mean(flags))


def classify(points: int) -> str:
    if points >= RISK_ON_AT:
        return "risk-on"
    if points <= RISK_OFF_AT:
        return "risk-off"
    return "neutral"


def compute_regime(
    benchmark: str,
    benchmark_close: pd.Series | None,
    breadth: float | None = None,
    vix_close: pd.Series | None = None,
    use_vix: bool = True,
) -> MarketRegime:
    """Score the backdrop from bars that are already cut at the decision date.

    Inputs that are missing or too short simply contribute no points and are listed
    in ``missing``, so a thin data day cannot fabricate a verdict. ``use_vix=False``
    (DAX, crypto) ignores the VIX entirely; the risk-on/off boundaries stay as they are.
    """
    facts: list[RegimeFact] = []
    missing: list[str] = []

    if benchmark_close is not None and len(benchmark_close) >= 200:
        close = benchmark_close.astype(float)
        avg200, avg50 = sma(close, 200).iloc[-1], sma(close, 50).iloc[-1]
        last = close.iloc[-1]
        if np.isfinite(avg200) and avg200 > 0 and np.isfinite(avg50) and np.isfinite(last):
            above = (last / avg200 - 1) * 100
            facts.append(RegimeFact("bench_vs_sma200", above, 1 if last > avg200 else -1))
            cross = (avg50 / avg200 - 1) * 100
            facts.append(RegimeFact("sma50_vs_sma200", cross, 1 if avg50 > avg200 else -1))
        else:
            missing.append("benchmark trend")
    else:
        missing.append("benchmark trend")
    if benchmark_close is not None and len(benchmark_close) >= 64:
        r63 = pct_return(benchmark_close.astype(float), 63).iloc[-1]
        if np.isfinite(r63):
            facts.append(RegimeFact("bench_ret63", r63 * 100, 1 if r63 > 0 else -1))
    else:
        missing.append("benchmark 3-month return")

    if breadth is None:
        missing.append("breadth")
    else:
        pts = 1 if breadth > BREADTH_STRONG else (-1 if breadth < BREADTH_WEAK else 0)
        facts.append(RegimeFact("breadth", breadth * 100, pts))

    if use_vix:
        vix = None
        if vix_close is not None and len(vix_close):
            vix = float(vix_close.iloc[-1])
        if vix is None or not np.isfinite(vix):
            missing.append("VIX")
        else:
            pts = 1 if vix < VIX_CALM else (-1 if vix > VIX_STRESS else 0)
            facts.append(RegimeFact("vix", vix, pts))

    points = sum(f.points for f in facts)
    return MarketRegime(classify(points), points, benchmark, facts, missing, vix_used=use_vix)
