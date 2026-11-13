"""Named sets of instruments to scan, each with the benchmark it is measured against.

The lists are hand-picked liquid names, not an index feed: members change, so a list
is a starting point and ``--tickers`` replaces it. Applying today's members to the
past is survivorship-biased, which the validation says out loud.
"""

from __future__ import annotations

from dataclasses import dataclass

_US = [
    # technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD",
    # communication services
    "GOOGL", "META", "NFLX", "DIS", "VZ",
    # consumer discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW",
    # consumer staples
    "PG", "KO", "PEP", "WMT", "COST", "PM",
    # health care
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT",
    # financials
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "BRK-B",
    # industrials
    "CAT", "GE", "HON", "UNP", "RTX", "UPS", "BA", "LMT",
    # energy
    "XOM", "CVX", "COP", "SLB",
    # utilities
    "NEE", "DUK", "SO",
    # materials
    "LIN", "SHW", "FCX",
    # real estate
    "PLD", "AMT", "EQIX",
]

_SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]

_GLOBAL = [
    # equity indices and regions
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "EWG", "EWJ", "EWU", "FXI", "INDA", "EWZ",
    # bonds
    "TLT", "IEF", "HYG", "LQD",
    # commodities and the dollar
    "GLD", "SLV", "USO", "UNG", "DBC", "UUP",
]

_DAX = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE",
    "CBK.DE", "CON.DE", "DTG.DE", "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE", "EOAN.DE",
    "FRE.DE", "FME.DE", "HNR1.DE", "HEI.DE", "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE",
    "MTX.DE", "MUV2.DE", "G24.DE", "HOT.DE", "QIA.DE", "RHM.DE", "RWE.DE", "SAP.DE",
    "G1A.DE", "SIE.DE", "ENR.DE", "SHL.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE",
]

_CRYPTO = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD", "LTC-USD", "TRX-USD",
]


def _union(*lists: list[str]) -> list[str]:
    return list(dict.fromkeys(t for items in lists for t in items))


# name -> (description, benchmark, tickers)
UNIVERSES: dict[str, tuple[str, str, list[str]]] = {
    "us": (f"{len(_US)} liquid US large caps across all 11 GICS sectors", "SPY", _US),
    "sectors": ("The 11 SPDR US sector ETFs", "SPY", _SECTORS),
    "global": ("Country, region, bond, commodity and dollar ETFs", "SPY", _GLOBAL),
    "dax": (f"DAX {len(_DAX)} members as of November 2026 (Xetra)", "^GDAXI", _DAX),
    "crypto": (f"{len(_CRYPTO)} large crypto coins vs USD", "BTC-USD", _CRYPTO),
    "all": ("US large caps + sector ETFs + global ETFs", "SPY", _union(_US, _SECTORS, _GLOBAL)),
}

DEFAULT_BENCHMARK = "SPY"


def universe_names() -> list[str]:
    return list(UNIVERSES)


def resolve_universe(
    name: str, tickers: list[str] | None = None
) -> tuple[str, str, list[str]]:
    """``(description, benchmark, members)`` for a named universe.

    ``tickers`` replaces the member list; the benchmark stays the named universe's
    (``SPY`` when the name is not one), so a custom list is judged against its market.
    Raises ``ValueError`` listing the valid names for an unknown one.
    """
    key = (name or "").strip().lower()
    if key not in UNIVERSES:
        raise ValueError(
            f"unknown universe {name!r}; choose one of: {', '.join(UNIVERSES)}"
        )
    description, benchmark, members = UNIVERSES[key]
    if tickers:
        cleaned = list(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))
        if not cleaned:
            raise ValueError("no ticker given")
        return f"Custom list ({len(cleaned)} tickers)", benchmark or DEFAULT_BENCHMARK, cleaned
    return description, benchmark, list(members)


# German descriptions, keyed like UNIVERSES (the English ones live in UNIVERSES itself).
_DESCRIPTION_DE = {
    "us": f"{len(_US)} liquide US-Standardwerte aus allen 11 GICS-Sektoren",
    "sectors": "Die 11 SPDR-Sektor-ETFs der USA",
    "global": "Länder-, Regionen-, Anleihen-, Rohstoff- und Dollar-ETFs",
    "dax": f"DAX-{len(_DAX)}-Mitglieder, Stand November 2026 (Xetra)",
    "crypto": f"{len(_CRYPTO)} große Kryptowährungen gegen USD",
    "all": "US-Standardwerte + Sektor-ETFs + globale ETFs",
}
_FAMILY = {"us": "stocks_us", "dax": "stocks_dax", "sectors": "etfs", "global": "etfs",
           "crypto": "crypto", "all": "mixed"}
# The US VIX says nothing about DAX stocks or coins, so those universes do not use it.
_USE_VIX = {"us", "sectors", "global", "all"}


@dataclass(frozen=True)
class UniverseInfo:
    """What the scanner must know about a universe beyond its member list.

    Crypto trades every calendar day, so its bars, "1 month" and annualisation differ
    from stocks (``month_bars``, ``bars_per_year``); the VIX only describes US markets
    (``use_vix``). ``family`` groups universes for those decisions.
    """

    name: str
    description: dict[str, str]
    benchmark: str
    members: list[str]
    family: str
    bars_per_year: int
    month_bars: int
    use_vix: bool


def universe_info(name: str, tickers: list[str] | None = None) -> UniverseInfo:
    """Metadata for a named universe; ``tickers`` replaces the members as in ``resolve_universe``."""
    description, benchmark, members = resolve_universe(name, tickers)
    key = name.strip().lower()
    if tickers:
        description_map = {"en": description,
                           "de": f"Eigene Liste ({len(members)} Ticker)"}
    else:
        description_map = {"en": description, "de": _DESCRIPTION_DE[key]}
    crypto = _FAMILY[key] == "crypto"
    return UniverseInfo(
        name=key, description=description_map, benchmark=benchmark, members=members,
        family=_FAMILY[key], bars_per_year=365 if crypto else 252,
        month_bars=30 if crypto else 21, use_vix=key in _USE_VIX,
    )
