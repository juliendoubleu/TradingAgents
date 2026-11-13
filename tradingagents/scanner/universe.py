"""Named sets of instruments to scan, each with the benchmark it is measured against.

The lists are hand-picked liquid names, not an index feed: members change, so a list
is a starting point and ``--tickers`` replaces it. Applying today's members to the
past is survivorship-biased, which the validation says out loud.
"""

from __future__ import annotations

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
    "MTX.DE", "MUV2.DE", "P911.DE", "PAH3.DE", "QIA.DE", "RHM.DE", "RWE.DE", "SAP.DE",
    "SRT3.DE", "SIE.DE", "ENR.DE", "SHL.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE",
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
    "dax": (f"The {len(_DAX)} DAX members (Xetra)", "^GDAXI", _DAX),
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
