"""Names people use for instruments (German and English) and the Yahoo symbol behind each.

Why a table and not a search: a question such as "Nvidia oder SAP auf 3 Monate?" has to
become ``NVDA`` and ``SAP.DE`` without guessing.  A wrong guess would produce an
assessment of the wrong instrument, which is worse than saying "I did not recognise it".
So only names listed here are understood, every symbol was checked to return bars on
Yahoo, and an unknown name simply stays unrecognised.

Country names ("Deutschland", "Japan") are deliberately NOT aliases: they occur in
questions that are not about an ETF at all.
"""

from __future__ import annotations

from tradingagents.scanner.universe import UNIVERSES

# (symbol, display name, aliases).  Aliases are lower case; "ü/ö" have an ASCII twin.
_ENTRIES: list[tuple[str, str, tuple[str, ...]]] = [
    # --- US large caps (universe "us")
    ("AAPL", "Apple", ("apple",)),
    ("MSFT", "Microsoft", ("microsoft",)),
    ("NVDA", "Nvidia", ("nvidia",)),
    ("AVGO", "Broadcom", ("broadcom",)),
    ("ORCL", "Oracle", ("oracle",)),
    ("CRM", "Salesforce", ("salesforce",)),
    ("AMD", "AMD", ("amd", "advanced micro devices")),
    ("GOOGL", "Alphabet (Google)", ("alphabet", "google")),
    ("META", "Meta Platforms", ("meta", "meta platforms", "facebook")),
    ("NFLX", "Netflix", ("netflix",)),
    ("DIS", "Disney", ("disney", "walt disney")),
    ("VZ", "Verizon", ("verizon",)),
    ("AMZN", "Amazon", ("amazon",)),
    ("TSLA", "Tesla", ("tesla",)),
    ("HD", "Home Depot", ("home depot",)),
    ("MCD", "McDonald's", ("mcdonald's", "mcdonalds", "mcdonald")),
    ("NKE", "Nike", ("nike",)),
    ("SBUX", "Starbucks", ("starbucks",)),
    ("LOW", "Lowe's", ("lowe's", "lowes")),
    ("PG", "Procter & Gamble", ("procter & gamble", "procter and gamble", "p&g")),
    ("KO", "Coca-Cola", ("coca-cola", "coca cola", "cocacola")),
    ("PEP", "PepsiCo", ("pepsico", "pepsi")),
    ("WMT", "Walmart", ("walmart",)),
    ("COST", "Costco", ("costco",)),
    ("PM", "Philip Morris", ("philip morris",)),
    ("LLY", "Eli Lilly", ("eli lilly", "lilly")),
    ("UNH", "UnitedHealth", ("unitedhealth", "united health")),
    ("JNJ", "Johnson & Johnson", ("johnson & johnson", "johnson and johnson")),
    ("ABBV", "AbbVie", ("abbvie",)),
    ("MRK", "Merck & Co.", ("merck", "merck & co", "merck and co")),
    ("TMO", "Thermo Fisher", ("thermo fisher",)),
    ("ABT", "Abbott", ("abbott",)),
    ("JPM", "JPMorgan", ("jpmorgan", "jp morgan", "jpmorgan chase")),
    ("BAC", "Bank of America", ("bank of america",)),
    ("WFC", "Wells Fargo", ("wells fargo",)),
    ("GS", "Goldman Sachs", ("goldman sachs", "goldman")),
    ("MS", "Morgan Stanley", ("morgan stanley",)),
    ("V", "Visa", ("visa",)),
    ("MA", "Mastercard", ("mastercard",)),
    ("BRK-B", "Berkshire Hathaway", ("berkshire hathaway", "berkshire")),
    ("CAT", "Caterpillar", ("caterpillar",)),
    ("GE", "GE Aerospace", ("ge aerospace", "general electric")),
    ("HON", "Honeywell", ("honeywell",)),
    ("UNP", "Union Pacific", ("union pacific",)),
    ("RTX", "RTX (Raytheon)", ("raytheon", "rtx corp")),
    ("UPS", "UPS", ("united parcel service",)),
    ("BA", "Boeing", ("boeing",)),
    ("LMT", "Lockheed Martin", ("lockheed martin", "lockheed")),
    ("XOM", "ExxonMobil", ("exxonmobil", "exxon mobil", "exxon")),
    ("CVX", "Chevron", ("chevron",)),
    ("COP", "ConocoPhillips", ("conocophillips", "conoco phillips")),
    ("SLB", "Schlumberger (SLB)", ("schlumberger",)),
    ("NEE", "NextEra Energy", ("nextera", "nextera energy")),
    ("DUK", "Duke Energy", ("duke energy",)),
    ("SO", "Southern Company", ("southern company",)),
    ("LIN", "Linde", ("linde",)),
    ("SHW", "Sherwin-Williams", ("sherwin-williams", "sherwin williams")),
    ("FCX", "Freeport-McMoRan", ("freeport-mcmoran", "freeport mcmoran", "freeport")),
    ("PLD", "Prologis", ("prologis",)),
    ("AMT", "American Tower", ("american tower",)),
    ("EQIX", "Equinix", ("equinix",)),
    # --- DAX (universe "dax")
    ("ADS.DE", "Adidas", ("adidas",)),
    ("AIR.DE", "Airbus", ("airbus",)),
    ("ALV.DE", "Allianz", ("allianz",)),
    ("BAS.DE", "BASF", ("basf",)),
    ("BAYN.DE", "Bayer", ("bayer",)),
    ("BEI.DE", "Beiersdorf", ("beiersdorf",)),
    ("BMW.DE", "BMW", ("bmw",)),
    ("BNR.DE", "Brenntag", ("brenntag",)),
    ("CBK.DE", "Commerzbank", ("commerzbank",)),
    ("CON.DE", "Continental", ("continental",)),
    ("DTG.DE", "Daimler Truck", ("daimler truck",)),
    ("DBK.DE", "Deutsche Bank", ("deutsche bank",)),
    ("DB1.DE", "Deutsche Börse", ("deutsche börse", "deutsche boerse")),
    ("DHL.DE", "DHL Group", ("dhl", "deutsche post")),
    ("DTE.DE", "Deutsche Telekom", ("deutsche telekom", "telekom")),
    ("EOAN.DE", "E.ON", ("e.on", "eon")),
    ("FRE.DE", "Fresenius", ("fresenius",)),
    ("FME.DE", "Fresenius Medical Care", ("fresenius medical care",)),
    ("HNR1.DE", "Hannover Rück", ("hannover rück", "hannover rueck", "hannover re")),
    ("HEI.DE", "Heidelberg Materials", ("heidelberg materials", "heidelbergcement")),
    ("HEN3.DE", "Henkel", ("henkel",)),
    ("IFX.DE", "Infineon", ("infineon",)),
    ("MBG.DE", "Mercedes-Benz", ("mercedes-benz", "mercedes benz", "mercedes")),
    ("MRK.DE", "Merck KGaA", ("merck kgaa",)),
    ("MTX.DE", "MTU Aero Engines", ("mtu aero engines", "mtu")),
    ("MUV2.DE", "Münchener Rück", ("münchener rück", "muenchener rueck", "munich re")),
    ("G24.DE", "Scout24", ("scout24",)),
    ("HOT.DE", "Hochtief", ("hochtief",)),
    ("QIA.DE", "Qiagen", ("qiagen",)),
    ("RHM.DE", "Rheinmetall", ("rheinmetall",)),
    ("RWE.DE", "RWE", ("rwe",)),
    ("SAP.DE", "SAP", ("sap",)),
    ("G1A.DE", "GEA", ("gea",)),
    ("SIE.DE", "Siemens", ("siemens",)),
    ("ENR.DE", "Siemens Energy", ("siemens energy",)),
    ("SHL.DE", "Siemens Healthineers", ("siemens healthineers", "healthineers")),
    ("SY1.DE", "Symrise", ("symrise",)),
    ("VOW3.DE", "Volkswagen", ("volkswagen", "vw")),
    ("VNA.DE", "Vonovia", ("vonovia",)),
    ("ZAL.DE", "Zalando", ("zalando",)),
    # --- sector ETFs (universe "sectors")
    ("XLK", "US-Technologiesektor (XLK)", ("tech-sektor", "technologiesektor", "tech sector")),
    ("XLF", "US-Finanzsektor (XLF)", ("finanzsektor", "financial sector")),
    ("XLE", "US-Energiesektor (XLE)", ("energiesektor", "energy sector")),
    ("XLV", "US-Gesundheitssektor (XLV)", ("gesundheitssektor", "healthcare sector")),
    ("XLI", "US-Industriesektor (XLI)", ("industriesektor", "industrial sector")),
    ("XLY", "US-Zykliker-Konsum (XLY)", ("consumer discretionary", "zyklischer konsum")),
    ("XLP", "US-Basiskonsum (XLP)", ("consumer staples", "basiskonsum")),
    ("XLU", "US-Versorger (XLU)", ("versorger", "utilities")),
    ("XLB", "US-Grundstoffe (XLB)", ("materials sector", "grundstoffsektor")),
    ("XLRE", "US-Immobiliensektor (XLRE)", ("immobiliensektor", "real estate sector")),
    ("XLC", "US-Kommunikationssektor (XLC)", ("kommunikationssektor", "communication services")),
    # --- global ETFs, indices and commodities (universe "global" and friends)
    ("SPY", "S&P 500 (SPY)", ("s&p 500", "s&p500", "sp500", "s&p", "s and p 500")),
    ("QQQ", "Nasdaq 100 (QQQ)", ("nasdaq", "nasdaq 100", "nasdaq100")),
    ("IWM", "US-Nebenwerte (IWM)", ("russell 2000", "small caps", "smallcaps")),
    ("DIA", "Dow Jones (DIA)", ("dow jones", "dow")),
    ("EFA", "Industrieländer ex US (EFA)", ("msci eafe",)),
    ("EEM", "Schwellenländer (EEM)", ("schwellenländer", "schwellenlaender",
                                       "emerging markets")),
    ("EWG", "Deutschland-ETF (EWG)", ("msci germany", "deutschland-etf")),
    ("EWJ", "Japan-ETF (EWJ)", ("msci japan", "japan-etf")),
    ("EWU", "Großbritannien-ETF (EWU)", ("msci uk", "uk-etf")),
    ("FXI", "China-ETF (FXI)", ("china-etf", "china large caps")),
    ("INDA", "Indien-ETF (INDA)", ("indien-etf", "india etf")),
    ("EWZ", "Brasilien-ETF (EWZ)", ("brasilien-etf", "brazil etf")),
    ("TLT", "US-Anleihen lang (TLT)", ("anleihen", "bonds", "staatsanleihen", "treasuries")),
    ("IEF", "US-Anleihen mittel (IEF)", ("10-year treasuries", "10-jährige anleihen")),
    ("HYG", "Hochzinsanleihen (HYG)", ("high yield", "hochzinsanleihen")),
    ("LQD", "Unternehmensanleihen (LQD)", ("unternehmensanleihen", "corporate bonds")),
    ("GLD", "Gold (GLD)", ("gold",)),
    ("SLV", "Silber (SLV)", ("silber", "silver")),
    ("USO", "Öl (USO)", ("öl", "oel", "oil", "rohöl", "crude oil")),
    ("UNG", "Erdgas (UNG)", ("erdgas", "natural gas")),
    ("DBC", "Rohstoffe (DBC)", ("rohstoffe", "commodities")),
    ("UUP", "US-Dollar-Index (UUP)", ("dollar-index", "dollar index")),
    ("^GDAXI", "DAX (Index)", ("dax", "dax 40", "dax40")),
    ("^STOXX50E", "Euro Stoxx 50", ("euro stoxx 50", "eurostoxx", "euro stoxx")),
    ("^N225", "Nikkei 225", ("nikkei", "nikkei 225")),
    # --- crypto (universe "crypto")
    ("BTC-USD", "Bitcoin", ("bitcoin", "btc")),
    ("ETH-USD", "Ethereum", ("ethereum", "ether", "eth")),
    ("BNB-USD", "BNB", ("bnb", "binance coin")),
    ("SOL-USD", "Solana", ("solana",)),
    ("XRP-USD", "XRP", ("xrp", "ripple")),
    ("ADA-USD", "Cardano", ("cardano",)),
    ("DOGE-USD", "Dogecoin", ("dogecoin", "doge")),
    ("AVAX-USD", "Avalanche", ("avalanche",)),
    ("LINK-USD", "Chainlink", ("chainlink",)),
    ("DOT-USD", "Polkadot", ("polkadot",)),
    ("LTC-USD", "Litecoin", ("litecoin",)),
    ("TRX-USD", "Tron", ("tron",)),
    # --- frequently asked about, outside the universes
    ("TTWO", "Take-Two Interactive", ("take-two", "take two", "take-two interactive", "ttwo")),
    ("SONY", "Sony", ("sony",)),
    ("GME", "GameStop", ("gamestop",)),
    ("NTDOY", "Nintendo (ADR)", ("nintendo",)),
    ("INTC", "Intel", ("intel",)),
    ("TSM", "TSMC", ("tsmc", "taiwan semiconductor")),
    ("PLTR", "Palantir", ("palantir",)),
    ("COIN", "Coinbase", ("coinbase",)),
    ("SPOT", "Spotify", ("spotify",)),
    ("PYPL", "PayPal", ("paypal",)),
]

NAMES: dict[str, str] = {}      # symbol -> display name
ALIASES: dict[str, str] = {}    # lower-case name -> symbol
for _sym, _name, _aliases in _ENTRIES:
    if _sym in NAMES:
        raise ValueError(f"duplicate symbol in the alias table: {_sym}")
    NAMES[_sym] = _name
    for _alias in _aliases:
        if ALIASES.setdefault(_alias, _sym) != _sym:
            raise ValueError(f"alias {_alias!r} maps to two symbols")

# Every member of the scanner's universes, the symbols a user may type directly.
UNIVERSE_SYMBOLS: frozenset[str] = frozenset(
    s for name, (_d, _b, members) in UNIVERSES.items() if name != "all" for s in members)
KNOWN_SYMBOLS: frozenset[str] = UNIVERSE_SYMBOLS | frozenset(NAMES)


def resolve(text: str) -> str:
    """The Yahoo symbol for a name or a symbol: a known name maps, anything else is
    upper-cased and left for the data source to accept or reject."""
    cleaned = (text or "").strip()
    return ALIASES.get(cleaned.lower(), cleaned.upper())


def display_name(symbol: str) -> str:
    """``"Nvidia (NVDA)"`` for a known symbol, the bare symbol otherwise."""
    name = NAMES.get(symbol)
    if not name or name.upper() == symbol or f"({symbol})" in name:
        return name or symbol
    return f"{name} ({symbol})"
