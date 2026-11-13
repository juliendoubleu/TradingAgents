# TradingAgents: notes for Claude sessions

## What this repository is
A multi-agent LLM trading framework plus a rule-based **market scanner**
(`tradingagents/scanner/`): per universe (US stocks, DAX, ETFs, crypto) it reads the market
regime, scores every instrument technically, and lists ideas per investment horizon (day,
swing, medium, long) with an **evidence label** that says what a walk-forward study measured.
The scanner needs no LLM. The owner reads results in German, on a phone.

## Setup and checks
- `pip install -e ".[dev]"`; Python 3.13.
- `python -m pytest -q` (not bare `pytest`); `ruff check .` must stay clean.
- `yfinance` may only be imported under `tradingagents/dataflows/`. Tests run offline and
  deterministic (inject loaders / monkeypatch the vendor); mark them `@pytest.mark.unit`.
- Numbers in scanner output go through `tradingagents/scanner/fmt.py` (German decimal commas).

## Answering the owner's market questions (the important part)
The owner asks things like "Nvidia oder SAP auf 3 Monate?", "Lohnt ein Short auf Tesla?" or an
open event question ("GTA VI kommt nächste Woche, ist am Markt etwas zu erwarten?"). Answer
**in German**, short, readable on a phone. Gather facts with the commands, then write the answer.

| Question | Run |
|---|---|
| about named instruments / a choice between them | `python -m cli.main ask "<question>" --lang de`, or `python -m cli.main assess NVDA SAP.DE --lang de` (add `--options` for implied moves) |
| what happened around past events | `python -m cli.main event-study SYMBOL --dates "2018-10-26:Red Dead Redemption 2,..." --lang de` |
| what the options market prices in | `python -m cli.main implied-move SYMBOL --lang de` |
| the picture of a whole market | `python -m cli.main scan --universe us --digest --lang de` (also `dax`, `crypto`, `global`, `sectors`) |

`ask` is deterministic (no LLM): it names the instruments it recognised and answers from the
data. If it recognises none, it says so and points here; **open event questions are answered
by you**, in this session: combine your own reasoning with the measured facts.

For an event question:
1. Reason which listed companies are plausibly affected and through which channel (publisher,
   console maker, retailers, ...), and which past events are comparable. Say that dates and
   comparisons come from your own knowledge and **verify every date** before using it.
2. Run `event-study` for those symbols and dates (an upcoming date shows only the run-up) and
   `implied-move` for the nearest expiries. Put **numbers only from command output**, with the
   number of events (n); a handful of events is a hint, not statistics.
3. Separate the effect on single stocks from the effect on the whole market, and say when an
   effect is too small to measure against normal daily swings.

Answer structure: **Kurzantwort** (one or two sentences) -> **Belege** (what the data show, with
sample sizes) -> **Einschätzung je Anlagehorizont** (Daytrade / Kurzfristig / Mittelfristig /
Langfristig, only the ones that matter) -> **Risiken und Unbekanntes**.

## Evidence labels (print `Evidence.text(lang)` verbatim, never upgrade a label)
- `validated` (belegt): measurable edge in the out-of-sample test after costs.
- `experimental` (schwach belegt): preliminary, may be chance.
- `relative` (nur relativ belegt): lagged the average, but no demonstrable absolute fall.
- `beta` (nur Markt-Beta): earns only what the market earned anyway.
- `unproven` (nicht belegt): no measurable edge. `not_studied`: no measurement.
Only `validated` / `experimental` ideas are recommendations; everything else is a **watch
list / technical picture** and must never be worded as a recommendation. The composite score has
no measured edge. A name can be an evidenced Long and technically weak at once: show both.
Shorts are not evidenced in any family so far; say what the latest study says, do not guess.
Current state is in `tradingagents/scanner/evidence.json` and `shortterm/validated.json`.

## Rules for every answer
- Never promise an outcome or a return; past measurements do not predict the future.
- Always close with "Keine Anlageberatung."
- If the data cannot answer (no instrument recognised, no history, no options), say so and show
  what to run instead; never invent a number or an assessment.

## Scheduled phone digest
The digest is produced by `python -m cli.main scan --universe <u> --digest --lang de`: it prints
only the plain-text digest (headline, blank line, body). A scheduled task's **final message must
be that digest text unchanged**: no preamble, no rewriting, no extra commentary.

## Code map
`scan.py` (scan), `horizons.py` (ideas + `Evidence`), `rules.py` (rule baskets), `assess.py`
(assessment of asked instruments), `aliases.py` + `ask.py` (question parsing and answers),
`events.py` (event study, snapshots), `report.py` / `digest.py` (reports), `cli/scanner_commands.py`
(`assess`, `ask`, `event-study`, `implied-move`).
