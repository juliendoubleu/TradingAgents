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
- Numbers in scanner output go through `tradingagents/scanner/fmt.py` (`fmt_num`, `format_text`:
  German decimal commas are produced where the sentence is built, never patched afterwards).

## Answering the owner's market questions (the important part)
The owner asks things like "Nvidia oder SAP auf 3 Monate?", "Lohnt ein Short auf Tesla?" or an
open event question ("GTA VI kommt nächste Woche, ist am Markt etwas zu erwarten?"). Answer
**in German**, short, readable on a phone. Gather facts with the commands, then write the answer.

| Question | Run |
|---|---|
| about named instruments / a choice between them | `python -m cli.main ask "<question>" --lang de`, or `python -m cli.main assess NVDA SAP.DE --lang de` (add `--options` for implied moves) |
| what happened around past events | `python -m cli.main event-study SYMBOL --dates "2018-10-26:Red Dead Redemption 2,..." --lang de` |
| what the options market prices in | `python -m cli.main implied-move SYMBOL --lang de` |
| the picture of a whole market | `python -m cli.main scan --universe us --digest --lang de` (also `dax`, `crypto`, `global`, `sectors`; several: `--universe us,dax,global,crypto`) |

All of these are also listed in `tradingagents --help` (the same app as `python -m cli.main`).

`ask` is deterministic (no LLM): it names the instruments it recognised and answers from the
data. The studies measure only about 1 month (medium) and about 6 months (long); for another
period (say 3 months) `ask` shows both and says that the asked period was not tested: repeat that
gap in your answer. If it recognises none, it says so and points here; **open event questions are answered
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
- `validated` (Belegt): passed the out-of-sample test after costs (median over all phases of the plain t of non-overlapping
  dates) and survives removing the three best-performing names.
- `experimental` (Schwach belegt): preliminary, may be chance.
- `relative` (Nur relativ schwächer): lagged the average, but no demonstrable absolute fall.
- `beta` (Nur Marktrendite): earns only what the market earned anyway.
- `unproven` (Nicht belegt): no measurable edge. `not_studied` (Nicht untersucht): no measurement (a custom ticker
  list is `not_studied` unless it is essentially the whole named universe).
Only `validated` / `experimental` ideas are recommendations; everything else is a **watch
list / technical picture** and must never be worded as a recommendation. The composite score has
no measured edge. A name can be an evidenced Long and technically weak at once: show both
(`Idea.conflict`). A caveat on the evidence (`Evidence.caveat`: the result hangs on a few names)
and a stale flag (study older than 120 days) are part of the sentence: never drop them.
Medium/long-term ideas have **no stop** (the study measured a time exit; a stop destroyed the
edge) and are sized equal-weight in the measured basket: never invent a stop for them.
**State of the evidence (study of 2026-11-13, corrected after an independent review): nothing is
validated or weakly supported at any horizon, no short is evidenced, so a scan says "Keine
belegte Empfehlung" and shows the technical picture as a labelled watch list.** The numbers are
in `tradingagents/scanner/evidence.json` (medium/long; refresh with
`python -m tradingagents.scanner.horizon_research --out DIR --cache DIR --update-package`) and
`tradingagents/scanner/shortterm/validated.json` (day/swing; refresh with
`python -m tradingagents.scanner.shortterm.research --out DIR --cache DIR --update-package`).
Do not tune a rule to rescue a result: the measurement decides. The names above live in one
place (`notices.LABEL_NAMES`); say them exactly so. The medium/long verdicts judge the median over
all phases of the non-overlapping t (a fixed phase had decided some cells; the rule was fixed after
an independent review showed it, the thresholds did not move): after the re-run 0 validated, 0
weakly supported, 0 relatively weaker, 15 "Nur Marktrendite", 129 "Nicht belegt". Watch items are
worded "technisch stark" / "technisch schwach", never as Long/Short calls.

## Rules for every answer
- Never promise an outcome or a return; past measurements do not predict the future.
- Always close with "Keine Anlageberatung."
- If the data cannot answer (no instrument recognised, no history, no options), say so and show
  what to run instead; never invent a number or an assessment.

## Scheduled phone digest
The digest is produced by `python -m cli.main scan --universe us,dax,global,crypto --digest --lang de --no-save`
(one or several universes; `--tz` sets the zone of the title clock and weekend flag, default
`Europe/Berlin`, any IANA name): it prints only the plain-text digest (headline of at most 180
characters, blank line, body). A scheduled task's **final message must
be that digest text unchanged**: no preamble, no rewriting, no extra commentary. If the command
exits non-zero (no market could be scanned) its stdout is one line ("Markt-Scan fehlgeschlagen:
Kursdaten nicht abrufbar – kein Digest"): the final message is that one line, unchanged.

## Code map
`scan.py` (scan), `universe.py` (universes, evidence family of a list), `horizons.py` (ideas +
`Evidence`), `rules.py` (rule baskets), `horizon_research.py` (the medium/long study, writes
`evidence.json` and the German report), `shortterm/` (day/swing study), `assess.py` (assessment of
asked instruments), `aliases.py` + `ask.py` (question parsing and answers), `events.py` (event
study, snapshots), `report.py` / `digest.py` (reports), `notices.py` (translatable messages),
`fmt.py` (numbers), `cli/main.py` (`scan`) and `cli/scanner_commands.py` (`assess`, `ask`,
`event-study`, `implied-move`, mounted with `register(app)`).
