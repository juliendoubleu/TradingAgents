# Mittel- und Langfrist: Was ist wirklich belegt?

Studie vom 2026-11-13. Alle Zahlen stammen aus einem echten Testlauf auf Yahoo-Kursdaten. Verworfene Regeln und Varianten stehen in den Tabellen mit drin.

**Dieser Bericht wurde nach einer unabhängigen Prüfung korrigiert.** Die erste Fassung stufte genau eine Zelle als "belegt" ein (US-Aktien, Momentum 12-1, langfristig, Long). Die Prüfung reproduzierte jede Zahl, zeigte aber, dass dieser Befund nicht trägt: er hing an drei Aktien, die mit Rückschau in der Liste stehen, sein t-Wert ignorierte die Überlappung der Fenster, und der mitgelieferte Stop zerstörte den gemessenen Vorteil. Beurteilt wird jetzt (1) mit dem Median des einfachen t auf nicht überlappenden Terminen über alle Phasen (siehe Abschnitt Phasenabhängigkeit), (2) nur mit Körben von im Schnitt mindestens drei Werten und (3) nur, wenn das Ergebnis auch ohne die drei Werte mit der höchsten Gesamtrendite besteht. Unten steht die Gegenüberstellung vorher / nachher.

## Kurzfassung

Einstufungen der 144 Regel-Seiten-Zellen (Mehrfachnennungen derselben Messung inbegriffen): belegt 0, schwach belegt 0, nur relativ schwächer 0, nur Marktrendite 15, nicht belegt 129.

**Keine der getesteten Regeln hat die korrigierte Prüfung bestanden.** Das ist das Ergebnis, kein Fehler: Die Anlageideen des Programms für Monate und Halbjahre sind deshalb als Beobachtungsliste gekennzeichnet, nicht als Empfehlung.

Nur **Marktrendite** (Gewinn in absoluten Zahlen, aber nicht mehr als der Durchschnitt aller Werte verdient hat): 15 Fälle.

## Korrektur nach der unabhängigen Prüfung

Gegenüberstellung aller Zellen, deren Urteil sich geändert hat:

| Zelle | Urteil vorher | Urteil nachher | t vorher (Newey-West: IS / OOS / gepoolt) | t nachher (einfach, nicht überlappend: IS / OOS / gepoolt) | Robustheitsprüfung |
|---|---|---|---|---|---|
| US-Aktien, T1, mittel (1 Monat), Long | nicht belegt | nur Marktrendite | 1,5 / 1,3 / 2,0 | 1,3 / 0,9 / 1,5 | - |
| US-Aktien, T1, lang (6 Monate), Long | belegt | nicht belegt | 2,2 / 2,2 / 3,0 | 1,5 / 1,4 / 2,0 | ohne NVDA, AVGO, AMD: gepooltes t = 1,4 (wäre durchgefallen) |
| US-Aktien, V1, lang (6 Monate), Long | belegt | nicht belegt | 2,1 / 2,7 / 3,2 | 1,8 / 1,7 / 2,4 | ohne NVDA, AVGO, AMD: gepooltes t = 1,6 (wäre durchgefallen) |
| ETFs (Welt + Sektoren), T1, lang (6 Monate), Long | nur Marktrendite | nicht belegt | 0,7 / 0,1 / 0,7 | 0,5 / -0,1 / 0,5 | - |
| ETFs (Welt + Sektoren), T2, lang (6 Monate), Long | nur Marktrendite | nicht belegt | -0,2 / -0,9 / -0,7 | -0,4 / -0,1 / -0,4 | - |
| ETFs (Welt + Sektoren), V1, lang (6 Monate), Long | nur Marktrendite | nicht belegt | 0,9 / -0,3 / 0,7 | 0,8 / -0,4 / 0,4 | - |
| ETFs (Welt + Sektoren), V4, lang (6 Monate), Long | nur Marktrendite | nicht belegt | -0,2 / -1,3 / -0,8 | -0,2 / 0,0 / -0,8 | - |
| ETFs (Welt + Sektoren), V5, mittel (1 Monat), Short | nur relativ schwächer | nicht belegt | 0,3 / 0,1 / 0,3 | 0,1 / 0,5 / 0,2 | - |

Was geändert wurde und warum:

- **Drei Aktien tragen den Befund.** Die Listen enthalten die heutigen Mitglieder; ein Momentum-Verfahren kauft verlässlich jene, die später zu den größten Werten wurden. Ohne NVDA, AVGO und AMD fiel das gepoolte Newey-West-t der US-Zelle von 2,97 auf 1,84, ohne die fünf besten auf 1,20 (Zahlen der Prüfung); NVDA allein lieferte 36 % des Bruttovorsprungs, NVDA, AMD, AVGO und TSLA zusammen 85 %. In den Jahren 2018 und 2022 lag der Vorsprung bei etwa -200 bzw. -266 Basispunkten. Jetzt gilt für jede Zelle, die belegt oder schwach belegt wäre: die Reihe wird ohne die 3 Werte mit der höchsten Gesamtrendite im Datenfenster der Familie neu berechnet (vorab festgelegt, k = 3, fest). Besteht das gepoolte t dann nicht mehr mit mindestens 2,0, wird das Urteil um eine Stufe herabgesetzt und trägt eine Warnung mit den Namen.
- **Das Newey-West-t war zu optimistisch.** Bei Haltedauern, die ein Mehrfaches des Terminabstands sind, setzt die verwendete Verzögerung die Überlappung nicht voll an (bei sechs überlappenden Fenstern ein Varianzfaktor von 4,06 statt 6, das t ist um etwa das 1,22-fache zu hoch). In einer Monte-Carlo-Prüfung ohne echten Effekt erreichten 7,2 % (lang, 94 Termine), 9,6 % (lang, 44 Termine) und 5,0 % (mittel) ein t von mindestens 2 – statt der nominalen 2,3 %. Beurteilt wird deshalb jetzt mit dem **einfachen t der nicht überlappenden Termine** (jeder k-te Termin; maßgeblich ist der **Median über alle k möglichen Phasen**, siehe Abschnitt Phasenabhängigkeit), getrennt für In-Sample, Out-of-Sample und gepoolt. Das Newey-West-t steht weiter in den Daten, als optimistisch gekennzeichnet; zusätzlich stehen im Datensatz die t-Werte aller Phasen (Startversätze).
- **Das Urteil hing an einer willkürlichen Phase.** Die erste korrigierte Fassung beurteilte nur eine feste Phase der nicht überlappenden Termine (die am letzten Termin endende). Eine weitere unabhängige Prüfung zeigte, dass sich bei derselben Zelle (US-Aktien, Momentum, lang, Long) das t je nach Phase stark ändert und die gewählte Phase die mit dem niedrigsten In-Sample-t war. Diese Regel wurde deshalb nach jener Prüfung festgelegt: beurteilt wird der **Median des einfachen t über alle Phasen**, für In-Sample, Out-of-Sample, gepoolt und ebenso in der Robustheitsprüfung; die Schwellen blieben unverändert. Der Mittelwert, der neben dem t genannt wird, stammt aus derselben Phase (bzw. dem Mittel der beiden mittleren Phasen).
- **Der Stop zerstörte den Vorteil.** Im eigenen Stop-Modus der Studie (siehe unten) wurde aus +597 Basispunkten außerhalb der Stichprobe für die frühere US-Zelle ein Minus. Für die Zeit-Ausstiegs-Regeln (mittel, lang) schlägt das Programm deshalb keinen Stop mehr vor, und die Positionsgröße ist gleichgewichtet im gemessenen Korb statt aus einem Stopabstand abgeleitet.
- **Mindestgröße des Korbs.** Ein Urteil über "nicht belegt" hinaus verlangt einen durchschnittlichen Korb von mindestens 3 Werten; die Short-Seite der Variante V5 bei ETFs stützte sich im Schnitt auf etwa einen ETF je Korb.
- **Verschiedene Prüfungen.** Die Varianten V3 und T3 haben dieselbe Long-Seite (8 identische Zellen). Gezählt werden deshalb verschiedene Messungen, nicht Zellen (136 verschiedene von 144 Zellen). Eine Variante bekommt nie das Siegel "stark".
- **Letzter abgeschlossener Kurs.** Der Download fand während einer Börsensitzung statt und enthielt Teilkurse des Tages. Die Studie schneidet jede Reihe beim letzten Kurs, der zum Zeitpunkt des Downloads abgeschlossen war (der Zeitpunkt wird mit dem Zwischenspeicher abgelegt).
- **Rückgang auf Tageskursen.** Die Risikosicht maß den größten Rückgang nur an den Halbjahrespunkten (etwa -6 % statt -10 %; auf Tageskursen desselben Pfads -30,7 % statt -34,5 %). Er wird jetzt auf täglichen Kursen gemessen.

## Was wurde getestet

Regeln, für die es in der Literatur einen Grund gibt, und die jeweils ohne Blick in die Zukunft berechnet werden (Signal am Schlusskurs von Tag t, Einstieg zur Eröffnung von t+1, Ausstieg nach einem Monat bzw. sechs Monaten):

| Kürzel | Regel | Beschreibung | Horizont | Art |
|---|---|---|---|---|
| T0 | Technisches Gesamtbild | Der Gesamtscore des Scanners (Trend, Momentum, relative Stärke, MACD, RSI, Volumen) mit den Regime-Schwellen. | mittel (1 Monat) | vorab festgelegt |
| T1 | Momentum 12-1 | Rendite von vor 12 bis vor 1 Monat; Long = oberste 20 % des Universums, Short = unterste 20 %. | mittel (1 Monat), lang (6 Monate) | vorab festgelegt |
| T2 | Trendzustand | Long, wenn der Kurs über dem 200-Tage-Schnitt und der 50-Tage-Schnitt darüber liegt; Short im Spiegelbild. | mittel (1 Monat), lang (6 Monate) | vorab festgelegt |
| T3 | Momentum mit Trend- und Marktfilter | Long = oberste 30 % nach 12-1-Momentum und über dem 200-Tage-Schnitt; Short = unterste 30 %, darunter, und der Vergleichsindex unter seinem eigenen 200-Tage-Schnitt. | mittel (1 Monat), lang (6 Monate) | vorab festgelegt |
| T4 | 52-Wochen-Hoch | Long, wenn der Schlusskurs höchstens 5 % unter dem 52-Wochen-Schlusshoch liegt; Short, wenn er mindestens 30 % darunter und unter dem 200-Tage-Schnitt liegt. | mittel (1 Monat) | vorab festgelegt |
| T5 | Rücksetzer im Trend | Long = die 20 % schwächsten 1-Monats-Renditen unter den Werten über dem 200-Tage-Schnitt; Short = die 20 % stärksten unter den Werten darunter. | mittel (1 Monat) | vorab festgelegt |
| V1 | Momentum 12-1, 10 % Ränder | Wie T1, aber nur die obersten und untersten 10 % des Universums. | mittel (1 Monat), lang (6 Monate) | Variante |
| V2 | Momentum 6-1 | Wie T1, aber die Rendite von vor 6 bis vor 1 Monat. | mittel (1 Monat), lang (6 Monate) | Variante |
| V3 | Momentum mit Trendfilter, ohne Marktfilter | Wie T3, aber die Short-Seite verlangt nicht mehr, dass der Index unter seinem 200-Tage-Schnitt liegt. | mittel (1 Monat), lang (6 Monate) | Variante |
| V4 | Schnellerer Trendzustand (20/100) | Wie T2 mit 20- und 100-Tage-Schnitt statt 50 und 200. | mittel (1 Monat), lang (6 Monate) | Variante |
| V5 | Rücksetzer, 10 % Ränder | Wie T5, aber die 10 % schwächsten/stärksten 1-Monats-Renditen. | mittel (1 Monat) | Variante |

Vergleichsmaßstab ist immer der Durchschnitt **aller** Werte des Universums im selben Zeitraum ("Drift"): Wer nur mit dem steigenden Markt mitschwimmt, hat keinen Vorteil gezeigt. Gerechnet wird nach Kosten (10 Basispunkte je Hin- und Rückweg bei Aktien/ETFs, 30 bei Krypto); der Test wird zusätzlich mit doppelten Kosten geprüft.

## Phasenabhängigkeit des Urteils

Bei Halbjahres- und Monatsfenstern überlappen sich benachbarte Termine; das einfache t nimmt deshalb nur jeden k-ten Termin. Welcher Termin der erste ist (die Phase), ist willkürlich, und das t kann sich dadurch stark ändern. Beurteilt wird darum der Median des t über alle k Phasen (In-Sample, Out-of-Sample und gepoolt); diese Regel wurde festgelegt, nachdem eine unabhängige Prüfung die Phasenabhängigkeit gezeigt hatte, und die Schwellen blieben unverändert. Unten die Zellen, die in mindestens einer einzelnen Phase "schwach belegt" oder besser erreicht hätten (vor der Robustheitsprüfung; reine Transparenz, keine Beurteilung):

| Zelle | Phasen | IS-t je Phase | OOS-t je Phase | Median IS / OOS | Urteil (Median) |
|---|---|---|---|---|---|
| US-Aktien, T1, lang (6 Monate), Long | in 2 von 6 Phasen wäre die Zelle schwach belegt oder besser | 2,6/1,4/1,0/0,7/1,5/2,5 | 1,4/1,3/1,3/1,5/1,4/1,9 | 1,5 / 1,4 | nicht belegt |
| US-Aktien, T3, lang (6 Monate), Long | in 1 von 6 Phasen wäre die Zelle schwach belegt oder besser | 1,5/0,5/-0,0/1,0/1,0/2,1 | 0,7/0,7/1,3/1,2/1,1/1,6 | 1,0 / 1,2 | nicht belegt |
| US-Aktien, V1, mittel (1 Monat), Long | in 1 von 5 Phasen wäre die Zelle schwach belegt oder besser | 1,3/0,9/2,6/0,9/1,7 | 1,4/1,0/1,4/2,4/1,0 | 1,3 / 1,4 | nicht belegt |
| US-Aktien, V1, lang (6 Monate), Long | in 1 von 6 Phasen wäre die Zelle schwach belegt oder besser | 2,1/1,7/1,9/0,9/1,3/2,0 | 1,9/1,7/1,7/1,7/1,5/2,0 | 1,8 / 1,7 | nicht belegt |
| US-Aktien, V2, lang (6 Monate), Long | in 1 von 6 Phasen wäre die Zelle schwach belegt oder besser | 1,4/0,2/1,7/0,1/0,9/2,5 | 0,9/1,4/1,8/1,1/0,8/2,2 | 1,1 / 1,3 | nicht belegt |
| US-Aktien, V3, lang (6 Monate), Long | in 1 von 6 Phasen wäre die Zelle schwach belegt oder besser | 1,5/0,5/-0,0/1,0/1,0/2,1 | 0,7/0,7/1,3/1,2/1,1/1,6 | 1,0 / 1,2 | nicht belegt |
| ETFs (Welt + Sektoren), V2, lang (6 Monate), Long | in 1 von 6 Phasen wäre die Zelle schwach belegt oder besser | 1,1/0,8/-1,8/-0,5/2,3/2,4 | -1,4/-0,8/1,4/1,6/1,1/0,3 | 1,0 / 0,7 | nur Marktrendite |

## Daten

| Familie | Werte | Kursdaten | Entscheidungstage | Termine mittel / lang | Testzeitraum (OOS, mittel) |
|---|---|---|---|---|---|
| US-Aktien | 61 | 2013-05-20 bis 2026-11-12 | 2014-05-20 bis 2026-10-08 | 624 / 144 | 2023-01-17 bis 2026-10-08 |
| DAX-Aktien | 40 | 2013-05-20 bis 2026-11-12 | 2014-05-20 bis 2026-10-12 | 631 / 146 | 2023-01-23 bis 2026-10-12 |
| ETFs (Welt + Sektoren) | 33 | 2013-05-20 bis 2026-11-12 | 2014-05-20 bis 2026-10-08 | 624 / 144 | 2023-01-17 bis 2026-10-08 |
| Kryptowährungen | 12 | 2014-09-17 bis 2026-11-12 | 2018-11-08 bis 2026-10-08 | 414 / 92 | 2024-05-23 bis 2026-10-08 |

Aufteilung: die ersten 70 % der Entscheidungstage dienen zum Auswählen (In-Sample, IS), die letzten 30 % werden genau einmal zum Prüfen verwendet (Out-of-Sample, OOS). IS-Tage, deren Haltefenster in den OOS-Zeitraum hineinreicht, werden entfernt (Purge). Die Kursdaten enden beim letzten Tag, der zum Zeitpunkt des Downloads abgeschlossen war.

## Ergebnisse je Familie

Spalten: Termine (n) und mittlere Rendite je Termin (alle, auch überlappende Termine) in Basispunkten (1 bp = 0,01 %) der **beurteilten Reihe** (Long: Vorsprung vor dem Durchschnitt aller Werte; Short: tatsächlicher Gewinn des Shorts), jeweils nach Kosten. t = Median des einfachen t der nicht überlappenden Termine über alle Phasen (so wird beurteilt; |t| ≥ 2 gilt als auffällig), "NW-t" = Newey-West-t über alle Termine (nur zur Information, optimistisch), "n ohne" = Zahl der nicht überlappenden Termine im OOS-Teil.

### US-Aktien

| Regel | Horizont | Seite | Ø Korb | IS n | IS bp | IS t | OOS n | OOS bp | OOS t | n ohne | OOS NW-t | Urteil |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 Technisches Gesamtbild | mittel (1 Monat) | Long | 28,6 | 430 | -11 | -0,4 | 188 | -55 | -1,5 | 38 | -2,1 | nur Marktrendite |
| T0 Technisches Gesamtbild | mittel (1 Monat) | Short | 9,1 | 404 | -142 | -1,8 | 185 | -211 | -2,8 | 37 | -4,1 | nicht belegt |
| T1 Momentum 12-1 | mittel (1 Monat) | Long | 13,0 | 432 | 33 | 1,3 | 188 | 56 | 0,9 | 38 | 1,3 | nur Marktrendite |
| T1 Momentum 12-1 | mittel (1 Monat) | Short | 13,0 | 432 | -156 | -2,3 | 188 | -192 | -2,9 | 38 | -3,9 | nicht belegt |
| T1 Momentum 12-1 | lang (6 Monate) | Long | 13,0 | 94 | 269 | 1,5 | 44 | 597 | 1,4 | 7 | 2,2 | nicht belegt |
| T1 Momentum 12-1 | lang (6 Monate) | Short | 13,0 | 94 | -1.045 | -2,6 | 44 | -1.264 | -2,5 | 7 | -3,6 | nicht belegt |
| T2 Trendzustand | mittel (1 Monat) | Long | 37,9 | 432 | -3 | -0,2 | 188 | -37 | -1,7 | 38 | -2,8 | nur Marktrendite |
| T2 Trendzustand | mittel (1 Monat) | Short | 13,9 | 432 | -124 | -1,9 | 188 | -175 | -2,7 | 38 | -4,0 | nicht belegt |
| T2 Trendzustand | lang (6 Monate) | Long | 39,0 | 94 | 45 | -0,5 | 44 | -41 | -0,2 | 7 | -0,3 | nur Marktrendite |
| T2 Trendzustand | lang (6 Monate) | Short | 13,0 | 94 | -982 | -2,3 | 44 | -934 | -2,5 | 7 | -3,9 | nicht belegt |
| T3 Momentum mit Trend- und Marktfilter | mittel (1 Monat) | Long | 17,2 | 432 | 19 | 0,8 | 188 | 13 | 0,4 | 38 | 0,4 | nur Marktrendite |
| T3 Momentum mit Trend- und Marktfilter | mittel (1 Monat) | Short | 16,7 | 85 | -304 | -1,5 | 13 | -253 | -0,2 | 3 | -1,1 | nicht belegt |
| T3 Momentum mit Trend- und Marktfilter | lang (6 Monate) | Long | 17,3 | 94 | 268 | 1,0 | 44 | 363 | 1,2 | 7 | 1,5 | nicht belegt |
| T3 Momentum mit Trend- und Marktfilter | lang (6 Monate) | Short | 16,9 | 13 | -1.858 | -6,0 | 7 | -2.421 | 0,0 | 1 | -7,4 | nicht belegt |
| T4 52-Wochen-Hoch | mittel (1 Monat) | Long | 25,3 | 427 | -31 | -1,2 | 188 | -76 | -1,7 | 38 | -2,8 | nur Marktrendite |
| T4 52-Wochen-Hoch | mittel (1 Monat) | Short | 4,6 | 289 | -74 | -0,4 | 174 | -272 | -1,7 | 35 | -2,3 | nicht belegt |
| T5 Rücksetzer im Trend | mittel (1 Monat) | Long | 8,9 | 431 | 20 | 0,5 | 188 | -24 | 0,1 | 38 | -0,7 | nur Marktrendite |
| T5 Rücksetzer im Trend | mittel (1 Monat) | Short | 4,3 | 404 | -135 | -1,6 | 188 | -234 | -2,5 | 38 | -3,6 | nicht belegt |
| V1 Momentum 12-1, 10 % Ränder | mittel (1 Monat) | Long | 7,0 | 432 | 66 | 1,3 | 188 | 127 | 1,4 | 38 | 1,8 | nicht belegt |
| V1 Momentum 12-1, 10 % Ränder | mittel (1 Monat) | Short | 7,0 | 432 | -166 | -2,0 | 188 | -215 | -2,6 | 38 | -3,5 | nicht belegt |
| V1 Momentum 12-1, 10 % Ränder | lang (6 Monate) | Long | 7,0 | 94 | 493 | 1,8 | 44 | 1.018 | 1,7 | 7 | 2,7 | nicht belegt |
| V1 Momentum 12-1, 10 % Ränder | lang (6 Monate) | Short | 7,0 | 94 | -1.106 | -2,3 | 44 | -1.588 | -2,6 | 7 | -3,5 | nicht belegt |
| V2 Momentum 6-1 | mittel (1 Monat) | Long | 13,0 | 432 | 22 | 0,9 | 188 | 1 | 0,1 | 38 | 0,0 | nur Marktrendite |
| V2 Momentum 6-1 | mittel (1 Monat) | Short | 13,0 | 432 | -156 | -2,3 | 188 | -207 | -3,0 | 38 | -4,2 | nicht belegt |
| V2 Momentum 6-1 | lang (6 Monate) | Long | 13,0 | 94 | 217 | 1,1 | 44 | 408 | 1,3 | 7 | 2,0 | nicht belegt |
| V2 Momentum 6-1 | lang (6 Monate) | Short | 13,0 | 94 | -1.028 | -3,3 | 44 | -1.132 | -2,4 | 7 | -3,5 | nicht belegt |
| V3 Momentum mit Trendfilter, ohne Marktfilter | mittel (1 Monat) | Long | 17,2 | 432 | 19 | 0,8 | 188 | 13 | 0,4 | 38 | 0,4 | nur Marktrendite |
| V3 Momentum mit Trendfilter, ohne Marktfilter | mittel (1 Monat) | Short | 10,7 | 432 | -125 | -1,5 | 188 | -166 | -2,3 | 38 | -3,5 | nicht belegt |
| V3 Momentum mit Trendfilter, ohne Marktfilter | lang (6 Monate) | Long | 17,3 | 94 | 268 | 1,0 | 44 | 363 | 1,2 | 7 | 1,5 | nicht belegt |
| V3 Momentum mit Trendfilter, ohne Marktfilter | lang (6 Monate) | Short | 10,3 | 94 | -1.019 | -2,4 | 44 | -1.121 | -2,7 | 7 | -3,7 | nicht belegt |
| V4 Schnellerer Trendzustand (20/100) | mittel (1 Monat) | Long | 35,0 | 432 | -14 | -0,7 | 188 | -64 | -1,9 | 38 | -2,5 | nur Marktrendite |
| V4 Schnellerer Trendzustand (20/100) | mittel (1 Monat) | Short | 17,7 | 432 | -136 | -2,0 | 188 | -192 | -2,8 | 38 | -4,0 | nicht belegt |
| V4 Schnellerer Trendzustand (20/100) | lang (6 Monate) | Long | 36,1 | 94 | -42 | -0,8 | 44 | -65 | -0,0 | 7 | -0,6 | nur Marktrendite |
| V4 Schnellerer Trendzustand (20/100) | lang (6 Monate) | Short | 16,7 | 94 | -943 | -2,6 | 44 | -1.121 | -2,9 | 7 | -5,3 | nicht belegt |
| V5 Rücksetzer, 10 % Ränder | mittel (1 Monat) | Long | 4,7 | 431 | 29 | 0,7 | 188 | 11 | 0,0 | 38 | 0,3 | nur Marktrendite |
| V5 Rücksetzer, 10 % Ränder | mittel (1 Monat) | Short | 2,4 | 404 | -138 | -1,3 | 188 | -219 | -1,7 | 38 | -2,9 | nicht belegt (Korb zu klein) |

### DAX-Aktien

| Regel | Horizont | Seite | Ø Korb | IS n | IS bp | IS t | OOS n | OOS bp | OOS t | n ohne | OOS NW-t | Urteil |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 Technisches Gesamtbild | mittel (1 Monat) | Long | 15,5 | 429 | -11 | -0,5 | 190 | 1 | 0,4 | 38 | 0,0 | nicht belegt |
| T0 Technisches Gesamtbild | mittel (1 Monat) | Short | 8,3 | 400 | -104 | -0,9 | 185 | -94 | -1,3 | 37 | -1,9 | nicht belegt |
| T1 Momentum 12-1 | mittel (1 Monat) | Long | 7,9 | 437 | -11 | -0,4 | 190 | 153 | 2,7 | 38 | 3,3 | nicht belegt |
| T1 Momentum 12-1 | mittel (1 Monat) | Short | 7,9 | 437 | -75 | -0,8 | 190 | -92 | -1,3 | 38 | -1,5 | nicht belegt |
| T1 Momentum 12-1 | lang (6 Monate) | Long | 7,9 | 96 | 39 | 0,2 | 44 | 995 | 2,2 | 7 | 2,6 | nicht belegt |
| T1 Momentum 12-1 | lang (6 Monate) | Short | 7,9 | 96 | -430 | -1,3 | 44 | -684 | -2,0 | 7 | -3,0 | nicht belegt |
| T2 Trendzustand | mittel (1 Monat) | Long | 20,7 | 437 | -22 | -0,9 | 190 | 32 | 1,4 | 38 | 1,9 | nicht belegt |
| T2 Trendzustand | mittel (1 Monat) | Short | 11,4 | 436 | -67 | -0,8 | 190 | -86 | -1,0 | 38 | -1,6 | nicht belegt |
| T2 Trendzustand | lang (6 Monate) | Long | 21,2 | 96 | 33 | 0,3 | 44 | 277 | 2,3 | 7 | 2,7 | nicht belegt |
| T2 Trendzustand | lang (6 Monate) | Short | 10,7 | 95 | -337 | -0,7 | 44 | -536 | -1,5 | 7 | -2,3 | nicht belegt |
| T3 Momentum mit Trend- und Marktfilter | mittel (1 Monat) | Long | 10,3 | 433 | -24 | -0,7 | 190 | 104 | 2,7 | 38 | 3,2 | nicht belegt |
| T3 Momentum mit Trend- und Marktfilter | mittel (1 Monat) | Short | 10,7 | 168 | -125 | -1,2 | 14 | -263 | -2,1 | 3 | -2,5 | nicht belegt |
| T3 Momentum mit Trend- und Marktfilter | lang (6 Monate) | Long | 10,4 | 95 | 61 | 0,3 | 44 | 671 | 2,0 | 7 | 2,8 | nicht belegt |
| T3 Momentum mit Trend- und Marktfilter | lang (6 Monate) | Short | 10,4 | 34 | -267 | -0,4 | 3 | -1.723 | 0,0 | 0 | -2,8 | nicht belegt |
| T4 52-Wochen-Hoch | mittel (1 Monat) | Long | 12,7 | 421 | -10 | -0,1 | 190 | 15 | 0,8 | 38 | 0,7 | nicht belegt |
| T4 52-Wochen-Hoch | mittel (1 Monat) | Short | 4,5 | 309 | -109 | -1,0 | 177 | -49 | 0,1 | 35 | -0,5 | nicht belegt |
| T5 Rücksetzer im Trend | mittel (1 Monat) | Long | 5,2 | 431 | -28 | -0,8 | 190 | 31 | 0,1 | 38 | 0,9 | nicht belegt |
| T5 Rücksetzer im Trend | mittel (1 Monat) | Short | 3,5 | 395 | -108 | -1,3 | 185 | -108 | -1,0 | 37 | -1,6 | nicht belegt |
| V1 Momentum 12-1, 10 % Ränder | mittel (1 Monat) | Long | 4,0 | 437 | -9 | -0,2 | 190 | 184 | 2,1 | 38 | 2,6 | nicht belegt |
| V1 Momentum 12-1, 10 % Ränder | mittel (1 Monat) | Short | 4,0 | 437 | -92 | -1,0 | 190 | -128 | -1,4 | 38 | -1,8 | nicht belegt |
| V1 Momentum 12-1, 10 % Ränder | lang (6 Monate) | Long | 4,0 | 96 | 168 | 0,8 | 44 | 1.503 | 2,2 | 7 | 2,9 | nicht belegt |
| V1 Momentum 12-1, 10 % Ränder | lang (6 Monate) | Short | 4,0 | 96 | -405 | -0,9 | 44 | -1.127 | -3,1 | 7 | -3,7 | nicht belegt |
| V2 Momentum 6-1 | mittel (1 Monat) | Long | 7,9 | 437 | -9 | -0,3 | 190 | 97 | 1,7 | 38 | 2,4 | nicht belegt |
| V2 Momentum 6-1 | mittel (1 Monat) | Short | 7,9 | 437 | -89 | -1,2 | 190 | -82 | -1,3 | 38 | -1,6 | nicht belegt |
| V2 Momentum 6-1 | lang (6 Monate) | Long | 7,9 | 96 | 61 | 0,4 | 44 | 576 | 1,6 | 7 | 2,0 | nicht belegt |
| V2 Momentum 6-1 | lang (6 Monate) | Short | 7,9 | 96 | -416 | -1,1 | 44 | -711 | -1,6 | 7 | -2,9 | nicht belegt |
| V3 Momentum mit Trendfilter, ohne Marktfilter | mittel (1 Monat) | Long | 10,3 | 433 | -24 | -0,7 | 190 | 104 | 2,7 | 38 | 3,2 | nicht belegt |
| V3 Momentum mit Trendfilter, ohne Marktfilter | mittel (1 Monat) | Short | 7,7 | 437 | -52 | -0,8 | 190 | -108 | -1,1 | 38 | -1,9 | nicht belegt |
| V3 Momentum mit Trendfilter, ohne Marktfilter | lang (6 Monate) | Long | 10,4 | 95 | 61 | 0,3 | 44 | 671 | 2,0 | 7 | 2,8 | nicht belegt |
| V3 Momentum mit Trendfilter, ohne Marktfilter | lang (6 Monate) | Short | 7,5 | 95 | -334 | -0,8 | 44 | -481 | -1,2 | 7 | -1,8 | nicht belegt |
| V4 Schnellerer Trendzustand (20/100) | mittel (1 Monat) | Long | 20,2 | 437 | -16 | -0,3 | 190 | 2 | 0,4 | 38 | 0,1 | nicht belegt |
| V4 Schnellerer Trendzustand (20/100) | mittel (1 Monat) | Short | 13,2 | 430 | -72 | -1,2 | 190 | -108 | -1,5 | 38 | -2,2 | nicht belegt |
| V4 Schnellerer Trendzustand (20/100) | lang (6 Monate) | Long | 20,4 | 96 | 69 | 0,6 | 44 | 211 | 1,7 | 7 | 2,6 | nicht belegt |
| V4 Schnellerer Trendzustand (20/100) | lang (6 Monate) | Short | 12,7 | 94 | -395 | -1,1 | 44 | -640 | -1,7 | 7 | -3,2 | nicht belegt |
| V5 Rücksetzer, 10 % Ränder | mittel (1 Monat) | Long | 2,8 | 431 | -4 | 0,2 | 190 | 17 | 0,2 | 38 | 0,3 | nicht belegt (Korb zu klein) |
| V5 Rücksetzer, 10 % Ränder | mittel (1 Monat) | Short | 2,0 | 395 | -84 | -0,9 | 185 | -85 | -1,1 | 37 | -1,1 | nicht belegt (Korb zu klein) |

### ETFs (Welt + Sektoren)

| Regel | Horizont | Seite | Ø Korb | IS n | IS bp | IS t | OOS n | OOS bp | OOS t | n ohne | OOS NW-t | Urteil |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 Technisches Gesamtbild | mittel (1 Monat) | Long | 14,7 | 422 | -28 | -0,8 | 188 | -42 | -1,7 | 38 | -2,5 | nicht belegt |
| T0 Technisches Gesamtbild | mittel (1 Monat) | Short | 5,9 | 382 | -25 | -0,6 | 171 | -183 | -1,4 | 34 | -2,5 | nicht belegt |
| T1 Momentum 12-1 | mittel (1 Monat) | Long | 7,0 | 432 | -3 | -0,4 | 188 | 18 | 0,3 | 38 | 0,5 | nicht belegt |
| T1 Momentum 12-1 | mittel (1 Monat) | Short | 7,0 | 432 | -23 | -0,4 | 188 | -122 | -2,4 | 38 | -3,7 | nicht belegt |
| T1 Momentum 12-1 | lang (6 Monate) | Long | 7,0 | 94 | 101 | 0,5 | 44 | 10 | -0,1 | 7 | 0,1 | nicht belegt |
| T1 Momentum 12-1 | lang (6 Monate) | Short | 7,0 | 94 | -172 | -0,4 | 44 | -593 | -2,5 | 7 | -3,9 | nicht belegt |
| T2 Trendzustand | mittel (1 Monat) | Long | 19,7 | 431 | -13 | -0,8 | 188 | -24 | -2,0 | 38 | -2,8 | nicht belegt |
| T2 Trendzustand | mittel (1 Monat) | Short | 8,4 | 422 | 7 | 0,3 | 188 | -132 | -1,3 | 38 | -1,8 | nicht belegt |
| T2 Trendzustand | lang (6 Monate) | Long | 20,1 | 94 | -17 | -0,4 | 44 | -99 | -0,1 | 7 | -0,9 | nicht belegt |
| T2 Trendzustand | lang (6 Monate) | Short | 7,9 | 93 | -9 | -0,1 | 44 | -505 | -1,1 | 7 | -3,5 | nicht belegt |
| T3 Momentum mit Trend- und Marktfilter | mittel (1 Monat) | Long | 8,7 | 431 | -8 | -0,2 | 188 | 9 | 0,1 | 38 | 0,3 | nicht belegt |
| T3 Momentum mit Trend- und Marktfilter | mittel (1 Monat) | Short | 9,1 | 85 | -112 | -0,8 | 13 | -167 | -0,6 | 3 | -1,5 | nicht belegt |
| T3 Momentum mit Trend- und Marktfilter | lang (6 Monate) | Long | 8,7 | 94 | 52 | 0,3 | 44 | 45 | 0,6 | 7 | 0,3 | nur Marktrendite |
| T3 Momentum mit Trend- und Marktfilter | lang (6 Monate) | Short | 9,1 | 13 | -950 | -2,4 | 7 | -968 | 0,0 | 1 | -3,3 | nicht belegt |
| T4 52-Wochen-Hoch | mittel (1 Monat) | Long | 16,6 | 432 | -26 | -0,9 | 188 | -41 | -1,6 | 38 | -3,0 | nicht belegt |
| T4 52-Wochen-Hoch | mittel (1 Monat) | Short | 2,2 | 289 | -35 | -0,4 | 164 | 118 | 0,3 | 33 | 0,8 | nicht belegt (Korb zu klein) |
| T5 Rücksetzer im Trend | mittel (1 Monat) | Long | 4,9 | 415 | -3 | -0,3 | 188 | -47 | -1,5 | 38 | -2,0 | nicht belegt |
| T5 Rücksetzer im Trend | mittel (1 Monat) | Short | 2,9 | 349 | 2 | 0,1 | 140 | -62 | -0,8 | 28 | -0,9 | nicht belegt (Korb zu klein) |
| V1 Momentum 12-1, 10 % Ränder | mittel (1 Monat) | Long | 4,0 | 432 | 1 | 0,1 | 188 | 35 | 0,4 | 38 | 0,7 | nicht belegt |
| V1 Momentum 12-1, 10 % Ränder | mittel (1 Monat) | Short | 4,0 | 432 | 9 | 0,3 | 188 | -120 | -1,9 | 38 | -2,4 | nicht belegt |
| V1 Momentum 12-1, 10 % Ränder | lang (6 Monate) | Long | 4,0 | 94 | 222 | 0,8 | 44 | -83 | -0,4 | 7 | -0,3 | nicht belegt |
| V1 Momentum 12-1, 10 % Ränder | lang (6 Monate) | Short | 4,0 | 94 | 51 | 0,2 | 44 | -456 | -1,5 | 7 | -2,4 | nicht belegt |
| V2 Momentum 6-1 | mittel (1 Monat) | Long | 7,0 | 432 | -16 | -0,5 | 188 | -14 | -0,2 | 38 | -0,4 | nicht belegt |
| V2 Momentum 6-1 | mittel (1 Monat) | Short | 7,0 | 432 | -28 | -0,5 | 188 | -131 | -2,9 | 38 | -3,7 | nicht belegt |
| V2 Momentum 6-1 | lang (6 Monate) | Long | 7,0 | 94 | 104 | 1,0 | 44 | 118 | 0,7 | 7 | 0,7 | nur Marktrendite |
| V2 Momentum 6-1 | lang (6 Monate) | Short | 7,0 | 94 | -81 | -0,2 | 44 | -527 | -2,1 | 7 | -4,9 | nicht belegt |
| V3 Momentum mit Trendfilter, ohne Marktfilter | mittel (1 Monat) | Long | 8,7 | 431 | -8 | -0,2 | 188 | 9 | 0,1 | 38 | 0,3 | nicht belegt |
| V3 Momentum mit Trendfilter, ohne Marktfilter | mittel (1 Monat) | Short | 6,0 | 430 | -10 | -0,4 | 188 | -86 | -1,0 | 38 | -1,7 | nicht belegt |
| V3 Momentum mit Trendfilter, ohne Marktfilter | lang (6 Monate) | Long | 8,7 | 94 | 52 | 0,3 | 44 | 45 | 0,6 | 7 | 0,3 | nur Marktrendite |
| V3 Momentum mit Trendfilter, ohne Marktfilter | lang (6 Monate) | Short | 5,8 | 93 | -78 | -0,3 | 44 | -525 | -1,7 | 7 | -3,0 | nicht belegt |
| V4 Schnellerer Trendzustand (20/100) | mittel (1 Monat) | Long | 18,8 | 432 | -31 | -0,9 | 188 | -42 | -1,6 | 38 | -2,2 | nicht belegt |
| V4 Schnellerer Trendzustand (20/100) | mittel (1 Monat) | Short | 9,6 | 425 | -39 | -0,8 | 188 | -117 | -1,9 | 38 | -3,0 | nicht belegt |
| V4 Schnellerer Trendzustand (20/100) | lang (6 Monate) | Long | 19,6 | 94 | -15 | -0,2 | 44 | -91 | 0,0 | 7 | -1,3 | nicht belegt |
| V4 Schnellerer Trendzustand (20/100) | lang (6 Monate) | Short | 9,1 | 90 | -28 | 0,1 | 44 | -609 | -2,6 | 7 | -6,4 | nicht belegt |
| V5 Rücksetzer, 10 % Ränder | mittel (1 Monat) | Long | 2,6 | 415 | 5 | 0,3 | 188 | -79 | -1,3 | 38 | -2,2 | nicht belegt (Korb zu klein) |
| V5 Rücksetzer, 10 % Ränder | mittel (1 Monat) | Short | 1,7 | 349 | 12 | 0,1 | 140 | 13 | 0,5 | 28 | 0,1 | nicht belegt (Korb zu klein) |

### Kryptowährungen

| Regel | Horizont | Seite | Ø Korb | IS n | IS bp | IS t | OOS n | OOS bp | OOS t | n ohne | OOS NW-t | Urteil |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 Technisches Gesamtbild | mittel (1 Monat) | Long | 4,6 | 203 | 67 | 0,2 | 83 | -370 | -1,1 | 17 | -1,6 | nicht belegt |
| T0 Technisches Gesamtbild | mittel (1 Monat) | Short | 5,8 | 165 | -953 | -1,6 | 95 | -426 | -0,6 | 19 | -0,9 | nicht belegt |
| T1 Momentum 12-1 | mittel (1 Monat) | Long | 2,6 | 285 | 247 | 0,7 | 125 | -223 | -0,9 | 25 | -1,1 | nicht belegt (Korb zu klein) |
| T1 Momentum 12-1 | mittel (1 Monat) | Short | 2,6 | 285 | -1.366 | -2,1 | 125 | -279 | -0,5 | 25 | -0,5 | nicht belegt (Korb zu klein) |
| T1 Momentum 12-1 | lang (6 Monate) | Long | 2,6 | 58 | -1.850 | -0,2 | 28 | -1.031 | -0,8 | 5 | -1,1 | nicht belegt (Korb zu klein) |
| T1 Momentum 12-1 | lang (6 Monate) | Short | 2,6 | 58 | -17.679 | -1,1 | 28 | -1.402 | -0,3 | 5 | -0,6 | nicht belegt (Korb zu klein) |
| T2 Trendzustand | mittel (1 Monat) | Long | 5,5 | 236 | -68 | -1,0 | 109 | -218 | -1,1 | 22 | -1,3 | nicht belegt |
| T2 Trendzustand | mittel (1 Monat) | Short | 6,3 | 215 | -987 | -2,2 | 104 | -505 | -0,9 | 21 | -1,2 | nicht belegt |
| T2 Trendzustand | lang (6 Monate) | Long | 5,9 | 45 | -3.791 | -0,1 | 23 | -517 | -0,5 | 4 | -1,0 | nicht belegt |
| T2 Trendzustand | lang (6 Monate) | Short | 6,2 | 44 | -31.373 | -1,0 | 22 | -1.433 | -0,4 | 4 | -0,7 | nicht belegt |
| T3 Momentum mit Trend- und Marktfilter | mittel (1 Monat) | Long | 2,6 | 242 | 239 | 0,4 | 105 | -310 | -0,8 | 21 | -1,2 | nicht belegt (Korb zu klein) |
| T3 Momentum mit Trend- und Marktfilter | mittel (1 Monat) | Short | 3,5 | 123 | -802 | -1,5 | 58 | 132 | 0,7 | 12 | 0,4 | nicht belegt |
| T3 Momentum mit Trend- und Marktfilter | lang (6 Monate) | Long | 2,7 | 47 | -3.273 | -0,7 | 21 | -864 | -0,8 | 4 | -1,3 | nicht belegt (Korb zu klein) |
| T3 Momentum mit Trend- und Marktfilter | lang (6 Monate) | Short | 3,5 | 25 | -2.216 | -0,5 | 11 | -3.963 | -0,6 | 2 | -1,1 | nicht belegt |
| T4 52-Wochen-Hoch | mittel (1 Monat) | Long | 2,4 | 89 | -581 | -0,9 | 37 | -414 | -0,3 | 7 | -0,7 | nicht belegt (Korb zu klein) |
| T4 52-Wochen-Hoch | mittel (1 Monat) | Short | 6,5 | 233 | -714 | -2,0 | 108 | -440 | -1,0 | 22 | -1,1 | nicht belegt |
| T5 Rücksetzer im Trend | mittel (1 Monat) | Long | 2,2 | 166 | -769 | -2,4 | 53 | -502 | -1,1 | 11 | -1,5 | nicht belegt (Korb zu klein) |
| T5 Rücksetzer im Trend | mittel (1 Monat) | Short | 2,3 | 147 | -651 | -1,5 | 89 | -354 | -0,5 | 18 | -1,0 | nicht belegt (Korb zu klein) |
| V1 Momentum 12-1, 10 % Ränder | mittel (1 Monat) | Long | 1,6 | 285 | -164 | -0,9 | 125 | -257 | -0,9 | 25 | -1,1 | nicht belegt (Korb zu klein) |
| V1 Momentum 12-1, 10 % Ränder | mittel (1 Monat) | Short | 1,6 | 285 | -1.419 | -1,6 | 125 | -175 | -0,3 | 25 | -0,4 | nicht belegt (Korb zu klein) |
| V1 Momentum 12-1, 10 % Ränder | lang (6 Monate) | Long | 1,6 | 58 | -5.526 | -0,4 | 28 | -1.130 | -0,7 | 5 | -1,2 | nicht belegt (Korb zu klein) |
| V1 Momentum 12-1, 10 % Ränder | lang (6 Monate) | Short | 1,6 | 58 | -12.971 | -1,1 | 28 | -988 | -0,4 | 5 | -0,4 | nicht belegt (Korb zu klein) |
| V2 Momentum 6-1 | mittel (1 Monat) | Long | 2,7 | 285 | -24 | -0,0 | 125 | -36 | -0,3 | 25 | -0,4 | nicht belegt (Korb zu klein) |
| V2 Momentum 6-1 | mittel (1 Monat) | Short | 2,7 | 285 | -1.503 | -2,2 | 125 | -312 | -0,6 | 25 | -0,6 | nicht belegt (Korb zu klein) |
| V2 Momentum 6-1 | lang (6 Monate) | Long | 2,7 | 58 | -671 | -0,0 | 28 | 345 | 0,3 | 5 | 0,8 | nicht belegt (Korb zu klein) |
| V2 Momentum 6-1 | lang (6 Monate) | Short | 2,7 | 58 | -26.430 | -1,1 | 28 | -903 | -0,3 | 5 | -0,4 | nicht belegt (Korb zu klein) |
| V3 Momentum mit Trendfilter, ohne Marktfilter | mittel (1 Monat) | Long | 2,6 | 242 | 239 | 0,4 | 105 | -310 | -0,8 | 21 | -1,2 | nicht belegt (Korb zu klein) |
| V3 Momentum mit Trendfilter, ohne Marktfilter | mittel (1 Monat) | Short | 3,0 | 217 | -724 | -1,9 | 108 | -335 | -0,7 | 22 | -0,8 | nicht belegt (Korb zu klein) |
| V3 Momentum mit Trendfilter, ohne Marktfilter | lang (6 Monate) | Long | 2,7 | 47 | -3.273 | -0,7 | 21 | -864 | -0,8 | 4 | -1,3 | nicht belegt (Korb zu klein) |
| V3 Momentum mit Trendfilter, ohne Marktfilter | lang (6 Monate) | Short | 3,1 | 44 | -28.663 | -1,0 | 22 | -2.280 | -0,6 | 4 | -0,8 | nicht belegt |
| V4 Schnellerer Trendzustand (20/100) | mittel (1 Monat) | Long | 6,2 | 233 | -76 | -1,1 | 93 | -203 | -0,8 | 19 | -1,3 | nicht belegt |
| V4 Schnellerer Trendzustand (20/100) | mittel (1 Monat) | Short | 6,9 | 206 | -1.031 | -2,0 | 95 | -709 | -0,8 | 19 | -1,2 | nicht belegt |
| V4 Schnellerer Trendzustand (20/100) | lang (6 Monate) | Long | 6,0 | 47 | -2.492 | -0,3 | 23 | -178 | -0,5 | 4 | -0,4 | nicht belegt |
| V4 Schnellerer Trendzustand (20/100) | lang (6 Monate) | Short | 6,9 | 43 | -22.088 | -1,2 | 21 | -2.687 | -0,8 | 4 | -1,5 | nicht belegt |
| V5 Rücksetzer, 10 % Ränder | mittel (1 Monat) | Long | 1,3 | 166 | -639 | -1,6 | 53 | -483 | -1,0 | 11 | -1,6 | nicht belegt (Korb zu klein) |
| V5 Rücksetzer, 10 % Ränder | mittel (1 Monat) | Short | 1,4 | 147 | -686 | -1,3 | 89 | -492 | -0,7 | 18 | -1,3 | nicht belegt (Korb zu klein) |

## Robustheit: ohne die drei Gewinner

Für jede Zelle, die belegt oder schwach belegt ist, und zur Information auch für jede, die es nach der früheren Regel (Newey-West-t) war, wurde die beurteilte Reihe ohne die 3 Werte mit der höchsten Gesamtrendite neu berechnet (Universum, Rangschwellen und Körbe auf der gekürzten Liste). Das Urteil bleibt nur bei einem gepoolten t von mindestens 2,0.

| Zelle | Entfernt | Gepoolt vorher | Gepoolt ohne sie | t ohne sie (IS / OOS) | Urteil vor Prüfung | Urteil danach |
|---|---|---|---|---|---|---|
| US-Aktien, T1, lang (6 Monate), Long | NVDA, AVGO, AMD | 374 bp / t 2,0 (Newey-West 3,0) | 215 bp / t 1,4 (Newey-West 1,8) | IS 0,7, OOS 1,2 | belegt (frühere Regel) | nicht belegt |
| US-Aktien, V1, lang (6 Monate), Long | NVDA, AVGO, AMD | 661 bp / t 2,4 (Newey-West 3,2) | 336 bp / t 1,6 (Newey-West 2,0) | IS 0,9, OOS 1,1 | belegt (frühere Regel) | nicht belegt |

Die Zellen mit dem Vermerk "frühere Regel" waren schon an der korrigierten Statistik gescheitert (getrenntes t im In-Sample- bzw. Out-of-Sample-Teil unter der Schwelle); die Robustheitsprüfung hätte sie für sich allein nur herabgesetzt, wenn das gepoolte t unter 2 fällt. Mit dem einfachen t bleibt es oberhalb (Spalte "Gepoolt ohne sie"), mit dem Newey-West-t der Prüfung darunter oder knapp darauf: das gepoolte t allein ist ein schwächeres Kriterium als die getrennte Prüfung beider Abschnitte. Ohne die drei Aktien bleibt im Auswahlzeitraum kaum ein Vorsprung (Spalte "t ohne sie").

## Stop-Modus (warum die Ideen keinen Stop tragen)

Dieselben Regeln wurden zusätzlich mit einem Schutz-Stop gerechnet (2,5 bzw. 4,0 x ATR(14), Lücken füllen zur Eröffnung). Gezeigt ist die beurteilte Reihe für alle Zellen, die ohne Robustheitsprüfung belegt oder schwach belegt gewesen wären:

| Zelle | ohne Stop: IS / OOS bp | t | mit Stop: IS / OOS bp | t |
|---|---|---|---|---|
| US-Aktien, T1, lang (6 Monate), Long | 269 / 597 | 1,5 / 1,4 | -186 / -193 | -0,5 / -0,2 |
| US-Aktien, V1, lang (6 Monate), Long | 493 / 1.018 | 1,8 / 1,7 | -60 / 191 | 0,2 / 0,3 |

## Risikosicht (Long-Seite)

Nicht überlappende Termine, der Korb wird gehalten (Bargeld, wenn leer), gegen den Durchschnitt aller Werte und den Vergleichsindex über dieselben Fenster. Eine Trendregel muss nicht mehr verdienen, kann aber Verluste begrenzen; deshalb stehen Rendite und größter Rückgang nebeneinander (Gesamtzeitraum). Der größte Rückgang ist auf **täglichen** Kursen innerhalb der Haltefenster gemessen.

### US-Aktien

| Regel | Horizont | Anteil investiert | Rendite p. a. (Korb / alle / Index) | Sharpe (Korb / alle / Index) | Max. Rückgang auf Tageskursen (Korb / alle / Index) |
|---|---|---|---|---|---|
| T0 | mittel (1 Monat) | 100 % | 8,4 % / 12,4 % / 8,8 % | 0,61 / 0,90 / 0,66 | -37 % / -32 % / -31 % |
| T1 | mittel (1 Monat) | 100 % | 16,2 % / 12,4 % / 8,8 % | 0,93 / 0,90 / 0,66 | -29 % / -32 % / -31 % |
| T1 | lang (6 Monate) | 100 % | 27,4 % / 18,8 % / 14,0 % | 1,78 / 1,67 / 1,21 | -31 % / -34 % / -34 % |
| T2 | mittel (1 Monat) | 100 % | 10,8 % / 12,4 % / 8,8 % | 0,83 / 0,90 / 0,66 | -30 % / -32 % / -31 % |
| T2 | lang (6 Monate) | 100 % | 18,3 % / 18,8 % / 14,0 % | 1,46 / 1,67 / 1,21 | -33 % / -34 % / -34 % |
| T3 | mittel (1 Monat) | 100 % | 12,5 % / 12,4 % / 8,8 % | 0,82 / 0,90 / 0,66 | -30 % / -32 % / -31 % |
| T3 | lang (6 Monate) | 100 % | 24,7 % / 18,8 % / 14,0 % | 1,76 / 1,67 / 1,21 | -32 % / -34 % / -34 % |
| T4 | mittel (1 Monat) | 99 % | 6,9 % / 12,4 % / 8,8 % | 0,59 / 0,90 / 0,66 | -30 % / -32 % / -31 % |
| T5 | mittel (1 Monat) | 100 % | 14,5 % / 12,4 % / 8,8 % | 0,88 / 0,90 / 0,66 | -37 % / -32 % / -31 % |
| V1 | mittel (1 Monat) | 100 % | 18,3 % / 12,4 % / 8,8 % | 0,91 / 0,90 / 0,66 | -34 % / -32 % / -31 % |
| V1 | lang (6 Monate) | 100 % | 32,8 % / 18,8 % / 14,0 % | 1,54 / 1,67 / 1,21 | -32 % / -34 % / -34 % |
| V2 | mittel (1 Monat) | 100 % | 13,3 % / 12,4 % / 8,8 % | 0,79 / 0,90 / 0,66 | -34 % / -32 % / -31 % |
| V2 | lang (6 Monate) | 100 % | 26,1 % / 18,8 % / 14,0 % | 1,51 / 1,67 / 1,21 | -34 % / -34 % / -34 % |
| V3 | mittel (1 Monat) | 100 % | 12,5 % / 12,4 % / 8,8 % | 0,82 / 0,90 / 0,66 | -30 % / -32 % / -31 % |
| V3 | lang (6 Monate) | 100 % | 24,7 % / 18,8 % / 14,0 % | 1,76 / 1,67 / 1,21 | -32 % / -34 % / -34 % |
| V4 | mittel (1 Monat) | 100 % | 8,2 % / 12,4 % / 8,8 % | 0,66 / 0,90 / 0,66 | -30 % / -32 % / -31 % |
| V4 | lang (6 Monate) | 100 % | 19,1 % / 18,8 % / 14,0 % | 1,59 / 1,67 / 1,21 | -34 % / -34 % / -34 % |
| V5 | mittel (1 Monat) | 100 % | 17,3 % / 12,4 % / 8,8 % | 0,87 / 0,90 / 0,66 | -38 % / -32 % / -31 % |

### DAX-Aktien

| Regel | Horizont | Anteil investiert | Rendite p. a. (Korb / alle / Index) | Sharpe (Korb / alle / Index) | Max. Rückgang auf Tageskursen (Korb / alle / Index) |
|---|---|---|---|---|---|
| T0 | mittel (1 Monat) | 99 % | 6,3 % / 7,6 % / 4,9 % | 0,48 / 0,57 / 0,39 | -35 % / -31 % / -37 % |
| T1 | mittel (1 Monat) | 100 % | 11,3 % / 7,6 % / 4,9 % | 0,70 / 0,57 / 0,39 | -30 % / -31 % / -37 % |
| T1 | lang (6 Monate) | 100 % | 17,7 % / 11,9 % / 8,2 % | 0,88 / 0,77 / 0,54 | -43 % / -39 % / -39 % |
| T2 | mittel (1 Monat) | 100 % | 5,7 % / 7,6 % / 4,9 % | 0,47 / 0,57 / 0,39 | -33 % / -31 % / -37 % |
| T2 | lang (6 Monate) | 100 % | 14,0 % / 11,9 % / 8,2 % | 0,87 / 0,77 / 0,54 | -39 % / -39 % / -39 % |
| T3 | mittel (1 Monat) | 99 % | 7,1 % / 7,6 % / 4,9 % | 0,52 / 0,57 / 0,39 | -29 % / -31 % / -37 % |
| T3 | lang (6 Monate) | 100 % | 16,8 % / 11,9 % / 8,2 % | 0,94 / 0,77 / 0,54 | -42 % / -39 % / -39 % |
| T4 | mittel (1 Monat) | 99 % | 6,6 % / 7,6 % / 4,9 % | 0,47 / 0,57 / 0,39 | -27 % / -31 % / -37 % |
| T5 | mittel (1 Monat) | 99 % | 4,4 % / 7,6 % / 4,9 % | 0,34 / 0,57 / 0,39 | -34 % / -31 % / -37 % |
| V1 | mittel (1 Monat) | 100 % | 8,9 % / 7,6 % / 4,9 % | 0,53 / 0,57 / 0,39 | -36 % / -31 % / -37 % |
| V1 | lang (6 Monate) | 100 % | 23,1 % / 11,9 % / 8,2 % | 0,90 / 0,77 / 0,54 | -39 % / -39 % / -39 % |
| V2 | mittel (1 Monat) | 100 % | 8,9 % / 7,6 % / 4,9 % | 0,59 / 0,57 / 0,39 | -35 % / -31 % / -37 % |
| V2 | lang (6 Monate) | 100 % | 16,8 % / 11,9 % / 8,2 % | 0,86 / 0,77 / 0,54 | -40 % / -39 % / -39 % |
| V3 | mittel (1 Monat) | 99 % | 7,1 % / 7,6 % / 4,9 % | 0,52 / 0,57 / 0,39 | -29 % / -31 % / -37 % |
| V3 | lang (6 Monate) | 100 % | 16,8 % / 11,9 % / 8,2 % | 0,94 / 0,77 / 0,54 | -42 % / -39 % / -39 % |
| V4 | mittel (1 Monat) | 100 % | 7,1 % / 7,6 % / 4,9 % | 0,55 / 0,57 / 0,39 | -32 % / -31 % / -37 % |
| V4 | lang (6 Monate) | 100 % | 14,0 % / 11,9 % / 8,2 % | 0,89 / 0,77 / 0,54 | -41 % / -39 % / -39 % |
| V5 | mittel (1 Monat) | 99 % | 4,3 % / 7,6 % / 4,9 % | 0,32 / 0,57 / 0,39 | -37 % / -31 % / -37 % |

### ETFs (Welt + Sektoren)

| Regel | Horizont | Anteil investiert | Rendite p. a. (Korb / alle / Index) | Sharpe (Korb / alle / Index) | Max. Rückgang auf Tageskursen (Korb / alle / Index) |
|---|---|---|---|---|---|
| T0 | mittel (1 Monat) | 98 % | 1,9 % / 4,5 % / 8,8 % | 0,24 / 0,44 / 0,66 | -28 % / -28 % / -31 % |
| T1 | mittel (1 Monat) | 100 % | 3,1 % / 4,5 % / 8,8 % | 0,28 / 0,44 / 0,66 | -39 % / -28 % / -31 % |
| T1 | lang (6 Monate) | 100 % | 10,6 % / 7,9 % / 14,0 % | 0,99 / 0,83 / 1,21 | -27 % / -29 % / -34 % |
| T2 | mittel (1 Monat) | 100 % | 2,8 % / 4,5 % / 8,8 % | 0,31 / 0,44 / 0,66 | -27 % / -28 % / -31 % |
| T2 | lang (6 Monate) | 100 % | 6,8 % / 7,9 % / 14,0 % | 0,78 / 0,83 / 1,21 | -26 % / -29 % / -34 % |
| T3 | mittel (1 Monat) | 100 % | 4,5 % / 4,5 % / 8,8 % | 0,40 / 0,44 / 0,66 | -27 % / -28 % / -31 % |
| T3 | lang (6 Monate) | 100 % | 11,7 % / 7,9 % / 14,0 % | 1,23 / 0,83 / 1,21 | -24 % / -29 % / -34 % |
| T4 | mittel (1 Monat) | 100 % | 1,7 % / 4,5 % / 8,8 % | 0,22 / 0,44 / 0,66 | -29 % / -28 % / -31 % |
| T5 | mittel (1 Monat) | 97 % | -0,3 % / 4,5 % / 8,8 % | 0,07 / 0,44 / 0,66 | -42 % / -28 % / -31 % |
| V1 | mittel (1 Monat) | 100 % | 3,2 % / 4,5 % / 8,8 % | 0,27 / 0,44 / 0,66 | -43 % / -28 % / -31 % |
| V1 | lang (6 Monate) | 100 % | 14,2 % / 7,9 % / 14,0 % | 0,93 / 0,83 / 1,21 | -28 % / -29 % / -34 % |
| V2 | mittel (1 Monat) | 100 % | 1,7 % / 4,5 % / 8,8 % | 0,19 / 0,44 / 0,66 | -42 % / -28 % / -31 % |
| V2 | lang (6 Monate) | 100 % | 11,9 % / 7,9 % / 14,0 % | 1,09 / 0,83 / 1,21 | -27 % / -29 % / -34 % |
| V3 | mittel (1 Monat) | 100 % | 4,5 % / 4,5 % / 8,8 % | 0,40 / 0,44 / 0,66 | -27 % / -28 % / -31 % |
| V3 | lang (6 Monate) | 100 % | 11,7 % / 7,9 % / 14,0 % | 1,23 / 0,83 / 1,21 | -24 % / -29 % / -34 % |
| V4 | mittel (1 Monat) | 100 % | 0,2 % / 4,5 % / 8,8 % | 0,08 / 0,44 / 0,66 | -40 % / -28 % / -31 % |
| V4 | lang (6 Monate) | 100 % | 6,6 % / 7,9 % / 14,0 % | 0,71 / 0,83 / 1,21 | -31 % / -29 % / -34 % |
| V5 | mittel (1 Monat) | 97 % | -0,2 % / 4,5 % / 8,8 % | 0,09 / 0,44 / 0,66 | -52 % / -28 % / -31 % |

### Kryptowährungen

| Regel | Horizont | Anteil investiert | Rendite p. a. (Korb / alle / Index) | Sharpe (Korb / alle / Index) | Max. Rückgang auf Tageskursen (Korb / alle / Index) |
|---|---|---|---|---|---|
| T0 | mittel (1 Monat) | 71 % | 33,8 % / 79,2 % / 51,8 % | 0,64 / 0,92 / 0,93 | -69 % / -76 % / -73 % |
| T1 | mittel (1 Monat) | 100 % | 75,6 % / 79,2 % / 51,8 % | 0,84 / 0,92 / 0,93 | -85 % / -76 % / -73 % |
| T1 | lang (6 Monate) | 100 % | 62,6 % / 66,6 % / 38,5 % | 0,61 / 0,43 / 0,58 | -90 % / -79 % / -77 % |
| T2 | mittel (1 Monat) | 83 % | 59,2 % / 79,2 % / 51,8 % | 0,78 / 0,92 / 0,93 | -64 % / -76 % / -73 % |
| T2 | lang (6 Monate) | 81 % | 59,4 % / 66,6 % / 38,5 % | 0,54 / 0,43 / 0,58 | -67 % / -79 % / -77 % |
| T3 | mittel (1 Monat) | 83 % | 65,4 % / 79,2 % / 51,8 % | 0,80 / 0,92 / 0,93 | -61 % / -76 % / -73 % |
| T3 | lang (6 Monate) | 81 % | 64,3 % / 66,6 % / 38,5 % | 0,55 / 0,43 / 0,58 | -68 % / -79 % / -77 % |
| T4 | mittel (1 Monat) | 31 % | 31,8 % / 79,2 % / 51,8 % | 0,75 / 0,92 / 0,93 | -49 % / -76 % / -73 % |
| T5 | mittel (1 Monat) | 51 % | 8,5 % / 79,2 % / 51,8 % | 0,38 / 0,92 / 0,93 | -75 % / -76 % / -73 % |
| V1 | mittel (1 Monat) | 100 % | 50,6 % / 79,2 % / 51,8 % | 0,76 / 0,92 / 0,93 | -91 % / -76 % / -73 % |
| V1 | lang (6 Monate) | 100 % | 13,5 % / 66,6 % / 38,5 % | 0,51 / 0,43 / 0,58 | -94 % / -79 % / -77 % |
| V2 | mittel (1 Monat) | 100 % | 58,3 % / 79,2 % / 51,8 % | 0,84 / 0,92 / 0,93 | -81 % / -76 % / -73 % |
| V2 | lang (6 Monate) | 100 % | 72,5 % / 66,6 % / 38,5 % | 0,45 / 0,43 / 0,58 | -81 % / -79 % / -77 % |
| V3 | mittel (1 Monat) | 83 % | 65,4 % / 79,2 % / 51,8 % | 0,80 / 0,92 / 0,93 | -61 % / -76 % / -73 % |
| V3 | lang (6 Monate) | 81 % | 64,3 % / 66,6 % / 38,5 % | 0,55 / 0,43 / 0,58 | -68 % / -79 % / -77 % |
| V4 | mittel (1 Monat) | 80 % | 35,1 % / 79,2 % / 51,8 % | 0,61 / 0,92 / 0,93 | -78 % / -76 % / -73 % |
| V4 | lang (6 Monate) | 94 % | 35,9 % / 66,6 % / 38,5 % | 0,45 / 0,43 / 0,58 | -82 % / -79 % / -77 % |
| V5 | mittel (1 Monat) | 51 % | 9,0 % / 79,2 % / 51,8 % | 0,38 / 0,92 / 0,93 | -80 % / -76 % / -73 % |

## Einordnung der Befunde

Auch ein bestandener Test wäre kein Versprechen: Er stützt sich auf eine einzige Testperiode mit wenigen unabhängigen Beobachtungen (bei 6 Monaten Haltedauer im Testabschnitt nur etwa 7 bis 8 nicht überlappende Termine), auf heutige Listenmitglieder (Survivorship) und auf Regeln, die nach Sicht der In-Sample-Zahlen um Varianten ergänzt wurden. Bei Varianten gilt bei gleichem Urteil die vorab festgelegte Regel als maßgeblich.

Zur Beschriftung "nur Marktrendite": Dieses Etikett (Long-Seite besteht auf absoluter Rendite, hat aber keinen belegten Vorsprung vor dem Durchschnitt aller Werte außerhalb der Stichprobe, also Mittelwert ≤ 0 oder t < 1) wurde nach dem ersten Gesamtlauf präzisiert, damit Regeln, die außerhalb der Stichprobe doch vorn lagen, aber im Auswahlzeitraum durchfielen, nicht fälschlich als Beta erscheinen. Die Urteile "belegt", "schwach belegt" und "nur relativ" waren davon nicht betroffen.

## Varianten und Mehrfachtests

Vorab festgelegt waren 9 Regel-Horizont-Kombinationen je Familie, dazu 5 Varianten (9 Kombinationen). Das ergibt 144 Zellen (Regel, Horizont, Seite, Familie), aber nur **136 verschiedene Messungen**: 8 Zellen (die Long-Seite von V3 ist die von T3) wiederholen eine andere und zählen nicht doppelt.

Wer genug Regeln probiert, findet allein durch Zufall welche mit t ≥ 2. Mit dem einfachen t auf nicht überlappenden Terminen (so wird jetzt beurteilt) gilt: mittel (1 Monat): 84 bzw. 38 nicht überlappende Termine im In- bzw. Out-of-Sample-Teil, Zufallstreffer je Teil 2,4 % bzw. 2,7 %; lang (6 Monate): 16 bzw. 7 nicht überlappende Termine im In- bzw. Out-of-Sample-Teil, Zufallstreffer je Teil 3,2 % bzw. 4,7 %. Aus den tatsächlichen Stichprobengrößen jeder verschiedenen Messung (Simulation mit festem Startwert, t-Verteilung, beide Abschnitte unabhängig, Kosten- und Robustheitsbedingung nicht eingerechnet, also eine obere Grenze) folgen **0,2 zufällige "belegt"** und **0,7 zufällige "belegt oder schwach belegt"** bei keinem echten Effekt. Die erste Fassung hatte "etwa 0,08" genannt; das war um den Faktor 8 bis 30 zu niedrig: die unabhängige Prüfung kam für die damalige Newey-West-Regel auf etwa 0,6 zufällige "belegt" und 2,5 zufällige "belegt oder schwach belegt". Die Familien sind aber untereinander stark verwandt (derselbe Markt), die effektive Zahl unabhängiger Tests ist kleiner.

| Variante | Regel | Begründung (aus dem In-Sample-Teil) |
|---|---|---|
| V1 | Momentum 12-1, 10 % Ränder | Im In-Sample-Teil zeigte sich die Momentum-Prämie nur an den Rändern (USA, lang): engere Ränder könnten sie bündeln. |
| V2 | Momentum 6-1 | Das 12-Monats-Fenster war im In-Sample-Teil bei den meisten Familien schwach; die klassische 6-Monats-Bildung reagiert schneller. |
| V3 | Momentum mit Trendfilter, ohne Marktfilter | Im In-Sample-Teil ließ der Marktfilter die Short-Seite fast leer (13 bis 85 von 432 Terminen): T3 war dort nicht beurteilbar. |
| V4 | Schnellerer Trendzustand (20/100) | Der 200-Tage-Schnitt hinkt hinterher; im In-Sample-Teil verdiente T2 nicht mehr als das Universum, ein schnellerer Zustand könnte früher drehen. |
| V5 | Rücksetzer, 10 % Ränder | Im In-Sample-Teil blieb die Short-Seite von T5 bei ETFs hinter dem Markt zurück: tiefere Extreme könnten die Umkehr deutlicher machen. |

V1 wurde mit genau der Zelle begründet, die später als einzige bestand (Momentum, US, lang): sie ist deshalb kein unabhängiger zweiter Test.

## Schwellen

Alle Schwellen sind vor dem Lauf festgelegt und wurden nicht angepasst:

| Schwelle | Wert | Bedeutung |
|---|---|---|
| VALIDATED_T | 2,0 | belegt: t in In-Sample und Out-of-Sample mindestens so hoch, und der Mittelwert bleibt bei doppelten Kosten positiv |
| EXPERIMENTAL_T_IS | 2,0 | schwach belegt: t im In-Sample-Teil |
| EXPERIMENTAL_T_OOS | 1,0 | schwach belegt: t im Out-of-Sample-Teil bei positivem Mittelwert; zugleich die Grenze, unter der ein Vorsprung vor dem Markt als "nicht vorhanden" gilt (nur Marktrendite) |
| STRONG_T | 3,0 | "stark": gepooltes t mindestens so hoch; nie für Varianten |
| VALIDATED_MIN_OOS_DATES | 30 | Mindestzahl Termine im Out-of-Sample-Teil für "belegt"; geprüft wird die Zahl aller Termine (auch überlappender), während das t aus den nicht überlappenden Terminen stammt; die Schwelle wurde nicht geändert |
| EXPERIMENTAL_MIN_OOS_DATES | 20 | Mindestzahl Termine im Out-of-Sample-Teil für "schwach belegt"; ein t ≥ 1 auf wenigen Terminen ist Rauschen; auch hier zählen alle (überlappenden) Termine, nicht die nicht überlappenden, aus denen das t stammt |
| MIN_BASKET | 3 | durchschnittliche Korbgröße, ab der ein Urteil über "nicht belegt" hinaus möglich ist |
| ROBUST_K / ROBUST_MIN_T | 3 / 2,0 | Zahl der entfernten Gewinner und gepooltes t, das ohne sie bestehen muss |
| IS_FRACTION | 0,70 | Anteil der Termine im In-Sample-Teil (mit Purge) |
| Kosten | US-Aktien 10 bp, DAX-Aktien 10 bp, ETFs (Welt + Sektoren) 10 bp, Kryptowährungen 30 bp | je Hin- und Rückweg und Position |
| Stop-Modus | 2,5 / 4,0 x ATR | nur zum Vergleich gerechnet, nicht Teil des Urteils |
| Veraltung | 120 Tage | ältere Studien werden im Programm als veraltet markiert |

## Grenzen

- **Drei Werte tragen den früheren Befund.** Das Ergebnis der früheren "belegten" Zelle hing an NVDA, AVGO und AMD; es gibt keinen Grund anzunehmen, dass nicht auch andere Zellen an wenigen späteren Gewinnern hängen. Darum die Robustheitsprüfung; sie ist vorab auf k = 3 festgelegt, nicht auf das beste Ergebnis abgestimmt.
- **Der Out-of-Sample-Zeitraum war überall momentum-freundlich.** In jeder Familie schnitt Momentum im letzten Abschnitt besser ab als im Auswahlzeitraum (beim DAX bei einem In-Sample-t von etwa 0). Ein Bestehen dort sagt wenig über andere Marktphasen.
- **Der Stop-Modus war negativ.** Mit Stop verschwand der Vorteil der früheren US-Zelle; die Regel funktioniert, wenn überhaupt, nur als reine Zeitregel ohne Stop, und das ist dann ein Risiko für den Anleger (keine Verlustbegrenzung).
- **V1 entstand aus derselben Zelle**, die sie später bestätigen sollte; die Variante ist kein unabhängiger Beleg.
- **Das Newey-West-t ist optimistisch** und wird nur zur Information mitgeführt; beurteilt wird mit dem Median des einfachen t der nicht überlappenden Termine über alle Phasen. Dieses beruht auf wenigen Beobachtungen (beim Halbjahr etwa 7 bis 8 im Testabschnitt) und ist entsprechend unscharf.
- **Der größte Rückgang** der Risikosicht wird auf täglichen Kursen innerhalb der Haltefenster gemessen, die Rendite und der Sharpe-Wert aber an den Periodenenden; Tage zwischen zwei Haltefenstern (Bargeld) sind nicht abgebildet.
- **Survivorship-Bias**: Es werden die heutigen Mitglieder der Listen rückwirkend getestet. Firmen, die zwischenzeitlich schwach waren, ausschieden oder pleitegingen, fehlen. Das begünstigt vor allem Long-Regeln und verzerrt Short-Ergebnisse zum Schlechteren (Verlierer fehlen).
- **Keine Leihgebühren und Dividendenkosten** bei Shorts; reale Shorts sind teurer und teils nicht möglich. Kostenannahmen sind pauschal (10 bp bzw. 30 bp je Hin- und Rückweg), ohne Abwägung von Spreads in schwachen Märkten oder Steuern.
- **Eine einzige Out-of-Sample-Periode** (die letzten 30 % der Termine): sie ist ein einzelnes Marktumfeld. Ein Bestehen belegt nicht, dass die Regel in anderen Umfeldern funktioniert, ein Durchfallen nicht, dass sie nie funktionieren kann.
- **Gleichgewichtete Körbe** ohne Rücksicht auf Liquidität und Positionsgrößen; die Rendite eines echten Depots mit wenigen Positionen streut deutlich stärker.
- Kurse sind rückwirkend um Dividenden und Splits bereinigt (Yahoo), Krypto-Daten reichen je nach Coin nur wenige Jahre zurück.
