# Kurzfrist-Handel: Simulation und ehrliche Bewertung

Studienstand: 2026-11-12. Alles ist eine **Simulation auf historischen Yahoo-Kursen**. Es wird nichts bei einem Broker ausgeführt, und eine Simulation ist keine echte Ausführung. Dies ist die Fassung **nach einer unabhängigen Prüfung**; Abschnitt 9 sagt, was korrigiert wurde und was sich dadurch geändert hat.

## 1. Was getestet wurde
- **Universum:** 61 liquide US-Einzelwerte (Large Caps aller Sektoren) plus die 4 ETFs SPY, QQQ, IWM, DIA = 65 Symbole.
- **Tagesbalken:** 2016-11-14 bis 2026-11-12 (≈10 Jahre, 2513 Handelstage).
- **60-Minuten-Balken:** 2024-11-14 bis 2026-11-12 (Yahoo liefert höchstens ≈2 Jahre; 493 vollständige Handelstage).
- **5-Minuten-Balken:** 2026-09-15 bis 2026-11-12 (Yahoo liefert höchstens 60 Tage; nur 43 Handelstage, **kleine Stichprobe**, nur ein Realismus-Check).
- **Kosten (Hin- und Rückweg):** 5 Basispunkte (bp) für Daytrades, 10 bp für Swing-Trades; zusätzlich 2× und 4× als Empfindlichkeit. Für deutsche Privatanleger (Ordergebühr, größere Spreads bei Tradegate/L&S, CFDs oder Knock-outs für Short-Positionen) sind eher 10–20 bp realistisch. 1 bp = 0,01 %.
- **Ausführung streng:** Ein Signal aus dem Schlusskurs eines Balkens wird erst zur Eröffnung des nächsten Balkens gehandelt. Stop und Ziel im selben Balken: der Stop gilt zuerst. Eröffnet ein Balken jenseits des Stops, wird zum schlechteren Eröffnungskurs ausgestoppt. Bei M3 (VWAP-Rückkehr) liegt der Stop 1 σ vom tatsächlichen Einstiegskurs entfernt (nicht 1 σ jenseits der 2-σ-Auslöseschwelle, wie ursprünglich geplant); das Ziel ist das VWAP.
- **Vorheriger Schlusskurs der Intraday-Regeln:** Er stammt aus dem letzten Balken der vorherigen Intraday-Sitzung, nicht aus den Tagesbalken (siehe Abschnitt 7, Datenfehler H-1).
- **Lern- und Testzeitraum:** Die ersten 70 % der Handelstage jedes Datensatzes sind der Lernzeitraum (IS), die letzten 30 % der Testzeitraum (OOS). Parameter wurden nur im IS gewählt; der OOS wurde je Kandidat einmal ausgewertet und wird hier unverändert berichtet.
- **Signifikanz, eine Größe für alle Entscheidungen:** Jede Strategie hat **eine** Tagesreihe, die „Basis des t-Werts“. Bei Daytrades ist es der Mittelwert der Trades des Einstiegstags. Bei Swing-Trades, deren Positionen bis zu 10 Tage offen sind und sich überlappen, ist es das **Kalenderzeit-Portfolio**: jeden Kalendertag der Durchschnitt der Tagesergebnisse aller an dem Tag offenen Positionen, Kosten am Ausstiegstag abgezogen. t-Wert, 95 %-Intervall (zieht ganze Tage neu, 2000 Ziehungen), die Kostenbedingung des Urteils (OOS-Ergebnis bei doppelten Kosten > 0) und die Frage „schlägt die Marktdrift?“ verwenden alle den Mittelwert **dieser** Reihe. Der Durchschnitt je Trade steht daneben nur zur Information (er gewichtet Tage mit vielen Signalen stärker). Als grobe Regel: |t| unter 2 ist von Zufall nicht zu unterscheiden.

**Anzahl getesteter Varianten insgesamt: 24** (17 vorab festgelegte, 7 nachträgliche).

## 2. Ergebnisse je Strategie (netto nach Kosten)
„Ø je Trade“ = durchschnittlicher Netto-Ertrag je Trade in Basispunkten (zur Information; kann an Tagen mit vielen Trades vom Tageswert abweichen, auch im Vorzeichen). „Ø je Handelstag (Basis des t-Werts)“ = Mittelwert der Tagesreihe, zu der der t-Wert gehört (Swing: Kalenderzeit-Portfolio, siehe Abschnitt 1). t = t-Wert dieser Reihe; Intervall = 95 %-Intervall von „Ø je Handelstag“ im OOS (bp). Treffer = Anteil der Trades mit Gewinn.

| ID | Strategie | Daten | IS Trades | IS Treffer | IS Ø je Trade | IS Ø je Handelstag (Basis des t-Werts) | IS t | OOS Trades | OOS Treffer | OOS Ø je Trade | OOS Ø je Handelstag (Basis des t-Werts) | OOS t | OOS 95 %-Intervall | Urteil |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | Gap-Fade (Tagesbalken) | Tagesbalken (10 J.) | 16721 | 49,1 % | -8,7 | -5,4 | -2,07 | 6890 | 50,8 % | -0,8 | -7,2 | -2,11 | [-13,5; -0,4] | verworfen |
| D2 | Gap-Fortsetzung (Tagesbalken) | Tagesbalken (10 J.) | 1216 | 42,0 % | -32,9 | -5,8 | -0,67 | 617 | 42,8 % | -13,9 | -25,0 | -1,91 | [-48,7; -0,5] | verworfen |
| D3 | Ausverkaufs-Umkehr | Tagesbalken (10 J.) | 1010 | 44,6 % | -26,3 | -18,1 | -2,07 | 472 | 52,1 % | 2,9 | -8,1 | -0,74 | [-30,1; 14,7] | verworfen |
| D4 | Trendtag-Fortsetzung | Tagesbalken (10 J.) | 1993 | 45,9 % | -15,8 | -5,8 | -1,04 | 757 | 46,4 % | -3,5 | 0,9 | 0,12 | [-14,5; 17,4] | verworfen |
| D5 | RSI(2)-Extrem, Daytrade | Tagesbalken (10 J.) | 4644 | 50,5 % | 2,4 | -3,0 | -0,87 | 2110 | 48,5 % | -4,2 | -1,2 | -0,25 | [-10,8; 8,5] | verworfen |
| H1 | Spätes Momentum (Schlussstunde) | 60-Min. (≈2 J.) | 22009 | 42,0 % | -6,1 | -6,1 | -8,46 | 9608 | 44,1 % | -4,7 | -4,7 | -7,21 | [-5,9; -3,4] | verworfen |
| H1b | Momentum der ersten Stunde bis Handelsschluss | 60-Min. (≈2 J.) | 22006 | 46,8 % | -7,9 | -7,9 | -3,47 | 9606 | 47,3 % | -5,5 | -5,5 | -2,75 | [-9,3; -1,6] | verworfen |
| H2 | Ausbruch aus der ersten Handelsstunde | 60-Min. (≈2 J.) | 14808 | 43,6 % | -4,3 | -6,3 | -3,30 | 6061 | 43,5 % | -8,3 | -9,4 | -5,09 | [-12,9; -5,7] | verworfen |
| H3 | Morgen-Fade | 60-Min. (≈2 J.) | 621 | 15,9 % | -7,8 | -8,3 | -1,78 | 302 | 21,2 % | -1,6 | -8,5 | -1,62 | [-18,6; 1,2] | verworfen |
| M1 | Opening-Range-Ausbruch (5 Min.) | 5-Min. (≈2 Mon.) | 1377 | 41,2 % | -5,9 | -5,9 | -1,39 | 624 | 45,2 % | 0,1 | -3,0 | -0,30 | [-20,8; 16,9] | verworfen |
| M2a | Gap-Fade, Einstieg 09:35 (5 Min.) | 5-Min. (≈2 Mon.) | 346 | 52,9 % | 2,7 | 5,8 | 0,44 | 259 | 49,8 % | -9,5 | -11,4 | -0,76 | [-39,5; 16,7] | verworfen |
| M2b | Gap-Fortsetzung, Einstieg 09:35 (5 Min.) | 5-Min. (≈2 Mon.) | 17 | 41,2 % | -57,5 | -76,5 | -1,44 | 15 | 40,0 % | -85,5 | -183,4 | -2,48 | [-336,7; -63,1] | verworfen |
| M3 | VWAP-Rückkehr (5 Min.) | 5-Min. (≈2 Mon.) | 750 | 30,1 % | -5,8 | -5,4 | -2,35 | 322 | 29,2 % | -5,9 | -4,9 | -2,07 | [-9,8; -0,7] | verworfen |
| S1 | RSI(2)-Rücksetzer | Tagesbalken (10 J.) | 5460 | 62,8 % | 6,5 | -0,5 | -0,15 | 2430 | 61,5 % | -2,0 | -0,7 | -0,22 | [-7,0; 5,7] | verworfen |
| S2 | 5-Tage-Umkehr | Tagesbalken (10 J.) | 1256 | 51,2 % | 12,5 | -4,2 | -0,88 | 471 | 53,9 % | 37,9 | 4,9 | 0,87 | [-5,5; 16,1] | verworfen |
| S3 | Ausbruch mit Volumen | Tagesbalken (10 J.) | 3200 | 44,2 % | -6,4 | -1,1 | -0,42 | 1287 | 46,9 % | -18,9 | -1,6 | -0,54 | [-7,4; 4,2] | verworfen |
| S4 | Drift nach Kurslücke | Tagesbalken (10 J.) | 393 | 52,4 % | 74,4 | 8,6 | 1,47 | 205 | 51,7 % | 27,9 | 8,7 | 0,91 | [-9,8; 27,5] | verworfen |
| S4s | Drift nach Kurslücke, nur Short | Tagesbalken (10 J.) | 210 | 53,8 % | 124,4 | 10,3 | 1,31 | 105 | 47,6 % | 10,9 | -4,9 | -0,37 | [-30,5; 21,4] | verworfen |
| S2s | 5-Tage-Umkehr, nur Short | Tagesbalken (10 J.) | 494 | 49,6 % | 38,4 | -5,1 | -0,64 | 151 | 48,3 % | -23,7 | -4,2 | -0,49 | [-21,3; 12,8] | verworfen |
| S3l | Ausbruch mit Volumen, nur Long | Tagesbalken (10 J.) | 1676 | 51,1 % | 46,8 | 7,0 | 2,47 | 655 | 52,5 % | 67,5 | 9,6 | 2,06 | [0,6; 18,1] | **formal bestanden, aber nur Marktrendite** |
| S1x | RSI(2)-Rücksetzer, extremere Schwelle | Tagesbalken (10 J.) | 2765 | 63,5 % | 20,1 | 1,5 | 0,42 | 1182 | 60,9 % | -0,8 | -1,7 | -0,37 | [-10,8; 6,7] | verworfen |
| S1v | RSI(2)-Rücksetzer, nur volatile Titel | Tagesbalken (10 J.) | 3000 | 63,0 % | 15,0 | -0,5 | -0,12 | 1477 | 61,2 % | 1,3 | -0,3 | -0,07 | [-8,3; 7,9] | verworfen |
| D5v | RSI(2)-Extrem Daytrade, nur volatile Titel | Tagesbalken (10 J.) | 2560 | 50,7 % | 7,1 | -0,2 | -0,03 | 1241 | 48,8 % | -4,2 | 2,0 | 0,31 | [-10,1; 14,3] | verworfen |
| H3x | Morgen-Fade, nur extreme erste Stunde | 60-Min. (≈2 J.) | 25 | 12,0 % | -10,5 | -9,5 | -0,54 | 14 | 14,3 % | -28,8 | -26,6 | -1,29 | [-61,0; 14,4] | verworfen |

Urteilsregeln (vorab festgelegt, alle Mittelwerte = „Ø je Handelstag“): *belegt* = IS-t ≥ 2 und OOS-t ≥ 2 (beide positiv), OOS-Mittel auch bei doppelten Kosten > 0, mindestens 100 OOS-Trades an ≥ 20 OOS-Tagen. *experimentell* = IS-t ≥ 2, OOS-Mittel > 0, OOS-t ≥ 1, ≥ 50 OOS-Trades. Alles andere *verworfen*. „stark“ nur bei gepooltem t ≥ 3 **und** geschlagener Marktdrift. **Etikett** (abgeleitet, nach dem Urteil): *belegt* / *experimentell* nur, wenn das Urteil so lautet UND die Regel die Marktdrift schlägt (Excess-t ≥ 2 und Excess > 0 im OOS); besteht sie das Urteil, schlägt aber die Marktdrift nicht, heißt sie *formal bestanden, aber nur Marktrendite* (in den Daten „beta“). Die 5-Minuten-Strategien haben nur 13 OOS-Tage und können die Regel für „belegt“ konstruktionsbedingt nicht erfüllen.

### Brutto, Excess und Long/Short (OOS)
Excess = Ertrag abzüglich des Durchschnitts des gesamten Universums im gleichen Zeitfenster (Short: umgekehrt). Nur der Excess ist Können; der Rest ist Marktdrift. Sharpe wird nur über die Handelstage mit Position gerechnet (Tage ohne Trade zählen nicht) und ist deshalb höher als bei einem ganzjährig investierten Portfolio. Max. Drawdown = größter Rückgang der **zusammengesetzten** Kapitalkurve der Tagesreihe in % (höchstens 100 %).

| ID | Ø brutto je Trade (bp) | Median netto je Trade (bp) | Profit-Faktor | Ø Excess je Trade (bp) | Ø Excess je Handelstag (bp, Basis des Excess-t) | Excess t | Long: Trades / Ø bp je Trade | Short: Trades / Ø bp je Trade | Sharpe (nur Handelstage mit Position) | Max. Drawdown (zusammengesetzt) |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 4,2 | 2,4 | 0,99 | -5,3 | -7,8 | -2,67 | 4545 / -1,3 | 2345 / 0,3 | -1,23 | 53,5 % |
| D2 | -8,9 | -58,8 | 0,87 | -14,7 | -17,9 | -1,38 | 312 / -21,4 | 305 / -6,2 | -1,78 | 56,0 % |
| D3 | 7,9 | 8,3 | 1,04 | -6,1 | -11,8 | -1,12 | 232 / -20,9 | 240 / 25,9 | -0,74 | 33,1 % |
| D4 | 1,5 | -10,8 | 0,95 | 3,3 | 4,9 | 0,63 | 376 / 8,5 | 381 / -15,4 | 0,10 | 21,1 % |
| D5 | 0,8 | -5,5 | 0,92 | -3,6 | -2,1 | -0,51 | 1423 / -4,3 | 687 / -4,1 | -0,16 | 29,0 % |
| H1 | 0,3 | -5,0 | 0,71 | -5,1 | -5,1 | -9,39 | 4908 / -6,0 | 4700 / -3,3 | -9,40 | 6,9 % |
| H1b | -0,5 | -5,9 | 0,87 | -5,0 | -5,0 | -2,88 | 4998 / -9,7 | 4608 / -0,9 | -3,59 | 8,2 % |
| H2 | -3,3 | -9,1 | 0,74 | -9,6 | -10,0 | -6,95 | 3041 / -9,9 | 3020 / -6,8 | -6,64 | 13,5 % |
| H3 | 3,4 | -15,7 | 0,94 | -2,3 | -9,0 | -1,73 | 142 / -10,0 | 160 / 5,9 | -2,47 | 10,3 % |
| M1 | 5,1 | -24,3 | 1,00 | -10,5 | -11,4 | -2,02 | 359 / -4,3 | 265 / 6,0 | -1,31 | 1,0 % |
| M2a | -4,5 | -0,7 | 0,85 | -13,7 | -13,1 | -1,16 | 155 / 8,5 | 104 / -36,4 | -3,36 | 2,7 % |
| M2b | -80,5 | -68,8 | 0,32 | -108,8 | -179,3 | -2,44 | 6 / 41,9 | 9 / -170,4 | -14,90 | 12,3 % |
| M3 | -0,9 | -25,4 | 0,76 | 1,1 | -0,4 | -0,11 | 185 / -9,5 | 137 / -1,0 | -9,11 | 0,6 % |
| S1 | 8,0 | 55,9 | 0,98 | -7,8 | -2,7 | -0,97 | 1672 / 15,3 | 758 / -40,0 | -0,13 | 17,8 % |
| S2 | 47,9 | 32,2 | 1,26 | 18,5 | 4,4 | 0,90 | 320 / 67,0 | 151 / -23,7 | 0,57 | 26,5 % |
| S3 | -8,9 | -50,5 | 0,91 | -19,9 | -2,4 | -1,07 | 654 / 66,9 | 633 / -107,6 | -0,31 | 26,5 % |
| S4 | 37,9 | 25,3 | 1,19 | 15,3 | 7,2 | 0,81 | 100 / 45,8 | 105 / 10,9 | 0,68 | 30,4 % |
| S4s | 20,9 | -17,1 | 1,07 | 28,8 | 5,2 | 0,43 | 0 / 0,0 | 105 / 10,9 | -0,34 | 27,9 % |
| S2s | -13,7 | -5,9 | 0,86 | 8,5 | 3,1 | 0,39 | 0 / 0,0 | 151 / -23,7 | -0,42 | 35,3 % |
| S3l | 77,5 | 27,9 | 1,39 | -1,8 | 1,7 | 0,37 | 655 / 67,5 | 0 / 0,0 | 1,21 | 20,6 % |
| S1x | 9,2 | 53,9 | 0,99 | -14,0 | -5,5 | -1,43 | 806 / 21,5 | 376 / -48,7 | -0,22 | 33,7 % |
| S1v | 11,3 | 71,4 | 1,01 | -2,5 | -1,5 | -0,40 | 949 / 26,3 | 528 / -43,8 | -0,04 | 27,4 % |
| D5v | 0,8 | -6,7 | 0,93 | -3,7 | 1,1 | 0,21 | 766 / -1,4 | 475 / -8,9 | 0,22 | 29,0 % |
| H3x | -23,8 | -40,4 | 0,32 | -2,9 | -8,1 | -0,39 | 9 / -9,5 | 5 / -63,6 | -6,19 | 2,9 % |

## 3. Vergleich mit Baselines (OOS)
Auf den **eigenen Trade-Fenstern der Strategie (einschließlich ihres Ausstiegstags)**: Jedes Fenster läuft von der Einstiegseröffnung bis zum Schluss des Balkens, an dem die Strategie ausgestiegen ist; ein ausgestoppter Trade endet also am Stopptag. Die Baselines sind damit nicht „ohne Stop“ im strengen Sinn, sondern an den Ausstieg der Strategie gebunden. Verglichen werden **immer Long**, die **eigenen Richtungen** der Strategie (jeweils ohne Stop innerhalb des Fensters) und **Zufallsrichtung** (200 Wiederholungen, Ø und 95 %-Band). Liegt die eigene Richtung nicht über dem Zufallsband, steckt in der Richtungswahl kein Können. Alle Werte sind Ø je Trade.

| ID | Immer Long Ø bp | Eigene Richtung Ø bp | Zufall Ø bp | Zufall 95 %-Band | Universum Ø bp |
|---|---|---|---|---|---|
| D1 | -3,8 | -1,2 | -5,0 | [-8,9; -0,8] | 1,4 |
| D2 | -15,2 | -13,0 | -4,8 | [-25,1; 15,4] | 0,1 |
| D3 | -29,1 | -3,8 | -4,5 | [-25,7; 17,0] | -22,6 |
| D4 | 3,2 | -0,3 | -5,8 | [-19,0; 6,1] | 4,9 |
| D5 | -4,3 | -4,1 | -5,0 | [-10,9; 2,2] | -0,8 |
| H1 | -6,3 | -4,7 | -5,0 | [-5,7; -4,3] | -1,4 |
| H1b | -9,4 | -5,5 | -4,9 | [-6,7; -2,8] | -4,4 |
| H2 | -6,7 | -8,0 | -5,1 | [-6,8; -3,3] | -2,7 |
| H3 | -21,6 | 12,3 | -5,4 | [-16,7; 5,4] | -4,8 |
| M1 | -9,3 | -0,5 | -5,3 | [-12,8; 2,7] | -4,8 |
| M2a | 15,6 | -9,7 | -5,3 | [-27,4; 12,2] | 19,0 |
| M2b | 113,1 | -82,2 | -0,1 | [-104,0; 96,4] | 24,5 |
| M3 | -8,9 | -8,0 | -4,7 | [-11,2; 2,1] | -2,2 |
| S1 | 19,8 | -0,6 | -10,0 | [-21,6; 1,9] | 20,7 |
| S2 | 54,9 | 46,8 | -13,1 | [-50,9; 17,9] | 40,1 |
| S3 | 73,5 | -15,5 | -9,0 | [-38,6; 16,3] | 69,6 |
| S4 | 4,1 | 34,9 | -10,6 | [-62,6; 38,3] | 30,9 |
| S4s | -40,0 | 20,0 | -13,9 | [-92,1; 57,4] | 17,8 |
| S2s | 2,5 | -22,5 | -16,9 | [-84,4; 52,6] | 32,2 |
| S3l | 67,4 | 67,4 | -8,1 | [-53,5; 44,4] | 69,4 |
| S1x | 26,3 | 1,6 | -10,6 | [-28,3; 5,8] | 27,7 |
| S1v | 30,0 | 2,2 | -10,2 | [-28,8; 10,9] | 21,6 |
| D5v | -0,3 | -4,0 | -5,0 | [-14,7; 4,0] | 0,5 |
| H3x | -8,1 | -20,7 | -4,7 | [-35,4; 26,5] | -13,2 |

Zum Vergleich die reine Marktdrift (gleichgewichtetes Universum, Tagesbalken): Eröffnung bis Schluss 2,0 bp (IS) / 2,4 bp (OOS) je Tag; 5-Tage-Halten 31,3 bp (IS) / 35,2 bp (OOS). Der Aktienmarkt stieg im Testzeitraum; jede Long-Regel profitiert davon.

## 4. Kostenempfindlichkeit (OOS, Ø je Handelstag (Basis des t-Werts) in bp, t-Wert der Reihe)
| ID | Basiskosten | 2× Kosten | 4× Kosten |
|---|---|---|---|
| D1 | -7,2 (t -2,11) | -12,2 (t -3,59) | -22,2 (t -6,54) |
| D2 | -25,0 (t -1,91) | -30,0 (t -2,30) | -40,0 (t -3,06) |
| D3 | -8,1 (t -0,74) | -13,1 (t -1,19) | -23,1 (t -2,09) |
| D4 | 0,9 (t 0,12) | -4,1 (t -0,51) | -14,1 (t -1,77) |
| D5 | -1,2 (t -0,25) | -6,2 (t -1,28) | -16,2 (t -3,32) |
| H1 | -4,7 (t -7,21) | -9,7 (t -14,91) | -19,7 (t -30,31) |
| H1b | -5,5 (t -2,75) | -10,5 (t -5,27) | -20,5 (t -10,31) |
| H2 | -9,4 (t -5,09) | -14,4 (t -7,80) | -24,4 (t -13,23) |
| H3 | -8,5 (t -1,62) | -13,5 (t -2,58) | -23,5 (t -4,50) |
| M1 | -3,0 (t -0,30) | -8,0 (t -0,79) | -18,0 (t -1,78) |
| M2a | -11,4 (t -0,76) | -16,4 (t -1,10) | -26,4 (t -1,77) |
| M2b | -183,4 (t -2,48) | -188,4 (t -2,55) | -198,4 (t -2,69) |
| M3 | -4,9 (t -2,07) | -9,9 (t -4,19) | -19,9 (t -8,42) |
| S1 | -0,7 (t -0,22) | -3,8 (t -1,21) | -10,0 (t -3,22) |
| S2 | 4,9 (t 0,87) | 2,8 (t 0,49) | -1,5 (t -0,26) |
| S3 | -1,6 (t -0,54) | -2,8 (t -0,96) | -5,3 (t -1,80) |
| S4 | 8,7 (t 0,91) | 6,5 (t 0,69) | 2,2 (t 0,24) |
| S4s | -4,9 (t -0,37) | -7,0 (t -0,53) | -11,3 (t -0,85) |
| S2s | -4,2 (t -0,49) | -6,3 (t -0,74) | -10,6 (t -1,23) |
| S3l | 9,6 (t 2,06) | 8,4 (t 1,79) | 5,9 (t 1,25) |
| S1x | -1,7 (t -0,37) | -4,7 (t -1,06) | -10,8 (t -2,48) |
| S1v | -0,3 (t -0,07) | -3,3 (t -0,77) | -9,3 (t -2,21) |
| D5v | 2,0 (t 0,31) | -3,0 (t -0,48) | -13,0 (t -2,07) |
| H3x | -26,6 (t -1,29) | -31,6 (t -1,54) | -41,6 (t -2,02) |

## 5. Was verbessert wurde (und was nicht), und was offenzulegen ist
Nach den ersten IS-Ergebnissen wurden 7 zusätzliche Varianten ausprobiert (erlaubt waren höchstens 8), jede mit einer Begründung, **alle protokolliert, auch die Fehlschläge**, und nur im IS gewählt. Danach wurde der OOS einmal für alle ausgewertet; es wurde nichts an den Regeln oder Parametern nachjustiert. **Zwei Dinge sind dabei offenzulegen:**

1. **S3l (Ausbruch mit Volumen, nur Long)** wurde im IS wegen des positiven Brutto-Ergebnisses der Long-Seite gewählt, obwohl sein IS-Excess gegenüber dem Marktdurchschnitt schon damals bei −0,4 bp lag (t = −0,09). Dass sich im OOS dasselbe Bild zeigt (Ertrag = Marktrendite), ist deshalb keine Überraschung.
2. **Das Kriterium „schlägt die Marktdrift“ (`beats_drift`) wurde erst nach dem ersten vollständigen Lauf ergänzt**, nachdem die OOS-Ergebnisse bereits bekannt waren (der erste Lauf, `validated_run1.json`, enthält diese Kennzahl nicht). Alle Zahlen des ersten Laufs waren identisch zu denen des zweiten, es wurde nichts neu eingestellt; die Kennzahl ist aber ein nachträglich hinzugefügter Filter und kein vorab festgelegter. Sie ändert nur, wie Ergebnisse benannt werden, nie, ob eine Regel Geld verdient hat.

Außerdem wurden **die Statistiken nach einer unabhängigen Prüfung korrigiert** (Datenfehler H-1, Überlappung der Swing-Trades M-1, einheitliche Größe M-2), die Regeln selbst aber nicht verändert. Die Vorher/Nachher-Tabelle steht in Abschnitt 9.

| ID | Variante | Begründung (nur IS-Evidenz) | IS Ø je Trade | IS Ø je Handelstag (Basis des t-Werts) | IS t | OOS Ø je Trade | OOS Ø je Handelstag (Basis des t-Werts) | OOS t | Etikett |
|---|---|---|---|---|---|---|---|---|---|
| S4s | Drift nach Kurslücke, nur Short | Im IS hatte die Short-Seite von S4 den deutlich höheren Trade-Durchschnitt (+123 bp gegenüber +19 bp Long); deshalb wurde die Short-Seite allein probiert. | 124,4 | 10,3 | 1,31 | 10,9 | -4,9 | -0,37 | verworfen |
| S2s | 5-Tage-Umkehr, nur Short | Im IS war der Trade-Durchschnitt der Short-Seite von S2 positiv (+38 bp; der t-Wert der Tagesmittel lag bei −0,45), der der Long-Seite negativ. | 38,4 | -5,1 | -0,64 | -23,7 | -4,2 | -0,49 | verworfen |
| S3l | Ausbruch mit Volumen, nur Long | Im IS war das Brutto-Ergebnis der Long-Seite positiv, Short-Positionen bei Ausbrüchen nach unten verloren; gewählt wurde auf dieses Brutto-Ergebnis, obwohl der IS-Excess gegenüber dem Markt schon damals bei −0,4 bp lag (t = −0,09). | 46,8 | 7,0 | 2,47 | 67,5 | 9,6 | 2,06 | formal bestanden, aber nur Marktrendite |
| S1x | RSI(2)-Rücksetzer, extremere Schwelle | Ein tieferer Rücksetzer sollte im Verhältnis zu den festen Kosten weiter zurückspringen. | 20,1 | 1,5 | 0,42 | -0,8 | -1,7 | -0,37 | verworfen |
| S1v | RSI(2)-Rücksetzer, nur volatile Titel | Die Kosten sind in bp fest, der Rücksprung wächst mit der Schwankung: nur Titel mit ATR% ≥ 2 %. | 15,0 | -0,5 | -0,12 | 1,3 | -0,3 | -0,07 | verworfen |
| D5v | RSI(2)-Extrem Daytrade, nur volatile Titel | D5 war brutto positiv, aber etwa so groß wie die 5 bp Kosten; größere Tagesspannen (ATR% ≥ 2 %) verdünnen die Kosten. | 7,1 | -0,2 | -0,03 | -4,2 | 2,0 | 0,31 | verworfen |
| H3x | Morgen-Fade, nur extreme erste Stunde | Eine erste Stunde von 1 ATR ist zu häufig zum Faden; verlangt werden 2 ATR%. | -10,5 | -9,5 | -0,54 | -28,8 | -26,6 | -1,29 | verworfen |

## 6. Ehrliche Gesamteinschätzung

Keine der 24 getesteten Varianten ist belegt: Keine erfüllt die vorab festgelegten Kriterien für „belegt“ UND schlägt zugleich die Marktdrift. Nach Kosten ist in diesen Daten kein verlässlicher Vorteil nachgewiesen.

**Wichtig zu S3l:** Die vorab festgelegten Kriterien für „belegt“ sind formal bestanden (der Netto-Ertrag nach Kosten war positiv), aber der Ertrag ist nur die Marktrendite. Gegenüber dem Durchschnitt des Universums im gleichen Zeitfenster beträgt der Excess im OOS 1,7 bp je Handelstag (t = 0,37), und die Basislinie „immer Long“ über dieselben Fenster erzielt 67,4 bp je Trade gegenüber 67,4 bp bei der eigenen Richtung. Der Ertrag stammt also aus der Marktdrift (ein steigender Aktienmarkt, plus Survivorship-Bias der heutigen Large Caps), nicht aus einem Können der Regel: kein gemessener Vorteil. Wer Long-Aktien hält, verdient dasselbe, ohne die Regel zu brauchen. Deshalb heißt S3l „nur Marktrendite“: keine Empfehlung, nur ein gekennzeichneter Beobachtungspunkt.

1 Variante(n) erreichten im Lernzeitraum (IS) einen t-Wert von mindestens 2 (S3l). Bei 24 Versuchen sind einzelne t-Werte um 2 allein durch Zufall zu erwarten. Maßgeblich ist der unberührte Testzeitraum (OOS): dort hatten 5 Variante(n) einen positiven Netto-Durchschnitt je Handelstag (D4, S2, S4, S3l, D5v).

Über die allgemeine Marktdrift hinaus (Excess gegenüber dem Durchschnitt des Universums) ließ sich im OOS bei keiner Variante ein Können mit t ≥ 2 belegen. Long-Regeln verdienen in einem steigenden Markt ohnehin etwas; das ist keine Fertigkeit.

Signifikant **negativ** (IS-t ≤ −2 und OOS-t ≤ −2) sind: D1 (t IS -2,07, OOS -2,11, gesamt -2,85; Ø brutto je Trade IS -3,7 bp, OOS 4,2 bp); H1 (t IS -8,46, OOS -7,21, gesamt -10,51; Ø brutto je Trade IS -1,1 bp, OOS 0,3 bp); H1b (t IS -3,47, OOS -2,75, gesamt -4,22; Ø brutto je Trade IS -2,9 bp, OOS -0,5 bp); H2 (t IS -3,30, OOS -5,09, gesamt -5,00; Ø brutto je Trade IS 0,7 bp, OOS -3,3 bp); M3 (t IS -2,35, OOS -2,07, gesamt -3,02; Ø brutto je Trade IS -0,8 bp, OOS -0,9 bp). Das ist überwiegend die Kostenwirkung und keine nachgewiesene Gegenrichtung: vor Kosten liegt der Ø brutto je Trade bei diesen Regeln nahe null (zwischen -3,7 und 4,2 bp), die Kosten (5 bp je Daytrade, 10 bp je Swing-Trade) machen das Ergebnis erst signifikant negativ. Die Umkehrung der Signale wäre deshalb kein Gewinn: auch die Gegenrichtung zahlt die Kosten.

## 7. Grenzen
- **Survivorship-Bias:** Das Universum sind die heutigen Large Caps, rückwirkend angewandt; Verlierer von gestern fehlen. Das begünstigt Long-Regeln und lässt Short-Ergebnisse eher zu schlecht aussehen (die Werte, auf deren Fall ein Short gesetzt hätte, fehlen).
- **Dividendenbereinigung (Datenfehler H-1, korrigiert):** Yahoo bereinigt Tagesbalken um Dividenden, Intraday-Balken nicht. Die erste Fassung verglich den (bereinigten) Tagesschlusskurs mit unbereinigten Intraday-Kursen; der „vorherige Schlusskurs“ lag zu Beginn des 60-Minuten-Fensters um bis zu 12 % zu niedrig (Median 370 bp in den ersten 20 Tagen), was H1/H1b fast immer auf „Long“ stellte und M2a/M2b verfälschte. Jetzt kommt der vorherige Schlusskurs aus dem letzten Intraday-Balken der Vorsitzung (gleiche Quelle, gleiche Bereinigung); Symbol-Tage ohne vorherige Intraday-Sitzung entfallen, ebenso solche, deren Vorsitzung nicht mit dem Schlussbalken endet (fehlender letzter Balken, 325 Symbol-Tage im 60-Minuten-Datensatz), weil deren letzter Kurs nicht der Schlusskurs ist. Restrisiko: Die Intraday-Daten enthalten Dividenden-Abschläge als echte Kurslücken (wie im realen Handel); Trendflags und ATR% stammen weiter aus bereinigten Tagesbalken (Verhältniszahlen, davon kaum berührt).
- **Yahoo-Datenqualität intraday:** fehlende Balken, Halbtage und gelegentliche Ausreißer; unvollständige Handelstage wurden ausgelassen (60 Min.: 32045 von 32500 Symbol-Tagen vollständig; 5 Min.: 2790 von 2795).
- **Keine Leerverkaufskosten:** Leihgebühren, Verfügbarkeit und Short-Squeezes sind nicht modelliert; das lässt Short-Ergebnisse eher zu gut aussehen, also in die entgegengesetzte Richtung wie der Survivorship-Bias.
- **Kleine 5-Minuten-Stichprobe:** nur etwa 43 Handelstage; Ergebnisse sind Anekdoten, kein Beweis.
- **Gap-Strategien auf Tagesbalken** (D1/D2) setzen voraus, dass man zum Eröffnungskurs handeln kann, den man erst kennt, wenn er entsteht; M2a/M2b prüfen das realistischer.
- **Swing-Trades überlappen:** Die Auswertung läuft deshalb auf dem Kalenderzeit-Portfolio (Abschnitt 1) statt auf Einstiegstagen. Auch die Tageswerte dieses Portfolios sind untereinander noch verwandt (Positionen laufen mehrere Tage); die t-Werte bleiben eher optimistisch. S1 verlässt die Position zum Schlusskurs, den man erst am Ende des Tages kennt.
- **Mehrfachtests:** Bei vielen Varianten entstehen Treffer zufällig. Der OOS ist nur ein einziger Zeitraum und kann selbst Zufall sein.
- **Simulation ≠ echte Ausführung:** Teilausführungen, Slippage in schnellen Märkten, Steuern und Verhalten sind nicht abgebildet.

## 8. Was daraus für die täglichen Empfehlungen folgt
Da keine Strategie belegt ist (Abschnitt 6), gibt es **keine** Kurzfrist-Empfehlung. Die tägliche Liste zeigt die Regeln nur als technisches Bild mit ihrem Etikett (Beobachtungsliste, nie als Empfehlung). „Kein Vorteil gefunden“ ist ein gültiges Ergebnis; Anlagehorizonte der übrigen Empfehlungen bleiben die der Analyse.
Formal bestandene Strategien ohne Vorteil gegenüber der Marktdrift (S3l, Etikett „nur Marktrendite“) sind **keine** Empfehlung. Das Programm zeigt sie nur als gekennzeichneten Beobachtungspunkt mit dem Pflichthinweis: „verdient nur, was der Markt in denselben Tagen ohnehin brachte; kein belegter Vorteil“.

## 9. Korrekturen nach der unabhängigen Prüfung
Eine unabhängige Prüfung hat die Zahlen der ersten Fassung nachgerechnet, die Schlussfolgerung bestätigt (nach Kosten zeigt keine Regel Können, nichts schlägt die Marktdrift, kein Look-ahead, der Simulator ist korrekt) und folgende Mängel gefunden, die jetzt behoben sind. Es wurde keine Strategie verändert oder hinzugefügt:

- **H-1 Datenfehler:** vorheriger Schlusskurs der Intraday-Regeln aus der vorherigen Intraday-Sitzung statt aus (dividendenbereinigten) Tagesbalken (Abschnitt 7).
- **H-2 Beschriftung:** Das vorab festgelegte Urteil bleibt unverändert, die Auszeichnung „belegt“ gilt aber nur noch, wenn die Regel auch die Marktdrift schlägt; sonst „formal bestanden, aber nur Marktrendite“. Die Kurzfrist-Ideen der Tagesliste folgen dem Etikett.
- **M-1 Überlappung:** t-Werte der Swing-Strategien jetzt auf dem Kalenderzeit-Portfolio statt nach Einstiegstag.
- **M-2 eine Größe:** Kostenbedingung und Marktdrift-Kriterium verwenden den Mittelwert derselben Reihe wie der t-Wert; „Ø je Trade“ bleibt als Information.
- Kleinere Punkte: Beschreibung der Baselines, Drawdown auf zusammengesetzter Kurve, Zählung des Universums, Wortlaut zu den signifikant negativen Regeln und zu den Variantenbegründungen, Alterskontrollen der Live-Ideen.

| ID | OOS Trades | OOS Ø je Trade (bp) | OOS Ø je Handelstag (Basis des t-Werts) (bp) | IS t | OOS t | Formales Urteil nach der vorab festgelegten Regel | Kennzeichnung im Programm |
|---|---|---|---|---|---|---|---|
| D1 | 6.890 | -0,8 | -7,2 | -2,07 | -2,11 | verworfen | verworfen |
| D2 | 617 | -13,9 | -25,0 | -0,67 | -1,91 | verworfen | verworfen |
| D3 | 472 | 2,9 | -8,1 | -2,07 | -0,74 | verworfen | verworfen |
| D4 | 757 | -3,5 | 0,9 | -1,04 | 0,12 | verworfen | verworfen |
| D5 | 2.110 | -4,2 | -1,2 | -0,87 | -0,25 | verworfen | verworfen |
| **H1** | 9.617 → 9.608 | -5,3 → -4,7 | -5,3 → -4,7 | -7,65 → -8,46 | -8,00 → -7,21 | verworfen | verworfen |
| **H1b** | 9.618 → 9.606 | -8,7 → -5,5 | -8,7 → -5,5 | -1,45 → -3,47 | -3,99 → -2,75 | verworfen | verworfen |
| **H2** | 6.061 | -8,3 | -9,4 | -3,38 → -3,30 | -5,09 | verworfen | verworfen |
| **H3** | 302 | -1,6 | -8,5 | -1,77 → -1,78 | -1,62 | verworfen | verworfen |
| **M1** | 624 | 0,1 | -3,0 | -1,38 → -1,39 | -0,30 | verworfen | verworfen |
| **M2a** | 255 → 259 | -7,8 → -9,5 | -10,0 → -11,4 | 0,00 → 0,44 | -0,67 → -0,76 | verworfen | verworfen |
| **M2b** | 15 | -85,4 → -85,5 | -183,4 | -1,41 → -1,44 | -2,48 | verworfen | verworfen |
| **M3** | 322 | -5,9 | -4,9 | -2,43 → -2,35 | -2,07 | verworfen | verworfen |
| **S1** | 2.430 | -2,0 | -15,7 → -0,7 | -1,20 → -0,15 | -1,71 → -0,22 | verworfen | verworfen |
| **S2** | 471 | 37,9 | 18,6 → 4,9 | -0,98 → -0,88 | 0,81 → 0,87 | verworfen | verworfen |
| **S3** | 1.287 | -18,9 | -11,8 → -1,6 | 0,13 → -0,42 | -0,57 → -0,54 | verworfen | verworfen |
| **S4** | 205 | 27,9 | 24,5 → 8,7 | 1,33 → 1,47 | 0,79 → 0,91 | verworfen | verworfen |
| **S4s** | 105 | 10,9 | -1,9 → -4,9 | 1,30 → 1,31 | -0,04 → -0,37 | verworfen | verworfen |
| **S2s** | 151 | -23,7 | -35,9 → -4,2 | -0,45 → -0,64 | -0,96 → -0,49 | verworfen | verworfen |
| **S3l** | 655 | 67,5 | 86,1 → 9,6 | 2,98 → 2,47 | 2,91 → 2,06 | belegt | formal bestanden, aber nur Marktrendite |
| **S1x** | 1.182 | -0,8 | -16,7 → -1,7 | 0,27 → 0,42 | -1,35 → -0,37 | verworfen | verworfen |
| **S1v** | 1.477 | 1,3 | -12,3 → -0,3 | -0,77 → -0,12 | -0,99 → -0,07 | verworfen | verworfen |
| D5v | 1.241 | -4,2 | 2,0 | -0,03 | 0,31 | verworfen | verworfen |
| H3x | 14 | -28,8 | -26,6 | -0,54 | -1,29 | verworfen | verworfen |

Die Spalte „Formales Urteil“ zeigt nur, ob die vorab festgelegten Kriterien erfüllt sind; „belegt“ dort ist keine Empfehlung. Maßgeblich ist die Spalte „Kennzeichnung im Programm“: Eine Regel, die die Marktdrift nicht schlägt, erscheint dort als „formal bestanden, aber nur Marktrendite“, und keine Variante ist belegt.

Fett = Zeile hat sich geändert (Trade-Zahl, IS-t, OOS-t, Urteil oder Etikett). „Vorher“ ist der Stand vor der Korrektur (Datei `validated_before_review.json`). Bei „vorher“ war „Ø je Handelstag“ der Mittelwert der Tages-Cluster nach Einstiegstag, „nachher“ derselbe Wert (Tagestrader) bzw. der des Kalenderzeit-Portfolios (Swing).

Geänderte Zeilen: H1, H1b, H2, H3, M1, M2a, M2b, M3, S1, S2, S3, S4, S4s, S2s, S3l, S1x, S1v.
