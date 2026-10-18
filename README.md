# Doppelt robust und Double ML – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-doppelt-robust-demo.streamlit.app/)**

Siebtes Stück der Linie **Kausale Diagramme und Adjustierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Nachfolger von [schaetzen-demo](https://github.com/sebastian-hanisch/schaetzen-demo): Dort stand, dass jeder Einzelschätzer ein zweites Modell braucht. Hier steht der Schätzer, der beide Modelle kombiniert und nur eines davon richtig braucht, und die Kreuzanpassung, die erlaubt, beide mit einem flexiblen Lernverfahren zu schätzen.

## Kernfrage

Der doppelt robuste Schätzer (AIPW) ist richtig, wenn Ergebnis- **oder** Zuteilungsmodell stimmt; Double ML erlaubt, beide Störfunktionen zu lernen, wenn der Fehler nur als Produkt eingeht und mit Kreuzanpassung berechnet wird. Was davon trägt, wenn das Lernverfahren ein Zufallswald ist, die Funktionen nichtlinear sind und n klein bleibt?

## Modell

- **Vehikel:** ein Standort wählt die neue Regel B (T = 1) abhängig von fünf Störfaktoren X₁ bis X₅; Ergebnis ist der Servicegrad. P(T = 1 | X) = Φ(γ s(X)), Y = m₀(X) + 0,5 T + Rauschen, konstante Wirkung **0,5** (die Wahrheit ist exakt). s und m₀ enthalten nichtlineare Terme (X₂², X₄X₅, sin 2X₃) mit der Stärke A, die in **beiden** Funktionen stehen und daher konfundieren. A ist der Schwierigkeitsregler (mild 0,4, schwer 1,0).
- **Daten:** `SplitMix64`-Ströme (Generator `kausal_generator.py`, wortgleich in allen Stücken der Linie).
- **Lernverfahren** (`dml_learners.py`): **Linear** (beide Modelle falsch), **Richtige Basis** (lineare Modelle auf genau den Merkmalen der wahren Funktionen, beide richtig) und **Zufallswald** (scikit-learn, 30 Bäume, Blattgröße 5), je für beide Ergebnisfunktionen und die Zuteilung.

## Methodik

- **Schätzer** (`dml_estimators.py`): naiver Vergleich; Plug-in (Mittel von m̂₁ − m̂₀); normiertes Gewichten; AIPW ohne Kreuzanpassung (Störfunktionen aus dem Modell, das auf denselben Daten lernte); AIPW mit Kreuzanpassung (K Falten); partiell lineares DML (Robinson) mit kreuzangepassten Resten.
- **Intervalle:** Einflussfunktion (Streuung der Einflusswerte durch √n); keine Bootstrap-Wiederholung.
- **Experimente** laufen über feste Zufallsströme; die zwei Wald-Experimente teilen ihre Läufe und starten auf Knopfdruck.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Wahrheit 0,50, Verzerrung = Mittel minus Wahrheit.

| Frage | Befund | Test |
|---|---|---|
| Standard-Stichprobe (n = 1 000, Wald, K = 5, A = 0,4, Seed 35) | Naiv +1,28; Plug-in 0,85; Gewichten 1,03; AIPW ohne Kreuzanpassung 0,81 (Intervall deckt nicht); **AIPW mit Kreuzanpassung 0,60 (deckt)**; partiell linear 0,70. Wurzel des Fehlers der Zuteilung 0,16 in der Stichprobe, 0,13 kreuzangepasst. | `test_default_sample` |
| Ein richtiges Modell genügt? (A = 1,0, n = 500, 500 Wiederholungen, parametrische Modelle) | Beide falsch: AIPW **+1,39** (Deckung 0 %). Nur Ergebnismodell richtig: AIPW **+0,01** (Deckung 90 %), das Gewichten allein bleibt bei +1,38. Nur Zuteilung richtig: AIPW **+0,19** (Deckung 85 %). Beide richtig: +0,01. | `test_experiment_1_double_robustness` |
| Kostet die Doppelsicherung etwas? | Bei richtigen Modellen streut der AIPW mit **0,151**, das Plug-in mit 0,115; bei knapper Überlappung (effektive Stichprobe der wahren Gewichte **245** von 500) ist er mit richtiger Zuteilung und falschem Ergebnismodell (SD 0,286) schlechter als das richtige Ergebnismodell allein (0,115). | `test_experiment_1_price_of_the_insurance` |
| Warum geht der Fehler als Produkt ein? (Population, n = 200 000) | Störung 0,4 im Ergebnismodell: Plug-in **+0,40**, AIPW −0,002 (gegenüber ohne Störung). Störung in beiden: AIPW-Zuwachs 0,0045 bei 0,2 und **0,0194** bei 0,4 (Faktor 4,3), das Plug-in wächst linear. | `test_experiment_2_orthogonality` |
| Was macht die Kreuzanpassung? (Wald, A = 0,4, 16 Wiederholungen je Zeile) | n = 500: AIPW ohne +0,25 (Deckung **19 %**), mit K = 5 +0,13 (**94 %**). n = 1 000: ohne +0,21 (**0 %**), mit +0,05 (**94 %**). Überanpassender Wald (Blattgröße 1, n = 500): ohne +0,25 (6 %), mit etwa −0,3 bei Streuung von etwa **0,7** (mit 16 Wiederholungen auf Plattform und Bibliotheksversion schwankend). Die Deckungen sind Zählungen von 16 Läufen; ein Gegenlauf mit je 100 Wiederholungen und zwei anderen Seeds ergab bei n = 1 000 ohne Kreuzanpassung 7 % und 4 %, mit ihr 93 % und 98 %. | `test_experiment_3_cross_fitting` |
| Reicht der Wald? (mild, n = 500 / 1 000 / 2 000) | AIPW mit Kreuzanpassung: Verzerrung **+0,13 → +0,05 → −0,01** (aus 16 Läufen, großenteils Stichprobenrauschen: ein Gegenlauf mit 64 Wiederholungen und anderen Seeds ergab +0,045 / +0,044 / +0,027), RMSE·√n **3,89 / 3,85 / 4,10** (bleibt bei etwa 4, mit großer Streuung aus 16 Läufen; vereinbar mit √n-Konsistenz). Plug-in-Wald **+0,29 → +0,17**, RMSE·√n 6,85 → 7,73; AIPW linear +0,28 bis +0,31, RMSE·√n 7,28 → 12,47; richtige Basis +0,01, +0,02 und −0,01, RMSE·√n 2,68 / 2,99 / 2,13. | `test_experiment_4_mild_and_hard` |
| Und im schweren Vehikel (A = 1,0)? | AIPW mit Kreuzanpassung bleibt bei **+0,51** (n = 500, Deckung 25 %) und **+0,39** (n = 1 000, Deckung 31 %); Plug-in-Wald +0,93 und +0,78; linear +1,41 und +1,35; richtige Basis +0,01. | `test_experiment_4_mild_and_hard` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Alle Störfaktoren sind beobachtet** | Ein verborgener Störfaktor verzerrt auch den doppelt robusten Schätzer; Kreuzanpassung ändert daran nichts. | Frontdoor, Instrumente, Sensitivität und Falsifikation |
| **Eines der Modelle stimmt oder beide sind gut genug** | Sind beide falsch oder zu schlecht gelernt, bleibt eine Restverzerrung (Linear, schwere Nichtlinearität). | Wie falsch darf das Diagramm sein? |
| **Überlappung: die Gewichte bleiben zahm** | Bei knapper Überlappung wird die Korrektur unruhig. | Schätzen (Stück 6) |
| **Das Lernverfahren ist gut genug** | Die Theorie verlangt, dass das Produkt der Fehler schneller als 1/√n fällt; der Wald mit 30 Bäumen im schweren Vehikel erfüllt das bei den gemessenen n (500, 1 000) nicht erkennbar. Die Hyperparameter sind nicht abgestimmt; Boosting war in einer nicht dokumentierten Vormessung als Zuteilungsmodell instabil und ist nicht aufgenommen. | nicht gebaut |
| **Konstante Wirkung (partiell lineares DML)** | Das Robinson-Verfahren schätzt sonst eine gewichtete Wirkung; die Demo hat nur eine konstante. | Heterogene Wirkungen sind nicht gebaut |
| **n ist groß genug für die Falten** | Bei kleinem n sieht jedes Modell zu wenige Daten; die Intervalle können zu selten decken (nicht gemessen). | Welche Menge? (Stück 5) |

Die Zahlen gelten für dieses erfundene Vehikel (fünf Störfaktoren, probit-Zuteilung, Normalrauschen); die Wald-Läufe haben 16 Wiederholungen je Zeile, Deckungen sind daher auf etwa ± 10 Prozentpunkte genau. „Wahrheit“ ist das simulierte Modell, nicht ein Befund über echte Standorte. Die Quellen im 📐-Expander sind aus dem Gedächtnis, nicht geprüft.

## Bezug zu den Nachbarn im Portfolio

- [off-policy-demo](https://github.com/sebastian-hanisch/off-policy-demo): Doppelt robust als „Modell plus gewichtete Korrektur“ mit **bekannter** Protokoll-Wahrscheinlichkeit und diskreten Kontexten. Hier ist die Zuteilung **unbekannt und gelernt**, die Störfaktoren stetig und mehrdimensional; neu sind Kreuzanpassung und Orthogonalität. Kein Neubau von IPS und SNIPS.
- [schaetzen-demo](https://github.com/sebastian-hanisch/schaetzen-demo): Einzelschätzer (Regression, Gewichten, Schichten, Matching) mit einem Störfaktor; sein README nennt Doppelt robust als nächstes Stück.
- [adjustierungsmenge-demo](https://github.com/sebastian-hanisch/adjustierungsmenge-demo): wählt die Menge; hier ist sie fest (alle fünf Störfaktoren).

## Tests

Mini-Instanzen von Hand (vier Einheiten: naive Differenz 1,5, Plug-in 1, ψ = (3, −1, 1, 3), AIPW 1,5), AIPW, Kreuzanpassung und partiell lineares DML gegen Schleifen und Normalgleichungen, Probit gegen scipy, Wald gegen den direkten Aufruf von scikit-learn, doppelte Robustheit und Orthogonalität in großen Stichproben, Auswertung, jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler, Permalink, PDF, die zwei Wald-Experimente auf Knopfdruck), Footer und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `kausal_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Normalverteilung, Strukturmodell) |
| `dml_world.py` | Vehikel: fünf Störfaktoren, Zuteilung, Ergebnis, wahre Störfunktionen |
| `dml_learners.py` | Lernverfahren: linear, richtige Basis, Zufallswald; Probit |
| `dml_estimators.py` | Plug-in, Gewichten, AIPW, Kreuzanpassung, partiell lineares DML |
| `dml_evaluation.py` | Einstellungen, Analyse, vier Experimente, Verdikte |
| `dml_visualization.py`, `dml_pdf_export.py` | Figuren, Tabellen, PDF |
| `dml_constants.py`, `dml_presets.py` | Regler, Experimente, Presets, Permalink |
| `tests/` | Mini-Instanzen, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Heterogene Wirkungen (CATE), kausale Wälder:** eigenes Thema; hier konstante Wirkung.
- **Abstimmung der Lernverfahren, Boosting, neuronale Netze, Stacking:** ein Wald genügt, um das Prinzip zu zeigen; kein Wettbewerb der Lernverfahren.
- **Bootstrap-Intervalle:** würden die Kreuzanpassung vielfach rechnen; die Einflussfunktion ist der Standard der Theorie.
- **Verborgene Störfaktoren, Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas, scikit-learn und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Kausale Diagramme: von Konfundierung bis Wahl](https://sebastianhanisch.net/konzepte-kausale-diagramme.html).
