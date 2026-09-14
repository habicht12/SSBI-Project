**Vollständiger Klassifikationsbenchmark und Abgleich der Gesamtpräsentation**

Geprüfter Stand: `main`, Commit `df5a10a`. Gesamtpräsentation:
`praesentation/main.pdf` mit 98 PDF-Seiten, letzter Änderungsstand `4045b3f`
vom 12.09.2026. Prüfung am 12.09.2026. PDF-Seitenzahlen unten zählen die
Titelseite mit; die eingeblendete Foliennummer ist jeweils um eins kleiner.

**Ergebnis:** Der vollständige Klassifikationsvergleich ist mit den ausdrücklich
zur Wiederverwendung freigegebenen, passenden Modellen abgeschlossen. Die
Ergebnisse für Aufgaben 3–6 stimmen mit den geprüften Folien überein. Aufgabe 2
enthält einen älteren UMAP-Stand und eine falsche t-SNE-Parameterliste. Das neueste
Aufgabe-2-Notebook enthält selbst Ausgaben unterschiedlicher Ausführungsstände;
ein bloßer Vergleich seiner sichtbaren Ergebnistabellen reicht daher nicht aus.

**Ausgeführter Umfang und Nachweise**

Alle Ausgaben liegen isoliert unter
[tables/benchmark_main_20260912](tables/benchmark_main_20260912).
Die Quellnotebooks und die Gesamtpräsentation wurden nicht verändert.

- `04a` wurde neu ausgeführt. Die 100 neu erzeugten spenderweisen Splitlisten
  stimmen bytegenau mit den bisherigen überein.
- `04b`, `04c`, `04d` und `04e` wurden jeweils mit frischem Kernel im Full-Modus
  ausgeführt: SVM/CellCNN jeweils 100 Splits, Citrus 30 Splits. Der
  Dreiervergleich nutzt dieselben 30 Splits; der zusätzliche SVM-/CellCNN-Vergleich
  alle 100. Pro Split: 14 äußere Trainingsspender, sechs Testspender, davon
  zwei CMV-positive und vier CMV-negative. Klassifikationsgate: `gated_alive`.
- `06a`, `06d`, `06b` mit lernbarem Pooling und `06e` wurden mit frischen Kerneln
  über **jeweils 100 Splits** ausgeführt. Der Runner überschreibt nur die
  bisherigen Standard-Splitlisten von drei bzw. zehn Splits und den Pfad zur
  Hard-Pooling-Referenz. Modell- und Trainingscode bleiben unverändert.
- Daten-, Split-, Implementierungs- und Konfigurationsnachweise wurden vor der
  Wiederverwendung geprüft. Alle 300 Bonus-Checkpoints wurden geladen,
  Kandidatenauswahl, Vorhersagen und Populationshäufigkeiten erneut geprüft.
  Die kleinen eingebauten Modell-/Smoke-Tests wurden ebenfalls ausgeführt.
  Es erfolgte **kein neues vollständiges Modelltraining**.
- Alle 600 SVM-Testvorhersagen wurden zusätzlich direkt aus Originalzellen,
  gespeichertem Scaler und Gewichten rekonstruiert: maximaler absoluter
  Scorefehler **0,0**. Die 600 CellCNN-Baseline-Vorhersagen wurden ebenfalls
  ohne Abweichung rekonstruiert. Die Bonusnotebooks prüfen ihre rekonstruierten
  Vorhersagen mit Toleranz `1e-6`.
- ROC-AUC, Average Precision und Balanced Accuracy wurden für alle **530
  Modell-Splits** zusätzlich mit unabhängigen arithmetischen Formeln geprüft:
  AUC durch positive/negative Scorepaare einschließlich halber Wertung bei
  Gleichständen, AP über die Präzision bei jeder Score-Schwelle, BA aus den
  beiden klassenweisen Trefferquoten.
- `05_interpretation.ipynb` wurde neu ausgeführt. Zusätzlich wurden die
  SVM-Markerprofile aller 180 Testauftritte und die 24 Werte der
  gemeinsamen Marker-Heatmap nachberechnet.
- Bei Aufgaben 2/3 wurden gespeicherte Ergebnisse und ihre Herkunft geprüft;
  t-SNE/UMAP und das Clustering wurden **nicht neu trainiert**. Für Aufgabe 2
  wurden kNN-Übereinstimmung, Distanzkorrelation und Procrustes auf den
  gespeicherten 40.000-Zell-Koordinaten erneut berechnet. Die vier zugehörigen
  Eingabedateien stimmen mit den SHA-256-Nachweisen des Folienexports überein.
- Der tatsächliche Text der 98-seitigen PDF wurde extrahiert und mit LaTeX,
  eingebundenen Tabellen, Abbildungen und den Notebook-Ausgaben abgeglichen.
  Dies ist ein Ergebnis-/Herkunftsabgleich, keine erneute Layoutprüfung oder
  unabhängige biologische Validierung.

Der vollständige Klassifikationsrunner einschließlich Wiederherstellung und
Vergleichsexport benötigte rund **195 Sekunden**. Separate Datenprüfung,
Interpretation und zusätzliche Audits sind darin nicht enthalten. Die kurzen
Zeiten entstehen durch Wiederverwendung, nicht durch neue Trainingsläufe.

**Ergebnisse des Klassifikationsbenchmarks**

| Modell | Splits | Mittlere ROC-AUC | Mediane ROC-AUC |
| --- | ---: | ---: | ---: |
| CellCNN | 30 | 0,8250 | 0,8750 |
| Lineare Single-Cell-SVM | 30 | 0,8083 | 0,8750 |
| Citrus | 30 | 0,6063 | 0,6250 |
| CellCNN | 100 | 0,8075 | 0,8750 |
| Lineare Single-Cell-SVM | 100 | 0,7838 | 0,8750 |
| Quadratic Top-1 % | 100 | 0,9125 | 1,0000 |
| Quadratic Soft-α | 100 | 0,9125 | 1,0000 |
| Mahalanobis/ReLU Soft-α | 100 | 0,7375 | 0,7500 |

Die Aufgaben-4-Folien berichten **Mediane**, die zentrale Aufgabe-6-Folie
zusätzlich **Mittelwerte**. Beispielsweise sind 0,875 und 0,8075 für CellCNN
über 100 Splits beide korrekt; sie bezeichnen unterschiedliche Kennzahlen.
Wiederholte Splits bleiben Auswertungen derselben 20 unabhängigen Spender.

**Abgleich nach Aufgaben und Herkunft der Abweichungen**

| Bereich / PDF-Seiten | Befund | Nachgewiesene Herkunft |
| --- | --- | --- |
| Aufgabe 2: PCA, S. 4 | 29 PCs erklären 91,50 %, die ersten zwei 20,29 %; passt zum gespeicherten Referenzstand. Der neueste Code berechnet zunächst 36 statt früher 30 PCs; das ändert nicht die dokumentierte Auswahl von 29. | Älterer Export auf `slides-aufgabe-2`; aktuelle gespeicherte Auswahl ebenfalls 29 PCs. |
| Aufgabe 2: t-SNE, S. 5 | Die Tabelle und Wahl von Perplexität 60 passen. Die ausgeschriebene Liste mit **acht** Werten einschließlich **15 und 40** passt nicht: `_clean` prüft **sechs** Werte `[5, 30, 50, 60, 70, 100]`. | Sechs Werte sowohl auf aktuellem `main` als auch `slides-aufgabe-2`. `02_dimensionality_reduction2.ipynb` prüft fünf Werte `[5, 15, 30, 50, 100]`; auch das erklärt die Folienliste nicht. In der geprüften Git-Historie wurde kein passender Acht-Werte-Lauf mit 40 gefunden. |
| Aufgabe 2: UMAP, S. 6 und 61–62 | Die Folien beziehen sich auf **16** Kombinationen: Nachbarn `[5,15,30,50]`, `min_dist=[0,0.1,0.3,0.5]`, Standard-Lernrate 1. Der aktuelle Code untersucht **64** Kombinationen: Nachbarn `[5,15,50,100]`, `min_dist=[0,0.1,0.5,1]`, Lernrate `[0.1,0.5,1,5]`. | Exakter älterer Notebook-SHA `69e81282d26d…` aus `slides-aufgabe-2`, Commit `1bbe6f7`, entspricht dem Folien-Provenienznachweis. Der 16er-Sweep findet sich auch in der späteren `main`-Historie, z. B. `8c5c977`. Der 64er-Stand kam mit `4045b3f`. |
| Aufgabe 2: bester UMAP-Wert, S. 6 | Folie: **T = 0,890139** bei 5 Nachbarn, `min_dist=0`. Neuester gespeicherter Sweep: **T = 0,895908**, zusätzlich **Lernrate 0,1**. Nachbarn und Mindestabstand bleiben gleich, Lernrate und T unterscheiden sich. | Der Folienwert entspricht exakt der älteren Lernrate 1 und erscheint sogar als solche Vergleichszeile im neuen Notebook. Kein ungeklärter Zahlenwert. |
| Aufgabe 2: Qualitäts-/Paarvergleich und Karten, S. 7–8, 63–67 | Die Zahlen passen zu den alten gespeicherten 40.000-Zell-Koordinaten und wurden dafür erneut bestätigt. Sie stehen weiterhin im neuesten Notebook, sind aber **nicht als durchgängiges Ergebnis des neuen Lernratenlaufs belegt**. | Zahlentabellen stimmen mit `slides-aufgabe-2` überein; die betroffenen Qualitäts-/Paarzellen haben im aktuellen Notebook keine Ausführungsnummer. Die übernommenen Karten-Ausgaben sind teilweise exakt die älteren. |
| Aufgabe 3, S. 9–21 und 68–73 | Passt zum aktuellen Notebook: 20.000 Zellen, 44 Konfigurationen; K-Means k=4, Ward k=6, Leiden 18 Cluster. Tabellen, Markerprofile und beide originalen K-Means-/Leiden-UMAPs stimmen bytegenau mit einem erneuten Export der gespeicherten Ausgaben überein. | `main:03_clustering.ipynb`, letzte Notebook-Änderung `27949d2`; vollständiger SHA `31d6fa89…` entspricht dem Folien-Provenienznachweis. Nicht das ältere 10.000-Zell-Notebook auf `GrHa`. |
| Aufgabe 4, S. 22–27 und 74–77 | Alle gedruckten AUC-/AP-/BA-Mediane über 30 bzw. 100 Splits passen. Hyperparameter, Testspenderzahlen und Methodenbeschreibungen entsprechen den Implementierungen. | Aktuelle `04a`–`04e` auf `main`, aus `GrHa`; vollständige Konfigurationsprüfung und erneute gemeinsame Auswertung bestanden. |
| Aufgabe 5, S. 28–33 und 78–86 | Zahlen und Markerprofile passen zur erneut ausgeführten Interpretation. | Aktuelles `05_interpretation.ipynb` aus `GrHa`. Die 10.000-Zell-Karte stammt vom früheren `GrHa`-Aufgabe-2-Notebook, jetzt `02_interpretation_reference.ipynb`. |
| Aufgabe 6, S. 34–40, 42–54 und 87–98 | Ergebnisse und beschriebene Modell-/Poolingdefinitionen passen zum vollständigen Vier-Modell-Vergleich. Alle sechs eingefrorenen Vergleichstabellen stimmen mit der erneuten Auswertung bei `atol=rtol=1e-12` überein. | Vollständiger Lauf `GrHa-learnable-pooling`, Commit `1f00db2`, Ordner `results/tables/task6_comparison_100/`. Mit den auf `main` übernommenen Modellen und expliziten 100 Splits reproduziert. |

**Konkrete bestätigte Details zu Aufgaben 5/6**

Aufgabe 5 ergibt unverändert 130 CellCNN- und 123 Citrus-Zentroiden.
Die gezeigten Gruppen kommen bei CellCNN in 30/20/15/8 und bei Citrus in
18/11/10/6/6 der 30 Splits vor. Es gibt 87 ausgewählte CellCNN-Karten-Zellen
und 93 mindestens einmal positiv ausgewählte SVM-Karten-Zellen. Die
Testauftritte je Spender reichen von vier bis 16. Alle vier neu erzeugten
Interpretationstabellen stimmen bytegenau mit den früheren überein.

Die Heatmap verwendet unverändert 168 nichtleere von 180 SVM-Testauftritten
und gewichtet anschließend die 20 Spender gleich. Alle 24 gezeigten
Markerwerte wurden bestätigt. Auch die im PDF verwendeten Aufgabe-4/5-Assets
stimmen mit den SHA-256-Werten ihres ursprünglichen Exports überein.

Aufgabe 6 hat 98 gemeinsam bestimmbare Frequency-Splits: Die Baseline hat
in Splits 44 und 84 keinen positiven Output-Kontrast. Das Fehlen bleibt
fehlend und wird nicht als Nullfrequenz behandelt. Die Mean-Network-AUCs
0,8075/0,9125/0,9125/0,7375, die 53/37/10 besseren/gleichen/schlechteren
Soft-Quadratic-Splits gegenüber CellCNN, die mittlere AUC-Differenz +0,1050
und die 416/414 gezeigten Alpha-Parameter sind bestätigt.

Abweichungen gegenüber den historischen Drei-/Zehn-Split-Ausgaben sind keine
Folienfehler: Beispielsweise gehört die frühere 06e-AUC 0,9500 auf `GrHa`
zu zehn Splits, die Folien-AUC 0,9125 zu 100 Splits. Auch das historische
`06b_ergebnisse_100_splits.ipynb` beschreibt eine andere Bonusvariante und
ist nicht die Quelle der aktuellen Vier-Modell-Folien.

**Was vor einer Aktualisierung der Folien zu klären bzw. zu korrigieren ist**

1. Auf S. 5 die t-SNE-Liste auf die tatsächlich geprüften sechs Werte berichtigen.
2. Für Aufgabe 2 einen konsistenten Stand verwenden: entweder ausdrücklich
   den älteren 16er-UMAP-Sweep als Vortragsgrundlage beibehalten oder den neuen
   64er-Lauf mit Lernrate 0,1 vollständig bis zu Karten, Qualitätsmaßen und
   Exporten ausführen und anschließend die betroffenen Folien gemeinsam
   aktualisieren. Nur die Zahl 0,890139 durch 0,895908 zu ersetzen reicht nicht.
3. Verweise in den Aufgabe-5-Backups auf „the Task 2 map“ präzisieren:
   Gemeint ist die ältere `gated_alive`-Interpretationskarte mit 10.000 Zellen,
   nicht die derzeitigen Aufgabe-2-Folien mit 40.000 `gated_NK`-Zellen.
   Die Datenbasis ist in den Aufgabe-5-Folien bereits numerisch beschriftet;
   der vereinfachte Aufgabenverweis ist dennoch missverständlich.

Diese Prüfung verändert weder Folien noch Analyseentscheidungen. Es wurden
keine Daten oder Checkpoints committed und kein Git-Push ausgeführt.

**Dateien zur Nachprüfung**

- [Laufstatus](tables/benchmark_main_20260912/benchmark_status.json),
  [Quellstände und Laufumfang](tables/benchmark_main_20260912/run_scope.json),
  [ausgeführte Notebooks](tables/benchmark_main_20260912/executed).
- [Unabhängige Metriken je Split](tables/benchmark_main_20260912/audit/independent_split_metrics.csv),
  [Aufgabe-4-Zusammenfassung](tables/benchmark_main_20260912/audit/task4_summary.csv),
  [Aufgabe-6-Zusammenfassung](tables/benchmark_main_20260912/results/tables/task6_comparison_100/comparison/summary.csv).
- [Ergebnis-/Assetprüfungen](tables/benchmark_main_20260912/audit/benchmark_checks.json),
  [Clustering-Abgleich](tables/benchmark_main_20260912/audit/task3_export_checks.json),
  [Task-2-Tabellen mehrerer Stände](tables/benchmark_main_20260912/audit/task2_stored_tables.json).
- [Neu berechnete Paarvergleiche](tables/benchmark_main_20260912/audit/task2_pairwise_recomputed.csv),
  [SVM-Rekonstruktion](tables/benchmark_main_20260912/audit/svm_reconstruction.csv),
  [Markerprofile](tables/benchmark_main_20260912/audit/marker_profiles_recomputed.csv).
