# Vier CellCNN-Varianten auf 100 gemeinsamen Spendersplits

Abgeschlossener Lauf auf `GrHa-learnable-pooling`: **Splits 0–99**, `gated_alive`,
20 unabhängige Spender, 37 Marker. Alle vier Modelle verwenden dieselben
spenderweisen Splits, dieselbe Datenvorverarbeitung und dieselbe strikte
Half-Max-Populationsdefinition. Es wurde kein Clustering ausgeführt.

## Einstieg

- [Vergleichsfolien als PDF](../../../praesentation/aufgabe_06_vergleich/slides.pdf)
- [Zusammenfassung aller Metriken](comparison/summary.csv)
- [Metriken je Methode und Split](comparison/split_metrics.csv)
- [Gepaarte AUC-Differenzen](comparison/paired_differences.csv)
- [Laufumfang und Quellstände](run_scope.json)
- [Nachweis der Plotdaten und Aggregation](comparison/provenance.json)

| Modell | Quellnotebook | Ausgeführte 100-Split-Fassung | Einzelmodellplots |
|---|---|---|---|
| CellCNN | [06a](../../../notebooks/06a_cellcnn_baseline_bonus.ipynb) | [Notebook](cellcnn_100_executed.ipynb) | [Dashboard](plots/cellcnn/dashboard.png) |
| Quadratic Top-1 % | [06d](../../../notebooks/06d_cellcnn_quadratic.ipynb) | [Notebook](quadratic_top1_100_executed.ipynb) | [Dashboard](plots/quadratic_top1/dashboard.png) |
| Quadratic Soft-α | [06e](../../../notebooks/06e_cellcnn_quadratic_learnable_pooling.ipynb) | [Notebook](quadratic_soft_100_executed.ipynb) | [Dashboard](plots/quadratic_soft/dashboard.png) |
| Prototyp/ReLU Soft-α | [06b](../../../notebooks/06b_cellcnn_mahalanobis_learnable_pooling.ipynb) | [Notebook](prototype_soft_100_executed.ipynb) | [Dashboard](plots/prototype_soft/dashboard.png) |

## Ergebnisse und Abbildungen

Die folgenden Werte sind Mittelwerte über Splits. Klassifikationsmetriken
verwenden alle 100 Splits; Frequency-Metriken verwenden dieselben **98
bei allen vier Methoden bestimmbaren Splits**. Frequency-Effect ist
CMV+-Mittel minus CMV−-Mittel als Anteil, nicht in Prozentpunkten.

| Modell | Network-AUC | AP | BA | Frequency-AUC | Frequency-Effect |
|---|---:|---:|---:|---:|---:|
| CellCNN · Top-1 % | 0,8075 | 0,8038 | 0,6787 | 0,8559 | 0,007403 |
| Quadratic · Top-1 % | 0,9125 | 0,9070 | 0,8013 | 0,8610 | 0,005698 |
| Quadratic · Soft-α | 0,9125 | 0,9058 | 0,7950 | 0,8559 | 0,005351 |
| Prototyp/ReLU · Soft-α | 0,7375 | 0,7207 | 0,6500 | 0,7245 | 0,003004 |

Verfügbare Frequency-Splits vor gemeinsamer Einschränkung:
CellCNN · Top-1 %: 98/100; Quadratic · Top-1 %: 100/100; Quadratic · Soft-α: 100/100; Prototyp/ReLU · Soft-α: 100/100.
Fehlende Populationen bleiben NaN und werden nicht als Frequency null gewertet.
Ein vorhandener Filter mit leerer Auswahl hat dagegen Frequency null.

- [Netzwerk-ROC](comparison/network_roc.png) und [Frequency-ROC](comparison/frequency_roc.png)
- [Klassifikationsmetriken](comparison/network_metrics.png): ROC-AUC, Average Precision, Balanced Accuracy
- [Frequency-AUC und -Effect](comparison/frequency_metrics.png)
- [Gepaarte Differenzen](comparison/paired_differences.png)
- [Donorfrequenzen auf gemeinsamer Y-Skala](comparison/donor_frequencies.png)
- [Gelernte Poolinganteile](comparison/learned_alphas.png)

Jede Abbildung liegt zusätzlich als Vektor-PDF vor. ROC-Kurven werden zuerst
je Split berechnet, auf eine gemeinsame FPR-Achse interpoliert und dann
gemittelt. Die Legende berichtet die mittlere ursprüngliche Split-AUC; es wird
keine ROC über alle 600 Testauftritte gepoolt. Boxen zeigen Q1–Q3 und Median,
Punkte die einzelnen Splits. Dies sind keine Konfidenzintervalle.

Die Donorplots mitteln zunächst die verfügbaren Testfrequenzen jedes Spenders;
jeder Spender erscheint einmal. Die Zellpopulation kann zwischen ausgewählten
Modellen/Splits variieren. Einzelmodell-Dashboards verwenden alle für das
jeweilige Modell verfügbaren Frequency-Splits; der Vierervergleich verwendet
die gemeinsame Schnittmenge. Das erklärt gegebenenfalls unterschiedliche
Frequency-Mittelwerte zwischen beiden Darstellungen.

## Gemeinsame Methodik

Pro Modell wird der Filter mit größtem **positiven** Output-Kontrast
`V[1,k] - V[0,k]` ausgewählt. Über alle Zellen seiner tatsächlichen
Inner-Train-Spender wird `M = max(h)` bestimmt; anschließend gilt auf jedem
Testspender dieselbe Schwelle `h > 0.5*M`. Die strikte Regel gilt auch bei M=0.
Inner-Validation und Outer-Test bestimmen die Schwelle nicht.

Die Frequency ist `n_selected_cells / n_cells` auf derselben festen Stichprobe
von bis zu 20.000 Zellen je Testspender (Seed 63000 plus sortierter Donorindex).
Ohne positiven Filter bleiben die Phenotype-Metriken fehlend. Eine AUC unter
0,5 wird nicht umgedreht. CMV-Donorlabels sind keine Zelltyp-Ground-Truth.
Die kombinierte [Frequency-Tabelle](comparison/frequencies.csv) enthält die
einheitlich benannte `halfmax_threshold`, Trainingsspender und Zellzahlen.

Transformation `arcsinh(x/5)`, Trainingsscaler, Bags, Seeds, Optimierer,
Lernrate, Early Stopping, Filterzahlen und Kandidatenauswahl bleiben gegenüber
den vorhandenen Implementierungen unverändert. Je Split werden neun innere
Kandidaten verglichen; das ausgewählte Inner-Modell wird ohne Refit getestet.
Die Baseline-Architektur und das Training stehen weiterhin in `04c_cellcnn.ipynb`.

Die Regularisierung unterscheidet sich weiterhin zwischen den Modellfamilien:
CellCNN: L2 auf W und V; Quadratic: L2 auf W, Q und V; jeweils 1e-4.
Prototyp: Output-L2 1e-4 plus 1e-3 mal die mittlere quadratische Abweichung
der normalisierten Markergewichte von eins. Keine zusätzliche Radius- oder
Alpha-Strafe. Quadratic Hard vs. Soft vergleicht die gesamte Umstellung auf
weiches lernbares Pooling, nicht isoliert das Lernen von Alpha bei fester Maske.

## Ausführung und Wiederverwendung

Die Baseline hatte bereits 100 ausgewählte Modelle in ihren Filter-CSV-Dateien.
Wiederverwendet wurden außerdem zehn Quadratic-Hard-, 50 Quadratic-Soft- und
50 Prototyp-Checkpoints. **190 fehlende Modell-Splits** wurden neu trainiert.
Es liegen somit 400 ausgewählte Modelle, 3.600 Kandidatenbewertungen und
2.400 Testvorhersagen über die vier Methoden vor. Der gemeinsame Lauf
einschließlich Rekonstruktion und Plotexport dauerte **67,8 Minuten**.
Die drei Bonusmodelle sind in insgesamt 300 `*.pt`-Checkpoints gespeichert;
die 100 Baseline-Modelle stehen in `task4_cellcnn_filters_gated_alive_full.csv`.

Die vier Quellnotebooks behalten ihre bisherigen Standard-Splitlisten. Der
[Runner](run.py) überschreibt für diesen Lauf lediglich Splitliste und
Artefaktpfade. Modell- und Trainingscode-Hashes bleiben kompatibel mit den
wiederverwendeten Modellen. Der ursprüngliche HEAD und die beim Laufstart
noch uncommittierten Notebook-Änderungen sind in `run_scope.json` dokumentiert.

Mit lokalen Originaldaten und passender Projekt-/CUDA-Umgebung:

```bash
python results/tables/task6_comparison_100/run.py
```

Der Runner prüft und lädt passende Checkpoints, trainiert fehlende Splits und
schreibt Ausgaben erneut. Er enthält die kleinen Notebook-Smoke-Tests.
Abweichende Konfigurationen werden abgelehnt. Tabellen und Notebookausgaben
können ohne Originaldaten und GPU gelesen werden. CPU/GPU-Ergebnisse müssen
nicht bitgleich sein; die Checkpoints enthalten die verwendeten Paketversionen.

Die Plot- und Folienassets lassen sich ohne neue Modellläufe aktualisieren:

```bash
python -m src.task6_comparison
```

## Verifikation und Grenzen

Die kleinen Tests prüfen Split- statt Testauftritt-Pooling der ROC, umgekehrte
und konstante Frequenzen, fehlende Populationen, gleiche Spendergewichtung,
Zellzahlnenner sowie den Ausschluss innerer Validierungsspender aus der
Threshold-Referenz. Alle vier Notebooks wurden mit frischen Kerneln ausgeführt.
Anschließend wurden alle Checkpoints, Quellen-/Eingabehashes, Wiederverwendung,
Kandidatenauswahl, Vorhersagen, Alpha-/Radiusparameter, Frequency-Nenner,
Trainingsspender und aus den Tabellen nachberechneten Metriken geprüft.
Folien wurden kompiliert, gerendert und visuell auf Lesbarkeit/Überläufe geprüft.

Die 100 Splits verwenden dieselben 20 unabhängigen Spender. Die Varianten
wurden nach früheren Ergebnissen entwickelt. Der Vergleich bleibt deskriptiv;
mehr Splits erzeugen keine neue unabhängige biologische Stichprobe.
