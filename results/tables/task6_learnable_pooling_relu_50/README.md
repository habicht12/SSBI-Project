# Radius/ReLU und Learnable Pooling: 50 Splits

Abgeschlossener GPU-Lauf vom **10.09.2026** auf `gated_alive`, Spender-Splits
**0–49**. Alle 50 Modelle wurden neu trainiert; die vorhandene CellCNN-Baseline
wurde wiederverwendet. Der Lauf dauerte einschließlich Vorprüfungen und
Auswertung **20,7 Minuten** auf einer NVIDIA GeForce RTX 5070 Laptop GPU.

## Ergebnisse ansehen

Für Tabellen und gespeicherte Notebook-Ausgaben werden weder die Originaldaten
noch eine GPU benötigt. Der Einstieg ist das
[ausgeführte Notebook](task6_learnable_pooling_relu_50_executed.ipynb).

- [Gepaarter Vergleich je Split](task6_learnable_pooling_relu_paired_comparison.csv)
- [Metriken der neuen Variante](task6_learnable_pooling_relu_metrics.csv)
- [300 Testvorhersagen](task6_learnable_pooling_relu_predictions.csv)
- [450 Kandidatenbewertungen](task6_learnable_pooling_relu_selection.csv)
- [Populationen: Frequenzen, Zellzahlen und Thresholds](task6_learnable_pooling_relu_frequencies.csv)
- [Alpha und Radius je Filter](task6_learnable_pooling_relu_alphas.csv)
- [Quellstand und Laufüberschreibungen](run_scope.json)

| Mittelwert | CellCNN-Baseline | Radius/ReLU + Learnable Pooling |
|---|---:|---:|
| Klassifikations-ROC-AUC, 50 Splits | 0,8025 | 0,7125 |
| Frequency-ROC-AUC, 49 gemeinsame Splits | 0,8482 | 0,7028 |
| Frequency-Effect, 49 gemeinsame Splits | 0,006197 | 0,003297 |

Der Frequency-Effect ist die mittlere Donorfrequenz bei CMV+ minus CMV−,
angegeben als Anteil; beispielsweise entsprechen 0,003297 etwa 0,330
Prozentpunkten. Die Baseline besitzt auf Split 44 keinen positiven Filter;
deshalb werden für die beiden Frequency-Zeilen dieselben 49 bestimmbaren Splits
verglichen. Die neue Variante liefert auf allen 50 Splits Frequency-Metriken;
ihre mittlere Frequency-AUC über alle 50 beträgt 0,70875.

Die mittlere gepaarte Klassifikations-AUC-Differenz zur Baseline beträgt
**−0,0900**, der Median **−0,1250**: 13 Splits besser, 8 gleich, 29 schlechter.
Der Median der Klassifikations-AUC beträgt 0,8750 für die Baseline und 0,7500
für die neue Variante. Unter diesen Einstellungen verbessert die Änderung den
Vergleich mit der Baseline nicht. Die 50 Splits überlappen und verwenden
dieselben **20 unabhängigen Spender**; 300 Testvorhersagen sind keine 300
unabhängigen Beobachtungen. Der Vergleich ist deskriptiv.

## Modell und Populationsdefinition

Die Implementierung steht im bestehenden
[06b-Quellnotebook](../../../notebooks/06b_cellcnn_mahalanobis_learnable_pooling.ipynb).
Die bisherige Zellantwort `−d²` wurde durch

$$d_k^2(x)=\frac1d\sum_j a_{kj}(x_j-c_{kj})^2,\qquad
h_k(x)=\operatorname{ReLU}(\rho_k-d_k^2(x)),\qquad
\rho_k=\operatorname{softplus}(\mathrm{raw\_rho}_k)+10^{-6}$$

ersetzt. Die trainierbaren Prototypzentren und positiven Markergewichte mit
Mittelwert eins je Filter bleiben erhalten. Die Radiusinitialisierung verwendet
wie `GrHa/06c` das 1%-Quantil der Distanzen auf der spenderbalancierten
Inner-Train-Referenz (maximal 2.000 Zellen je Spender), mindestens `2 * EPS`,
mit der dortigen stabilen inversen Softplus.

Danach folgt weiterhin das weiche Top-Alpha-Pooling: absteigend sortierte
Responses, Sigmoid-Rangmaske, `alpha = sigmoid(raw_alpha)`, Start bei 1 % und
feste Temperatur `tau = 0.002`, anschließend der lineare Output.
Sampling, Scaler, Bags, Splits, Filterzahlen, Lernrate und Early Stopping wurden
beibehalten. Die Geometriestrafe bleibt
`0.001 * mean((normalized_marker_weights - 1) ** 2)` ausschließlich im
Trainingsloss; L2 betrifft weiterhin nur die Output-Gewichte. Es gibt keine
zusätzliche Radius-Regularisierung. Gegenüber der linearen CellCNN-Baseline
unterscheiden sich mehrere Modellbestandteile; der Vergleich isoliert daher
nicht allein den Effekt des lernbaren Poolings.

Je ausgewähltem Modell bestimmt der größte positive Output-Kontrast
`V[1, k] - V[0, k]` den Phenotype-Filter. Sein Half-Max-Threshold ist
`0.5 * max(h)` über **sämtliche Zellen der tatsächlichen Inner-Train-Spender**
dieses Modells. Inner-Validation und Outer-Test gehen nicht in das Maximum ein.
Die ausgewählten inneren Modelle werden ohne Refit auf alle Outer-Train-Spender
getestet.

Auf derselben festen Stichprobe von bis zu 20.000 Zellen je Testspender wie in
der Baseline wird die Maske strikt als `h > halfmax_threshold` berechnet.
`frequency = n_selected_cells / n_cells`; daraus folgen die unverändert
orientierte ROC-AUC und die Differenz der CMV-Gruppenmittel. Ohne positiven
Output-Kontrast bleiben die Phenotype-Metriken fehlend. Auch bei Trainingsmaximum
null gilt die strikte Formel; in diesem Lauf trat dieser Fall bei keinem der
ausgewählten Phenotype-Filter auf. Radius, Alpha und Half-Max-Threshold haben
unterschiedliche Rollen. Pooled-Response-Metriken bleiben zusätzliche Diagnostik.

Die Frequenzdatei enthält Filter-ID, Alpha, Radius, Half-Max-Threshold,
Trainingsspender und Zellzahlen. Die booleschen `population_masks` sind während
der Notebook-Auswertung verfügbar und beziehen sich auf `evaluation_indices`;
sie werden nicht als eigene Datei gespeichert. Ihre erneute Berechnung benötigt
die Originalzellen und den jeweiligen Checkpoint. Es wurde kein neuer
Clustering-Schritt ergänzt.

## Modelle laden und den Lauf nachvollziehen

Die 50 Dateien `task6_learnable_pooling_relu_split_*.pt` enthalten jeweils
`state_dict` einschließlich `raw_rho` und `raw_alpha`, Trainings-Scaler,
Konfiguration, ausgewählten Inner-Fold, Vorhersagen und Kandidatenauswahl.
Vom Repository-Stamm aus lässt sich ein Checkpoint ohne GPU einlesen:

```python
import torch

checkpoint = torch.load(
    "results/tables/task6_learnable_pooling_relu_50/task6_learnable_pooling_relu_split_0.pt",
    map_location="cpu",
    weights_only=True,
)
```

Für neue Inferenz ist zusätzlich die `PrototypeCellCNN`-Klasse aus Zelle 5 des
06b-Notebooks (Zählung ab 0), der gespeicherte Scaler und die ursprüngliche
Markerreihenfolge nötig. Modell und Datenvorverarbeitung werden im Notebook
gemeinsam rekonstruiert.

Der mitgelieferte [Runner](run.py) verwendet das unveränderte Quellnotebook und
überschreibt nur Splitliste und lokale Artefaktpfade. Mit lokalen Originaldaten,
der [Projektumgebung](../../../environment.yml), dem registrierten Kernel
`ssbi-group-project` und einer passenden CUDA-Umgebung lautet der Aufruf vom
Repository-Stamm:

```bash
python results/tables/task6_learnable_pooling_relu_50/run.py
```

Der Runner führt die kleinen Notebook-Prüfungen einschließlich Smoke-Training
aus, lädt passende Checkpoints und trainiert fehlende Splits. Er schreibt die
Auswertung erneut. Abweichende Konfigurationen werden abgelehnt; CPU und GPU
müssen nicht bitgleiche Ergebnisse liefern. Der tatsächlich verwendete Stand
war Python 3.12.14 und PyTorch 2.14.0+cu130. Das normale 06b-Notebook behält
seinen Standardumfang von drei Splits.

`task4_*` und `task6_baseline_*` sind die verwendeten Split- und Baseline-Dateien.
Original-FCS-Dateien und ZIPs sind nicht enthalten. Absolute Pfade in
`run_scope.json` dokumentieren den ursprünglichen Laufrechner; der Runner
ermittelt den Repository-Stamm relativ zu seinem eigenen Speicherort.

## Quellstand und geprüfte Konsistenz

Der Modellcode wurde nach dem Lauf als Commit **c0597750bb5bdbfb927e0f7479f48aa35e983836** gesichert.
`run_scope.json` nennt noch den damaligen HEAD `1941243`, weil die Notebook-
Änderung beim Start uncommittiert war. Der dort gespeicherte Notebook-SHA-256
identifiziert exakt die anschließend committete Datei. Das ausgeführte Notebook
enthält zusätzlich die dokumentierten Überschreibungen für 50 Splits.

Bestanden sind die synthetischen Modell-/Gradienten-/Half-Max-Prüfungen, der
kleine echte Smoke-Split und der vollständige Notebooklauf. Anschließend wurden
alle 50 Checkpoints, Konfigurationen und Trainingscode-Hashes, die
Kandidatenauswahl, 300 Testspender-Zuordnungen, gespeicherte Filterparameter,
Threshold-Trainingsspender sowie die Metriken aus den exportierten Vorhersagen
und Frequenzen geprüft. CPU-Rekonstruktion und GPU-Parameterexport wurden mit
Float32-Toleranz verglichen. Eine erneute vollständige Zellinferenz war nicht
Teil dieser zusätzlichen Artefaktprüfung.

SHA-256 der gepaarten Vergleichstabelle:

```text
1938eb5a4be29a2215613350135e11da62d92f79b6b359c4990b3e3aa144b990
```

Die früheren Läufe [ohne Radius/ReLU mit Geometriestrafe](../task6_learnable_pooling_geometry_50/README.md)
und [ohne Radius/ReLU und ohne Geometriestrafe](../task6_learnable_pooling_100/README.md)
bleiben getrennte historische Ergebnisstände.
