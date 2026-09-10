# Quadratic CellCNN mit Learnable Pooling: 50 Splits

Dieser Ergebnisstand gehört zu **06e**, dem quadratischen Filter mit freien
linearen und quadratischen Gewichten und lernbarem Soft-Top-Alpha-Pooling.
Der Lauf vom 10.09.2026 umfasst Spender-Splits **0–49** auf `gated_alive`.
Zehn unverändert passende Modelle wurden wiederverwendet, 40 neu trainiert.
Die Erweiterung einschließlich Vorprüfungen und Auswertung dauerte
**13,9 Minuten** auf der NVIDIA GeForce RTX 5070 Laptop GPU.

## Ergebnisse ansehen

- [Ausgeführtes 50-Split-Notebook](task6_quadratic_learnable_50_executed.ipynb)
- [Vergleich je Split](task6_quadratic_learnable_paired_comparison.csv)
- [Baseline-vs.-Soft-Zusammenfassung](task6_quadratic_learnable_comparison_summary.csv)
- [Hard-vs.-Soft-Zusammenfassung der zehn gemeinsamen Splits](task6_quadratic_learnable_hard_comparison_summary.csv)
- [300 Testvorhersagen](task6_quadratic_learnable_predictions.csv)
- [450 Kandidatenbewertungen](task6_quadratic_learnable_selection.csv)
- [Donorfrequenzen und Trainings-Thresholds](task6_quadratic_learnable_frequencies.csv)
- [Alpha je Filter](task6_quadratic_learnable_alphas.csv)
- [Quellstand und Wiederverwendung](run_scope.json)

Tabellen und Notebookausgaben sind ohne Originaldaten oder GPU lesbar.

| Mittelwert | Lineare CellCNN-Baseline | Quadratic Soft-Top-Alpha |
|---|---:|---:|
| Network-ROC-AUC, 50 Splits | 0,8025 | 0,9200 |
| Frequency-ROC-AUC, 49 gemeinsam bestimmbare Splits | 0,8482 | 0,8520 |
| Frequency-Effect, dieselben 49 Splits | 0,006197 | 0,005357 |

Der Frequency-Effect ist die mittlere Donorfrequenz bei CMV+ minus CMV−,
als Anteil angegeben. Fehlende Phenotype-Werte werden nicht als null gewertet.
Die Anzahl gemeinsam bestimmbarer Frequency-Splits steht auch in der CSV.
Die Baseline hat auf Split 44 keinen positiven Filter; für das neue Modell
sind die Phenotype-Metriken auf allen 50 Splits bestimmbar.

Die mittlere gepaarte Network-AUC-Differenz zur linearen Baseline beträgt
**0,1175**: **28 Splits besser, 17 gleich, 5 schlechter**.
Die 50 Splits verwenden dieselben **20 unabhängigen Spender** und überlappen;
300 Testvorhersagen sind keine 300 unabhängigen Beobachtungen. Die Variante
wurde nach Sichtung früherer Ergebnisse entwickelt. Der Vergleich ist daher
deskriptiv und keine unabhängige Bestätigung.

**Die feste Quadratic-Hard-Top-1%-Referenz liegt nur für Splits 0–9 vor.**
Auf diesen gemeinsamen zehn Splits beträgt die mittlere Network-AUC
**0,9375 für Hard** und **0,9500 für Soft**. Diese Zahlen dürfen nicht mit
50-Split-Mittelwerten verglichen werden. In der vollständigen Vergleichstabelle
bleiben die Hard-Spalten für Splits 10–49 NaN. Es wurden keine zusätzlichen
Hard-Pooling-Modelle trainiert.

## Modell und Auswertung

Der [06e-Quellcode](../../../notebooks/06e_cellcnn_quadratic_learnable_pooling.ipynb)
verwendet für standardisierte `arcsinh(x/5)`-Markerwerte

$$h_k(x)=\operatorname{ReLU}\left(b_k+w_k^\top x+\sum_j q_{kj}x_j^2\right).$$

Die freien `quadratic_weights` starten bei null; Kreuzterme sind nicht enthalten.
Die Zellresponse entspricht bei der Initialisierung der linearen Baseline.
Die Antworten werden absteigend sortiert und über eine Sigmoid-Rangmaske
gepoolt, mit `alpha = sigmoid(raw_alpha)`, Start bei **1 %** und fester
Temperatur **0,002**. Danach folgen linearer Output und Softmax.
Alpha ist ein Ranganteil, kein absoluter Distanzthreshold.

Der Loss ist Cross Entropy plus
`1e-4 * (sum(W²) + sum(Q²) + sum(V²))`. Biases und Alpha werden nicht zusätzlich
regularisiert. Die Zellfilter dieses Modells haben keine Prototypzentren oder
Radiusparameter. Das separate
[06b-Prototypmodell mit Radius/ReLU](../task6_learnable_pooling_relu_50/README.md)
bleibt als eigene Variante erhalten.

Die Trainingskonfiguration wurde aus dem ursprünglichen 06e übernommen:
3.000 Zellen pro Trainingsbag, 200 Bags je Donor, Lernrate 0,01,
maximal 100 Epochen und Early-Stopping-Patience 5. Je Outer-Split werden drei
Inner-Folds und drei Filterzahlen (3, 4, 5), also neun Kandidaten, bewertet.
Das ausgewählte Inner-Fold-Modell wird direkt getestet, ohne Refit.
Scaler und Training verwenden ausschließlich die jeweiligen Inner-Train-Donoren.

Für die Population wird der Filter mit größtem positiven Output-Kontrast
gewählt. Sein Half-Max-Threshold ist `0.5 * max(h)` über alle Zellen der
tatsächlichen Inner-Train-Donoren dieses Modells. Die Frequency zählt
`h > threshold` auf einer festen Stichprobe von bis zu 20.000 Zellen pro
Testdonor. Die strikte Regel gilt auch bei Trainingsmaximum null; ohne positiven
Output-Kontrast bleiben die Phenotype-Metriken fehlend. Eine AUC unter 0,5 wird
nicht umgedreht. Donorlabels sind keine Zelltyp-Ground-Truth.

## Laden und reproduzieren

Die 50 Dateien `task6_quadratic_learnable_split_*.pt` enthalten den vollständigen
Modellzustand einschließlich `quadratic_weights` und `raw_alpha`, den Scaler,
Konfiguration, Inner-Fold, Vorhersagen und Kandidatenauswahl. Beispiel vom
Repository-Stamm:

```python
import torch

checkpoint = torch.load(
    "results/tables/task6_quadratic_learnable_50/task6_quadratic_learnable_split_0.pt",
    map_location="cpu",
    weights_only=True,
)
```

Neue Inferenz benötigt zusätzlich die `QuadraticCellCNN`-Klasse aus 06e,
gespeicherte Skalierung und Originaldaten mit korrekter Markerreihenfolge.
Unter `hard_reference/` liegen die unveränderten zehn Hard-Checkpoints und
ihre Tabellen. Das [06d-Notebook](../../../notebooks/06d_cellcnn_quadratic.ipynb)
wurde zur Prüfung dieser Referenz ebenfalls unverändert übernommen.

Der mitgelieferte [Runner](run.py) verwendet das Quellnotebook und überschreibt
nur die Splitliste und Ergebnis-Pfade. In der passenden Projekt-/CUDA-Umgebung
mit dem registrierten Kernel `ssbi-group-project` und lokalen Originaldaten:

```bash
python results/tables/task6_quadratic_learnable_50/run.py
```

Der Runner führt die kleinen Prüfungen einschließlich Smoke-Training aus,
rekonstruiert die Baseline, lädt passende Checkpoints und trainiert fehlende
Splits. Die Auswertung wird erneut geschrieben. Abweichende Konfigurationen
werden abgelehnt. Verwendet wurden Python 3.12.14 und PyTorch 2.14.0+cu130;
CPU und GPU müssen nicht bitgleiche Werte liefern. Der normale 06e-Quellcode
behält seinen Standardumfang von zehn Splits.

## Herkunft und Verifikation

Der ursprüngliche 06e-Code stammt aus dem lokalen `GrHa`-Worktree. Auf
`GrHa-learnable-pooling` wurde nur die Vergleichsausgabe für unterschiedliche
Splitumfänge angepasst; Modell- und Trainingszellen blieben identisch.
Der Übernahme-Commit ist **93c38e2efcde4481e4d325163efcf7cda65f20c6**. `run_scope.json` dokumentiert
den früheren HEAD bei Vorbereitung, beide Notebook-Hashes und die zehn
wiederverwendeten Splits. Absolute Pfade darin gehören zum ursprünglichen
Laufrechner; der Runner verwendet relative Projektpfade.

Bestanden sind die Modell-, Initialisierungs-, Pooling-, Gradienten- und
Regularisierungschecks, der kleine echte Smoke-Split und der gesamte
Notebooklauf. Anschließend wurden alle 50 Checkpoints, Konfigurationen,
Code-Hashes, 450 Kandidatenbewertungen, 300 Testvorhersagen, Alpha-Werte,
Threshold-Trainingsspender und nachberechneten AUCs geprüft. Die zehn
wiederverwendeten Checkpoints sind bytegleich mit den Originalen. Für den
CPU/GPU-Vergleich der Alpha-Werte wurde Float32-Toleranz verwendet.

SHA-256 der gepaarten Vergleichstabelle:

```text
a6c8938b9f6254a11d965d76dddfcce3f3554980f7a35a0521a7969733b20738
```
