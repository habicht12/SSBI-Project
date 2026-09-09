# CellCNN: geprüfter Ergebnisstand über 100 Splits

Dieser Ordner enthält den am 09.09.2026 abgeschlossenen GPU-Benchmark der Variante mit lernbarem Pooling. Die historische Implementierung ohne Geometriestrafe steht in [06b bei Commit 544192d](https://github.com/habicht12/SSBI-Project/blob/544192d0a4a8bd010c67043ce2d2e3ccf2c65b26/notebooks/06b_cellcnn_mahalanobis_learnable_pooling.ipynb); Quellstand und Laufüberschreibungen sind in [run_scope.json](run_scope.json) dokumentiert. Splits 0–2 verwenden die bereits vorhandenen Checkpoints, Splits 3–99 wurden neu trainiert.

Zum Ansehen der Ergebnisse sind weder Originaldaten noch eine GPU nötig:

- Das [ausgeführte Notebook](task6_learnable_pooling_100_executed.ipynb) zeigt die gespeicherten Ausgaben des tatsächlichen Laufs. Es ist ein unveränderter Ausführungsnachweis; Hinweise darin auf `run.py` beziehen sich auf das lokale Startskript.
- Die [Vergleichstabelle](task6_learnable_pooling_paired_comparison.csv) enthält je Split die AUCs, gepaarten Differenzen, gewählte Filterzahl, Alpha-Werte und Laufzeit. Mittelwert und Median der AUCs stehen in der [Zusammenfassung](task6_learnable_pooling_comparison_summary.csv).
- `task6_learnable_pooling_predictions.csv` enthält 600 Testvorhersagen; `task6_learnable_pooling_selection.csv` dokumentiert alle 900 Kandidatenbewertungen. Alpha-Werte, Poolingantworten und Laufzeiten liegen zusätzlich in den entsprechend benannten CSV-Dateien.
- `task4_*` und `task6_baseline_*` liefern die verwendeten Spender-Splits, Baseline-Vorhersagen, Filterparameter, Auswahl- und Konfigurationsnachweise. `run_scope.json` enthält zur Provenienz auch die ursprünglichen lokalen Pfade; die mitgelieferten Dateien liegen gemeinsam in diesem Ordner.

Die 100 Dateien `task6_learnable_pooling_split_*.pt` enthalten jeweils den ausgewählten Modellzustand einschließlich `raw_alpha`, den Trainings-Scaler, die Konfiguration, Vorhersagen und Kandidatenauswahl. Mit installiertem PyTorch lassen sie sich ohne GPU laden, beispielsweise vom Repository-Stamm aus:

```python
import torch

checkpoint = torch.load(
    "results/tables/task6_learnable_pooling_100/task6_learnable_pooling_split_0.pt",
    map_location="cpu",
    weights_only=True,
)
```

Die mittlere ROC-AUC beträgt 0,80750 für die Baseline und 0,71875 für die neue Variante. Die mittlere gepaarte Differenz ist −0,08875. Die 100 Splits überlappen und verwenden dieselben 20 unabhängigen Spender; 600 Testvorhersagen sind daher keine 600 unabhängigen Beobachtungen.

Die Rohdaten sind nicht enthalten. Eine erneute Berechnung von Zellantworten oder Vorhersagen benötigt die Original-FCS-Dateien, Marker-/Labeldateien und die passende Umgebung. Der reguläre Code in 06b startet weiterhin mit drei Splits; die gespeicherten Notebook-Ausgaben und `run_scope.json` dokumentieren den zusätzlich angeforderten Lauf über 100 Splits. Logs, Prozessdateien, Startskript und temporäre Vorprüfungen bleiben lokal.

Der spätere Lauf mit Geometriestrafe ist separat unter [task6_learnable_pooling_geometry_50](../task6_learnable_pooling_geometry_50/README.md) dokumentiert.
