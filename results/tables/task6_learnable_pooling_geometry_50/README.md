# Learnable Pooling mit Geometrie-Regularisierung: 50 Splits

Abgeschlossener Lauf vom 09.09.2026 auf `gated_alive`, Splits 0–49.
Das Modell verwendet negative diagonale Prototypdistanzen, weiches Rangpooling
mit lernbarem Alpha und Temperatur 0,002 sowie die zusätzliche Trainingsstrafe
`0.001 * mean((normalized_marker_weights - 1) ** 2)`.
Early Stopping verwendet weiterhin Cross Entropy plus Output-L2.

## Ergebnisse lesen

- [Ausgeführtes Notebook](task6_learnable_pooling_geometry_50_executed.ipynb)
- [Vergleich auf denselben 50 Splits](task6_geometry_effect_comparison.csv)
- [Metriken](task6_learnable_pooling_metrics.csv)
- [300 Donor-Testvorhersagen](task6_learnable_pooling_predictions.csv)
- [450 Kandidatenbewertungen](task6_learnable_pooling_selection.csv)
- [Gelernte Alpha-Werte](task6_learnable_pooling_alphas.csv)
- [Quellstand und Laufüberschreibungen](run_scope.json)

| Variante | Mittlere Network-AUC | Median |
|---|---:|---:|
| Lineare Baseline | 0,8025 | 0,8750 |
| Learnable Pooling ohne Geometriestrafe | 0,7200 | 0,7500 |
| Learnable Pooling mit Geometriestrafe | 0,7225 | 0,7500 |

Gegenüber der Variante ohne Geometriestrafe: drei Splits besser, 45 gleich,
zwei schlechter; mittlere gepaarte Differenz +0,0025. Die 50 Splits verwenden
dieselben 20 Donoren und sind keine 50 unabhängigen Datensätze.

## Modelle und Baseline

Die 50 Dateien `task6_learnable_pooling_split_*.pt` enthalten jeweils Modell,
Scaler, Konfiguration, Vorhersagen und Kandidatenauswahl. Laden ohne GPU:

```python
import torch

checkpoint = torch.load(
    "results/tables/task6_learnable_pooling_geometry_50/task6_learnable_pooling_split_0.pt",
    map_location="cpu",
    weights_only=True,
)
```

Die mitgelieferten `task4_*`- und `task6_baseline_*`-Dateien sind die verwendete
Baseline und Splitreferenz. Der Lauf wurde mit damals noch uncommittierten
Änderungen vorbereitet; deshalb nennt `run_scope.json` zusätzlich den früheren
HEAD. Der Trainingscode-Hash aller 50 Checkpoints stimmt mit den relevanten
Notebookzellen des Commits `1d696d6d28561ccda5e080d8b9cfc3d40122b250` überein:

```text
15ea74ebad557a1a3d34efd959643f8199913e002fc28994603eec068f362039
```

Originaldaten und GPU sind zum Lesen der Tabellen und Notebookausgaben nicht
nötig. Neue Inferenz benötigt die lokalen FCS-Dateien und die passende
Architektur. Absolute Pfade in `run_scope.json` gehören zum ursprünglichen
Laufrechner; die Artefakte liegen hier gemeinsam im Ergebnisordner.
