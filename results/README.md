# Geteilte Ergebnisse

Stand der Veröffentlichung: 9. September 2026. Die Ergebnisdateien sind zum
Lesen und Weiterverarbeiten versioniert; dafür ist kein erneutes Training nötig.
Die Original-FCS-Dateien und die bereitgestellten Paper gehören weiterhin zur
lokalen Datengrundlage jedes Mitarbeitenden.

## Einstieg

| Fragestellung | Dateien auf `GrHa` |
|---|---|
| Projektionen und Nachbarschaftsvergleich | `tables/task2_*`, `figures/task2_*` |
| Clustering und Markerprofile | `tables/task3_*`, `figures/task3_*` |
| Klassifikation: gemeinsamer Dreiervergleich | `tables/task4_metric_summary.csv`, `tables/task4_split_metrics.csv`; CellCNN/SVM/Citrus auf Splits 0–29 |
| CellCNN und SVM über 100 Splits | `tables/task4_cellcnn_svm_100_*`; zugehörige Predictions, Selection und Modellparameter unter `tables/task4_*` |
| Aktuelle Interpretation über 30 Splits | `tables/task5_paper_*` |
| Radius/Mean-Bonus über 100 Splits | [bonus_100](tables/bonus_100/); vollständige Checkpoints `tables/task6_modified_split_*.pt` |
| Radius/Top1 über drei Splits | `tables/task6_mahalanobis_top1_*` |
| Linear-quadratischer Bonus über zehn Splits | `tables/task6_quadratic_*` |
| Technische und mathematische Architektur-Analyse | [PDF](../report/SSBI_CellCNN_Architecture_Analysis.pdf), [Markdown](../report/SSBI_CellCNN_Architecture_Analysis.md) |
| Präsentation zu Aufgaben 4 und 5 | [Folien, Quellen und Sprechernotizen](../praesentation/aufgaben_04_05/README.md) |

Die Analyse dokumentiert den damaligen lokalen/GitHub-Stand vor dieser
Veröffentlichung. Zuvor als lokal oder noch nicht versioniert bezeichnete
Ergebnisdateien werden mit dieser Veröffentlichung geteilt. Absolute Pfade in
historischen Provenienzen und in der Analyse beschreiben den ursprünglichen
Laufrechner; die obigen relativen Pfade sind der Einstieg im eigenen Checkout.

## Learnable Pooling liegt auf dem eigenen Branch

Auf [GrHa-learnable-pooling](https://github.com/habicht12/SSBI-Project/tree/GrHa-learnable-pooling/results):

- `tables/task6_learnable_pooling_100/`: historischer Lauf ohne Geometriestrafe,
  100 Splits, 900 Kandidaten, mittlere Network-AUC 0,71875.
- `tables/task6_learnable_pooling_geometry_50/`: Lauf mit Geometriestrafe 0,001,
  50 Splits, 450 Kandidaten, mittlere Network-AUC 0,72250; der Vergleich zur
  bisherigen Variante und Baseline verwendet dieselben Splits 0–49.

Beide Pakete enthalten ausgeführte Notebooks, Vorhersagen, Kandidatenauswahl,
Alpha-Werte, Konfigurationen und ausgewählte Modelle. Der Quellstand steht in
der jeweiligen `run_scope.json` und im Trainingscode-Hash des Checkpoints.

## Historische Ergebnisse nicht mit aktuellen Läufen mischen

- `task4_*gated_NK_smoke*` sind technische Smoke-Läufe.
- `task5_*` ohne `paper_` dokumentieren die frühere Interpretation; die aktuelle
  Zentroidanalyse verwendet `task5_paper_*`.
- `tables/citrus_10_split_backup/` enthält den früheren Zehn-Split-Stand.
- `tables/archive_citrus_1000cells_5percent_20260908/` verwendet die frühere
  Citrus-Konfiguration und gehört nicht zum aktuellen 30-Split-Dreiervergleich.
- `tables/task6_full_100/` enthält Notebook-Snapshots des Radius/Mean-Laufs;
  `previous/` dokumentiert dessen vorherigen kleinen Ergebnisstand.

Die Auswertungseinheit bleibt der Donor. Wiederholte Splits sind keine neuen
unabhängigen Datensätze. AUC-Vergleiche nur auf identischen Split-IDs durchführen.

## Modelle und Reproduzierbarkeit

PyTorch-Checkpoints können ohne GPU gelesen werden:

```python
import torch

checkpoint = torch.load(
    "results/tables/task6_modified_split_0.pt",
    map_location="cpu",
    weights_only=True,
)
```

Für neue Zellantworten sind zusätzlich die Originaldaten und die zum
Checkpoint passende Architektur nötig. Der aktuelle lokale `GrHa`-Stand hat
eine bekannte Importinkonsistenz: Die Bonusnotebooks importieren noch
`SavedCellCNN`, das in der neueren Interpretation entfernt wurde. Vor einem
erneuten Bonuslauf den passenden historischen Quellstand verwenden oder den
Import gezielt korrigieren. Die gespeicherten Ergebnisse können unabhängig
davon gelesen werden; für diese Veröffentlichung wurden keine Modelle neu trainiert.
