# Geteilte Learnable-Pooling-Ergebnisse

- [100 Splits ohne Geometriestrafe](tables/task6_learnable_pooling_100/README.md):
  historischer Modellstand `544192d`, Network-AUC 0,71875.
- [50 Splits mit Geometriestrafe](tables/task6_learnable_pooling_geometry_50/README.md):
  aktueller Trainingscode `1d696d6`, Network-AUC 0,72250. Der direkte Vergleich
  verwendet die identischen ersten 50 Splits, nicht den gesamten 100-Split-Mittelwert.

Die Dateien direkt unter `tables/` enthalten zusätzlich den ursprünglichen
Drei-Split-Bonuslauf und seine Baseline-Referenzen. Sie sind historisch und
kein Lauf der aktuellen Geometrie-Regularisierung. Die jeweiligen Ordner mit
100 beziehungsweise 50 Splits sind die maßgeblichen vollständigen Laufpakete.

Weitere Projektresultate, die Architektur-Analyse und Präsentationsmaterialien
liegen auf [GrHa](https://github.com/habicht12/SSBI-Project/tree/GrHa/results).
Für Tabellen und gespeicherte Notebookausgaben ist kein neues Training nötig.
Originaldaten, Logs und Prozessdateien werden nicht mitgeteilt. Absolute lokale
Pfade in Provenienzen dokumentieren den ursprünglichen Rechner.
