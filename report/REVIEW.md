# Unabhängiges Review der neuen Gesamtfassung

Abgeschlossen am 14. September 2026 durch den separaten Review-Agenten
`independent_review`. Das Review erfolgte lesend, ohne neue Benchmarks.

## Ergebnis

Keine offenen kritischen Findings. Alle sechs Aufgaben werden im Hauptbericht
beantwortet. Die beiden konkreten Befunde wurden korrigiert und die PDFs danach
erneut gebaut und geprüft:

1. Die Geometriestrafe des Mahalanobis/ReLU-Modells ist eindeutig als
   `10^{-3} mean[(a - 1)^2]` formuliert. Die frühere Schreibweise konnte als
   Quadrat des Mittelwerts gelesen werden und entsprach dann nicht dem Code.
2. Die im CellCNN-Paper ebenfalls verglichenen Single-Marker-Gates wurden in
   der Introduction und der Literaturübersicht des Supplements ergänzt.

## Prüfumfang

- Aufgabenblatt und CellCNN-Originalpaper; vollständige Aufgabenabdeckung.
- Aktuelle Notebookformeln, Ergebnisexports und Quellenprüfsummen.
- Alle fünf Hauptseiten visuell; Abbildungen, Lesbarkeit und Bonusumfang.
- Die drei Procrustes-Vergleiche erneut nachgerechnet; Clusterstabilitäten,
  Bonuskennzahlen und Formeln mit ihren Quellen verglichen.
- Analysegrundlagen getrennt: 40.000 NK-Zellen, 20.000 Live-PBMCs,
  30 gemeinsame Klassifikationssplits, 100 Bonus-Netzwerksplits und
  98 gemeinsam auswertbare Bonus-Frequenzsplits.
- Konsistente Mittelwerte/Mediane und IQR-/z-Skalen; unveränderte Übernahme der
  bereits geprüften Task-4/5-Abbildungen.

Die abschließende Dokumentprüfung bestätigt fünf Hauptseiten, 26 Supplementseiten,
vier Hauptabbildungen und einen Bonusblock mit 32,3 % der nutzbaren Seitenhöhe.
Es bestehen keine überlaufenden Elemente oder unaufgelösten Zitate/Querverweise.
Der endgültige Prüfstand mit PDF-Prüfsummen steht in
[data/validation.json](data/validation.json).

Die kleine, teilweise verwandte Kohorte, überlappende Spendersplits und fehlende
unabhängige biologische Validierung bleiben Grenzen der Analysen; der Bericht
benennt sie ausdrücklich.
