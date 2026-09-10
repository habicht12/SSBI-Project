# Aufgabe 6: vier CellCNN-Varianten vergleichen

[slides.pdf](slides.pdf) enthält elf deutsche Folien im vorhandenen
16:9-Beamer-Theme [theme.tex](../theme.tex). Die Quelle ist [slides.tex](slides.tex).
Die Folien vergleichen alle vier Modelle auf denselben 100 Spendersplits.
Half-Max ist eine Zellselektion; es wird kein Clustering ausgeführt.

Ergebnisdaten, Wiederverwendung und Metrikdefinitionen stehen in der
[Benchmark-README](../../results/tables/task6_comparison_100/README.md).
CSV-Tabellen und `data/provenance.json` belegen die Folienwerte. Die Zahlen
und Takeaway-Texte werden durch `src.task6_comparison.export_slide_assets`
direkt aus den Vergleichstabellen erzeugt.

## Aktualisieren und bauen

Vom Repository-Stamm in der vorhandenen Projektumgebung:

```bash
python -m src.task6_comparison
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -outdir=build praesentation/aufgabe_06_vergleich/slides.tex
cp praesentation/aufgabe_06_vergleich/build/slides.pdf praesentation/aufgabe_06_vergleich/slides.pdf
```

Der Plotexport liest gespeicherte Ergebnistabellen; er trainiert keine Modelle.
Der LaTeX-Build benötigt nur Theme, Quelle, `figures/` und `data/*.tex`.
Das Theme stammt unverändert von `GrHa`; Sprache und Fußzeile werden in der
Folienquelle angepasst. Originaldaten und GPU sind zum Lesen und zum
LaTeX-Build nicht nötig.

## Darstellungen

- ROC-Kurven: Mittel interpolierter Split-ROCs; die AUC-Legenden mitteln die
  ursprünglichen Split-AUCs. Kein Pooling von 600 Testauftritten.
- Boxplots: Q1–Q3, Median und Splitpunkte; keine Konfidenzintervalle.
- Frequency-Vergleiche: dieselben bei allen vier Modellen bestimmbaren Splits.
- Donorfrequenzen: ein Punkt je Spender, Mittel über seine Testauftritte;
  gemeinsame Y-Skala. Populationen können zwischen Splits variieren.
- Alpha-Verteilung: Parameter aller ausgewählten Modellfilter; keine
  gemessene Populationshäufigkeit.

Die PDF wurde nach dem Build vollständig gerendert und visuell geprüft.
Die Folien bestätigen keine Zelltypen und keinen unabhängigen
Generalisierungsvorteil; alle Auswertungen verwenden dieselben 20 Spender.
