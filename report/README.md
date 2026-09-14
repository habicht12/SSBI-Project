# Neuer Gesamtbericht

## Version 2

- **[Hauptbericht main02: fünf Seiten](build/main02.pdf)**
- **[Supplement supplement02: 26 Seiten](build/supplement02.pdf)**
- Quellen: [main02.tex](main02.tex), [supplement02.tex](supplement02.tex).

Die drei erklärenden Fußzeilen unter den Hauptabbildungen 1, 3 und 4 entfallen.
Ihre Informationen stehen vollständig in den regulären Bildunterschriften,
einschließlich aller drei Procrustes-Werte. Achsen, Legenden, Panelangaben,
Farbskalen und zugrunde liegende Daten sind erhalten. Abbildung 2 ist unverändert.
Die neuen Exporte liegen getrennt unter `figures/v02`.

Beide Texte sind sprachlich überarbeitet. Die Semikolons im Hauptquelltext
sind von 21 auf 0, im Supplement von 76 auf 29 reduziert; dort bleiben sie in
kompakten Tabellen stehen. Der Hauptbericht ergänzt die Literaturprinzipien,
Clustering-Methoden und Markerinterpretation, konkrete Parameterwirkungen,
Auswahlstabilität sowie Grenzen des Klassifikationsvergleichs. Er nutzt weiterhin
A4 und 11-Punkt-Haupttext. Der Bonus steht geschlossen auf Seite 5 und umfasst
einschließlich Tabelle 32,3 % der nutzbaren Seitenhöhe.

Version 2 kompilieren:

```bash
bash report/build.sh main02 supplement02
```

Die drei V2-Abbildungen erneut aus den geprüften Ergebnisexports zeichnen und
die Dokumente prüfen:

```bash
MPLCONFIGDIR=/tmp/ssbi-report02-mpl .venv/bin/python report/export_assets02.py
bash report/build.sh main02 supplement02
PYTHONPATH=/tmp/report_korrigiert_pdf_tools .venv/bin/python report/check_report02.py
```

Dabei werden keine Modelle, Zellselektionen oder Parametersuchen neu berechnet.
[V2-Provenienz](data/v02/figure_provenance.json),
[V2-Prüfergebnis](data/v02/validation.json) und
[unabhängiges V2-Review](REVIEW02.md) dokumentieren den neuen Stand.
61 Dateien der ersten Fassung sowie die sechs geschützten Originaldateien sind
unverändert. Auch sämtliche mathematischen Ausdrücke und Zahlentokens des
Supplements stimmen zwischen den Fassungen überein.

## Version 1

- **[Hauptbericht: fünf Seiten](build/main.pdf)**
- **[Supplement: 26 Seiten](build/supplement.pdf)**

Englische Neufassung von *Benchmarking Comparative Single-Cell Analysis of CMV
Status*, von Sebastian Fay, Marie Schygulla und Gregor Habitzreither.
Die Hauptfassung behandelt alle sechs Aufgaben mit vier Ergebnisabbildungen.
Die Bonusaufgabe steht geschlossen auf Seite 5 und belegt einschließlich
Überschrift und Tabelle rund 32 % der nutzbaren Seitenhöhe.

Die alten generierten Voll- und Detailberichte wurden entfernt. Der vorhandene
Einleitungs-/Aufgabe-2-Bericht der Kollegin, seine Korrekturfassung und die
gehaltene Präsentation sind unverändert; ihre Prüfsummen sind in
[protected_files.json](data/protected_files.json) festgehalten.

## Inhalt und Aufgabenabdeckung

| Aufgabe | Hauptbericht | Vertiefung |
| --- | --- | --- |
| 1: Literatur | Introduction: CellCNN, Citrus, SVM und weitere Paper-Baselines | S1.1: Prinzipien und Projektstatus |
| 2: Dimensionsreduktion | Methods 2.1–2.2, Results 3.1, Figure 1 | S1/S2: PCA-Spektrum, sechs t-SNE- und 64 UMAP-Einstellungen, Qualitätsmaße und alle Kartenpaare |
| 3: Clustering | Methods 2.3, Results 3.2, Figure 2 | S2.2: alle 44 Konfigurationen, Linkage-Varianten, Stabilität und alle 37 Marker |
| 4: Klassifikation | Methods 2.4, Results 3.3, Figure 3 | S3/S5: Implementierungen, 30 gemeinsame Splits, zusätzlicher 100-Split-SVM/CellCNN-Vergleich |
| 5: Zellsubsets | Methods 2.5, Results 3.4, Figure 4 | S3/S5: quantitative Scores, OOF-Nenner, Citrus-Zuordnung und ergänzende Profile |
| 6: Bonus | Results 3.5, maximal eine halbe Seite | S4/S5: exakte Modelle, Pooling und 100-/98-Split-Vergleiche |

## Kompilieren

Aus dem Projektstamm:

```bash
bash report/build.sh
```

Benötigt werden `latexmk` und eine TeX-Live-kompatible Installation mit den
Standardpaketen in [style.tex](style.tex). Abbildungen und Tabellen sind enthalten;
zum Kompilieren werden weder die FCS-Dateien noch ein Modelltraining benötigt.
Die gemeinsame Literaturdatei [references.tex](references.tex) enthält kompakte
numerische Einträge und Links zu den Primärquellen. BibTeX/Biber ist nicht nötig.

Quelltexte: [main.tex](main.tex), [supplement.tex](supplement.tex).

## Abbildungen und Datenherkunft

- Figure 1 verwendet die aktuellen gespeicherten 40.000-Zell-Koordinaten der
  Aufgabe-2-Neuberechnung: Perplexität 70, UMAP mit 5 Nachbarn, Mindestabstand 0
  und Lernrate 0,1. Donorfarben werden gegen die Labeldatei geprüft; alle drei
  Procrustes-Werte werden aus den Koordinaten erneut berechnet.
- Figure 2 verwendet die unverändert ausgewählten K-means-/Ward-/Leiden-Modelle
  auf 20.000 PBMCs. Nur diese drei Konfigurationen wurden für vollständige
  numerische Markerprofile rekonstruiert. Clustergrößen, DB-Werte und alle
  sichtbaren ursprünglichen Markerangaben stimmen; ein bereits abgeschnittener
  Leiden-16-Text kann nur anhand seines sichtbaren Präfixes verglichen werden.
  Es gab keinen neuen Grid- oder Stabilitätslauf.
- Figures 3 und 4 übernehmen die bereits geprüften Aufgabe-4/5-Report-Assets.
  Für diesen Bericht wurde kein Klassifikator neu trainiert und kein
  Citrus-Benchmark wiederholt.
- Supplement-Tabellen werden direkt aus den aktuellen Ergebnis-CSVs erzeugt.
  Die dargestellten UMAP-Karten zeigen die 16 Nachbar-/Abstandskombinationen
  bei Lernrate 0,1; die numerische Tabelle enthält alle 64 Einstellungen.

Abbildungen ohne erneute Clusterfits erzeugen:

```bash
MPLCONFIGDIR=/tmp/ssbi-report-mpl .venv/bin/python report/export_assets.py
.venv/bin/python report/build_tables.py
```

Nur falls die gespeicherten vollständigen Clusterprofile neu hergestellt werden
müssen, ist `--reconstruct-clusters` vorgesehen. Es benötigt Originaldaten und
die bestehende Python-Umgebung, rekonstruiert aber ausschließlich die drei
bereits festgelegten Lösungen. Die Originalnotebooks bleiben unangetastet.

## Prüfungen

Ein separates [unabhängiges Review](REVIEW.md) ist abgeschlossen. Beide konkreten
Befunde wurden korrigiert; es gibt keine offenen kritischen Findings.

[validation.json](data/validation.json) hält den tatsächlichen PDF-Umfang,
die Bonus-Höhe, Quelldateischutz und numerische Prüfungen fest.
[figure_provenance.json](data/figure_provenance.json) und
[table_provenance.json](data/table_provenance.json) enthalten Quellen und
Dateiprüfsummen. Alle fünf Hauptseiten und die Supplement-Abbildungstypen
wurden außerdem visuell auf Lesbarkeit, Skalen und Bildunterschriften geprüft.

Die automatische Dokumentprüfung benötigt zusätzlich PyMuPDF. In der vorhandenen
Arbeitsumgebung lautet der Aufruf:

```bash
PYTHONPATH=/tmp/report_korrigiert_pdf_tools MPLCONFIGDIR=/tmp/ssbi-report-mpl \
  .venv/bin/python report/check_report.py
```

Numerische Konsistenz und saubere Testspender-Trennung sind keine unabhängige
biologische Validierung. Diese Grenze, die kleine verwandtschaftlich teilweise
verbundene Kohorte und die verschiedenen Auswahlregeln werden im Bericht genannt.
