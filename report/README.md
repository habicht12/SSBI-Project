# Editing the report

## Deutscher Detail- und Prüfbericht

**Historischer Stand:** Der Detailbericht und seine Assets dokumentieren den
Lauf vom 6. September 2026 (einschließlich der alten Citrus-Konfiguration und
Aufgabe 5). Der aktuelle Aufgabe-4-Vergleich steht im Kurzbericht `main.tex`
und im Comparison-Notebook. Den Detailexport erst nach Aktualisierung von
Aufgabe 5 erneut ausführen; seine bisherigen Zahlen sind nicht die neuen
Klassifikationsergebnisse.

`detailbericht_de.tex` erklärt die Aufgaben 1–5 einschließlich der einzelnen
Schritte von CellCNN, Citrus und SVM, der Zellinterpretation und der Grenzen der
Schlussfolgerungen. Die eigene PDF liegt neben der englischen Kurzfassung unter
`build/detailbericht_de.pdf`. LaTeX Workshop erkennt beide Hauptdateien anhand
ihrer jeweiligen Root-Direktive.

```bash
latexmk -cd -pdf -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=build report/detailbericht_de.tex
```

Zusätzlich zur Ausstattung der Kurzfassung werden `babel-german` und
`hyphen-german` für deutsche Beschriftungen und Silbentrennung benötigt. In der
vorhandenen TinyTeX-Installation sind diese Pakete eingerichtet.

Der separate Export wird nur nach einer bewussten Aktualisierung der Analyse
gestartet, in der Python-Projektumgebung (hier `.venv/bin/python`):

```bash
python -m src.detail_report_assets
python -m unittest discover -s tests -p 'test_detail_report_assets.py' -v
```

Er liest die vorhandenen Tabellen der Aufgaben 2–5 und die Original-FCS-Dateien
beider Gates ausschließlich lesend. Er prüft die ursprünglichen Stichproben,
Zellzuordnungen, 1.200 Metrikwerte, Auswahlregeln, Häufigkeitsnenner,
5.180 Spender-Markerprofile und die Aufgabe-5-Provenienz. Gespeicherte
Modellrekonstruktionsprüfungen werden auf Dateiidentität geprüft und zusammengefasst;
sie werden nicht mit neu ausgeführten Modellinferenztests gleichgesetzt.

Alle Ausgaben liegen getrennt unter `detail_assets/`: deutsche Grafiken,
LaTeX-Tabellen, Zahlenmakros sowie `pruefung.json` und `quellen.json`.
Diese ausgewählten Berichtsassets sind zur Versionierung vorgesehen. Der Export
trainiert keine Modelle und verändert keine Dateien unter `results/`.
Der normale LaTeX-Build benötigt nur die fertigen Berichtsassets, keine lokalen
Originaldaten und keine erneute Analyse.

Nach Text- oder Abbildungsänderungen die maximale Gesamtlänge von 40 Seiten und
die Lesbarkeit kontrollieren. Neu entdeckte Probleme werden im Prüfteil benannt;
die Analysepipeline wird durch den Bericht nicht stillschweigend korrigiert.

## English five-page report

Open `main.tex` in VS Code. LaTeX Workshop builds on save and shows the PDF in a
tab beside the source. Use **LaTeX Workshop: View LaTeX PDF file** in the command
palette to open the preview. SyncTeX connects source locations and PDF positions.
The output is `report/build/main.pdf`.

From the repository root, the same build is:

```bash
latexmk -cd -pdf -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=build report/main.tex
```

TeX Live with `latexmk`, pdfLaTeX, Latin Modern, `natbib`, `booktabs`, `geometry`,
`caption` and `microtype` is sufficient. In this WSL environment, TinyTeX (a small
TeX Live distribution) is installed in `~/.TinyTeX`, with commands linked into
`~/.local/bin`; LaTeX Workshop is installed on the WSL side. A fresh machine needs
these tools installed separately. If VS Code was already running during installation
and cannot find `latexmk`, reload the window and ensure `~/.local/bin` is on PATH.
See the [TinyTeX installation guide](https://yihui.org/tinytex/) and
[LaTeX Workshop documentation](https://github.com/James-Yu/LaTeX-Workshop/wiki).

## Content and figures

Für eine gezielte Aktualisierung ausschließlich der Aufgabe-4-Tabellen:

```bash
python -m src.report_assets --classification-only
```

Dies erzeugt `tables/classification.tex` für den Dreiervergleich über zehn
gemeinsame Splits sowie `tables/classification_cellcnn_svm_100.tex` für den
zusätzlichen CellCNN–SVM-Vergleich über 100 Splits. Aufgabe-5-Dateien und
Abbildungen werden dabei nicht gelesen oder verändert. Der Interpretationsteil
des Kurzberichts bleibt bis zur gesonderten Bearbeitung von Aufgabe 5 historisch.

- Edit prose and captions in `main.tex`, and references in `references.bib`.
- Replace the author placeholder before submission.
- The Task 6 minipage reserves half the text height of one page. Replace the whole
  minipage when the extension is ready. No extension results are implied by it.
- Explicit page breaks organize the initial five-page draft. Recheck page count
  after larger edits; the five-page limit includes figures, tables and references.
- The selected PDFs in `figures/` and LaTeX table in `tables/` belong to the report.
  They make normal compilation independent of the local dataset, Python and R.

To refresh those assets after an intentional analysis update, activate the project's
Python environment and run from the repository root:

```bash
python -m src.report_assets
```

The exporter reads saved results under `results/tables/`. Figure 1 also reads the
exact saved events from the original `gated_alive` FCS files for CD3 colours and
checks their exploration checksum. It does not fit projections, clustering or
classifiers, and does not change the analysis outputs. Missing or inconsistent
inputs must be resolved in the analysis before exporting. Run this command
separately; normal LaTeX builds never run analyses or regenerate figures.

Figure sources: `task2_embeddings`, `task2_pairwise_jaccard` and `task2_cells`
(Figure 1); `task3_marker_profiles`, `task3_recommendations` and saved exploratory
scaling (Figure 2); `task5_cell_scores` and `task5_report_marker_table` (Figure 3).
The classification table uses `task4_metric_summary`, checked against
`task4_split_metrics`. All these inputs are CSV/JSON exports in `results/tables/`.
The first report draft uses the complete `gated_alive` runs from 5 September 2026.

Generated assets do not automatically rewrite claims in the prose: check numbers
and conclusions in the text whenever the underlying analyses change. Build files
under `build/` are ignored by Git; source, selected assets and project VS Code
settings are intended to be versioned. No raw data are included.
