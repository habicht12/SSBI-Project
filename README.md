# NK/CMV Single-Cell Analysis

Dieses Repository enthält die Analysen des SSBI-Gruppenprojekts zum
NK/CMV-Datensatz. Der Datensatz selbst wird nicht mit Git versioniert. Alle
Mitwirkenden verwenden dieselbe Original-ZIP-Datei und entpacken sie lokal.

## Voraussetzungen

- Git
- Miniforge oder Miniconda
- die Datei `nk_cell_dataset.zip`

Unter Windows sollten die folgenden Befehle im Miniforge Prompt oder Anaconda
Prompt ausgeführt werden.

## Installation

```bash
git clone <REPOSITORY-URL>
cd <REPOSITORY-ORDNER>
conda env create -f environment.yml
conda activate ssbi-group-project
```

## Datensatz einrichten

Die vorhandene ZIP-Datei unverändert direkt in den Projektordner entpacken. Die
Verschachtelung des Originalarchivs bleibt dabei erhalten. Danach muss dieser
Pfad existieren:

```text
NK_cell_dataset/NK_cell_dataset/NK_cell_dataset/gated_NK/a_001_NK.fcs
```

Die Installation und den Datenpfad prüfen:

```bash
python untersuchung/inspect_fcs.py
```

Die Ausgabe sollte unter anderem 40 echte FCS-Dateien, 20 Spender und 37
Analysemarker melden.

## Projekt starten

```bash
jupyter lab
```

JupyterLab muss aus der aktivierten Conda-Umgebung gestartet werden.
Für die übernommenen Python-Notebooks den Projektkernel registrieren:

```bash
python -m ipykernel install --user --name ssbi-group-project --display-name "Python (SSBI Group Project)"
```

## Verbindliche Grundlage für den Bericht

Maries vorhandener Berichtsteil liegt als [LaTeX](praesentation/report.tex)
und [PDF](praesentation/report.pdf) vor. Ihr Text, ihre PDF und die bisherigen
Berichtsabbildungen bleiben auf ausdrücklichen Wunsch unverändert. Neue
Abschnitte zu Aufgaben 3–6 werden noch nicht geschrieben.

Die gehaltene Gesamtpräsentation bleibt als [PDF](praesentation/main.pdf) und
[LaTeX-Quelle](praesentation/main.tex) mit ihren bisherigen Assets unverändert.
Neue Berichtsassets liegen getrennt unter `praesentation/report_assets/`.

| Aufgabe | Aktuelles Notebook | Herkunft bei der Zusammenführung |
| --- | --- | --- |
| 2 | [02_dimensionality_reduction_clean.ipynb](notebooks/02_dimensionality_reduction_clean.ipynb) | Vollständiger Neulauf vom 13.09.2026 auf Basis von `main`/`4045b3f` |
| 3 | [03_clustering.ipynb](notebooks/03_clustering.ipynb) | `main`, Notebook-Änderung `27949d2` |
| 4 | `04a_data_qc_and_splits.ipynb`, `04b_svm.ipynb`, `04c_cellcnn.ipynb`, `04d_citrus.ipynb`, `04e_comparison.ipynb` | `GrHa`, `90fbcbf` |
| 5 | [05_interpretation.ipynb](notebooks/05_interpretation.ipynb) | `GrHa`, `90fbcbf` |
| 6 | Ausgeführte `06a`-, `06b`-, `06d`- und `06e`-Fassungen mit Suffix `_100.ipynb` (Links unten) | Je 100 gemeinsame Splits; unveränderter Modellcode aus `GrHa-learnable-pooling`/`1f00db2`, erneut geprüft |

Die übrigen Aufgabe-2-Varianten bleiben historische Referenzen. Für den Bericht
ist ausschließlich der durchgehend ausgeführte `_clean`-Stand maßgeblich. Die historischen Bonusnotebooks
`06b_cellcnn_mahalanobis_relu_threshold.ipynb`, `06b_ergebnisse_100_splits.ipynb`
und `06c_cellcnn_mahalanobis_top1.ipynb` stammen aus `GrHa`; ihre Experimente
sind vom aktuellen Vier-Modell-Vergleich zu unterscheiden. Insbesondere nutzt
`06b_ergebnisse_100_splits.ipynb` die historischen Tabellen in
`results/tables/bonus_100/`.

### Vollständige Aufgabe-6-Ergebnisse

| Modell | Ausgeführtes Notebook |
| --- | --- |
| CellCNN-Baseline | [06a – 100 Splits](notebooks/06a_cellcnn_baseline_bonus_100.ipynb) |
| Quadratic Top-1 % | [06d – 100 Splits](notebooks/06d_cellcnn_quadratic_100.ipynb) |
| Mahalanobis/ReLU Soft-α | [06b – 100 Splits](notebooks/06b_cellcnn_mahalanobis_learnable_pooling_100.ipynb) |
| Quadratic Soft-α | [06e – 100 Splits](notebooks/06e_cellcnn_quadratic_learnable_pooling_100.ipynb) |

Die Ausgaben sind gespeichert und ohne erneutes Training lesbar. Die Fassungen
setzen den Umfang ausdrücklich auf 100 Splits und deaktivieren neues
Benchmark-Training. Die kleinen eingebauten Smoke-Tests laufen weiterhin.
Für eine erneute Ausführung aus dem Projektstamm oder Notebook-Ordner werden die
Originaldaten, eine zur gespeicherten Konfiguration passende CUDA-Umgebung
und lokale Modellartefakte in `results/tables/` benötigt. Für den deterministischen
CUDA-Lauf muss `CUBLAS_WORKSPACE_CONFIG=:4096:8` vor dem Kernelstart gesetzt
sein, beispielsweise unter Bash mit
`CUBLAS_WORKSPACE_CONFIG=:4096:8 jupyter lab`.
Die ursprünglichen Trainingsnotebooks ohne `_100` bleiben unverändert: Ihre
Code-Hashes werden bei der Modellprüfung als Referenz verwendet.

Kompakte gemeinsame Auswertung:
[Zusammenfassung](praesentation/report_assets/aufgabe_06/data/summary.csv),
[Metriken je Split](praesentation/report_assets/aufgabe_06/data/split_metrics.csv),
[gepaarte Differenzen](praesentation/report_assets/aufgabe_06/data/paired_differences.csv)
und [Herkunftsnachweis](praesentation/report_assets/aufgabe_06/data/provenance.json).
Die weiteren CSVs dort enthalten Vorhersagen, Frequenzen, ROC-Kurven und
Poolingparameter. Alle vier Modelle verwenden 100 Netzwerk-Splits und dieselben
98 bestimmbaren Frequency-Splits; bei der Baseline fehlen positive Filter in
Splits 44 und 84. Ergebnisse entsprechen dem vollständigen Vergleich der
gehaltenen Präsentation. Mediane aus Aufgabe 4 sind von Mittelwerten in Aufgabe 6
zu unterscheiden.

### Aufgabe 2: Bericht und gehaltene Präsentation

Die aktuelle Notebook-Grundlage nutzt sechs t-SNE-Einstellungen und 64 UMAP-Kombinationen
(einschließlich Lernrate). Alle finalen Karten, Qualitätsmaße und Exporte stammen
aus demselben frischen Kernel. Die exakte blockweise Trustworthiness-Berechnung
wurde gegen scikit-learn geprüft und wertet alle 40.000 Zellen aus.
[Ergebnisse und Parameter](praesentation/report_assets/aufgabe_02/data/provenance.json),
[UMAP-Sweep](praesentation/report_assets/aufgabe_02/data/umap_sweep.csv),
[Qualitätsmaße](praesentation/report_assets/aufgabe_02/data/quality.csv).

Der vollständige Lauf vom 13.09.2026 wählt t-SNE-Perplexität **70**
(Sweep-Trustworthiness **0,927102**). UMAP wählt weiterhin **5 Nachbarn,
min_dist=0 und Lernrate 0,1**. Die Auswahl erfolgt zweistufig: zuerst die
Lernrate nach mittlerer Trustworthiness zweier festgelegter Konfigurationen,
dann Nachbarzahl und Mindestabstand innerhalb dieser Lernrate. Das ist keine
Auswahl des einzelnen Maximums über alle 64 Kombinationen.

Die finalen 40.000-Zell-Karten ergeben Trustworthiness **0,725804 / 0,960849 /
0,883767** für PCA / t-SNE / UMAP. Im aktuellen Procrustes-Vergleich sind
t-SNE und UMAP das ähnlichste Paar (**0,368308**). Diese neuen Werte ersetzen
die früheren gemischten Ausgaben ausschließlich im Ergebnisnotebook und den
separaten Assets. Maries bestehender Text und seine alten Tabellen bleiben
unverändert; die abweichenden Parameter und Schlussfolgerungen müssen vor der
späteren Berichtsfertigstellung gemeinsam abgeglichen werden.

Die gehaltenen Folien zeigen dagegen den früheren 16er-UMAP-Sweep. Außerdem
nennt ihr t-SNE-Text acht Werte, obwohl die zugehörigen `_clean`-Notebooks sechs
prüfen. Diese historischen Folien werden nicht nachträglich umgeschrieben;
Maries Berichtsteil bleibt ebenfalls unverändert. Die neuen Assets liegen für
eine spätere, gesondert abgestimmte Überarbeitung bereit; ihre bestehenden
Tabellen und Abbildungen werden nicht automatisch ersetzt.
Die Aufgabe-5-Karte bleibt die ältere `gated_alive`-Referenz mit 10.000 Zellen
(`02_interpretation_reference.ipynb`), nicht die aktuelle `gated_NK`-Karte mit
40.000 Zellen aus Aufgabe 2. Ergebnisse und Methoden zu Aufgaben 3–5 wurden im
vorangegangenen Benchmark geprüft und werden durch diese Vorbereitung nicht
verändert.

### Voraussetzungen und Ausführungsreihenfolge

1. `01_data_exploration.ipynb` erzeugt die Eingaben für Aufgabe 2;
   anschließend `02_dimensionality_reduction_clean.ipynb` ausführen.
   Aufgabe 3 lädt ihre Daten eigenständig.
2. Für Aufgabe 4 zunächst `04a`, dann die drei Methoden `04b`–`04d`, zuletzt
   `04e` ausführen. Citrus benötigt den separaten R-Kernel (siehe unten).
   Der Hauptvergleich verwendet `gated_alive`: Methodenläufe mit
   `TASK4_RUN_MODE=full`; `TASK4_RUN_TRAINING=0` lädt passende vorhandene
   Ergebnisse. Ohne Überschreibung verwenden die Methoden den technischen
   Smoke-Modus auf `gated_NK`.
3. Aufgabe 5 benötigt die vollständigen Aufgabe-4-Ergebnisse sowie die
   ursprüngliche `gated_alive`-Referenzkarte. Diese wird durch
   [02_interpretation_reference.ipynb](notebooks/02_interpretation_reference.ipynb)
   erzeugt: unveränderte Kopie des früheren Aufgabe-2-Notebooks aus `GrHa`,
   mit 500 Zellen je Spender und `task2_*`-Exporten. Sie dient ausschließlich
   der Reproduktion der Interpretationskarten; das aktuelle Aufgabe-2-Notebook
   verwendet andere Daten und ersetzt diese Eingaben nicht.
4. Aufgabe 6 setzt die vollständigen CellCNN-Ergebnisse aus `04c` voraus.
   Zunächst `06a`, dann die gewünschten Varianten ausführen; `06e` benötigt
   zusätzlich die Hard-Pooling-Referenz aus `06d`. Die Notebooks dokumentieren
   ihre Laufparameter (`TASK6_*`) und prüfen gespeicherte Konfigurationen.
   Historische und aktuelle Versuche in getrennte Ergebnisordner schreiben,
   da einzelne Standarddateinamen übereinstimmen. Vorhandene Notebook-Ausgaben
   ersetzen die für eine Neuausführung benötigten Tabellen und Modelle nicht.

Originaldaten, lokale Arbeitsdateien und Modell-Checkpoints bleiben unversioniert.
Die ausdrücklich ausgewählten kompakten Ergebnisexports für den Bericht sind
unter `praesentation/report_assets/` versioniert. Ein vollständiger Benchmark
ist kein routinemäßiger Integrationstest.

### Separate Citrus-Umgebung

```bash
conda env create -f environment-citrus.yml
conda run -n ssbi-citrus Rscript -e 'remotes::install_github("nolanlab/Rclusterpp@a07380683ce7a6849af8ec27db6439ea3a707890", upgrade="never", dependencies=FALSE, build_vignettes=FALSE)'
conda run -n ssbi-citrus Rscript -e 'remotes::install_github("nolanlab/citrus@d02baae544abdc403704aaceb75d1e7931a0331c", upgrade="never", dependencies=FALSE, build_vignettes=FALSE)'
conda run -n ssbi-citrus Rscript -e 'IRkernel::installspec(name="ssbi-citrus", displayname="R (SSBI Citrus)", user=TRUE)'
```

## Projektstruktur

```text
notebooks/       Analysen in Ausführungsreihenfolge
src/             wiederverwendbarer Python-Code
untersuchung/    Datensatzprüfung und Voruntersuchung
results/figures/ erzeugte Abbildungen
results/tables/  erzeugte Tabellen
```

Der lokale Datensatz, die Conda-Umgebung und temporäre Analyseergebnisse werden
nicht mit Git versioniert. Die ausgewählten Berichtsassets und ausgeführten
Ergebnisnotebooks bilden die gemeinsam lesbare Ausnahme.

## Datenquelle und Referenzen

- Datensatz: [Benchmark Datasets CellCnn, Zenodo](https://doi.org/10.5281/zenodo.5597098), CC BY 4.0
- Paper: [Arvaniti & Claassen, *Sensitive detection of rare disease-associated cell subsets via representation learning*, Nature Communications (2017)](https://doi.org/10.1038/ncomms14825)
