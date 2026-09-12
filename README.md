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

## Aktueller gemeinsamer Stand: Aufgaben 2–6

Die Gesamtpräsentation liegt als [PDF](praesentation/main.pdf) und
[LaTeX-Quelle](praesentation/main.tex) auf `main`. Die dort bereits vorhandene
weiterbearbeitete Fassung vom 12.09.2026 bleibt erhalten.

| Aufgabe | Aktuelles Notebook | Herkunft bei der Zusammenführung |
| --- | --- | --- |
| 2 | [02_dimensionality_reduction_clean.ipynb](notebooks/02_dimensionality_reduction_clean.ipynb) | `main`, `4045b3f` |
| 3 | [03_clustering.ipynb](notebooks/03_clustering.ipynb) | `main`, Notebook-Änderung `27949d2` |
| 4 | `04a_data_qc_and_splits.ipynb`, `04b_svm.ipynb`, `04c_cellcnn.ipynb`, `04d_citrus.ipynb`, `04e_comparison.ipynb` | `GrHa`, `90fbcbf` |
| 5 | [05_interpretation.ipynb](notebooks/05_interpretation.ipynb) | `GrHa`, `90fbcbf` |
| 6 | `06a_cellcnn_baseline_bonus.ipynb`, `06b_cellcnn_mahalanobis_learnable_pooling.ipynb`, `06d_cellcnn_quadratic.ipynb`, `06e_cellcnn_quadratic_learnable_pooling.ipynb` | `GrHa-learnable-pooling`, `1f00db2` |

Die Notebook-Inhalte werden unverändert aus diesen Ständen übernommen.
Die übrigen Aufgabe-2-Varianten bleiben erhalten. Die historischen Bonusnotebooks
`06b_cellcnn_mahalanobis_relu_threshold.ipynb`, `06b_ergebnisse_100_splits.ipynb`
und `06c_cellcnn_mahalanobis_top1.ipynb` stammen aus `GrHa`; ihre Experimente
sind vom aktuellen Vier-Modell-Vergleich zu unterscheiden. Insbesondere nutzt
`06b_ergebnisse_100_splits.ipynb` die historischen Tabellen in
`results/tables/bonus_100/`.

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

Originaldaten, ignorierte Ergebnisdateien und Checkpoints werden bei dieser
Zusammenführung nicht hinzugefügt. Ein vollständiger Benchmark ist kein
routinemäßiger Integrationstest.

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

Der lokale Datensatz, die Conda-Umgebung und erzeugte Ergebnisse werden nicht
mit Git versioniert.

## Datenquelle und Referenzen

- Datensatz: [Benchmark Datasets CellCnn, Zenodo](https://doi.org/10.5281/zenodo.5597098), CC BY 4.0
- Paper: [Arvaniti & Claassen, *Sensitive detection of rare disease-associated cell subsets via representation learning*, Nature Communications (2017)](https://doi.org/10.1038/ncomms14825)
