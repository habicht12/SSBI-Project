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

JupyterLab muss aus der aktivierten Conda-Umgebung gestartet werden. Die
Notebooks verwenden den Standardkernel `Python 3`; eine manuelle
Kernelregistrierung ist nicht erforderlich.

Die Notebooks sind in dieser Reihenfolge vorgesehen:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_dimensionality_reduction.ipynb`
3. `notebooks/03_clustering.ipynb`
4. `notebooks/04_classification.ipynb`

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
