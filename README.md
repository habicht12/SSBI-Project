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
python -m ipykernel install --user --name ssbi-group-project --display-name "Python (SSBI Group Project)"
```

Optional kann CellCNN auf einem Linux-System mit kompatibler NVIDIA-GPU über
den offiziellen CUDA-Wheel beschleunigt werden. Die portable Standardumgebung
bleibt CPU-fähig:

```bash
python -m pip install --upgrade torch==2.14.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Das CellCNN-Notebook wählt CUDA automatisch, wenn PyTorch eine GPU erkennt.

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
Python-Notebooks verwenden den oben registrierten Kernel
`Python (SSBI Group Project)`; das Citrus-Notebook verwendet den getrennten
R-Kernel `R (SSBI Citrus)`.

### Separate Citrus-Umgebung

Die originale Citrus-Implementierung benötigt R und wird getrennt von der
Python-Hauptumgebung installiert:

```bash
conda env create -f environment-citrus.yml
conda run -n ssbi-citrus Rscript -e 'remotes::install_github("nolanlab/Rclusterpp@a07380683ce7a6849af8ec27db6439ea3a707890", upgrade="never", dependencies=FALSE, build_vignettes=FALSE)'
conda run -n ssbi-citrus Rscript -e 'remotes::install_github("nolanlab/citrus@d02baae544abdc403704aaceb75d1e7931a0331c", upgrade="never", dependencies=FALSE, build_vignettes=FALSE)'
conda run -n ssbi-citrus Rscript -e 'IRkernel::installspec(name="ssbi-citrus", displayname="R (SSBI Citrus)", user=TRUE)'
```

Das Notebook `notebooks/04d_citrus.ipynb` verwendet anschließend den Kernel
`R (SSBI Citrus)`. Alle übrigen Notebooks bleiben in der Python-Umgebung.

### Aufgabe 4 ausführen

Ein normales `Run All` in den Methoden-Notebooks verwendet den schnellen
Smoke-Modus auf `gated_NK`. Vorhandene Ergebnisse werden erkannt und nicht neu
trainiert. Nach `04a_data_qc_and_splits.ipynb` werden die vollständigen Läufe
unter Bash beziehungsweise WSL so gestartet:

```bash
TASK4_RUN_MODE=full TASK4_RUN_TRAINING=1 jupyter nbconvert --to notebook --execute --inplace notebooks/04b_svm.ipynb --ExecutePreprocessor.kernel_name=ssbi-group-project --ExecutePreprocessor.timeout=14400
TASK4_RUN_MODE=full TASK4_RUN_TRAINING=1 jupyter nbconvert --to notebook --execute --inplace notebooks/04c_cellcnn.ipynb --ExecutePreprocessor.kernel_name=ssbi-group-project --ExecutePreprocessor.timeout=14400
TASK4_RUN_MODE=full TASK4_RUN_TRAINING=1 jupyter nbconvert --to notebook --execute --inplace notebooks/04d_citrus.ipynb --ExecutePreprocessor.kernel_name=ssbi-citrus --ExecutePreprocessor.timeout=14400
```

Mit `TASK4_RUN_TRAINING=0` werden stattdessen vorhandene vollständige
Ergebnisse geladen. `TASK4_SPLIT_LIMIT=N` begrenzt einen technischen Lauf auf
die ersten `N` Splits. In PowerShell werden die Variablen vor dem Aufruf mit
`$env:TASK4_RUN_MODE="full"` und `$env:TASK4_RUN_TRAINING="1"` gesetzt.

Die erzeugten Dateien unter `results/` sind lokale, von Git ignorierte
Analyseartefakte. Daher müssen auf einem frischen Clone zuerst `04a` und danach
die drei vollständigen Methodenläufe ausgeführt werden. Erst anschließend kann
`04e_comparison.ipynb` den vollständigen Vergleich neu berechnen. Die im
Repository gespeicherten Notebook-Ausgaben bleiben auch ohne diese lokalen
CSV-Dateien sichtbar.

Die Notebooks sind in dieser Reihenfolge vorgesehen:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_dimensionality_reduction.ipynb`
3. `notebooks/03_clustering.ipynb`
4. `notebooks/04a_data_qc_and_splits.ipynb`
5. `notebooks/04b_svm.ipynb`
6. `notebooks/04c_cellcnn.ipynb`
7. `notebooks/04d_citrus.ipynb`
8. `notebooks/04e_comparison.ipynb`

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
