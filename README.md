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

### Bericht bearbeiten

Der englische Bericht liegt in [report/main.tex](report/main.tex). LaTeX Workshop
kompiliert ihn in VS Code beim Speichern; die PDF entsteht unter
`report/build/main.pdf`. Der Entwurf umfasst fünf Seiten einschließlich Literatur
und einer halben reservierten Seite für Aufgabe 6. Anleitung und separater
Abbildungsexport: [report/README.md](report/README.md).

Der ergänzende deutsche Detail- und Prüfbericht liegt in
[report/detailbericht_de.tex](report/detailbericht_de.tex). Er erklärt die
Aufgaben 1–5 mit Formeln, Ergebnisgrafiken und einem Abgleich der
Schlussfolgerungen; seine PDF entsteht separat unter
`report/build/detailbericht_de.pdf`. Bauanleitung und eigener Export stehen
ebenfalls in der [Berichts-README](report/README.md).

### Notebooks starten

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

### Aufgaben 2 und 3 ausführen

Die Notebooks `02_dimensionality_reduction.ipynb` und `03_clustering.ipynb`
werden jeweils mit einem frischen Python-Kernel von oben nach unten ausgeführt.
Beide laden eigenständig dieselben 500 zufällig ausgewählten Zellen pro Spender
aus `gated_alive`, also 10.000 Zellen mit 37 Markern. Aufgabe 3 benötigt keine
gespeicherten Ergebnisse oder Kernelvariablen aus Aufgabe 2.

Für diese rein explorativen Analysen ist ausdrücklich vereinbart, alle 20 Spender
gemeinsam zu verwenden und die Standardisierung auf dieser Stichprobe zu fitten.
Das ist eine auf Aufgaben 2/3 begrenzte Ausnahme von der Trainingsspender-Regel;
die Vorverarbeitung und Ergebnisse des Klassifikationsbenchmarks bleiben getrennt.
Stabile Zellkennungen ermöglichen später das Zuordnen von Zellbewertungen zu den
explorativen Karten, ohne die Aufgabe-4-Modelle oder deren Scaler zu verändern.

Aufgabe 2 untersucht acht PCA-/t-SNE-/UMAP-Einstellungen und vergleicht ihre
lokalen Nachbarschaften untereinander sowie mit dem ursprünglichen Markerraum.
Aufgabe 3 vergleicht jeweils drei Einstellungen von K-Means, Ward-Clustering und
Leiden anhand spendergemittelter Silhouetten und biologischer Markerprofile.
Die Empfehlungen gelten für die jeweils untersuchten Einstellungen und Kriterien;
geometrische Clusterqualität ist kein Nachweis biologischer Zelltypen.

### Aufgabe 4 ausführen

Ein normales `Run All` in den Methoden-Notebooks verwendet den schnellen
Smoke-Modus auf `gated_NK`. Vorhandene Ergebnisse werden erkannt und nicht neu
trainiert, sofern Konfiguration und Artefakte vollständig zusammenpassen.
Nach `04a_data_qc_and_splits.ipynb` werden die vollständigen Läufe
unter Bash beziehungsweise WSL so gestartet:

```bash
TASK4_RUN_MODE=full TASK4_RUN_TRAINING=1 jupyter nbconvert --to notebook --execute --inplace notebooks/04b_svm.ipynb --ExecutePreprocessor.kernel_name=ssbi-group-project --ExecutePreprocessor.timeout=14400
TASK4_RUN_MODE=full TASK4_RUN_TRAINING=1 jupyter nbconvert --to notebook --execute --inplace notebooks/04c_cellcnn.ipynb --ExecutePreprocessor.kernel_name=ssbi-group-project --ExecutePreprocessor.timeout=14400
TASK4_RUN_MODE=full TASK4_RUN_TRAINING=1 jupyter nbconvert --to notebook --execute --inplace notebooks/04d_citrus.ipynb --ExecutePreprocessor.kernel_name=ssbi-citrus --ExecutePreprocessor.timeout=-1
```

Citrus verwendet im Full-Modus 10.000 Zellen je Trainings- und Testspender,
30 äußere Splits (IDs 0–29) und drei innere Folds. Zellzahl und Wiederholungen
sind gegenüber den 20.000 Zellen und 100 Splits im Paper als Rechenkompromiss
reduziert; die Mindestclustergröße bleibt
papernah bei 0,05 % (Anteil `0.0005`).
Der frühere Lauf mit 1.000 Zellen und 5 % wird vom Comparison zurückgewiesen.
Vor dem vollständigen Neulauf zunächst mit `TASK4_SPLIT_LIMIT=1` Laufzeit
und Speicherbedarf prüfen; das hierarchische Clustering ist deutlich aufwendiger.
Nach allen 30 Citrus-Splits `04e_comparison.ipynb` erneut ausführen; dieses
verwendet für den Dreiervergleich dieselben 30 Splits aller Methoden.
Zusätzlich wertet es CellCNN und SVM über alle 100 vorhandenen Splits aus
(`task4_cellcnn_svm_100_*`); die Vorhersagedateien dieser Methoden bleiben unverändert.
Vor dem Comparison werden die Konfigurationsnachweise aller drei Methoden,
Eingabeprüfsummen und Trainingscode geprüft. Für die Citrus-RDS-Datei wird die
vorhandene Conda-Umgebung `ssbi-citrus` benötigt. Rclusterpp verwendet eigene
OpenMP-Threads; `mc.cores = 1` begrenzt nur die R-Prozessparallelität.
Die aktuelle Interpretation in Aufgabe 5 und die Kurzberichtsabbildung verwenden
`task5_paper_*` auf denselben 30 Splits. Frühere `task5_*`-Dateien und der Detailbericht
bleiben historische Ergebnisse der alten Citrus-Konfiguration.
Nur die Aufgabe-4-Berichtstabellen lassen sich mit
`python -m src.report_assets --classification-only` aktualisieren.

Mit `TASK4_RUN_TRAINING=0` werden vorhandene Ergebnisse mit passendem
Konfigurationsnachweis geladen. `TASK4_SPLIT_LIMIT=N` begrenzt einen technischen Lauf auf
die ersten `N` Splits. In PowerShell werden die Variablen vor dem Aufruf mit
`$env:TASK4_RUN_MODE="full"` und `$env:TASK4_RUN_TRAINING="1"` gesetzt.

Neue Methodenläufe speichern neben den CSV-Dateien eine `.config.json`-Datei
(SVM/CellCNN) bzw. `.config.rds`-Datei (Citrus). Sie dokumentiert Modellparameter,
Paketversionen, relevante Implementierung und Inhaltsprüfsummen der Eingabedateien
einschließlich aller inneren und äußeren Splits. Abweichende Konfigurationen und
unvollständige Zwischenstände werden vor der Wiederverwendung abgewiesen.
Ein Split-Limit verändert diese Konfiguration nicht; bereits vorhandene weitere
Splits bleiben in den Dateien erhalten.

Die Methoden-Notebooks übernehmen Altbestände ohne Konfigurationsnachweis nicht
zum Fortsetzen oder als vermeintlich passend zur aktuellen Konfiguration.
Der Comparison setzt für Citrus die oben genannte Konfiguration voraus.
Vor einem neuen Lauf müssen die bisherigen Dateien der betroffenen Methode und
Gate-/Modus-Kombination einschließlich Konfigurations- und Modellparameterdateien
separat gesichert und aus den aktiven Ergebnispfaden verschoben werden. Es gibt
keine automatische Überschreibung bei Konfigurationskonflikten.

SVM-Läufe speichern zusätzlich `task4_svm_models_<gate>_<mode>.csv` mit
Markergewichten, Intercept, Klassenrichtung, Entscheidungsschwelle und
Trainings-Scaler. Neue CellCNN-Filterdateien enthalten beide Ausgabegewichte und
beide Ausgabebiases. Die Markerreihenfolge folgt `NK_markers.csv`; für die
Rekonstruktion gilt zuerst `arcsinh(x / 5)`, anschließend der gespeicherte Scaler.
Parameter-CSV-Dateien sollten mit `pd.read_csv(..., float_precision="round_trip")`
eingelesen werden. Für dieselbe Float32-Arithmetik wie beim Fit wird ein
`StandardScaler` mit den gespeicherten `mean_`, `scale_` und `n_features_in_`
wiederhergestellt und dessen `transform` verwendet.
Der vollständige Neulauf auf `gated_alive` vom 5. September 2026 enthält
für alle 100 Splits die Konfigurationsnachweise und diese Modellparameter.
Die älteren unvollständigen Full-Artefakte wurden ersetzt.

Der Vergleich berichtet primär ROC-AUC, ergänzend Average Precision und Balanced
Accuracy. Die bisherige trapezoidale PR-AUC bleibt als separate Metrik erhalten.
Die neue Spalte `average_precision` wird direkt aus vorhandenen Vorhersagen
berechnet und erfordert kein Neutraining. Für Citrus gelten bei Clusterzählung
und Profil-Export nur Koeffizienten mit `abs(coefficient) > 1e-10` als wirksam.

Die erzeugten Dateien unter `results/` sind lokale, von Git ignorierte
Analyseartefakte. Daher müssen auf einem frischen Clone zuerst `04a` und danach
die drei vollständigen Methodenläufe ausgeführt werden. Erst anschließend kann
`04e_comparison.ipynb` den vollständigen Vergleich neu berechnen. Die im
Repository gespeicherten Notebook-Ausgaben bleiben auch ohne diese lokalen
CSV-Dateien sichtbar.

### Ergebnisse der ersten Bonuslösung teilen

Der vollständige Lauf der ersten Bonusvariante (`06b`, diagonale Prototypfilter
mit ReLU-Radius und Mittelwert-Pooling) wird nach erfolgreichem Abschluss in
`notebooks/06b_ergebnisse_100_splits.ipynb` zusammengefasst. Dieses separate
Ergebnisnotebook zeigt alle 100 Splits und verwendet ausschließlich die kleinen,
versionierbaren Ergebnistabellen unter `results/tables/bonus_100/`. Es benötigt
zum erneuten Auswerten weder FCS-Daten noch Modell-Checkpoints oder eine GPU.
Notebook und Tabellen müssen gemeinsam committet und gepusht werden;
gespeicherte Notebook-Ausgaben sind dann direkt nach dem Pull sichtbar.
Die Veröffentlichung erfolgt erst nach Prüfung aller 100 Splits. Die übrigen
lokalen Analyseartefakte bleiben von Git ausgeschlossen.

### Aufgabe 5 ausführen

`05_interpretation.ipynb` interpretiert die gespeicherten Modelle der **30 gemeinsamen
Splits 0–29** auf `gated_alive`. Es benötigt die vollständigen Aufgabe-4-Artefakte
sowie Zell-IDs, t-SNE-Koordinaten und Provenienz aus Aufgabe 2. Solange Citrus
noch nicht alle 30 Splits exportiert hat, stoppt die Hauptauswertung mit einer
Meldung über die fehlenden Splits; es gibt keine automatische Teil-Auswertung.
Der vollständige Lauf über 30 Splits wurde am 9. September 2026 geprüft und ausgeführt.
Für eine Neuberechnung das Notebook mit frischem Python-Kernel von oben nach unten
ausführen. R/`ssbi-citrus` wird ausschließlich für die bestehende
Konfigurationsprüfung benötigt; es werden keine Citrus-Bäume rekonstruiert.

CellCNN verwendet pro Filter die Halbmaximum-Auswahl auf der ursprünglichen
Scaler-Stichprobe: 20.000 Zellen je tatsächlichem inneren Trainingsspender,
mit ursprünglichem Kandidatenseed und gespeichertem Modell-Scaler. Citrus
verwendet die bereits exportierten Zentroiden der Cluster mit wirksamen
Regressionskoeffizienten. Die Zentroiden werden je Methode im gemeinsamen
explorativen z-Markerraum aus Aufgabe 2 hierarchisch gruppiert: Average-Linkage,
Kosinusdistanz, Schnitt 0,4. Diese Regel ist eine **dokumentierte Übertragung der
Filtergruppierung des Originalcodes**, keine belegte Originalparametrisierung
der NK-Zentroidanalyse. Es wird kein neuer Scaler gefittet.

Die Wiederkehr zählt unterschiedliche Splits mit mindestens einem Gruppenmitglied,
geteilt durch 30 einschließlich Nullmodellen. Gruppen ab sechs Vorkommen werden
auf der bestehenden t-SNE-Karte gezeigt. Repräsentanten minimieren die summierten
Kosinusdistanzen innerhalb ihrer Gruppe. Ein repräsentativer CellCNN-Filter wird
zusätzlich explorativ auf alle Karten-Zellen angewandt. Die SVM behält die bisherigen
höchsten 1 % der vollständigen Testspender und ihre gerichteten OOF-Zellhäufigkeiten,
beschränkt auf die 30 gemeinsamen Splits. Gruppen- und Zellhäufigkeiten sind
unterschiedliche Größen und keine vergleichbaren Effektstärken.

Der neue Lauf exportiert ausschließlich `task5_paper_*`: Zentroiden mit Gruppen,
Gruppenzusammenfassung, SVM-Zellhäufigkeiten, CellCNN-Repräsentantenauswahl,
Provenienz und zwei Abbildungen. Die bisherigen `task5_*`-Ergebnisse bleiben
unverändert als historischer Stand. Der Kurzberichtsexporter liest ausschließlich
die aktuellen `task5_paper_*`-Ergebnisse und prüft ihre Quellen- und Ausgabeprüfsummen;
der Detailbericht bleibt historisch.
Es werden keine Klassifikatoren trainiert und keine Leistungsmetriken geändert.

Die Notebooks sind in dieser Reihenfolge vorgesehen:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_dimensionality_reduction.ipynb`
3. `notebooks/03_clustering.ipynb`
4. `notebooks/04a_data_qc_and_splits.ipynb`
5. `notebooks/04b_svm.ipynb`
6. `notebooks/04c_cellcnn.ipynb`
7. `notebooks/04d_citrus.ipynb`
8. `notebooks/04e_comparison.ipynb`
9. `notebooks/05_interpretation.ipynb`

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

## Gezielte Regressionstests

```bash
python -m unittest discover -s tests -p 'test_task23_analysis.py' -v
python -m unittest discover -s tests -p 'test_task4_artifacts.py' -v
conda run -n ssbi-citrus Rscript tests/test_task4_artifacts.R
python -m unittest discover -s tests -p 'test_task5_interpretation.py' -v
```

Die Aufgabe-2/3-Tests prüfen Sampling, Nachbarschaftsvergleich, Parameterwahl,
Clusterverfahren, Silhouetten und Markerprofile mit kleinen deterministischen
Daten. Die Aufgabe-4-Tests prüfen Konfigurationskonflikte, fehlende Modellparameter,
Spenderzuordnungen und die numerische Citrus-Nulltoleranz ohne Benchmarktraining.
Die Aufgabe-5-Tests prüfen Trainingsspender und Referenzsampling, gespeicherte
Scaler, Halbmaximum und Zentroiden, einmalige Splitzählung einschließlich
Nullmodellen, Clustering und Repräsentanten, Projektionszuordnung und die
SVM-Auswahl. Kleine technische Läufe verwenden explizit einzelne fertige Splits
über die exportfreien Helfer; `run_interpretation` und das Notebook verlangen
immer alle 30 Splits.

## Datenquelle und Referenzen

- Datensatz: [Benchmark Datasets CellCnn, Zenodo](https://doi.org/10.5281/zenodo.5597098), CC BY 4.0
- Paper: [Arvaniti & Claassen, *Sensitive detection of rare disease-associated cell subsets via representation learning*, Nature Communications (2017)](https://doi.org/10.1038/ncomms14825)
