# Untersuchung von Aufgabenstellung, Paper und NK-Datensatz

Stand: 31. August 2026. Die Originaldateien wurden nicht verändert.

## Kurzfazit

Der Datensatz ist der NK/CMV-Beispieldatensatz aus dem offiziellen CellCNN-
Repository. Er umfasst 20 Spender mit zwei klinischen Gruppen: 11 CMV-negative
(`label=0`) und 9 CMV-positive (`label=1`) Personen. Für jeden Spender liegen
zwei FCS-Dateien mit unterschiedlichen Gates vor:

- `gated_alive`: lebende, von Doubletten bereinigte PBMCs; 3.438.750 Events.
- `gated_NK`: daraus gegatete NK-Zellen; 261.593 Events.

Ein Event entspricht einer einzelnen gemessenen Zelle. Jede FCS-Datei enthält
43 Kanäle. Die Datei `NK_markers.csv` wählt davon 37 biologische Marker für die
Analyse aus; sechs technische/QC-Kanäle bleiben außen vor.

Die tatsächlichen FCS-Daten belegen 607,09 MiB. Die vielen Dateien unter
`__MACOSX` und alle `.DS_Store`-Dateien sind nur macOS-Verpackungsmetadaten und
keine weiteren Messdaten.

## Dateien und Verzeichnisse

```text
Group_projects_ssbi_2026.pdf                   Aufgabenstellung (2 Seiten)
CellCNN.pdf                                    Hauptpaper (10 Seiten)
NK_cell_dataset/
├── NK_cell_dataset/
│   ├── NK_fcs_samples_with_labels.csv         20 NK-Dateinamen + binäres Label
│   ├── NK_markers.csv                         37 zu verwendende Marker
│   └── NK_cell_dataset/
│       ├── gated_alive/                       20 echte FCS-Dateien
│       └── gated_NK/                          20 echte FCS-Dateien
└── __MACOSX/                                  ignorierbare AppleDouble-Dateien
```

Die verschachtelte Wiederholung von `NK_cell_dataset` stammt offenbar aus dem
entpackten Archiv. Sie ist unschön, aber funktional.

## Technisches Format

Alle 40 echten Messdateien sind FCS 3.0 mit 32-Bit-Fließkommawerten
(`$DATATYPE=F`) und Big-Endian-Bytefolge (`$BYTEORD=4,3,2,1`). Die Daten wurden
2012 mit einem DVS Sciences CyTOF aufgenommen und aus Cytobank exportiert. Die
Dateien haben dieselben 43 Kurznamen in derselben Reihenfolge. Bei einem Teil
der Dateien unterscheiden sich lediglich Langnamen bzw. Wertebereiche in den
FCS-Metadaten; die Markerzuordnung über `$PnS` ist konsistent.

Die Rohwerte liegen nicht auf einer gemeinsamen, direkt PCA-tauglichen Skala:
Markerwerte reichen je nach Kanal von leicht negativen Hintergrundwerten bis in
den Hunderterbereich. Transformation und Skalierung müssen deshalb als eigener,
reproduzierbarer Vorverarbeitungsschritt festgelegt werden.

Die sechs nicht als Analysemarker ausgewählten Kanäle sind:

- `Time`
- `Cell_length`
- `Dead`
- `(La139)Dd` (kein zugeordneter biologischer Marker)
- `DNA1`
- `DNA2`

Alle Kanäle stehen in `channel_overview.csv`, alle Spender und Eventzahlen in
`sample_overview.csv`.

## Welche Gate-Stufe wofür?

Die offizielle CellCNN-Beispielkommandozeile verwendet genau
`NK_fcs_samples_with_labels.csv`, `NK_markers.csv` und den Ordner `gated_NK`.
Damit ist `gated_NK` die naheliegende Eingabe für den geforderten
Klassifikationsbenchmark.

Für die Clusteraufgabe mit der Frage nach biologischen Zelltypen ist
`gated_alive` informativer, weil es verschiedene PBMC-Zelltypen enthält. Im
Hauptpaper wird außerdem beschrieben, dass die NK/CMV-Analyse auf ungated PBMCs
nach Entfernung toter Zellen und Doubletten durchgeführt wurde. Welche Gate-
Stufe für jede Teilaufgabe benotet werden soll, sollte deshalb im Bericht
explizit begründet und im Zweifel mit dem Tutor geklärt werden.

## So lassen sich die Dateien ansehen

### Schnell und grafisch

FlowJo kann ganze Ordner mit FCS-Dateien per Drag-and-drop öffnen. Dafür nur
einen der beiden echten Ordner `gated_alive` oder `gated_NK` laden, nicht
`__MACOSX`. In FlowJo lassen sich einzelne Marker als Histogramm oder zwei
Marker als Scatter-/Density-Plot ansehen und Gates überprüfen. Das ist der
schnellste Weg für eine visuelle Plausibilitätskontrolle, aber nicht die beste
Basis für eine vollständig reproduzierbare Projektpipeline.

### Reproduzierbar in Python

FlowKit liest FCS 2.0/3.0/3.1 und kann Messwerte als NumPy-Array, Pandas-
DataFrame oder CSV bereitstellen. Es unterstützt außerdem ArcSinh-
Transformationen und Visualisierungen. Für dieses Projekt bietet sich danach
eine Python-/Notebook-Pipeline mit scikit-learn, UMAP und Scanpy/igraph für PCA,
t-SNE, UMAP, k-means, hierarchisches Clustering und Leiden an.

Wichtig: niemals einzelne Zellen desselben Spenders auf Training und Test
verteilen. Splits und Cross-Validation müssen auf Spenderebene erfolgen.

### Reproduzierbar in R

Bioconductors `flowCore` liest mit `read.FCS()` eine Datei oder mit
`read.flowSet()` mehrere FCS-Dateien. Das ist eine gute Alternative, falls die
Gruppe bereits R/Bioconductor verwendet.

### Ohne Zusatzpakete: der lokale Inspektor

`inspect_fcs.py` benutzt nur die Python-Standardbibliothek und verändert keine
Quelldatei:

```bash
python3 untersuchung/inspect_fcs.py
python3 untersuchung/inspect_fcs.py --channels
python3 untersuchung/inspect_fcs.py --preview a_001_NK.fcs --preview-rows 10
```

Der letzte Befehl zeigt die ersten Zellen CSV-artig im Terminal. Für echte
Plots und Analysen sollte FlowKit oder `flowCore` verwendet werden.

## Was die Aufgabenstellung verlangt

1. Paper und darin verglichene Single-Cell-Ansätze zusammenfassen.
2. PCA, t-SNE und UMAP einschließlich Parameterwirkung visualisieren und alle
   Visualisierungspaare mit einem quantitativen Maß vergleichen.
3. k-means, hierarchisches Clustering und Leiden biologisch und quantitativ
   vergleichen; dazu auch das ursprüngliche Datensatzpaper lesen.
4. Drei vergleichende Single-Cell-Klassifikationsansätze implementieren und
   ihre Klassifikationsleistung bewerten.
5. Einen quantitativen Score für die klassifikationsentscheidende Zellmenge
   definieren und diese Zellen in einer Einbettung visualisieren.
6. Optional CellCNN verbessern, zum Beispiel über eine andere Filterantwort.

Der Abschlussbericht soll einschließlich Abbildungen, Tabellen und Referenzen
fünf Seiten lang sein.

## Paper: das Wesentliche für diesen Datensatz

CellCNN formuliert die Aufgabe als Multiple-Instance-Learning: Ein Spender ist
ein ungeordnetes Set vieler Zellen und besitzt genau ein klinisches Label. Eine
gemeinsame lineare Filterbank bewertet jede Zelle. Pooling fasst pro Filter die
stärksten Zellantworten zu einer spenderbezogenen Repräsentation zusammen; eine
Ausgabeschicht sagt daraus das klinische Label vorher. Die Filter können danach
wieder auf einzelne Zellen angewendet werden, um die prädiktive Subpopulation
zu lokalisieren.

Für den NK/CMV-Benchmark beschreibt das Paper:

- 20 verwendete Spender nach Ausschluss von Sample 008: 11 CMV−, 9 CMV+.
- 100 Monte-Carlo-CV-Wiederholungen; jeweils 7 CMV− und 7 CMV+ im Training,
  sechs zurückgehaltene Spender als Testdaten.
- Innerhalb des Trainings eine verschachtelte dreifache CV zur Modellauswahl.
- 3.000 Zellen je Multi-Cell-Input, 200 Inputs je Sample, 3–5 Filter,
  Lernrate 0,01 und Mittelung der stärksten 1 % Zellantworten.
- Testmetrik: ROC-AUC auf den sechs zurückgehaltenen Spendern.
- Wichtigster Befund: eine seltene, CMV+-assoziierte, memory-like
  `NKG2C+`/`CD57+`-Population; mediane Test-ROC-AUC von CellCNN 1,0 und schlechtere
  Ergebnisse für Citrus.

Im gesamten Paper werden unter der Überschrift „Baseline methods“ jedoch mehr
als zwei Alternativen beschrieben: Distanz-basierte Outlier Detection,
Single-Cell-LR/SVM/Random Forest, momentbasierte Repräsentation, Denoising
Autoencoder und Citrus. Die Aufgabenstellung spricht dagegen von „two reported
baseline approaches“, ohne sie namentlich festzulegen. Für den NK-Teil zeigt die
zentrale Benchmark-Abbildung nur CellCNN gegen Citrus. Die genaue Auswahl der
zwei zusätzlich zu CellCNN zu implementierenden Baselines sollte deshalb vor
größerer Implementierungsarbeit beim Tutor bestätigt werden.

## Wichtige methodische Stolperfallen

- Nur 20 unabhängige Beobachtungseinheiten: Zellen sind keine unabhängigen
  Patienten. Alle Splits müssen gruppiert nach Spender erfolgen.
- Die Samples enthalten stark unterschiedliche Zellzahlen. Für gemeinsame
  Einbettungen und Klassifikatoren pro Spender balanciert subsamplen bzw.
  Spender gleich gewichten.
- Drei Aufnahmetermine und zwei Instrument-Softwarestände sind im FCS-Header
  sichtbar. Batch-Effekte sollten geprüft und nicht versehentlich als
  CMV-Signal gelernt werden.
- Transformation, Skalierung, PCA und andere gelernte Vorverarbeitung innerhalb
  jedes CV-Trainingsfolds fitten, nicht vorab auf allen Spendern.
- Für faire Vergleiche dieselben spenderweisen Splits, Seeds und Primärmetriken
  für alle Klassifikatoren verwenden. Bei 9/11 Klassen sind ROC-AUC und zusätzlich
  PR-AUC bzw. Balanced Accuracy sinnvoller als reine Accuracy.
- Die originale Implementierung zielt primär auf Python 2.7/Theano; es gibt
  zwar einen Python-3-Branch, aber für ein neues Projekt ist eine kleine moderne
  Reimplementierung typischerweise robuster als das alte Environment.

## Externe Referenzen

- Offizielles CellCNN-Repository: https://github.com/eiriniar/CellCnn
- CellCNN-Dokumentation: https://eiriniar.github.io/CellCnn/
- FlowKit: https://github.com/whitews/flowkit
- FlowJo FCS-Dokumentation:
  https://flowjo.com/docs/flowjo10/workspaces-and-samples/samples-and-file-types/ws-fcsfiles
- Bioconductor flowCore: https://bioconductor.org/packages/flowCore/
