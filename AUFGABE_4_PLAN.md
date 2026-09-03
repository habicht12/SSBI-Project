# Plan für Aufgabe 4: Klassifikation

## Ziel

Wir vergleichen drei Single-Cell-Klassifikationsansätze auf dem NK/CMV-Datensatz:

1. CellCNN
2. Citrus
3. lineare Single-Cell-SVM

Optional ergänzen wir eine einfache Moment-Baseline als Plausibilitätscheck. Sie
zählt nicht zu den drei Hauptmethoden.

## Gemeinsame Datenbasis

- Für den finalen Hauptvergleich `gated_alive` verwenden. Diese Dateien enthalten
  lebende, von Doubletten bereinigte PBMCs ohne zusätzliches NK-Gate und entsprechen
  damit der Analyse im Paper.
- `gated_NK` enthält dieselben 20 Spender nach einer zusätzlichen NK-Selektion. Die
  kleineren Dateien eignen sich für schnelle technische Tests und optional für eine
  Sensitivitätsanalyse, sind aber nicht die primäre Datenbasis.
- Die einzige Labeldatei `NK_fcs_samples_with_labels.csv` gilt für beide Gate-Stufen.
  Die Zuordnung erfolgt über die Spender-ID im Dateinamen; `0 = CMV−` und
  `1 = CMV+`.
- Die 37 Marker aus `NK_markers.csv` verwenden. Das Paper nennt 36 Marker, die
  offizielle CellCNN-Beispieldatei enthält jedoch dieselben 37 Marker wie unser
  Datensatz; diese kleine Abweichung wird im Bericht erwähnt.
- Markerwerte mit `arcsinh(x / 5)` transformieren.
- Spender gleich gewichten beziehungsweise pro Spender gleich viele Zellen ziehen.
- Training und Test immer nach Spender trennen, niemals einzelne Zellen eines
  Spenders auf beide Mengen verteilen.
- Eine Standardisierung nur auf den Trainingsspendern fitten und unverändert auf
  Validierungs- und Testspender anwenden.

### Qualitätskontrolle und Vorverarbeitung

- Der bereitgestellte Datensatz ist bereits vorverarbeitet und gegatet. Für den
  Hauptvergleich enthält `gated_alive` lebende, von Doubletten bereinigte PBMCs.
  Eine erneute umfangreiche Rohdaten-QC, insbesondere eine bead-basierte
  Normalisierung, ist daher zunächst nicht vorgesehen.
- Als Standard-QC pro Spender die Zellzahl, die Vollständigkeit und Reihenfolge
  der gemessenen Marker sowie fehlende, nicht-endliche oder offensichtlich
  ungültige Werte prüfen.
- Die Verteilungen der 37 Analysemarker allgemein und nach Spender
  kontrollieren, um auffällige Wertebereiche, einzelne Ausreißerproben oder
  mögliche technische Unterschiede zu erkennen. Diese Prüfung ist deskriptiv
  und darf keine Informationen aus Testlabels zur Vorverarbeitung nutzen.
- Zusätzliche Vorverarbeitung nur dann einführen, wenn diese Kontrollen einen
  konkreten Bedarf zeigen. Die Entscheidung, einschließlich einer Entscheidung
  gegen weitere Vorverarbeitung, anhand der QC-Ergebnisse begründen und im
  Bericht dokumentieren.
- Unabhängig vom QC-Ergebnis die papernahe Transformation `arcsinh(x / 5)` auf
  die 37 Analysemarker anwenden. Gelernte Schritte wie eine Standardisierung
  weiterhin ausschließlich auf den jeweiligen Trainingsspendern fitten.

## Vergleichsprotokoll

- Für alle Methoden dieselben zufälligen Seeds und spenderweisen Splits verwenden.
- Als Orientierung am Paper pro Wiederholung 7 CMV− und 7 CMV+ zum Training
  verwenden; die übrigen 6 Spender bilden den Testdatensatz.
- Hyperparameter nur mit den jeweiligen Trainingsspendern auswählen.
- Primäre Metrik: ROC-AUC auf Spender-Ebene.
- Zusätzlich: PR-AUC und Balanced Accuracy.
- Ergebnisse über wiederholte Splits als Median und Streuung berichten.
- In der optionalen Gate-Sensitivitätsanalyse für jeden identischen Split die
  gepaarte Differenz
  `ΔROC-AUC = ROC-AUC(gated_NK) − ROC-AUC(gated_alive)` berechnen. Sie zeigt,
  wie stark eine Methode von der manuellen NK-Vorauswahl profitiert. `gated_NK`
  ist dabei kein unabhängiger Validierungsdatensatz.
- Zusätzlich pro Split die gepaarten ROC-AUC-Differenzen zwischen CellCNN und den
  beiden Hauptbaselines auswerten. Dafür sind keine weiteren Modellläufe nötig.

## Methoden

### CellCNN

- Die offiziellen alten Notebooks `NK_cell.ipynb` und `NK_cell_ungated.ipynb`
  dienen als Referenz für Datenfluss, Modellaufbau, Parameter und Auswertung.
- Die Notebooks selbst basieren auf Python 2.7; der offizielle `python3`-Branch
  verwendet eine separate alte Python-3.7-/TensorFlow-Umgebung und enthält die
  Notebooks nicht mehr. Diese Legacy-Abhängigkeiten übernehmen wir nicht.
- CellCNN wird passend zur bestehenden Python-3.12-Projektumgebung klein und
  nachvollziehbar neu implementiert, vorzugsweise mit aktuellem PyTorch.
- Die portable Projektumgebung bleibt CPU-fähig. Falls ein kompatibler
  CUDA-PyTorch-Build verfügbar ist, darf CellCNN mit identischer Konfiguration
  auf der GPU ausgeführt werden; das verwendete Gerät wird in den Artefakten
  dokumentiert.
- Zufällige Multi-Cell-Inputs mit etwa 3.000 Zellen erzeugen.
- Wenige lernbare Filter verwenden und die stärksten Zellantworten poolen.
- Als Startwerte dienen die Angaben aus dem Paper: 3–5 Filter und Mittelung der
  stärksten 1 % der Zellantworten.
- Mehrere Multi-Cell-Vorhersagen zu einer Spenderwahrscheinlichkeit mitteln.
- Die Zellantworten für die anschließende Bearbeitung von Aufgabe 5 aufbewahren.

### Citrus

- Die offizielle R-Implementierung `nolanlab/citrus` in einer getrennten,
  reproduzierbaren R-Umgebung verwenden. Citrus und Rclusterpp auf feste
  Git-Commits pinnen.
- Clustering und Merkmalsauswahl innerhalb der Trainingsdaten durchführen.
- Clusterhäufigkeiten beziehungsweise Markerstatistiken als spenderbezogene
  Merkmale verwenden.
- Die Citrus-Ausgabe in dieselben Splits und Metriken wie CellCNN einordnen.

## Notebook-Aufteilung

- `04a_data_qc_and_splits.ipynb`: Laden, Standard-QC, Vorverarbeitungsentscheidung
  und gemeinsame spenderweise Splits.
- `04b_svm.ipynb`: technischer `gated_NK`-Test und lineare Single-Cell-SVM.
- `04c_cellcnn.ipynb`: moderne PyTorch-Reimplementierung von CellCNN.
- `04d_citrus.ipynb`: originale Citrus-R-Pipeline.
- `04e_comparison.ipynb`: ausschließlich Laden, Validieren und Vergleichen der
  gespeicherten vollständigen Spendervorhersagen.
- Lange Methodenläufe speichern nach jedem äußeren Split einen Zwischenstand
  und können fortgesetzt werden. Smoke- und Full-Ergebnisse werden getrennt.

### Lineare Single-Cell-SVM

- Trainingszellen erhalten zunächst das Label ihres Spenders.
- Aus jedem Trainingsspender gleich viele Zellen verwenden.
- Eine lineare SVM auf den transformierten und standardisierten Markern trainieren.
- Für jede Zelle den kontinuierlichen SVM-Score berechnen.
- Pro Spender den Mittelwert der höchsten 1 % der Zellscores als Vorhersagescore
  verwenden. Diese Aggregation wird vor dem finalen Test festgelegt.

### Optionale Moment-Baseline

- Pro Spender und Marker die ersten vier Momente berechnen.
- Daraus mit einer regularisierten logistischen Regression ein Spenderlabel
  vorhersagen.
- Die Baseline dient nur als Kontrolle, ob einfache globale Verteilungsmerkmale
  bereits für die Klassifikation ausreichen.

## Geplante Arbeitsreihenfolge

1. Gemeinsames Laden und die Standard-QC implementieren: Spender- und
   Zellzahlen, Marker-Konsistenz, fehlende oder ungültige Werte sowie kompakte
   Marker-Verteilungen prüfen. Danach den begründeten Vorverarbeitungsumfang
   festlegen, `arcsinh(x / 5)` anwenden und spenderweise Splits erzeugen.
2. Die Pipeline zunächst mit `gated_NK` und wenigen Splits technisch prüfen.
3. Die lineare SVM als schnellen Test der gesamten Auswertungspipeline umsetzen.
4. CellCNN implementieren und zunächst auf wenigen Splits testen.
5. Citrus einrichten und in dasselbe Vergleichsprotokoll integrieren.
6. Optional die Moment-Baseline ergänzen.
7. Den finalen Vergleich auf `gated_alive` mit allen festgelegten Wiederholungen
   ausführen.
8. Eine Ergebnistabelle, einen kompakten Methodenvergleich und die wichtigsten
   Abbildungen für den Bericht erstellen.

## Erwartete Darstellung

- Tabelle mit ROC-AUC, PR-AUC und Balanced Accuracy je Methode.
- Box- oder Punktplot der ROC-AUC über die identischen Splits.
- Kompakte Darstellung der gepaarten ROC-AUC-Differenzen zwischen den Methoden
  und, falls durchgeführt, zwischen den beiden Gate-Stufen.
- Kurze Diskussion von Leistung, Interpretierbarkeit und Laufzeit.
- Als kleinen Qualitätscheck untersuchen, ob Messtag oder Instrumentversion das
  CMV-Label erkennbar erklären könnten; das Ergebnis knapp dokumentieren.
- Für die Verbindung zu Aufgabe 5 prüfen, ob CellCNN über die wiederholten Splits
  stabil eine biologisch ähnliche `NKG2C+`/`CD57+` Population auswählt.
- Nur wenn nach der Hauptanalyse Zeit bleibt, auf wenigen festen Splits optional
  eine Ablation ohne `NKG2C` und `CD57` rechnen. Sie ist kein Pflichtbestandteil.
- Deutlicher Hinweis auf die kleine Zahl von nur 20 unabhängigen Spendern und die
  daraus resultierende Unsicherheit.
