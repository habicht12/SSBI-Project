# Arbeitsregeln für dieses Repository

## Ziel und Prioritäten

- Arbeite exakt auf die Aufgabenstellung und den fünfseitigen Gruppenbericht hin.
- Priorität bei methodischen Entscheidungen: Aufgabenstellung, Paper,
  Referenzimplementierung, eigene Erweiterungen.
- Halte dich möglichst genau an das CellCNN-Paper. Kennzeichne und begründe jede
  bewusste Abweichung.
- Frage nach, wenn eine fachliche Unklarheit das Ergebnis oder die Paper-Nähe
  wesentlich verändern würde.

## Arbeitsweise

- Implementiere immer die kleinste korrekte und nachvollziehbare Lösung.
- Vermeide Overengineering: keine unnötigen Abstraktionen,
  Konfigurationsschichten, Frameworks, Dateien, Ausgaben oder Visualisierungen.
- Behebe bei Fixes nur die eigentliche Ursache. Vermeide unaufgeforderte
  Refactorings, kosmetische Nebenänderungen und Erweiterungen außerhalb des
  Auftrags.
- Verwende bestehende Projektkonventionen und Abhängigkeiten. Füge neue oder
  schwere Abhängigkeiten nur hinzu, wenn sie für die Aufgabe notwendig sind.
- Formuliere Ergebnisse präzise. Behaupte keine Verifikation, die nicht
  tatsächlich ausgeführt wurde.

## Daten und methodische Sicherheit

- Behandle Originaldaten, ZIP-Dateien und lokale Paper-PDFs als unveränderlich.
  FCS-Dateien dürfen nur gelesen, nicht bearbeitet oder umbenannt werden.
- Spender sind die unabhängigen Beobachtungen. Teile niemals Zellen desselben
  Spenders auf Training, Validierung und Test auf.
- Fitte Skalierung, Clustering, Merkmalsauswahl und andere gelernte
  Vorverarbeitung ausschließlich mit den jeweiligen Trainingsspendern.
- Verwende feste Seeds und für alle verglichenen Methoden identische
  spenderweise Splits.
- Gewichte Spender gleich und dokumentiere Sampling, Transformationen,
  Hyperparameter und Metriken so, dass die Analyse reproduzierbar bleibt.
- Verwende für den papernahen Hauptvergleich `gated_alive`. Nutze `gated_NK`
  nur für schnelle technische Tests oder eine klar bezeichnete optionale
  Sensitivitätsanalyse.

## Aufgabe 4

- Hauptmethoden sind CellCNN, Citrus und eine lineare Single-Cell-SVM.
- Eine Moment-Baseline ist optional und zählt nicht zu den drei Hauptmethoden.
- Bewerte die Klassifikation primär auf Spender-Ebene.
- Nutze die offiziellen Notebooks `NK_cell.ipynb` und
  `NK_cell_ungated.ipynb` als Referenz für CellCNN, führe sie aber nicht als
  Bestandteil der Hauptpipeline aus.
- Implementiere CellCNN passend zur bestehenden Python-3.12-Umgebung neu,
  vorzugsweise mit aktuellem PyTorch. Übernimm keine alte
  Python-2.7-/3.7-/TensorFlow-Umgebung in das Hauptprojekt.

## Code, Notebooks und Bericht

- Halte erklärende Analysen in den Notebooks und wiederverwendbare Logik in
  `src/`.
- Notebooks müssen mit einem frischen Kernel von oben nach unten ausführbar sein.
- Erzeuge nur Tabellen und Abbildungen, die eine konkrete Frage der
  Aufgabenstellung beantworten und kompakt in den Bericht passen.
- Beschrifte Abbildungen und Tabellen eindeutig und dokumentiere die dafür
  verwendeten Daten, Parameter und Metriken.
- Schreibe Berichtstext, Notebook-Erklärungen und Abbildungsbeschriftungen auf
  Deutsch; verwende konsistente englische Python-Bezeichner.

## Tests und Verifikation

- Prüfe jede Änderung unabhängig mit dem kleinsten aussagekräftigen Test.
- Beginne mit gezielten Unit- oder Smoke-Tests. Führe größere Integrationsläufe
  nur aus, wenn der kleine Test bestanden ist und der zusätzliche Lauf nötig ist.
- Teste jede Klassifikationsmethode zunächst isoliert auf einem kleinen,
  deterministischen Lauf und danach einmal gemeinsam über die Vergleichspipeline.
- Starte die vollständige wiederholte Cross-Validation nicht als Routinetest.
  Sie ist ein finaler Benchmark und wird nur ausgeführt, wenn sie ausdrücklich
  benötigt wird.
- Spare Laufzeit und Tokens, ohne relevante Fehlerpfade, Datenlecks oder die
  Aussagekraft des Tests zu opfern.

## Git und Artefakte

- Erhalte vorhandene Nutzeränderungen und beschränke Diffs auf den aktuellen
  Auftrag.
- Committe nur auf ausdrücklichen Wunsch und halte Commits klein und thematisch
  geschlossen.
- Committe keine lokalen Eingangsdaten oder durch Analyseläufe erzeugten
  Artefakte, die laut `.gitignore` ausgeschlossen sind.
