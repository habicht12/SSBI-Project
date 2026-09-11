# Aufgaben 4 und 5: Klassifikation und Interpretation

[slides.pdf](slides.pdf) enthält zwölf englische Hauptfolien und 13 Reservefolien
im gemeinsamen 16:9-Beamer-Theme aus `main`. Die [LaTeX-Quelle](slides.tex) ist direkt
editierbar. Die bisherigen Sprechernotizen auf [Deutsch](speaker_notes.md) und
[Englisch](speaker_notes_en.md) beziehen sich auf die frühere Fassung mit
13 Hauptfolien und zehn Minuten. Sie wurden beim Anhängen der neuen Backups
nicht verändert.

## Aufbau

| Folien | Inhalt | Fußzeile |
|---|---|---|
| 1–6 | Daten und Splits, CellCNN, Citrus-Einrichtung, SVM-Zellscore und -Spenderscore, gemeinsamer Methodenvergleich | SSBI · Task 4 |
| 7–9 | Je eine kurze Erklärung der quantitativen Auswahlregel für CellCNN, Citrus und SVM | SSBI · Task 5 |
| 10–12 | Wiederkehrende Subsets, ausgewählte Karten-Zellen und biologische Markerprofile | SSBI · Task 5 |
| 13 | Reserve: Hyperparameter/Paper-Abweichungen | SSBI · Task 4 |
| 14–15 | Reserve: gesamter SVM-Ablauf, von Training und Verlustfunktion bis Modellauswahl und Test | SSBI · Task 4 |
| 16 | Reserve: separater CellCNN–SVM-Vergleich über 100 Splits | SSBI · Task 4 |
| 17–19 | Reserve: ausführliche Auswahlregeln für CellCNN, Citrus und SVM | SSBI · Task 5 |
| 20–22 | Neue Backups zu Folie 10: Zellsubsets, Zentroiden, Gruppierung, Medoid und überlagerte Kartenpositionen | SSBI · Task 5 |
| 23 | Neuer Backup zu Folie 11: einzelne Karten-Zellen, CellCNN-Repräsentant und positive SVM-Auswahlhäufigkeit | SSBI · Task 5 |
| 24–25 | Neue Backups zu Folie 12: Datenbasis jeder Heatmap-Zeile, gleiche Spendergewichte und gemeinsame z-Skala | SSBI · Task 5 |

Die Fußzeilen enthalten zusätzlich die Seitenzahl. Die Variablenerklärung auf
Folie 4 ist entfernt; die Symbole werden auf Folie 9 erläutert. Die angepassten
Variablenbeschreibungen auf Folien 2, 5, 7–9 und im Backup verwenden fette Symbole;
auf Folien 7–9 stehen die Erklärungen untereinander. Die Schemata sind direkt in TikZ geschrieben.
Inhaltliche Grundlage sind die [Methodenerklärungen zu Aufgabe 4](../../results/AUFGABE_4_METHODENERKLAERUNGEN.md)
und [Aufgabe 5](../../results/AUFGABE_5_METHODENERKLAERUNGEN.md), abgeglichen mit
den bestehenden Notebooks und `src/task5_interpretation.py`.

Die sechs angehängten Backups verwenden englische Erläuterungen, Tabellen und
ein Rechenbeispiel ohne Botschaftskästen. Ihre Zahlen stammen aus den vorhandenen
`task5_paper_centroids.csv`, `task5_paper_groups.csv`, den Karten-Zelltabellen und
`data/svm_profiles_by_test_visit.csv`. Für die Überlagerung werden die eindeutigen
gespeicherten Kartenpositionen gezählt: CellCNN 111 dargestellte Zentroiden an
44 Positionen, Citrus 77 an 52 Positionen. Zellmittelwert, Gruppierung anhand
von Profilen und Auswahl eines vorhandenen Medoids werden getrennt erklärt.

Die Citrus-Einrichtungsfolie beschreibt die separate Umgebung aus
`environment-citrus.yml`: R 4.5, Bibliotheken, Compiler und `make`; anschließend
Originalpakete und R-Jupyter-Kernel gemäß [Projekt-README](../../README.md#separate-citrus-umgebung).
Citrus 0.8 und Rclusterpp sind auf die dort dokumentierten Git-Commits festgelegt.
Die gespeicherte Notebook-Ausgabe nennt R 4.5.3 und Rclusterpp 0.2.6. „Legacy“
bezieht sich auf den ursprünglichen Paketcode. Die Folie behauptet weder eine
alte R-Laufzeit noch zusätzliche Quellcode-Reparaturen.

## PDF erstellen

Aus dem Repository-Hauptordner:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -outdir=build praesentation/aufgaben_04_05/slides.tex
cp praesentation/aufgaben_04_05/build/slides.pdf praesentation/aufgaben_04_05/slides.pdf
```

Benötigt werden die vorhandene LaTeX-Installation mit Beamer, TikZ, Latin Modern,
Booktabs und englischem Babel. Der Build benötigt ausschließlich `slides.tex`,
`../theme.tex`, `figures/` und `data/*.tex`. Er benötigt weder Rohdaten noch Python,
R oder neue Modellläufe. Für Overleaf den Ordner `praesentation/` einschließlich
Theme hochladen und `aufgaben_04_05/slides.tex` als Hauptdatei setzen.

## Grafiken und Tabellen aktualisieren

Aus dem Repository-Hauptordner in der vorhandenen Python-Projektumgebung:

```bash
python -m src.presentation_assets
```

Der Exporter verwendet die vorhandenen Projektabhängigkeiten und Prüfhelfer.
Zusätzlich zu gespeicherten CSV-/JSON-Dateien liest er die ursprünglichen
`gated_alive`-FCS-Dateien für das ergänzende SVM-Markerprofil. Dabei werden gespeicherte
SVMs samt Scalern rekonstruiert, keine Modelle trainiert und keine Projektionen
oder Skalierungen neu gefittet. Die Aufgabe-5-Quellen, FCS-Prüfsummen, Testvorhersagen
und Karten-Auswahlzählungen werden kontrolliert; Metrikquantile werden gegen die
Splitmetriken geprüft. Alle Ausgaben liegen in diesem Präsentationsordner.
Die bestehenden Analyseergebnisse und Berichtsdateien bleiben erhalten.

| Material | Grundlage und Darstellung |
|---|---|
| `performance_30.pdf` und Tabelle | Drei Methoden, gemeinsame Splits 0–29. ROC-AUC-Boxplots, Einzelpunkte mit festem Jitterseed 45; Tabelle: Mediane von ROC-AUC, AP und Balanced Accuracy. |
| `centroids.pdf` | Aktuelle `task5_paper_centroids` und `task5_paper_groups`; Gruppen ab sechs unterschiedlichen Splits, gespeicherte Repräsentanten als Sterne. |
| `selected_cells.pdf` | Aktuelle `task5_paper_representative_cells` und `task5_paper_svm_cells`; explorative CellCNN-Auswahl und positive SVM-Auswahlhäufigkeit über Testmodelle des jeweiligen Spenders. |
| `marker_profiles.pdf` | Gespeicherte G1-Repräsentanten von CellCNN/Citrus und spendergleich gewichtete positive SVM-Testzellprofile; acht Marker, vorhandene explorative z-Skala und gemeinsame symmetrische Farbskala. |
| `performance_100.pdf` und Tabelle | Separater CellCNN–SVM-Vergleich über Splits 0–99, ausschließlich auf Reservefolie 16. |

`data/representative_profiles.csv` enthält die 24 dargestellten Markerwerte auf
ArcSinh- und z-Skala samt Profiltyp; einzelne Modell-/Subset-IDs gelten nur für die
CellCNN-/Citrus-Repräsentanten. `data/svm_profiles_by_test_visit.csv` enthält für
alle 180 Testauftritte die Auswahlzahlen und acht Marker-Mittelwerte.
`data/provenance.json` dokumentiert Eingabe- und Ausgabeprüfsummen, Exportcode,
SVM-Aggregationsregel und Abdeckung. `data/numbers.tex` übernimmt
die ausgewiesenen Wiederkehr- und Zellzahlen. Nach einer Ergebnisaktualisierung
müssen auch die manuell formulierten Aussagen und Sprechernotizen geprüft werden.

## Inhaltliche Abgrenzung

- Hauptvergleich und Interpretation verwenden `gated_alive` und dieselben 30 Splits.
  Historische `task5_*`-Dateien und die alten Detailberichtsassets werden nicht gelesen.
- Die Karte enthält 10.000 PBMCs, 500 je Spender, t-SNE-Perplexität 30. Sie unterscheidet
  sich von der 40.000-NK-Zellen-Karte in den Aufgabe-2-Folien auf `main`; übernommen
  wurde das Theme, nicht die dortige Datenbasis.
- CellCNN-/Citrus-Kartenpunkte sind Trainingssubset-Zentroiden, über die nächste
  Karten-Zelle projiziert. Die CellCNN-Zellauswahl eines Repräsentanten ist explorativ;
  die SVM-Häufigkeit verwendet ausschließlich die äußeren Testauftritte ihres Spenders.
- Auf Folie 12 stehen die gespeicherten Repräsentanten der häufigsten Gruppen:
  CellCNN Split 9/Filter 2 und Citrus Split 25/Cluster 139891. Ihre Auswahl wurde
  nicht anhand gewünschter Marker verändert. z-Werte sind keine Positivitätsgates.
- Die SVM-Zeile mittelt positiv ausgewählte Zellen vollständiger Testspender
  zunächst je Testauftritt, dann über nichtleere Auftritte je Spender, zuletzt
  gleichgewichtet über Spender. Alle 20 Spender tragen bei; 168/180 Auftritte
  sind nichtleer. Leere Auswahlen sind fehlende Profile, keine Nullwerte.
  Die 82.189 ausgewählten Zellvorkommen können dieselben Zellen mehrfach enthalten.
- Quellenzeilen auf den Folien enthalten ausschließlich passende Paperreferenzen;
  Projektergebnisse ohne passende Paperquelle erhalten keine Quellenzeile.
  Technische Herkunftsnachweise stehen hier, in den Methodenerklärungen und in der
  Provenienz. Für die Interpretation nötiger Kontext bleibt im Folieninhalt.
- Die Diagramme auf Folien 4 und 5 sind schematisch; die Beispielschwelle 1,80
  auf Folie 5 ist kein Ergebnis des Benchmarks. Auch die Zahlenbeispiele zur
  CellCNN-Auswahl auf Folie 7 und SVM-Auswahl auf Folie 19 illustrieren Regeln.
  Die SVM-Aggregation ist als Projektanpassung gekennzeichnet.
- CellCNN verwendet zur Interpretation eine strenge Halbmaximum-Schwelle auf
  der ursprünglichen Trainingsreferenz, zur Vorhersage dagegen Top-1-%-Pooling.
  Citrus wählt Cluster über wirksame Häufigkeitskoeffizienten aus; die Exporte
  erlauben keine exakte individuelle Zuordnung beliebiger Karten-Zellen.
  Die SVM wählt die höchsten 1 % des vollständigen Testspenders und für positive
  Auswahl zusätzlich Margins strikt oberhalb der gelernten Spenderschwelle.
- Die neue Notation verwendet `m` für Zellmargins, `s` für Spenderscores und `t`
  für zentrierte Interpretationsscores. Mit zusätzlichen Indizes bezeichnet
  `s` auch den äußeren Split; jede Folie erklärt ihre Indizes. Trainingslabels
  in der SVM-Verlustfunktion werden mit `y` bezeichnet.

Die PDF sowie die Quellen und ausgewählten Präsentationsassets können gemeinsam
versioniert werden. LaTeX-Zwischendateien unter `build/` werden ignoriert.

## Prüfung dieser Fassung

- LaTeX-Build mit 25 Seiten erfolgreich; abschließender Compilerlauf ohne
  Warnungen und ohne übervolle horizontale oder vertikale Boxen.
- Alle 25 Seiten gerendert; die sechs neuen Backups visuell auf Lesbarkeit,
  Formeln und Überläufe geprüft. Die bisherigen 19 Folien bleiben in der Quelle
  und im gerenderten PDF unverändert.
- Aufgabenbezogene Fußzeilen, Seitenzahlen und Verweise auf die aktuellen
  Hauptfolien 10–12 geprüft. Die neuen Backups enthalten keine Botschaftskästen.
- Auswahlregeln und Zahlen gegen Methodenerklärungen, vorhandenen Code und
  gespeicherte Ergebnistabellen geprüft. Bestehende Abbildungen, Datendateien,
  Provenienz und Sprechernotizen bleiben unverändert.
- Keine Modelle trainiert, keine Analyse oder Asset-Erzeugung erneut ausgeführt.

## Prüfstand der unveränderten Ergebnisassets

Die vorherige Fassung dokumentiert folgende Prüfungen. Sie wurden für diese reine
Folienüberarbeitung nicht erneut ausgeführt:

- Zusätzlicher Build in einem separaten Ordner ausschließlich aus LaTeX-Quelle,
  Theme und Präsentationsassets erfolgreich.
- 870 dargestellte beziehungsweise zusammengefasste Metrikwerte unabhängig aus den
  gespeicherten Spendervorhersagen nachvollzogen; Testspender gegen die Splitdatei geprüft.
- Alle 24 Heatmap-Werte gegen Originalzentroiden beziehungsweise die SVM-Testzellprofile
  und die gespeicherte Skalierung geprüft. Die 180 SVM-Scores, Schwellen und Top-Zellzahlen
  sowie die Karten-Auswahlzählungen und Häufigkeitsnenner stimmen mit dem bisherigen Export überein.
- Kleine deterministische Tests für Top-Auswahl, Gleichstände, strenge Schwelle,
  gleiche Spendergewichtung, ungleiche Zellzahlen/Testauftritte und leere Profile:
  `python -m unittest discover -s tests -p 'test_presentation_assets.py' -v`.
