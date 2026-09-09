# Aufgaben 4 und 5: Klassifikation und Interpretation

[slides.pdf](slides.pdf) enthält neun englische Hauptfolien und zwei Reservefolien
im gemeinsamen 16:9-Beamer-Theme aus `main`. Die [LaTeX-Quelle](slides.tex) ist direkt
editierbar. Die Sprechernotizen sind auf [Deutsch](speaker_notes.md) und
[Englisch](speaker_notes_en.md) verfügbar und auf acht Minuten ausgelegt;
die tatsächliche Dauer hängt vom Vortrag ab.

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
| `performance_100.pdf` und Tabelle | Separater CellCNN–SVM-Vergleich über Splits 0–99, ausschließlich auf Reservefolie 11. |

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
- Auf Folie 9 stehen die gespeicherten Repräsentanten der häufigsten Gruppen:
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
  auf Folie 5 ist kein Ergebnis des Benchmarks. Die SVM-Aggregation ist als
  Projektanpassung gekennzeichnet.

Die PDF sowie die Quellen und ausgewählten Präsentationsassets können gemeinsam
versioniert werden. LaTeX-Zwischendateien unter `build/` werden ignoriert.

## Prüfung dieser Fassung

- Normaler Build und zusätzlicher Build in einem separaten Ordner mit ausschließlich
  LaTeX-Quelle, Theme und Präsentationsassets erfolgreich; beide abschließenden
  Compilerläufe ohne LaTeX-Warnungen.
- Alle elf Seiten gerendert und visuell auf Lesbarkeit und Überläufe geprüft.
- 870 dargestellte beziehungsweise zusammengefasste Metrikwerte unabhängig aus den
  gespeicherten Spendervorhersagen nachvollzogen; Testspender gegen die Splitdatei geprüft.
- Alle 24 Heatmap-Werte gegen Originalzentroiden beziehungsweise die SVM-Testzellprofile
  und die gespeicherte Skalierung geprüft. Die 180 SVM-Scores, Schwellen und Top-Zellzahlen
  sowie die Karten-Auswahlzählungen und Häufigkeitsnenner stimmen mit dem bisherigen Export überein.
- Kleine deterministische Tests für Top-Auswahl, Gleichstände, strenge Schwelle,
  gleiche Spendergewichtung, ungleiche Zellzahlen/Testauftritte und leere Profile:
  `python -m unittest discover -s tests -p 'test_presentation_assets.py' -v`.
- Acht Minuten sind die geplante Dauer, keine gemessene Vortragszeit.
