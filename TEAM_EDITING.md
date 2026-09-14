# Präsentation und Report gemeinsam bearbeiten

Alle unten verlinkten Quellen, PDFs, Tabellen und Plotdateien liegen auf `main`.
Nach `git pull --ff-only origin main` sind sie lokal verfügbar. Für die reine
Text-/Layoutbearbeitung sind weder ein Python-Training noch die FCS-Dateien nötig.

## Die richtigen Hauptdateien

| Dokument | Bearbeiten | Fertige PDF |
| --- | --- | --- |
| Aktueller Gesamtbericht, Version 2 | [report/main02.tex](report/main02.tex) | [main02.pdf](report/build/main02.pdf) |
| Aktuelles Supplement, Version 2 | [report/supplement02.tex](report/supplement02.tex) | [supplement02.pdf](report/build/supplement02.pdf) |
| Gesamtbericht und Supplement, Version 1 | [main.tex](report/main.tex), [supplement.tex](report/supplement.tex) | [main.pdf](report/build/main.pdf), [supplement.pdf](report/build/supplement.pdf) |
| Gemeinsame Gesamtpräsentation | [praesentation/main.tex](praesentation/main.tex) | [main.pdf](praesentation/main.pdf) |
| Maries ursprünglicher Berichtsteil | [report.tex](praesentation/report.tex) | [report.pdf](praesentation/report.pdf) |
| Korrekturfassung dieses Berichtsteils | [report_korrigiert.tex](praesentation/report_korrigiert.tex) | [report_korrigiert.pdf](praesentation/report_korrigiert.pdf) |

Der neue Bericht verwendet den geprüften aktuellen Ergebnisstand. Die gehaltene
Präsentation und die älteren Texte wurden erhalten. Insbesondere Aufgabe 2 und
die Zellinterpretation von Aufgabe 5 haben dort teilweise einen älteren Stand.
Für neue Aussagen die Ergebnisse unter `report/` und
`praesentation/report_assets/` verwenden, wie in der Haupt-README beschrieben.

## Text, Abbildungen und Literatur ändern

Im Report stehen Text und Bildunterschriften direkt in der jeweiligen `.tex`-
Hauptdatei. Gemeinsames Layout: [style.tex](report/style.tex). Gemeinsame Literatur:
[references.tex](report/references.tex). Die aktuellen Abbildungen 1/3/4 liegen
unter [report/figures/v02](report/figures/v02), die Clustering-Abbildung unter
[report/figures/clustering.pdf](report/figures/clustering.pdf). Supplement-Tabellen
liegen in [report/tables](report/tables), weitere Abbildungen in
[report/figures/supplement](report/figures/supplement).

Die Gesamtpräsentation enthält eigene Kopien der Folientexte. Deshalb direkt
`praesentation/main.tex` bearbeiten, wenn sich die gemeinsame Präsentation ändern
soll. Änderungen an Einzelfoliensätzen erscheinen dort nicht automatisch.

Einzelfoliensätze und Notizen:

- [Aufgabe 2](praesentation/aufgabe_02/README.md), einschließlich Sprechernotizen.
- [Aufgabe 3](praesentation/aufgabe_03_clustering.tex), mit
  [Folieninhalt](praesentation/slides/aufgabe_03.tex) und
  [Sprechernotizen](praesentation/sprechernotizen_aufgabe_03.md).
- [Aufgaben 4/5](praesentation/aufgaben_04_05/README.md), einschließlich deutscher
  und englischer Sprechernotizen sowie ausführlicher Methodenerklärung.
- [Aufgabe 6](praesentation/aufgabe_06_vergleich/README.md).
- [Ältere Folien im Ordner slides](slides/main.tex) bleiben als separate Fassung
  erhalten. Dieser Ordner benötigt auch die geteilten Assets aus dem Projektstamm.

[Aufgabenblatt](Group_projects_ssbi_2026.pdf) und [CellCNN-Paper](CellCNN.pdf)
sind ebenfalls enthalten. Die Quellen-PDFs bleiben unverändert.

## PDFs kompilieren

Benötigt werden TeX Live/MiKTeX oder Overleaf, pdfLaTeX und lokal `latexmk`.
Es werden übliche Pakete wie `beamer`, `pgf`, `lmodern`, `microtype`, `natbib`,
`titlesec`, `caption`, `booktabs` und `hyperref` verwendet. Die lokale optionale
TeX-Suche im Report-Bauskript ersetzt keine vollständige TeX-Installation.

Aus dem Projektstamm:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -outdir=build report/main02.tex
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -outdir=build report/supplement02.tex
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -outdir=build praesentation/main.tex
```

Unter Bash geht für die beiden Reports auch `bash report/build.sh main02 supplement02`.
In Overleaf den vollständigen Ordner `report/` bzw. `praesentation/` importieren
und die gewünschte Hauptdatei wählen. Für den neuen Gesamtbericht ist kein BibTeX
nötig. Maries Berichtsteil verwendet dagegen die mitgelieferte `references.bib`.

Report-Ausgaben entstehen unter `report/build/`. Die neu kompilierte gemeinsame
Präsentation entsteht unter `praesentation/build/main.pdf`. Wenn sie geteilt werden
soll, die geprüfte Ausgabe nach `praesentation/main.pdf` kopieren und zusammen
mit den Quellen committen. Hilfsdateien wie `.aux` und `.log` bleiben lokal.

## Plots aus den vorhandenen Ergebnissen neu zeichnen

Die Python-Umgebung ist in [environment.yml](environment.yml) beschrieben.
Die Report-Exporter benötigen unter anderem NumPy, pandas, PyArrow, SciPy und
Matplotlib. Die numerischen Eingangsdaten sind auf `main` enthalten, auch die
gezielt eingefrorenen Dateien unter den datierten `results/tables/`-Pfaden.

```bash
python report/export_assets02.py
python report/build_tables.py
```

Für die ursprünglichen Report-Abbildungen einschließlich Clustering:
`python report/export_assets.py`. Für die zusätzlichen Aufgabe-4/5-Abbildungen:
`python -m src.report_task45 --render-only`. Für die Aufgabe-6-Abbildungen:
`python -m src.export_task6`. Diese Exporte nutzen gespeicherte Ergebnisse und
starten kein Modelltraining. Sie schreiben neue Abbildungsdateien und
gegebenenfalls deren Prüfnachweise; danach die betroffenen PDFs neu kompilieren.

Die Koordinaten des aktuellen 40.000-Zell-Reports liegen unter
`results/tables/report_preparation_20260913/embeddings_full.parquet`.
Die gleichnamige Datei im Projektstamm gehört zu einem älteren Stand und darf
nicht ohne Ergebnisprüfung als Ersatz verwendet werden.

Eine vollständige Neuberechnung der Analysen ist ein anderer Arbeitsschritt:
Dafür benötigt man weiterhin das bereitgestellte FCS-Archiv, passende
Modellartefakte oder neues Training. Die großen Trainings-/Citrus-Caches und
temporären Laufdateien werden zum Bearbeiten der Dokumente nicht benötigt.

## Änderungen prüfen und teilen

Der Hauptbericht soll fünf Seiten bei 11-Punkt-Haupttext behalten. Der Bonus
bleibt einschließlich Tabelle unter einer halben Seite. Bei Änderungen an
Skalen, Zahlen oder Auswahlregeln auch die zugehörigen Aussagen prüfen.

`python report/check_report02.py` prüft den eingefrorenen V2-Stand einschließlich
Prüfsummen und benötigt PyMuPDF. Nach absichtlichen Änderungen kann eine
Prüfsumme erwartungsgemäß abweichen. Das ist ein Hinweis, die Änderung zu prüfen
und ihren Nachweis zu aktualisieren, kein Grund, die Prüfung einfach zu entfernen.
Die bestehenden Reviews beziehen sich auf die jeweils geprüfte Fassung.

Die Vollständigkeit wurde am 14.09.2026 in einer separaten Kopie ausschließlich
der versionierten Dateien geprüft: zwölf LaTeX-Einstiege und fünf
Plot-/Tabellenexporte liefen erfolgreich, ohne lokale FCS-Dateien oder
Modellcaches. Der Text aller vier neu kompilierten Gesamtberichte/Supplemente
stimmt mit den geteilten PDFs überein. Die Zahlen, Dateien und Wiederherstellungen
sind in [TEAM_SHARING_CHECK.json](TEAM_SHARING_CHECK.json) dokumentiert.

Zum Teilen die geänderten Quellen, Abbildungen und die geprüften fertigen PDFs
gemeinsam committen und pushen. Eine kurze Beschreibung der geänderten Abschnitte
hilft den anderen beim Abgleich.
