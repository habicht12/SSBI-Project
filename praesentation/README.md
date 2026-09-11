# Gemeinsame SSBI-Präsentation

[main.tex](main.tex) enthält die gemeinsame Präsentation auf `GrHa`.
Die fertige, mitversionierte Fassung liegt unter [main.pdf](main.pdf).

Aus dem Repository-Hauptordner kompilieren:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -outdir=build praesentation/main.tex
```

Der Build erzeugt `praesentation/build/main.pdf`. Nach einer Prüfung diese
Datei zum Teilen als `praesentation/main.pdf` übernehmen; der Build-Ordner
bleibt von Git ausgeschlossen.

Benötigt werden pdfLaTeX/latexmk und die Pakete des vorhandenen Beamer-Themes.
Alle benötigten Abbildungen und Tabellen liegen im Präsentationsordner;
Analysen oder Datenexporte müssen dafür nicht erneut ausgeführt werden.
Für Overleaf den gesamten Ordner `praesentation/` übernehmen und `main.tex`
als Hauptdokument wählen.

## Reihenfolge

Die PDF hat **77 Seiten**. Die Titelseite ist unnummeriert; anschließend
laufen die eingeblendeten Foliennummern von 1 bis 76 durch. Die Seitenangaben
in dieser Tabelle beziehen sich auf die PDF einschließlich Titelseite.

| Abschnitt | PDF-Seiten |
| --- | --- |
| Group-Project SSBI: Sebastian Fay, Marie Schygulla, Gregor Habitzreither | 1 |
| Aufgabe 2 | 2–8 |
| Aufgabe 3 | 9–21 |
| Aufgabe 4 | 22–27 |
| Aufgabe 5 | 28–33 |
| Aufgabe 6 | 34–38 |
| References | 39 |
| Backups zu Aufgabe 2 | 40–52 |
| Backups zu Aufgabe 3 | 53–58 |
| Backups zu Aufgabe 4 | 59–62 |
| Backups zu Aufgabe 5 | 63–71 |
| Backups zu Aufgabe 6 | 72–77 |

Die beiden ursprünglich mit „SKIP“ markierten Leiden-Folien eröffnen den
Backupblock von Aufgabe 3. Beide vorhandenen UMAP- und Procrustes-Varianten
von Aufgabe 2 bleiben erhalten. Die früheren Quellenfolien sind in der
gemeinsamen References-Folie zusammengefasst; die Angaben zum Clustering-Lauf
stehen zusätzlich auf der Backupfolie zur Stabilitätsschwelle.

## Herkunft und Bearbeitung

- Aufgabe 2 und 6: `main`, Commit `0829915`, aus `aufgabe_02/slides.tex`
  beziehungsweise `aufgabe_06_vergleich/slides.tex`; zugehörige benötigte
  Assets und Provenienzdateien wurden unverändert übernommen.
- Aufgabe 3: lokale `aufgabe_03_final.tex`; Abbildungen und Ergebnistabelle
  aus dem bereits vorhandenen `ssbi_project`-Ordner des separaten
  `Group-Project-slides`-Worktrees.
- Aufgabe 4/5: lokaler, uncommittierter Stand von `aufgaben_04_05/slides.tex`
  bei der Zusammenführung am 11.09.2026.

`main.tex` enthält eine eigenständige Kopie der Folieninhalte. Änderungen an
Einzelquellen werden deshalb nicht automatisch in die gemeinsame Fassung
übernommen. Für Anpassungen an der Gesamtpräsentation direkt `main.tex`
bearbeiten. Das Theme und die ursprünglichen Einzelquellen bleiben unverändert.

Die Zusammenführung vereinheitlicht Pfade, Fußzeilen und Folienverweise.
Auf fünf Folien wurden Größen oder Abstände angepasst, damit der bestehende
Inhalt vollständig auf die Seite passt. Die englischen Inhalte und
Ergebnisse wurden beibehalten; es wurden keine Analysen neu ausgeführt.

## Prüfung

Die PDF wurde mit 77 Seiten kompiliert. Alle Seiten wurden als gerenderte
Übersicht, Titel/References und die Layoutkorrekturen zusätzlich vergrößert
kontrolliert. Geprüft wurden außerdem die aus den Quellen abgeleitete
Folienreihenfolge, Nummerierung, interne Verweise, PDF-Lesezeichen und
Metadaten. Der abschließende LaTeX-Lauf enthält keine Warnungen oder
über-/untervollen Boxen.
