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

Die PDF hat **98 Seiten**. Die Titelseite ist unnummeriert; anschließend
laufen die eingeblendeten Foliennummern von 1 bis 97 durch. Die Seitenangaben
in dieser Tabelle beziehen sich auf die PDF einschließlich Titelseite.

| Abschnitt | PDF-Seiten |
| --- | --- |
| Group-Project SSBI: Sebastian Fay, Marie Schygulla, Gregor Habitzreither | 1 |
| Aufgabe 2 | 2–8 |
| Aufgabe 3 | 9–21 |
| Aufgabe 4 | 22–27 |
| Aufgabe 5 | 28–33 |
| Aufgabe 6: Populationsbewertung und Filterauswahl | 34–35 |
| References | 36 |
| Aufgabe 6: Filtergeometrie und lernbares Pooling | 37–49 |
| Aufgabe 6: Vergleich der Modelle | 50–54 |
| Backups zu Aufgabe 2 | 55–67 |
| Backups zu Aufgabe 3 | 68–73 |
| Backups zu Aufgabe 4 | 74–77 |
| Backups zu Aufgabe 5 | 78–86 |
| Backups zu Aufgabe 6 | 87–98 |

Direkt nach der eingeblendeten Folie 32 stehen „How should we score the
learned populations?“ und „Select a predictive filter, then count responding
cells“ als Folien 33/34. References ist damit Folie 35. Die übrigen
Task-6-Erklärfolien und die bisherigen fünf Vergleichsfolien folgen dahinter;
die Backups bleiben nach Aufgaben sortiert am Ende.

Die beiden ursprünglich mit „SKIP“ markierten Leiden-Folien eröffnen den
Backupblock von Aufgabe 3. Beide vorhandenen UMAP- und Procrustes-Varianten
von Aufgabe 2 bleiben erhalten. Die „SKIP“-Beispielfolie zu hartem und weichem
Pooling eröffnet den Backupblock von Aufgabe 6. Die früheren Quellenfolien sind in der
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
- Ergänzende Task-6-Erklärfolien: `ssbi_project/slides/aufgabe_06.tex` aus
  dem separaten `Group-Project-slides`-Worktree, englische Fassung vom
  10.09.2026. Alle 21 Folien wurden übernommen; die Einzelquelle bleibt
  unverändert.

`main.tex` enthält eine eigenständige Kopie der Folieninhalte. Änderungen an
Einzelquellen werden deshalb nicht automatisch in die gemeinsame Fassung
übernommen. Für Anpassungen an der Gesamtpräsentation direkt `main.tex`
bearbeiten. Das Theme und die ursprünglichen Einzelquellen bleiben unverändert.

Die Zusammenführung vereinheitlicht Pfade, Fußzeilen und Folienverweise.
Auf fünf Folien wurden Größen oder Abstände angepasst, damit der bestehende
Inhalt vollständig auf die Seite passt. In einer zusätzlichen Task-6-Tabelle
wurde der Text linksbündig gesetzt. Die englischen Inhalte und
Ergebnisse wurden beibehalten; es wurden keine Analysen neu ausgeführt.

## Prüfung

Die ursprünglichen 77 Seiten wurden als gerenderte Übersicht,
Titel/References und Layoutkorrekturen zusätzlich vergrößert kontrolliert.
Nach der Task-6-Ergänzung wurde die PDF mit 98 Seiten kompiliert; die neuen
Folien und Übergänge wurden erneut gerendert und geprüft. Außerdem wurden
der Erhalt aller bisherigen Folien, die neue Reihenfolge, Nummerierung,
internen Verweise, PDF-Lesezeichen und Metadaten geprüft. Der abschließende
LaTeX-Lauf enthält keine Warnungen oder über-/untervollen Boxen.
