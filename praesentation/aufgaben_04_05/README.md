# Historischer Einzelfoliensatz zu Aufgaben 4 und 5

- [Fertige Folien](slides.pdf) und [bearbeitbare LaTeX-Quelle](slides.tex).
- [Englische Sprechernotizen](speaker_notes_en.md), [deutsche Notizen](speaker_notes.md).
- [Ausführliche Erklärung von Aufgabe 4](AUFGABE_4_FEYNMAN_ERKLAERUNG.md).
- Tabellen unter `data/`, Abbildungen unter `figures/`.

Die fehlenden Quellen und Notizen wurden aus Commit `d516499` wiederhergestellt.
Sie beschreiben den damaligen Stand der Einzelpräsentation. Die spätere
Gesamtpräsentation ist `../main.tex` und enthält eigene Kopien der Folieninhalte.
Änderungen hier werden nicht automatisch in die Gesamtpräsentation übernommen.

Die aktuelle Reportauswertung aller drei Methoden mit vollständiger
Testspender-Trennung steht unter
[../report_assets/aufgaben_04_05](../report_assets/aufgaben_04_05/README.md).
Für neue Berichtsaussagen diese geprüften Ergebnisse verwenden.

Aus dem Projektstamm kompilieren:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=build praesentation/aufgaben_04_05/slides.tex
```

Zum Bearbeiten und Kompilieren genügen die geteilten Quellen und Assets.
Die FCS-Dateien und Modellcaches werden dafür nicht benötigt.
