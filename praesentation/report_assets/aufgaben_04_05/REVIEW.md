# Unabhängiger Review – Aufgaben 4 und 5

Abgeschlossen am 13.09.2026 durch den separaten Agenten
`/root/independent_review`, mit eigenständigem Kontext und ohne Änderungen an
den geprüften Dateien. Nach einem methodischen Vorabreview wurden die fertigen
Ergebnisse, Abbildungen und die drucknahe LaTeX-Vorschau erneut geprüft.

**Ergebnis: keine offenen blockierenden oder ergebnisverfälschenden Befunde.**
Der im Vorabreview gefundene fehlende Bezug der Mapping-Caches zu Paketversionen
wurde ergänzt; die betreffenden Mappings wurden anschließend neu berechnet.

Unabhängig nachgeprüft wurden:

- Verwendete Quellen-, Code-, Cache-, Export- und Abbildungsprüfsummen sowie
  Paketversionen.
- Klassifikationsmetriken aller 90 Methode/Split-Kombinationen, gepaarte
  Unterschiede und die in der README angegebenen Zahlen.
- 540 Testspender-Auftritte, 30.000 Kartenzeilen und die gemeinsamen
  spenderweisen Nenner von 4–16 Testauftritten.
- 90.000 CellCNN-Kartenbewertungen aus den gespeicherten Filtern; erneute
  Aggregation von 90.000 Citrus-Kartenbewertungen aus den Split-Caches;
  Übereinstimmung der SVM-Häufigkeiten mit dem geprüften Original.
- Alle 30 Citrus-Audits mit 9.984 nativen Kontrollentscheidungen. Zusätzlich
  wurde für jedes der 26 Modelle mit positiver Auswahl eine Nachbarsuche
  unabhängig über sämtliche 140.000 Trainingszellen berechnet: 26/26 korrekt.
- Alle 127.233.750 rohen Markerwerte aus den 20 FCS-Dateien, erneut mit FlowKit
  eingelesen und gegen die nativen Citrus/FlowCore-Prüfsummen geprüft.
- Sämtliche Spenderprofile, Markerzusammenfassungen, ergänzenden Gruppenprofile
  und die Behandlung leerer Selektionen. Alle drei Methoden haben
  20 beitragende Spender.
- 26 gezielte Tests; beide Hauptabbildungen, ihre Captions, Auswahlformeln,
  README und die A4-Vorschau bei 16 cm Textbreite.

Die beiden Ergebnisabbildungen samt Bildunterschriften passen auf eine A4-Seite.
Die Auswahlregeln und der übrige Berichtstext sind zusätzlich einzuplanen.

Die Aussagegrenzen sind dokumentiert: dieselben 20 Spender in wiederholten
Splits, methodenspezifische Auswahlregeln und explorative t-SNE-Koordinaten.
Der technische und methodische Review ist keine kausale oder unabhängige
biologische Validierung der ausgewählten Zellmengen.

Geprüfte Hauptartefakte:
[Report-Vorschau](report_layout_preview.pdf),
[Aufgabe 4](figures/task4_performance_paired.pdf),
[Aufgabe 5](figures/task5_report_combined.pdf),
[Datenprovenienz](data/provenance.json),
[Abbildungsprovenienz](figures/provenance.json).
