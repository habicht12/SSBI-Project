# Report-Abbildungen für Aufgaben 4 und 5

Für den fünfseitigen Bericht sind genau zwei Ergebnisabbildungen vorgesehen.
Die bestehenden Folien und Berichtsteile werden nicht automatisch geändert.
Alle Beschriftungen und LaTeX-Bildunterschriften sind auf Englisch.
Der [unabhängige Agentenreview](REVIEW.md) der fertigen Ergebnisse und
Abbildungen ist abgeschlossen; es bestehen keine offenen ergebnisverfälschenden Befunde.

## Empfohlene Auswahl

1. [Aufgabe 4: Leistung und gepaarte Unterschiede](figures/task4_performance_paired.pdf).
   Drei AUC-Verteilungen und die Unterschiede auf denselben 30 Splits.
2. [Aufgabe 5: drei Zellkarten und eine Marker-Heatmap](figures/task5_report_combined.pdf).
   CellCNN, SVM und Citrus auf derselben t-SNE-Karte, mit gleicher Definition
   der angezeigten Auswahlhäufigkeit und gleicher Profilaggregation.

Beide Hauptabbildungen zusammen: [report_figures.pdf](figures/report_figures.pdf).
Bei 16 cm Textbreite passen beide einschließlich Bildunterschriften auf
eine A4-Seite: [Layout-Vorschau](report_layout_preview.pdf).
Einbindung: [figures_for_report.tex](figures_for_report.tex).
Die für Aufgabe 5 erforderlichen quantitativen Auswahlregeln stehen zusätzlich
als kompakter englischer LaTeX-Baustein in [selection_rules.tex](selection_rules.tex).
Diese Dateien werden nicht selbstständig in einen bestehenden Report eingefügt.

Die ausführlichen [Gruppenprofile mit Wiederkehr](figures/task5_group_profiles_recurrence.pdf),
[Markerprofile mit Spenderstreuung](figures/task5_oof_marker_profiles.pdf) und
[separaten Zellkarten](figures/task5_oof_selection.pdf) bleiben Zusatzresultate.
Sie sind nicht zusätzlich für den knappen Hauptbericht vorgesehen.
[plot_overview.pdf](figures/plot_overview.pdf) enthält alle fünf Abbildungen.
Zu jedem Einzel-PDF gibt es eine PNG-Vorschau. Schriften und Zahlen bleiben
in den PDFs als Vektoren erhalten; die dichten Zellpunkte werden mit 400 dpi rasterisiert.

## Aufgabe 4

Die gespeicherten Klassifikatoren und ihre Vorhersagen bleiben unverändert.
ROC-AUC, Average Precision und Balanced Accuracy werden aus den Vorhersagen
neu berechnet und gegen den geprüften Benchmark abgeglichen.

| Methode | Mediane AUC | Mediane AP | Mediane BA |
| --- | ---: | ---: | ---: |
| CellCNN | 0,875 | 0,833 | 0,625 |
| SVM | 0,875 | 0,833 | 0,625 |
| Citrus | 0,625 | 0,542 | 0,500 |

| Gepaarter Vergleich | Mittlere AUC-Differenz | Besser / gleich / schlechter |
| --- | ---: | ---: |
| CellCNN − SVM | +0,0167 | 12 / 12 / 6 |
| CellCNN − Citrus | +0,2188 | 22 / 6 / 2 |
| SVM − Citrus | +0,2021 | 19 / 6 / 5 |

Boxen: Q1–Q3, Linie: Median, Punkte: Splits. Die 30 Aufteilungen verwenden
immer dieselben 20 Spender; sie sind keine unabhängigen Kohorten und die
Boxen keine Konfidenzintervalle. Je Split werden sechs Spender getestet,
zwei CMV-positive und vier CMV-negative. Weitere Kennzahlen stehen in
[data/metric_summary.csv](data/metric_summary.csv) und
[data/split_metrics.csv](data/split_metrics.csv).

## Aufgabe 5: vergleichbare Darstellung aller drei Methoden

| Auf derselben 10.000-Zell-Karte | CellCNN | SVM | Citrus |
| --- | ---: | ---: | ---: |
| Mindestens einmal ausgewählt | 863 | 93 | 1.583 |
| In mindestens 50 % der eigenen Testauftritte ausgewählt | 84 | 9 | 11 |
| In jedem eigenen Testauftritt ausgewählt | 15 | 0 | 0 |
| Nichtleere Vollspender-Auswahl | 180/180 | 168/180 | 156/180 |
| Spender mit Markerprofil | 20 | 20 | 20 |

Die NKG2C-Mittelwerte auf der gemeinsamen Referenz-z-Skala betragen
2,862 / 3,004 / 0,972 für CellCNN / SVM / Citrus. Dies charakterisiert die
jeweils ausgewählten Zellmengen; die Häufigkeiten allein bewerten nicht
deren biologische Qualität.

Die Auswertung konzentriert sich ausdrücklich auf **CMV-positiv unterstützende
Subsets**. Die einzelnen Auswahlregeln sind methodenspezifisch; ihre Ergebnisse
sind keine kausale Zerlegung der Vorhersage.

- **CellCNN:** Vereinigung aller Filter mit positivem Ausgangskontrast.
  Eine Zelle muss strikt mehr als die Hälfte des ursprünglichen
  Trainingsreferenzmaximums eines solchen Filters erreichen. Referenzziehung,
  Scaler und Seed bleiben unverändert; es wird kein Testmaximum bestimmt.
- **SVM:** Höchste `ceil(0.01 * N)` Margins des vollständigen Testspenders,
  anschließend `margin > saved_donor_threshold`. Grenzgleichstände werden
  durch die ursprüngliche FCS-Ereignisnummer aufgelöst.
- **Citrus:** Vereinigung der Mitgliedschaften aller gespeicherten Cluster
  mit `coefficient > 1e-10`. Neue Zellen erben die Cluster-Mitgliedschaften
  ihrer nächstgelegenen ursprünglichen Trainingszelle im unskalierten
  ArcSinh-Markerraum. Dies ist die originale Citrus-Zuordnungsregel,
  keine Zuordnung zum nächstgelegenen Clusterzentroiden.

Der ursprüngliche Citrus-Benchmark wertet 10.000 gezogene Testzellen je
Spender aus. Für diese zusätzliche Interpretation wird dieselbe feste
Zuordnungsregel auf alle Ereignisse des Testspenders angewandt. Die gespeicherten
Aufgabe-4-Vorhersagen werden dabei nicht durch neue Vollspender-Vorhersagen ersetzt.

Alle Methoden bewerten ausschließlich **äußere Testspender**. Jede Zelle zählt
je Split höchstens einmal, auch bei überlappenden Filtern oder Clustern.
Der Nenner ist die Zahl aller Testauftritte ihres Spenders in Splits 0–29,
hier 4–16. Leere Auswahlen bleiben im Nenner. Bei Citrus wählen die Splits
19, 24 und 26 keine wirksamen Cluster und Split 27 nur negative Cluster;
alle vier liefern daher eine leere positive Auswahl.

Die Farbe zeigt die Auswahlhäufigkeit von 0 bis 1, keine biologische
Klassenzugehörigkeitswahrscheinlichkeit. Eine größere oder kompaktere
markierte Zellmenge belegt keine bessere biologische Qualität, da sich
die Auswahlregeln unterscheiden.

Die Karte ist die bestehende **gated_alive**-Referenz: 10.000 Zellen,
500 je Spender, t-SNE-Perplexität 30. Sie ist nicht die aktuelle
Aufgabe-2-Karte mit 40.000 gated_NK-Zellen. Ihre Koordinaten und die gemeinsame
Marker-z-Skalierung dienen der explorativen Darstellung und gehen nicht
in die Klassifikatorvorhersagen oder Selektionsschwellen ein.

Die Heatmap verwendet die ausgewählten Zellen des **vollständigen
Testspenders**, nicht nur die auf der Karte sichtbare Stichprobe:
Mittelwert der Zellen pro nichtleerem Testauftritt, dann Mittelwert der
Auftritte pro Spender, dann gleicher Beitrag aller beitragenden Spender.
Eine leere Auswahl hat kein Markerprofil und wird nicht als Null-Expression
behandelt. Beide CMV-Klassen können beitragen. Spenderzahlen und nichtleere
Auftritte werden ausdrücklich exportiert. Alle 37 Marker stehen in den CSVs;
gezeigt werden dieselben acht Marker wie in der Präsentation.

Die ergänzende Gruppenanalyse ist weiterhin separat: Mittelwert der
Trainingszentroiden innerhalb einer Gruppe und eines Splits, danach
Median und Q1/Q3 über Splits mit dieser Gruppe. Die Gruppierung bleibt
unverändert (37 gemeinsame z-skalierte Marker; Kosinusdistanz;
Average-Linkage; Schnitt 0,4; Anzeige ab sechs Splits). Diese retrospektiven
Gruppenprofile werden nicht als Testzellprofile in die Hauptabbildung gemischt.

## Citrus-Rekonstruktion und Prüfungen

Die ursprünglichen finalen Trainingsbäume werden mit dem originalen R-Paket,
Seed und 10.000 Zellen pro Trainingsspender rekonstruiert. Gespeicherte
Cluster-IDs und Koeffizienten werden wiederverwendet; Klassifikatoren und
Hyperparameter werden nicht neu gewählt. Die Profile aller gespeicherten
Cluster eines rekonstruierten Baums müssen mit den bisherigen Zentroiden
übereinstimmen. Splits ohne positive Cluster benötigen keinen neuen Baum
und keine Nachbarsuche; dies ist in der Audit-Tabelle als Nullauswahl markiert.

Die Zellzuordnung wird exakt beschleunigt: Ein tatsächlich näherer negativer
Trainingspunkt kann Nichtauswahl beweisen, wenn seine Distanz kleiner als
die exakte Distanz zum nächsten positiven Punkt ist. Eine approximative
Suche liefert ausschließlich solche negativen Kandidaten. Alle nicht
bewiesenen Fälle werden im vollständigen Trainingsbaum exakt gesucht;
nahe Gleichstände werden mit sequentieller Distanzsummation nachgerechnet.
Die Approximation beeinflusst damit nur die Laufzeit, nicht die Auswahlregel.
„Positiv“ bedeutet hier Mitglied der positiven Clustervereinigung;
„negativ“ bezeichnet deren Komplement, nicht den CMV-Status des Trainingsspenders.

Für jeden nichtleeren positiven Split werden zusätzlich 384 echte Testkarten-
Zellen mit der nativen Citrus-Funktion ausgewertet. Nächste Trainingsindizes
und Auswahlentscheidungen müssen exakt übereinstimmen. Alle 127.233.750 rohen
Markerwerte der 3.438.750 Zellen in 20 FCS-Dateien werden außerdem vollständig
zwischen Citrus/FlowCore und Python/FlowKit anhand von Byte-Prüfsummen verglichen.
Im vollständigen Lauf stimmen alle **9.984 nativen Kontrollabfragen** exakt;
der größte Zentroidfehler beträgt **5,33 × 10⁻¹⁵**. Die 26 gezielten Tests bestehen.

Der vollständige Lauf exportiert 540 Testspender-Auftritte (180 je Methode)
und 30.000 Karten-Zellzeilen (10.000 je Methode), einschließlich Nullauswahlen.
Die SVM-Kartenhäufigkeiten und alle Klassifikationsmetriken werden erneut mit
den ursprünglichen Ergebnissen abgeglichen. Große Trainingsreferenzen und
Bäume bleiben im ignorierten lokalen Cache; seine Dateien und die Mappings
werden vor Wiederverwendung geprüft. Mapping-Caches binden zusätzlich die
Versionen von NumPy, SciPy, Pandas und FlowKit.

## Dateien und Reproduktion

- [provenance.json](data/provenance.json): Quellen, Definitionen, Dateiprüfsummen,
  Paketversionen, Zellzahlen und nichtleere Testauftritte.
- [citrus_mapping_audit.csv](data/citrus_mapping_audit.csv): native
  Zuordnungsprüfungen und Zentroidfehler je Split.
- [citrus_input_checks.csv](data/citrus_input_checks.csv): vollständiger Vergleich
  der beiden FCS-Leser.
- [cell_selection_frequencies.csv](data/cell_selection_frequencies.csv):
  Zähler, Nenner und Frequenz jeder Karten-Zelle je Methode.
- [profiles_by_test_visit.csv](data/profiles_by_test_visit.csv),
  [profiles_by_donor.csv](data/profiles_by_donor.csv),
  [oof_marker_profiles.csv](data/oof_marker_profiles.csv): alle Aggregationsstufen.
- [figures/provenance.json](figures/provenance.json): Eingabe- und Ausgabehashes
  der gerenderten Abbildungen.

Aus dem Projektstamm mit vorhandenen Umgebungen und Originaldaten:

```bash
OMP_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 OPENBLAS_NUM_THREADS=1 \
  /home/gregor/miniconda3/envs/ssbi-citrus/bin/Rscript \
  src/task5_citrus_reconstruct.R results/tables/benchmark_main_20260912 \
  results/tables/task5_citrus_oof_cache

/home/gregor/miniconda3/envs/ssbi-citrus/bin/Rscript \
  src/task5_citrus_input_check.R results/tables/benchmark_main_20260912 \
  results/tables/task5_citrus_oof_cache/native_input_checks.csv

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/ssbi-task45-mpl .venv/bin/python -m src.report_task45
```

Rscript ist alternativ über die aktivierte ssbi-citrus-Umgebung aufrufbar.
Die Rekonstruktion ist teuer und benötigt auf dieser Maschine Minuten pro
nichtleerem Split. Bereits passende Caches werden wiederverwendet. Optional
akzeptiert das R-Skript als drittes Argument eine kommagetrennte Splitliste.
`--source-root`, `--citrus-cache` und `--output` erlauben andere Arbeitsorte.

Nur die Abbildungen aus den geprüften kleinen CSVs neu erzeugen, ohne
Originaldaten, R oder Modellrekonstruktion:

```bash
MPLCONFIGDIR=/tmp/ssbi-task45-mpl .venv/bin/python -m src.report_task45 --render-only
```

Gezielte Tests:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m unittest tests.test_report_task45 \
  tests.test_task5_citrus_oof tests.test_task5_interpretation
```
