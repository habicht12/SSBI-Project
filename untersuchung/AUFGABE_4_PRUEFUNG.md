# Prüfung von Aufgabe 4 am 5. September 2026

**Stand nach Umsetzung:** Die nachfolgende Prüfung beschreibt den Ausgangsstand.
Inzwischen prüfen alle Methoden ihre Zwischenstände gegen gespeicherte
Konfigurationen. Neue SVM-/CellCNN-Läufe exportieren die vollständigen
Modellparameter. Die sieben numerischen Citrus-Nullprofile wurden aus den
historischen Full-Artefakten entfernt (288 statt 295 Profile), ohne die
Spendervorhersagen zu ändern. Average Precision und die Batch-Labeltabelle sind
im Vergleich ergänzt. Anschließend wurden auf ausdrücklichen Wunsch alle drei Methoden auf
`gated_alive` über dieselben 100 Splits vollständig neu berechnet. Jetzt liegen
für jeden Split die vollständigen Artefakte und Konfigurationsnachweise vor.
SVM und CellCNN reproduzieren die alten Scores exakt; bei Citrus beträgt die
maximale Abweichung etwa `1e-15`. Alle 1.200 Metrikwerte des gemeinsamen
Vergleichs stimmen überein. Die Modellrekonstruktion aus den neuen SVM- und
CellCNN-Parametern wurde zusätzlich für die festen Full-Splits 0, 49 und 99
geprüft. Die Eingangsdaten und Splitzuordnungen sind unverändert. Details zur
Benutzung stehen in der README.

Die aktuelle Klassifikation enthält in den geprüften Rechenwegen keinen
nachgewiesenen Fehler bei FCS-Zuordnung, Labelrichtung, spenderweiser Trennung,
ROC-AUC oder Balanced Accuracy. Die auffälligen SVM- und Citrus-Ergebnisse
lassen sich gezielt reproduzieren und erklären. Es gibt aber eine bestätigte
Lücke bei der Wiederverwendung vorhandener SVM-Ergebnisse, numerisch
irreführende Citrus-Clusterzahlen und Grenzen der Interpretation des Batch-
und PR-AUC-Vergleichs.

Die bestehenden Notebooks und Benchmark-CSV-Dateien wurden bei dieser Prüfung
nicht verändert. Es wurde keine vollständige wiederholte CV neu ausgeführt.

## 1. Daten, Splits und Ergebniszuordnung

- Die Original-Labeldatei stimmt mit allen 2.000 Zeilen der Splitdatei überein.
- Jeder der 100 Splits enthält 14 Trainingsspender (7 CMV−, 7 CMV+) und
  6 Testspender (4 CMV−, 2 CMV+), ohne Überschneidung.
- Jede Methode enthält genau 600 vollständige Testvorhersagen. Spender, Labels
  und Split-Seeds stimmen eins zu eins mit der zentralen Splitdatei überein.
- Der Hauptvergleich verwendet `gated_alive`. `gated_NK` wird in den
  getrennten Smoke-Ergebnissen verwendet. Der Suffix `_NK` in der gemeinsamen
  Labeldatei führt nicht dazu, dass der Full-Modus NK-Dateien lädt: Die Zuordnung
  erfolgt über die Spender-ID und den zum Modus passenden FCS-Pfad.
- Für alle 300 Methoden-/Split-Kombinationen wurden ROC-AUC zusätzlich direkt
  aus den acht positiven-negativen Spenderpaaren und Balanced Accuracy aus den
  beiden klassenspezifischen Trefferquoten berechnet. Sie stimmen mit den
  gespeicherten Werten überein. Die binären Vorhersagen entsprechen überall
  `score >= decision_threshold`.

**Unabhängiger FCS-Abgleich:** Die von Citrus tatsächlich gezogenen Ereignisse
wurden anhand ihrer ursprünglichen Ereignisnummern mit FlowKit und einem direkten
Read-only-Zugriff auf die binären Big-Endian-Float32-Daten verglichen. Geprüft
wurden alle 37 Marker und alle 20 Dateien pro Gate: 148.000 Markerwerte aus zwei
Smoke-Splits und 2.220.000 Markerwerte aus drei Full-Splits, einschließlich
Mehrfachvorkommen eines Ereignisses in unterschiedlichen Splits. Die Rohwerte
von FlowKit und dem direkten Leser sind identisch. R-ArcSinh und unabhängig
berechnetes Float64-ArcSinh unterscheiden sich maximal um `5,33e-15`; zur
Float32-Python-Pipeline beträgt die größte Differenz `3,98e-7`. Negative
Hintergrundwerte bleiben erhalten. Für diese geprüften Werte trat keine
Begrenzung an der Kanalobergrenze auf.

Damit gibt es in den überprüften Daten keinen Hinweis auf eine falsche
Gate-Stufe, vertauschte Marker, doppelte ArcSinh-Transformation, falsche
Byte-Offsets oder eine relevante Abweichung zwischen R und Python.

## 2. Warum Citrus in 15 Splits konstant vorhersagt

Die vollständigen Vorhersagen enthalten 15 Splits mit sechsmal `score = 0.5`:

`2, 14, 16, 19, 24, 27, 38, 53, 55, 59, 71, 75, 86, 96, 97`.

- In neun davon sind keine Clusterkoeffizienten ausgewählt.
- In sechs weiteren meldet Citrus formal einen Nichtnull-Koeffizienten,
  dessen Betrag nur zwischen `1,79e-15` und `9,41e-15` liegt. Das ist numerisch
  ein Nullmodell.
- Bei sieben positiven und sieben negativen Trainingsspendern sagt ein
  logistisches Modell ohne wirksame Merkmale eine Wahrscheinlichkeit von 0,5
  vorher. ROC-AUC und Balanced Accuracy sind dann beide 0,5.

**Gezielte Wiederholung:** Zuerst wurden die beiden Smoke-Splits 0 und 1,
anschließend die Full-Splits 2, 14 und 0 mit der unveränderten Notebook-Funktion
neu berechnet. Letztere decken ein exaktes Nullmodell, ein numerisches Nullmodell
und ein nichtkonstantes Modell ab. Alle drei Full-Splits reproduzieren die
gespeicherten Scores ohne Differenz; die größte Smoke-Differenz liegt bei
`1,40e-16` durch die CSV-Darstellung.

Zusätzlich wurden die trainierten Zwischenobjekte geprüft:

- Zeilennamen der Trainings-, Validierungs- und Testfeatures entsprechen den
  jeweiligen FCS-Dateien in derselben Reihenfolge wie die Labels.
- Die innere Clusterbildung umfasst nur 9, 9 beziehungsweise 10 Trainingsspender;
  die äußere Clusterbildung umfasst die 14 Trainingsspender. Der Mindestumfang
  von 5 % entspricht bei 14.000 Zellen tatsächlich 700 Zellen.
- Trainings- und Testfeatures haben dieselbe Spaltenreihenfolge.
- Die positive glmnet-Klasse ist `CMV+`.
- Die Vorhersagen stimmen mit einer direkten Berechnung von
  `sigmoid(Intercept + Clusterhäufigkeiten @ Koeffizienten)` überein.
- Die CV-Fehlerraten wurden aus den inneren Spendervorhersagen unabhängig
  rekonstruiert und stimmen überein.

Die originale Funktion `citrus.getCVMinima` nimmt bei gleicher minimaler
Klassifikationsfehlerrate den ersten Eintrag des absteigenden Lambda-Pfades,
also die stärkste Regularisierung unter den gleich guten Kandidaten. In
Full-Split 2 gibt es 83 gleich gute Lambda-Werte; ausgewählt wird der erste,
mit CV-Fehler `8/14`. In Split 14 gibt es drei gleich gute Werte bei `7/14`.
Die sparsamen bzw. konstanten Modelle sind damit ein Ergebnis der offiziellen
Modellauswahl und kein versehentliches Überspringen des Trainings.

**Rundung:** In Split 14 beträgt ein ungerundeter positiver Score
`0.50000000000000022`, während die anderen fünf bei 0,5 liegen. Ohne Behandlung
solcher Rundungsreste entstünde allein dadurch eine scheinbare ROC-AUC von
0,75. Die vorhandene Rundung auf 15 Nachkommastellen entfernt hier numerisches
Rauschen; sie vernichtet kein nachgewiesenes biologisches Signal. Bei den 85
nichtkonstanten Splits beträgt die kleinste gespeicherte Score-Spannweite
etwa 0,0144 und liegt damit weit von dieser Größenordnung entfernt.

**Tatsächliche Unstimmigkeit:** Die exportierte `selected_cluster_count` zählt
formal nichtnull Werte ohne Toleranz. Insgesamt sind sieben von 295 exportierten
Cluster-/Split-Kombinationen kleiner als `1e-10`, darunter eine zusätzliche in
Split 54 mit einem ansonsten nichtkonstanten Modell. Der kleinste Betrag der
übrigen Koeffizienten ist etwa 0,00742. Für Aufgabe 5 müssen Clusterzahlen und
Profile deshalb mit derselben numerischen Nulltoleranz interpretiert werden.
Die aktuellen Klassifikationsmetriken sind dadurch nicht falsch.

## 3. Warum alle drei SVM-C-Werte dieselbe AUC liefern

In allen 300 vollständigen inneren Folds und allen sechs Smoke-Folds sind die
AUC-Werte für `C = 0.01, 0.1, 1.0` identisch. Die Auswahl nimmt bei Gleichstand
das kleinste C; deshalb wird in allen 100 vollständigen Splits `C = 0.01`
ausgewählt.

Das ist kein Beleg für identische Modelle: Bei einer Neuberechnung ändern sich
die Gewichte und die kontinuierlichen Scores, während die Spenderrangfolge
gleich bleibt. Beispiel Full-Split 0, innerer Fold 0:

| C | Norm des Gewichtsvektors | Score für a_001 | Score für a_003 | Validierungs-AUC |
|---|---:|---:|---:|---:|
| 0,01 | 0,438320 | 0,755603 | 1,250753 | 0,5 |
| 0,1 | 0,438953 | 0,756363 | 1,251937 | 0,5 |
| 1,0 | 0,439017 | 0,756439 | 1,252056 | 0,5 |

ROC-AUC hängt von den positiven-negativen Rangvergleichen ab, nicht vom
numerischen Abstand der Scores. In einem inneren Fold mit vier bzw. fünf
Spendern gibt es nur vier bzw. sechs solche Paare. Die Metrik ist entsprechend
grob.

Außerdem verwendet LinearSVC eine Summe der Zellverluste. Bei 90.000 bzw.
100.000 inneren Trainingszellen ist die relative Regularisierung in diesem
C-Bereich bereits schwach. Das passt zu den nur geringfügig unterschiedlichen
Gewichten. Der gültige Regularisierungsparameter wird bei jedem Fit tatsächlich
übergeben; es wird jeweils eine neue SVM erzeugt. Siehe auch die
[LinearSVC-Dokumentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).

**Ausgeführte Tests:** Smoke-Split 0/Fold 0; Full-Split 0/Folds 0–2 sowie
Full-Splits 1 und 14/Fold 0, jeweils alle drei C-Werte. Alle 18 Fits reproduzieren
die gespeicherte AUC und ändern über C die Gewichte, ohne die Spenderrangfolge zu
ändern. Der vollständige Auswahl-/Refit-/Testweg wurde zusätzlich für Split 0
im Smoke- und Full-Modus wiederholt; Test-Scores und gelernte Schwellenwerte
stimmen mit den gespeicherten Ergebnissen innerhalb `1e-10` überein.

Eine strengere Optimierungstoleranz von `1e-6` bzw. `1e-8` in Full-Split 0/Fold 0
ändert die AUC nicht; die maximale Scoreänderung gegenüber dem Standardfit bei
C=0,01 beträgt nur `5,69e-6`. Auch ein rein diagnostischer Test von
`C = 1e-7, 1e-6, 1e-5, 1e-4, 1e-3` in diesem inneren Fold verändert die
Gewichtsnormen stark, aber nicht die AUC. Dabei wurde kein äußeres Testergebnis
zur Wahl eines neuen C-Bereichs verwendet.

Die bisherige Suche ist korrekt ausgeführt, unterscheidet ihre drei Kandidaten
aber empirisch nicht. Sie beweist nicht, dass C=0,01 global optimal ist.

**Weitere Score-Auffälligkeit:** Alle 600 vollständigen SVM-Spenderscores sind
positiv. Das ist bei einem Mittelwert der obersten 1 % der Zellscores plausibel
und keine Wahrscheinlichkeit. Die gelernte Spenderschwelle ist deshalb wichtig;
ihre Werte liegen zwischen etwa 0,676 und 1,664. Eine nachträglich eingesetzte
Schwelle von null wäre hier falsch. Der Code verwendet die gelernte Schwelle.
Dass rohe Out-of-fold-Scores aus verschiedenen Modellen zu einer Schwelle
zusammengeführt und anschließend auf einen neu gefitteten Klassifikator
übertragen werden, bleibt eine Kalibrierungsannahme. Sie ist kein nachgewiesener
Rechenfehler und beeinflusst die schwellenunabhängige ROC-AUC nicht.

## 4. Batch-Check: korrekt gerechnet, begrenzt aussagekräftig

Alle 200 gespeicherten AUC-Werte wurden mit einem unabhängigen Leser der
FCS-Metadaten rekonstruiert. Messtag und Instrument stimmen außerdem zwischen
den Alive- und NK-Dateien desselben Spenders überein. Es liegt keine Vertauschung
von CMV− und CMV+ vor.

Die tatsächlichen Labelzahlen sind:

| Messtag | CMV− | CMV+ |
|---|---:|---:|
| 01.05.2012 | 3 | 3 |
| 08.05.2012 | 3 | 2 |
| 12.06.2012 | 5 | 4 |

Die Instrumentgruppen enthalten entsprechend 6/5 und 5/4 CMV−/CMV+-Spender.
Die Gruppen haben also fast gleiche CMV-Anteile. Wenn beim zufälligen Split
überproportional viele positive Spender eines Batches im Training landen,
bleiben dort eher negative Spender für den Test übrig. Der Trainingsscore
ordnet diesen Batch dann hoch ein, obwohl sein Testanteil niedriger ausfällt.

Zur Prüfung dieses Mechanismus wurden alle
`choose(9,2) * choose(11,4) = 11.880` möglichen äußeren Testmengen auf genau
dieser Kohorte ausgewertet, ohne Zellmodelle zu trainieren:

| Kontrollvariable | Mittlere AUC über alle Testmengen | Mediane AUC |
|---|---:|---:|
| Messtag | 0,2960 | 0,250 |
| Instrument | 0,3636 | 0,375 |

Damit reproduzieren sich die niedrigen Mediane auch über sämtliche möglichen
Splits. Sie entstehen aus der konkreten Zusammensetzung dieser kleinen Kohorte
und dem Training/Test-Komplement, nicht aus einem Vorzeichenfehler.

Ein zusätzlicher Kontrollversuch mit 500 zufälligen Zuordnungen der neun
positiven Labels und jeweils 100 Splitziehungen (Seed 20260905) ergab über diese
zufälligen Kohorten mittlere AUCs von 0,4963 bzw. 0,4930. Die niedrige AUC ist
also kein allgemeiner Zwang des Codes. Bedingt auf die vorliegende, nahezu
labelbalancierte Batchzusammensetzung ist sie jedoch nachvollziehbar.

Ein Fisher-Test für die Instrumenttabelle und ein exakt enumerierter
konditionaler Pearson-Test für die Messtagstabelle ergeben jeweils p=1.
Das bestätigt lediglich, dass diese 20 Spender keinen auffälligen direkten
Batch-Label-Zusammenhang zeigen. Es beweist weder das Fehlen technischer
Expressionsverschiebungen noch die Übertragbarkeit auf einen neuen Messtag.
Die Instrumentversion ist zudem an den Messtag gekoppelt; beide Kontrollen
sind keine unabhängigen Bestätigungen.

**Angemessene Formulierung:** „CMV-Gruppen sind über die erfassten Messtage und
Instrumentversionen annähernd gleich verteilt. Ein ausschließlich aus diesen
Metadaten gebildeter Kontrollscore liefert im vorliegenden Splitprotokoll
keine positive Vorhersageleistung. Technische Einflüsse auf Markerexpression
und Modellleistung werden dadurch nicht ausgeschlossen.“

## 5. Weitere bestätigte Auffälligkeiten

### Trapezoidale PR-AUC bei konstanten Scores

Bei zwei positiven und vier negativen Testspendern hat der konstante Score
0,5 mit der verwendeten trapezoidalen Berechnung eine PR-AUC von **2/3**.
Average Precision wäre dagegen **1/3**, entsprechend dem positiven Anteil.
Die lineare Interpolation zum PR-Endpunkt mit Precision 1 erzeugt den großen
Unterschied. Das ist kein Rechenfehler der implementierten Definition, kann
aber besonders die 15 konstanten Citrus-Ergebnisse irreführend gut erscheinen
lassen. Die
[scikit-learn-Dokumentation zu Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
unterscheidet die Definitionen ausdrücklich.

Aus denselben bestehenden Scores, ohne Neutraining, ergeben sich:

| Methode | Mediane trapezoidale PR-AUC | Mediane Average Precision |
|---|---:|---:|
| CellCNN | 0,7917 | 0,8333 |
| SVM | 0,7917 | 0,8333 |
| Citrus | 0,4167 | 0,5000 |

AP ist über alle Splits nicht zwangsläufig niedriger als die trapezoidale Fläche.
Die Definition sollte transparent beibehalten oder eine zusätzliche AP-Spalte
ausgewiesen werden; ein stillschweigender Austausch wäre nicht korrekt.

### Unvollständige Prüfung wiederverwendeter SVM-Ergebnisse

Ein kontrollierter Test setzte im Arbeitsspeicher `SVM_C_VALUES = [1.0]` bei
vollständig vorhandenen Full-Artefakten und aktivierter automatischer
Wiederverwendung. Das Notebook lud trotzdem die alten Ergebnisse mit
`best_C = 0.01`, führte keinen Fit aus und bestand alle Ergebnisprüfungen.

Das ist ein bestätigter Fehler im Konfigurationsabgleich. Das Notebook prüft
beim Wiederverwenden nicht, ob das gespeicherte C-Raster der aktuellen
Konfiguration entspricht. Auch weitere Modellparameter werden nicht vollständig
mit den Artefakten abgeglichen. Vorhandene CSV-Dateien dürfen nach einer
Parameteränderung deshalb nicht ungeprüft weiterverwendet werden.

Für die aktuellen drei C-Werte gibt es keinen Nachweis veralteter Ergebnisse:
Die direkten Neuberechnungen bestätigen ihre gespeicherten AUCs und die
Testvorhersagen von Split 0. Der Wiederverwendungsfehler erklärt die jetzigen
Gleichstände nicht.

### CellCNN-Auswahl und gespeicherte Modelle

Alle 100 ausgewählten CellCNN-Kandidaten entsprechen den gespeicherten
Validierungskennzahlen und der im Code festgelegten Sortierregel. Filterzahl,
innerer Fold und die exportierten Filterzeilen sind konsistent. Die Architektur
mit linearem Zellfilter, ReLU und Top-k-Mittelung stimmt mit dem
[offiziellen Architekturcode](https://eiriniar.github.io/CellCnn/_modules/cellCnn/model.html)
überein. Für CellCNN wurde bei dieser Prüfung kein neues Training ausgeführt.

Die aktuelle Filterdatei enthält keine Ausgabe-Biases, die SVM speichert keine
Gewichte und keinen Scaler des äußeren Fits. Das verfälscht die vorhandenen
Metriken nicht, begrenzt aber die spätere Modellrekonstruktion ohne Neutraining.
CellCNN-Zellantworten sind aus den gespeicherten Filtern und Scalern berechenbar;
vollständige Klassenwahrscheinlichkeiten sind daraus allein nicht rekonstruierbar.

## 6. Konsequenz für den aktuellen Stand

Ein neuer vollständiger Benchmark ist durch diese Prüfung nicht begründet.
Die vorhandenen ROC-AUC-Ergebnisse sind in den geprüften Rechenwegen konsistent.
Vor der weiteren Nutzung sollten die Batch-Aussage präzisiert, numerische
Nullkoeffizienten bei Citrus einheitlich behandelt und die SVM-Wiederverwendung
gegen geänderte Parameter abgesichert werden. Für die PR-Auswertung bietet sich
ein ergänzender, eindeutig als Average Precision bezeichneter Wert an.

Die Bewertung „kein nachgewiesener Rechenfehler“ bezieht sich auf die genannten
Prüfungen. Sie ersetzt weder ein Neutraining aller 100 Splits noch eine externe
Validierung der Modelle.
