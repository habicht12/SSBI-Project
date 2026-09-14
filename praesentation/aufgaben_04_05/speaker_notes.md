# Sprechernotizen zu Aufgaben 4 und 5

Englische Folien, deutscher Vortragstext. Geplant sind **8:00 Minuten** für die
neun Hauptfolien. Die beiden Reservefolien gehören nicht zum regulären Vortrag.
Die Zeitangaben sind eine Planung, keine gemessene Vortragsdauer.

| Folie | Dauer | Bis |
|---|---:|---:|
| 1 – Daten und Versuchsaufbau | 0:45 | 0:45 |
| 2 – CellCNN | 1:00 | 1:45 |
| 3 – Citrus | 0:45 | 2:30 |
| 4 – SVM-Zellscore | 0:45 | 3:15 |
| 5 – SVM-Spenderscore | 1:00 | 4:15 |
| 6 – Vergleich | 1:00 | 5:15 |
| 7 – Wiederkehrende Subsets | 1:00 | 6:15 |
| 8 – Ausgewählte Zellen | 0:45 | 7:00 |
| 9 – Biologische Einordnung | 1:00 | 8:00 |

## 1. From cells to donor-level prediction — 45 Sekunden

„Für Aufgabe 4 wollen wir den CMV-Status eines Spenders vorhersagen. Dafür verwenden
wir die lebenden, von Doubletten bereinigten PBMCs ohne zusätzliches NK-Gate:
20 Spender und 37 Marker. Wir transformieren die Intensitäten mit ArcSinh. CellCNN
und SVM erhalten zusätzlich eine Standardisierung, die nur auf Trainingsspendern
gelernt wird. Pro Wiederholung stehen 14 Spender zum Training und sechs zum Test
bereit. Die Modellauswahl erfolgt innerhalb der Trainingsspender mit drei Folds.
Alle drei Methoden verwenden dieselben 30 äußeren Splits. Entscheidend ist:
Die unabhängigen Beobachtungen sind die Spender, und ihre Zellen bleiben zusammen.“

Auf das Splitdiagramm zeigen. Keine erneute allgemeine Einführung in CyTOF nötig.

## 2. CellCNN learns filters for informative cells — 60 Sekunden

„CellCNN bekommt Gruppen von jeweils 3.000 Zellen. Jeder Filter bildet eine
gewichtete Kombination der 37 Marker; ReLU setzt negative Antworten auf null.
Anschließend mitteln wir pro Filter die stärksten ein Prozent der Antworten.
So kann eine kleine, informative Zellpopulation zum Spenderscore beitragen.
Die gepoolten Werte liefern über die Ausgabeschicht eine CMV-Wahrscheinlichkeit.
Filter und Vorhersage werden gemeinsam aus den Spenderlabels gelernt. Wir erzeugen
200 Zellgruppen pro Trainingsspender und wählen das beste Netz anhand der inneren
Validierung. Unsere PyTorch-Umsetzung folgt dem Paper, verwendet aber eine feste
Suche mit drei, vier oder fünf Filtern. Für den Test mitteln wir fünf
Input-Wahrscheinlichkeiten statt einer. Das ausgewählte Netz wird nicht noch einmal
auf allen 14 Spendern trainiert.“

Am Diagramm entlanggehen. Das Pooling erklärt die Vorhersage; die spätere
Halbmaximum-Schwelle zur Subset-Abgrenzung ist ein eigener Schritt. Ein Testinput
enthält höchstens 20.000 Zellen; die feste Filtersuche ersetzt die Zufallssuche des
Papers.

## 3. Citrus predicts from population frequencies — 45 Sekunden

„Citrus geht in zwei Schritten vor. Zunächst entsteht aus gleich vielen Zellen je
Trainingsspender ein hierarchischer Ward-Baum. Cluster ab 0,05 Prozent der
Trainingszellen liefern für jeden Spender eine Zellhäufigkeit. Diese Häufigkeiten
gehen in eine logistische Regression mit L1-Regularisierung ein. Viele
Koeffizienten werden dadurch null; die übrigen Cluster sind für das Modell
relevant. Neue Zellen übernehmen die Clusterzugehörigkeiten ihrer nächsten
Trainingszelle. Wir nutzen die originale R-Implementierung mit 10.000 Zellen pro
Spender. Zellzahl und 30 Wiederholungen sind ein Rechenkompromiss gegenüber dem
Paper. Anders als bei CellCNN werden die Zellpopulationen vor der überwachten
Klassifikation gebildet.“

Die Cluster des Baums sind verschachtelt, also keine disjunkte Zelltyp-Tabelle.
Verwendet wird die originale Citrus-R-Implementierung v0.8.

## 4. A linear SVM assigns a score to each cell — 45 Sekunden

„Unsere dritte Methode ist eine lineare Single-Cell-SVM. Jede Trainingszelle erhält
zunächst das CMV-Label ihres Spenders. Das ist eine schwache Zuordnung: Ein
positiver Spender besteht nicht ausschließlich aus krankheitsassoziierten Zellen.
Die Grafik zeigt das Prinzip schematisch mit zwei Markern; tatsächlich verwenden
wir alle 37. Die SVM lernt eine lineare Entscheidungsfunktion. Ihr Wert ist die
Margin einer Zelle: Je größer der Wert, desto stärker liegt die Zelle in der
positiven Modellrichtung. Dieser Score ist keine Wahrscheinlichkeit. Eine
Single-Cell-SVM kommt bereits als Baseline im Paper vor. Unsere Anpassung betrifft
den nächsten Schritt, nämlich die Zusammenfassung zum Spenderscore.“

Die Punktfarben bezeichnen Spenderlabels, keine validierten Einzelzellzustände.

## 5. Our adaptation: aggregate the highest cell margins — 60 Sekunden

„Für die Spendervorhersage sortieren wir alle Zellmargins eines Testspenders und
mitteln die höchsten ein Prozent. Bei tausend Zellen sind das zehn. Im gezeigten
Rechenbeispiel haben diese zehn Zellen Margins von 2,1 bis 3,0; der Mittelwert ist
2,55. Bei einer beispielhaften Schwelle von 1,8 wäre die Vorhersage positiv.
Diese Zahlen illustrieren nur die Regel. Im tatsächlichen Benchmark wählen wir
zunächst den SVM-Parameter C anhand der mittleren inneren Spender-AUC. Die
Entscheidungsschwelle lernen wir aus inneren Out-of-fold-Spenderscores, indem wir
Sensitivität plus Spezifität maximieren. Danach wird die SVM auf allen 14
Trainingsspendern neu gefittet. Der äußere Test liefert weder C noch die Schwelle.
Die Top-1-Prozent-Aggregation ist unsere vorab festgelegte Projektanpassung.“

Die Breiten im Schema sind nicht proportional zur Zellzahl. Für AUC wird der
kontinuierliche Spenderscore verwendet, nicht das bereits binarisierte Label.

## 6. Performance across 30 shared donor splits — 60 Sekunden

„Hier vergleichen wir die drei Methoden auf genau denselben 30 Splits. Jeder Punkt
ist die ROC-AUC auf sechs Testspendern; die Box zeigt das mittlere Quartilsintervall.
CellCNN und SVM erreichen beide eine mediane AUC von 0,875. Citrus liegt bei 0,625.
Die Streuung ist deutlich, besonders bei Citrus. Die ergänzenden Kennzahlen sind
Average Precision und Balanced Accuracy. Auch dort haben CellCNN und SVM hier
dieselben Mediane, obwohl ihre einzelnen Vorhersagen unterschiedlich sein können.
Bei Balanced Accuracy hängt das Ergebnis zusätzlich von der Entscheidungsschwelle
ab. Wir können daraus keinen allgemeinen Überlegenheitsnachweis ableiten: Es
bleiben 20 unabhängige Spender, und die wiederholten Splits überlappen. Die
Quartile beschreiben diese Splitstreuung und sind keine Konfidenzintervalle.“

Die BA-Schwellen sind 0,5 für CellCNN/Citrus und innerlich gelernt für die SVM.

Wenn nach der groben AUC-Abstufung gefragt wird: zwei positive und vier negative
Testspender ergeben nur acht positive-negative Paare; Gleichstände tragen halb bei.

## 7. Which CellCNN and Citrus subsets recur? — 60 Sekunden

„Aufgabe 5 fragt, welche Zellsubsets mit den Modellen verbunden sind. Für CellCNN
teilen wir die Filterantwort durch ihr Maximum auf der ursprünglichen
Trainingsreferenz. Antworten über 0,5 definieren das Subset; sein mittleres
Markerprofil ist der Zentroid. Bei Citrus nutzen wir die gespeicherten Zentroiden
der Cluster mit wirksamen Regressionskoeffizienten. Ähnliche Zentroiden gruppieren
wir getrennt für jede Methode und zählen, in wie vielen unterschiedlichen Splits
eine Gruppe erscheint. Ab sechs von 30 zeigen wir sie auf der t-SNE-Karte. Die
häufigste CellCNN-Gruppe erscheint in allen 30, die häufigste Citrus-Gruppe in 18
Splits. Beide sind positiv mit CMV assoziiert. Die Punkte sind projizierte
Trainingszentroiden; gleiche Farben zwischen den Panels bedeuten keine identische
Population.“

Die Karte umfasst hier **10.000 `gated_alive`-Zellen**, 500 je Spender, Perplexität
30. Das ist nicht die 40.000-NK-Zellen-Karte der bisherigen Aufgabe-2-Folien auf
`main`. Der gemeinsame Darstellungsraum wurde explorativ auf allen Spendern
festgelegt und geht nicht in die Klassifikatorbewertung ein. Details bei Nachfrage:
Average-Linkage, Kosinusdistanz, Schnitt 0,4; dokumentierte Übertragung aus der
Filtergruppierung des Originalcodes. Die drei Citrus-Nullmodelle zählen im Nenner.

## 8. Which cells does the SVM repeatedly select? — 45 Sekunden

„Links sehen wir 87 Karten-Zellen, die ein repräsentativer CellCNN-Filter auswählt.
Das ist eine explorative Anwendung dieses einen Filters auf alle Spender.
Rechts sind 93 Karten-Zellen markiert, die die SVM mindestens einmal positiv
ausgewählt hat. Dafür muss eine Zelle zu den höchsten ein Prozent der Margins
ihres vollständigen Testspenders gehören und über der gespeicherten Schwelle
liegen. Die Farbe zeigt, wie oft das über alle Testauftritte ihres Spenders
passiert. Je Spender sind das vier bis 16 Auftritte. Diese Zellhäufigkeit und die
Gruppenwiederkehr der vorherigen Folie haben unterschiedliche Nenner und sind
keine unmittelbar vergleichbaren Effektstärken.“

Die Kartenstichprobe begrenzt die Darstellung: 93 ist die Zahl markierter
Karten-Zellen, keine Gesamtpopulation und kein fester Anteil der dargestellten Karte.

## 9. What do the marker profiles support? — 60 Sekunden

„Diese Heatmap ergänzt jetzt das SVM-Profil. CellCNN und Citrus zeigen weiterhin
je einen gespeicherten Gruppenrepräsentanten. Für die SVM mitteln wir positiv
ausgewählte Zellen zunächst je Testauftritt, dann je Spender und schließlich über
alle 20 Spender mit gleichem Gewicht. 168 von 180 Testauftritten tragen bei.
CellCNN zeigt erhöhte NKG2C- und CD57-Werte zusammen mit NK-assoziierten Markern,
vereinbar mit dem memory-like NK-Profil des Papers. Beim Citrus-Repräsentanten
sprechen CD3 und CD57 eher für ein T-assoziiertes Muster. Auch die SVM zeigt
erhöhte NKG2C-, CD57- und NK-assoziierte Marker; CD3 liegt etwas über dem
Referenzmittel. Die gemeinsame z-Skala macht die Werte vergleichbar, aber die
Subset-Definitionen unterscheiden sich. Diese Profile stützen biologische
Hypothesen, keine gesicherten Zelltypzuordnungen.“

Die SVM verwendet sämtliche Zellen der jeweiligen äußeren Testspender in Splits
0–29, unabhängig von deren wahrem CMV-Label; die 93 Karten-Zellen begrenzen dieses
Profil nicht. Zwölf Testauftritte haben keine positive Auswahl. Ihre Markerprofile
sind fehlend und werden nicht als Nullwerte eingerechnet. Alle 20 Spender tragen
mit mindestens einem nichtleeren Auftritt bei. Zunächst werden die nichtleeren
Auftritte je Spender gleich gewichtet, danach die Spender. Das Profil beschreibt
Markerausprägung bedingt auf positive Auswahl, nicht deren Häufigkeit.

CellCNN: Split 9, Filter 2; Citrus: Split 25, Cluster 139891. Die Repräsentanten
wurden unverändert übernommen. Die SVM-Zeile ist kein Gruppenrepräsentant.
Nicht aus z-Werten „positiv/negativ gegatet“ ableiten. Mittelprofile können
heterogene Zellen zusammenfassen; CD3 ist auch beim CellCNN-Profil nicht null.

## Reserve 10. Hyperparameter und Abweichungen

Nur bei Nachfrage öffnen. CellCNN verwendet das ausgewählte innere Netz mit neun
oder zehn Trainingsspendern; Citrus und SVM fitten abschließend auf 14. Bei Citrus
werden innere Bäume foldweise gelernt, das originale Lambda-Raster verwendet aber
alle 14 äußeren Trainingsspender. Alle äußeren Testspender bleiben ausgeschlossen.
Die Frequenzmerkmale werden innerhalb der glmnet-Fits standardisiert; auf den
ArcSinh-Markern selbst verwendet Citrus kein zusätzliches z-Scoring.

CellCNN-Validierungs-Gleichstände: AUC, Verlust, weniger Filter, Fold-Reihenfolge.
SVM-Gleichstände bei C: kleinerer Wert. Citrus cv.min minimiert den inneren
Klassifikationsfehler, bei Gleichstand mit stärkerer Regularisierung. Die
37 bereitgestellten Marker weichen von der Paperangabe von 36 ab. Unterschiedliche
Zellstichproben und GPU/CPU-Hardware begrenzen einen direkten Laufzeitvergleich.

## Reserve 11. Zusätzlicher Vergleich über 100 Splits

Dieser Vergleich enthält ausschließlich CellCNN und SVM. Beide erreichen auch
hier eine mediane AUC von 0,875. Die mediane Balanced Accuracy beträgt hier 0,750
für CellCNN und 0,625 für SVM. Die 600 Testvorhersagen je Methode sind wiederholte
Vorhersagen derselben 20 Spender. Diese Werte nicht mit dem Dreiervergleich über
30 Splits zu einer gemeinsamen Tabelle oder Rangfolge vermischen.

## Quellen und Ergebnisstand

Grundlagen: [CellCNN-Paper](https://doi.org/10.1038/ncomms14825),
[Citrus-Paper](https://doi.org/10.1073/pnas.1408792111),
[Horowitz et al.](https://doi.org/10.1126/scitranslmed.3006702).
Projektstand: gespeicherte Aufgabe-4-/Aufgabe-5-Ergebnisse vom 9. September 2026;
Dateiprüfsummen und verwendete Exporte stehen in `data/provenance.json`.
