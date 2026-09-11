# Sprechernotizen zu Aufgaben 4 und 5

Englische Folien, deutscher Vortragstext. Geplant sind **10:00 Minuten** für die
13 Hauptfolien. Die sieben Reservefolien gehören nicht zum regulären Vortrag.
Die Zeitangaben sind Ziele, keine gemessene Vortragsdauer. Die Zitate bilden den
Sprechtext; die weiteren Absätze enthalten Zeigehinweise und Material für Rückfragen.

| Folie | Dauer | Bis |
|---|---:|---:|
| 1 – Daten und Versuchsaufbau | 0:35 | 0:35 |
| 2 – CellCNN | 0:55 | 1:30 |
| 3 – Citrus einrichten | 0:30 | 2:00 |
| 4 – Citrus-Ablauf | 0:45 | 2:45 |
| 5 – SVM-Zellscore | 0:40 | 3:25 |
| 6 – SVM-Spenderscore | 0:55 | 4:20 |
| 7 – Methodenvergleich | 0:50 | 5:10 |
| 8 – CellCNN-Zellauswahl | 0:35 | 5:45 |
| 9 – Citrus-Clusterauswahl | 0:35 | 6:20 |
| 10 – SVM-Zellauswahl | 0:45 | 7:05 |
| 11 – Wiederkehrende Subsets | 1:00 | 8:05 |
| 12 – Ausgewählte Zellen | 0:55 | 9:00 |
| 13 – Biologische Einordnung | 1:00 | 10:00 |

## 1. From cells to donor-level prediction — 35 Sekunden

> Aufgabe 4 sagt den CMV-Status eines Spenders voraus. Unsere Daten umfassen
> 20 Spender und 37 Marker. ArcSinh transformiert jeden Markerwert mit Kofaktor
> fünf. CellCNN und SVM erhalten zusätzlich eine Skalierung aus Trainingsspendern.
> Pro Split verwenden wir 14 Spender zum Training und sechs zum Test; drei innere
> Folds dienen der Modellauswahl. Alle Methoden teilen dieselben 30 Splits.
> Die Zellen eines Spenders bleiben dabei zusammen.

Auf das Splitdiagramm zeigen. Die unabhängigen Beobachtungen sind die Spender.
`gated_alive` umfasst lebende, von Doubletten bereinigte PBMCs ohne zusätzliches
NK-Gate. In der Formel bezeichnet der erste Index die Zelle, der zweite den Marker;
x ist der Rohwert, a der transformierte Wert.

## 2. CellCNN learns filters for informative cells — 55 Sekunden

> CellCNN bekommt Gruppen von jeweils 3.000 Zellen. Jeder Filter bildet aus den
> standardisierten Markern eine gewichtete Summe plus Bias. ReLU setzt negative
> Antworten auf null. Anschließend mitteln wir pro Filter die stärksten ein Prozent
> der Antworten. So kann eine kleine Population zur CMV-Wahrscheinlichkeit beitragen.
> Filter und Vorhersage werden gemeinsam aus Spenderlabels gelernt. Wir erzeugen
> 200 Zellgruppen pro Trainingsspender und wählen das beste Netz in der inneren
> Validierung. Unsere PyTorch-Umsetzung folgt Paper und Referenzcode, verwendet aber
> eine feste Suche mit drei, vier oder fünf Filtern. Für den Test mitteln wir fünf
> Input-Wahrscheinlichkeiten. Das ausgewählte Netz wird anschließend nicht neu trainiert.

Am Diagramm entlanggehen. Die Variablenlegende erklärt Zellindex i, Filterindex f,
Markervektor z, Gewichte w, Bias b und Filterantwort r. Ein Testinput enthält
höchstens 20.000 Zellen. Die feste Filtersuche ersetzt die Zufallssuche des Papers.
Die spätere Halbmaximum-Auswahl auf Folie 8 hat eine andere Aufgabe als das Pooling.

## 3. Running the original Citrus implementation — 30 Sekunden

> Für Citrus verwenden wir den originalen Paketcode in einer separaten
> Conda-Umgebung mit R 4.5. Sie stellt auch die Bibliotheken und den C++-Compiler
> bereit. Citrus 0.8 und Rclusterpp installieren wir aus festgelegten Git-Commits.
> Ein eigener R-Jupyter-Kernel führt das Notebook mit unseren gemeinsamen
> Spenderfolds aus. CellCNN haben wir anhand des Papers neu implementiert;
> bei Citrus verwenden wir die ursprüngliche Implementierung. Beide Wege sind papernah.

„Legacy“ bezeichnet das Citrus-Paket; der dokumentierte Lauf verwendet R 4.5.3
und Rclusterpp 0.2.6. `flowCore` liest FCS-Dateien, `glmnet` berechnet die
regularisierte Regression. Die Einrichtung ist in der Projekt-README und
`environment-citrus.yml` dokumentiert; keine zusätzlichen Quellcode-Reparaturen
behaupten. Feste Git-Commits legen die Quellstände von Citrus und Rclusterpp fest,
nicht sämtliche transitiven Paketversionen.

## 4. Citrus predicts from population frequencies — 45 Sekunden

> Citrus beginnt mit gleich vielen Zellen pro Trainingsspender und baut daraus
> einen hierarchischen Ward-Baum. Cluster ab 0,05 Prozent der Trainingszellen
> werden berücksichtigt; Eltern- und Kindcluster können sich überschneiden.
> Danach beschreiben wir jeden Spender durch die Zellhäufigkeit in diesen Clustern.
> Neue Zellen übernehmen die Zugehörigkeiten ihrer nächsten Trainingszelle.
> Schließlich wählt eine L1-logistische Regression informative Häufigkeiten aus
> und liefert eine CMV-Wahrscheinlichkeit. Die Populationen entstehen vor der
> überwachten Klassifikation. Unsere 10.000 Zellen pro Spender und 30 Splits sind
> ein Rechenkompromiss gegenüber 20.000 Zellen und 100 Splits im Paper.

Links die drei Schritte erklären, rechts am vertikalen Ablauf entlanggehen.
Die Cluster sind keine disjunkte Zelltyp-Tabelle. Die inneren Clusterbäume verwenden
jeweils nur innere Trainingsspender. Die Einschränkung des ursprünglichen
Lambda-Rasters steht auf Reservefolie 14.

## 5. A linear SVM assigns a score to each cell — 40 Sekunden

> Die lineare Single-Cell-SVM erhält 10.000 Trainingszellen pro Spender. Jede Zelle
> trägt zunächst das CMV-Label ihres Spenders; das ist eine schwache Zuordnung.
> Aus ihren standardisierten Markern berechnet die SVM eine gewichtete Summe
> plus Intercept. Das ist der Zellscore, auch Margin genannt. Höhere Werte zeigen
> stärker in die positive Modellrichtung; sie sind keine Wahrscheinlichkeiten.
> Die SVM lernt mit L2-Regularisierung und Squared-Hinge-Verlust. Aus diesen
> Zellscores müssen wir anschließend einen Spenderscore bilden.

Die Zeichnung illustriert zwei Marker; gerechnet wird mit allen 37. Die Punktfarben
bezeichnen Spenderlabels, keine validierten Zellzustände. Die Margin ist der
funktionale Entscheidungsscore und kein auf die Gewichtsnorm normierter geometrischer
Abstand. Die gesamte Trainingsregel steht auf Reservefolie 15.

## 6. Our adaptation: aggregate the highest cell margins — 55 Sekunden

> Für einen Spender mitteln wir die höchsten ein Prozent seiner Zellmargins.
> Die Anzahl wird aufgerundet. Bei tausend Zellen behalten wir zehn; die gezeigten
> Margins von 2,1 bis 3,0 ergeben den Spenderscore 2,55. Bei einer beispielhaften
> Schwelle von 1,8 wäre die Vorhersage positiv. Im Benchmark wählen wir zunächst
> den Verlustparameter C nach der mittleren inneren Spender-AUC. Die Schwelle
> lernen wir aus inneren Out-of-fold-Spenderscores durch Maximierung von
> Sensitivität plus Spezifität. Danach werden Scaler und SVM auf allen 14
> Trainingsspendern neu gefittet. Die Top-1-Prozent-Aggregation ist unsere vorab
> festgelegte Projektanpassung.

Die Formel benennt Spender d, Zellzahl N, ausgewählte Anzahl k, Indexmenge I,
Zellmargin m, Spenderscore s und Schwelle tau. C gewichtet die Zellverluste gegenüber
der Regularisierung; es ist keine Lernrate. Beispielzahlen und Diagrammbreiten
sind schematisch. AUC verwendet den kontinuierlichen Spenderscore. Die äußeren
Testspender bestimmen weder C noch die Schwelle. Details: Reservefolie 16.

## 7. Performance across 30 shared donor splits — 50 Sekunden

> Hier vergleichen wir die drei Methoden auf denselben 30 Splits. Jeder Punkt
> zeigt die ROC-AUC auf sechs Testspendern; die Box umfasst das erste bis dritte
> Quartil. CellCNN und SVM erreichen eine mediane AUC von 0,875, Citrus 0,625.
> Die Streuung ist deutlich. Die Tabelle ergänzt Average Precision und Balanced
> Accuracy; auch hier teilen CellCNN und SVM dieselben Mediane, obwohl einzelne
> Vorhersagen abweichen können. Balanced Accuracy hängt zusätzlich von der
> Entscheidungsschwelle ab. Die Quartile beschreiben Splitstreuung. Weil die
> Splits überlappen und weiterhin dieselben 20 Spender verwenden, folgt daraus
> kein allgemeiner Überlegenheitsnachweis.

Die Bildunterschrift steht direkt unter den Boxplots. BA-Schwellen: 0,5 für
CellCNN/Citrus, innerlich gelernt für SVM. Zwei positive und vier negative
Testspender ergeben acht positive-negative Paare; deshalb ist die AUC grob
abgestuft. Gleichstände zählen halb. Quartile sind keine Konfidenzintervalle.

## 8. CellCNN: select cells with a strong filter response — 35 Sekunden

> Aufgabe 5 fragt jetzt nach den ausgewählten Zellen. Bei CellCNN teilen wir die
> Antwort eines Filters durch dessen Maximum auf der ursprünglichen
> Trainingsreferenz. Der normierte Score muss strikt größer als 0,5 sein.
> Ist das Maximum zehn, werden Antworten von acht und sechs ausgewählt, fünf
> jedoch nicht. Diese Halbmaximum-Regel beschreibt das interpretierte Subset.
> Das Top-1-Prozent-Pooling aus Aufgabe 4 dient dagegen der Vorhersage.

Ein Score gehört zu einer Zelle und einem Filter. Die Richtung ergibt sich aus
dem Unterschied der beiden Ausgangsgewichte, nicht aus der stets nichtnegativen
ReLU-Antwort. Nur Filter mit positivem Referenzmaximum und wirksamem Ausgangskontrast
werden interpretiert. Die Referenz gehört zu den neun oder zehn tatsächlichen
inneren Trainingsspendern des ausgewählten Netzes. Details: Reservefolie 18.

## 9. Citrus: select clusters used by the classifier — 35 Sekunden

> Citrus wählt ganze Cluster über ihre Rolle im Klassifikator aus. Unser
> quantitativer Auswahlwert ist der Betrag des Regressionskoeffizienten ihrer
> Häufigkeit. Werte über der kleinen numerischen Nulltoleranz bleiben erhalten.
> Das Vorzeichen gibt die Richtung an: Ein positiver Koeffizient bedeutet,
> dass eine höhere Clusterhäufigkeit bei sonst gleichen Merkmalen die Vorhersage
> Richtung CMV-positiv verschiebt. Wir verwenden die gespeicherten mittleren
> Markerprofile dieser Cluster. Deshalb zeigt unsere Citrus-Karte Zentroiden.

Die Toleranz beträgt 10 hoch minus 10. Sie definiert numerische Null, keine
biologische Relevanz oder statistische Signifikanz. Es werden alle wirksamen
Cluster berücksichtigt. Die vorhandenen Exporte enthalten nicht den vollständigen
Baum und seine Mitgliedschaften; daher lässt sich keine exakte individuelle
Citrus-Zellauswahl auf der Karte rekonstruieren. Details: Reservefolie 19.

## 10. SVM: select the strongest positive cell scores — 45 Sekunden

> Bei der SVM berechnen wir zunächst die Margin jeder Zelle eines vollständigen
> Testspenders. Dann wählen wir die höchsten ein Prozent innerhalb dieses
> Spenders aus. Für positive Auswahl muss die Margin zusätzlich über der
> gelernten Spenderschwelle liegen. Dafür ziehen wir die Schwelle vom Zellscore
> ab; ein positiver Restwert erfüllt diese zweite Bedingung. Beide Bedingungen
> müssen gemeinsam gelten. Jedes Modell bewertet ausschließlich seine äußeren
> Testspender. Erst danach schränken wir die Darstellung auf die Zellen der
> vorhandenen Karte ein.

Die Schwelle stammt aus innerer Validierung. Sie ist weder eine neu gelernte
Zellschwelle noch automatisch null. Der Anteil wird auf der vollständigen Probe
bestimmt, nicht auf den 500 Karten-Zellen des Spenders. „Positiv ausgewählt“ ist
eine Modellrichtung und kein Nachweis, dass eine Zelle CMV-infiziert ist.
Details zu Gleichständen, negativen Auswahlen und Häufigkeiten: Reservefolie 20.

## 11. Which CellCNN and Citrus subsets recur? — 60 Sekunden

> Nachdem die Subsets feststehen, vergleichen wir ihre mittleren Markerprofile
> über die Splits. Ähnliche Zentroiden gruppieren wir getrennt für CellCNN und
> Citrus. Dann zählen wir die unterschiedlichen Splits, in denen eine Gruppe
> erscheint. Mehrere Mitglieder aus demselben Split zählen nur einmal. Ab sechs
> von 30 Splits zeigen wir eine Gruppe auf der t-SNE-Karte. Die häufigste
> CellCNN-Gruppe erscheint in allen 30 Splits, die häufigste Citrus-Gruppe in 18;
> beide sind positiv mit CMV assoziiert. Die Sterne markieren gespeicherte
> Repräsentanten. Die Punkte zeigen projizierte Trainingszentroiden; gleiche
> Farben zwischen den Panels bezeichnen keine identischen Populationen.

Die Karte umfasst **10.000 `gated_alive`-Zellen**, 500 je Spender, Perplexität 30.
Sie unterscheidet sich von der 40.000-NK-Zellen-Karte in den Aufgabe-2-Folien auf
`main`. Zentroiden werden über die nächste Karten-Zelle projiziert. Karte und
gemeinsame Profilskalierung wurden explorativ mit allen Spendern festgelegt;
sie gehen nicht in die Klassifikatorbewertung ein.

Gruppierung: Average-Linkage, Kosinusdistanz, Schnitt 0,4. Die Übertragung aus der
Filtergruppierung des Originalcodes auf Subset-Zentroiden ist eine dokumentierte
Anpassung. Die drei Citrus-Nullmodelle zählen im Nenner. Gruppen-IDs gelten je
Methode. Die Wiederkehrschwelle ist kein Signifikanztest.

## 12. Which cells does the SVM repeatedly select? — 55 Sekunden

> Links markiert ein repräsentativer CellCNN-Filter 87 Karten-Zellen. Das ist seine
> explorative Anwendung auf alle Spender. Rechts sind 93 Karten-Zellen dargestellt,
> die die SVM mindestens einmal positiv ausgewählt hat. Die Farbe beschreibt,
> bei welchem Anteil der Testauftritte ihres Spenders das passiert. Der Nenner
> enthält alle Testauftritte, auch solche ohne Auswahl; pro Spender sind das
> vier bis 16. Eine Zelle mit drei positiven Auswahlen in acht Testauftritten
> erhält also drei Achtel. Diese Wiederkehr derselben Zelle unterscheidet sich
> von der Wiederkehr ähnlicher Zentroidgruppen auf der vorherigen Folie.

Die Häufigkeiten haben unterschiedliche Nenner und sind keine vergleichbaren
Effektstärken. Die 93 beziehen sich nur auf markierte Karten-Zellen, nicht auf
die Gesamtpopulation. Die SVM-Zellauswahl erfolgt auf vollständigen Testspendern.
CellCNN verwendet für diese Zellkarte den unveränderten Repräsentanten der
häufigsten Gruppe; es handelt sich nicht um eine äußere Testbewertung aller Zellen.

## 13. What do the marker profiles support? — 60 Sekunden

> Die Heatmap zeigt für CellCNN und Citrus gespeicherte Gruppenrepräsentanten.
> Für die SVM mitteln wir positiv ausgewählte Zellen zunächst je Testauftritt,
> dann je Spender und schließlich über alle 20 Spender mit gleichem Gewicht.
> 168 von 180 Testauftritten tragen bei. CellCNN zeigt erhöhte NKG2C- und CD57-Werte
> zusammen mit NK-assoziierten Markern, vereinbar mit einem memory-like NK-Profil.
> Beim Citrus-Repräsentanten sprechen CD3 und CD57 eher für ein T-assoziiertes
> Muster. Die SVM zeigt ebenfalls erhöhte NKG2C-, CD57- und NK-Marker; CD3 liegt
> über dem Referenzmittel. Die gemeinsame z-Skala unterstützt den Vergleich,
> aber die Subset-Definitionen unterscheiden sich. Die Profile stützen biologische
> Hypothesen, keine gesicherten Zelltypzuordnungen.

Das SVM-Profil verwendet sämtliche Zellen der jeweiligen äußeren Testspender in
Splits 0–29, unabhängig von wahrem Label oder vorhergesagter Klasse. Es ist nicht
auf die 93 Karten-Zellen beschränkt. Zwölf Testauftritte haben keine positive
Auswahl; diese Profile sind fehlend, keine Nullwerte. Zuerst werden nichtleere
Auftritte je Spender gleich gewichtet, danach die Spender. Das Profil beschreibt
Markerausprägung bedingt auf positive Auswahl, nicht deren Häufigkeit.

Die Repräsentanten bleiben CellCNN Split 9/Filter 2 und Citrus Split 25/Cluster
139891. Die SVM-Zeile ist kein Gruppenrepräsentant. z-Werte sind keine
Positivitätsgates. Mittelprofile können heterogene Zellen zusammenfassen; CD3
ist auch beim CellCNN-Profil nicht null.

## 14. Backup: hyperparameters and paper deviations

CellCNN behält das ausgewählte innere Netz mit neun oder zehn Trainingsspendern.
Citrus und SVM verwenden für ihre finalen Modelle alle 14 äußeren Trainingsspender.
Citrus fittet seine inneren Bäume pro Fold; das originale Lambda-Raster verwendet
jedoch Clustermerkmale und Labels aller 14 äußeren Trainingsspender. Die äußeren
Testspender bleiben vom Fit ausgeschlossen. Deshalb keine pauschale vollständige
Trennung aller inneren datenabhängigen Entscheidungen behaupten.

Bei CellCNN entscheiden nach Validierungsgenauigkeit die AUC, der Loss, weniger
Filter und die Fold-Reihenfolge. SVM-Gleichstände bevorzugen kleineres C.
Citrus `cv.min` minimiert den inneren Klassifikationsfehler und bevorzugt bei
Gleichstand stärkere Regularisierung. Die 37 bereitgestellten Marker unterscheiden
sich von den 36 im Paper. Zellstichproben und unterschiedliche GPU-/CPU-Nutzung
begrenzen direkte Laufzeitvergleiche.

Die Average-Linkage-/Kosinus-/0,4-Regel wurde aus der Filtergruppierung des
Referenzcodes auf Subset-Zentroiden übertragen. Sie ist keine belegte identische
Originalparametrisierung der NK-Zentroidanalyse.

## 15. Backup: SVM approach 1 — training the cell scorer

Links steht die Vorverarbeitung: Pro Trainingsspender werden mit festen Seeds
10.000 Zellen ohne Zurücklegen gezogen. ArcSinh verwendet Kofaktor fünf; ein
StandardScaler wird gemeinsam auf genau diesen Trainingszellen gefittet und
unverändert auf Validierung oder Test angewendet. Gleiche Zellzahlen gewichten
Spender in der Verlustfunktion gleich; es gibt keine zusätzliche Klassengewichtung.
Die Zellen erben das Spenderlabel als schwache Zuordnung.

Rechts steht das tatsächliche Optimierungsziel von `LinearSVC`: kleine Parameter
werden durch L2 bestraft, Verletzungen der angestrebten Label-Margin durch den
quadrierten Hinge-Verlust. C gewichtet den Verlust relativ zur Regularisierung.
Das mathematische Label y ist hier minus oder plus eins; der Buchstabe t bleibt
für den zentrierten Interpretationsscore reserviert. Der Intercept wird bei
`intercept_scaling=1` ebenfalls regularisiert, deshalb steht sein Quadrat in der
Formel. Die SVM lernt 37 Markergewichte und einen Intercept, ohne Kalibrierung.

Numerische Einstellungen: `dual="auto"`, maximal 10.000 Iterationen, Toleranz 0,0001.
Bei deutlich mehr Zellen als Markern wird hier primal optimiert. Es gibt kein
validierungsbasiertes Early Stopping wie bei CellCNN.

## 16. Backup: SVM approach 2 — model selection and test

Innerhalb der 14 äußeren Trainingsspender vergleichen wir C = 0,01, 0,1 und 1
in drei festen Spenderfolds. Alle C-Werte eines Folds verwenden dieselbe Zellstichprobe
und denselben Scaler. Die höchste mittlere innere Spender-AUC entscheidet;
bei Gleichstand gewinnt das kleinere C. Für dieses C liegen 14 Out-of-fold-Scores
vor, je einer pro Trainingsspender. Daraus wählen wir die endliche ROC-Schwelle
mit maximalem Youden-Index, also Sensitivität plus Spezifität minus eins.
Bei Gleichstand gewinnt die höchste angebotene optimale Schwelle.

Danach ziehen wir eine neue deterministische Stichprobe aus allen 14 Trainingsspendern
und fitten Scaler sowie finale SVM neu. Die innere Schwelle bleibt bestehen.
Für jeden der sechs Testspender berechnen wir sämtliche Zellmargins, mitteln die
höchsten ein Prozent und vergleichen mit der Schwelle. Gleichheit bedeutet eine
positive Spendervorhersage; bei der positiven Einzelzellauswahl aus Aufgabe 5 gilt
dagegen eine strenge Überschreitung. AUC verwendet kontinuierliche Scores, BA
binarisierte Vorhersagen.

Die Scores der inneren Modelle sind unkalibriert; ihre Skalen können sich untereinander
und gegenüber dem neu trainierten Modell unterscheiden. Die Übertragung der Schwelle
ist deshalb eine Einschränkung. Weder Modellauswahl noch Schwellenwahl verwendet
die äußeren Testspender. Die gespeicherten inneren Einzelscores werden nicht erneut
berechnet; wir erläutern den vorhandenen Ablauf.

## 17. Backup: CellCNN and SVM across 100 splits

Hier werden ausschließlich CellCNN und SVM verglichen. Beide erreichen erneut eine
mediane AUC von 0,875. Die mediane Balanced Accuracy beträgt 0,750 für CellCNN und
0,625 für SVM. Die 600 Testvorhersagen je Methode betreffen dieselben 20 Spender
wiederholt. Diesen Zweiervergleich getrennt vom gemeinsamen Dreiervergleich über
30 Splits erläutern; Citrus gehört nicht in diese Tabelle.

## 18. Backup: CellCNN subset selection in detail

Die linke Spalte zeigt Antwort und Referenz. Das ausgewählte Netz wurde auf neun
oder zehn inneren Trainingsspendern gelernt. Mit dem ursprünglichen Kandidatenseed
rekonstruieren wir je 20.000 Referenzzellen, ohne Scaler oder Modell neu zu fitten.
Die nichtnegative ReLU-Antwort wird durch ihr filterbezogenes Referenzmaximum geteilt.
Eine Antwort genau auf dem Halbmaximum wird ausgeschlossen.

Rechts folgt die Modellrichtung aus dem Unterschied der CMV+- und CMV−-Ausgangsgewichte.
Filter mit Referenzmaximum null oder Ausgangskontrast null liefern kein relevantes
Subset. Für wirksame Filter mitteln wir die ArcSinh-Markervektoren der ausgewählten
Referenzzellen. Das ergibt einen Zentroiden mit 37 Werten. Die ausgewählten Zellen
sind gleich gewichtet; Spender mit mehr ausgewählten Zellen tragen stärker bei.
Der normierte Score ist keine Wahrscheinlichkeit und kann für neue Zellen über eins
liegen. Das Subset ist keine exakte Liste der in jeder Vorhersage gepoolten Zellen.

## 19. Backup: Citrus subset selection in detail

Der lineare logistische Prädiktor verbindet Clusterhäufigkeiten mit ihren
Koeffizienten. `glmnet` standardisiert die Häufigkeitsmerkmale intern. Vor dem
Clustering erhalten die ArcSinh-Marker dagegen keine zusätzliche z-Skalierung.
Alle Koeffizienten mit Betrag über 10 hoch minus 10 werden berücksichtigt.
Das Vorzeichen beschreibt den bedingten Effekt der jeweiligen Häufigkeit.

Jeder gespeicherte Zentroid mittelt die ArcSinh-Profile der ursprünglichen
Trainingszellen seines Clusters. Eltern- und Kindcluster können dieselben Zellen
enthalten. Zentroiden und Koeffizienten reichen nicht aus, um beliebige neue
Karten-Zellen exakt dem ursprünglichen Baum zuzuordnen. Eine Zuordnung zum nächsten
Zentroiden wäre eine zusätzliche Regel. Die drei Nullmodelle ohne wirksamen Cluster
bleiben bei der Wiederkehrzählung im Nenner von 30 Splits.

## 20. Backup: SVM cell selection and recurrence

Zuerst werden exakt die aufgerundeten höchsten ein Prozent aller Zellen eines
Testspenders gewählt. Bei Gleichständen an der Grenze entscheidet die ursprüngliche
FCS-Ereignisnummer. Innerhalb dieser Auswahl bezeichnet der Abstand zur gelernten
Spenderschwelle positive oder negative Auswahl; Gleichheit gehört zu keiner Richtung.
Negative Auswahl umfasst daher keine separat ausgewählten niedrigsten Zell-Scores.

Die positive Häufigkeit zählt erfolgreiche positive Auswahlen derselben Zelle,
geteilt durch alle äußeren Testauftritte ihres Spenders. Der negative Anteil entsteht
analog. Innerhalb eines Splits schließen sich beide Richtungen aus; über verschiedene
Splits kann dieselbe Zelle beide Richtungen annehmen. Der Mittelwert der zentrierten
Top-Scores ist exakt der Abstand des Spenderscores zur Schwelle. Das ist eine
Zerlegung bei fester Auswahl und kein kausaler Entfernungseffekt einer Zelle.

Beispiel: 101 Zellen liefern zwei Top-Zellen. Bei Margins neun und acht und Schwelle
8,5 ist eine positiv und eine negativ; der Spenderscore liegt genau auf der Schwelle.

## Quellen und Ergebnisstand

Maßgeblich sind [Aufgabe 4: Methodenerklärungen](../../results/AUFGABE_4_METHODENERKLAERUNGEN.md)
und [Aufgabe 5: Methodenerklärungen](../../results/AUFGABE_5_METHODENERKLAERUNGEN.md),
die vorhandenen Notebooks und die Interpretationsimplementierung.
Hintergrund: [CellCNN-Paper](https://doi.org/10.1038/ncomms14825),
[Citrus-Paper](https://doi.org/10.1073/pnas.1408792111) und
[Horowitz et al.](https://doi.org/10.1126/scitranslmed.3006702).
Die Ergebnisassets stammen unverändert aus dem gespeicherten Stand vom
9. September 2026; ihre Herkunft dokumentiert `data/provenance.json`.
