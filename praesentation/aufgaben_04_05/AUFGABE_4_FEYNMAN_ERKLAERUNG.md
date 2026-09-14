# Aufgabe 4 – die gesamte Analyse mit der Feynman-Methode erklärt

Diese Notiz erklärt die fünf Notebooks so, dass man jeden Schritt in eigenen
Worten wiedergeben kann. Der Schwerpunkt liegt auf der linearen
Single-Cell-SVM. Danach folgen CellCNN, Citrus und der gemeinsame Vergleich.

Die Feynman-Idee ist einfach:

1. Erkläre das Problem ohne Fachsprache.
2. Zerlege jeden Rechenschritt in kleine, überprüfbare Teile.
3. Benenne Stellen, an denen Information unzulässig „in die Zukunft“ fließen
   könnte.
4. Prüfe am Ende, ob du den Ablauf ohne Notebook erklären könntest.

## 1. Das Problem in einem Bild

Stell dir jeden Spender als einen großen Beutel mit sehr vielen Karten vor. Eine
Karte entspricht einer Zelle. Auf jeder Karte stehen 37 Messwerte – die Marker.
Auf dem Beutel steht genau ein Zielwert:

- `0`: CMV−
- `1`: CMV+

Wir besitzen 20 Beutel: 11 CMV− und 9 CMV+. Insgesamt enthält das Haupt-Gate
`gated_alive` 3.438.750 Zellen.

Die entscheidende Frage lautet nicht: „Ist diese einzelne Zelle CMV+?“ Eine
einzelne Zelle hat in diesen Daten gar kein unabhängig gemessenes CMV-Label.
Die Frage lautet vielmehr: „Kann das Muster in allen Zellen eines bisher
ungesehenen Spenders vorhersagen, ob dieser Spender CMV+ oder CMV− ist?“

### Warum wir niemals Zellen zufällig auf Train und Test verteilen

Zellen desselben Spenders ähneln einander. Würden einige Zellen eines Spenders
im Training und andere im Test landen, könnte ein Modell spender-spezifische
Eigenheiten wiedererkennen. Das sähe wie gute Vorhersage aus, wäre aber keine
Verallgemeinerung auf neue Menschen.

Darum ist der **Spender** die unabhängige Beobachtung. Ein Spender liegt in
einem Split entweder vollständig im Training oder vollständig im Test.

## 2. Gemeinsame Datenverarbeitung aller Methoden

### 2.1 Welches Gate wird verwendet?

Der papernähere Hauptvergleich benutzt `gated_alive`: bereits gegatete, lebende
und von Doubletten bereinigte PBMCs. `gated_NK` ist kleiner und wird nur für
schnelle technische Smoke-Tests benutzt.

Das ist wichtig: Wenn wir nur vorselektierte NK-Zellen als Hauptanalyse nähmen,
würden wir dem Modell bereits biologisches Wissen vorgeben. Der Hauptvergleich
soll aber auf dem breiteren `gated_alive`-Datensatz stattfinden.

### 2.2 Welche Spalten werden eingelesen?

Die FCS-Dateien werden nur gelesen und nie verändert. Aus jeder Datei werden
die 37 Marker aus `NK_markers.csv` anhand ihrer FCS-Kurznamen (`PnS`) und in
genau dieser Reihenfolge ausgewählt. Technische Kanäle wie `Time`,
`Cell_length`, `Dead`, `DNA1` und `DNA2` gehören nicht zu dieser Markerliste und
gehen nicht in die Klassifikation ein.

Das CellCNN-Paper spricht an dieser Stelle von 36 Markern. Die mitgelieferte
`NK_markers.csv` und das offizielle CellCNN-Beispiel verwenden für diesen
Datensatz jedoch übereinstimmend 37 Marker. Wir folgen deshalb der konkreten
Referenzimplementierung und der gelieferten Markerliste; dies ist eine bewusst
dokumentierte Abweichung von der Zahl im Papertext.

Die Loader prüfen unter anderem:

- genau 20 eindeutige Spender,
- Übereinstimmung von FCS-Dateien und Labeltabelle,
- genau 37 eindeutige Marker,
- identische Marker in allen Dateien,
- erwartete Zellzahlen,
- ausschließlich endliche Messwerte nach der Transformation.

`source="raw"` bedeutet hier: FlowKit soll beim Laden keine versteckte
Transformation oder Kompensation anwenden. Ein bekannter falscher FCS-Offset
wird kontrolliert toleriert (`ignore_offset_error=True`); die Dateien bleiben
unverändert.

### 2.3 Warum `arcsinh(x / 5)`?

Für jeden Rohwert `x` wird berechnet:

```text
x_transformed = arcsinh(x / 5)
```

Die ArcSinh-Funktion verhält sich nahe null ungefähr linear und für große
Beträge ungefähr logarithmisch. Anschaulich komprimiert sie sehr große
Messwerte, ohne kleine oder negative Werte unbrauchbar zu machen. Der Kofaktor
5 übernimmt die papernahe Skala.

Wichtig ist die Unterscheidung:

- Die ArcSinh-Transformation hat einen vorher festgelegten Kofaktor. Sie lernt
  nichts aus Testdaten und darf deshalb direkt auf alle Spender angewendet
  werden.
- Mittelwert und Standardabweichung einer Standardisierung werden aus Daten
  gelernt. Sie dürfen deshalb ausschließlich aus den jeweiligen
  Trainingsspendern stammen.

### 2.4 Warum werden Spender gleich gewichtet?

Ein Spender mit 300.000 Zellen darf nicht automatisch zehnmal wichtiger sein
als einer mit 30.000 Zellen. Darum ziehen SVM, CellCNN und Citrus für das Lernen
pro Trainingsspender dieselbe Anzahl von Zellen beziehungsweise Ereignissen.

Für Validierung und Vorhersage unterscheiden sich die Methoden anschließend in
ihrer paperbedingten Aggregation. Das Ziel bleibt aber immer genau ein Score pro
Spender.

## 3. Notebook 04a: Daten, QC und gemeinsame Splits

Datei: `notebooks/04a_data_qc_and_splits.ipynb`

Dieses Notebook beantwortet zwei Fragen:

1. Sind die gelieferten Daten technisch plausibel und konsistent?
2. Welche Spender verwendet jede Methode in jedem Train-/Test-Split?

### 3.1 Labels, Marker und Dateien zusammenführen

Aus der Labeldatei wird aus dem Dateinamen die Spender-ID gebildet. Danach
werden Labeltabelle und tatsächlich vorhandene FCS-Dateien eins zu eins über
die Spender-ID verbunden. Ein fehlender, doppelter oder zusätzlicher Spender
führt zu einem Fehler statt zu einer stillen Annahme.

Ergebnis:

- 20 Spender,
- 11 CMV−,
- 9 CMV+,
- 37 Analysemarker.

### 3.2 Eine kleine sichtbare Datenprobe

Für den Spender `a_001` zeigt das Notebook die ersten fünf Zellen für sechs
ausgewählte Marker einmal roh und einmal nach `arcsinh(x / 5)`. Diese Tabelle
ist eine einfache Plausibilitätskontrolle: Wir sehen unmittelbar, ob die
richtigen Spalten gelesen und sinnvoll transformiert wurden.

### 3.3 Standard-QC statt unnötiger Rohzytometrie-Normalisierung

Der Datensatz ist bereits vorverarbeitet und gegatet. Deshalb wird keine
nachträgliche bead-basierte Normalisierung erfunden. Stattdessen prüfen wir die
Punkte, die mit den gelieferten Daten tatsächlich überprüfbar sind:

- Zellzahl je Spender,
- Anzahl und Konsistenz der Marker,
- fehlende oder nicht-endliche Werte,
- Verteilungen der transformierten Marker,
- spenderweise Marker-Mediane,
- Messdatum und Instrument als mögliche technische Störgrößen.

Für kompakte Verteilungsplots wird mit Seed 12.345 eine ausgewogene
QC-Stichprobe von 2.000 Zellen je Spender gezogen, also 40.000 Zellen. Diese
Stichprobe verändert die Originaldaten nicht. Ausreißer werden in einzelnen
Boxplots nur optisch ausgeblendet, nicht aus den Daten entfernt.

Die robust standardisierten spenderweisen Mediane dienen nur der
Visualisierung. Sie fließen nicht als gelernte Transformation in die drei
Klassifikatoren ein.

Die rationale Entscheidung nach dem QC lautet: keine Zellen oder Spender
zusätzlich entfernen und keine weitere globale Normalisierung einführen.
Methodenspezifische Standardisierung wird später splitweise und nur auf
Trainingsspendern gelernt.

### 3.4 Die 100 gemeinsamen äußeren Splits

Mit Master-Seed 12.345 werden 100 reproduzierbare Monte-Carlo-Splits erzeugt.
Jeder Split enthält:

- Training: 7 CMV− + 7 CMV+ = 14 Spender,
- Test: 4 CMV− + 2 CMV+ = 6 Spender.

Das Verhältnis im Test ergibt sich aus den insgesamt vorhandenen 11 negativen
und 9 positiven Spendern. Innerhalb der 14 Trainingsspender wird zusätzlich
eine geschichtete 3-fache Kreuzvalidierung erzeugt. Jeder Trainingsspender ist
in genau einem inneren Fold Validierungsspender. Äußere Testspender erhalten
`inner_fold = -1`.

Die Datei `task4_donor_splits.csv` ist anschließend für alle drei Methoden die
einzige Wahrheit über die Split-Zuordnung. Damit werden Methoden tatsächlich
auf denselben Testpersonen verglichen.

## 4. Notebook 04b: lineare Single-Cell-SVM – ganz genau

Datei: `notebooks/04b_svm.ipynb`

### 4.1 Die Grundidee ohne Formel

Die SVM sieht beim Training einzelne Zellen. Weil nur der Spender ein Label
besitzt, erhält jede gezogene Zelle vorübergehend das Label ihres Spenders.
Eine Zelle aus einem CMV+-Spender wird also als positiv behandelt, obwohl nicht
jede einzelne Zelle ein CMV-Signal tragen muss.

Das ist bewusst eine **schwache Zellbeschriftung**. Die Hoffnung lautet: Einige
Zellen aus positiven Spendern tragen ein wiederkehrendes Muster. Darum wird am
Ende nicht der mittlere Score aller Zellen verwendet, sondern der Mittelwert
der auffälligsten 1 %.

### 4.2 Schritt 1: Modus und Daten festlegen

Im Smoke-Modus werden für zwei Splits `gated_NK` und 2.000 Trainingszellen je
Spender verwendet. Im vollständigen Lauf sind es 100 Splits, `gated_alive` und
10.000 Trainingszellen je Spender.

Die Kandidaten für den SVM-Hyperparameter sind:

```text
C ∈ {0.01, 0.1, 1.0}
```

Der Anteil für die spätere Spenderaggregation ist fest `1 %`.

### 4.3 Schritt 2: Alle FCS-Dateien spenderweise laden

Für jeden Spender entsteht eine Matrix:

```text
Anzahl Zellen des Spenders × 37 Marker
```

Die 37 Rohwerte jeder Zelle werden mit `arcsinh(x / 5)` transformiert. Die
Matrizen bleiben in `data_by_donor` getrennt. Diese Trennung erschwert es,
versehentlich Zellen eines Spenders in mehrere Partitionen zu verteilen.

### 4.4 Schritt 3: Einen äußeren Split nehmen

Betrachten wir beispielhaft Split 0. Seine 14 Trainingsspender sind zur
Modellauswahl da. Die Daten der 6 Testspender werden zwar technisch eingelesen
und waren Teil des globalen, unüberwachten QC, beeinflussen aber weder Scaler,
Modellfit, Hyperparameterwahl noch Entscheidungsschwelle.

Innerhalb der 14 Trainingsspender gibt es drei innere Folds. Für einen inneren
Durchlauf gilt:

- zwei Folds: inneres Training,
- ein Fold: innere Validierung,
- äußere 6 Testspender: vollständig unberührt.

Der Code prüft, dass sich diese drei Spendergruppen nicht überschneiden.

### 4.5 Schritt 4: Gleich viele Trainingszellen je Spender ziehen

Aus jedem inneren Trainingsspender werden im vollständigen Lauf genau 10.000
Zellen **ohne Zurücklegen** gezogen. Dadurch hat jeder Trainingsspender das
gleiche Gewicht.

Alle Zellen eines inneren Validierungsspenders bleiben dagegen erhalten. Sie
werden nicht zum Fitten benutzt, sondern später nur bewertet.

Warum nicht alle Trainingszellen verwenden? Erstens würde ein zellreicher
Spender stärker gewichtet. Zweitens wäre die Rechnung wesentlich größer, ohne
dass dadurch mehr unabhängige Personen hinzukämen.

### 4.6 Schritt 5: Marker nur mit inneren Trainingsdaten standardisieren

Nach ArcSinh können Marker sehr unterschiedliche Wertebereiche besitzen. Eine
lineare SVM reagiert auf solche Skalen. Daher wird pro Marker standardisiert:

```text
z = (x - Mittelwert_Training) / Standardabweichung_Training
```

`StandardScaler.fit(...)` sieht ausschließlich die gezogenen Zellen der
inneren Trainingsspender. Dieselben gelernten Mittelwerte und
Standardabweichungen werden danach unverändert auf die vollständigen Zellen
der inneren Validierungsspender angewendet.

Das ist ein zentraler Schutz vor Leakage. Würde der Scaler auch
Validierungs- oder Testspender sehen, hätte die Trainingspipeline bereits
Information über deren Markerlagen.

### 4.7 Schritt 6: Was lernt die lineare SVM?

Für eine standardisierte Zelle mit 37 Werten `z` berechnet die SVM:

```text
s(z) = w₁z₁ + w₂z₂ + ... + w₃₇z₃₇ + b
```

`w` sind die gelernten Markergewichte, `b` ist der Achsenabschnitt und `s` ist
der Abstandsscore zur Trennfläche. Ein größerer Score spricht stärker für die
positive Klasse. Der Score ist keine kalibrierte Wahrscheinlichkeit.

`C` steuert den Kompromiss:

- kleines `C`: stärkere Regularisierung, breitere Fehlertoleranz,
- großes `C`: Trainingsfehler werden härter bestraft, das Modell kann sich
  stärker an die Trainingsdaten anpassen.

Für jedes `C` und jeden der drei inneren Folds wird eine eigene `LinearSVC`
trainiert. Konkret nutzt sie `dual="auto"`, eine L2-Strafe und den
standardmäßigen `squared_hinge`-Verlust. Es wird keine zusätzliche
Klassengewichtung gesetzt. `random_state` wird reproduzierbar aus Split-Seed
und innerem Fold abgeleitet; `max_iter=10.000` gibt dem Optimierer genug
Schritte.

### 4.8 Schritt 7: Zell-Scores zu einem Spender-Score zusammenfassen

Die trainierte SVM berechnet für jede Zelle eines Validierungsspenders einen
Score. Bei `n` Zellen werden

```text
k = max(1, ceil(0.01 × n))
```

Zellen ausgewählt: die `k` Zellen mit den höchsten Scores. Deren Mittelwert ist
der Spender-Score.

Kleines Beispiel: Ein Spender habe 1.000 Zellen. Dann sind `k = 10`. Wenn die
zehn auffälligsten Zellen zu einer biologisch relevanten Unterpopulation
gehören, die übrigen 990 aber unauffällig sind, würde ein Mittelwert über alle
1.000 Zellen das Signal stark verdünnen. Die Top-1-%-Aggregation fragt
gezielter: „Gibt es eine kleine, besonders positiv aussehende Teilpopulation?“

Die Aggregation ist fest vorgegeben und wird nicht anhand der Testdaten
optimiert.

### 4.9 Schritt 8: Das beste `C` nur in der inneren CV wählen

Für jeden inneren Fold entstehen Spender-Scores für die dort zurückgehaltenen
Validierungsspender. Aus diesen Scores wird die ROC-AUC auf Spender-Ebene
berechnet.

Für jedes `C` werden die drei inneren ROC-AUC-Werte gemittelt. Gewählt wird das
`C` mit der größten mittleren ROC-AUC. Bei Gleichstand gewinnt das kleinere
`C`, also die stärker regularisierte, einfachere Variante.

Wichtig: Die vielen Zell-Scores sind nicht viele unabhängige Testfälle. Die
ROC-AUC wird aus den wenigen **Spender-Scores** berechnet.

### 4.10 Schritt 9: Den Entscheidungsschwellenwert lernen

ROC-AUC bewertet die Rangfolge und benötigt keinen festen Schwellenwert. Für
Balanced Accuracy und ein binäres `y_pred` brauchen wir aber einen.

Für das gewählte `C` werden alle out-of-fold Spender-Scores der inneren CV
zusammengenommen. Jeder dieser Scores stammt von einem Modell, das den
betreffenden Spender nicht trainiert hat. Auf diesen Scores wird der endliche
Schwellenwert gewählt, der den Youden-Index maximiert:

```text
Youden J = Sensitivität - Falsch-Positiv-Rate
         = Sensitivität + Spezifität - 1
```

Auch dieser Schwellenwert sieht keinen äußeren Testspender.

### 4.11 Schritt 10: Das äußere Modell neu fitten

Nun sind `C` und Schwellenwert fest. Aus jedem der 14 äußeren
Trainingsspender werden erneut gleich viele Zellen gezogen. Auf genau diesen
Zellen wird ein neuer `StandardScaler` gefittet und danach eine neue lineare
SVM mit dem gewählten `C` trainiert.

Das ist der endgültige Fit für diesen äußeren Split. Anders als während der
inneren CV darf er alle 14 äußeren Trainingsspender verwenden. Die sechs
Testspender bleiben weiterhin unsichtbar.

### 4.12 Schritt 11: Die sechs Testspender vorhersagen

Für jeden Testspender geschieht nun:

1. Alle seine ArcSinh-transformierten Zellen werden mit dem **äußeren
   Trainings-Scaler** standardisiert.
2. Die SVM berechnet einen Score für jede Zelle.
3. Der Mittelwert der höchsten 1 % wird zum Spender-Score.
4. `score >= gelernter Schwellenwert` ergibt `y_pred = 1`, andernfalls `0`.

Erst jetzt wird das echte Spenderlabel zur Bewertung herangezogen.

### 4.13 Schritt 12: Ergebnisse speichern und fortsetzen können

Pro Testspender werden unter anderem Split, Seed, Label, Score,
Schwellenwert, Vorhersage, gewähltes `C`, Top-Anteil und Zahl verwendeter
Trainingszellen gespeichert. Eine zweite Tabelle dokumentiert für jede
Kombination aus `C` und innerem Fold die Validierungs-ROC-AUC. Die einzelnen
inneren Out-of-fold-Spender-Scores werden für die Schwellenwahl im Speicher
verwendet, aber nicht als eigene CSV-Tabelle abgelegt.

Nach jedem abgeschlossenen Split werden die CSV-Dateien aktualisiert. Bei einem
erneuten Start erkennt das Notebook bereits vollständige Splits und überspringt
sie. Dadurch muss ein langer Lauf nach einer Unterbrechung nicht von vorne
beginnen.

### 4.14 Die SVM als Mini-Pseudocode

```text
für jeden äußeren Split:
    halte 6 ganze Spender als Test zurück

    für C in [0.01, 0.1, 1.0]:
        für jeden der 3 inneren Folds:
            ziehe gleich viele Zellen je innerem Trainingsspender
            fitte Scaler nur dort
            trainiere lineare SVM nur dort
            score alle Zellen der inneren Validierungsspender
            bilde je Spender den Mittelwert der höchsten 1 %
            berechne spenderweise ROC-AUC

    wähle C mit bester mittlerer innerer ROC-AUC
    lerne Youden-Schwelle aus out-of-fold Trainingsspender-Scores

    ziehe gleich viele Zellen aus allen 14 Trainingsspendern
    fitte neuen Scaler und neue SVM auf diesen 14 Spendern
    score alle Zellen jedes der 6 Testspender
    aggregiere wieder die höchsten 1 %
    speichere genau eine Vorhersage je Testspender
```

### 4.15 Leakage-Check für die SVM

Für einen sauberen Split müssen alle Antworten „ja“ sein:

- Sind Spender statt Zellen getrennt? Ja.
- Sieht der innere Scaler nur innere Trainingsspender? Ja.
- Sieht die Wahl von `C` nur innere Validierungsspender? Ja.
- Sieht die Youden-Schwelle nur out-of-fold Trainingsspender? Ja.
- Sieht der äußere Scaler nur die 14 äußeren Trainingsspender? Ja.
- Werden Testlabels erst bei der Metrik verwendet? Ja.
- Werden alle Spender durch gleich große Trainingsstichproben gewichtet? Ja.

## 5. Notebook 04c: CellCNN

Datei: `notebooks/04c_cellcnn.ipynb`

### 5.1 Die Grundidee

Die SVM bewertet jede Zelle mit genau einer linearen Regel. CellCNN lernt
mehrere kleine „Detektoren“, im Notebook Filter genannt. Jeder Filter sucht
nach einem Markerprofil. Danach fragt das Netz pro Filter, ob die stärksten
1 % der Zellen dieses Profil zeigen.

### 5.2 Gemeinsame Vorverarbeitung

Auch hier werden dieselben 37 Marker geladen und `arcsinh(x / 5)` angewendet.
Für jeden Kandidaten wird ein eigener `StandardScaler` ausschließlich aus den
inneren Trainingsspendern gelernt.

Für den Scaler werden im vollständigen Lauf 20.000 Zellen ohne Zurücklegen je
Trainingsspender gezogen. Das hält die Spendergewichte gleich.

### 5.3 Aus Spendern werden Multi-Cell-Inputs

Ein Netzbeispiel ist keine einzelne Zelle, sondern ein kleines künstliches
Päckchen aus Zellen eines Spenders. Im vollständigen Lauf enthält ein solches
Päckchen 3.000 Zellen. Pro Trainingsspender werden 200 Päckchen erzeugt.

Die Zellen werden dabei mit Zurücklegen gezogen. Mehrere Päckchen desselben
Spenders überlappen also möglicherweise. Sie sind Datenaugmentation, keine
neuen unabhängigen Personen.

Die Päckchen werden einmal deterministisch materialisiert und als Tensoren auf
CPU oder GPU gehalten. Die GPU ändert nicht die Methode; sie beschleunigt nur
die Matrixoperationen.

### 5.4 Was geschieht im Netz?

Für jede Zelle und jeden Filter wird zunächst eine lineare Antwort berechnet:

```text
Filterantwort = ReLU(Markerwerte · Filtergewichte + Filterbias)
```

`ReLU` setzt negative Antworten auf null. Ein Filter „feuert“ somit nur auf
Zellen, die zu seinem gelernten Profil passen.

Für jeden Filter werden anschließend die höchsten 1 % der Antworten innerhalb
des 3.000-Zellen-Päckchens gemittelt. Das sind 30 Zellen. Aus jedem Filter wird
also genau eine zusammengefasste Zahl. Die Ausgabeschicht verbindet diese
Zahlen zu zwei Logits; Softmax macht daraus einen CMV+-Score zwischen 0 und 1.

### 5.5 Training und Regularisierung

Im vollständigen Lauf werden 3, 4 oder 5 Filter ausprobiert. Für jeden der drei
inneren Folds ergibt das 9 Kandidatenmodelle pro äußerem Split.

Trainiert wird mit:

- Adam-Optimierer,
- Lernrate `0.01`,
- Batchgröße `128`,
- höchstens 100 Epochen,
- L2-Strafe `1e-4` auf Filter- und Ausgabegewichte,
- Early Stopping nach 5 Epochen ohne relevante Verbesserung.

Der beste Parameterzustand nach innerem Validierungsverlust wird behalten.

### 5.6 Kandidatenauswahl und wichtiger Unterschied zur SVM

Jedes Kandidatenmodell sagt seine inneren Validierungsspender vorher. Sortiert
wird nach:

1. höchster Validierungsgenauigkeit bei Schwelle 0,5,
2. dann höchster Validierungs-ROC-AUC,
3. dann niedrigstem Validierungsverlust,
4. dann weniger Filtern,
5. dann kleinerer Fold-Nummer.

Das ausgewählte **einzelne innere Modell** wird direkt auf die äußeren
Testspender angewendet. Je nach Größe seines Validierungsfolds wurde es nur auf
9 oder 10 Spendern der beiden anderen inneren Folds trainiert; die übrigen 4
oder 5 äußeren Trainingsspender dienten diesem Kandidaten als Validierung. Es
wird nicht noch einmal auf allen 14 äußeren Trainingsspendern gefittet. Das
folgt hier bewusst dem Vorgehen der CellCNN-Referenz, unterscheidet sich aber
von der SVM.

### 5.7 Vorhersage eines Spenders

Für einen Spender werden fünf getrennte Zufallsstichproben mit jeweils bis zu
20.000 Zellen gezogen. Innerhalb einer Stichprobe erfolgt die Ziehung ohne
Zurücklegen; zwischen den fünf Stichproben dürfen dieselben Zellen erneut
vorkommen. Das ausgewählte Netz gibt für jedes Päckchen einen Softmax-basierten
CMV+-Score aus; der Mittelwert der fünf Scores ist der Spender-Score. Wir
nennen ihn nicht kalibrierte Wahrscheinlichkeit, weil keine Kalibrierung
geprüft wurde. Bei `score >= 0.5` lautet die Klassenentscheidung CMV+.

Die fünf Vorhersage-Päckchen sind eine dokumentierte Stabilisierung gegenüber
einer einzelnen zufälligen Teststichprobe im Paper.

### 5.8 Was wird für die Interpretation aufgehoben?

Zusätzlich zu Vorhersagen und Auswahlprotokoll speichert das Notebook:

- Filtergewichte je Marker,
- Filterbias,
- Unterschied der beiden Ausgabeschicht-Gewichte,
- Mittelwert und Skala des zugehörigen Trainings-Scalers.

Damit kann später nachvollzogen werden, welche Markerprofile ein Filter sucht
und ob er eher in Richtung CMV+ oder CMV− wirkt.

## 6. Notebook 04d: Citrus mit der originalen R-Plattform

Datei: `notebooks/04d_citrus.ipynb`

### 6.1 Die Grundidee

Citrus geht nicht direkt von einzelnen Zellen zur Klasse. Es bildet zuerst
Zellgruppen, also Cluster. Danach beschreibt es jeden Spender durch die
Häufigkeit dieser Cluster und sucht Clusterhäufigkeiten, die CMV+ und CMV−
unterscheiden.

Das Notebook läuft deshalb in der separaten R-Umgebung `ssbi-citrus` mit dem
originalen Nolan-Lab-Paket `citrus` und `Rclusterpp`.

### 6.2 Daten einlesen und fair samplen

Die 37 Marker werden auch in R anhand ihrer Kanalbeschreibungen ausgewählt und
mit Kofaktor 5 transformiert. Im vollständigen Lauf zieht Citrus 1.000
Ereignisse je Spender. Somit trägt jeder Spender gleich viele Zellen zur
Clusterbildung bei.

### 6.3 Cluster ausschließlich aus Trainingsdaten bilden

Für jeden äußeren Split werden 14 Trainings- und 6 Testspender getrennt. Citrus
führt auf den Trainingszellen hierarchisches Clustering durch. Die Testspender
dürfen die Clustergrenzen nicht mitbestimmen.

Für die innere Validierung wird aus den bereits gemeinsamen Fold-Zuordnungen
ein Citrus-kompatibles Fold-Objekt gebaut. Dafür werden offizielle
Citrus-Funktionen zur Clusterbildung und Abbildung auf den Clusterraum benutzt.
So verwendet Citrus exakt dieselben inneren Spender-Folds wie die beiden
anderen Methoden.

### 6.4 Aus Clustern werden Spendermerkmale

Für jeden Spender wird berechnet, welcher Anteil seiner Zellen in jedem Cluster
liegt. Diese Anteile sind die „abundance features“.

Citrus erzeugt hierarchisch verschachtelte Cluster. Ein Zellereignis kann daher
zu einem Kindcluster und zugleich zu dessen Elterncluster beitragen. Die
Abundance-Werte verschiedener Cluster sind folglich nicht wie disjunkte
Kuchenstücke zu verstehen und müssen sich nicht zu 100 % summieren.

Nur Cluster mit mindestens `0.05` Anteil werden betrachtet. In der Citrus-API
ist dieser Wert ein Anteil und bedeutet hier 5 % der geclusterten Ereignisse,
nicht 0,05 %.

Anschaulich wird aus einem riesigen Zellbeutel nun eine kurze Tabelle:

```text
Spender A: Cluster 1 = 12 %, Cluster 2 = 4 %, ...
Spender B: Cluster 1 =  6 %, Cluster 2 = 9 %, ...
```

### 6.5 Welche Cluster sagen CMV voraus?

Auf den Clusterhäufigkeiten wird eine logistische `glmnet`-Klassifikation mit
L1-Regularisierung trainiert. L1 kann Koeffizienten exakt auf null setzen und
damit eine kleine Zahl informativer Cluster auswählen.

Die Regularisierungsstärke `lambda` wird über die fest vorgegebenen inneren
Spender-Folds bestimmt. `cv.min` bezeichnet den Wert mit minimalem innerem
Klassifikationsfehler.

### 6.6 Testspender abbilden

Die sechs Testspender werden nicht neu geclustert. Ihre gesampelten Zellen
werden in den bereits aus den Trainingsspendern gelernten Clusterraum
abgebildet. Danach werden dieselben Clusterhäufigkeiten berechnet und vom
finalen `glmnet`-Modell in einen CMV+-Score zwischen 0 und 1 übersetzt.

Nach der inneren Wahl von `lambda` werden das finale Clustering und das finale
`glmnet`-Modell auf allen 14 äußeren Trainingsspendern verwendet. Darin ähnelt
Citrus der SVM und unterscheidet sich vom hier papernah direkt ausgewählten
CellCNN-Kandidaten.

Die feste Klassenschwelle ist 0,5. Ausgewählte Cluster, ihre Koeffizienten und
ihre 37-dimensionalen Zentroidprofile werden für die Interpretation
gespeichert.

### 6.7 Numerische Reproduzierbarkeit

Citrus-Scores werden vor dem Speichern auf 15 Nachkommastellen gerundet. Sehr
kleine numerische Abweichungen aus parallelen Bibliotheken dürfen bis zu einer
strengen Toleranz auftreten, sollen aber die inhaltliche Vorhersage nicht
ändern. Analyse-Seeds, Stichprobengröße und Clustergrenze werden mitgespeichert.

## 7. Notebook 04e: fairer Vergleich

Datei: `notebooks/04e_comparison.ipynb`

Dieses Notebook trainiert keine Hauptmethode neu. Es liest die vollständigen
Ergebnisdateien ein und prüft zuerst, dass wirklich Gleiches mit Gleichem
verglichen wird:

- jede Methode hat 100 Splits,
- jeder Split hat 6 Testspender,
- also 600 Vorhersagen je Methode,
- Gate und Modus sind `gated_alive` und `full`,
- Split-ID, Seed, Spender-ID und wahres Label stimmen exakt mit der gemeinsamen
  Splittabelle überein,
- Scores und Schwellenwerte sind endlich,
- `y_pred` passt rechnerisch zu Score und Schwelle.

### 7.1 Drei Metriken pro Split

Für jede Methode und jeden Split werden auf den sechs Testspendern berechnet:

**ROC-AUC:** Wie oft erhält ein zufälliger CMV+-Spender einen höheren Score als
ein zufälliger CMV−-Spender? 0,5 entspricht zufälliger Rangfolge, 1 perfekter
Rangfolge.

**PR-AUC:** Fläche unter der Precision-Recall-Kurve, im Notebook trapezförmig
integriert. Das ist ausdrücklich nicht zwingend identisch mit „Average
Precision“.

**Balanced Accuracy:**

```text
(Sensitivität + Spezifität) / 2
```

Sie gewichtet beide Klassen gleich, obwohl ein Testsplit 4 negative und nur 2
positive Spender enthält.

Über die 100 Splits werden Median, erstes Quartil, drittes Quartil, Mittelwert
und Standardabweichung berichtet. Der Median ist bei nur sechs Testspendern pro
Split robuster als ein einzelner Lauf.

### 7.2 Warum gepaarte Differenzen?

Ein Split kann zufällig leichter oder schwerer sein. Darum wird nicht nur der
Median von Methode A mit dem Median von Methode B verglichen. Für jeden
identischen Split wird direkt gerechnet:

```text
ROC-AUC(CellCNN) - ROC-AUC(Citrus)
ROC-AUC(CellCNN) - ROC-AUC(SVM)
```

Damit wird die Schwierigkeit des jeweiligen Testsets kontrolliert.

### 7.3 Technische Batch-Kontrolle

Messdatum und Instrument könnten unbeabsichtigt mit dem CMV-Label
zusammenhängen. Deshalb wird pro Split und technischer Variable ein bewusst
einfacher Kontrollprädiktor gebaut:

1. Nur in den 14 Trainingsspendern wird pro Datum beziehungsweise Instrument
   die positive Labelrate berechnet.
2. Diese Rate wird als Score auf passende Kategorien der Testspender
   übertragen.
3. Unbekannte Kategorien erhalten die mittlere Trainingsrate.
4. Auf den Testspendern wird ROC-AUC berechnet.

Auch diese Kontrolle ist train-only. Die beobachteten medianen ROC-AUC-Werte
von 0,250 für Messdatum und 0,375 für Instrument liefern keine stabile
übertragbare Erklärung des CMV-Labels.

### 7.4 Ergebnis des vollständigen Vergleichs

Die Medianwerte über 100 identische Splits sind:

| Methode | ROC-AUC | PR-AUC | Balanced Accuracy |
|---|---:|---:|---:|
| CellCNN | 0,875 | 0,792 | 0,750 |
| Lineare Single-Cell-SVM | 0,875 | 0,792 | 0,625 |
| Citrus | 0,625 | 0,417 | 0,500 |

Die mediane gepaarte ROC-AUC-Differenz beträgt `CellCNN − Citrus = 0,25` und
`CellCNN − SVM = 0`. Das zeigt in diesen Splits eine ähnliche Rangordnung von
CellCNN und SVM, aber eine bessere schwellenabhängige Balanced Accuracy von
CellCNN. Wegen nur 20 Spendern und überlappender Monte-Carlo-Splits sind dies
keine 100 unabhängigen Studien.

## 8. Warum ein Notebook-Neustart nicht automatisch alles neu rechnet

Die rechenintensiven Notebooks besitzen drei praktische Steuerungen:

- `TASK4_RUN_MODE=smoke`: kleiner technischer Lauf.
- `TASK4_RUN_MODE=full`: vollständiger Hauptlauf.
- `TASK4_RUN_TRAINING=0`: vorhandene Ergebnisdateien laden, nicht trainieren.

Bei aktivem Training werden bereits abgeschlossene Splits aus den lokalen CSVs
erkannt und übersprungen. Darum kann ein Notebook sehr schnell durchlaufen,
obwohl das ursprüngliche Training lange dauerte. Schnell bedeutet dann: Die
gespeicherten Ergebnisse wurden geprüft und dargestellt – nicht: Alle Modelle
wurden in Sekunden neu trainiert.

Die Wiederverwendung wird methodenspezifisch geprüft. Beim CellCNN müssen bei
aktivem Training zusätzlich Gerätetyp (`cpu` oder `cuda`) und
Implementierungsversion zum vorhandenen Ergebnis passen; sonst wird dieser
Cache nicht als kompatibel übernommen.

Die Ergebnisdateien unter `results/tables/` und Abbildungen unter
`results/figures/` sind Git-ignoriert. Ein frischer Clone besitzt diese
Laufergebnisse nicht und muss sie neu erzeugen oder separat erhalten.

## 9. Der gesamte Ablauf in einem Satz pro Notebook

1. **04a:** Daten plausibilisieren und eine einzige spenderweise Split-Tabelle
   für alle Methoden erzeugen.
2. **04b:** Zellweise lineare SVM trainieren, ihre stärksten 1-%-Zellsignale zu
   Spender-Scores bündeln und sauber verschachtelt validieren.
3. **04c:** Mehrere lernbare Zellfilter auf Multi-Cell-Inputs trainieren und
   seltene starke Filterantworten zu Spenderwahrscheinlichkeiten bündeln.
4. **04d:** Trainingszellen clustern, Clusterhäufigkeiten je Spender bilden und
   mit L1-logistischer Regression informative Cluster auswählen.
5. **04e:** Alle drei Methoden auf exakt denselben Testspendern mit identischen
   Metriken und gepaarten Differenzen vergleichen.

## 10. Feynman-Selbsttest

Wenn du die folgenden Fragen ohne Nachschlagen beantworten kannst, hast du den
Ablauf verstanden.

### Frage 1: Warum ist eine Million Zellen nicht gleich einer Million unabhängiger Fälle?

Weil das Zielwert-Label am Spender hängt. Zellen desselben Spenders teilen
biologische und technische Eigenschaften. Die unabhängigen Einheiten sind 20
Spender, nicht 3,4 Millionen Zellen.

### Frage 2: Warum darf ArcSinh auf alle Daten angewendet werden, der Scaler aber nicht global gefittet werden?

ArcSinh benutzt nur die vorher festgelegte Formel und den festen Kofaktor 5.
Der Scaler schätzt Mittelwert und Standardabweichung aus beobachteten Daten. Ein
globaler Fit würde Validierungs- und Testinformationen in das Training tragen.

### Frage 3: Warum bekommt bei der SVM jede Zelle das Spenderlabel, obwohl das biologisch ungenau ist?

Es gibt kein Zelllabel. Die schwache Beschriftung macht ein Zellmodell trotzdem
trainierbar. Die Top-1-%-Aggregation erlaubt anschließend, dass nur eine kleine
Unterpopulation das Spenderlabel trägt.

### Frage 4: Woher kommen `C` und SVM-Schwellenwert?

Beide ausschließlich aus den 14 äußeren Trainingsspendern: `C` aus mittlerer
innerer Spender-ROC-AUC, die Schwelle aus den out-of-fold Spender-Scores des
gewählten `C` über den Youden-Index.

### Frage 5: Worin unterscheiden sich SVM und CellCNN konzeptionell?

Die SVM hat eine lineare Zell-Score-Regel und aggregiert danach die höchsten
Zell-Scores. CellCNN lernt mehrere Zellfilter gemeinsam mit einer nichtlinearen
ReLU- und Top-1-%-Pooling-Struktur sowie einer Ausgabeschicht.

### Frage 6: Worin unterscheidet sich Citrus?

Citrus entdeckt zuerst Zellcluster und macht aus deren Häufigkeiten
spenderweise Merkmale. Die Klassifikation erfolgt dann auf diesen
Clusterhäufigkeiten, nicht direkt auf jeder einzelnen Zelle.

### Frage 7: Warum werden gepaarte Split-Differenzen verwendet?

Weil alle Methoden denselben Split sehen. Die direkte Differenz innerhalb
desselben Splits entfernt einen Teil der Schwankung durch leichte oder schwere
Testspenderkombinationen.

## 11. Die drei wichtigsten Merksätze

1. **Getrennt wird immer nach Spendern, nie nach Zellen.**
2. **Alles Gelernte – Scaler, Hyperparameter, Cluster und Schwellenwerte – wird
   ohne Zugriff auf die äußeren Testspender bestimmt.**
3. **Jede Methode liefert am Ende genau einen Score pro Testspender, erst dann
   werden die Methoden verglichen.**
