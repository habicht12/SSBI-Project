# Aufgabe 5: Die Auswahl und Interpretation von Zellsubsets ausführlich erklärt

Diese Datei erklärt die Umsetzung von Aufgabe 5 im Stil der [Methodenerklärungen zu Aufgabe 4](AUFGABE_4_METHODENERKLAERUNGEN.md): vom gespeicherten Modell über eine quantitative Auswahlregel bis zur Darstellung der Zellsubsets, einschließlich Formeln, Entscheidungsbegründungen und Grenzen.

**Dokumentationsstand: 9. September 2026, ergänzt um das SVM-Markerprofil.** Grundlage ist die aktuelle Auswertung von **30 gemeinsamen äußeren Splits, IDs 0–29**, auf `gated_alive`. Maßgeblich sind das Aufgabe-5-Notebook, seine Implementierung und die gespeicherten `task5_paper_*`-Ergebnisse. Das ergänzende SVM-Markerprofil wird im Präsentationsexporter aus denselben gespeicherten Testmodellen berechnet.

Es wurden keine Modelle neu trainiert. Die bisherigen Analyseergebnisse bleiben unverändert; das ergänzende SVM-Markerprofil und die Präsentationsassets werden separat exportiert. Die beschriebenen Auswahlregeln sind nachvollziehbare Operationalisierungen; dass jede ausgewählte Zelle biologisch relevant ist, folgt daraus nicht.

## Quellen und Orientierung

- [Aufgabenstellung](../Group_projects_ssbi_2026.pdf), Aufgabe 5.
- [Notebook zur Interpretation](../notebooks/05_interpretation.ipynb).
- [Implementierung der Interpretation](../src/task5_interpretation.py).
- [Präsentationsexporter mit ergänzendem SVM-Markerprofil](../src/presentation_assets.py).
- [Datenaufbereitung und Projektionen aus Aufgabe 2](../src/task23_analysis.py).
- [CellCNN-Notebook](../notebooks/04c_cellcnn.ipynb), [Citrus-Notebook](../notebooks/04d_citrus.ipynb) und [SVM-Notebook](../notebooks/04b_svm.ipynb).
- [Lokales CellCNN-Paper](../CellCNN.pdf), insbesondere Seite 8, Abschnitt „NK-cell benchmark data set“; [Onlinefassung des Papers](https://www.nature.com/articles/ncomms14825).
- [Provenienz der aktuellen Auswertung](tables/task5_paper_provenance.json): Parameter, Quellen, Softwareversionen und Prüfsummen.

## Inhalt

1. [Fragestellung und gemeinsame Datenbasis](#1-fragestellung-und-gemeinsame-datenbasis)
2. [CellCNN: Vom Filter zur ausgewählten Zellpopulation](#2-cellcnn-vom-filter-zur-ausgewählten-zellpopulation)
3. [Citrus: Welche Cluster verwendet das Modell?](#3-citrus-welche-cluster-verwendet-das-modell)
4. [Wiederkehrende Subsets über mehrere Splits erkennen](#4-wiederkehrende-subsets-über-mehrere-splits-erkennen)
5. [SVM: Welche Einzelzellen gehen in die Entscheidung ein?](#5-svm-welche-einzelzellen-gehen-in-die-entscheidung-ein)
6. [Darstellung und vorhandene Ergebnisse](#6-darstellung-und-vorhandene-ergebnisse)
7. [Paper-Nähe, Aussagegrenzen und Nachvollziehbarkeit](#7-paper-nähe-aussagegrenzen-und-nachvollziehbarkeit)

---

# 1. Fragestellung und gemeinsame Datenbasis

## 1.1 Was verlangt Aufgabe 5?

Aufgabe 5 verlangt einen **quantitativen Score, mit dem sich das für die Klassifikatoren relevante Zellsubset bestimmen lässt**, und eine Visualisierung dieser Subsets mit einer Methode zur Dimensionsreduktion.

Aus Aufgabe 4 liegen bereits Modelle vor, die den CMV-Status eines Spenders vorhersagen. Jetzt untersuchen wir, auf welche Zellmuster diese Modelle ansprechen und welche davon bei unterschiedlichen Trainingsaufteilungen erneut auftreten.

Dabei unterscheiden wir drei Fragen:

- **Auswahl:** Welche Zellen beziehungsweise Cluster erfüllen die Auswahlregel eines Modells?
- **Richtung:** Unterstützt die zugehörige Modellkomponente eher CMV+ oder CMV−?
- **Wiederkehr:** Wie regelmäßig erscheint ein ähnliches Subset über mehrere Splits?

Die Modelle erhalten in Aufgabe 5 keine neuen Gewichte. Ihre gespeicherten Parameter werden ausgewertet und zusammengefasst.

**Warum so?** Die Interpretation soll die bereits bewerteten Klassifikatoren erklären. Ein neues Training würde andere Modelle erzeugen und damit den Bezug zu den Ergebnissen aus Aufgabe 4 verändern.

## 1.2 Was bedeutet „relevante Zelle“ hier?

Wir kennen den CMV-Status des **Spenders**, aber kein unabhängig gemessenes CMV-Relevanzlabel für jede einzelne Zelle.

„Positiv ausgewählt“ bedeutet deshalb eine Auswahl mit Bezug zur positiven Modellrichtung. Es bedeutet weder, dass diese Zelle selbst CMV-infiziert ist, noch, dass sie ausschließlich bei CMV+-Spendern vorkommt.

Die drei Verfahren liefern unterschiedliche Arten von Auswahl:

| Methode | Ausgangspunkt | Quantitative Auswahl |
|---|---|---|
| CellCNN | Antwort einer Zelle auf einen gelernten Filter | Antwort größer als die Hälfte des filterbezogenen Referenzmaximums |
| Citrus | Clusterhäufigkeit als Merkmal der logistischen Regression | Cluster mit wirksamem Regressionskoeffizienten |
| SVM | Linearer Score jeder Zelle | Höchste 1 % der Zell-Scores eines vollständigen Testspenders; Richtung relativ zur Spenderschwelle |

**Warum keine gemeinsame Schwelle für alle Methoden?** Die Größen haben verschiedene Bedeutungen und Skalen. Eine identische Zahl würde keine identische biologische Relevanz herstellen. Wir knüpfen deshalb an die Berechnung des jeweiligen Klassifikators an.

## 1.3 Welche Daten verwenden wir?

Die Interpretation verwendet die breite Zellpopulation `gated_alive` mit **20 Spendern und 37 Markern**. Die Messwerte werden wie in Aufgabe 4 transformiert:

$$
a_{ij}=\operatorname{arcsinh}(x_{ij}/5).
$$

Dabei ist $x_{ij}$ der ursprüngliche Wert des Markers $j$ in Zelle $i$ und $a_{ij}$ der transformierte Wert. Der Kofaktor 5 ist festgelegt.

Die verwendeten Mengen erfüllen unterschiedliche Aufgaben:

| Menge | Umfang | Verwendung |
|---|---|---|
| Vorhandene t-SNE-Karte | 500 Zellen je Spender, insgesamt 10.000 | Gemeinsamer Hintergrund und Darstellung einzelner Karten-Zellen |
| CellCNN-Referenz | 20.000 Zellen je tatsächlichem inneren Trainingsspender | Referenzmaximum, Selektionsschwelle und Subset-Zentroid |
| Citrus-Trainingsstichprobe aus Aufgabe 4 | 10.000 Zellen je äußerem Trainingsspender | Grundlage der bereits gespeicherten Clusterzentroiden |
| SVM-Testspender | Sämtliche `gated_alive`-Zellen des jeweiligen Testspenders | Exakte Bestimmung der höchsten 1 % |

**Die 10.000 Karten-Zellen sind damit keine gemeinsame Berechnungsgrundlage für alle Auswahlregeln.** Insbesondere bestimmt die SVM ihre höchsten 1 % auf der vollständigen Probe, bevor das Ergebnis auf die Karten-Zellen eingeschränkt wird.

## 1.4 Welche Splits werden berücksichtigt?

Wir verwenden die **30 gemeinsamen Splits 0–29** aus Aufgabe 4. Pro Split liegen 14 Spender im äußeren Training und sechs im äußeren Test. Die Spenderzuordnung ist für alle Methoden identisch.

Die Splits überlappen zwischen Wiederholungen. Innerhalb eines Splits bleiben die Spender vollständig getrennt. Ein Spender kann in einem anderen Split wieder im Training oder Test liegen.

Für CellCNN und SVM existieren auch Modelle aus weiteren Splits. Diese gehen in die aktuelle Aufgabe-5-Auswertung nicht ein.

**Warum so?** Unterschiedliche Mengen von Wiederholungen würden die Interpretation des Methodenvergleichs erschweren. Die gemeinsame Grundlage umfasst hier 30 vollständig gespeicherte Modelle je Methode.

## 1.5 Wie wird sichergestellt, dass Modelle und Karte zusammenpassen?

Die Implementierung prüft die Vollständigkeit der benötigten Modellexporte vor dem Einlesen der umfangreichen FCS-Daten. Fehlende Splits führen zu einem Fehler; sie werden nicht als erfolglose Auswahl behandelt.

Anschließend werden unter anderem Gate, Markerreihenfolge, Transformation und die Zuordnung der Zell-IDs kontrolliert. Eine Karten-Zelle ist durch Spender und ursprüngliche Ereignisnummer in der FCS-Datei identifizierbar.

Die Kartenstichprobe aus Aufgabe 2 lässt sich deterministisch rekonstruieren: 500 Zellen ohne Zurücklegen, Seed `42 + Index des alphabetisch sortierten Spenders`, anschließend nach Ereignisnummer sortiert. Die ursprünglichen t-SNE-Koordinaten werden übernommen.

Für die Modelle wird die ursprüngliche Zahlenverarbeitung beibehalten: zuerst Rohwerte in `float32`, dann `arcsinh(x/5)` und die gespeicherte Modellskalierung. Die explorativen Markerprofile der Karte werden wie in Aufgabe 2 in `float64` verarbeitet.

**Warum so?** Bereits kleine Änderungen der Skalierung oder Zellzuordnung könnten andere Zellen über eine Schwelle heben oder Punkte an falscher Stelle einfärben. Die Originaldateien bleiben unverändert.

# 2. CellCNN: Vom Filter zur ausgewählten Zellpopulation

## 2.1 Welche Modellparameter werden verwendet?

Pro äußerem Split verwenden wir genau das in Aufgabe 4 ausgewählte CellCNN-Modell. Gespeichert sind unter anderem seine Filtergewichte, Filter-Biases, Ausgangsgewichte, Skalierungsparameter und der ursprüngliche Kandidatenseed.

Das ausgewählte Modell wurde auf den Trainingsspendern eines inneren Folds gelernt. Da es nach der Auswahl nicht auf allen 14 äußeren Trainingsspendern neu trainiert wurde, besteht seine tatsächliche Trainingsmenge aus **neun oder zehn Spendern**.

**Warum ist das wichtig?** Die Referenz für die Filterantworten muss zu diesem konkreten Modell passen. Die übrigen äußeren Trainingsspender wurden bei seiner Auswahl zur inneren Validierung verwendet und gehören nicht zu seiner Referenzstichprobe.

## 2.2 Wie reagiert ein Filter auf eine Zelle?

Zuerst wird das ArcSinh-Markerprofil mit dem gespeicherten Scaler des Modells standardisiert:

$$
z_{ij}=\frac{a_{ij}-\mu_{sj}}{\sigma_{sj}}.
$$

Der Index $s$ bezeichnet den äußeren Split und damit das ausgewählte Modell. Mittelwerte $\mu_{sj}$ und Standardabweichungen $\sigma_{sj}$ werden in Aufgabe 5 nicht neu geschätzt.

Für Filter $f$ berechnen wir:

$$
r_{isf}=\max\left(0,\sum_{j=1}^{37}w_{sfj}z_{ij}+b_{sf}\right).
$$

Dabei bedeutet:

- $w_{sfj}$: Gewicht des Markers $j$ in Filter $f$.
- $b_{sf}$: Bias des Filters.
- $r_{isf}$: nichtnegative Filterantwort der Zelle.

Die äußere Maximumfunktion ist die **ReLU-Aktivierung**: Negative lineare Antworten werden auf null gesetzt.

Anschaulich prüft jeder Filter eine gelernte Kombination von Markern. Eine hohe Antwort bedeutet, dass die Zelle stark auf diese Kombination anspricht. Ob dieser Filter CMV+ oder CMV− unterstützt, entscheidet erst seine Verbindung zum Ausgang des Modells.

## 2.3 Wie entsteht die Referenz für die Auswahl?

Wir rekonstruieren die ursprüngliche Stichprobe, mit der der Scaler dieses Modellkandidaten gefittet wurde:

- 20.000 Zellen je tatsächlichem inneren Trainingsspender;
- ohne Zurücklegen;
- dieselbe sortierte Spenderreihenfolge;
- derselbe Kandidatenseed wie in Aufgabe 4.

Damit enthält die Referenz $T_s$ **180.000 oder 200.000 Zellen**. Der Kandidatenseed wird anhand der gespeicherten Auswahl geprüft:

$$
\text{Kandidatenseed}
=\text{Splitseed}+10.000\cdot\text{innerer Fold}
+100\cdot\text{Filterzahl}.
$$

**Warum so?** Die Referenz ist reproduzierbar, berücksichtigt jeden beteiligten Trainingsspender mit gleich vielen Zellen und benötigt keine Informationen aus den äußeren Testspendern. Wir verwenden eine bereits festgelegte Stichprobe, um die Zahl zusätzlicher Entscheidungen klein zu halten.

Die Beschränkung auf diese Referenz ist eine Projektentscheidung. Das höchste Filteransprechen unter sämtlichen FCS-Ereignissen kann höher sein als das Maximum innerhalb dieser Stichprobe.

## 2.4 Wie lautet der quantitative Zellscore?

Für jeden Filter bestimmen wir sein Maximum innerhalb der Referenz:

$$
M_{sf}=\max_{i\in T_s}r_{isf}.
$$

Für $M_{sf}>0$ lautet der normierte Zellscore:

$$
q_{isf}=\frac{r_{isf}}{M_{sf}}.
$$

Eine Zelle wird ausgewählt, wenn

$$
q_{isf}>0{,}5
\quad\Longleftrightarrow\quad
r_{isf}>0{,}5M_{sf}.
$$

Die Ungleichung ist **streng**. Eine Antwort genau auf der Schwelle gehört nicht zum Subset.

**Beispiel:** Beträgt das Referenzmaximum 10, liegt die Schwelle bei 5. Antworten von 8 und 6 werden ausgewählt; Antworten von 5, 2 und 0 nicht.

Der Score ist keine Wahrscheinlichkeit. Innerhalb der Referenz liegt er zwischen null und eins. Bei Anwendung auf weitere Zellen kann er größer als eins werden, weil deren Antwort das Referenzmaximum übersteigen kann.

**Warum so?** Die Normierung macht „starkes Ansprechen“ relativ zur Antwortskala des einzelnen Filters beschreibbar. Die Hälfte des Maximums ist eine feste Auswahlregel; wir haben diese Schwelle nicht anhand der gewünschten Markerpopulation oder der Testleistung optimiert.

## 2.5 Warum ist das etwas anderes als das Top-1-%-Pooling?

In Aufgabe 4 mittelt CellCNN für die Vorhersage die stärksten 1 % der Filterantworten innerhalb eines Zellinputs. Aufgabe 5 verwendet für die Beschreibung des Subsets die Halbmaximum-Schwelle.

| Regel | Welche Größe wird festgelegt? | Konsequenz |
|---|---|---|
| Top-1-%-Pooling | Anteil der berücksichtigten Zellen | Die Antwortschwelle hängt von der jeweiligen Zellmenge ab. |
| Halbmaximum-Auswahl | Mindestantwort relativ zum Referenzmaximum | Der ausgewählte Zellanteil kann kleiner oder größer als 1 % sein. |

Ein Filter kann deshalb beispielsweise 0,8 % der Referenzzellen auswählen, obwohl sein Vorhersagescore über die höchsten 1 % berechnet wird.

**Das interpretierte Subset ist keine exakte Liste aller Zellen, die in jeder einzelnen Vorhersage gepoolt wurden.** Es beschreibt die Zellen, die besonders stark auf das gelernte Muster reagieren.

## 2.6 Was bedeutet die positive oder negative Richtung eines Filters?

Das Modell hat je einen Ausgang für CMV− und CMV+. Für Filter $f$ betrachten wir den Unterschied seiner beiden Ausgangsgewichte:

$$
\Delta_{sf}=v_{sf,1}-v_{sf,0}.
$$

Bei ansonsten festen gepoolten Antworten gilt:

- $\Delta_{sf}>0$: Eine höhere gepoolte Antwort dieses Filters verschiebt die Entscheidung in Richtung CMV+.
- $\Delta_{sf}<0$: Sie verschiebt die Entscheidung in Richtung CMV−.
- $\Delta_{sf}=0$: Der Filter verändert den Unterschied der beiden Klassenausgänge nicht.

Filter mit Referenzmaximum null oder Ausgangskontrast genau null liefern in unserer Interpretation kein relevantes Subset und werden ausgeschlossen.

**Warum der Unterschied der Ausgangsgewichte?** Entscheidend für die binäre Softmax-Ausgabe ist der Unterschied zwischen beiden Klassenausgängen. Ein großes Gewicht zum positiven Ausgang allein reicht zur Einordnung nicht aus, wenn die Verbindung zum negativen Ausgang ebenfalls groß ist.

Der Betrag dieses Kontrasts wird nicht zur Gewichtung der Subset-Zentroiden verwendet. Die Richtung beschreibt den Modellbezug, die Zentroiden beschreiben die Markerprofile.

## 2.7 Wie wird aus ausgewählten Zellen ein Zentroid?

Für jeden wirksamen Filter bilden wir innerhalb der Referenz das Subset

$$
S_{sf}=\{i\in T_s:r_{isf}>0{,}5M_{sf}\}.
$$

Sein Zentroid ist der mittlere **ArcSinh-transformierte, nicht standardisierte** Markervektor:

$$
c_{sfj}=\frac{1}{|S_{sf}|}\sum_{i\in S_{sf}}a_{ij}.
$$

Das ergibt 37 Werte je Subset: einen Mittelwert pro Marker. Es wird pro Filter ein eigener Zentroid gebildet. Subsets verschiedener Filter können sich überschneiden.

**Warum so?** Ein Zentroid beschreibt kompakt, welche Markerwerte die ausgewählten Zellen tatsächlich haben. Die unmittelbare Gruppierung von Filtergewichten wäre schwerer interpretierbar, weil verschiedene Modelle unterschiedliche Scaler besitzen und ähnliche Zellpopulationen mit unterschiedlichen Gewichten erkennen können.

Die Mittelung erfolgt nach ArcSinh-Transformation. Sie ist weder ein Median noch die ArcSinh-Transformation eines Rohwertmittelwerts.

**Wie sind Spender gewichtet?** Die Referenz enthält gleich viele Zellen je Trainingsspender. Innerhalb des ausgewählten Subsets werden anschließend alle ausgewählten Zellen gleich gewichtet. Ein Spender mit mehr ausgewählten Zellen trägt deshalb stärker zum Subset-Zentroiden bei. Der Zentroid ist kein Mittel gleich gewichteter Spenderprofile.

## 2.8 Ein tatsächliches Beispiel: Split 9, Filter 2

Der spätere Repräsentant der häufigsten CellCNN-Gruppe besitzt folgende gespeicherte Werte:

| Größe | Wert |
|---|---:|
| Äußerer Split | 9 |
| Filter-ID | 2 |
| Tatsächliche Trainingsspender | 10 |
| Referenzzellen | 200.000 |
| Antwortschwelle $0{,}5M_{sf}$ | ca. 6,9707 |
| Ausgewählte Referenzzellen | 1.633 |
| Anteil in der Referenz | 0,8165 % |
| Ausgangskontrast $\Delta_{sf}$ | ca. +1,6586 |

Der Filter spricht somit auf ein relativ kleines Subset an und unterstützt bei steigender gepoolter Antwort die positive Modellrichtung. Die 1.633 ausgewählten Referenzzellen bestimmen seinen Zentroiden.

Quelle: [gespeicherte Subset-Zentroiden](tables/task5_paper_centroids.csv), Zeile für `CellCNN`, `split_id=9`, `subset_id=2`.

# 3. Citrus: Welche Cluster verwendet das Modell?

## 3.1 Welche Größe erklärt Citrus?

Citrus bildet in Aufgabe 4 eine hierarchische Struktur aus Trainingszellen. Aus den darin enthaltenen Clustern entstehen Merkmale, welche die Häufigkeit der jeweiligen Zellpopulation pro Spender beschreiben.

Eine regularisierte logistische Regression kombiniert diese Merkmale schematisch zu

$$
\eta_d=\beta_0+\sum_j\beta_jh_{dj}.
$$

Dabei ist $h_{dj}$ das im Modell verwendete Häufigkeitsmerkmal des Clusters $j$ für Spender $d$, $\beta_j$ sein Koeffizient und $\eta_d$ der lineare Prädiktor für CMV+.

Aufgabe 5 übernimmt die bereits gespeicherten Clusterprofile und Koeffizienten des ausgewählten Modells. Es werden keine neuen Citrus-Bäume aufgebaut.

## 3.2 Wann gilt ein Cluster als ausgewählt?

Die Auswahlregel lautet:

$$
|\beta_j|>10^{-10}.
$$

Der Betrag entscheidet, ob ein Cluster wirksam im Modell vertreten ist. Das Vorzeichen beschreibt seine Richtung:

- Positiver Koeffizient: Ein höheres zugehöriges Häufigkeitsmerkmal erhöht bei sonst festen Merkmalen den linearen Prädiktor für CMV+.
- Negativer Koeffizient: Es senkt ihn.

**Warum die kleine Toleranz?** Rechnerisch winzige Werte sollen nicht als wirksame Auswahl interpretiert werden. Die Toleranz definiert hier numerische Null und ist kein biologischer Relevanztest.

Es werden weder die größten Koeffizienten nachträglich herausgesucht noch eine feste Anzahl von Clustern erzwungen. Alle Cluster, die diese Regel erfüllen, gehen in die Zentroidanalyse ein.

## 3.3 Woher stammen die Clusterzentroiden?

Im Citrus-Notebook wurden für jeden ausgewählten Cluster die Markerwerte seiner Trainingszellen gemittelt und als Zentroid exportiert. Die Werte liegen auf der ArcSinh-Skala vor.

Die zugehörige Full-Konfiguration verwendet 10.000 Zellen je äußerem Trainingsspender und eine Mindestclustergröße von **0,05 %**. Aufgabe 5 prüft diese Einstellungen und liest die vorhandenen Zentroiden.

**Warum so?** Der exportierte Zentroid gehört unmittelbar zu einem Merkmal des bereits getesteten Modells. Ein neu berechnetes Clustering könnte andere Zellgruppen und damit andere Merkmale liefern.

Hierarchische Cluster können ineinander enthalten sein. Zwei ausgewählte Cluster eines Splits müssen deshalb keine getrennten Zellpopulationen darstellen.

## 3.4 Warum färben wir keine Citrus-Zellen auf der Karte ein?

Die verwendeten Exporte enthalten die Trainingszentroiden und ihre Koeffizienten. Daraus lässt sich die exakte Zugehörigkeit beliebiger Karten-Zellen zu einem ursprünglichen hierarchischen Cluster nicht rekonstruieren.

Eine nachträgliche Zuordnung zur nächsten Cluster-Mitte wäre eine zusätzliche Auswahlmethode und nicht automatisch dieselbe Regel wie im gespeicherten Citrus-Modell.

Deshalb zeigt die aktuelle Citrus-Darstellung **die Zentroiden ausgewählter Cluster**. Ein farbiger Punkt steht für ein Clusterprofil, nicht für eine neu klassifizierte Karten-Zelle oder eine vollständig eingezeichnete Clustergrenze.

## 3.5 Was passiert bei einem Modell ohne wirksame Cluster?

Im gespeicherten Lauf liefern die Citrus-Splits **19, 24 und 26** keine wirksamen Cluster. Solche Modelle liefern keinen Zentroiden, bleiben aber bei der späteren Wiederkehr im Nenner enthalten.

**Warum so?** Ein Split ohne ausgewähltes Subset ist ein Ergebnis der Modellselektion. Würden wir ihn aus dem Nenner entfernen, erschiene die Auswahl regelmäßiger, als sie über alle untersuchten Splits tatsächlich war.

# 4. Wiederkehrende Subsets über mehrere Splits erkennen

## 4.1 Warum reicht eine Filter- oder Cluster-ID nicht aus?

Die IDs gelten jeweils innerhalb eines Modells. Filter 2 aus Split 9 muss nicht dasselbe Zellmuster erkennen wie Filter 2 aus Split 10. Entsprechendes gilt für Citrus-Cluster.

Um ähnliche Subsets über Splits hinweg wiederzuerkennen, vergleichen wir ihre 37-dimensionalen Zentroidprofile. CellCNN und Citrus werden dabei **getrennt voneinander** gruppiert.

Eine Gruppe fasst mehrere ähnlich profilierte Subsets zusammen. Sie ist keine automatische Annotation eines biologischen Zelltyps. Auch „Gruppe 1“ bezeichnet bei den beiden Methoden nicht dieselbe Population.

## 4.2 Welche Skalierung verwenden wir zum Vergleich?

Die Zentroiden werden mit den bereits gespeicherten explorativen Mittelwerten und Standardabweichungen aus Aufgabe 2 transformiert:

$$
u_{sfj}=\frac{c_{sfj}-\mu_j^{\mathrm{expl}}}{\sigma_j^{\mathrm{expl}}}.
$$

**Warum so?** Die Marker sollen auf einer gemeinsamen Skala verglichen werden. Die einzelnen Modellscaler wären dafür ungeeignet, weil sie unterschiedliche Trainingsmengen und damit unterschiedliche Bezugspunkte haben.

Der explorative Scaler wurde in Aufgabe 2 aus der Kartenstichprobe **aller 20 Spender** geschätzt. Seine Wiederverwendung ist eine retrospektive, explorative Auswertung und keine ausschließlich auf Trainingsspendern gelernte Gruppierung. Er verändert weder die Klassifikatorvorhersagen noch die CellCNN-Halbmaximum-Schwellen oder die SVM-Auswahl.

Diese Trennung begrenzt die Aussage: Die resultierenden Gruppen sind eine Beschreibung der bereits vorhandenen Modelle und Daten, keine unabhängig validierte Entdeckungsregel für neue Spender.

## 4.3 Wie wird die Ähnlichkeit zweier Subsets gemessen?

Wir verwenden die Kosinusdistanz zweier standardisierter Zentroidprofile $u$ und $v$:

$$
d_{\cos}(u,v)=1-\frac{u^\top v}{\|u\|\,\|v\|}.
$$

Ähnliche Richtungen der Profile ergeben eine kleine Distanz. Das betrifft das Muster relativ zum explorativen Mittel: Welche Marker liegen gemeinsam darüber oder darunter?

Die Kosinusdistanz gewichtet die Richtung des Profils, nicht seinen gesamten Betrag. Zwei gleich gerichtete Profile können deshalb sehr ähnlich sein, obwohl ihre Abweichungen vom Mittel unterschiedlich stark ausfallen.

**Warum so?** Für die Zusammenfassung interessiert uns, ob Modelle ähnliche Markerkombinationen erfassen. Das konkrete Distanzmaß bleibt eine methodische Wahl; ein anderer Abstand kann andere Gruppen ergeben. Für einen Nullvektor wäre die Kosinusdistanz undefiniert, weshalb die Implementierung solche Eingaben zurückweist.

## 4.4 Wie entstehen daraus Gruppen?

Wir verwenden hierarchisches Clustering mit **Average-Linkage** und einem **Distanzschnitt von 0,4**.

Average-Linkage beschreibt den Abstand zweier Gruppen durch den Mittelwert aller paarweisen Abstände zwischen ihren Mitgliedern. Schrittweise werden die jeweils nächstliegenden Gruppen zusammengeführt. Der Distanzschnitt legt fest, welche Zusammenführungen in den endgültigen Gruppen enthalten sind.

**Ein Schnitt bei 0,4 bedeutet nicht, dass jedes Paar innerhalb einer Gruppe höchstens Distanz 0,4 haben muss.** Die Zusammenführung richtet sich nach dem durchschnittlichen Abstand der beteiligten Gruppen.

Die [Provenienz](tables/task5_paper_provenance.json) dokumentiert diese Einstellung als Übertragung aus der Filtergruppierung des [CellCNN-Referenzcodes, Commit `0413a9f`](https://github.com/eiriniar/CellCnn/tree/0413a9f49fe0831c8fe3280957fb341f9e028d2d/cellCnn). Die Übertragung auf unsere Subset-Zentroiden ist eine bewusste Anpassung. Sie darf nicht als belegte Originalparametrisierung der NK-Zentroidanalyse ausgegeben werden.

Die Einstellung wird fest verwendet. Sie wurde nicht anhand einer gewünschten Zelltyp-Zuordnung optimiert.

## 4.5 Wie wird die Wiederkehr berechnet?

Für eine Zentroidgruppe $G$ zählen wir, in wie vielen unterschiedlichen äußeren Splits mindestens ein Mitglied dieser Gruppe vorkommt:

$$
F_G=\frac{\#\{s\in\{0,\ldots,29\}:\text{mindestens ein Mitglied von }G\text{ in Split }s\}}{30}.
$$

Wichtig sind drei Regeln:

- Mehrere Mitglieder aus demselben Split zählen einmal.
- Modelle ohne ausgewähltes Subset bleiben im Nenner.
- Fehlende Modellexporte sind Fehler und dürfen nicht als Nullmodelle gezählt werden.

**Beispiel aus unseren Daten:** CellCNN-Gruppe 1 enthält 58 Zentroiden, verteilt auf alle 30 Splits. Ihre Wiederkehr beträgt $30/30=1$, nicht $58/30$.

Damit werden zwei verschiedene Größen getrennt: `n_centroids` zählt die gruppierten Subsets, `occurrences` die unterschiedlichen Splits.

Für die Darstellung behalten wir Gruppen mit **mindestens sechs Vorkommen**. Bei 30 Splits entspricht das einer Wiederkehr von mindestens 20 %.

**Warum so?** Einzelne selten auftretende Modellkomponenten sollen die zusammenfassende Darstellung nicht dominieren. Die Grenze beschreibt eine festgelegte Auswahl für die Darstellung, keinen Signifikanztest und keine Wahrscheinlichkeit biologischer Richtigkeit.

## 4.6 Werden positive und negative Subsets getrennt gruppiert?

Die Gruppierung verwendet die Markerprofile, nicht das Vorzeichen des Ausgangskontrasts oder Regressionskoeffizienten. Ähnliche Profile können daher gemeinsam in einer Gruppe liegen, obwohl ihre Modellrichtung unterschiedlich ist.

Zusätzlich zählen wir je Gruppe die Splits mit mindestens einem positiven beziehungsweise negativen Mitglied. Beide Zählungen können sich überschneiden.

**Tatsächliches Beispiel:** Citrus-Gruppe 3 tritt in zehn Splits auf. Vier Splits enthalten ein positives und sieben ein negatives Mitglied. Die Summe elf ist möglich, weil ein Split beide Richtungen enthält.

**Warum so?** Ähnlichkeit des Markerprofils und Richtung im Modell sind unterschiedliche Eigenschaften. Eine gemischte Richtung ist eine relevante Einschränkung der Interpretation und wird sichtbar ausgewiesen.

## 4.7 Wie wird ein repräsentatives Subset ausgewählt?

Für jede Gruppe suchen wir den vorhandenen Zentroiden mit der kleinsten Summe der Kosinusdistanzen zu allen anderen Mitgliedern:

$$
c_G^*=\underset{c\in G}{\operatorname{argmin}}\sum_{c'\in G}d_{\cos}(u_c,u_{c'}).
$$

Ein solcher Repräsentant heißt **Medoid**. Er gehört zu einem tatsächlich gespeicherten Filter beziehungsweise Cluster. Bei Gleichstand innerhalb einer Toleranz von $10^{-12}$ entscheidet die aufsteigende Split- und Subset-ID.

**Warum so?** Zu einem vorhandenen Zentroiden existiert ein konkretes Subset mit einer konkreten Auswahlregel. Ein künstlicher Durchschnitt mehrerer Filter könnte eine neue Zellpopulation auswählen und wäre nicht mehr eines der getesteten Modelle.

Die Gruppen werden nach abnehmender Wiederkehr nummeriert; weitere Gleichstände werden anhand der Repräsentanten-IDs aufgelöst. Für die zusätzliche CellCNN-Zellkarte wird der Repräsentant der häufigsten beibehaltenen Gruppe verwendet. Es erfolgt keine nachträgliche Auswahl anhand erwarteter Marker oder der höchsten Test-AUC.

# 5. SVM: Welche Einzelzellen gehen in die Entscheidung ein?

## 5.1 Wie entsteht der Zellscore?

Für jeden äußeren Split wird die gespeicherte lineare SVM mit ihrem eigenen Scaler rekonstruiert. Für jede Zelle eines vollständigen Testspenders berechnen wir den linearen Score, auch Margin genannt:

$$
m_{is}=w_s^\top z_i+b_s.
$$

Hohe Werte liegen stärker in der positiven Richtung der SVM. Es handelt sich um einen Entscheidungsscore, nicht um eine kalibrierte Wahrscheinlichkeit.

Die Modelle werden ausschließlich auf den Spendern ausgewertet, die im jeweiligen Split zum äußeren Test gehören.

## 5.2 Welche Zellen werden für einen Spender berücksichtigt?

Hat ein Testspender $N_d$ Zellen, wählen wir genau

$$
k_d=\left\lceil0{,}01N_d\right\rceil
$$

Zellen mit den höchsten Scores aus. $\lceil\cdot\rceil$ bedeutet Aufrunden. Die entsprechende Indexmenge bezeichnen wir mit $I_{ds}$.

Bei gleichen Scores an der Grenze entscheidet die ursprüngliche Ereignisnummer. So ist die Auswahl deterministisch und enthält exakt $k_d$ Zellen.

**Warum auf dem vollständigen Spender?** Genau diese Zellmenge ging in den gespeicherten Spenderscore aus Aufgabe 4 ein. Die höchsten fünf von 500 Karten-Zellen wären eine andere Auswahl und könnten den eigentlichen oberen Rand der vollständigen Probe verfehlen.

## 5.3 Wann ist eine ausgewählte Zelle positiv oder negativ?

Die SVM-Spenderschwelle $\tau_s$ wurde in Aufgabe 4 anhand innerer Validierung bestimmt. Innerhalb der ausgewählten Top-1-%-Zellen betrachten wir

$$
t_{is}=m_{is}-\tau_s.
$$

Dann gilt:

- $i\in I_{ds}$ und $t_{is}>0$: positive Auswahl;
- $i\in I_{ds}$ und $t_{is}<0$: negative Auswahl;
- $i\in I_{ds}$ und $t_{is}=0$: weder positive noch negative Auswahl;
- $i\notin I_{ds}$: keine Auswahl, unabhängig vom Vorzeichen.

**Eine negative Auswahl bedeutet also nicht, dass die niedrigsten 1 % ausgewählt wurden.** Sie bezeichnet Zellen innerhalb der höchsten 1 %, deren Score trotzdem unterhalb der gespeicherten Spenderschwelle liegt.

## 5.4 Warum beziehen wir die Zellscores auf die Spenderschwelle?

Der in Aufgabe 4 verwendete Spenderscore ist der Mittelwert der ausgewählten Margins:

$$
g_{ds}=\frac{1}{k_d}\sum_{i\in I_{ds}}m_{is}.
$$

Daraus folgt:

$$
g_{ds}-\tau_s
=\frac{1}{k_d}\sum_{i\in I_{ds}}(m_{is}-\tau_s).
$$

Die zentrierten Scores der ausgewählten Zellen ergeben im Mittel somit exakt den Abstand des Spenderscores zur Entscheidungsschwelle.

**Rechenbeispiel:** Von 101 Zellen seien die beiden höchsten Margins 9 und 8. Wegen $\lceil1{,}01\rceil=2$ werden beide ausgewählt. Bei $\tau_s=8{,}5$ ist die erste positiv, die zweite negativ; ihr mittlerer zentrierter Score beträgt null.

Diese Zerlegung gilt für die feste Top-Auswahl. Sie ist kein kausaler Entfernungseffekt einer Zelle: Beim Entfernen einer Zelle könnten sich die Auswahlmenge und ihr Mittelwert ändern.

Die Implementierung kontrolliert für jeden Testspender, dass der rekonstruierte Top-Mittelwert, die ausgewählte Zellzahl und die Entscheidungsschwelle zu den gespeicherten Vorhersagen passen. Für den Mittelwert wird eine absolute Toleranz von $10^{-10}$ verwendet.

## 5.5 Was bedeutet die Auswahlhäufigkeit einer Karten-Zelle?

Ein Spender kann in mehreren der 30 Splits im äußeren Test liegen. Für jede seiner Karten-Zellen zählen wir, wie oft sie positiv beziehungsweise negativ ausgewählt wird.

Sei $\mathcal T_d$ die Menge der Splits, in denen Spender $d$ zum äußeren Test gehört. Dann gilt für seine Zelle $i$:

$$
f_i^+=\frac{\sum_{s\in\mathcal T_d}\mathbf{1}[i\in I_{ds}\ \text{und}\ m_{is}>\tau_s]}{|\mathcal T_d|},
$$

$$
f_i^-=\frac{\sum_{s\in\mathcal T_d}\mathbf{1}[i\in I_{ds}\ \text{und}\ m_{is}<\tau_s]}{|\mathcal T_d|}.
$$

Die Indikatorfunktion $\mathbf{1}[\cdot]$ ist eins, wenn die Bedingung zutrifft, sonst null.

Der Nenner ist die Zahl **aller Testauftritte dieses Spenders**, einschließlich der Splits, in denen die Zelle nicht ausgewählt wurde. Im vorhandenen Lauf sind es je Spender vier bis 16 Testauftritte.

**Beispiel:** Wird eine Zelle bei acht Testauftritten dreimal positiv und einmal negativ ausgewählt, beträgt $f_i^+=3/8$ und $f_i^-=1/8$. Die übrigen vier Auftritte zählen ebenfalls zum Nenner.

Ohne einen Testauftritt wäre die Häufigkeit nicht bestimmbar und würde als fehlender Wert gespeichert. Im vorliegenden Lauf sind alle 20 Spender im Test vertreten.

## 5.6 Was bedeutet „OOF“ und warum unterscheiden sich diese Häufigkeiten von den Zentroidgruppen?

**Out-of-fold**, kurz OOF, bedeutet hier: Die Zelle wird mit einem Modell bewertet, das ihren Spender im betreffenden äußeren Split nicht zum Training oder zur Modellauswahl verwendet hat.

Die beiden Häufigkeitsbegriffe beantworten verschiedene Fragen:

| Größe | Einheit, die wiederkehrt | Nenner | Bedeutung |
|---|---|---|---|
| $F_G$ bei CellCNN/Citrus | Ähnlich profilierte Subsets einer Gruppe | 30 äußere Splits | Wie regelmäßig findet das Training ein entsprechendes Muster? |
| $f_i^\pm$ bei der SVM | Dieselbe ursprüngliche Zelle eines Spenders | Testauftritte ihres Spenders | Wie regelmäßig wird diese Zelle bei Bewertung ihres Testspenders ausgewählt? |

Diese Werte dürfen nicht als vergleichbare Effektstärken gelesen werden. Ein Wert von 0,6 bedeutet in beiden Spalten etwas anderes.

Über verschiedene Splits kann dieselbe SVM-Zelle sowohl positiv als auch negativ ausgewählt werden. Innerhalb eines Splits schließen sich beide Richtungen aus.

## 5.7 Wie entsteht das spendergleich gewichtete SVM-Markerprofil?

Für die biologische Einordnung auf Folie 9 wird zusätzlich ein Profil der **positiv ausgewählten Zellen vollständiger Testspender** berechnet. Es verwendet dieselben 30 äußeren Modelle, ihre gespeicherten Scaler und Schwellen sowie die Auswahlregel aus Abschnitt 5.3. Beide tatsächlichen CMV-Klassen gehen ein. Es gibt keine Einschränkung auf die 93 positiv markierten Karten-Zellen oder auf positiv klassifizierte Spender.

Für Spender $d$ und dessen äußeren Testauftritt $s$ sei

$$
P_{ds}=\{i\in I_{ds}:m_{is}>\tau_s\}.
$$

Bei nichtleerer Auswahl mitteln wir für Marker $j$ die ArcSinh-Werte:

$$
c_{dsj}=\frac{1}{|P_{ds}|}\sum_{i\in P_{ds}}a_{ij}.
$$

Anschließend mitteln wir zuerst über die **nichtleeren Testauftritte eines Spenders** und danach gleichgewichtet über die vertretenen Spender:

$$
\mathcal T_d^+=\{s\in\mathcal T_d:|P_{ds}|>0\},\qquad
c_{dj}=\frac{1}{|\mathcal T_d^+|}\sum_{s\in\mathcal T_d^+}c_{dsj},
$$

$$
D^+=\{d:|\mathcal T_d^+|>0\},\qquad
c_j^{\mathrm{SVM}}=\frac{1}{|D^+|}\sum_{d\in D^+}c_{dj}.
$$

**Warum diese Reihenfolge?** Ein Spender soll weder wegen mehr gemessener oder ausgewählter Zellen noch wegen mehr Testauftritten stärker gewichtet werden. Innerhalb eines nichtleeren Subsets zählen alle Zellen gleich. Die Zusammenfassung ist eine eigene deskriptive Ergänzung, keine im Paper vorgegebene SVM-Interpretation.

**Leere Auswahlen:** Für $P_{ds}=\varnothing$ ist das Markerprofil undefiniert und wird als fehlend gespeichert. Es geht nicht als Nullprofil in die Mittelung ein. Spender ohne irgendeine positive Auswahl würden ebenfalls kein Profil beitragen. Die Abdeckung wird deshalb stets mitberichtet. Fehlt jede positive Auswahl, stoppt der Export mit einer Fehlermeldung. Das Profil beschreibt bedingt auf positive Auswahl die Markerausprägung; deren Häufigkeit wird dadurch nicht geschätzt.

Für die Auswahl bleibt die ursprüngliche Arithmetik erhalten: Rohwerte zunächst als `float32`, dann `arcsinh(x/5)`, gespeicherter Scaler und gespeicherte SVM-Gewichte. Die Markerwerte werden in `float64` aufsummiert. Erst nach der dreistufigen Mittelung wird das Profil mit der **bestehenden explorativen** Marker-Skalierung aus Aufgabe 2 dargestellt:

$$
z_j^{\mathrm{SVM}}=\frac{c_j^{\mathrm{SVM}}-\mu_j^{\mathrm{expl}}}{\sigma_j^{\mathrm{expl}}}.
$$

Es wird kein neuer Scaler gefittet. Die acht dargestellten Marker sind CD3, CD19, CD56, CD16, CD94, NKG2A, NKG2C und CD57. SVM-Gewichte selbst werden nicht als Markerexpression ausgegeben.

## 5.8 Welche Abdeckung hat das zusätzliche Profil?

Alle **20 Testspender** tragen zum Profil bei. **168 von 180 Testauftritten** enthalten mindestens eine positiv ausgewählte Zelle; zwölf Auswahlen sind leer. Pro Spender tragen vier bis 16 nichtleere Auftritte bei. Insgesamt werden **82.189 ausgewählte Zellvorkommen über Testauftritte** erfasst. Das ist keine Anzahl eindeutiger Zellen: Eine ursprüngliche Zelle kann in mehreren Splits erneut ausgewählt werden.

Die [Tabelle je Testauftritt](../praesentation/aufgaben_04_05/data/svm_profiles_by_test_visit.csv) enthält alle 180 Spender-/Split-Kombinationen, vollständige Zellzahl, Top-Zellzahl, positive Auswahlzahl und die acht Marker-Mittelwerte. Leere Profile sind ausdrücklich als fehlend enthalten. Diese Tabelle ermöglicht die unabhängige Nachrechnung der beiden anschließenden Mittelungsschritte.

# 6. Darstellung und vorhandene Ergebnisse

## 6.1 Welche Karte verwenden wir?

Wir verwenden die bestehende **t-SNE-Karte mit Perplexität 30** aus Aufgabe 2, Variante `tsne_p30`. Sie enthält 10.000 Zellen, gleichmäßig verteilt auf die 20 Spender.

Die Karte wurde mit Seed 42, PCA-Initialisierung, automatischer Lernrate und maximal 1.000 Iterationen auf den explorativ standardisierten 37 Markerwerten berechnet. Aufgabe 5 übernimmt die Koordinaten unverändert.

**Warum so?** Alle Darstellungen erhalten denselben Hintergrund. Unterschiede entstehen durch die eingezeichneten Modellauswahlen. Der quantitative Zellscore wird im Markerraum berechnet; eine optisch auffällige t-SNE-Insel wird nicht nachträglich zur Selektionsregel erklärt.

Die beiden Achsen sind Darstellungskoordinaten. Ihre Werte besitzen keine unmittelbare biologische Einheit. Eine räumliche Nachbarschaft ist ein Hinweis zur Exploration und keine Zelltypbestimmung.

## 6.2 Wie wird ein Zentroid auf die bestehende Karte übertragen?

Ein Subset-Zentroid ist ein neu berechnetes 37-dimensionales Profil und besitzt keine eigene gespeicherte t-SNE-Koordinate.

Deshalb suchen wir unter den 10.000 Karten-Zellen diejenige mit dem kleinsten **euklidischen Abstand im explorativ standardisierten Markerraum**:

$$
i^*(c)=\underset{i\in\text{Karten-Zellen}}{\operatorname{argmin}}\|u_c-u_i\|_2.
$$

Der Zentroid erhält die t-SNE-Koordinaten dieser Zelle. Bei Gleichstand entscheidet die bestehende Kartenreihenfolge.

**Wichtig:** Die Gruppierung der Zentroiden verwendet Kosinusdistanz. Ihre Positionierung auf der Karte verwendet euklidische Distanz. Das sind zwei getrennte Schritte.

**Warum so?** Damit lässt sich ein neues Profil auf der vorhandenen Karte verorten. Die Lage ist eine Approximation; mehrere Zentroiden können auf dieselbe Karten-Zelle fallen. Ein dargestellter Zentroidpunkt beweist keine Zugehörigkeit dieser Karten-Zelle zum ursprünglichen Trainingssubset.

## 6.3 Was zeigt die Zentroidabbildung?

[Die Abbildung der Subset-Zentroiden](figures/task5_paper_centroids.png) enthält je ein Feld für CellCNN und Citrus:

- Graue Punkte: die gemeinsame Kartenstichprobe.
- Farbige Punkte: Zentroiden der Gruppen mit mindestens sechs Vorkommen.
- Unterschiedliche Farben: unterschiedliche Gruppen innerhalb der jeweiligen Methode.
- Sterne: die Repräsentanten der gezeigten Gruppen.

Ein farbiger Punkt entspricht einem ausgewählten Subset aus einem bestimmten Split. Seine Gruppenzugehörigkeit beruht auf dem Markerprofil und wird nicht aus der Lage im zweidimensionalen Bild bestimmt.

## 6.4 Was zeigt die zusätzliche Zellabbildung?

[Die Abbildung ausgewählter Zellen](figures/task5_paper_subsets.png) zeigt drei Felder:

| Feld | Gezeigte Auswahl | Einordnung |
|---|---|---|
| CellCNN | Halbmaximum-Auswahl des Repräsentanten der häufigsten Gruppe | Explorative Anwendung auf alle 20 Spender, einschließlich Trainingsspendern |
| SVM positiv | Karten-Zellen mit $f_i^+>0$; Farbe zeigt $f_i^+$ | Ausschließlich aus äußeren Testauftritten berechnet |
| SVM negativ | Karten-Zellen mit $f_i^->0$; Farbe zeigt $f_i^-$ | Ausschließlich aus äußeren Testauftritten berechnet |

Der CellCNN-Repräsentant verwendet seinen ursprünglichen Scaler und seine ursprüngliche Referenzschwelle. Die Karte erhält keine neu angepasste Halbmaximum-Schwelle.

**Warum nur ein CellCNN-Repräsentant?** Er macht eine konkrete, regelmäßig auftretende Auswahlregel sichtbar. Die übrigen wiederkehrenden Gruppen bleiben in der Zentroidabbildung und Ergebnistabelle enthalten.

Die CellCNN-Zellkarte ist somit keine OOF-Auswertung. Auch die SVM-Häufigkeiten werden auf einer explorativen Karte aller Spender dargestellt; OOF bezeichnet dort die Modellbewertung der Zellen, nicht das Lernen der Karte.

## 6.5 Welche Zentroidgruppen liegen aktuell vor?

Insgesamt enthalten die Exporte **130 CellCNN-Zentroiden und 123 Citrus-Zentroiden**. Daraus entstehen 15 beziehungsweise 17 Gruppen. Folgende vier CellCNN- und fünf Citrus-Gruppen erreichen mindestens sechs Vorkommen:

| Methode | Gruppe | Zentroiden | Splits mit Vorkommen | Wiederkehr | Splits mit positiver Richtung | Splits mit negativer Richtung |
|---|---:|---:|---:|---:|---:|---:|
| CellCNN | 1 | 58 | 30/30 | 100,0 % | 30 | 0 |
| CellCNN | 2 | 25 | 20/30 | 66,7 % | 0 | 20 |
| CellCNN | 3 | 20 | 15/30 | 50,0 % | 0 | 15 |
| CellCNN | 4 | 8 | 8/30 | 26,7 % | 2 | 6 |
| Citrus | 1 | 28 | 18/30 | 60,0 % | 18 | 0 |
| Citrus | 2 | 19 | 11/30 | 36,7 % | 11 | 0 |
| Citrus | 3 | 17 | 10/30 | 33,3 % | 4 | 7 |
| Citrus | 4 | 6 | 6/30 | 20,0 % | 0 | 6 |
| Citrus | 5 | 7 | 6/30 | 20,0 % | 6 | 0 |

Quelle: [Gruppentabelle](tables/task5_paper_groups.csv). Positive und negative Splitzahlen können sich überschneiden.

Die häufigste CellCNN-Gruppe ist über alle untersuchten Splits vertreten und hat ausschließlich positive Mitglieder. Auch die häufigste Citrus-Gruppe enthält ausschließlich positive Mitglieder, tritt aber in 18 Splits auf.

Das zeigt eine unterschiedliche Wiederkehr unter den verwendeten Regeln. Es belegt weder eine höhere Klassifikationsleistung noch, dass beide Gruppen denselben Zelltyp darstellen.

## 6.6 Wie viele Karten-Zellen werden ausgewählt?

| Auswertung | Vorhandenes Ergebnis |
|---|---:|
| CellCNN-Repräsentant: Gruppe 1, Split 9, Filter 2 | 87 von 10.000 Karten-Zellen |
| SVM: mindestens einmal positiv ausgewählt | 93 Karten-Zellen |
| SVM: mindestens einmal negativ ausgewählt | 286 Karten-Zellen |
| SVM: über verschiedene Splits in beiden Richtungen ausgewählt | 73 Karten-Zellen |
| Testauftritte je Spender innerhalb der 30 Splits | 4–16 |

Quellen: [CellCNN-Repräsentant auf der Karte](tables/task5_paper_representative_cells.csv) und [SVM-Zellhäufigkeiten](tables/task5_paper_svm_cells.csv).

Die 87 CellCNN-Zellen entsprechen 0,87 % der Karte. Das ist ein Anteil in der Kartenstichprobe unter einer konkreten Filterregel. Er ist keine geschätzte Prävalenz eines biologisch bestätigten Zelltyps in sämtlichen FCS-Ereignissen.

Die SVM-Zahlen zählen jede Karten-Zelle, sobald sie mindestens einmal in der jeweiligen Richtung ausgewählt wurde. Sie dürfen weder mit den 87 Zellen eines einzelnen CellCNN-Filters noch mit der Anzahl von Citrus-Zentroiden als Maß der Methodenqualität verglichen werden.

## 6.7 Was zeigt die ergänzte Markerprofil-Folie?

Die [Markerprofilgrafik](../praesentation/aufgaben_04_05/figures/marker_profiles.pdf) stellt drei Profile auf derselben explorativen z-Skala dar. Werte in dieser Tabelle sind wie auf der Folie auf eine Nachkommastelle gerundet:

| Profil | CD3 | CD19 | CD56 | CD16 | CD94 | NKG2A | NKG2C | CD57 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CellCNN G1 | −0,2 | −0,1 | 2,1 | 1,7 | 2,4 | 0,5 | 4,6 | 1,9 |
| Citrus G1 | 0,7 | 0,0 | −0,2 | −0,1 | 0,4 | −0,4 | 0,3 | 2,2 |
| SVM positive OOF | 0,3 | 0,1 | 1,2 | 1,6 | 1,9 | 0,4 | 3,0 | 2,0 |

CellCNN und Citrus verwenden unverändert die gespeicherten G1-Repräsentanten: Split 9/Filter 2 beziehungsweise Split 25/Cluster 139891. Die SVM-Zeile verwendet die spendergleich gewichteten Testzellprofile aus Abschnitt 5.7. Sie ist **kein dritter Gruppenrepräsentant** und besitzt deshalb keine einzelne Split-, Filter- oder Cluster-ID.

CellCNN zeigt ein NKG2C-/CD57- und NK-assoziiertes Muster, das mit dem im Paper beschriebenen memory-like NK-Phänotyp vereinbar ist. Beim Citrus-Repräsentanten fallen CD3 und CD57 bei deutlich geringerer NKG2C-Erhöhung auf, was ein T-assoziiertes Muster nahelegt. Auch die SVM-Auswahl zeigt erhöhte NKG2C-/CD57-Werte zusammen mit CD56, CD16 und CD94; CD3 liegt ebenfalls etwas über dem Referenzmittel. Diese Mittelwerte erlauben weder eine gesicherte Zelltypzuordnung noch eine Aussage über die Reinheit der Subsets.

Die gemeinsame Farbskala macht Markerwerte lesbar vergleichbar, beseitigt aber nicht die unterschiedlichen Subset-Definitionen und Aggregationen. Insbesondere ist ein höherer z-Wert keine bessere Klassifikationsleistung und kein Positivitätsgate. Die ungerundeten 24 Werte samt Profiltyp stehen in der [Profiltabelle](../praesentation/aufgaben_04_05/data/representative_profiles.csv).

# 7. Paper-Nähe, Aussagegrenzen und Nachvollziehbarkeit

## 7.1 Welche Schritte übernehmen wir aus dem Paper?

Die NK-Benchmark-Interpretation im Paper definiert CellCNN-Subsets über die Hälfte der maximalen Filterantwort, fasst sie als Zentroiden zusammen und gruppiert ähnliche Zentroiden über Wiederholungen. Citrus wird über Cluster mit nichtnulligen Regressionskoeffizienten entsprechend ausgewertet. Berichtet werden Gruppen mit mindestens 20 Vorkommen in 100 Wiederholungen. Zentroiden werden über ihre nächste Zelle im Markerraum auf die t-SNE-Karte übertragen; ein zentraler vorhandener Zentroid repräsentiert eine häufige Gruppe. Diese Grundstruktur übernimmt unsere CellCNN-/Citrus-Auswertung. Quelle: [CellCNN-Paper, Methoden, NK-cell benchmark](https://www.nature.com/articles/ncomms14825), lokal Seite 8.

## 7.2 Welche Anpassungen und Einschränkungen müssen genannt werden?

| Punkt | Umsetzung im Projekt | Begründung beziehungsweise Folge |
|---|---|---|
| Zahl der Wiederholungen | 30 gemeinsame Splits statt 100 | Gemeinsame verfügbare Grundlage; entsprechend begrenzte Stabilitätsaussage |
| Wiederkehrschwelle | Mindestens 6 von 30 | Übertragung des 20-%-Kriteriums |
| CellCNN-Referenzmaximum | Ursprüngliche Scaler-Stichprobe der inneren Trainingsspender | Reproduzierbare, spenderbalancierte Referenz; seltene Zellen außerhalb dieser Stichprobe beeinflussen die Schwelle nicht |
| Zentroidgruppierung | Explorative z-Skalierung, Average-Linkage, Kosinusdistanz, Schnitt 0,4 | Konkrete Übertragung der dokumentierten Filtergruppierungsregel; keine belegte Originalparametrisierung für NK-Zentroiden |
| Kartenstichprobe | 500 Zellen je Spender | Wiederverwendung von Aufgabe 2; deutlich weniger Zellen als die 20.000 je Person der Paperabbildung |
| SVM-Interpretation | Top-1-%-Auswahl, Richtung relativ zur Spenderschwelle, OOF-Zellhäufigkeiten | Eigene, an den gespeicherten SVM-Spenderscore angeschlossene Lösung |
| Ergänzendes SVM-Markerprofil | Zellen je Testauftritt, nichtleere Auftritte je Spender, Spender gleichgewichtet mitteln | Deskriptives Profil bedingt auf positive Auswahl; keine zusätzliche Zentroidgruppe oder neue Leistungsbewertung |
| Citrus-Darstellung | Ausgewählte Clusterzentroiden | Keine exakte Zellzuordnung ohne ursprüngliche Zuordnungsstruktur |
| Weitere Zelltypanalyse | Keine zusätzliche Unterteilung der CellCNN-Subsets im aktuellen Aufgabe-5-Lauf | Ein Subset kann mehrere Zelltypen enthalten; seine Homogenität ist nicht nachgewiesen |

Die reduzierte Kartenstichprobe ist besonders für seltene Subsets relevant: Ein im vollständigen Datensatz vorhandenes Muster kann auf der Karte nur wenige oder keine Zellen haben. Das Ausbleiben sichtbarer Punkte beweist daher nicht, dass die Population fehlt.

## 7.3 Welche Aussagen sind durch diese Auswertung gedeckt?

Wir können angeben, welche Filter oder Cluster ausgewählt wurden, wie ihre Markerprofile aussehen, welche Modellrichtung sie besitzen und wie häufig ähnliche Profile über die betrachteten Splits wiederkehren. Für die SVM lässt sich zusätzlich nachvollziehen, welche ursprünglichen Karten-Zellen bei Bewertung ihrer Testspender in das Top-Subset gelangen.

Für weitergehende Aussagen sind folgende Grenzen wesentlich:

- **Keine Zelllabels:** Eine biologische Trefferquote der ausgewählten Zellen ist ohne unabhängige Referenzannotation nicht bestimmbar.
- **Keine Kausalität:** Modellassoziation und Wiederkehr beweisen keinen ursächlichen Zusammenhang mit CMV.
- **Keine unabhängigen Wiederholungen:** Die 30 Splits verwenden überlappende Teile derselben 20 Spender. Wiederkehr ist kein Signifikanztest.
- **Keine automatische Zelltypbestätigung:** Zentroid, Kartenlage und Filterrichtung reichen zur biologischen Benennung eines Subsets nicht aus. Hierfür müssten passende Markerprofile und zusätzliche biologische Evidenz betrachtet werden.
- **Keine vollständige Zerlegung der Vorhersage:** Besonders die CellCNN-Halbmaximum-Auswahl beschreibt stark antwortende Zellen und ist nicht identisch mit den gepoolten Zellen jeder Vorhersage. Bei Citrus wirken mehrere möglicherweise überlappende Cluster gemeinsam.
- **Begrenzte Homogenität:** Ein Mittelprofil kann unterschiedliche Zellzustände zusammenfassen. Ähnliche Zentroiden bedeuten nicht, dass die zugrunde liegenden Zellverteilungen identisch sind.

Für den Bericht ist deshalb die Formulierung **„wiederkehrende, modellassoziierte Subsets“** angemessen. Eine Bezeichnung als gesicherte CMV-spezifische Zelltypen wäre durch diese Auswertung allein nicht gedeckt.

## 7.4 Welche Dateien enthalten das aktuelle Ergebnis?

| Datei unter `results/` | Inhalt |
|---|---|
| [tables/task5_paper_centroids.csv](tables/task5_paper_centroids.csv) | Ein Profil je wirksamem CellCNN-Filter oder Citrus-Cluster, einschließlich Richtung, Gruppenzuordnung und Kartenposition |
| [tables/task5_paper_groups.csv](tables/task5_paper_groups.csv) | Gruppen, Wiederkehr, Richtungszählungen, Repräsentanten und Entscheidung über die Darstellung |
| [tables/task5_paper_svm_cells.csv](tables/task5_paper_svm_cells.csv) | Positive und negative Auswahlzählungen und Häufigkeiten je Karten-Zelle |
| [tables/task5_paper_representative_cells.csv](tables/task5_paper_representative_cells.csv) | Filterantwort und Auswahlstatus aller Karten-Zellen für den CellCNN-Repräsentanten |
| [tables/task5_paper_provenance.json](tables/task5_paper_provenance.json) | Parameter, Splits, Softwareversionen sowie Quellen-, Implementierungs- und Ausgabeprüfsummen |
| [figures/task5_paper_centroids.png](figures/task5_paper_centroids.png) | Wiederkehrende CellCNN-/Citrus-Subset-Zentroiden auf der vorhandenen t-SNE-Karte |
| [figures/task5_paper_subsets.png](figures/task5_paper_subsets.png) | CellCNN-Repräsentant sowie positive und negative SVM-Zellhäufigkeiten |

Die ergänzenden Markerprofile liegen getrennt unter `praesentation/aufgaben_04_05/data/`: `svm_profiles_by_test_visit.csv` und `representative_profiles.csv`. Die dortige `provenance.json` dokumentiert die ursprünglichen FCS-/Modellquellen, Auswahl und Aggregation, Abdeckung sowie die Prüfsummen der Präsentationsausgaben. Die bestehende `task5_paper_provenance.json` und ihre Ergebnisse werden nicht überschrieben.

Die ebenfalls vorhandenen älteren Dateien wie `task5_cell_scores.csv`, `task5_marker_profiles.csv` und `task5_provenance.json` gehören zu einem historischen Auswertungsstand. Ihre Zahlen und Auswahlregeln werden für diese Erklärung nicht mit den aktuellen Ergebnissen vermischt.

## 7.5 Was wurde für diese Erklärung tatsächlich geprüft?

Die Beschreibung wurde mit dem aktuellen Notebook und der Implementierung abgeglichen. Die lokale Aufgabenstellung und der NK-Methodenabschnitt des lokalen Papers wurden gelesen.

Aus den gespeicherten CSV-Dateien wurden die Zentroid- und Gruppenzahlen, die Citrus-Splits ohne wirksame Cluster, die Angaben zum CellCNN-Repräsentanten und die SVM-Zellzahlen einschließlich ihrer Nenner nachgerechnet. Alle **16 in der Provenienz erfassten Artefaktprüfsummen**, die **vier Ergebnis-CSV-Prüfsummen** und die **Prüfsumme der Implementierung** stimmen mit den vorhandenen Dateien überein.

Für die ergänzenden SVM-Profile wurden auch die ursprünglichen FCS-Eingabeprüfsummen kontrolliert und die 180 Testauswertungen rekonstruiert. Top-Zellzahlen und Schwellen stimmen mit den gespeicherten Vorhersagen überein; die größte absolute Abweichung eines rekonstruierten Spenderscores beträgt $4{,}44\cdot10^{-16}$. Die positiven und negativen Auswahlzählungen sowie Testauftrittsnenner der 10.000 Karten-Zellen stimmen exakt mit dem bestehenden Export überein. Kein Klassifikator, Scaler oder Projektionsmodell wurde neu gefittet; das Notebook und der Benchmark wurden nicht erneut ausgeführt.
