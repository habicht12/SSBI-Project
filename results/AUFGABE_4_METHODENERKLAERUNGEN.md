# Aufgabe 4: CellCNN, lineare Single-Cell-SVM und Citrus ausführlich erklärt

Diese Datei führt die drei Erklärungen aus dem Gespräch zusammen: von den FCS-Daten bis zu den gespeicherten Spendervorhersagen, einschließlich Formeln, Entscheidungsbegründungen und Vergleich mit dem CellCNN-Paper.

**Dokumentationsstand: 9. September 2026.** Die Angaben beziehen sich jeweils auf den Full-Modus. CellCNN und SVM liegen mit 100 äußeren Splits vor. Die aktuelle Citrus-Konfiguration verwendet 10.000 Zellen je Spender, 0,05 % Mindestclustergröße und 30 äußere Splits, IDs 0–29. Der aktuelle Dreiervergleich verwendet dieselben 30 Splits aller Methoden. Die CellCNN- und SVM-Zusammenfassungen über 100 Splits sind separat gekennzeichnet.

Die Erklärungen begründen die tatsächlich gewählten Einstellungen. Eine plausible Begründung ist kein experimenteller Nachweis, dass die Wahl gegenüber Alternativen besser ist. Für diese Dokumentation wurden keine Modelle neu trainiert.

## Quellen und Orientierung

- [CellCNN-Notebook](../notebooks/04c_cellcnn.ipynb)
- [SVM-Notebook](../notebooks/04b_svm.ipynb)
- [Citrus-Notebook](../notebooks/04d_citrus.ipynb)
- [Gemeinsame Datenaufbereitung und Spendersplits](../notebooks/04a_data_qc_and_splits.ipynb)
- [Methodenvergleich](../notebooks/04e_comparison.ipynb)
- [Lokales CellCNN-Paper](../CellCNN.pdf), insbesondere die Methoden auf Seiten 6–9
- [Supplementary Methods des CellCNN-Papers](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fncomms14825/MediaObjects/41467_2017_BFncomms14825_MOESM2828_ESM.pdf), insbesondere Seite 6 zu Citrus
- [Offizieller CellCNN-Modellcode](https://raw.githubusercontent.com/eiriniar/CellCnn/master/cellCnn/model.py)
- [Offizieller Citrus-Quellstand](https://github.com/nolanlab/citrus/tree/d02baae544abdc403704aaceb75d1e7931a0331c)
- [LinearSVC-Dokumentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- [Projektplan](../AUFGABE_4_PLAN.md)

## Inhalt

1. [CellCNN](#1-cellcnn)
2. [Lineare Single-Cell-SVM](#2-lineare-single-cell-svm)
3. [Citrus](#3-citrus)

---

# 1. CellCNN

**Unser Notebook setzt die zentrale CellCNN-Methode und die wichtigsten Einstellungen des NK-/CMV-Benchmarks um. Es ist eine papernahe Neuimplementierung mit einigen bewussten Abweichungen.** Die folgende Erklärung beschreibt den tatsächlichen Ablauf des Full-Modus einschließlich der vorhandenen Ergebnisse.

Grundlage sind das CellCNN-Notebook, das Notebook zur Datenaufbereitung und Split-Erzeugung sowie das lokale Paper. Wenn ein Detail nur im Referenzcode nachvollziehbar ist, wird das ausdrücklich von einer Angabe im Paper unterschieden.

Bei den Begründungen sind zwei Dinge auseinanderzuhalten: Manche Entscheidungen übernehmen wir direkt aus dem Paper; andere folgen unseren Projektregeln. **Dass unsere Abweichungen bessere Ergebnisse erzeugen, haben wir nicht durch einen eigenen Vergleich nachgewiesen.**

## 1.1 Was soll CellCNN bei uns lernen?

Die Eingabe besteht aus vielen einzelnen Zellmessungen eines Spenders. Das bekannte Ziel ist der **CMV-Status des Spenders**:

- `0`: CMV−.
- `1`: CMV+.

CellCNN soll aus den Zellmessungen vorhersagen, welcher dieser beiden Gruppen ein Spender angehört. Dabei soll es selbst lernen, welche Kombinationen von Markern und welche Zellpopulationen dafür informativ sind.

Wir kennen **keine Zelllabels**, die für jede einzelne Zelle angeben, ob sie „CMV-typisch“ ist. Deshalb wird auch keine einzelne Zelle direkt gegen ein solches Label trainiert. Das Modell erhält eine **Gruppe von Zellen zusammen mit dem Spenderlabel**.

Das heißt *Multiple Instance Learning*: Eine Menge von Beobachtungen erhält ein gemeinsames Ziel.

**Warum so?** Der CMV-Status ist auf Spenderebene bekannt, und möglicherweise enthält nur eine kleine Untergruppe seiner Zellen das relevante Signal. Würden wir jede Zelle eines CMV+-Spenders als individuell „positiv“ behandeln, wäre das eine stärkere und biologisch fragwürdige Annahme.

**Im Paper:** Genau dieses Lernen aus Zellgruppen und zugehörigen Probenlabels ist die Grundidee von CellCNN.

## 1.2 Welche Daten lesen wir ein?

Im Full-Modus verwenden wir `gated_alive`:

- **20 Spender**.
- **11 CMV− und 9 CMV+**.
- **3.438.750 Zellen insgesamt**.
- **37 Marker pro Zelle**.

Die FCS-Dateien sind bereits auf lebende Zellen und gegen Doubletten bereinigt. Unser Notebook beginnt also mit den **Messwerten aus bereits vorgegaten Dateien**, nicht mit vollständig unverarbeiteten Instrumentdaten.

`gated_alive` enthält die breite PBMC-Population. PBMCs sind mononukleäre Blutzellen, beispielsweise T-, B- und NK-Zellen.

Wir verwenden für den Hauptvergleich **keine vorherige Einschränkung auf NK-Zellen**.

**Warum so?** Das Modell soll die relevante Population innerhalb einer größeren Zellmischung finden. Wenn wir vorher nur NK-Zellen auswählen würden, würden wir bereits biologisches Vorwissen in die Eingabe einbauen und die Suchaufgabe verändern.

**Im Paper:** Der NK-/CMV-Benchmark verwendet ebenfalls PBMCs nach Entfernung toter Zellen und Doubletten. Von ursprünglich 21 Personen wird Probe 008 ausgeschlossen. Unsere bereitgestellten 20 Proben entsprechen dieser Benchmark-Zusammensetzung; das CellCNN-Notebook trifft keine neue Ausreißerentscheidung.

Der voreingestellte **Smoke-Modus** verwendet dagegen `gated_NK` und stark reduzierte Trainingsparameter. Er dient technischen Tests. Die nachfolgende Erklärung und die genannten Resultate beziehen sich auf den Full-Modus.

## 1.3 Wie ordnen wir Messwerte, Marker und Labels korrekt zu?

Jede FCS-Datei wird über ihre Spender-ID mit dem Label aus der Labeldatei verbunden.

Die Markerliste legt fest, welche Spalten verwendet werden und in welcher Reihenfolge. Technische Kanäle wie `Time`, `Cell_length`, `Dead`, `DNA1` und `DNA2` gehen nicht in das Modell ein.

Die Marker werden über die konsistenten FCS-Kurznamen ausgewählt. Dadurch vermeiden wir Probleme durch unterschiedliche Schreibweisen der Detektornamen.

FlowKit liest die Werte mit `source="raw"` ein. Anschließend führen wir die Transformation selbst explizit aus.

Die Dateien haben einen bekannten Header-Offsetfehler. `ignore_offset_error=True` ermöglicht das Lesen; die Originaldateien werden dadurch nicht verändert.

**Warum so?**

- Ein falsch zugeordnetes Spenderlabel würde das gesamte Training verfälschen.
- Eine vertauschte Markerreihenfolge würde beim Modell dieselbe Gewichtung auf unterschiedliche Marker anwenden.
- Eine explizite Markerliste verhindert, dass technische Messkanäle versehentlich als biologische Prädiktoren dienen.
- Explizite Transformationen machen nachvollziehbar, welche Werte tatsächlich in das Modell gelangen.

**Im Paper:** Die biologische Markerauswahl ist grundsätzlich vergleichbar. Das Paper nennt 36 Marker; wir verwenden die 37 Einträge der bereitgestellten Markerdatei. Das ist eine dokumentationspflichtige Abweichung. Die genaue FlowKit-Behandlung ist eine technische Entscheidung unserer Neuimplementierung.

## 1.4 Zuerst transformieren wir die Messwerte mit ArcSinh

Für jeden Messwert berechnen wir:

$$
a=\operatorname{arcsinh}(x/5)
$$

Dabei bedeutet:

- $x$: ursprünglicher FCS-Messwert eines bestimmten Markers in einer bestimmten Zelle.
- $5$: der festgelegte Kofaktor.
- $\operatorname{arcsinh}$: die inverse hyperbolische Sinusfunktion.
- $a$: der transformierte Messwert.

Die Funktion verhält sich nahe null ungefähr linear. Große positive Werte werden ähnlich wie durch einen Logarithmus komprimiert. Auch null und negative Werte sind zulässig.

**Warum so?** Cytometrische Markerwerte können sehr unterschiedliche und stark rechtsschiefe Wertebereiche haben. Ohne Kompression könnten besonders große Messwerte das Lernen stark beeinflussen. Ein einfacher Logarithmus wäre für null und negative Hintergrundwerte problematisch.

Der Kofaktor wird **nicht aus unseren Daten geschätzt**. Deshalb dürfen wir die Transformation bereits vor der Aufteilung auf Training und Test anwenden: Dabei fließt keine Information zwischen Spendern.

Wir entfernen anhand der Werte keine weiteren Zellen und führen keine zusätzliche Batch-Korrektur durch. Die vorgeschaltete QC dokumentiert mögliche technische Unterschiede; sie liefert hier keine ausreichende Grundlage für eine zusätzliche Korrektur mit Referenzdaten.

**Im Paper:** Wir orientieren uns an der ArcSinh-Vorverarbeitung der CellCNN-Analyse. Die konkrete Wahl `arcsinh(x / 5)` ist unsere festgelegte papernahe Vorverarbeitung; sie sollte nicht mit einer auf unserem Datensatz optimierten Transformation verwechselt werden.

## 1.5 Wir teilen die Spender in äußere Trainings- und Testmengen auf

Die Splits werden einmal im vorgeschalteten Notebook erzeugt und von allen drei Klassifikationsmethoden übernommen.

Für jede der **100 Wiederholungen** wählen wir zufällig:

| Gruppe | Äußeres Training | Äußerer Test |
|---|---:|---:|
| CMV− | 7 | 4 |
| CMV+ | 7 | 2 |
| Insgesamt | 14 | 6 |

Innerhalb einer Wiederholung gehört jeder Spender vollständig zu genau einer dieser Mengen.

Die Zufallszahlen basieren auf dem Master-Seed `12345`. Daraus wird für jede Wiederholung ein eigener Split-Seed erzeugt.

**Warum so?**

- **Spenderweise Aufteilung:** Zellen desselben Spenders ähneln sich. Eine Aufteilung seiner Zellen auf Training und Test würde die Leistung für neue Spender nicht sauber messen.
- **Sieben Trainingsspender pro Klasse:** Das äußere Training ist damit ausgeglichen.
- **100 Wiederholungen:** Bei nur 20 Spendern hängt das Ergebnis stark von der konkreten Testmenge ab. Die Wiederholungen zeigen diese Schwankung.
- **Identische Splits für alle Methoden:** Leistungsunterschiede sollen nicht dadurch entstehen, dass eine Methode einfachere Testspender bekommt. Der aktuelle Dreiervergleich verwendet die 30 auch für Citrus vorliegenden Splits.

Das ist **Monte-Carlo-Cross-Validation**, keine gewöhnliche 100-fache Aufteilung in disjunkte Folds. Ein Spender kann über verschiedene Wiederholungen mehrfach im Test erscheinen.

**Im Paper:** Dieses Schema mit 100 Wiederholungen, 7 Trainingsspendern pro Klasse und 6 Testspendern entspricht dem NK-Benchmark. Unsere konkreten Zufallsaufteilungen sind damit nicht automatisch dieselben wie im Paper.

## 1.6 Innerhalb der 14 Trainingsspender erzeugen wir drei innere Folds

Die 14 äußeren Trainingsspender werden mittels stratifizierter dreifacher Cross-Validation aufgeteilt.

*Stratifiziert* bedeutet: Die Klassenverteilung wird möglichst gleichmäßig auf die Folds verteilt.

Nacheinander übernimmt jeder Fold die Rolle der inneren Validierung:

| Innerer Durchlauf | Spender zum Lernen der Gewichte | Spender zur Validierung |
|---|---:|---:|
| Ein Durchlauf | 9 | 5 |
| Ein weiterer Durchlauf | 9 | 5 |
| Der verbleibende Durchlauf | 10 | 4 |

Die äußeren sechs Testspender bleiben dabei unberührt.

**Warum so?** Wir brauchen Daten, die uns beim Stoppen des Trainings und bei der Modellauswahl helfen. Wenn wir dafür die äußeren Testspender verwenden würden, wären sie anschließend kein unabhängiger Test mehr.

Die inneren Validierungsdaten sind allerdings ebenfalls **keine abschließende unabhängige Leistungsprüfung**: Wir verwenden sie ausdrücklich zur Auswahl.

**Im Paper:** Die innere dreifache CV entspricht dem beschriebenen NK-Benchmark.

## 1.7 Für jeden Modellkandidaten fitten wir einen eigenen StandardScaler

Für jeden inneren Trainingsspender ziehen wir **20.000 Zellen ohne Zurücklegen**. Wir führen diese Stichproben zusammen und bestimmen für jeden Marker den Mittelwert und die Standardabweichung.

Danach standardisieren wir:

$$
z_{ij}=\frac{a_{ij}-\mu_j}{\sigma_j}
$$

Dabei bedeutet:

- $i$: eine Zelle.
- $j$: ein Marker.
- $a_{ij}$: ArcSinh-transformierter Wert des Markers $j$ in Zelle $i$.
- $\mu_j$: Mittelwert des Markers $j$, geschätzt aus den ausgewählten inneren Trainingszellen.
- $\sigma_j$: zugehörige Standardabweichung.
- $z_{ij}$: standardisierter Wert, den CellCNN erhält.

Bei neun Trainingsspendern wird der Scaler somit auf 180.000 Zellen gefittet, bei zehn auf 200.000.

**Es gibt einen gemeinsamen Scaler pro Kandidat, keinen eigenen Scaler pro Spender.** Seine Parameter werden unverändert auf sämtliche benötigten Zellen der Trainings-, Validierungs- und Testspender angewendet.

**Warum so?**

- Vergleichbare Markerskalen erleichtern das Lernen und machen die L2-Regularisierung sinnvoller.
- Gleich viele Zellen pro Trainingsspender verhindern eine stärkere Gewichtung zellreicher Proben bei der Skalierung.
- Trainingsexklusive Schätzung verhindert Datenleckage.
- Eine separate Standardisierung jedes Spenders könnte biologische Unterschiede zwischen Spendern entfernen.

Die Begrenzung auf 20.000 Zellen ist eine praktische Projektentscheidung. Wir haben nicht nachgewiesen, dass genau diese Zahl optimal ist.

**Referenzvergleich:** Der offizielle Code fittet den Scaler auf allen zusammengeführten Trainingszellen. Unsere spenderbalancierte Stichprobe ist eine Abweichung.

Ein Beispiel verdeutlicht die Gewichtung:

| Beispiel | Spender A | Spender B |
|---|---:|---:|
| Vorhandene Zellen | 100.000 | 20.000 |
| Anteil an der Skalierung im Referenzcode | 83,3 % | 16,7 % |
| Bei uns verwendete Zellen | 20.000 | 20.000 |
| Anteil an unserer Skalierung | 50 % | 50 % |

Die übrigen Trainingszellen werden nicht verworfen; sie helfen lediglich nicht beim Schätzen dieses Scalers. Unsere Skalierung kann andere Mittelwerte und Standardabweichungen liefern und damit das Training beeinflussen. Dass sie bessere Ergebnisse liefert, ist nicht nachgewiesen.

Ein zusätzliches Detail: Der Seed hängt bei uns auch von der Filterzahl ab. Deshalb können Kandidaten mit unterschiedlichen Filterzahlen leicht unterschiedliche Scaler-Stichproben erhalten.

## 1.8 Wir erzeugen die Zellgruppen, auf denen das Modell lernt

Für jeden inneren Trainingsspender erzeugen wir:

- **200 Inputs**;
- mit jeweils **3.000 Zellen**;
- zufällig **mit Zurücklegen** gezogen.

Ein Input hat somit die Form **3.000 Zellen × 37 Marker**. Alle 200 Inputs eines Spenders erhalten dessen CMV-Label.

Bei neun Trainingsspendern entstehen 1.800 Trainingsinputs; bei zehn entstehen 2.000.

Für die inneren Validierungsspender erzeugen wir ebenfalls 200 Inputs mit jeweils 3.000 Zellen.

**Mit Zurücklegen** bedeutet: Dieselbe ursprüngliche Zelle kann innerhalb eines Inputs mehrfach gezogen werden. Auch verschiedene Inputs können dieselben Zellen enthalten.

Die Inputs werden für einen Kandidaten **einmal erzeugt und gespeichert**. In späteren Epochen werden sie nicht neu gezogen; lediglich die Reihenfolge der Trainingsinputs wird gemischt.

**Warum so?**

- Unterschiedliche Zellgruppen zeigen dem Modell verschiedene Ausschnitte desselben Spenders.
- Eine feste Größe erlaubt eine einfache gemeinsame Verarbeitung.
- 3.000 Zellen erhöhen gegenüber sehr kleinen Inputs die Chance, seltene relevante Zellen zu erfassen.
- 200 Inputs pro Spender geben jedem Spender dasselbe Gewicht in der Trainingsverlustfunktion.
- Feste Validierungsinputs verhindern, dass der beobachtete Validierungsverlust allein wegen einer neuen Stichprobe springt.

Die vielen Inputs erhöhen **nicht** die Zahl unabhängiger Spender. 1.800 Inputs aus neun Spendern bleiben biologisch neun unabhängige Beobachtungen.

**Im Paper:** 3.000 Zellen, 200 Inputs pro Probe und Sampling mit Zurücklegen entsprechen den angegebenen NK-Benchmark-Einstellungen. Das Paper nennt außerdem grundsätzlich klassenbalancierte Inputs. Bei uns sind die Spender gleich gewichtet; wegen der inneren Aufteilung können die Klassen dort leicht unterschiedlich viele Inputs haben.

## 1.9 Die erste Modellschicht berechnet lernbare Zellantworten

Wir testen Modelle mit **3, 4 oder 5 Filtern**.

Jeder Filter besitzt ein Gewicht für jeden der 37 Marker und einen zusätzlichen Verschiebungswert, den Bias.

Für eine Zelle berechnet ein Filter:

$$
r_{if}=\max\left(0,\ b_f+\sum_{j=1}^{37}w_{fj}z_{ij}\right)
$$

Dabei bedeutet:

- $i$: die betrachtete Zelle.
- $f$: der betrachtete Filter.
- $j$: der Marker.
- $z_{ij}$: standardisierter Markerwert.
- $w_{fj}$: lernbares Gewicht von Marker $j$ im Filter $f$.
- $b_f$: lernbarer Bias dieses Filters.
- $\sum$: Addition der Beiträge aller 37 Marker.
- $\max(0,\ldots)$: ReLU; negative Ergebnisse werden null.
- $r_{if}$: nichtnegative Antwort der Zelle auf diesen Filter.

Ein Filter kann beispielsweise lernen, stark auf Zellen mit einer bestimmten Kombination hoher und niedriger Markerwerte zu reagieren. Diese Kombination geben wir nicht vorher biologisch vor.

**Warum so?**

- Einzelne Marker können weniger informativ sein als ihre Kombination.
- Dieselben Gewichte werden auf jede Zelle angewendet.
- Es wird keine räumliche Nachbarschaft zwischen Zellen angenommen.
- ReLU ermöglicht eine nichtlineare Antwort: Manche Zellen aktivieren den Filter, andere nicht.

Unser `nn.Linear` wird auf die letzte Dimension jeder Zellmessung angewendet. Das erfüllt hier die Funktion einer Faltung mit Kernelbreite 1.

**Im Paper/Referenzcode:** Dieser Aufbau entspricht der CellCNN-Architektur. Der offizielle Code verwendet ebenfalls einen linearen zellweisen Filter und anschließend ReLU.

## 1.10 Pro Filter mitteln wir die stärksten 1 % der Zellantworten

Nach der ersten Schicht hat jede der 3.000 Zellen eine Antwort auf jeden Filter.

Für jeden Filter wählen wir separat die **30 größten Antworten** aus und bilden deren Mittelwert.

Allgemein:

$$
h_f=\frac{1}{k}\sum_{i\in T_f}r_{if}
$$

Dabei bedeutet:

- $f$: ein Filter.
- $r_{if}$: Antwort der Zelle $i$ auf diesen Filter.
- $T_f$: Menge der Zellpositionen mit den höchsten Antworten auf Filter $f$.
- $k$: Anzahl dieser Positionen, bei uns 1 % der Inputgröße, abgerundet und mindestens eine.
- $h_f$: gepoolter Wert dieses Filters für den gesamten Input.

Bei 3.000 Zellen ist $k=30$, bei 20.000 Zellen ist $k=200$.

**Warum so?**

- Ein Mittelwert über alle Zellen könnte ein Signal einer seltenen Zellpopulation stark verdünnen.
- Nur das einzelne Maximum wäre stärker von einer einzelnen extremen Zellmessung abhängig.
- Der Mittelwert der höchsten 1 % konzentriert sich auf stark reagierende Zellen und verwendet gleichzeitig mehrere Antworten.

Dabei wählen unterschiedliche Filter möglicherweise unterschiedliche Zellen aus. Dieselbe Zelle darf auf mehrere Filter stark reagieren.

**Die 1 % sind kein vorab bestimmtes biologisches Zellcluster.** Es sind jeweils die Zellen mit den höchsten gelernten Antworten innerhalb des aktuellen Inputs.

**Im Paper:** Genau dieses Top-1-%-Pooling ist für den NK-Benchmark angegeben. Wir haben den Anteil übernommen und nicht selbst optimiert.

Die Zellreihenfolge spielt für das Ergebnis keine Rolle: Eine Umordnung verändert weder die zellweisen Antworten noch deren höchste Werte.

## 1.11 Aus den gepoolten Antworten entsteht die CMV-Wahrscheinlichkeit

Nach dem Pooling bleiben pro Input nur noch 3, 4 oder 5 Zahlen übrig: eine pro Filter.

Eine zweite lineare Schicht berechnet zwei Klassenwerte:

$$
\ell_c=d_c+\sum_{f=1}^{F}v_{cf}h_f
$$

Dabei bedeutet:

- $c$: Klasse, entweder 0 für CMV− oder 1 für CMV+.
- $F$: Anzahl der Filter.
- $h_f$: gepoolte Antwort des Filters $f$.
- $v_{cf}$: lernbares Gewicht von Filter $f$ zur Klasse $c$.
- $d_c$: Bias der Klasse $c$.
- $\ell_c$: unbeschränkter Klassenwert, auch *Logit* genannt.

Softmax wandelt die beiden Klassenwerte in Wahrscheinlichkeiten um:

$$
p_1=\frac{\exp(\ell_1)}{\exp(\ell_0)+\exp(\ell_1)}
$$

Dabei ist:

- $\exp$: die Exponentialfunktion.
- $\ell_0,\ell_1$: die beiden Klassenwerte.
- $p_1$: modellierte Wahrscheinlichkeit für CMV+.
- Die Wahrscheinlichkeit für CMV− ist $1-p_1$.

**Warum so?** Das Modell soll mehrere gefundene Zellmuster zu einer gemeinsamen Spenderentscheidung kombinieren. Softmax liefert dafür zwei zusammengehörige Klassenwahrscheinlichkeiten.

**Im Paper:** Eine Ausgabe mit einem Knoten pro Klasse und Softmax entspricht der beschriebenen Klassifikationsarchitektur.

Im Trainingscode gibt `forward()` zunächst die Logits zurück. Die verwendete Cross-Entropy-Funktion verarbeitet diese intern passend; bei der Vorhersage wird Softmax ausdrücklich aufgerufen.

Die Werte sind Modellwahrscheinlichkeiten. Eine zusätzliche Wahrscheinlichkeitskalibrierung findet nicht statt.

## 1.12 Was wird beim Training optimiert?

Wir minimieren **Cross-Entropy plus L2-Strafe**:

$$
L=-\frac{1}{B}\sum_{b=1}^{B}\log p_{b,y_b}
+\lambda\left(\sum_{f,j}w_{fj}^{2}+\sum_{c,f}v_{cf}^{2}\right)
$$

Dabei bedeutet:

- $L$: gesamter Verlust eines Trainingsbatches.
- $B$: Anzahl der Zellgruppen im Batch, normalerweise 128.
- $b$: Index einer Zellgruppe im Batch; dieser Index ist nicht der zuvor verwendete Filterbias $b_f$.
- $y_b$: tatsächliches Spenderlabel dieser Zellgruppe.
- $p_{b,y_b}$: Wahrscheinlichkeit, die das Modell der richtigen Klasse zuweist.
- $\log$: natürlicher Logarithmus.
- $w_{fj}$: Gewichte der Zellfilter.
- $v_{cf}$: Gewichte der Ausgabeschicht.
- $\lambda=0{,}0001$: Stärke der L2-Strafe.
- Die Summen der Quadrate umfassen alle Gewichte der jeweiligen Schicht.

Die Bias-Werte werden bei uns **nicht** mit dieser L2-Strafe belegt.

Der erste Term wird kleiner, wenn das Modell der richtigen Klasse eine höhere Wahrscheinlichkeit gibt. Der zweite Term bestraft große Gewichte.

**Warum so?**

- Cross-Entropy passt zum Ziel einer binären Klassifikation über Klassenwahrscheinlichkeiten.
- L2 begrenzt sehr große Gewichte und soll Überanpassung reduzieren.
- Bei nur wenigen unabhängigen Spendern ist eine Begrenzung der Modellkomplexität sinnvoll.

Es gibt keinen zusätzlichen Verlust, der einzelne Zellen klassifiziert. Die Filter lernen ausschließlich über ihre Wirkung auf die Vorhersage der Zellgruppe.

**Im Paper:** Cross-Entropy und L2 mit Stärke $10^{-4}$ entsprechen den Angaben. Die konkrete Strafe auf die beiden Gewichtsmatrizen entspricht dem Referenzaufbau.

## 1.13 Wie werden die Gewichte aktualisiert und wann stoppen wir?

Die Gewichte beginnen mit der zufälligen Standardinitialisierung der PyTorch-Schichten.

Für jeden Batch läuft:

1. Vorhersagen berechnen.
2. Verlust berechnen.
3. Mit Backpropagation bestimmen, wie die Parameter den Verlust beeinflussen.
4. Parameter mit **Adam** aktualisieren.

Adam verwendet Informationen aus bisherigen Gradienten, um die Aktualisierung jeder Modellgröße anzupassen.

Die Einstellungen sind:

| Einstellung | Wert | Grund |
|---|---:|---|
| Lernrate | 0,01 | Aus dem NK-Benchmark übernommen |
| Batchgröße | 128 Zellgruppen | Paper-Einstellung; begrenzt den Speicherbedarf pro Schritt |
| Maximale Epochen | 100 | Paper-Einstellung |
| Dropout | keines | Entspricht dem beschriebenen NK-Benchmark |
| Early-Stopping-Geduld | 5 Epochen | Paper-Einstellung |

Eine **Epoche** ist ein vollständiger Durchlauf durch die einmal erzeugten Trainingsinputs.

Nach jeder Epoche berechnen wir den regularisierten Verlust auf den festen inneren Validierungsinputs. Dabei werden keine Gewichte verändert.

Wir sichern den Modellzustand, wenn sich dieser Verlust um mehr als `0.000001` verbessert. Nach fünf aufeinanderfolgenden Epochen ohne solche Verbesserung stoppen wir. Anschließend laden wir den **besten gesicherten Zustand**, nicht einfach den zuletzt erreichten.

**Warum so?** Ein weiter sinkender Trainingsverlust garantiert keine bessere Leistung auf neuen Spendern. Early Stopping verwendet die Validierung als Signal dafür, wann weiteres Training nicht mehr hilft.

**Im Paper/Referenzcode:** Die grundlegenden Einstellungen und das Stoppen anhand des Validierungsverlusts stimmen überein. Die explizite Verbesserungstoleranz ist ein Detail unserer Implementierung. PyTorch-Initialisierung und numerische Abläufe sind keine bitgenaue Reproduktion der ursprünglichen Keras-/Theano-Umgebung.

Die gespeicherten Full-Ergebnisse wurden auf CUDA berechnet. Seeds und deterministische Einstellungen erhöhen die Reproduzierbarkeit; identische Ergebnisse über beliebige Hardware- und Bibliotheksversionen hinweg sind damit nicht garantiert.

## 1.14 Wie wählen wir unter den trainierten Kandidaten aus?

Pro äußerem Split entstehen **3 innere Folds × 3 Filterzahlen = 9 Kandidaten**.

Ein Kandidat besteht aus seinem trainierten Netz und seinem Scaler.

Nach dem Training bewerten wir jeden Kandidaten noch einmal **spenderweise** auf seinen inneren Validierungsspendern:

- Pro Validierungsspender fünf Stichproben.
- Jeweils 20.000 Zellen ohne Zurücklegen.
- Für jede Stichprobe eine CMV+-Wahrscheinlichkeit.
- Mittelwert dieser fünf Wahrscheinlichkeiten.
- Vorhersage CMV+, wenn der Mittelwert mindestens 0,5 ist.

Dann sortieren wir die neun Kandidaten nach:

1. Höchster Validierungsaccuracy.
2. Bei Gleichstand: höchster Validierungs-ROC-AUC.
3. Danach: niedrigstem besten Validierungsverlust.
4. Danach: kleinerer Filterzahl.
5. Danach: kleinerem Fold-Index.

*Accuracy* ist hier der Anteil korrekt klassifizierter Validierungsspender.

**Warum so?**

- Die primäre Auswahl anhand der Accuracy folgt dem Paper.
- Eine Stimme pro Spender entspricht unserer unabhängigen Beobachtungseinheit.
- Bei vier oder fünf Validierungsspendern gibt es häufig Gleichstände.
- Die zusätzlichen Regeln machen die Entscheidung eindeutig und reproduzierbar.

Die letzten beiden Regeln bevorzugen bei sonst vollständigem Gleichstand das kleinere Modell und legen abschließend eine feste Reihenfolge fest.

**Hier gibt es zwei getrennte Validierungsschritte:**

| Zweck | Verwendete Datenrepräsentation |
|---|---|
| Beste Trainingsepoche finden | 200 feste Inputs mit je 3.000 Zellen pro Validierungsspender |
| Besten Kandidaten finden | Fünf Vorhersagen mit je 20.000 Zellen, pro Spender gemittelt |

**Papervergleich:** Das Paper beschreibt Random Search und anschließend die Wahl des Netzes mit der höchsten Validierungsgenauigkeit. Wir verwenden stattdessen die feste Suche über 3, 4 und 5 Filter und eigene Gleichstandsregeln. Der Referenzcode bewertet generierte Validierungsinputs; unsere spenderweise Bewertung verändert dieses Detail.

Da jeder Kandidat einen eigenen Seed hat, variieren neben den Gewichten auch die gezogenen Zellstichproben. Der Kandidatenvergleich isoliert somit nicht ausschließlich den Effekt der Filterzahl.

## 1.15 Wir trainieren das ausgewählte Modell nicht noch einmal neu

Das ausgewählte Netz wird direkt auf die sechs äußeren Testspender angewendet.

**Seine Gewichte wurden daher mit neun oder zehn Spendern gelernt.** Die übrigen vier oder fünf äußeren Trainingsspender dienten zur Validierung und Auswahl.

Wir mitteln weder die drei inneren Modelle noch trainieren wir abschließend auf allen 14 äußeren Trainingsspendern neu.

**Warum so?** Damit testen wir genau den bereits trainierten Modellzustand, der die Auswahl gewonnen hat. Ein erneutes Training auf allen 14 Spendern wäre ein anderer Ablauf und würde auch eine neue Entscheidung über die Trainingsdauer erfordern.

**Im Paper:** Für den NK-Benchmark wird ausdrücklich das Netz mit der höchsten Validierungsaccuracy aus den inneren CV-Läufen für die finale Testvorhersage verwendet. Dieses Vorgehen übernehmen wir.

Eine klassische Auswahl der Filterzahl anhand des **Durchschnitts über drei Folds** mit anschließendem Neutraining wäre ebenfalls denkbar, ist aber nicht unser Verfahren.

## 1.16 Wie entsteht die endgültige Vorhersage eines Testspenders?

Für jeden der sechs Testspender:

1. Ziehen wir 20.000 Zellen ohne Zurücklegen.
2. Verwenden wir die ArcSinh-transformierten Werte und wenden den Scaler des ausgewählten Modells an.
3. Berechnen wir Filterantworten.
4. Mitteln wir pro Filter die höchsten 200 Antworten.
5. Berechnen wir die CMV+-Wahrscheinlichkeit.
6. Wiederholen wir das für insgesamt fünf Zellstichproben.

Die bereits eingelesenen Daten sind schon ArcSinh-transformiert; diese Transformation wird nicht ein zweites Mal angewendet.

Der endgültige Score lautet:

$$
s_d=\frac{1}{5}\sum_{r=1}^{5}p_{dr}
$$

Dabei bedeutet:

- $d$: ein Testspender.
- $r$: eine der fünf Stichproben.
- $p_{dr}$: CMV+-Wahrscheinlichkeit für Stichprobe $r$ von Spender $d$.
- $s_d$: endgültiger CMV+-Score dieses Spenders.

Es wird **nach Softmax gemittelt**, also über Wahrscheinlichkeiten, nicht über Logits.

Für das Klassenlabel gilt:

- $s_d\geq0{,}5$: CMV+.
- $s_d<0{,}5$: CMV−.

Innerhalb einer Stichprobe gibt es keine doppelt gezogenen Zellen. Die fünf Stichproben dürfen sich allerdings überschneiden. Es sind also nicht zwangsläufig 100.000 verschiedene Zellen.

Falls weniger als 20.000 Zellen vorhanden wären, würde die Vorhersagefunktion alle verfügbaren Zellen verwenden.

**Warum so?** Mehrere Stichproben sollen die Abhängigkeit von einer einzelnen zufälligen Zellwahl reduzieren. Die feste Schwelle 0,5 vermeidet eine zusätzliche Schwellenoptimierung.

**Im Paper:** Für den NK-Test wird **eine** Stichprobe mit 20.000 Zellen pro Spender angegeben. Unsere Mittelung über fünf Stichproben ist eine bewusste Abweichung. Sie ist im Projektplan und Notebook dokumentiert.

## 1.17 Was bedeutet das an einem echten Beispiel?

Im gespeicherten äußeren **Split 0** wird der Kandidat mit innerem Fold 1, vier Filtern und Validierungsaccuracy 0,80 ausgewählt.

Ein anderer Kandidat erreicht zwar eine Validierungs-ROC-AUC von 1,00, aber nur eine Accuracy von 0,75. Er wird deshalb nicht gewählt: **Accuracy hat in unserer Auswahl Vorrang.**

Das ausgewählte Netz liefert für die sechs Testspender:

| Spender | Tatsächliche Klasse | CMV+-Score | Vorhersage |
|---|---|---:|---|
| a_002 | CMV+ | 0,9817 | CMV+ |
| a_004 | CMV− | 0,0415 | CMV− |
| a_006 | CMV− | 0,4476 | CMV− |
| a_1a | CMV− | 0,9160 | CMV+ |
| a_2a | CMV− | 0,1475 | CMV− |
| a_4a | CMV+ | 0,9467 | CMV+ |

Der Spender `a_1a` wird falsch positiv klassifiziert.

Trotzdem liegt die ROC-AUC in diesem Split bei 1,00, weil beide tatsächlich positiven Spender höhere Scores haben als alle tatsächlich negativen Spender. Das verdeutlicht: **ROC-AUC bewertet die Rangfolge, nicht die Fehlerzahl an der Schwelle 0,5.**

## 1.18 Welche Kennzahlen berechnen wir?

Die Kennzahlen werden **für jeden äußeren Split separat auf seinen sechs Testspendern** berechnet.

| Kennzahl | Was sie misst | Warum wir sie verwenden |
|---|---|---|
| ROC-AUC | Wie gut positive Spender höhere Scores erhalten als negative | Primäre Benchmark-Metrik des Papers |
| Average Precision | Präzision bei zunehmender Erfassung positiver Spender, gewichtet nach Recall-Zuwächsen | Ergänzt die Rangbewertung aus Sicht der positiven Klasse |
| Trapezoidale PR-AUC | Fläche unter der Precision-Recall-Kurve durch trapezoidale Integration | Zusätzliche dokumentierte PR-Kennzahl |
| Balanced Accuracy | Mittelwert aus Erkennungsrate der positiven und der negativen Klasse bei Schwelle 0,5 | Gewichtet beide Klassen trotz 2 positiven und 4 negativen Testspendern gleich |

Dabei bedeutet:

- **Precision:** Anteil tatsächlich positiver Spender unter den als positiv eingestuften Spendern.
- **Recall/Sensitivität:** Anteil erkannter positiver Spender an allen tatsächlich positiven Spendern.
- **Spezifität:** Anteil korrekt negativ eingestufter Spender an allen tatsächlich negativen Spendern.

Average Precision und trapezoidale PR-AUC sind unterschiedliche Berechnungen und müssen nicht denselben Wert ergeben.

**Im Paper:** Für den NK-Benchmark wird die Test-ROC-AUC über die 100 Wiederholungen berichtet. Unsere weiteren Kennzahlen ergänzen diese Auswertung.

Aus den vorhandenen Testvorhersagen wurden folgende Werte neu berechnet:

| Kennzahl | Mittelwert über 100 Splits | Median |
|---|---:|---:|
| ROC-AUC | **0,8075** | 0,8750 |
| Average Precision | 0,8038 | 0,8333 |
| Trapezoidale PR-AUC | 0,7613 | 0,7917 |
| Balanced Accuracy | 0,6788 | 0,7500 |

Der ROC-AUC-Mittelwert bedeutet nicht „80,75 % aller Spender richtig klassifiziert“. Er beschreibt die mittlere Trennfähigkeit der Scores.

Die Ergebnisse schwanken zwischen den Splits. Wegen der wiederholten Verwendung derselben 20 Spender sind die 100 Split-Ergebnisse außerdem nicht 100 unabhängige Studien.

## 1.19 Was speichern wir als Endergebnis?

Der vollständige Lauf umfasst **900 Kandidatentrainings** und **100 ausgewählte Modelle**.

Gespeichert werden drei zentrale Tabellen:

| Ausgabe | Inhalt | Zweck |
|---|---|---|
| [Vorhersagen](tables/task4_cellcnn_predictions_gated_alive_full.csv) | 600 Zeilen: 6 Testspender × 100 Splits, jeweils Label, Score und Vorhersage | Leistungsbewertung und Methodenvergleich |
| [Kandidatenauswahl](tables/task4_cellcnn_selection_gated_alive_full.csv) | 900 Kandidaten mit Einstellungen, Validierungsmetriken und Auswahlmarkierung | Nachvollziehen, warum welches Modell gewählt wurde |
| [Modellparameter](tables/task4_cellcnn_filters_gated_alive_full.csv) | Filtergewichte, Biases, Ausgabegewichte und Scalerparameter der gewählten Netze | Modelle rekonstruieren und Zellantworten untersuchen |

Dazu kommt eine Konfigurationsdatei mit Parametern, Bibliotheksversionen sowie Prüfsummen von Eingaben und Trainingscode.

Nach abgeschlossenen Splits werden Zwischenstände geschrieben. Passende vorhandene Ergebnisse können später geladen werden.

**Warum so?** Wir können lange Läufe fortsetzen, Ergebnisse auf ihre Konfiguration zurückführen und die ausgewählten Modelle später ohne erneutes Training untersuchen.

**Papervergleich:** Das ist unsere technische Umsetzung der Reproduzierbarkeit. Das statistische Ergebnis des Benchmarks bleibt die Verteilung der Testleistungen über die Wiederholungen.

Es gibt am Ende dieses Notebooks **kein einziges endgültiges Modell, das auf allen 20 Spendern trainiert wurde**. Es gibt 100 ausgewählte Modelle zur Bewertung des Verfahrens.

## 1.20 Was sagen die gespeicherten Filter biologisch aus?

Ein Filter beschreibt zunächst nur eine gelernte Kombination standardisierter Markerwerte. Ob seine Aktivität die Entscheidung Richtung CMV+ oder CMV− verschiebt, hängt von der Ausgabeschicht ab.

Deshalb speichern wir zusätzlich:

$$
\Delta_f=v_{1f}-v_{0f}
$$

Dabei bedeutet:

- $f$: der Filter.
- $v_{1f}$: Ausgabegewicht dieses Filters für CMV+.
- $v_{0f}$: Ausgabegewicht dieses Filters für CMV−.
- $\Delta_f$: Unterschied der beiden Gewichte.

Bei ansonsten gleichen gepoolten Antworten bedeutet:

- $\Delta_f>0$: Eine höhere Antwort dieses Filters verschiebt die Entscheidung Richtung CMV+.
- $\Delta_f<0$: Sie verschiebt sie Richtung CMV−.

Ein großes positives Markergewicht allein bedeutet dagegen noch nicht, dass dieser Marker „CMV verursacht“ oder dass jede stark reagierende Zelle eine eindeutig identifizierte Zellart ist.

**Warum speichern wir das?** Aufgabe 5 soll nachvollziehen können, welche Zellen und Markerprofile mit den Vorhersagen verbunden sind.

**Im Paper:** Dort folgt auf die Klassifikation eine ausführliche Charakterisierung der ausgewählten Populationen, unter anderem mit einem Halbmaximum-Schwellenwert und einer Zusammenfassung wiederkehrender Populationen über die CV-Läufe.

Unser CellCNN-Notebook bereitet dafür die Parameter vor. Die eigentliche biologische Interpretation erfolgt anschließend separat. Das **Top-1-%-Pooling zur Vorhersage** und die **spätere Abgrenzung einer biologischen Population** sind zwei unterschiedliche Arbeitsschritte.

Für diese Erklärung wurden der Code und die gespeicherten Tabellen geprüft und die genannten Kennzahlen neu berechnet. Ein neues CellCNN-Training wurde nicht gestartet.

---

# 2. Lineare Single-Cell-SVM

**Unsere SVM lernt zunächst eine lineare Bewertungsregel für einzelne Zellen. Anschließend fassen wir die höchsten Zellbewertungen eines Spenders zu einem Spenderscore zusammen und vergleichen diesen mit einer zuvor bestimmten Entscheidungsschwelle.**

Das unterscheidet sie wesentlich von CellCNN: **Bei der SVM erfolgt das Lernen auf einzelnen, mit dem Spenderlabel beschrifteten Zellen. Die Zusammenfassung auf Spenderebene kommt erst danach.**

Diese Erklärung beschreibt den Full-Modus des SVM-Notebooks einschließlich der vorhandenen Ergebnisse.

Beim Papervergleich ist eine Einschränkung entscheidend: Das CellCNN-Paper beschreibt eine Single-Cell-SVM als Baseline. **Unser vollständiges Verfahren mit Top-1-%-Aggregation, innerer AUC-Auswahl und gelernter Spenderschwelle ist aber eine Projektanpassung.** Es ist keine identische Reproduktion eines vollständig beschriebenen CMV-SVM-Benchmarks.

## 2.1 Was wollen wir vorhersagen?

Das Ziel ist der CMV-Status eines Spenders:

- `0`: CMV−.
- `1`: CMV+.

Zur Verfügung stehen viele einzelne Zellmessungen, aber nur **ein bekanntes Label pro Spender**.

Unsere SVM benötigt jedoch ein Label für jede Trainingszeile. Deshalb erhält **jede ausgewählte Trainingszelle das Label ihres Spenders**.

Beispielsweise erhalten alle 10.000 verwendeten Zellen eines CMV+-Spenders das Trainingslabel `1`.

**Warum so?** Wir haben keine verlässlichen Labels für einzelne krankheitsassoziierte Zellen. Die Übertragung des Spenderlabels macht es möglich, trotzdem einen gewöhnlichen überwachten Zellklassifikator zu trainieren.

Das ist eine **schwache Beschriftung**: Eine Zelle eines CMV+-Spenders muss selbst kein CMV-typisches Merkmal besitzen. Viele Zelltypen können in beiden Gruppen nahezu gleich aussehen.

**Im Paper:** Diese Übertragung des Probenlabels auf jede einzelne Trainingszelle ist ausdrücklich Teil der beschriebenen Single-Cell-Baselines.

**Unterschied zu CellCNN:** CellCNN erhält das Label für die gesamte Zellgruppe. Es muss nicht jede einzelne Zelle dieser Gruppe in dieselbe Richtung klassifizieren. Unsere SVM wird dagegen über einen Verlust für jede einzelne Trainingszelle optimiert.

## 2.2 Welche Rohdaten verwenden wir?

Der Full-Modus verwendet dieselben Daten wie unser CellCNN-Hauptvergleich:

| Eigenschaft | Wert |
|---|---:|
| Datenstufe | `gated_alive` |
| Spender | 20 |
| CMV− | 11 |
| CMV+ | 9 |
| Zellen insgesamt | 3.438.750 |
| Marker pro Zelle | 37 |

Die Dateien enthalten bereits auf lebende Zellen und gegen Doubletten bereinigte PBMCs. Unser Notebook führt dieses Gating nicht erneut durch.

**Warum `gated_alive`?** Wir wollen untersuchen, ob die Methode informative Zellen innerhalb einer größeren Zellmischung findet. Eine vorherige Beschränkung auf NK-Zellen würde diese Aufgabe verändern und biologisches Vorwissen einbauen.

Der voreingestellte Smoke-Modus verwendet `gated_NK`, zwei äußere Splits und weniger Trainingszellen. Seine Ergebnisse dienen technischen Tests.

**Im Paper:** Für den NK-/CMV-Benchmark werden ebenfalls PBMCs nach Entfernung toter Zellen und Doubletten verwendet. Unsere gemeinsame Datengrundlage orientiert sich daran. Die konkrete SVM-Auswertung auf diesen Daten ist unsere zusätzliche Vergleichsmethode.

## 2.3 Wie lesen wir die Daten korrekt ein?

Das Notebook verbindet:

- die Spender-ID aus dem Dateinamen;
- das CMV-Label aus der Labeldatei;
- die Marker aus `NK_markers.csv`.

Die Markerliste bestimmt sowohl die Auswahl als auch die Reihenfolge der 37 Spalten. Technische Kanäle wie Zeit, Zelllänge oder DNA-Kanäle werden nicht als Klassifikationsmerkmale verwendet.

FlowKit liest die FCS-Werte mit `source="raw"`. Marker werden über die konsistenten FCS-Kurznamen identifiziert.

Der bekannte Header-Offsetfehler wird beim Lesen mit `ignore_offset_error=True` toleriert. Die Dateien bleiben unverändert.

Das Notebook kontrolliert unter anderem:

- Anzahl der Dateien und Marker;
- Übereinstimmung der Spenderlabels mit den Split-Tabellen;
- fehlende oder doppelte Markernamen;
- nicht-endliche Messwerte.

**Warum so?** Fehler bei Labels oder Markerreihenfolge würden unmittelbar falsche Zusammenhänge erzeugen. Explizites Einlesen verhindert außerdem unbemerkte zusätzliche Transformationen.

**Papervergleich:** Die biologische Datengrundlage ist papernah; FlowKit und diese Prüfungen sind technische Entscheidungen unserer Umsetzung. Wir verwenden 37 bereitgestellte Marker, während das Paper 36 nennt. Diese Differenz bleibt auch für die SVM bestehen.

## 2.4 Wir transformieren jeden Marker mit ArcSinh

Für jeden einzelnen Messwert gilt:

$$
a=\operatorname{arcsinh}(x/5)
$$

Dabei bedeutet:

- $x$: ursprünglicher Messwert eines Markers in einer Zelle.
- $5$: festgelegter Kofaktor.
- $\operatorname{arcsinh}$: inverse hyperbolische Sinusfunktion.
- $a$: transformierter Messwert.

Große Werte werden komprimiert; nahe null verhält sich die Funktion ungefähr linear. Null und negative Hintergrundwerte sind zulässig.

**Warum so?** Cytometrische Messwerte können stark rechtsschief sein. Ohne Transformation könnten extreme Intensitäten die nachfolgende Skalierung und Modellanpassung stark beeinflussen. Ein einfacher Logarithmus wäre bei null und negativen Werten problematisch.

Die Transformation benötigt keine aus den Daten gelernten Parameter. Deshalb darf sie bereits vor dem Training auf alle Spender angewendet werden.

**Papervergleich:** Wir verwenden dieselbe festgelegte papernahe Vorverarbeitung wie bei CellCNN. Die konkrete SVM-Pipeline mit diesen Einstellungen ist damit noch keine exakte Paper-Reproduktion.

Die Daten bleiben anschließend in getrennten Arrays pro Spender gespeichert.

## 2.5 Wir verwenden dieselben äußeren Spendersplits wie CellCNN

Für jede der 100 Wiederholungen gilt:

| Gruppe | Äußeres Training | Äußerer Test |
|---|---:|---:|
| CMV− | 7 | 4 |
| CMV+ | 7 | 2 |
| Insgesamt | 14 | 6 |

Die Splits stammen aus dem vorgeschalteten Daten-Notebook. Der Master-Seed ist `12345`; daraus wurden die einzelnen Split-Seeds erzeugt.

**Warum so?**

- Zellen desselben Spenders bleiben vollständig zusammen.
- Die äußeren Trainingsgruppen sind klassenbalanciert.
- Alle Methoden erhalten dieselben Trainings- und Testspender; der aktuelle Dreiervergleich nutzt die 30 auch für Citrus vorliegenden Splits.
- Die 100 Wiederholungen zeigen, wie empfindlich die Leistung gegenüber der Auswahl der Spender ist.

Ein Spender darf in verschiedenen Wiederholungen unterschiedliche Rollen haben. Innerhalb eines Splits darf er aber nicht gleichzeitig im Training und Test vorkommen.

**Im Paper:** Dieses äußere Schema entspricht dem NK-/CMV-Benchmark. Wir übertragen es auf unsere SVM, damit der Methodenvergleich unter denselben Bedingungen stattfindet.

**Die sechs Testspender werden weder zur Wahl der SVM-Einstellung noch zur Bestimmung der Entscheidungsschwelle verwendet.**

## 2.6 Innerhalb der 14 Trainingsspender verwenden wir dreifache Cross-Validation

Die äußeren Trainingsspender sind in drei stratifizierte innere Folds eingeteilt.

Nacheinander dient jeder Fold als Validierungsmenge:

- zweimal neun Spender zum Trainieren und fünf zum Validieren;
- einmal zehn Spender zum Trainieren und vier zum Validieren.

Stratifizierung verteilt beide CMV-Gruppen möglichst gleichmäßig.

**Warum so?** Die innere Validierung hilft uns, die Regularisierungsstärke und die spätere Spenderschwelle zu bestimmen. Der äußere Test bleibt für die abschließende Bewertung reserviert.

**Papervergleich:** Die dreifache innere CV folgt dem gemeinsamen NK-Benchmark-Schema. Wie wir daraus konkret SVM-Parameter und Schwelle bestimmen, ist unsere Ausgestaltung.

## 2.7 Wir ziehen 10.000 Trainingszellen pro Spender

Für jeden inneren Trainingsspender werden **10.000 Zellen zufällig ohne Zurücklegen** ausgewählt.

Daraus entstehen:

| Innere Trainingsspender | Trainingszellen |
|---:|---:|
| 9 | 90.000 |
| 10 | 100.000 |

Jede Trainingszeile enthält 37 Markerwerte und das Label ihres Spenders.

**Warum so?**

- Gleich viele Zellen pro Spender verhindern, dass zellreiche Proben die Trainingsverlustfunktion allein durch ihre Größe dominieren.
- 10.000 Zellen begrenzen den Rechenaufwand.
- Ohne Zurücklegen wird innerhalb einer Trainingsstichprobe keine Zelle doppelt verwendet.

Die Zahl 10.000 ist eine praktische Projektentscheidung. Wir haben nicht nachgewiesen, dass sie optimal ist.

Es gibt keine zusätzliche Klassengewichtung. Jeder Spender trägt gleich viele Zellverluste bei. Wenn ein innerer Trainingsfold beispielsweise fünf negative und vier positive Spender enthält, enthält er entsprechend mehr negativ beschriftete Trainingszellen.

**Papervergleich:** Das Paper beschreibt Trainingszellen aus den für CellCNN erzeugten Multi-Cell-Inputs. Wir ziehen stattdessen eine eigene feste Stichprobe ohne Zurücklegen. Diese SVM-Trainingsdaten sind also nicht identisch mit den CellCNN-Trainingsinputs.

Die 90.000 Zellzeilen bedeuten weiterhin nur neun unabhängige Trainingsspender.

## 2.8 Auf genau diesen Trainingszellen fitten wir den StandardScaler

Für jeden Marker berechnen wir auf der zusammengeführten Trainingsstichprobe Mittelwert und Standardabweichung.

Anschließend:

$$
z_{ij}=\frac{a_{ij}-\mu_j}{\sigma_j}
$$

Dabei bedeutet:

- $i$: eine Zelle.
- $j$: ein Marker.
- $a_{ij}$: ArcSinh-transformierter Markerwert.
- $\mu_j$: Mittelwert dieses Markers über die ausgewählten Trainingszellen.
- $\sigma_j$: entsprechende Standardabweichung.
- $z_{ij}$: standardisierter Wert für die SVM.

**Ein gemeinsamer Scaler gilt für alle Spender eines Modellfits.** Die Validierungszellen werden mit denselben Trainingsparametern transformiert.

**Warum so?** Die SVM bewertet Abstände und bestraft die Größe ihrer Gewichte. Unterschiedliche Markerskalen würden beeinflussen, wie leicht ein Marker durch ein kleines Gewicht einen großen Beitrag erzeugt.

Eine separate Standardisierung jedes Spenders könnte biologische Unterschiede zwischen Spendern entfernen. Ein Scaler auf allen Spendern würde dagegen Informationen aus Validierung und Test verwenden.

**Unterschied zu CellCNN:** Bei der SVM sind die 10.000 Zellen pro Spender gleichzeitig die Scaler-Stichprobe und die tatsächlichen Trainingszellen. CellCNN schätzt seinen Scaler separat auf 20.000 Zellen pro innerem Trainingsspender.

**Papervergleich:** Die genaue spenderbalancierte Skalierung dieser SVM ist eine Projektentscheidung; sie ist nicht als identische SVM-Einstellung aus dem Paper belegt.

Für denselben inneren Fold erhalten alle getesteten SVM-Einstellungen **dieselbe Zellstichprobe und denselben Scaler**. Der Sampling-Seed hängt vom Split und Fold ab, aber nicht vom getesteten Parameter.

## 2.9 Was lernt die lineare SVM?

Die SVM lernt eine einzige lineare Bewertungsfunktion:

$$
g_i=b+\sum_{j=1}^{37}w_jz_{ij}
$$

Dabei bedeutet:

- $i$: die betrachtete Zelle.
- $j$: ein Marker.
- $z_{ij}$: standardisierter Wert dieses Markers.
- $w_j$: gelerntes Gewicht des Markers.
- $b$: gelernter Intercept, also ein gemeinsamer Verschiebungswert.
- $g_i$: kontinuierlicher SVM-Score der Zelle.
- $\sum$: Summe der Beiträge aller 37 Marker.

Das Modell hat damit **37 Gewichte und einen Intercept**, also 38 gelernte Zahlen.

Bei sonst unveränderten Markerwerten erhöht ein positiver Markerkoeffizient den Zellscore; ein negativer senkt ihn.

Die Fläche mit $g_i=0$ ist die lineare Zellentscheidungsgrenze. Im 37-dimensionalen Markerraum nennt man sie eine Hyperebene.

**Warum eine lineare SVM?**

- Sie ist als Hauptbaseline im Projekt vorgegeben.
- Sie liefert eine einfache, nachvollziehbare Markergewichtung.
- Sie lässt sich auf vielen Zellzeilen effizient trainieren.
- Sie benötigt keine zusätzliche Suche nach einem nichtlinearen Kernel und dessen Parametern.

„Linear“ bezieht sich dabei auf die **transformierten und standardisierten Markerwerte**.

**Papervergleich:** Das Paper beschreibt eine SVM auf einzelnen Zellprofilen. Unsere konkrete Festlegung auf `LinearSVC` und die hier gewählten Einstellungen ist die Umsetzung der linearen Projektbaseline.

**Der Zellscore ist keine Wahrscheinlichkeit.** Ein Score von 1,2 bedeutet nicht 120 % CMV-Wahrscheinlichkeit. Wir verwenden weder Softmax noch eine nachträgliche Wahrscheinlichkeitskalibrierung.

## 2.10 Wie lernt die SVM ihre Gewichte?

Wir verwenden `LinearSVC` mit **L2-Regularisierung und Squared-Hinge-Verlust**.

Für unsere Einstellungen lässt sich das Optimierungsziel so schreiben:

$$
\min_{w,b}\left[
\frac12\left(\sum_{j=1}^{37}w_j^2+b^2\right)
+C\sum_{i=1}^{N}\left[\max(0,1-t_i g_i)\right]^2
\right]
$$

Dabei bedeutet:

- $\min_{w,b}$: Suche nach Gewichten und Intercept, die den Ausdruck möglichst klein machen.
- $w$: Gesamtheit der 37 Markergewichte.
- $b$: Intercept.
- $w_j$: Gewicht des Markers $j$.
- $N$: Anzahl der Trainingszellen.
- $i$: Index einer Trainingszelle.
- $t_i$: Spenderlabel in der mathematischen Kodierung −1 für CMV− und +1 für CMV+.
- $g_i$: SVM-Score der Zelle.
- $C$: Gewichtung der Zellverluste gegenüber der Regularisierung.
- $\max(0,\ldots)$: negative Ergebnisse werden null.
- Das Quadrat erzeugt den Squared-Hinge-Verlust.

Der erste Teil bestraft große Parameter. Der zweite Teil bestraft Zellen, deren Score nicht ausreichend zur zugewiesenen Klasse passt.

Für eine positiv beschriftete Zelle gilt beispielsweise:

| Zellscore | Einzelner Squared-Hinge-Verlust |
|---:|---:|
| 1,5 | 0 |
| 0,2 | 0,64 |
| −0,5 | 2,25 |

Ein positiver Score von 0,2 liegt zwar auf der richtigen Seite der Zellgrenze, erreicht aber noch nicht den angestrebten funktionalen Abstand von 1.

**Warum so?** Die SVM sucht eine Trennung mit Abstand zur Entscheidungsgrenze und erlaubt zugleich Verletzungen dieser Trennung. Das ist bei unseren schwachen Zelllabels nötig: Eine perfekte Trennung aller Zellen beider Spendergruppen ist nicht zu erwarten.

Bei unseren `LinearSVC`-Standardeinstellungen wird auch der Intercept regularisiert. Deshalb enthält die Formel $b^2$. Die Bibliothek repräsentiert den Intercept bei `intercept_scaling=1` über eine konstante zusätzliche Featurekomponente.

**Papervergleich:** Eine SVM als Zellbaseline ist beschrieben. Die genaue Squared-Hinge-Variante einschließlich Intercept-Regularisierung ist unsere konkrete Bibliotheksimplementierung; sie darf nicht ohne weiteren Nachweis dem Paper zugeschrieben werden.

## 2.11 Welche Rolle spielt der Parameter C?

Wir testen:

$$
C\in\{0{,}01;\ 0{,}1;\ 1\}
$$

$C$ ist keine Lernrate. Es steuert den Kompromiss zwischen kleinen Gewichten und dem Einhalten der Zelllabels:

- **Kleines C:** Die Regularisierung hat relativ mehr Einfluss.
- **Großes C:** Verletzungen der Zelllabels beziehungsweise des angestrebten Abstands erhalten relativ mehr Gewicht.

**Warum diese drei Werte?** Sie bilden eine kleine Suche über zwei Größenordnungen und halten den Rechenaufwand begrenzt. Das ist eine Projektentscheidung, keine nachgewiesen optimale Suchspanne.

Die weiteren Einstellungen lauten:

| Parameter | Wert |
|---|---|
| Verlust | `squared_hinge` |
| Regularisierung | `l2` |
| Maximale Optimierungsschritte | 10.000 |
| Optimierungstoleranz | 0,0001 |
| Solverwahl | `dual="auto"` |

Bei deutlich mehr Zellzeilen als Markern wählt `dual="auto"` hier die primale Optimierung. Die Toleranz steuert das numerische Abbruchkriterium. Es gibt **kein validierungsbasiertes Early Stopping wie bei CellCNN** und keine von uns festgelegte Adam-Lernrate.

**Papervergleich:** Das Paper beschreibt Random Search für Baseline-Hyperparameter. Unsere feste Suche über drei C-Werte ist eine vereinfachte, reproduzierbare Alternative.

## 2.12 Wie machen wir aus Zell-Scores einen Spenderscore?

Nach einem inneren SVM-Training bewerten wir **alle Zellen** der jeweiligen Validierungsspender.

Pro Spender wählen wir die höchsten 1 % seiner Zellscores und bilden deren Mittelwert:

$$
k_d=\max\left(1,\left\lceil0{,}01N_d\right\rceil\right)
$$

$$
S_d=\frac{1}{k_d}\sum_{i\in T_d}g_i
$$

Dabei bedeutet:

- $d$: ein Spender.
- $N_d$: Anzahl seiner verfügbaren Zellen.
- $\lceil\ldots\rceil$: Aufrunden auf die nächste ganze Zahl.
- $k_d$: Anzahl der ausgewählten Zellen.
- $T_d$: Menge der Zellpositionen mit den höchsten $k_d$ Scores.
- $g_i$: Score einer dieser Zellen.
- $S_d$: aggregierter Spenderscore.

Bei 100.000 Zellen werden die höchsten 1.000 Scores gemittelt.

**Warum so?**

- Ein Mittelwert über alle Zellen könnte ein Signal einer seltenen Population verdünnen.
- Das einzelne Maximum wäre stark von einer extremen Zelle abhängig.
- Ein fester Anteil lässt sich auf Spender mit unterschiedlichen Zellzahlen anwenden.
- Jeder Spender liefert anschließend genau einen Score.

Die höchsten 1 % werden anhand des **vorzeichenbehafteten Scores** gewählt. Es geht um Zellen mit besonders hoher Bewertung in Richtung CMV+, nicht um die größten absoluten Werte.

**Diese Aggregation wird nicht mittrainiert.** Während des SVM-Fits beeinflussen alle ausgewählten Trainingszellen den Verlust. Erst bei der spenderweisen Bewertung wird auf die höchsten 1 % eingeschränkt.

**Papervergleich:** Diese genaue SVM-Aggregation ist im Projektplan vorab festgelegt. Sie ist vom Fokus auf seltene Populationen motiviert, aber keine nachgewiesen identische Paper-SVM-Regel.

**Unterschied zu CellCNN:** Unsere SVM aggregiert einen Zellscore über alle Zellen eines Spenders. CellCNN besitzt mehrere Filter und poolt deren Antworten innerhalb gezogener Zellgruppen bereits während des Trainings.

## 2.13 Wie wählen wir das beste C?

Für jeden C-Wert trainieren wir eine SVM in jedem der drei inneren Folds: **3 C-Werte × 3 Folds = 9 innere Modelle**.

Jedes Modell liefert spenderweise Validierungsscores. Daraus berechnen wir pro Fold die ROC-AUC.

Für jedes C mitteln wir die drei Fold-AUCs:

$$
\overline{A}(C)=\frac{A_0(C)+A_1(C)+A_2(C)}{3}
$$

Dabei bedeutet:

- $C$: getesteter Regularisierungsparameter.
- $A_f(C)$: ROC-AUC im inneren Fold $f$ für diesen C-Wert.
- $f$: Fold-Index 0, 1 oder 2.
- $\overline{A}(C)$: ungewichteter Mittelwert der drei AUCs.

Das C mit der höchsten mittleren AUC gewinnt. Bei Gleichstand wählen wir das **kleinste C**.

**Warum so?**

- ROC-AUC bewertet die Rangfolge der Spender, ohne bereits eine Schwelle festlegen zu müssen.
- Die Mittelung nutzt die Ergebnisse aller drei Folds.
- Das kleinere C ist bei Gleichstand die vorab bestimmte Wahl mit stärkerer relativer Regularisierung.

Das ist **keine einzige AUC über zusammengeführte Scores verschiedener Modelle**. Jeder Fold wird zunächst für sich bewertet.

**Papervergleich:** Hyperparameter werden auch im Paper mit Validierungsdaten ausgewählt. Die konkrete Regel „mittlere innere Spender-AUC, danach kleinstes C“ ist unsere Festlegung.

**Tatsächliches Ergebnis:** In allen 100 äußeren Splits haben die drei C-Werte dieselbe mittlere innere AUC. Deshalb wird überall **C = 0,01** ausgewählt.

Das beweist keine Überlegenheit von 0,01. Unterschiedliche Scores können dieselbe Rangfolge und damit dieselbe AUC besitzen. Bei nur vier oder fünf Validierungsspendern ist die Auflösung dieser Metrik zudem grob.

## 2.14 Warum brauchen wir eine eigene Spenderschwelle?

Ein SVM-Zellscore von null bezeichnet die gelernte **Zellentscheidungsgrenze**. Unser Spenderscore ist aber der Mittelwert der höchsten 1 % aller Zellscores.

Durch diese Auswahl liegt er systematisch höher als der durchschnittliche Zellscore.

**Deshalb ist null nicht automatisch eine geeignete Spenderschwelle.** Auch bei CMV−-Spendern können die am höchsten bewerteten Zellen positive Scores haben.

Tatsächlich sind **alle 600 gespeicherten Test-Spenderscores positiv**. Mit einer nachträglich eingesetzten Schwelle null würden alle diese Testfälle als CMV+ eingestuft.

Wir bestimmen deshalb eine Schwelle aus der inneren Validierung.

**Papervergleich:** Diese Schwellenbestimmung gehört zu unserer spenderweisen SVM-Anpassung. Sie ist nicht als identische NK-SVM-Regel aus dem Paper übernommen.

## 2.15 Wie bestimmen wir die Schwelle ohne Testdaten?

Für das ausgewählte C sammeln wir die inneren **Out-of-fold-Vorhersagen**.

Das bedeutet: Für jeden der 14 äußeren Trainingsspender liegt genau ein Score von einem inneren Modell vor, das diesen Spender nicht zum Lernen seiner Gewichte oder seines Scalers verwendet hat.

Auf diesen 14 Scores wählen wir eine Schwelle nach dem **Youden-Kriterium**:

$$
J(\tau)=\operatorname{Sensitivität}(\tau)+\operatorname{Spezifität}(\tau)-1
$$

Dabei bedeutet:

- $\tau$: eine mögliche Spenderschwelle.
- $J(\tau)$: Qualität dieser Schwelle nach dem Youden-Kriterium.
- Sensitivität: Anteil der tatsächlich positiven Spender, deren Score mindestens $\tau$ erreicht.
- Spezifität: Anteil der tatsächlich negativen Spender, deren Score unter $\tau$ liegt.

Wir wählen die Schwelle mit dem höchsten Wert.

**Warum so?** Das Kriterium berücksichtigt beide Klassen gleich. Seine Maximierung entspricht der Maximierung der Balanced Accuracy auf diesen Validierungsscores.

Die Kandidatenschwellen stammen aus `roc_curve`. Unendliche Schwellen werden ausgeschlossen. Bei Gleichstand gewinnt der erste angebotene endliche Maximierer; bei der absteigenden Schwellenreihenfolge ist das die höchste der angebotenen optimalen Schwellen.

Die Vorhersage lautet anschließend:

- $S_d\geq\tau$: CMV+.
- $S_d<\tau$: CMV−.

Diese 14 Scores sind für die Schwellenwahl geeignet, aber keine unabhängige abschließende Leistungsprüfung: Sowohl C als auch die Schwelle werden mithilfe der inneren Validierung gewählt.

**Methodische Einschränkung:** Die Scores stammen aus drei unterschiedlichen Modellen. Wir nehmen an, dass ihre numerischen Skalen ausreichend vergleichbar sind. Eine zusätzliche Kalibrierung findet nicht statt.

**Papervergleich:** Youden-Schwellenwahl aus inneren Out-of-fold-Spenderscores ist unsere Projektentscheidung.

## 2.16 Danach trainieren wir eine neue SVM auf allen 14 äußeren Trainingsspendern

Nach Auswahl von C und Schwelle:

1. Ziehen wir erneut 10.000 Zellen pro äußerem Trainingsspender.
2. Fitten wir einen neuen Scaler auf diesen **140.000 Zellen**.
3. Trainieren wir eine neue SVM mit dem ausgewählten C.
4. Übernehmen wir die zuvor bestimmte Spenderschwelle unverändert.

Die finale Trainingsstichprobe verwendet einen eigenen reproduzierbaren Seed.

**Warum so?** Die innere CV hat die Einstellungen bestimmt. Für die eigentliche Testvorhersage können wir nun alle 14 verfügbaren äußeren Trainingsspender zum Lernen der linearen Regel nutzen.

**Das ist ein wichtiger Unterschied zu unserem CellCNN-Ablauf:** Bei CellCNN testen wir das ausgewählte innere Netz direkt. Bei der SVM trainieren wir nach der Parameterauswahl neu auf allen 14 äußeren Trainingsspendern.

**Methodische Einschränkung:** Die innerlich bestimmte Schwelle wird auf dieses neue Modell übertragen. Dessen Scoreskala kann sich durch das Neutraining verändern. Das kann die Klassifikation an der Schwelle beeinflussen; die ROC-AUC hängt dagegen nicht von der Schwelle ab.

**Papervergleich:** Dieses konkrete Neutraining einschließlich Übernahme der OOF-Schwelle ist unsere SVM-Auswertungspipeline.

## 2.17 Wie bewerten wir die sechs Testspender?

Für jeden Testspender führen wir aus:

1. Alle bereits ArcSinh-transformierten Zellmessungen mit dem finalen Trainings-Scaler standardisieren.
2. Für jede Zelle den linearen SVM-Score berechnen.
3. Die höchsten 1 % der Scores auswählen.
4. Deren Mittelwert als Spenderscore berechnen.
5. Den Score mit der innerlich bestimmten Schwelle vergleichen.

Wir verwenden **alle Testzellen**, keine zufällige Teststichprobe und keine Mittelung über mehrere Stichproben.

**Warum so?** Eine lineare Bewertung ist rechnerisch günstig. Alle Zellen zu verwenden vermeidet zusätzliche zufällige Schwankungen durch Testsampling.

**Papervergleich:** Auch diese genaue Kombination aus vollständiger Zellbewertung, Top-1-%-Pooling und gelernter Schwelle ist unsere Anpassung.

## 2.18 Ein tatsächliches Beispiel: äußerer Split 0

Für Split 0 ergeben alle drei C-Werte dieselben inneren Fold-AUCs:

| Innerer Fold | ROC-AUC |
|---|---:|
| 0 | 0,5000 |
| 1 | 0,3333 |
| 2 | 0,7500 |

Ihr Mittelwert beträgt etwa 0,5278. Wegen des Gleichstands wird C = 0,01 gewählt.

Die gespeicherte Spenderschwelle beträgt **0,986707**.

Das finale Modell liefert:

| Spender | Tatsächliche Klasse | Spenderscore | Vorhersage |
|---|---|---:|---|
| a_002 | CMV+ | 0,9344 | CMV− |
| a_004 | CMV− | 0,4732 | CMV− |
| a_006 | CMV− | 0,7300 | CMV− |
| a_1a | CMV− | 0,7009 | CMV− |
| a_2a | CMV− | 0,5567 | CMV− |
| a_4a | CMV+ | 1,0693 | CMV+ |

`a_002` wird falsch negativ klassifiziert, weil sein Score unter der Schwelle liegt.

Trotzdem ist die **ROC-AUC dieses Splits 1,00**: Beide positiven Spender haben höhere Scores als alle negativen.

Die Balanced Accuracy beträgt **0,75**:

- Einer von zwei positiven Spendern erkannt: Sensitivität 0,5.
- Vier von vier negativen Spendern erkannt: Spezifität 1,0.
- Mittelwert beider Raten: 0,75.

Dieses Beispiel zeigt, weshalb wir Rangqualität und Klassifikation an einer Schwelle getrennt betrachten.

## 2.19 Welche Endergebnisse erhalten wir über alle 100 Splits?

Pro Split berechnen wir die Kennzahlen auf den sechs Testspendern:

| Kennzahl | Bedeutung |
|---|---|
| ROC-AUC | Wie gut positive Spender höhere Scores erhalten als negative |
| Average Precision | Zusammenfassung von Precision über steigenden Recall |
| Trapezoidale PR-AUC | Trapezoidal integrierte Fläche unter der Precision-Recall-Kurve |
| Balanced Accuracy | Mittelwert aus Sensitivität und Spezifität an der gewählten Schwelle |

Dabei ist **Precision** der Anteil tatsächlich positiver Spender unter den positiv eingestuften Spendern. **Recall** ist die Sensitivität.

Average Precision und trapezoidale PR-AUC verwenden unterschiedliche Berechnungen und sind deshalb nicht austauschbar.

Aus den vorhandenen Vorhersagen wurden neu berechnet:

| Kennzahl | Mittelwert über 100 Splits | Median |
|---|---:|---:|
| **ROC-AUC** | **0,78375** | **0,8750** |
| Average Precision | 0,77983 | 0,8333 |
| Trapezoidale PR-AUC | 0,73117 | 0,7917 |
| Balanced Accuracy | 0,6200 | 0,6250 |

**Warum diese Auswertung?** ROC-AUC ist die primäre Vergleichsmetrik des NK-Benchmarks. Die weiteren Kennzahlen ergänzen die Bewertung der positiven Klasse und der konkreten Entscheidungen.

Der Mittelwert wird über die **100 separat berechneten Split-Metriken** gebildet. Wir behandeln die 600 Vorhersagezeilen nicht als 600 unabhängige Personen. Sie stammen aus wiederholten Tests derselben 20 Spender.

**Papervergleich:** Die ROC-AUC-Auswertung über das gemeinsame Monte-Carlo-Schema ist papernah. Diese konkreten SVM-Ergebnisse und die zusätzliche Schwellenbewertung gehören zu unserem Projekt.

## 2.20 Was speichern wir?

Pro äußerem Split werden neun innere SVMs und eine finale SVM trainiert. Der Full-Lauf umfasst damit insgesamt **1.000 SVM-Fits**.

Gespeichert werden:

| Datei | Inhalt | Zweck |
|---|---|---|
| [Testvorhersagen](tables/task4_svm_predictions_gated_alive_full.csv) | 600 Testzeilen mit Score, Schwelle, Label, gewähltem C und Anzahl gepoolter Zellen | Leistungsbewertung |
| [Innere Auswahl](tables/task4_svm_selection_gated_alive_full.csv) | 900 Kombinationen aus Split, C und innerem Fold mit Validierungs-AUC | Nachvollziehbare Parameterauswahl |
| [Finale Modelle](tables/task4_svm_models_gated_alive_full.csv) | Je Split 37 Gewichte, Intercept, Scalerparameter und Schwelle | Rekonstruktion und Interpretation |

Eine Konfigurationsdatei dokumentiert außerdem Parameter, Bibliotheksversionen und Prüfsummen der Eingaben sowie des relevanten Codes.

**Warum so?** Das ermöglicht das Fortsetzen eines Laufs und die spätere Rekonstruktion der finalen Vorhersagen ohne erneutes Training.

Eine Grenze der gespeicherten Dokumentation: **Die einzelnen inneren Out-of-fold-Spenderscores werden nicht exportiert.** Die gewählte Schwelle ist gespeichert, ihre vollständige Neuberechnung würde aber die inneren Fits erneut benötigen.

Am Ende liegen 100 finale Split-Modelle vor. Ein einzelnes zusätzliches Modell auf allen 20 Spendern wird in diesem Notebook nicht trainiert.

## 2.21 Was können wir daraus biologisch interpretieren?

Die Markergewichte beschreiben, welche Kombination standardisierter Marker einen hohen Zellscore erzeugt.

Die höchsten 1 % dieser Scores bestimmen den Spenderscore. Diese Zellen können anschließend auf ihre Markerprofile untersucht werden.

Dabei gelten drei Grenzen:

- Ein hohes Markergewicht ist keine kausale Aussage über CMV.
- Eine hoch bewertete Zelle ist keine durch unabhängige Zelllabels bestätigte CMV-assoziierte Zelle.
- Die zur Spenderbewertung ausgewählten Top-1-%-Zellen sind **nicht dasselbe wie die Support-Vektoren**, die dem SVM-Verfahren seinen Namen geben. Die Top-Zellen entstehen durch unsere nachgelagerte Aggregationsregel.

**Warum ist diese Trennung nötig?** Unsere SVM wurde mit schwachen Zelllabels trainiert. Ihre Zellbewertungen liefern Hinweise auf informative Profile, keine biologische Wahrheit pro Zelle.

**Papervergleich:** Die Single-Cell-Baseline übernimmt ebenfalls Probenlabels für Zellen. Die Interpretation unserer Top-1-%-Auswahl muss zusätzlich im Kontext unserer eigenen Aggregationsregel erfolgen.

Das ergänzte [SVM-Markerprofil aus Aufgabe 5](AUFGABE_5_METHODENERKLAERUNGEN.md#57-wie-entsteht-das-spendergleich-gewichtete-svm-markerprofil) beschreibt positiv ausgewählte Zellen der vollständigen äußeren Testspender in Splits 0–29. Es mittelt zunächst Zellen pro Testauftritt, anschließend nichtleere Auftritte je Spender und schließlich alle vertretenen Spender gleichgewichtet. Dies ergänzt die Interpretation; Training, Auswahlregeln und Leistungskennzahlen bleiben unverändert.

Die SVM-Pipeline selbst und die Kennzahlen wurden anhand des Codes und der gespeicherten Tabellen geprüft. Für diese Erklärung wurden keine neuen SVMs trainiert und die historischen OOF-Schwellen nicht neu berechnet.

---

# 3. Citrus

**Citrus bildet zuerst Zellpopulationen durch hierarchisches Clustering. Danach beschreibt es jeden Spender durch die Häufigkeiten dieser Populationen. Eine L1-regularisierte logistische Regression wählt informative Populationen aus und berechnet daraus die CMV+-Wahrscheinlichkeit.**

Anders als CellCNN lernt Citrus die Zellgruppen und den Klassifikator in getrennten Schritten. Anders als unsere SVM trainiert der abschließende Klassifikator direkt auf **Spendermerkmalen**, nicht auf einzelnen Zellen.

**Aktualisierung:** Das aktuelle Citrus-Notebook verwendet:

| Einstellung | Aktueller Full-Modus |
|---|---:|
| Datenstufe | `gated_alive` |
| Zellen pro Trainings- und Testspender | 10.000 |
| Mindestclustergröße | 0,05 % |
| Äußere Splits | 30, IDs 0–29 |
| Innere Folds | 3 |

Der frühere Lauf mit 1.000 Zellen, 5 % Mindestclustergröße und 100 Splits ist archiviert. Einige Berichtsteile beziehen sich noch darauf. **Die folgende Erklärung beschreibt die aktuelle Umsetzung und die aktuellen Ergebnisse.**

Für den Papervergleich sind insbesondere die Supplementary Methods auf Seite 6 relevant. Dort stehen die konkreten Citrus-Einstellungen des NK-/CMV-Benchmarks.

## 3.1 Was soll Citrus vorhersagen?

Das Ziel ist wieder der CMV-Status eines Spenders:

- `0`: CMV−.
- `1`: CMV+.

Die biologische Annahme lautet: Bestimmte Zellpopulationen könnten bei CMV+-Spendern häufiger oder seltener vorkommen als bei CMV−-Spendern.

Citrus untersucht deshalb zwei Fragen nacheinander:

1. Welche Zellgruppen lassen sich anhand ihrer Markerprofile bilden?
2. Welche ihrer Häufigkeiten helfen, den CMV-Status vorherzusagen?

**Warum so?** Ein biologisches Signal muss nicht darin bestehen, dass sich alle Zellen eines Spenders verändern. Es kann auch darin liegen, dass eine bestimmte Unterpopulation häufiger vorkommt.

Für das Clustering benötigt Citrus keine Zelllabels. Das Spenderlabel wird erst beim Lernen des Klassifikators verwendet.

**Im Paper:** Genau diese Kombination aus Zellclustering, spenderbezogenen Clusterhäufigkeiten und regulärer Klassifikation wird für den Citrus-NK-Benchmark beschrieben.

## 3.2 Wir verwenden die originale Citrus-Implementierung

Das Notebook läuft in einer separaten R-Umgebung und verwendet:

- Citrus 0.8;
- `Rclusterpp` für das hierarchische Clustering;
- `flowCore` für FCS-Dateien;
- `glmnet` für die regularisierte logistische Regression.

Die Quellstände von Citrus und Rclusterpp sind auf konkrete Git-Commits festgelegt:

- Citrus: `d02baae544abdc403704aaceb75d1e7931a0331c`.
- Rclusterpp: `a07380683ce7a6849af8ec27db6439ea3a707890`.

**Warum so?** Wir wollen die tatsächliche Citrus-Methode verwenden. Ein selbst geschriebenes Clustering mit anschließender Regression wäre nicht automatisch dieselbe Implementierung.

Die getrennte R-Umgebung ermöglicht außerdem, Citrus zu verwenden, ohne die Python-Umgebung von SVM und CellCNN umzubauen.

**Im Paper:** Für den NK-Benchmark wurde ebenfalls Citrus 0.8 verwendet. Gleiche Citrus-Version bedeutet allerdings nicht automatisch identische Ergebnisse: Bibliotheksversionen, Datenstichproben und Zufallsaufteilungen können sich unterscheiden.

## 3.3 Welche Daten gehen hinein?

Die Datengrundlage ist dieselbe wie bei den anderen Hauptmethoden:

| Eigenschaft | Wert |
|---|---:|
| Spender | 20 |
| CMV− | 11 |
| CMV+ | 9 |
| Verfügbare Zellen | 3.438.750 |
| Verwendete biologische Marker | 37 |
| Datenstufe | `gated_alive` |

Die FCS-Dateien sind bereits auf lebende Zellen und gegen Doubletten bereinigt. Unser Notebook wiederholt dieses Gating nicht.

Die Markerliste legt fest, welche 37 Dimensionen für Transformation, Clustering und Zuordnung neuer Zellen verwendet werden. Technische Kanäle gehen nicht in diese Berechnungen ein.

Die Spenderlabels und Aufteilungen kommen aus der gemeinsamen Split-Tabelle.

**Warum so?** Alle Methoden sollen dieselbe biologische Aufgabe bearbeiten. Eine Beschränkung auf vorher ausgewählte NK-Zellen würde die Suche nach einer informativen Population verändern.

**Im Paper:** Die Analyse verwendet ebenfalls PBMCs nach Entfernung toter Zellen und Doubletten. Wir verwenden die 37 bereitgestellten Marker; die bereits besprochene Differenz zur Angabe von 36 Markern im Paper bleibt bestehen.

## 3.4 Wir verwenden 30 gemeinsame äußere Spendersplits

Pro äußerem Split gilt:

| Gruppe | Äußeres Training | Äußerer Test |
|---|---:|---:|
| CMV− | 7 | 4 |
| CMV+ | 7 | 2 |
| Insgesamt | 14 | 6 |

Citrus verwendet aktuell die bereits festgelegten Split-IDs **0 bis 29**.

**Warum so?**

- Die spenderweise Trennung verhindert, dass Zellen desselben Spenders gleichzeitig im Training und Test liegen.
- Sieben Trainingsspender pro Klasse ergeben ein ausgeglichenes äußeres Training.
- Die gemeinsamen Splits ermöglichen einen direkten Methodenvergleich.
- Die Begrenzung auf 30 Wiederholungen reduziert den Aufwand des hierarchischen Clusterings.

**Im Paper:** Dort wurden **100** gemeinsame Monte-Carlo-Splits für CellCNN und Citrus verwendet. Unsere Reduktion auf 30 Splits ist ein ausdrücklich dokumentierter Rechenkompromiss.

Für den aktuellen Dreiervergleich müssen deshalb auch CellCNN und SVM auf genau diesen 30 Splits betrachtet werden. Der zusätzliche Zweiervergleich verwendet alle 100 gemeinsamen CellCNN-/SVM-Splits und wird separat ausgewiesen.

## 3.5 Pro Spender ziehen wir 10.000 Zellen

Aus jedem der 14 äußeren Trainingsspender zieht Citrus zufällig **10.000 Zellen ohne Zurücklegen**.

Damit stehen für das äußere Trainingsclustering **14 × 10.000 = 140.000 Zellen** zur Verfügung.

Auch aus jedem Testspender werden später 10.000 Zellen gezogen.

Die Dateien werden in einer festen Spenderreihenfolge eingelesen. Ein aus dem gemeinsamen Split-Seed abgeleiteter R-Seed macht die Stichprobe reproduzierbar. Die Anpassung an Rs Integerbereich verändert die Spenderaufteilung nicht. Konkret wird der Split-Seed modulo `2^31 - 1` verwendet; das Testsampling erhält den so gebildeten Seed plus eins.

**Warum so?**

- Gleiche Zellzahlen verhindern, dass zellreiche Spender den Clusterbaum allein durch mehr Messungen dominieren.
- Die Begrenzung reduziert den Rechen- und Speicherbedarf.
- Ohne Zurücklegen enthält die Stichprobe eines Spenders keine künstlich wiederholten Zellen.

Die 10.000 Zellen sind ein Rechenkompromiss, keine nachgewiesen optimale Stichprobengröße.

**Im Paper:** Citrus zieht im NK-Benchmark **20.000 Zellen pro Trainings- und Testspender**. Wir verwenden jeweils die Hälfte.

Bei seltenen Populationen beeinflusst diese Reduktion, wie viele ihrer Zellen überhaupt in der Stichprobe vorhanden sind und wie stabil ihre Häufigkeit geschätzt werden kann.

## 3.6 Wir transformieren die Markerwerte mit ArcSinh

Für jeden verwendeten Markerwert berechnet Citrus:

$$
a_{ij}=\operatorname{arcsinh}(x_{ij}/5)
$$

Dabei bedeutet:

- $i$: eine Zelle.
- $j$: ein Marker.
- $x_{ij}$: eingelesener Messwert dieses Markers in dieser Zelle.
- $5$: festgelegter Kofaktor.
- $\operatorname{arcsinh}$: inverse hyperbolische Sinusfunktion.
- $a_{ij}$: transformierter Wert.

Große Intensitäten werden komprimiert; Werte nahe null bleiben ungefähr linear. Auch negative Werte sind zulässig.

**Warum so?** Die Transformation begrenzt den Einfluss stark rechtsschiefer Intensitätsverteilungen und entspricht unserer gemeinsamen Vorverarbeitung.

**Entscheidender Unterschied zu SVM und CellCNN:** Wir führen bei Citrus **keine zusätzliche z-Standardisierung der Marker vor dem Clustering** durch. Die Clusterdistanzen werden auf den ArcSinh-transformierten Markerwerten berechnet.

Das bedeutet: Marker mit größerer verbleibender Streuung können die Distanzen stärker beeinflussen.

**Paper-/Referenzvergleich:** Wir verwenden den originalen Citrus-Leseweg mit expliziter ArcSinh-Transformation und ohne aktivierte zusätzliche Markerskalierung. Die konkrete Vorverarbeitung ist damit an der Referenz orientiert; ihre Überlegenheit gegenüber einer zusätzlichen Standardisierung wurde hier nicht getestet.

Später standardisiert `glmnet` andere Größen: die **Clusterhäufigkeiten**. Das ist ein separater Schritt.

## 3.7 Wir bilden eine hierarchische Struktur aus den Trainingszellen

Citrus führt das Clustering im Raum der **37 transformierten Marker** aus.

Die verwendeten Rclusterpp-Standardeinstellungen sind:

- euklidische Distanz;
- Ward-Linkage.

Anschaulich beginnt das hierarchische Verfahren mit einzelnen Zellen. Schrittweise werden Gruppen zusammengeführt. Ward bevorzugt Zusammenführungen, die die Streuung innerhalb der Gruppen möglichst wenig erhöhen.

Es entsteht ein Baum:

- Kleine Gruppen beschreiben feinere Zellpopulationen.
- Größere übergeordnete Gruppen enthalten mehrere kleinere Gruppen.
- Der oberste Knoten umfasst alle Trainingszellen.

**Warum so?** Biologisch informative Populationen können auf unterschiedlichen Auflösungsebenen liegen. Citrus muss deshalb nicht vorab eine feste Zahl disjunkter Zelltypen festlegen.

**Wichtig:** Citrus schneidet den Baum hier nicht einfach auf beispielsweise zehn Cluster zurück. Es berücksichtigt viele geeignete Knoten der Hierarchie.

Dadurch kann dieselbe Zelle zu mehreren betrachteten Clustern gehören: zu einem kleinen Cluster und gleichzeitig zu dessen übergeordneten Clustern.

**Im Paper:** Hierarchische Zellgruppen sind ein Kernbestandteil von Citrus. Ward und euklidische Distanz wurden als tatsächliche Einstellungen der installierten Referenzimplementierung geprüft.

Das Clustering verwendet die Markerwerte, **nicht die CMV-Labels**. Die Labels bestimmen später, welche Clusterhäufigkeiten prädiktiv sind.

## 3.8 Wir behalten nur ausreichend große Cluster

Die aktuelle Mindestgröße beträgt **0,05 % = 0,0005**.

Citrus erwartet diesen Wert als Anteil. Deshalb steht im Code:

```r
minimumClusterSizePercent = 0.0005
```

Bei 140.000 äußeren Trainingszellen entspricht das **140.000 × 0,0005 = 70 Zellen**.

Ein betrachteter Cluster muss somit mindestens 70 Trainingszellen enthalten.

**Die Grenze bezieht sich auf die gesamte Trainingszellmenge des jeweiligen Baums, nicht auf jeden Spender einzeln.**

Ein zulässiger Cluster darf deshalb bei einem bestimmten Spender nur wenige oder gar keine Zellen enthalten.

**Warum so?** Sehr kleine Gruppen können instabil sein und die Zahl möglicher Merkmale stark erhöhen. Gleichzeitig muss die Grenze niedrig genug sein, um seltene Populationen zuzulassen.

**Im Paper:** Die Mindestclustergröße beträgt ebenfalls **0,05 %**. Diese Einstellung entspricht jetzt dem NK-Benchmark.

Durch unsere kleinere Zellstichprobe unterscheidet sich die absolute Grenze: Im Paper wären es bei 14 × 20.000 Zellen mindestens 140 Zellen.

Der frühere Wert `0.05` entsprach dagegen **5 %**. Das war eine hundertfach höhere relative Mindestgröße und gehört nicht zur aktuellen Konfiguration.

## 3.9 Aus den Clustern entsteht eine Tabelle mit einer Zeile pro Spender

Für jeden zulässigen Cluster bestimmen wir, welcher Anteil der Zellen eines Spenders zu ihm gehört:

$$
F_{dk}=\frac{n_{dk}}{N_d}
$$

Dabei bedeutet:

- $d$: ein Spender.
- $k$: ein betrachteter Cluster.
- $n_{dk}$: Anzahl der gezogenen Zellen dieses Spenders im Cluster.
- $N_d$: Gesamtzahl seiner gezogenen Zellen, hier 10.000.
- $F_{dk}$: relative Clusterhäufigkeit für diesen Spender.

Beispiel: Gehören 120 seiner 10.000 Zellen zum Cluster, ist das Merkmal $F_{dk}=120/10.000=0{,}012$. Das entspricht 1,2 %.

Die finale Trainingsmatrix hat:

- **14 Zeilen**, eine pro Trainingsspender;
- eine Spalte pro zulässigem Cluster.

Im aktuellen Split 0 sind das **3.117 Clustermerkmale**.

**Warum Häufigkeiten?** Absolute Zellzahlen wären von der Zahl eingelesener Zellen abhängig. Relative Häufigkeiten beschreiben die Zusammensetzung einer Probe.

Wir verwenden `featureType="abundances"`. Die Klassifikationsmerkmale sind damit **Clusterhäufigkeiten**, nicht die mittleren Markerintensitäten innerhalb eines Clusters.

**Im Paper:** Für den NK-Benchmark werden ebenfalls spenderbezogene Clusterhäufigkeiten als Eingabe der logistischen Regression verwendet.

Weil sich die Cluster hierarchisch überlappen, müssen ihre Häufigkeiten pro Spender **nicht zu eins summieren**.

## 3.10 Die innere Validierung benötigt eigene Clusterbäume

Die 14 äußeren Trainingsspender werden anhand derselben inneren Folds wie bei SVM und CellCNN aufgeteilt:

- zweimal neun innere Trainingsspender und fünf Validierungsspender;
- einmal zehn Trainingsspender und vier Validierungsspender.

Für jeden Fold wird das Clustering ausschließlich auf dessen inneren Trainingsspendern neu aufgebaut.

Damit enthalten die inneren Bäume:

| Innere Trainingsspender | Zellen im Baum | Mindestclustergröße |
|---:|---:|---:|
| 9 | 90.000 | 45 Zellen |
| 10 | 100.000 | 50 Zellen |

Die einmal gezogenen 10.000 Zellen pro Spender werden dabei entsprechend seiner Fold-Rolle verwendet.

**Warum eigene Bäume?** Würden wir einen Clusterbaum aus allen 14 Spendern für die innere Validierung verwenden, hätten die Validierungsspender bereits die Merkmalsbildung beeinflusst.

Die Cluster verschiedener innerer Bäume sind nicht identisch. Eine Cluster-ID bezeichnet nur einen Knoten innerhalb des zugehörigen Baums.

**Referenzvergleich:** Die öffentliche Citrus-Funktion würde eigene zufällige Folds erzeugen. Unser Notebook baut das benötigte Fold-Objekt deshalb aus den offiziellen Citrus-Funktionen mit unseren vorgegebenen Spenderzuordnungen auf.

Clustering, Mapping, Häufigkeitsberechnung und Regression werden dabei weiterhin von Citrus durchgeführt.

## 3.11 Wie ordnen wir Validierungszellen einem bereits vorhandenen Baum zu?

Die Validierungszellen werden nicht gemeinsam mit den Trainingszellen neu geclustert.

Stattdessen verwendet Citrus ein **Nearest-Neighbor-Mapping**:

1. Für jede neue Zelle wird eine nächste Trainingszelle im verwendeten Markerraum bestimmt.
2. Die neue Zelle übernimmt deren Zugehörigkeit zu den betrachteten Trainingsclustern.
3. Daraus berechnen wir die Clusterhäufigkeiten des neuen Spenders.

**Warum so?** Der neue Spender muss durch dieselben Merkmale beschrieben werden, auf denen das Modell gelernt hat. Würden wir ihn separat clustern, hätten seine Cluster nicht automatisch dieselbe Bedeutung.

Das Mapping ordnet die Zelle dabei über Trainingszellen zu. Es ist nicht einfach eine Zuordnung zum nächstgelegenen exportierten Clusterzentroiden.

**Im Paper:** Neue Testproben werden ebenfalls auf die Trainingscluster abgebildet. Für unsere innere Validierung verwenden wir denselben Gedanken innerhalb jedes Folds.

So entstehen vergleichbare Trainings- und Validierungsmerkmale **innerhalb eines Folds**, obwohl unterschiedliche Folds verschiedene Clusterbäume besitzen.

## 3.12 Die logistische Regression kombiniert die Clusterhäufigkeiten

Aus den Clusterhäufigkeiten berechnet das Modell zunächst einen linearen Wert:

$$
\eta_d=\beta_0+\sum_{k=1}^{K}\beta_kF_{dk}
$$

Dabei bedeutet:

- $d$: ein Spender.
- $K$: Anzahl zulässiger Clustermerkmale dieses Modells.
- $F_{dk}$: Häufigkeit des Clusters $k$ bei Spender $d$.
- $\beta_k$: gelerntes Gewicht dieser Clusterhäufigkeit.
- $\beta_0$: Intercept, also der gemeinsame Grundwert.
- $\eta_d$: linearer Vorhersagewert, auch Logit genannt.
- $\sum$: Addition aller Clusterbeiträge.

Daraus entsteht:

$$
p_d=\frac{1}{1+\exp(-\eta_d)}
$$

Dabei ist:

- $\exp$: Exponentialfunktion.
- $p_d$: modellierte CMV+-Wahrscheinlichkeit des Spenders.

Bei ansonsten unveränderten Merkmalen gilt:

- Positives $\beta_k$: Eine höhere Häufigkeit dieses Clusters erhöht die CMV+-Wahrscheinlichkeit.
- Negatives $\beta_k$: Eine höhere Häufigkeit senkt sie.

**Warum logistische Regression?** Sie passt zur binären Zielvariable und liefert eine direkte Spenderwahrscheinlichkeit. Die Clusterhäufigkeiten werden dabei gemeinsam betrachtet.

**Im Paper:** Eine L1-regularisierte logistische Regression auf Clusterhäufigkeiten entspricht der beschriebenen Citrus-Konfiguration des NK-Benchmarks.

Die Gewichte beziehen sich auf **Clusterhäufigkeiten**. Es handelt sich nicht um Gewichte einzelner Marker wie bei der SVM.

## 3.13 Warum brauchen wir L1-Regularisierung?

Wir haben nur 14 Trainingsspender, aber mehrere Tausend mögliche Clustermerkmale. Viele davon sind außerdem stark korreliert, weil sie aus demselben Baum stammen.

Deshalb verwenden wir L1-Regularisierung, auch Lasso genannt.

Das Optimierungsziel lautet sinngemäß:

$$
L=-\frac{1}{D}\sum_{d=1}^{D}
\left[y_d\log(p_d)+(1-y_d)\log(1-p_d)\right]
+\lambda\sum_{k=1}^{K}|\widetilde{\beta}_k|
$$

Dabei bedeutet:

- $L$: zu minimierender Gesamtverlust.
- $D$: Anzahl der Trainingsspender im jeweiligen Fit.
- $d$: Index eines Trainingsspenders.
- $y_d$: tatsächliches Label, 0 oder 1.
- $p_d$: vorhergesagte CMV+-Wahrscheinlichkeit.
- $\log$: natürlicher Logarithmus.
- $K$: Anzahl der Clustermerkmale.
- $\widetilde{\beta}_k$: Gewicht des intern standardisierten Clustermerkmals.
- $|\ldots|$: Betrag.
- $\lambda$: Stärke der Regularisierung.

Der erste Teil bestraft schlechte Klassenwahrscheinlichkeiten. Der zweite Teil bestraft die Summe der absoluten Gewichte.

**Warum L1?** L1 kann Gewichte auf genau null setzen. Damit werden aus den vielen möglichen Clustern wenige prädiktive Cluster ausgewählt.

Bei `glmnet` wird dafür `alpha=1` verwendet. Das ist reines Lasso, keine Mischung aus L1 und L2.

`glmnet` standardisiert die Clusterhäufigkeiten innerhalb des jeweiligen Trainingsfits. Das unterscheidet sich von einer Standardisierung der ursprünglichen Marker vor dem Clustering.

Die zurückgegebenen Koeffizienten beziehen sich wieder auf die ursprüngliche Häufigkeitsskala. Der Intercept wird nicht regularisiert.

**Im Paper:** Die L1-Strafe ist ausdrücklich vorgesehen. Die genaue interne Standardisierung folgt der verwendeten Citrus-/glmnet-Implementierung.

Die zahlreichen Clustermerkmale erhöhen nicht die Anzahl unabhängiger Beobachtungen: Der finale Klassifikator hat weiterhin nur 14 Trainingszeilen.

## 3.14 Wie wählen wir die Regularisierungsstärke Lambda?

Großes $\lambda$ bevorzugt eine sparsame Lösung mit wenigen wirksamen Clustern. Kleineres $\lambda$ erlaubt mehr und größere Koeffizienten.

Citrus erzeugt einen absteigenden **Lambda-Pfad**, also eine Folge möglicher Regularisierungsstärken. Angefordert werden standardmäßig 100 Werte; die tatsächliche zurückgegebene Pfadlänge kann kürzer sein.

Für diese Werte werden die drei inneren Regressionsmodelle trainiert und auf ihren Validierungsspendern bewertet.

Für jedes Lambda werden die Klassifikationsfehler der insgesamt 14 inneren Out-of-fold-Spendervorhersagen zusammengefasst.

Wir wählen:

```r
regression$cvMinima[["cv.min"]]
```

Das ist das Lambda mit dem **kleinsten inneren Klassifikationsfehler**.

Bei Gleichstand gewinnt der erste Wert im absteigenden Pfad, also die stärkere Regularisierung.

**Warum so?** Lambda soll anhand bisher nicht zum jeweiligen Regressionsfit verwendeter Spender gewählt werden. Die Gleichstandsregel bevorzugt die sparsamere Erklärung.

Wir wählen hier nicht nach ROC-AUC wie bei unserer SVM. Auch die zusätzlich von Citrus berechnete `cv.1se`-Alternative wird nicht verwendet.

**Papervergleich:** Dreifache CV zur Auswahl der L1-Strafe ist im NK-Benchmark beschrieben. Unsere konkrete Auswahl über `cv.min` verwendet die entsprechende Funktion des Originalpakets.

## 3.15 Eine wichtige Einschränkung der inneren Trennung

Die originale Citrus-Funktion bestimmt den gemeinsamen Lambda-Pfad anhand der Clustermerkmale und Labels **aller 14 äußeren Trainingsspender**.

Erst danach werden die inneren Modelle an diesen Lambda-Werten bewertet.

Daher gilt:

- Die inneren Clusterbäume verwenden nur innere Trainingsspender.
- Die inneren Regressionsfits verwenden nur innere Trainingsspender.
- **Das datenabhängige Lambda-Raster berücksichtigt auch die inneren Validierungsspender.**

**Warum wurde das so übernommen?** Unser Notebook verwendet bewusst den originalen Citrus-Auswahlablauf. Eine vollständig getrennte Erzeugung des Suchrasters wäre eine Änderung dieses Ablaufs.

Das ist aber eine tatsächliche methodische Einschränkung. Man darf deshalb für unsere Citrus-Pipeline nicht pauschal behaupten, dass jede datenabhängige Entscheidung ausschließlich aus inneren Trainingsspendern stammt.

**Die sechs äußeren Testspender bleiben weiterhin vollständig ausgeschlossen.** Die äußere Auswertung bewertet also das Verfahren einschließlich dieser Eigenschaft der inneren Auswahl.

**Referenzvergleich:** Dieser Ablauf wurde direkt in der installierten Citrus-Funktion `citrus.endpointRegress` geprüft. Es ist eine Eigenschaft der übernommenen Implementierung, nicht eine explizite Aussage aus der kurzen Benchmarkbeschreibung.

## 3.16 Welches Modell wird schließlich getestet?

Zusätzlich zu den drei inneren Bäumen wird ein Baum aus **allen 14 äußeren Trainingsspendern** erstellt.

Auf dessen Clusterhäufigkeiten wird die finale logistische Regression trainiert. Die innere Validierung bestimmt, welcher Lambda-Wert dieses finalen Modellpfads verwendet wird.

**Warum so?** Nach der Auswahl der Regularisierungsstärke sollen alle verfügbaren äußeren Trainingsspender zur finalen Merkmalsbildung und Regression beitragen.

Technisch berechnet Citrus den finalen Regressionspfad bereits innerhalb seines Auswahlablaufs. Es ist daher nicht ganz exakt, die interne Reihenfolge als „erst Lambda auswählen, dann erstmals final trainieren“ zu beschreiben.

**Unterschied zu CellCNN:** Das finale Citrus-Modell nutzt alle 14 äußeren Trainingsspender. Unser ausgewähltes CellCNN-Netz hat seine Gewichte dagegen auf neun oder zehn inneren Trainingsspendern gelernt.

**Im Paper:** Das Testmodell beruht ebenfalls auf Trainingsclustern und einer auf Trainingsproben angepassten, per CV regulierten logistischen Regression.

## 3.17 Wie entsteht die Testvorhersage?

Für jeden der sechs äußeren Testspender:

1. 10.000 Zellen zufällig ohne Zurücklegen ziehen.
2. Die 37 Marker mit ArcSinh transformieren.
3. Die Zellen in den fertigen äußeren Trainingsbaum abbilden.
4. Die Häufigkeiten derselben zulässigen Trainingscluster berechnen.
5. Diese Häufigkeiten an die finale logistische Regression übergeben.
6. Die CMV+-Wahrscheinlichkeit am ausgewählten Lambda berechnen.

Die Entscheidung lautet:

- Wahrscheinlichkeit mindestens 0,5: CMV+.
- Wahrscheinlichkeit unter 0,5: CMV−.

Es gibt keine zusätzliche Youden-Schwelle und keine Mittelung über fünf Stichproben wie bei unseren anderen Verfahren.

**Warum so?** Die logistische Regression liefert bereits eine Klassenwahrscheinlichkeit. Die feste Schwelle 0,5 vermeidet eine weitere Schwellenoptimierung.

Das Notebook rundet Wahrscheinlichkeiten vor dem Speichern und der Entscheidung auf 15 Nachkommastellen. Das ist eine numerische Konvention, keine zusätzliche Kalibrierung.

**Im Paper:** Eine zufällige Teststichprobe, Mapping auf Trainingscluster und anschließende logistische Vorhersage entsprechen dem beschriebenen Verfahren. Die Abweichung ist unsere reduzierte Teststichprobe von 10.000 statt 20.000 Zellen.

## 3.18 Was bedeutet ein ausgewählter Cluster?

Ein Cluster gilt beim Export als wirksam, wenn:

$$
|\beta_k|>10^{-10}
$$

Dabei bedeutet:

- $\beta_k$: Koeffizient des Clusters im ausgewählten finalen Modell.
- $|\beta_k|$: sein Betrag.
- $10^{-10}$: numerische Toleranz.

**Warum so?** Sehr kleine Rundungsreste sollen nicht als biologisch ausgewählte Cluster gezählt werden.

Diese Toleranz steuert Clusterzählung und Profilexport. Sie ersetzt nicht die L1-Optimierung.

Es ist möglich, dass die Regularisierung alle Clusterkoeffizienten auf null setzt. Dann bleibt nur der Intercept. Bei sieben Trainingsspendern pro Klasse ergibt ein solches Nullmodell grundsätzlich eine Wahrscheinlichkeit von 0,5.

Das wäre eine gültige Modellauswahl, keine automatisch fehlgeschlagene Berechnung.

**Im aktuellen Lauf treten drei Nullmodelle auf:** In den Splits 19, 24 und 26 ist kein Cluster wirksam. Pro Split werden zwischen **null und zwölf** wirksame Cluster ausgewählt. Diese Splits bleiben in der Leistungsbewertung und im Nenner der Aufgabe-5-Wiederkehr enthalten.

**Papervergleich:** Die Auswahl von Clustern über Nichtnull-Koeffizienten entspricht der L1-Idee. Unsere konkrete numerische Toleranz ist ein Implementierungsdetail.

## 3.19 Ein tatsächliches Beispiel: Split 0

Im aktuellen Split 0 gibt es:

- 140.000 Trainingszellen;
- 3.117 zulässige Cluster;
- ausgewähltes Lambda etwa **0,199902**;
- **sechs** wirksame Cluster im finalen Modell.

Die Vorhersagen lauten:

| Spender | Tatsächliche Klasse | CMV+-Wahrscheinlichkeit | Vorhersage |
|---|---|---:|---|
| a_002 | CMV+ | 0,3624 | CMV− |
| a_004 | CMV− | 0,2538 | CMV− |
| a_006 | CMV− | 0,4807 | CMV− |
| a_1a | CMV− | 0,7824 | CMV+ |
| a_2a | CMV− | 0,6433 | CMV+ |
| a_4a | CMV+ | 0,7131 | CMV+ |

Damit werden:

- einer von zwei positiven Spendern erkannt;
- zwei von vier negativen Spendern korrekt negativ eingestuft.

Sensitivität und Spezifität betragen jeweils 0,5. Daher beträgt auch die Balanced Accuracy 0,5.

Die ROC-AUC dieses Splits beträgt ebenfalls 0,5. Das folgt hier aus der Rangfolge der sechs Wahrscheinlichkeiten, nicht unmittelbar aus der Fehlerzahl an der Schwelle.

## 3.20 Wie gut ist Citrus im aktuellen Lauf?

Die Kennzahlen wurden aus den **180 vorhandenen Testvorhersagen über 30 Splits** neu berechnet:

| Kennzahl | Mittelwert | Median |
|---|---:|---:|
| **ROC-AUC** | **0,60625** | **0,6250** |
| Average Precision | 0,5783 | 0,5417 |
| Trapezoidale PR-AUC | 0,5331 | 0,6646 |
| Balanced Accuracy | 0,5292 | 0,5000 |

Die Bedeutungen sind:

- **ROC-AUC:** Wie häufig positive Spender höhere Scores als negative erhalten; Gleichstände zählen halb.
- **Average Precision:** Zusammenfassung der Precision bei zunehmendem Recall.
- **Trapezoidale PR-AUC:** Anders berechnete Fläche unter der Precision-Recall-Kurve.
- **Balanced Accuracy:** Mittelwert aus Sensitivität und Spezifität an der Schwelle 0,5.

Precision ist der Anteil tatsächlich positiver Spender unter den positiv eingestuften. Recall entspricht der Sensitivität.

Das Citrus-Notebook selbst berechnet ROC-AUC und Balanced Accuracy. Die zusätzlichen PR-Kennzahlen werden im gemeinsamen Vergleich einheitlich bestimmt.

**Einordnung:** Die mittlere ROC-AUC von 0,60625 liegt über 0,5, bei deutlicher Streuung zwischen den Splits. Dieses deskriptive Ergebnis beschreibt die aktuelle Konfiguration auf einer kleinen Kohorte; es ist kein allgemeiner Leistungsnachweis.

Für einen fairen Vergleich ergeben sich auf **denselben 30 Splits**:

| Methode | Mittlere ROC-AUC | Mediane ROC-AUC |
|---|---:|---:|
| CellCNN | 0,8250 | 0,8750 |
| SVM | 0,8083 | 0,8750 |
| Citrus | 0,60625 | 0,6250 |

Die Wiederholungen verwenden dieselben 20 Spender mehrfach. Sie sind keine 30 unabhängigen Studien.

**Papervergleich:** Das Paper bewertet Citrus ebenfalls über Test-ROC-AUC. Seine 100 Wiederholungen und größeren Zellstichproben unterscheiden sich von unserem aktuellen Umfang.

## 3.21 Welche Ergebnisse speichern wir?

Die zentralen Ausgaben sind:

| Datei | Inhalt | Zweck |
|---|---|---|
| [Testvorhersagen](tables/task4_citrus_predictions_gated_alive_full.csv) | 180 Zeilen mit Wahrscheinlichkeit, Label, Lambda und Clusterzahl | Leistungsbewertung |
| [Auswahlübersicht](tables/task4_citrus_selection_gated_alive_full.csv) | Pro Split gewähltes Lambda, zulässige und ausgewählte Clusterzahl | Dokumentation der Modellauswahl |
| [Clusterprofile](tables/task4_citrus_clusters_gated_alive_full.csv) | Markerzentroiden und Koeffizienten ausgewählter Cluster | Vorbereitung der biologischen Interpretation |

Die Konfigurationsdatei dokumentiert Parameter, Paketversionen, Quellstände, Eingabeprüfsummen und relevante Trainingsfunktionen.

Nach jedem abgeschlossenen Split werden Zwischenstände gespeichert.

**Warum so?** Das ermöglicht Fortsetzung und verhindert, dass Ergebnisse anderer Parameter unbemerkt wiederverwendet werden.

Die CSVs enthalten allerdings **nicht den vollständigen Clusterbaum, den vollständigen Modellzustand oder sämtliche inneren Lambda-Fehlerraten**. Anders als bei unseren gespeicherten SVM-Parametern reichen die Clusterprofile allein nicht aus, um beliebige neue Spender vollständig vorherzusagen.

Es gibt am Ende 30 finale Split-Modelle im Ablauf, aber kein zusätzliches Modell, das auf allen 20 Spendern trainiert wurde.

## 3.22 Wie interpretieren wir die ausgewählten Populationen?

Für jeden wirksamen Cluster berechnet das Notebook den Mittelwert jedes transformierten Markers über dessen Trainingszellen. Diese 37 Mittelwerte bilden den **Markerzentroiden**.

Er beschreibt beispielsweise, ob die ausgewählte Population im Mittel hohe Werte für bestimmte NK-Marker besitzt.

Dabei müssen drei Dinge unterschieden werden:

| Größe | Bedeutung |
|---|---|
| Clusterhäufigkeit | Wie häufig die Population bei einem Spender vorkommt |
| Regressionskoeffizient | Wie ihre Häufigkeit zur gemeinsamen Vorhersage beiträgt |
| Markerzentroid | Welches durchschnittliche Markerprofil ihre Trainingszellen besitzen |

**Warum diese Trennung?** Ein hoher Markerwert im Zentroiden ist kein Markergewicht des Klassifikators. Der Klassifikator verwendet die Häufigkeit des gesamten Clusters.

Außerdem sind hierarchische Cluster überlappend und korreliert. Ein positiver Koeffizient bedeutet deshalb keine unabhängige kausale Wirkung eines Zelltyps.

**Im Paper:** Ausgewählte Citrus-Populationen werden ebenfalls biologisch charakterisiert und über Wiederholungen zusammengefasst. Unser Notebook exportiert dafür Profile; die weiterführende Interpretation gehört zu Aufgabe 5.

Die aktuelle Interpretation verwendet `task5_paper_*` über dieselben 30 Splits und ist in den [Methodenerklärungen zu Aufgabe 5](AUFGABE_5_METHODENERKLAERUNGEN.md) beschrieben. Ältere `task5_*`-Dateien und der Detailbericht bleiben historische Ergebnisse.

Für diese Erklärung wurden das aktuelle Notebook, die tatsächlich installierten Citrus-Funktionen, die Supplementary Methods und die gespeicherten Ergebnisse geprüft. Die Kennzahlen wurden neu berechnet; neue Clusterbäume oder Regressionsmodelle wurden nicht trainiert.
