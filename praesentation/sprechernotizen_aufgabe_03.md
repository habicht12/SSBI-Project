**Sprechernotizen zu Aufgabe 3 · etwa 8–9 Minuten**

Die acht Hauptfolien bilden den Vortrag. Reservefolien 9–11 sind für Rückfragen gedacht. Die Zeiten sind Richtwerte und ergeben zusammen etwa 8 Minuten 50 Sekunden.

**Folie 1 · Fragestellung und Daten · 45 Sekunden**

„Für Aufgabe 3 vergleichen wir K-means, hierarchisches Clustering und Leiden. Unsere Frage ist, welche Gruppierung biologisch sinnvolle Zellpopulationen widerspiegelt. Dafür verwenden wir 20.000 lebende PBMCs mit 37 Markern. Jeder der 20 Spender trägt genau 1.000 Zellen bei. Der zentrale Punkt ist: Ein guter Clusterscore allein reicht für die biologische Entscheidung nicht aus.“

PBMC bei Bedarf als mononukleäre Zellen des peripheren Bluts erklären. Es werden die `alive`-Dateien verwendet, keine reine Stichprobe aus den bereits gegateten NK-Zellen.

**Folie 2 · Vorgehen und Auswahl · 75 Sekunden**

Die Arcsinh-Transformation reduziert den Einfluss großer Intensitätsunterschiede; anschließend werden die Marker standardisiert. Alle Verfahren bekommen dieselbe Matrix. Auch die Scores werden auf dieser Matrix berechnet, nicht auf UMAP.

K-means und die vier hierarchischen Varianten testen sieben Clusterzahlen. Leiden testet neun Kombinationen aus Nachbarzahl und Resolution. Das ergibt 44 Konfigurationen. Bei Leiden legt die Resolution die Clusterzahl nicht direkt fest.

Jede Konfiguration wird zehnmal auf einer 80-Prozent-Teilstichprobe pro Spender neu gefittet. Die Skalierung wird dabei ebenfalls neu angepasst. Mittlere Clusterstabilität ab 0,6 ist die Zulassung; danach wählen wir je Methode den kleinsten Davies–Bouldin-Index. Ein kleinerer DB-Wert steht für eine geometrisch günstigere Trennung.

**Folie 3 · Ergebnistabelle · 75 Sekunden**

Die Tabelle zeigt die Gewinner innerhalb der jeweiligen Methode, keinen automatisch bestimmten Gesamtsieger. K-means liefert vier Cluster und eine mittlere Stabilität von 0,995. Leiden liefert 18 Cluster bei 30 Nachbarn und Resolution 1,2, mit Stabilität 0,683. Ihre DB-Werte 2,363 und 2,361 liegen sehr nahe beieinander; daraus leiten wir keinen belegten Qualitätsunterschied ab.

Die auffälligen Zahlen stehen in der letzten Spalte: Single, Average und Complete legen jeweils über 99,9 Prozent der Zellen in ein Cluster. Ward liefert dagegen sechs größere Gruppen, ist aber deutlich weniger stabil als K-means.

**Folie 4 · Problem des Scores · 50 Sekunden**

„Single Linkage hat den niedrigsten DB-Wert und Stabilität 1. Trotzdem besteht die Lösung aus 19.999 Zellen plus einer einzelnen Zelle. Das ist für die Frage nach Zelltypen keine hilfreiche Aufteilung.“

Ein isoliertes Event kann beim Entfernen anderer Zellen weiterhin isoliert bleiben. Deshalb ist hohe Stabilität hier kein Gegenargument. Das zweite Cluster ist im maßstabsgetreuen Balken so klein, dass es nicht erkennbar wäre; die Anzahl steht deshalb daneben. Nicht behaupten, alle kleinen Cluster seien generell Artefakte.

**Folie 5 · Gemeinsame UMAP · 60 Sekunden**

Beide Bilder zeigen dieselben Zellen an denselben UMAP-Koordinaten. Links werden vier große Gruppen eingefärbt, rechts wird die Struktur in 18 Gruppen aufgeteilt. Die Farben sind innerhalb jeder Methode vergeben; gleiche Farben auf beiden Bildern bedeuten keine Übereinstimmung.

Die UMAP dient nur zur Darstellung. Sie ist kein zusätzlicher Nachweis für eine gute Trennung im ursprünglichen Merkmalsraum. Die feinere Leiden-Lösung erzeugt interessante Kandidaten für Untergruppen, die als Nächstes einzeln beurteilt werden müssen.

**Folie 6 · Einzelne Cluster statt nur Mittelwert · 75 Sekunden**

Jeder Punkt ist der Median-Jaccard eines Clusters über die zehn Wiederholungen. Bei K-means liegen alle vier nahe bei 1. Bei Ward liegen drei von sechs, bei Leiden vier von 18 unter 0,6. Besonders die Leiden-Cluster C15 und C16 sind mit etwa 0,11 und 0,12 instabil.

Die gestrichelte Linie markiert 0,6 zum Vergleich. Wichtig: Im Notebook wird diese Grenze nur auf den Mittelwert der Cluster-Mediane angewendet. Sie ist kein Einzelclusterfilter. Die Lösung kann daher zugelassen werden, obwohl einzelne Gruppen instabil sind.

**Folie 7 · Biologische Interpretation · 90 Sekunden**

Hier werden vorsichtige Hypothesen aus den relativen Markerprofilen formuliert. CD19 und HLA-DR sind in K-means-C0 relativ erhöht, CD3 ist erniedrigt. Ein ähnliches Profil erscheint bei Leiden-C3. Das passt zu einer B-Zell-artigen Gruppe. K-means-C3 und Leiden-C4 zeigen NK-assoziierte Marker und sind Kandidaten für NK-Zell-artige Gruppen.

Nicht sagen, dass die beiden Verfahren damit nachweislich dieselben Zellen gruppieren: Verglichen werden hier die Profile. Auch nicht alle vier K-means-Cluster als vier identifizierte Zelltypen bezeichnen. Die übrigen Profile sind nicht eindeutig genug.

Die Pfeile bedeuten Abweichungen des Cluster-Medians vom Stichprobenmedian, skaliert mit dem Stichproben-IQR. Sie bedeuten keine experimentell festgelegte Positivität. Für belastbare Zelltypen müssten weitere Marker und Koexpression auf Einzelzellebene geprüft werden. Horowitz et al. dient als biologischer Kontext; unsere Zuordnungen bleiben eigene, vorläufige Interpretationen.

**Folie 8 · Begründete Bewertung · 60 Sekunden**

„Für eine robuste grobe Aufteilung ist K-means in diesem Lauf unsere Ausgangslösung. Leiden liefert feinere Hypothesen, ist aber bei mehreren Untergruppen instabil. Die stark degenerierten Single-, Average- und Complete-Lösungen helfen bei der Zelltypfrage wenig; Ward ist weniger stabil.“

Für eine endgültige biologische Entscheidung fehlen validierte Markerzuordnungen sowie die Kontrolle von Spender- und Aufnahmetagseffekten. Neue Ausgangsstichproben sind besonders bei kleinen Gruppen wichtig. Damit beantworten wir die quantitative Frage und begründen unsere derzeitige Einschätzung, ohne einen biologischen Sieger zu behaupten, den die vorhandenen Daten noch nicht belegen.

**Reservefolien**

- **9:** Der Vergleich zum 200er-Lauf ändert Stichprobengröße und k-Raster gleichzeitig. Die Referenz verwendet die korrekte Stabilitätsgrenze 0,6. Keine kausale Zuordnung des Unterschieds allein zur Zellzahl.
- **10:** Jaccard auf genau denselben behaltenen Zell-IDs erklären. Cluster-IDs dürfen sich ändern; gesucht wird die beste Mengenüberlappung. Die Referenz ist keine Ground Truth. Die Skala wird bei jedem Resampling neu gefittet.
- **11:** Quellen und Reproduzierbarkeit; das ausgeführte Notebook enthält alle Tabellen und vollständigen Grafiken. Kein erneutes 20-Minuten-Clustering nötig, um die Präsentation zu kompilieren.
