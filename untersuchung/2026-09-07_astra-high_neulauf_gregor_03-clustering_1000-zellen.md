**Neulauf von Aufgabe 3: 1.000 Zellen pro Spender und zusätzliche k-Werte**

Datum: 07.09.2026 · Kontext: Gregor / Astra High

Das [Clustering-Notebook](../notebooks/03_clustering.ipynb) wurde vollständig mit einem frischen Python-Kernel ausgeführt. Die Tabellen, Markerprofile und UMAP-Abbildungen sind im Notebook gespeichert und nach dem Push im Repository einsehbar.

**Einstellungen und Umfang**

- 20 Spender mit jeweils 1.000 lebenden PBMCs: **20.000 Zellen × 37 Marker**.
- `K_VALUES = [2, 3, 4, 6, 8, 10, 12]`; k=2 und k=3 erweitern das Raster des ursprünglichen 200-Zellen-Laufs.
- K-means sowie Single-, Average-, Complete- und Ward-Linkage; Leiden mit 10/30/60 Nachbarn und Resolution 0,3/0,7/1,2.
- Vorverarbeitung: Arcsinh mit Cofaktor 5, anschließend Z-Standardisierung. Seed: 42.
- Auswahl: minimaler Davies–Bouldin-Index unter den Konfigurationen mit mittlerer Clusterstabilität ≥ 0,6.
- Zehn Resampling-Wiederholungen mit jeweils 80 % der ursprünglichen Zellen pro Spender: 16.000 Zellen je Wiederholung, Skalierung jeweils neu angepasst.
- **44 ursprüngliche Konfigurationen und 440 Resampling-Konfigurationen**.
- Gesamtdauer einschließlich Abbildungen und Abschlussprüfungen: etwa 20.1 Minuten auf der lokalen Maschine.

Vor diesem Neulauf enthielt der aktuelle Repository-Stand bereits die Korrekturen für Datenpfad, Konfigurationszuordnung und Ausgabe der Auswahlgrenze. Er war auf 500 Zellen pro Spender und zusätzlich k=14 eingestellt. Für diesen Lauf wurden die ausdrücklich vereinbarten 1.000 Zellen und das oben genannte Raster verwendet. Der Datenpfad benötigte keine weitere Korrektur.

Für die größere Stichprobe werden die sieben Schnitte jedes hierarchischen Baums gemeinsam mit `cut_tree(tree, n_clusters=K_VALUES)` berechnet. Dadurch entfällt das wiederholte Durchlaufen desselben Baums. Vor dem Neulauf wurden 56 Schnitte über alle vier Linkages und zwei Reihenfolgen der k-Werte mit den bisherigen Einzelaufrufen verglichen: Die Labels stimmen exakt überein. Ein vorheriger Lauf mit Einzelaufrufen wurde dafür abgebrochen; die angegebene Laufzeit beschreibt den anschließenden vollständigen Lauf.

**Ausgewählte Konfigurationen**

| Methode | Ausgewählte Parameter | Davies–Bouldin ↓ | Mittlere Stabilität |
| --- | --- | --- | --- |
| k-means | k=4 | 2.363 | 0.995 |
| Single linkage | k=2 | 0.264 | 1.000 |
| Average linkage | k=2 | 0.894 | 0.602 |
| Complete linkage | k=2 | 0.948 | 0.617 |
| Ward linkage | k=6 | 2.522 | 0.611 |
| Leiden | 30 neighbours, resolution=1.2; 18 clusters | 2.361 | 0.683 |

Die Auswahl erfolgt separat für jede Methode beziehungsweise Linkage. „No eligible configuration“ bedeutet, dass keine getestete Einstellung die Auswahlbedingungen erfüllt; es wird kein Ersatzgewinner eingesetzt.

**Clustergrößen und einzelne instabile Gruppen**

| Methode | Cluster | Kleinstes Cluster | Anteil größtes Cluster | Cluster mit Median-Jaccard < 0,6 |
| --- | --- | --- | --- | --- |
| k-means | 4 | 3148 | 37.780 % | 0 |
| Hierarchical (single) | 2 | 1 | 99.995 % | 0 |
| Hierarchical (average) | 2 | 5 | 99.975 % | 1 |
| Hierarchical (complete) | 2 | 8 | 99.960 % | 1 |
| Hierarchical (ward) | 6 | 519 | 38.220 % | 3 |
| Leiden | 18 | 223 | 17.940 % | 4 |

Die mittlere Stabilität gewichtet Cluster gleich und kann schwache Gruppen verdecken. Hohe Stabilität oder ein niedriger Davies–Bouldin-Wert beweisen insbesondere bei sehr ungleichen Clustergrößen oder Einzelzellen keine biologisch sinnvolle Zellpopulation. Die individuellen Stabilitäten und Markerprofile stehen im Notebook.

**Vergleich mit dem geprüften 200-Zellen-Lauf**

| Methode | 200 Zellen/Spender: Parameter | DB bei 200 | 1.000 Zellen/Spender: Parameter | DB bei 1.000 |
| --- | --- | --- | --- | --- |
| k-means | k=4 | 2.385 | k=4 | 2.363 |
| Single linkage | k=4 | 0.364 | k=2 | 0.264 |
| Average linkage | k=4 | 0.699 | k=2 | 0.894 |
| Complete linkage | k=4 | 1.834 | k=2 | 0.948 |
| Ward linkage | No eligible configuration | — | k=6 | 2.522 |
| Leiden | 30 neighbours, resolution=0.7; 8 clusters | 2.345 | 30 neighbours, resolution=1.2; 18 clusters | 2.361 |

Die Referenz verwendet die beim früheren Prüflauf neu berechneten Ergebnisse mit Stabilitätsgrenze 0,6 und `K_VALUES = [4, 6, 8, 10, 12]`, nicht die damals veralteten gespeicherten Ausgaben mit einer zu 0,75 passenden Zulassung. Sie ist im [Untersuchungsbericht](2026-09-07_astra-high_untersuchung_gregor_03-clustering.md) beschrieben.

Stichprobengröße und Parameterraster ändern sich gleichzeitig. Unterschiede lassen sich deshalb nicht allein der größeren Zellzahl zuschreiben. Scores auf verschiedenen Stichproben sind kein kontrollierter Nachweis einer Qualitätsverbesserung. Numerische Cluster-IDs sind zwischen Läufen nicht direkt vergleichbar.

**Abschlussprüfungen**

- Vollständige Ausführung aller Codezellen ohne Fehler; alle gespeicherten Ausgaben stammen aus diesem Neulauf.
- Matrixgröße, endliche Werte, 1.000 Zellen je Spender und eindeutige Spender/Event-Paare geprüft.
- 44 Konfigurationen, 440 Resampling-Konfigurationen und 16.000 berücksichtigte Zellen je Konfiguration und Wiederholung geprüft.
- Label-Längen, Clusterzahlen und Summe der Clustergrößen geprüft; Scores aus den gespeicherten Partitionen neu berechnet und abgeglichen.
- Zulassung mit der Grenze 0,6 sowie Auswahl des kleinsten Scores je zugelassener Methode geprüft.
- Markerprofile unabhängig als Mediane der zugehörigen Arcsinh-Zeilen nachgerechnet.
- Kleinste Anzahl auswertbarer Wiederholungen über alle ursprünglichen Cluster: 6 von 10. Eine Mindestanzahl ist weiterhin kein Filterkriterium.

Verwendete Paketversionen: numpy 2.5.2, pandas 2.3.3, scipy 1.18.0, scikit-learn 1.9.0, scanpy 1.12.4, leidenalg 0.12.0, flowio 1.4.0.

Die biologische Zelltypzuordnung und Bewertung möglicher Spender-/Aufnahmetagseffekte bleiben die im Untersuchungsbericht beschriebenen nächsten fachlichen Schritte. Dieser Neulauf erweitert die Stichprobe und aktualisiert die Rechenergebnisse.
