**Untersuchung von `02_dimensionality_reduction_clean.ipynb` und Einordnung für Aufgabe 2**

Datum: 08.09.2026 · Kontext: Prüfung auf Vollständigkeit, Codefehler und Abgabereife

Untersucht wurden ausschließlich das [bereinigte Notebook](../notebooks/02_dimensionality_reduction_clean.ipynb) und Aufgabe 2 auf Seite 1 der [Aufgabenstellung](../Group_projects_ssbi_2026.pdf). Dieser Bericht ersetzt die vorherige Untersuchung, die irrtümlich auf die Variante ohne `_clean` bezogen war. Die Notebooks und ihre Analyseartefakte wurden nicht verändert.

**Die `_clean`-Version deckt die technische Aufgabenstellung weitgehend ab.** PCA, sechs t-SNE-Parameterbeispiele, 16 UMAP-Parameterbeispiele und quantitative Vergleiche aller drei Verfahrenspaare sind vorhanden. Der Procrustes-Vergleich erfüllt bereits die Forderung nach einem paarweisen Maß. Vor der Abgabe müssen jedoch der zusätzliche kNN-Vergleich korrigiert, Widersprüche zwischen Text und tatsächlichem Versuchsaufbau bereinigt und die Ergebnisse abschließend interpretiert werden.

**Die zuvor berichteten Ausführungsfehler gelten für diese Version nicht:** Die GMM-Funktion ist vor ihrer Verwendung definiert, `labels` wird nicht mit Plotbeschriftungen überschrieben, eine Verwendung des nicht importierten Moduls `warnings` und der problematische t-SNE-Cache kommen hier nicht vor. Auch der frühere Vorwurf fehlender t-SNE-Beispiele und eines 48er-UMAP-Sweeps mit nur 16 Bildern trifft hier nicht zu.

Zellangaben zählen alle Code- und Markdown-Zellen ab 1, unabhängig von Jupyters Ausführungsnummern. Untersucht wurden **50 Zellen, darunter 29 Codezellen**. SHA-256 des Notebooks: `59b57fa69d587ed04d126f17d86be19e2e995c964a7919ac127fcf977788543b`.

---

**1. Ist jede Teilanforderung von Aufgabe 2 beantwortet?**

Die Aufgabenstellung verlangt die Visualisierung mit PCA, t-SNE und UMAP, die Erklärung wählbarer Parameter und ihrer Auswirkungen anhand von Beispielen sowie die Auswahl eines quantitativen Maßes und dessen Anwendung auf alle Visualisierungspaare.

| Teilanforderung | Fundstelle | Bewertung | Begründung |
|---|---|---|---|
| Bereitgestellten Datensatz verwenden | Zellen 4, 8, 10 | Erfüllt | Geladen werden 40.000 spenderbalanciert ausgewählte NK-Zellen und der vollständige NK-Bestand mit 261.593 Zellen. Der Datenimport wurde lokal ausgeführt. |
| PCA, t-SNE und UMAP visualisieren | Zellen 20–22, 28–39 | Erfüllt im dokumentierten Stand | Berechnungen und gespeicherte Abbildungen aller drei Verfahren sind vorhanden, einschließlich gemeinsamer Darstellung mit CMV-Färbung. Die großen t-SNE-/UMAP-Einbettungen wurden im Audit nicht neu trainiert. |
| Wählbare Parameter erklären | Zellen 19–21, 27, 31 | Weitgehend erfüllt | PCA-Komponentenzahl, t-SNE-Perplexity und weitere Optimierungseinstellungen sowie UMAP-Nachbarschaftsgröße und Mindestabstand werden erläutert. Einzelne Aussagen sind zu pauschal; siehe Abschnitt 3. |
| Parameterwirkung anhand von Beispielen dokumentieren | Zellen 20–22, 28–29, 32–33 | Teilweise erfüllt | Varianz-/Rekonstruktionskurve, sechs t-SNE-Beispiele und 16 UMAP-Beispiele sind vorhanden. Die Texte erklären allgemeine Erwartungen, diskutieren die konkret gezeigten Unterschiede aber noch zu wenig. |
| Geeignete quantitative Maße erläutern und auswählen | Zellen 26, 42, 46 | Erfüllt mit begrifflichen Korrekturen | Lokale Nachbarschaftsmaße, Pearson-Korrelation paarweiser Distanzen, Procrustes und CMV-Silhouette werden unterschieden. Die Bezeichnung „Distance correlation“ ist missverständlich. |
| Ein Maß für alle Visualisierungspaare berechnen | Zelle 47 | Erfüllt durch Procrustes | PCA–t-SNE, PCA–UMAP und t-SNE–UMAP werden berechnet. Die Procrustes-Werte wurden auf den vorhandenen Exporten reproduziert und unabhängig kontrolliert. Der zusätzlich berichtete kNN-Vergleich ist fehlerhaft. |
| Ergebnisse als verständliche Antwort zusammenführen | Insbesondere Zelle 48 | Noch nicht abgeschlossen | Die Zusammenfassung enthält Platzhalter und Arbeitsanweisungen. Es fehlt eine konkrete Einordnung der Ergebnisse, besonders der unterschiedlichen Rangfolgen je Vergleichsmaß. |

Die letzte Zeile beschreibt die Ausarbeitung der Antwort, keine zusätzliche Rechenanforderung. **Die Aufgabe braucht vor allem Korrekturen und eine fertige Ergebnisdiskussion, nicht weitere Verfahren oder umfangreiche zusätzliche Sweeps.**

---

**2. Bestätigte Fehler und konkrete Auswirkungen**

**2.1 Die kNN-Preservation vergleicht die falschen Nachbarn**

In Zelle 26 fordert `knn_preservation()` jeweils `k + 1` Nachbarn an und entfernt anschließend den ersten Eintrag mit `[:, 1:]`. Bei `kneighbors()` ohne übergebenes Abfragearray schließt scikit-learn den eigenen Punkt bereits aus. Die Funktion verwendet deshalb die Nachbarränge 2 bis k+1 statt 1 bis k. Dieses Verhalten ist in der [NearestNeighbors-Dokumentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors) beschrieben und wurde mit der lokalen Implementierung bestätigt.

**Unabhängiger Funktionstest:** Zwei Zufallsarrays mit Seed 42, 20 Punkten und drei beziehungsweise zwei Dimensionen wurden bei k=3 verglichen. Eine unabhängige Kontrollrechnung sortiert die vollständigen euklidischen Distanzmatrizen nach Ausschluss der Diagonale.

| Berechnung | Nachbarschaftsüberlappung |
|---|---:|
| Originalfunktion aus Zelle 26 | 0,150000 |
| Unabhängige Distanzmatrixrechnung | 0,216667 |
| API-Aufruf mit genau k Nachbarn ohne Abschneiden | 0,216667 |

**Nachweis auf den tatsächlichen gespeicherten Einbettungen:** Aus `embeddings_full.parquet` wurden die drei zweidimensionalen Koordinatensätze geladen. Die Metadaten stimmen zeilenweise mit den aktuellen 40.000 Eingabezellen überein; die gespeicherten PCA-Koordinaten stimmen exakt mit der neu berechneten PCA überein. Die Originalfunktion reproduziert die im Notebook gespeicherten Paarwerte auf die gezeigte Genauigkeit. Mit der korrigierten Nachbarwahl ergeben sich:

| Paar | Originaler kNN-Wert | Korrigierter kNN-Wert | Procrustes-Disparität, unverändert |
|---|---:|---:|---:|
| PCA–t-SNE | 0,005585 | **0,005850** | 0,451891 |
| PCA–UMAP | 0,003975 | **0,004095** | 0,733897 |
| t-SNE–UMAP | 0,105957 | **0,125318** | 0,723692 |

Die korrigierten Werte sind Neuberechnungen **auf den vorhandenen Einbettungsexporten**, kein erneutes Training von t-SNE oder UMAP. Der Export hat SHA-256 `e38dcc0e75593564ed8d85750c4c2b36e4dcb7d5c1607e07c3f7d774b69b620a`.

**Auswirkung:** Betroffen sind die kNN-Spalten beider Sweeps in Zellen 28 und 32, der Qualitätsvergleich in Zelle 43 sowie Tabelle und Heatmap in Zelle 47. Die finale Parameterauswahl basiert auf Trustworthiness und wird durch diesen kNN-Fehler allein nicht geändert. Auch Procrustes ist unabhängig davon korrekt.

**Korrektur:** Auf beiden Seiten `NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)` verwenden. Alle kNN-Ausgaben anschließend aktualisieren. Neben Identität und Symmetrie unbedingt gegen eine unabhängige Nachbarrechnung testen: Der Identitätstest allein erkennt den aktuellen Fehler nicht.

**2.2 Text und Code verwenden unterschiedliche Auswertungsdaten**

Zelle 42 behauptet, alle nachfolgenden Nachbarschafts- und Distanzkorrelationsmaße würden dieselben 5.000 Zellen und `X_pca_sweep` wie die Sweeps verwenden. Tatsächlich setzt Zelle 43 `reference_space = X_pca_full` und iteriert über `embeddings_full`.

| Auswertung | Zellen | Referenz / Koordinaten |
|---|---:|---|
| t-SNE- und UMAP-Sweeps | 5.000 | `X_pca_sweep`: 29 PCs |
| Finale Trustworthiness und kNN gegen Referenz | 40.000 | `X_pca_full`: 30 PCs |
| Pearson-Korrelation paarweiser Distanzen | 2.000 aus den 40.000 | Auswahl innerhalb der finalen Referenz und Einbettungen |
| CMV-Silhouette | 40.000 | Finale 2D-Koordinaten und zugehörige CMV-Labels |
| Paarvergleiche in Zelle 47 | 40.000 | Finale 2D-Koordinaten |

**Auswirkung:** Die Aussage, finale Scores seien direkt mit den Sweep-Tabellen vergleichbar, stimmt nicht. Das ist kein nachgewiesener Zeilenversatz: Innerhalb der jeweiligen finalen Auswertung sind die Eingaben korrekt aufeinander ausgerichtet. Ein 30-PC-Referenzraum ist ebenfalls nicht grundsätzlich unzulässig; er entspricht nur nicht der beschriebenen 29-PC-Sweep-Referenz.

Zusätzlich berechnet `trustworthiness()` in der lokalen scikit-learn-Version quadratische Distanz- und Rangmatrizen. Bei 40.000 Punkten hat eine solche Matrix 1,6 Milliarden Einträge, also etwa 12,8 GB bei Float64/Int64. Mehrere dieser Matrizen existieren gleichzeitig. Der Aufwand kann damit viele zehn GB RAM erreichen. Das ist eine aus dem Quellcode abgeleitete Ressourcenanforderung, kein im Audit beobachteter Speicherabsturz.

**Korrektur:** Den Versuchsaufbau eindeutig festlegen. Naheliegend ist der bereits angekündigte Vergleich auf `embeddings_sweep` mit `X_pca_sweep` und `clinical_group[sweep_idx]`; die großen Einbettungen bleiben für die Übersicht und Markerplots. Wenn stattdessen die 40.000-Zell-Auswertung beibehalten wird, müssen Text, Referenzdimension, Teilstichprobe der Distanzkorrelation und Ressourcenbedarf ausdrücklich dazu passen. Die Tabelle oben beschreibt den aktuellen Code, keine Empfehlung für neue Pflichtanalysen.

**2.3 Das Ergebnisfazit ist noch eine Vorlage**

Zelle 48 enthält unter anderem `N`, `N_COMPONENTS_SELECTED`, alternative Formulierungen in eckigen Klammern und die Aufforderung, Zahlen später einzutragen. Dabei liegen die Ergebnisse bereits vor.

**Auswirkung:** Leser müssen die Antwort aus vielen Ausgaben selbst zusammensuchen. Die zentrale Frage, wie ähnlich sich die drei Visualisierungen sind, wird noch nicht abschließend beantwortet.

**Korrektur:** Die Vorlage durch ein fertiges Fazit ersetzen. Dazu gehören 37 behaltene Marker, die Auswahl von 29 PCs für mindestens 90 % erklärte Varianz, die dokumentierten finalen Parameter sowie eine Interpretation der Paarvergleiche. Die korrigierten kNN-Werte zeigen t-SNE und UMAP als ähnlichstes Paar hinsichtlich lokaler Nachbarn. Procrustes bewertet dagegen PCA und t-SNE als ähnlichstes Paar hinsichtlich der globalen Punktanordnung nach Ausrichtung. Dieser Unterschied ist ein erklärungsbedürftiges Ergebnis, kein Widerspruch der Berechnungen.

**2.4 Die t-SNE-Ergebnistabelle wird nicht angezeigt**

In Zelle 28 steht `tsne_sweep_df.sort_values(...)` vor der anschließenden Zuweisung von `best_tsne_perplexity`. Ein solcher Zwischenausdruck wird in einer üblichen Notebook-Zelle nicht automatisch angezeigt. Die Zelle enthält tatsächlich keine gespeicherte Ausgabe. Das Raster in Zelle 29 zeigt nur auf drei Nachkommastellen gerundete Scores; mehrere Varianten erscheinen dadurch gleich gut.

**Korrektur:** Die sortierte Tabelle ausdrücklich mit `display(...)` ausgeben. Damit wird die Auswahl von Perplexity 60 nachvollziehbar. Es fehlen hier weder der Sweep noch die Auswahlberechnung, sondern die genaue Ergebnistabelle für Leser.

**2.5 Kleinere numerische beziehungsweise bedingte Fehler**

- **PCA-Rekonstruktionsfehler, Zellen 20–21:** `1 - cumulative_var` ist der relative Restvarianzanteil. Als absoluter MSE stimmt er nur bei passender Normierung exakt. Die verwendete Scanpy-Skalierung ergibt für den mit `np.mean` berechneten Fehler hier den zusätzlichen Faktor `(n-1)/n`. Neu berechnet wurden MSE **0,0850197352** und die unbereinigte Formel **0,0850218607**; mit dem mittleren Merkmalvarianzwert bei `ddof=0` als Faktor stimmen beide bis auf Rundung überein. Bei vier Nachkommastellen ist der Unterschied unsichtbar. Die Formel präzisieren oder die Kurve als relativen Rekonstruktionsfehler beschriften; das ist kein wesentlicher Ergebnisfehler.
- **Nicht erreichte PCA-Varianzschwelle, Zelle 21:** Liegt die gesamte berechnete Kurve unter 0,90, gibt `searchsorted` einen Index hinter dem letzten Eintrag zurück. Die folgende Indizierung erzeugt einen `IndexError`. Mit einer verkürzten Testkurve reproduziert; auf den aktuellen Daten werden dagegen korrekt 29 PCs gewählt. Vor der Auswahl prüfen, ob die Schwelle erreicht wird, und andernfalls mehr zulässige PCs berechnen oder den Fall ausdrücklich melden.
- **GMM ohne Schwellenübergang, Zelle 16:** Wenn `posterior_pos > 0.5` überall falsch ist, liefert `argmax` dennoch 0 und die Funktion gibt den ersten Rasterwert als vermeintliche Schwelle zurück. Mit einem simulierten Modell reproduziert, nicht für die realen Marker nachgewiesen. Einen tatsächlich vorhandenen Übergang prüfen. Die aktuellen NKG2C-/CD57-Schwellen wurden hingegen erfolgreich reproduziert.
- **Unvollständiger Subsample-Export, Zellen 49–50:** Angekündigt werden Einbettungen für den Vergleich aller drei Verfahren; die t-SNE-Spalten sind im 5.000-Zell-Export auskommentiert. Die vorhandene Datei bestätigt dies. Der 40.000-Zell-Export enthält alle drei Verfahren. Den kleinen Export vervollständigen oder seine Beschreibung anpassen; das ist keine fehlende Pflichtberechnung innerhalb des Notebooks.

---

**3. Methodische Begründung, Interpretation und Lesbarkeit**

**Vorverarbeitung und Datenfluss funktionieren im geprüften Abschnitt.** Alle Codezellen bis einschließlich Zelle 26 wurden mit den vorhandenen Daten in einem neuen Python-Prozess ausgeführt. Es gab keinen Ausführungsfehler. AnnData wird beim Plotten teilweise auf kategoriale Metadatentypen umgestellt; ein strikter DataFrame-Typvergleich fällt dadurch unterschiedlich aus. Die Metadatenwerte und ihre Reihenfolge stimmen jedoch vollständig überein.

| Kontrolle mit realen Daten | Ergebnis |
|---|---|
| Eingabestichprobe | 40.000 Zellen; 2.000 pro Spender |
| Vollständiger NK-Bestand | 261.593 Zellen |
| Marker / Varianzfilter | Alle 37 behalten; kein Marker unter 0,05 |
| Vorverarbeitete Matrix | 40.000 × 37; alle Werte endlich |
| PCA | 30 Komponenten berechnet, 29 ausgewählt |
| Erklärte Varianz | PC1–2: 20,29 %; PC1–29: 91,50 %; PC1–30: 92,96 % |
| Sweep-Stichprobe | 5.000 × 29; alle 20 Spender vertreten, 220–281 Zellen pro Spender |
| GMM-Schwellen | NKG2C etwa 0,0858455; CD57 etwa 0,0845767 |
| Metadaten der großen Einbettungsexporte | Werte und Reihenfolge passen zur Eingabe; PCA-Koordinaten exakt reproduziert |

**Parametererklärungen konkretisieren, ohne weitere große Sweeps zu verlangen.**

- In Zelle 19 ist die Behauptung, die Komponentenzahl sei der einzige echte PCA-Parameter, zu absolut. Solver und gegebenenfalls Whitening sind ebenfalls Einstellungen; die vorliegende Fixierung des Solvers ist zulässig. Komponentenzahl und dargestelltes Achsenpaar unterscheiden. Der erklärte Varianzanteil begründet die 29-dimensionalen Eingaben für t-SNE/UMAP, sagt aber nicht, dass die ersten beiden PCs 90 % der Varianz zeigen. Die [PCA-Dokumentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) beschreibt diese Einstellungen.
- Die sechs t-SNE-Beispiele sind vorhanden und lesbar. Der gespeicherte Plot zeigt bei Perplexity 5 eine diffusere Darstellung und bei größeren Werten deutlicher abgegrenzte Bereiche. Die pauschale Erwartung „kleine Perplexity erzeugt viele kleine Inseln“ sollte daher durch eine Beschreibung der tatsächlich sichtbaren Resultate ergänzt werden. Lernrate und Iterationszahl können auch das endgültige Layout beeinflussen; „750 Iterationen“ allein ist kein Konvergenznachweis.
- Der 16er-UMAP-Sweep passt zu Tabelle und Bild. `n_neighbors` und `min_dist` werden systematisch variiert; die Auswahl von 5 und 0,0 wird aus der Tabelle übernommen. Die Aussage, größere `min_dist` mache Distanzen grundsätzlich besser interpretierbar, sollte abgeschwächt werden. UMAP-Abstände sind dadurch nicht automatisch originalgetreu; die [offizielle Parametererklärung](https://umap-learn.readthedocs.io/en/latest/parameters.html) unterscheidet Nachbarschaftsskala und Packung im Embedding.

**Die Vergleichsmaße sind sinnvoll gewählt, brauchen aber präzise Namen und Grenzen.**

`distance_correlation()` in Zelle 43 berechnet Pearson-r zwischen zwei Vektoren paarweiser Distanzen. Der Funktionskörper macht das klar; die Bezeichnung kann aber mit einer anders definierten statistischen Distance Correlation verwechselt werden. „Pearson-Korrelation paarweiser Distanzen“ wäre eindeutig. Die zufälligen 2.000 Indizes werden für beide Räume identisch verwendet; hier wurde kein Zuordnungsfehler gefunden.

Procrustes vergleicht korrespondierende Punkte nach Zentrierung, Skalierung und optimaler Ausrichtung einschließlich möglicher Spiegelung. Die direkte Anwendung auf gleich geordnete 2D-Einbettungen ist korrekt; die Disparität hängt nicht von einer willkürlichen Drehung des Bildes ab. Die drei Exportwerte wurden zusätzlich über eine unabhängige SVD-Formel kontrolliert und stimmen überein. Die [SciPy-Dokumentation zu Procrustes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html) beschreibt diese Eigenschaften. Eine niedrigere Disparität zeigt Formähnlichkeit, keine bessere biologische Wahrheit.

Die Aussage in Zelle 44, ein perfektes Embedding müsse im Shepard-Diagramm auf der Diagonalen liegen, braucht eine Skalierungsbedingung. Schon eine gleichmäßige Streckung verändert die Steigung, obwohl die Pearson-Korrelation der Distanzen weiterhin 1 sein kann.

Die gespeicherten finalen Trustworthiness-Werte betragen PCA 0,724320, t-SNE 0,959955 und UMAP 0,899225. Die Pearson-Korrelationen betragen entsprechend 0,617264, 0,484853 und 0,394753. Diese Werte wurden **nicht auf allen 40.000 Zellen neu berechnet**. Sie illustrieren im dokumentierten Stand unterschiedliche Rangfolgen für lokale und globale Eigenschaften. Die CMV-Silhouetten von etwa 0,0016 bis 0,0178 sollten als schwache Trennung nach diesen Labels diskutiert werden; sie sind kein allgemeines Gütemaß einer Dimensionsreduktion und keine spenderübergreifende Klassifikationsleistung.

**Biologische Zusatzprüfungen nicht überinterpretieren.**

Die Repräsentativitätsprüfung einer über GMM-Schwellen definierten NKG2C/CD57-Gruppe ist eine hilfreiche Zusatzkontrolle. Die daraus geschätzten Anteile im Gesamtbestand liegen je Spender zwischen etwa 7,67 % und 41,55 %; in der Stichprobe werden mindestens 145 solche Zellen je Spender gefunden. Diese konkrete Definition sollte daher nicht ohne weitere Einordnung als Nachweis einer seltenen biologischen Population behandelt werden. Ein gutes Ergebnis für diese eine Gruppe validiert zudem nicht automatisch alle für die Dimensionsreduktion relevanten Strukturen, wie Zelle 18 nahelegt.

Auch die Begründung des Cofaktors in Zelle 7 sollte korrigiert werden: Bei festem positivem Rohwert vergrößert ein kleinerer Cofaktor den Wert von `arcsinh(x/c)`. Die Aussage, kleinere Cofaktoren drückten den positiven Peak an die Achse, passt weder dazu noch zu den gespeicherten Histogrammen. Cofaktor 5 kann als begründete Wahl bestehen bleiben; die begleitende Erklärung muss den sichtbaren Effekt richtig beschreiben.

„Full data“ bezeichnet bei den Einbettungen 40.000 ausgewählte Zellen, nicht alle 261.593 NK-Zellen. Das sollte in Abschnittsüberschriften, Markerplot-Titeln und Exportbeschreibung einheitlich benannt werden. Zudem zeichnet Zelle 39 sämtliche CMV+-Punkte nach den CMV−-Punkten; bei starker Überlagerung kann dies die sichtbare Farbdominanz beeinflussen. Eine gemeinsame zufällige Zeichenreihenfolge oder getrennte Panels wären eine optionale Verbesserung.

---

**4. Priorisierte Änderungen vor der Abgabe**

1. **kNN-Funktion korrigieren und betroffene Ausgaben aktualisieren.** Die drei Paarwerte sind auf den vorhandenen Exporten bereits unabhängig überprüfbar; Procrustes beibehalten.
2. **Versuchsaufbau und Beschreibung in Einklang bringen.** Insbesondere 5.000 versus 40.000 Zellen, 29 versus 30 Referenz-PCs sowie 2.000 Zellen für die Distanzkorrelation eindeutig festhalten. Die Auswahl auf dem Sweep ist eine explorative Optimierung, keine unabhängige Validierung.
3. **Die Ergebniszusammenfassung tatsächlich schreiben.** Konkrete Zahlen einsetzen, Parameterbeispiele auswerten und die unterschiedliche Aussage von kNN, Procrustes und Referenzraumtreue erklären.
4. **Präzision und Nachvollziehbarkeit verbessern.** t-SNE-Tabelle explizit anzeigen; Cofaktor-Erklärung, Metriknamen, PCA-MSE-Normierung und „full data“-Bezeichnungen korrigieren. Randfälle der PCA-Auswahl und GMM-Schwelle absichern; den Export zur Beschreibung passend machen.
5. **Abschließenden konsistenten Notebooklauf durchführen.** In einem frischen Kernel mit dem festgelegten Auswertungsumfang ausführen und sämtliche Ausgaben gemeinsam speichern. Die derzeitigen großen Einbettungen wurden für diesen Audit nicht neu berechnet.

Zusätzliche Methoden, umfangreiche weitere Parameterstudien oder Klassifikationsanalysen sind für Aufgabe 2 nicht erforderlich. Mehrere Seeds wären eine optionale Robustheitsprüfung. Die Fünf-Seiten-Vorgabe auf Seite 2 der Aufgabenstellung gilt für den gemeinsamen Abschlussbericht, nicht für diese interne Untersuchung.

---

**5. Prüfprotokoll und Aussagegrenzen**

| Prüfung | Tatsächlich durchgeführt | Ergebnis / Grenze |
|---|---|---|
| Aufgabenabgleich | Wortlaut aus Original-PDF mit der `_clean`-Version abgeglichen | Teilanforderungen in Abschnitt 1 zugeordnet |
| Statische Prüfung | Alle 50 Zellen gelesen, alle 29 Codezellen kompiliert | Keine Syntaxfehler; kein Beweis eines vollständigen erfolgreichen Laufs |
| Frischer Daten-/PCA-Lauf | Sämtliche Codezellen bis einschließlich Zelle 26 mit realen Exportdateien ausgeführt | Erfolgreich, einschließlich GMM, PCA-Auswahl und Stichprobenbildung |
| Datenzuordnung | Eingaben, AnnData-Metadaten und große Einbettungsexporte geprüft | Werte und Reihenfolge stimmen; AnnData ändert teilweise nur die Metadatentypen |
| Paarvergleich | Originalzelle 47 auf vorhandenen 40.000-Zell-Einbettungsexporten ausgeführt | Gespeicherte kNN- und Procrustes-Werte reproduziert |
| Unabhängige Metrikprüfung | kNN gegen Distanzmatrixrechnung, Procrustes gegen SVD-Formel geprüft | kNN-Fehler bestätigt; Procrustes korrekt; korrigierte Paarwerte berechnet |
| Qualitätszelle 43 | Auf 200 über den Bestand verteilten, passend zugeordneten Exportzeilen ausgeführt | Alle vier Metriken laufen; kein vollständiger 40.000-Zell-Score-Neulauf |
| Randfälle | Nicht erreichte PCA-Schwelle und GMM ohne Übergang simuliert | Bedingte Fehler reproduziert; aktuelle Daten erreichen die PCA-Schwelle |
| Abbildungen | Unter anderem Cofaktor-Raster, sechs t-SNE-Beispiele, 16 UMAP-Beispiele und gemeinsame Dreierdarstellung direkt angesehen | Beispielvisualisierungen vorhanden; konkrete Interpretationshinweise oben |

Verwendet wurde die Projekt-`.venv` mit NumPy 2.5.2, pandas 2.3.3, SciPy 1.18.0, scikit-learn 1.9.0, Scanpy 1.12.4, UMAP 0.5.12 und AnnData 0.13.3.post0. Diagramme wurden ohne interaktives Fenster erzeugt und `display()` im Prüflauf unterdrückt; die numerischen Ausdrücke blieben unverändert. Temporäre Prüfdateien liegen außerhalb der versionierten Analyse unter `/tmp`.

**Nicht durchgeführt:** erneutes Training aller sechs t-SNE- und 16 UMAP-Sweep-Modelle, erneutes Training der großen Einbettungen, vollständige 40.000-Zell-Neuberechnung von Trustworthiness und Silhouette sowie erneute FCS-Extraktion. Die Originaleinbettungen sind durch Metadatenabgleich, exakte PCA-Reproduktion und passende Paarwerte plausibilisiert; daraus folgt keine vollständige Reproduktion der Trainingsläufe. Es gibt in diesem Audit keinen belegten aktuellen Durchlaufabbruch der `_clean`-Version. Die verbleibenden Hauptprobleme sind der bestätigte kNN-Fehler, die widersprüchliche Dokumentation des Vergleichsaufbaus und die unfertige Ergebnisdiskussion.
