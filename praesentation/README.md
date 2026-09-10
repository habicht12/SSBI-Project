**Präsentation zum SSBI-Gruppenprojekt**

## Ordnerstruktur

Jede Aufgabe mit eigenen Folien, Tabellen und Abbildungen liegt in einem
separaten Unterordner. Gemeinsame LaTeX-Definitionen liegen direkt in diesem
Ordner.

```text
praesentation/
|-- theme.tex                         gemeinsames Beamer-Theme
|-- aufgabe_02/                       Dimensionality reduction
|   |-- slides.tex                    Haupteinstiegspunkt
|   |-- figures/                      Folienabbildungen
|   `-- data/                         Tabellen und Zahlenmakros
|-- aufgabe_06_vergleich/             Vergleich der CellCNN-Varianten
|   |-- slides.tex                    Haupteinstiegspunkt
|   |-- figures/                      Folienabbildungen
|   `-- data/                         Tabellen, Zahlen und Provenienz
`-- aufgabe_03_clustering.tex         Clustering-Einstiegspunkt
```

Die jeweilige `figures/`- und `data/`-Mappe gehört zur Aufgabe und wird nicht
mit den Assets einer anderen Aufgabe vermischt. LaTeX-Zwischendateien und
neu erzeugte PDFs gehören in einen lokalen `build/`-Ordner.

### Aufgabe 6

Der vollständige Foliensatz zum Vergleich liegt in
[aufgabe_06_vergleich](aufgabe_06_vergleich). Die Folienquelle ist
[slides.tex](aufgabe_06_vergleich/slides.tex), die geprüfte PDF liegt daneben;
Tabellen und Grafiken liegen in den gleichnamigen Unterordnern.

Build aus dem Repository-Hauptordner:

```bash
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error \
	-outdir=build praesentation/aufgabe_06_vergleich/slides.tex
```

Die exportierten Task-6-Daten und die Auswertungsbeschreibung bleiben im
jeweiligen Aufgabenordner. Die vollständige Ergebnisdokumentation liegt unter
`results/tables/task6_comparison_100/`, sofern sie in diesem Branch vorhanden
ist.

`aufgabe_03_clustering.tex` erstellt einen eigenständigen Foliensatz zu **Aufgabe 3: Clustering** auf Englisch, ausgelegt auf **7–10 Minuten**. Er enthält acht Hauptfolien und drei Reservefolien. Die fertige [PDF](aufgabe_03_clustering.pdf) kann ohne LaTeX angesehen werden.

**Dateien**

- [aufgabe_03_clustering.tex](aufgabe_03_clustering.tex): Einstiegspunkt; weitere Aufgaben können später hier eingebunden werden.
- [theme.tex](theme.tex): gemeinsames Beamer-Layout im Format 16:9.
- [slides/aufgabe_03.tex](slides/aufgabe_03.tex): editierbare Folieninhalte.
- [sprechernotizen_aufgabe_03.md](sprechernotizen_aufgabe_03.md): Englische Sprechernotizen mit Zeitaufteilung.
- `figures/`: originale Notebook-UMAPs und eine aus den Ergebnissen erstellte Stabilitätsgrafik.
- `data/`: exportierte Ergebnistabellen, generierte LaTeX-Ergebnistabelle und Herkunftsnachweis.
- [export_notebook.py](export_notebook.py): erneuter Export aus gespeicherten Notebook-Ausgaben; führt kein Clustering aus.

**PDF erstellen**

Aus diesem Ordner mit TeX Live, TinyTeX oder MiKTeX:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build aufgabe_03_clustering.tex
```

Danach `build/aufgabe_03_clustering.pdf` als `aufgabe_03_clustering.pdf` in diesen Ordner kopieren, wenn die Version für die gemeinsame Ansicht aktualisiert werden soll. Unter Linux/macOS:

```bash
cp build/aufgabe_03_clustering.pdf aufgabe_03_clustering.pdf
```

Benötigte LaTeX-Pakete sind unter anderem `beamer`, `pgf`, `booktabs`, `babel-english` und `lm`. Es ist kein Shell-Escape erforderlich. Die mitgelieferten Abbildungen und Tabellen genügen zum Kompilieren; die FCS-Rohdaten werden dafür nicht benötigt. Für Overleaf den gesamten Ordner hochladen und `aufgabe_03_clustering.tex` als Hauptdatei wählen (pdfLaTeX).

**Daten und Abbildungen aktualisieren**

Aus dem Repository-Hauptordner in der Python-Projektumgebung:

```bash
python praesentation/export_notebook.py
```

Der Export benötigt `pandas`, `lxml`, `numpy` und `matplotlib`. Er übernimmt Tabellen und UMAP-Bilder aus dem vollständig ausgeführten [Notebook](../notebooks/03_clustering.ipynb). Die numerischen Notebook-Tabellen sind teilweise auf drei Dezimalstellen gerundet; die Stabilitätsgrafik verwendet diese gespeicherten Werte. Die UMAP-Dateien bleiben unverändert, lediglich ihre langen Legenden werden auf Folie 5 durch LaTeX ausgeblendet. Die komplette Ansicht steht im Notebook.

Nach einem neuen Analyselauf müssen auch die Aussagen und manuell gesetzten Zahlen in den Folien und Sprechernotizen geprüft werden. Der Export aktualisiert die Ergebnistabelle, Daten und Abbildungen, nicht den Vortragstext. `data/provenance.json` hält den SHA-256-Hash des exportierten Notebooks fest.

**Inhaltlicher Stand**

Grundlage ist der geprüfte Lauf vom **07.09.2026** mit **20.000 Zellen, 37 Markern, 44 Konfigurationen und 440 Resampling-Konfigurationen**. Details stehen im [Bericht zum Neulauf](../untersuchung/2026-09-07_astra-high_neulauf_gregor_03-clustering_1000-zellen.md).

Die Folien trennen die rechnerische Auswahl von der biologischen Bewertung. „B-cell-like“ und „NK-cell-like“ sind vorläufige Interpretationen relativer Markerprofile, keine validierten Zelltyp-Labels. Eine abschließende biologische Rangfolge, neue Spenderprüfungen oder eine Batch-Korrektur wurden für diese Präsentation nicht berechnet. Die Quellen für Datensatz, biologischen Kontext und Methoden sind auf der letzten Reservefolie verlinkt.

Die PDF, LaTeX-Quellen, `figures/` und `data/` gehören gemeinsam ins Repository. Temporäre Dateien unter `build/` werden ignoriert.
