**Präsentation zum SSBI-Gruppenprojekt**

`aufgabe_03_clustering.tex` erstellt einen eigenständigen Foliensatz zu **Aufgabe 3: Clustering** auf Deutsch, ausgelegt auf **7–10 Minuten**. Er enthält acht Hauptfolien und drei Reservefolien. Die fertige [PDF](aufgabe_03_clustering.pdf) kann ohne LaTeX angesehen werden.

**Dateien**

- [aufgabe_03_clustering.tex](aufgabe_03_clustering.tex): Einstiegspunkt; weitere Aufgaben können später hier eingebunden werden.
- [theme.tex](theme.tex): gemeinsames Beamer-Layout im Format 16:9.
- [slides/aufgabe_03.tex](slides/aufgabe_03.tex): editierbare Folieninhalte.
- [sprechernotizen_aufgabe_03.md](sprechernotizen_aufgabe_03.md): Vorschlag für Vortrag und Zeitaufteilung.
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

Benötigte LaTeX-Pakete sind unter anderem `beamer`, `pgf`, `booktabs`, `babel-german` und `lm`. Es ist kein Shell-Escape erforderlich. Die mitgelieferten Abbildungen und Tabellen genügen zum Kompilieren; die FCS-Rohdaten werden dafür nicht benötigt. Für Overleaf den gesamten Ordner hochladen und `aufgabe_03_clustering.tex` als Hauptdatei wählen (pdfLaTeX).

**Daten und Abbildungen aktualisieren**

Aus dem Repository-Hauptordner in der Python-Projektumgebung:

```bash
python praesentation/export_notebook.py
```

Der Export benötigt `pandas`, `lxml`, `numpy` und `matplotlib`. Er übernimmt Tabellen und UMAP-Bilder aus dem vollständig ausgeführten [Notebook](../notebooks/03_clustering.ipynb). Die numerischen Notebook-Tabellen sind teilweise auf drei Dezimalstellen gerundet; die Stabilitätsgrafik verwendet diese gespeicherten Werte. Die UMAP-Dateien bleiben unverändert, lediglich ihre langen Legenden werden auf Folie 5 durch LaTeX ausgeblendet. Die komplette Ansicht steht im Notebook.

Nach einem neuen Analyselauf müssen auch die Aussagen und manuell gesetzten Zahlen in den Folien und Sprechernotizen geprüft werden. Der Export aktualisiert die Ergebnistabelle, Daten und Abbildungen, nicht den Vortragstext. `data/provenance.json` hält den SHA-256-Hash des exportierten Notebooks fest.

**Inhaltlicher Stand**

Grundlage ist der geprüfte Lauf vom **07.09.2026** mit **20.000 Zellen, 37 Markern, 44 Konfigurationen und 440 Resampling-Konfigurationen**. Details stehen im [Bericht zum Neulauf](../untersuchung/2026-09-07_astra-high_neulauf_gregor_03-clustering_1000-zellen.md).

Die Folien trennen die rechnerische Auswahl von der biologischen Bewertung. „B-Zell-artig“ und „NK-Zell-artig“ sind vorläufige Interpretationen relativer Markerprofile, keine validierten Zelltyp-Labels. Eine abschließende biologische Rangfolge, neue Spenderprüfungen oder eine Batch-Korrektur wurden für diese Präsentation nicht berechnet. Die Quellen für Datensatz, biologischen Kontext und Methoden sind auf der letzten Reservefolie verlinkt.

Die PDF, LaTeX-Quellen, `figures/` und `data/` gehören gemeinsam ins Repository. Temporäre Dateien unter `build/` werden ignoriert.
