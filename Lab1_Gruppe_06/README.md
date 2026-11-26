# Labor 1 – Gruppe 06
Julius Weber, Jannis Hammerer, Matthis Winter

1. ZIP vollständig entpacken.
2. Python-Abhängigkeiten installieren: `pip install -r requirements.txt`.
3. `MainCode1.ipynb` im entpackten Ordner öffnen und alle Zellen ausführen.
4. Die Auswertung legt Plots und Tabellen in `ergebnisse/` ab.

Der Bericht verwendet die vorhandene MCI-Vorlage in einer PDF-kompatiblen Fassung.
Zur Neukompilierung im Ordner `bericht/` zweimal `pdflatex Bericht_1_Gruppe_06.tex` ausführen.
Das Archiv enthält beide Arduino-Sketches in ihren jeweiligen Sketch-Ordnern.

GitHub-Datenbasis: 0e878e9c892acd65b4d91fb24b0a2f1dd0da02fe.
Die Daten zu Aufgabe 9 enthalten zwei Zeitrücksprünge. Das Notebook wertet
die ersten zehn Sekunden des letzten zusammenhängenden Abschnitts aus.

Alternativ erzeugt `python main.py` dieselben Plots und Kennwerte.
Mit den installierten Abhängigkeiten nutzt die Auswertung SciPy und Matplotlib.
Die Diagramme entstehen über die direkten DataFrame-Aufrufe von Pandas und Matplotlib.
