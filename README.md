# Python-Programme zur CBCT-Dosimetrie

Begleitmaterial zur Bachelorarbeit von Marcello Paul Mondadori,
Physik B.Sc., Ludwig-Maximilians-Universität München, 2026.

| Programm | Inhalt |
|---|---|
| `programme/spektren_hvl_xcom.py` | Spektrumsimulation, Halbwertschichtdicke, effektive Energie und Wechselwirkungsanteile in PMMA |
| `programme/energiekorrekturfaktoren.py` | Bestimmung der Energiekorrekturfaktoren für TLD und Farmer-Kammer |
| `programme/kollimator_kreisfit.py` | Bestimmung des Kollimatormittelpunkts mit einem Kreisfit |
| `programme/dosisprofile.py` | Auswertung der TLD- und Farmer-Kammer-Messungen und Darstellung der Dosisprofile |

## Ausführen

Python 3 verwenden. Im Projektordner die benötigten Pakete installieren:

```bash
python -m pip install -r requirements.txt
```

Danach das gewünschte Programm starten, beispielsweise:

```bash
python programme/dosisprofile.py
```

Die Spektrumsimulation liest `daten/65kVp_measurement_interp.mat` und
`daten/xcom_pmma.txt`. Die zweite Spektrumsdatei
`daten/65kVp_filter_measurement_interp.mat` dokumentiert den weiteren
Datensatz aus dem Anhang. Der Kreisfit benötigt
`daten/beamradius_TE.EVI`. Fehlende Eingabedateien müssen ergänzt werden.
Die Dosiswerte und die Tabellenwerte für die Korrekturfaktoren stehen
direkt in den entsprechenden Programmen. Das Programm für die
Korrekturfaktoren fragt die effektive Energie und die Al-Halbwertschichtdicke
im Terminal ab.

Für die Bachelorarbeit ist der unter `v1.0` veröffentlichte Stand maßgeblich.
Fachliche Grundlagen und die Einordnung der Ergebnisse stehen in der Arbeit.
