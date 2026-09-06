#!/usr/bin/env python3
"""FCS-Metadaten prüfen, ohne die Quelldateien zu verändern oder ganz zu laden.

Das Skript verwendet bewusst nur die Python-Standardbibliothek. Es liest die
HEADER- und TEXT-Segmente der FCS-Dateien, prüft die Kanalkonsistenz und führt
die beiden Gate-Stufen über die mitgelieferten Spenderlabels zusammen. Dadurch
eignet es sich als schneller, von der eigentlichen Analyse unabhängiger
Plausibilitätstest für den lokalen Datensatz.
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
from collections import Counter
from pathlib import Path


# Alle Pfade werden relativ zum Skript bestimmt. Das macht den Aufruf unabhängig
# vom aktuellen Arbeitsverzeichnis und vermeidet maschinenspezifische Pfade.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "NK_cell_dataset" / "NK_cell_dataset"
FCS_ROOT = DATA_ROOT / "NK_cell_dataset"
LABELS_PATH = DATA_ROOT / "NK_fcs_samples_with_labels.csv"
MARKERS_PATH = DATA_ROOT / "NK_markers.csv"


def _header_int(header: bytes, start: int, end: int) -> int:
    """Ein rechtsbündiges ASCII-Zahlenfeld aus dem 58-Byte-FCS-Header lesen."""

    # Leere Offsetfelder sind laut FCS-Konvention möglich und werden als 0
    # behandelt; tatsächliche Offsets können dann im TEXT-Segment stehen.
    value = header[start:end].decode("ascii").strip()
    return int(value) if value else 0


def _split_fcs_text(payload: str) -> list[str]:
    """Ein FCS-TEXT-Segment unter Beachtung maskierter Trennzeichen zerlegen.

    Das erste Zeichen definiert das Trennzeichen. Zwei direkt aufeinanderfolgende
    Trennzeichen stehen innerhalb eines Feldes für ein einzelnes Literalzeichen.
    """

    delimiter = payload[0]
    fields: list[str] = []
    field: list[str] = []
    index = 1
    while index < len(payload):
        char = payload[index]
        if char != delimiter:
            field.append(char)
            index += 1
        # Ein verdoppeltes Trennzeichen gehört zum Feldinhalt und beendet das
        # aktuelle Feld daher nicht.
        elif index + 1 < len(payload) and payload[index + 1] == delimiter:
            field.append(delimiter)
            index += 2
        else:
            fields.append("".join(field))
            field = []
            index += 1
    if field:
        fields.append("".join(field))
    return fields


def read_fcs_metadata(path: Path) -> dict[str, object]:
    """Header, Schlüsselwörter und Kanaldefinitionen einer FCS-Datei lesen.

    Es wird ausschließlich bis zum Ende des TEXT-Segments gelesen. Der große
    binäre Messdatenblock bleibt unangetastet, weshalb diese Prüfung auch für
    alle 40 Dateien schnell und speichersparend ist.
    """

    with path.open("rb") as stream:
        header = stream.read(58)
        if not header.startswith(b"FCS"):
            raise ValueError(f"Not an FCS file: {path}")
        text_start = _header_int(header, 10, 18)
        text_end = _header_int(header, 18, 26)
        stream.seek(text_start)
        payload = stream.read(text_end - text_start + 1).decode("latin-1")

    # Der TEXT-Bereich ist eine alternierende Folge aus Schlüssel und Wert.
    fields = _split_fcs_text(payload)
    if len(fields) % 2:
        # Manche Exporte enden mit einem nicht gepaarten leeren Feld. Es trägt
        # keine Information und wird entfernt, bevor Schlüssel/Werte gepaart werden.
        fields = fields[:-1]
    text = dict(zip(fields[::2], fields[1::2]))

    parameter_count = int(text["$PAR"])
    channels = []
    for number in range(1, parameter_count + 1):
        # PnS bezeichnet hier den biologischen Markernamen, PnN den technischen
        # Detektorkanal. Beide werden benötigt, um Dateilayouts sicher zu vergleichen.
        channels.append(
            {
                "number": number,
                "short_name": text.get(f"$P{number}S", ""),
                "detector_name": text.get(f"$P{number}N", ""),
                "bits": int(text.get(f"$P{number}B", "0")),
                "range": int(text.get(f"$P{number}R", "0")),
            }
        )

    return {
        "version": header[:6].decode("ascii"),
        "events": int(text["$TOT"]),
        "parameters": parameter_count,
        "data_type": text.get("$DATATYPE", ""),
        "byte_order": text.get("$BYTEORD", ""),
        "instrument": text.get("$CYT", ""),
        "date": text.get("$DATE", ""),
        "start_time": text.get("$BTIM", ""),
        "end_time": text.get("$ETIM", ""),
        "source_name": text.get("$FIL", ""),
        "comment": text.get("$COM", ""),
        "data_start": int(text["$BEGINDATA"]),
        "data_end": int(text["$ENDDATA"]),
        "channels": channels,
    }


def preview_events(path: Path, row_count: int) -> None:
    """Die ersten Messereignisse einer unterstützten FCS-Datei als CSV ausgeben.

    Diese Vorschau liest nur ``row_count`` Zeilen aus dem DATA-Segment. Sie ist
    absichtlich auf das im Projektdatensatz verwendete Float-/Byteformat begrenzt.
    """

    metadata = read_fcs_metadata(path)
    if metadata["data_type"] != "F" or metadata["byte_order"] != "4,3,2,1":
        raise ValueError(
            "Preview currently supports the dataset's big-endian 32-bit floats only"
        )
    parameter_count = int(metadata["parameters"])
    # Datentyp F bedeutet 32-Bit-Float, also vier Byte je Kanal und Ereignis.
    row_size = parameter_count * 4
    names = [str(channel["short_name"]) for channel in metadata["channels"]]
    print(",".join(names))
    with path.open("rb") as stream:
        stream.seek(int(metadata["data_start"]))
        for _ in range(min(row_count, int(metadata["events"]))):
            row = stream.read(row_size)
            # ``>`` erzwingt die im Datensatz deklarierte Big-Endian-Bytefolge.
            values = struct.unpack(f">{parameter_count}f", row)
            print(",".join(f"{value:.7g}" for value in values))


def sample_id(path: Path) -> str:
    """Die gemeinsame Spender-ID aus Alive- oder NK-Dateinamen ableiten."""

    return re.sub(r"_(?:NK|alive)\.fcs$", "", path.name)


def load_labels() -> dict[str, int]:
    """CMV-Label als Zuordnung ``Spender-ID -> 0/1`` laden."""

    with LABELS_PATH.open(newline="", encoding="utf-8-sig") as stream:
        return {
            sample_id(Path(row["fcs_filename"])): int(row["label"])
            for row in csv.DictReader(stream)
        }


def load_analysis_markers() -> list[str]:
    """Die festgelegte Reihenfolge der 37 biologischen Marker laden."""

    with MARKERS_PATH.open(newline="", encoding="utf-8-sig") as stream:
        return next(csv.reader(stream))


def main() -> None:
    """Kommandozeile auswerten und Vorschau oder vollständige Metadaten-QC starten."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channels",
        action="store_true",
        help="also print the full channel table",
    )
    parser.add_argument(
        "--preview",
        metavar="FILENAME",
        help="print the first event rows of one real FCS file as CSV",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="number of rows printed with --preview (default: 5)",
    )
    args = parser.parse_args()

    if args.preview:
        # Der Dateiname muss über beide Gate-Ordner eindeutig sein, damit nicht
        # versehentlich die falsche Population angezeigt wird.
        matches = list(FCS_ROOT.glob(f"gated_*/{args.preview}"))
        if len(matches) != 1:
            parser.error(f"expected one matching real FCS file, found {len(matches)}")
        preview_events(matches[0], args.preview_rows)
        return

    # Ab hier läuft die Standard-QC über beide Gate-Stufen. Die Originaldateien
    # werden dabei ausschließlich binär zum Lesen geöffnet.
    labels = load_labels()
    analysis_markers = load_analysis_markers()
    fcs_paths = sorted(FCS_ROOT.glob("gated_*/*.fcs"))
    records = [(path, read_fcs_metadata(path)) for path in fcs_paths]

    # Ein Set vollständiger Layout-Tupel deckt jede Abweichung in Reihenfolge,
    # biologischem Namen oder Detektornamen zwischen den Dateien auf.
    channel_layouts = {
        tuple(
            (channel["short_name"], channel["detector_name"])
            for channel in metadata["channels"]
        )
        for _, metadata in records
    }
    versions = Counter(metadata["version"] for _, metadata in records)
    data_types = Counter(metadata["data_type"] for _, metadata in records)
    byte_orders = Counter(metadata["byte_order"] for _, metadata in records)

    # Getrennte Zuordnungen erlauben anschließend den direkten Vergleich der
    # Ereigniszahl vor und nach dem zusätzlichen NK-Gating pro Spender.
    alive = {
        sample_id(path): int(metadata["events"])
        for path, metadata in records
        if path.parent.name == "gated_alive"
    }
    nk = {
        sample_id(path): int(metadata["events"])
        for path, metadata in records
        if path.parent.name == "gated_NK"
    }

    print("Dataset overview")
    print(f"  Real FCS files: {len(records)}")
    print(f"  Donors/samples: {len(labels)}")
    print(f"  Label counts: {dict(sorted(Counter(labels.values()).items()))}")
    print(f"  FCS versions: {dict(versions)}")
    print(f"  Data types: {dict(data_types)}")
    print(f"  Byte orders: {dict(byte_orders)}")
    print(f"  Distinct channel layouts: {len(channel_layouts)}")
    print(f"  Analysis markers: {len(analysis_markers)}")
    print(f"  Alive events: {sum(alive.values()):,}")
    print(f"  NK events: {sum(nk.values()):,}")
    print()
    print("Per-sample event counts")
    print("sample,label,alive_events,nk_events,nk_percent_of_alive")
    for identifier in sorted(labels):
        # Der Anteil beschreibt nur den Effekt des vorhandenen Gates; es werden
        # weder Zellen neu ausgewählt noch FCS-Dateien verändert.
        percentage = 100 * nk[identifier] / alive[identifier]
        print(
            f"{identifier},{labels[identifier]},{alive[identifier]},"
            f"{nk[identifier]},{percentage:.3f}"
        )

    if args.channels:
        # Das Kanallayout ist zuvor als identisch geprüft worden. Deshalb genügt
        # für die ausführliche Tabelle die erste Datei als Repräsentant.
        first_channels = records[0][1]["channels"]
        analysis_marker_set = set(analysis_markers)
        print()
        print("Channels")
        print("number,short_name,detector_name,bits,range,analysis_marker")
        for channel in first_channels:
            marker = channel["short_name"]
            print(
                f'{channel["number"]},{marker},{channel["detector_name"]},'
                f'{channel["bits"]},{channel["range"]},'
                f'{"yes" if marker in analysis_marker_set else "no"}'
            )


if __name__ == "__main__":
    main()
