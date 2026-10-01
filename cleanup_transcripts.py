"""
cleanup_transcripts.py — Räumt output_transcripts/ auf

Jedes Transkript entsteht als Paar:
    name.txt              (reiner Text, ohne Zeitstempel)
    name_timestamps.txt   (mit Zeitstempeln)

Dieses Script löscht rekursiv alle "name.txt", für die ein passendes
"name_timestamps.txt" im selben Ordner existiert — übrig bleiben nur
noch die Versionen mit Zeitstempeln.

Standardmäßig nur Dry-Run (zeigt an, was gelöscht würde).
Mit --run wird tatsächlich gelöscht.

Nutzung:
    python cleanup_transcripts.py                       # Dry-Run
    python cleanup_transcripts.py --run                  # löscht wirklich
    python cleanup_transcripts.py --folder output_transcripts --run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TIMESTAMP_SUFFIX = "_timestamps"


def find_plain_txts_with_timestamp_sibling(root: Path) -> list[Path]:
    """Findet alle *.txt-Dateien, deren *_timestamps.txt-Gegenstück existiert."""
    to_delete = []
    for txt in root.rglob("*.txt"):
        if txt.stem.endswith(TIMESTAMP_SUFFIX):
            continue
        sibling = txt.with_name(f"{txt.stem}{TIMESTAMP_SUFFIX}.txt")
        if sibling.exists():
            to_delete.append(txt)
    return sorted(to_delete)


def main() -> int:
    p = argparse.ArgumentParser(description="Löscht die Nicht-Timestamp-Transkripte.")
    p.add_argument("--folder", "-f", type=Path, default=Path("output_transcripts"),
                   help="Wurzelordner (Standard: output_transcripts)")
    p.add_argument("--run", action="store_true",
                   help="Tatsächlich löschen (ohne diese Flag nur Anzeige)")
    args = p.parse_args()

    if not args.folder.exists():
        print(f"[ERR]  Ordner nicht gefunden: {args.folder}")
        return 1

    candidates = find_plain_txts_with_timestamp_sibling(args.folder)

    if not candidates:
        print("[OK]   Nichts zu tun — keine Datei-Paare gefunden.")
        return 0

    for f in candidates:
        print(f"[DEL]  {f.relative_to(args.folder)}")

    print(f"\n{len(candidates)} Datei(en) gefunden.")

    if not args.run:
        print("Dry-Run — nichts gelöscht. Mit --run tatsächlich löschen.")
        return 0

    for f in candidates:
        f.unlink()

    print(f"[OK]   {len(candidates)} Datei(en) gelöscht.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
