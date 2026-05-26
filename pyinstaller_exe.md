# PyInstaller — .exe Build

## Ziel

CPU-only `.exe` die auf jedem Windows 10/11 läuft.
Kein Python, kein uv, kein Terminal für den Endnutzer.

CUDA wird **nicht** eingebündelt (zu groß, zu komplex, CUDA-Nutzer nutzen den uv-Weg).
ffmpeg wird **nicht** eingebündelt — wird vom Nutzer selbst installiert (wie bisher).

---

## Was vor dem Build geändert werden muss

### 1. `config.py` — Pfade für schreibbare Dateien

Im PyInstaller-Bundle zeigt `Path(__file__).parent` auf einen Read-only-Temp-Ordner (`_MEIPASS`).
Config, Logs und Output-Ordner müssen stattdessen **neben der .exe** liegen.

```python
# Anfang von config.py ersetzen:
import json
import sys
from pathlib import Path

_FROZEN    = getattr(sys, "frozen", False)
_WRITE_DIR = Path(sys.executable).parent if _FROZEN else Path(__file__).parent
_READ_DIR  = Path(sys._MEIPASS) if _FROZEN else Path(__file__).parent  # type: ignore[attr-defined]

CONFIG_PATH = _WRITE_DIR / "transcript_config.json"
THEMES_PATH = _READ_DIR  / "themes.json"

_DEFAULTS: dict = {
    "default_folder": str(Path.home()),
    "use_default_folder": False,
    "output_folder_yt": str(_WRITE_DIR / "output_yt"),
    "model": "large-v3",
    "ui_scale": "medium",
    "theme": "discord_dark",
}
```

### 2. `main.py` — Log-Ordner

`_BASE_DIR` hat das gleiche Problem. Im Bundle auf `sys.executable` umstellen:

```python
# Zeile ersetzen:
_BASE_DIR = Path(__file__).parent

# Ersetzen durch:
_BASE_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
```

---

## PyInstaller installieren

In der uv-Umgebung (ohne cuda-Extra):

```bash
uv sync
uv pip install pyinstaller
```

---

## Build-Befehl

```bash
uv run pyinstaller `
  --name "Transkript" `
  --onedir `
  --windowed `
  --add-data "themes.json;." `
  --collect-all faster_whisper `
  --collect-all ctranslate2 `
  --hidden-import "faster_whisper" `
  main.py
```

> `--onedir` statt `--onefile`: ctranslate2 ist groß (~150 MB an nativen DLLs).
> Mit `--onefile` würde die App bei jedem Start alles in einen Temp-Ordner entpacken — das dauert 10–20 Sekunden. Mit `--onedir` entsteht ein Ordner mit der .exe + DLLs daneben, Startzeit normal.

---

## Ergebnis

```
dist/
└── Transkript/
    ├── Transkript.exe       ← das ist die Datei die Nutzer starten
    ├── _internal/           ← DLLs, Python, Pakete (nicht anfassen)
    └── ...
```

Der gesamte `Transkript/`-Ordner wird ausgeliefert (z.B. als .zip).

Beim ersten Start durch den Nutzer entstehen automatisch daneben:
```
Transkript/
├── Transkript.exe
├── transcript_config.json   ← wird beim ersten Speichern angelegt
├── logs/                    ← wird beim ersten Start angelegt
└── output_yt/               ← wird beim ersten YouTube-Transkript angelegt
```

---

## Bekannte Stolpersteine

**ctranslate2 / faster-whisper**: Hat native DLLs die PyInstaller nicht automatisch findet.
`--collect-all` für beide ist Pflicht, sonst gibt es beim Start einen Import-Fehler.

**Whisper-Modelle**: Werden beim ersten Start noch runtergeladen (~3 GB für large-v3).
Das passiert automatisch in der App, Nutzer müssen nichts tun — aber sie brauchen Internet beim ersten Mal.

**Fenstermodus**: `--windowed` unterdrückt das schwarze Konsolenfenster.
Wenn etwas nicht startet: `--windowed` temporär weglassen um Fehlerausgabe zu sehen.

**themes.json**: Wird mit `--add-data "themes.json;."` ins Bundle gelegt und über `_READ_DIR` gelesen.
Ohne diesen Flag startet die App ohne Theme (weißes Fenster / Fehler).

---

## Testen vor der Auslieferung

1. `dist/Transkript/Transkript.exe` auf einem **frischen Windows ohne Python** testen
2. Prüfen ob `transcript_config.json` nach dem ersten Einstellungs-Speichern neben der .exe liegt
3. Prüfen ob `logs/` nach dem Start neben der .exe liegt
4. Eine Datei transkribieren, prüfen ob `.txt` neben der Quell-Datei landet
5. Ein YouTube-Video transkribieren, prüfen ob `output_yt/` neben der .exe entsteht
