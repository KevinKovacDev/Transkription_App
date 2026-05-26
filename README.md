# Transkript

## Installation

### 1. uv installieren

uv verwaltet Python und alle Pakete. Einmal installieren, danach nie wieder anfassen.

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Danach Terminal neu starten (damit `uv` im PATH ist).

---

### 2. ffmpeg installieren

ffmpeg extrahiert den Ton aus Videos. Ohne ffmpeg startet die App nicht.

**Windows:**
```powershell
winget install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt install ffmpeg
```

Terminal nach der Installation neu starten.

> Prüfen ob es geklappt hat: `ffmpeg -version` im Terminal eingeben — es sollte eine Versionsnummer erscheinen.

---

### 3. Abhängigkeiten installieren

Im Projektordner (dort wo `pyproject.toml` liegt):

```bash
uv sync
```

Das installiert Python 3.14 (falls nicht vorhanden) und alle Pakete automatisch — inklusive yt-dlp für YouTube. Beim ersten Mal kann das einige Minuten dauern.

---

### 4. App starten

```bash
uv run python main.py
```

Beim allerersten Start lädt Whisper das gewählte Modell herunter (Large v3 ≈ 3 GB). Das passiert nur einmal und wird danach gecacht.

---

### Tipp: Startskript anlegen (Windows)

Einmal eine `start.bat` im Projektordner anlegen:

```bat
@echo off
cd /d "%~dp0"
uv run python main.py
```

Danach reicht ein Doppelklick.

---

### CUDA (optional, empfohlen bei NVIDIA-GPU)

CUDA beschleunigt die Transkription deutlich — Large v3 läuft damit ca. 5–10× schneller als auf der CPU. Die App erkennt automatisch ob CUDA verfügbar ist (grüner/roter Punkt unten links in der App).

**Windows:**
```powershell
winget install cuda
```

**Linux:** [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

CUDA nach der Installation neu starten und dann `uv sync` nochmal ausführen.
