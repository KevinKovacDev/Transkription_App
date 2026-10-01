"""
transcriber_isolated.py — Standalone-Transkriptions-Modul

Einzige externe Abhängigkeit: faster-whisper  (pip install faster-whisper)
System-Tools:                  ffmpeg, optional nvidia-smi + ctranslate2 (CUDA)

Nutzung als Bibliothek — Ordner-Modus (gespiegelte Struktur):
    from transcriber_isolated import transcribe_folder

    transcribe_folder(
        input_root=Path("input_videos"),
        output_root=Path("output_transcripts"),
        model="large-v3",
    )

Nutzung als Bibliothek — Datei-Liste:
    from transcriber_isolated import transcribe_videos

    transcribe_videos(
        videos=[Path("video1.mp4"), Path("video2.mp4")],
        output_dir=Path("output"),
    )

Nutzung als CLI — Ordner:
    python transcriber_isolated.py --folder input_videos --output output_transcripts

Nutzung als CLI — Einzeldateien:
    python transcriber_isolated.py video1.mp4 video2.mp4 --output ./out --model medium
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Windows-Konsole auf UTF-8 setzen (verhindert Encoding-Fehler bei Umlauten/Sonderzeichen)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv"}


# ---------------------------------------------------------------------------
# Inline-Logger
# ---------------------------------------------------------------------------

class _Log:
    def info(self, msg: str)    -> None: print(f"[INFO]  {msg}", flush=True)
    def success(self, msg: str) -> None: print(f"[OK]    {msg}", flush=True)
    def error(self, msg: str)   -> None: print(f"[ERR]   {msg}", flush=True)
    def warn(self, msg: str)    -> None: print(f"[WARN]  {msg}", flush=True)
    def raw(self, msg: str)     -> None: print(msg, flush=True)

    def progress(self, current: int, total: int, label: str = "") -> None:
        bar_width = 30
        filled = int(bar_width * current / total) if total else 0
        bar = "#" * filled + "-" * (bar_width - filled)
        suffix = f"  {label}" if label else ""
        print(f"[{current:>{len(str(total))}}/{total}] [{bar}]{suffix}", flush=True)


_log = _Log()


# ---------------------------------------------------------------------------
# System-Checks
# ---------------------------------------------------------------------------

def cuda_available() -> bool:
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if r.returncode != 0:
            return False
    except Exception:
        return False
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def ffmpeg_available() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# DLL-Pfade für Windows + CUDA
# ---------------------------------------------------------------------------

def _register_cuda_dlls() -> None:
    if sys.platform != "win32":
        return
    for sp in sys.path:
        for pkg in ("cublas", "cudnn", "cuda_runtime", "cufft", "cusparse", "curand"):
            dll_dir = Path(sp) / "nvidia" / pkg / "bin"
            if dll_dir.exists():
                os.add_dll_directory(str(dll_dir))
                os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ["PATH"]


# ---------------------------------------------------------------------------
# Datentypen
# ---------------------------------------------------------------------------

@dataclass
class TranscriptResult:
    text: str
    language: str
    segments: list[dict]
    processing_seconds: float
    media_seconds: float = 0.0
    source: str = ""


# ---------------------------------------------------------------------------
# Video-Erkennung
# ---------------------------------------------------------------------------

def discover_videos(root: Path) -> list[Path]:
    """Findet alle Video-Dateien rekursiv, sortiert nach relativem Pfad."""
    return sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------

class Transcriber:
    """
    Whisper-Wrapper mit "einmal laden, viele füttern".

    t = Transcriber(use_cuda=True, model="large-v3")
    t.init()                               # Whisper einmalig laden
    result = t.transcribe(Path("x.mp4"))   # Video transkribieren
    result = t.transcribe_audio(wav_path)  # bereits extrahiertes WAV
    t.unload()                             # Referenzen freigeben
    """

    def __init__(self, use_cuda: bool = False, model: str = "large-v3"):
        self._use_cuda = use_cuda
        self._model_name = model
        self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def init(self) -> None:
        if self._model is not None:
            _log.info("Whisper bereits geladen — überspringe.")
            return

        _register_cuda_dlls()

        from faster_whisper import WhisperModel  # type: ignore

        device  = "cuda" if self._use_cuda else "cpu"
        compute = "float16" if self._use_cuda else "int8"

        _log.info(f"Lade Whisper {self._model_name} auf {device.upper()}...")
        self._model = WhisperModel(self._model_name, device=device, compute_type=compute)
        _log.success("Whisper geladen.")

        if self._model.model.device.startswith("cuda"):
            _log.success("GPU-Beschleunigung aktiv.")
        else:
            _log.info("GPU-Beschleunigung inaktiv (CPU).")

    def unload(self) -> None:
        if self._model is None:
            return
        try:
            del self._model
        finally:
            self._model = None
        gc.collect()
        _log.info("Modell-Referenz freigegeben.")

    # ------------------------------------------------------------------

    def transcribe(
        self,
        video_path: Path,
        on_progress: Callable[[int], None] | None = None,
        language: str = "de",
    ) -> TranscriptResult:
        if not self.ready:
            raise RuntimeError("Transcriber nicht initialisiert — init() aufrufen.")

        _log.raw(f"\n{'=' * 60}")
        _log.info(f"Datei:  {video_path.name}")
        _log.info(f"Größe:  {video_path.stat().st_size / (1024 * 1024):.1f} MB")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = Path(tmp.name)

        try:
            self._extract_audio(video_path, audio_path)
            result = self._run_whisper(audio_path, on_progress, language=language)
            result.source = str(video_path)
            return result
        finally:
            if audio_path.exists():
                audio_path.unlink()

    def transcribe_audio(
        self,
        audio_path: Path,
        label: str = "",
        on_progress: Callable[[int], None] | None = None,
        language: str = "de",
    ) -> TranscriptResult:
        if not self.ready:
            raise RuntimeError("Transcriber nicht initialisiert — init() aufrufen.")

        _log.raw(f"\n{'=' * 60}")
        if label:
            _log.info(f"Verarbeite: {label}")
        _log.info(f"Größe: {audio_path.stat().st_size / (1024 * 1024):.1f} MB")

        result = self._run_whisper(audio_path, on_progress, language=language)
        result.source = label or str(audio_path)
        return result

    # ------------------------------------------------------------------

    def _extract_audio(self, video_path: Path, audio_path: Path) -> None:
        _log.info("ffmpeg: Extrahiere Audio...")
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg Fehler:\n{r.stderr}")
        _log.success(f"Audio extrahiert: {audio_path.stat().st_size / (1024 * 1024):.1f} MB")

    def _run_whisper(
        self,
        audio_path: Path,
        on_progress: Callable[[int], None] | None = None,
        language: str = "de",
    ) -> TranscriptResult:
        _log.info("Whisper: Starte Transkription...")
        t_start = time.time()

        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
        )

        _log.info(f"Sprache: {info.language} ({info.language_probability:.0%})")
        total = float(getattr(info, "duration", 0.0) or 0.0)

        segments: list[dict] = []
        for seg in segments_iter:
            segments.append({
                "start": round(seg.start, 2),
                "end":   round(seg.end,   2),
                "text":  seg.text.strip(),
            })
            if on_progress and total > 0:
                on_progress(min(100, int(seg.end / total * 100)))

        elapsed = time.time() - t_start
        _log.success(f"{len(segments)} Segmente in {elapsed:.1f}s")

        return TranscriptResult(
            text=" ".join(s["text"] for s in segments),
            language=info.language,
            segments=segments,
            processing_seconds=elapsed,
            media_seconds=total,
        )

    @staticmethod
    def to_text(segments: list[dict]) -> str:
        def ts(s: float) -> str:
            t = int(s)
            m, sec = (t % 3600) // 60, t % 60
            return f"{m:02d}:{sec:02d}" if t < 3600 else f"{t // 3600}:{m:02d}:{sec:02d}"
        return "\n".join(f"[{ts(seg['start'])}] {seg['text']}" for seg in segments)


# ---------------------------------------------------------------------------
# Ausgabepfad berechnen
# ---------------------------------------------------------------------------

def _output_path(video: Path, output_root: Path, input_root: Path | None) -> Path:
    """Gibt den Zielordner für ein Video zurück, gespiegelt aus input_root."""
    if input_root is not None:
        try:
            rel = video.relative_to(input_root)
            return output_root / rel.parent
        except ValueError:
            pass
    return output_root


# ---------------------------------------------------------------------------
# Haupt-API: Ordner transkribieren (gespiegelte Struktur)
# ---------------------------------------------------------------------------

def transcribe_folder(
    input_root: Path,
    output_root: Path,
    model: str = "large-v3",
    use_cuda: bool | None = None,
    language: str = "de",
    skip_existing: bool = True,
    include_plain_text: bool = False,
) -> list[TranscriptResult]:
    """
    Scannt input_root rekursiv nach Videos, transkribiert alle und
    legt die .txt-Dateien in output_root mit identischer Ordnerstruktur ab.

    Args:
        input_root:         Wurzelordner mit Videos (z.B. input_videos/).
        output_root:        Zielordner (z.B. output_transcripts/).
        model:              Whisper-Modell.
        use_cuda:           True/False/None (None = automatisch).
        language:           Sprach-Code für Whisper.
        skip_existing:      Überspringt Videos deren Transkript bereits existiert.
        include_plain_text: Zusätzlich zur _timestamps.txt auch die reine
                             .txt ohne Zeitstempel erzeugen (Standard: nur Timestamps).

    Returns:
        Liste aller TranscriptResult (fehlgeschlagene haben text="").
    """
    input_root  = Path(input_root)
    output_root = Path(output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Eingabeordner nicht gefunden: {input_root}")

    # --- Videos zählen ---
    _log.raw(f"\n{'=' * 60}")
    _log.info(f"Scanne: {input_root}")
    all_videos = discover_videos(input_root)

    if not all_videos:
        _log.warn("Keine Video-Dateien gefunden.")
        return []

    _log.success(f"{len(all_videos)} Videos gefunden.")

    # --- bereits erledigte überspringen ---
    if skip_existing:
        pending = []
        skipped = 0
        for v in all_videos:
            dest_dir = _output_path(v, output_root, input_root)
            if (dest_dir / f"{v.stem}_timestamps.txt").exists():
                skipped += 1
            else:
                pending.append(v)
        if skipped:
            _log.info(f"{skipped} bereits transkribiert — übersprungen.")
    else:
        pending = all_videos

    if not pending:
        _log.success("Alle Videos bereits transkribiert.")
        return []

    _log.info(f"{len(pending)} Videos werden transkribiert.")
    _log.raw(f"{'=' * 60}\n")

    return transcribe_videos(
        videos=pending,
        output_root=output_root,
        input_root=input_root,
        model=model,
        use_cuda=use_cuda,
        language=language,
        include_plain_text=include_plain_text,
    )


# ---------------------------------------------------------------------------
# Haupt-API: Video-Liste transkribieren
# ---------------------------------------------------------------------------

def transcribe_videos(
    videos: list[Path],
    output_root: Path | None = None,
    input_root: Path | None = None,
    output_dir: Path | None = None,       # Fallback: alle in einen Ordner
    model: str = "large-v3",
    use_cuda: bool | None = None,
    language: str = "de",
    on_progress: Callable[[int, int, str], None] | None = None,
    include_plain_text: bool = False,
) -> list[TranscriptResult]:
    """
    Transkribiert eine Liste von Video-Dateien.

    Struktur-Modus (empfohlen):
        output_root + input_root → gespiegelte Ordnerstruktur

    Flach-Modus:
        output_dir → alle .txt in einem Ordner

    Args:
        videos:             Pfade zu den Video-Dateien.
        output_root:        Wurzel des Ausgabeordners (bei gespiegelter Struktur).
        input_root:         Wurzel des Eingabeordners (für relative Pfadberechnung).
        output_dir:         Einzelner Zielordner (Fallback wenn kein output_root).
        model:              Whisper-Modell.
        use_cuda:           True/False/None (None = automatisch).
        language:           Sprach-Code für Whisper.
        on_progress:        Callback(video_index, prozent, dateiname).
        include_plain_text: Zusätzlich zur _timestamps.txt auch die reine
                             .txt ohne Zeitstempel erzeugen (Standard: nur Timestamps).

    Returns:
        Liste von TranscriptResult, gleiche Reihenfolge wie `videos`.
    """
    if not videos:
        _log.warn("Keine Videos übergeben.")
        return []

    if not ffmpeg_available():
        _log.error("ffmpeg nicht gefunden. Bitte installieren und PATH setzen.")
        raise RuntimeError("ffmpeg nicht gefunden.")

    if use_cuda is None:
        use_cuda = cuda_available()
        _log.info(f"CUDA: {'erkannt -> GPU' if use_cuda else 'nicht verfuegbar -> CPU'}")

    t = Transcriber(use_cuda=use_cuda, model=model)
    t.init()

    total = len(videos)
    results: list[TranscriptResult] = []

    for idx, video in enumerate(videos, 1):
        video = Path(video)

        _log.raw("")
        _log.progress(idx, total, video.name)

        def _prog(pct: int, _i=idx, _name=video.name) -> None:
            if on_progress:
                on_progress(_i, pct, _name)

        try:
            result = t.transcribe(video, on_progress=_prog, language=language)
        except Exception as exc:
            _log.error(f"Fehler: {exc}")
            results.append(TranscriptResult(
                text="", language="", segments=[],
                processing_seconds=0.0, media_seconds=0.0, source=str(video),
            ))
            continue

        # --- Zielordner bestimmen ---
        if output_root is not None:
            dest = _output_path(video, output_root, input_root)
        elif output_dir is not None:
            dest = output_dir
        else:
            dest = video.parent

        dest.mkdir(parents=True, exist_ok=True)

        out_ts = dest / f"{video.stem}_timestamps.txt"
        out_ts.write_text(Transcriber.to_text(result.segments), encoding="utf-8")

        if include_plain_text:
            out_plain = dest / f"{video.stem}.txt"
            out_plain.write_text(result.text, encoding="utf-8")

        _log.success(f"Gespeichert → {out_ts.relative_to(output_root) if output_root else out_ts.name}")
        results.append(result)

    _log.raw(f"\n{'=' * 60}")
    ok = sum(1 for r in results if r.text)
    _log.success(f"Fertig: {ok}/{total} Videos erfolgreich transkribiert.")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transkribiert Videos mit Whisper (gespiegelte Ordnerstruktur).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Ordner-Modus (empfohlen):
  python transcriber_isolated.py --folder input_videos --output output_transcripts

  # Einzeldateien:
  python transcriber_isolated.py video1.mp4 video2.mp4 --output ./texte

  # Optionen:
  python transcriber_isolated.py --folder ./videos --output ./out --model medium --language en
        """,
    )

    p.add_argument("videos", nargs="*", type=Path,
                   help="Einzelne Video-Dateien (alternativ zu --folder)")
    p.add_argument("--folder", "-f", type=Path, default=None,
                   help="Eingabeordner (rekursiver Scan)")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Ausgabeordner")
    p.add_argument("--model", "-m", default="large-v3",
                   choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                   help="Whisper-Modell (Standard: large-v3)")
    p.add_argument("--gpu", action="store_true",
                   help="GPU erzwingen")
    p.add_argument("--cpu", action="store_true",
                   help="CPU erzwingen")
    p.add_argument("--language", "-l", default="de",
                   help="Sprach-Code für Whisper (Standard: de)")
    p.add_argument("--no-skip", action="store_true",
                   help="Bereits transkribierte Videos nicht überspringen")
    p.add_argument("--with-plain-text", action="store_true",
                   help="Zusätzlich zur _timestamps.txt auch die reine .txt ohne Zeitstempel erzeugen")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.cpu:
        use_cuda = False
    elif args.gpu:
        use_cuda = True
    else:
        use_cuda = None

    # --- Ordner-Modus ---
    if args.folder:
        if args.output is None:
            _log.error("--output ist erforderlich wenn --folder verwendet wird.")
            return 1
        transcribe_folder(
            input_root=args.folder,
            output_root=args.output,
            model=args.model,
            use_cuda=use_cuda,
            language=args.language,
            skip_existing=not args.no_skip,
            include_plain_text=args.with_plain_text,
        )
        return 0

    # --- Datei-Liste-Modus ---
    if not args.videos:
        _log.error("Keine Videos angegeben. Nutze --folder oder liste Dateien auf.")
        return 1

    missing = [v for v in args.videos if not v.exists()]
    if missing:
        for m in missing:
            _log.error(f"Datei nicht gefunden: {m}")
        return 1

    results = transcribe_videos(
        videos=list(args.videos),
        output_dir=args.output,
        model=args.model,
        use_cuda=use_cuda,
        language=args.language,
        include_plain_text=args.with_plain_text,
    )
    return 0 if any(r.text for r in results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _log.warn("Abgebrochen.")
        sys.exit(130)
