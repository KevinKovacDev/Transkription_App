"""
Transcriber-Modul: Whisper Singleton + Transkriptions-Logik.

Whisper wird einmal beim Aufruf von `init()` geladen und dann
für alle folgenden Transkriptionen wiederverwendet.

Nutzung:
    from transcriber import Transcriber
    from logger import Logger

    log = Logger()
    t = Transcriber(logger=log, use_cuda=True)
    t.init()  # Lädt Whisper — nur einmal nötig

    result = t.transcribe(Path("video.mp4"))
    print(result.text)
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from logger import Logger


@dataclass
class TranscriptResult:
    text: str
    language: str
    segments: list[dict]
    duration_seconds: float


class Transcriber:
    def __init__(self, logger: Logger, use_cuda: bool = False, model: str = "large-v3"):
        self._log = logger
        self._use_cuda = use_cuda
        self._model_name = model
        self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def init(self) -> None:
        """Lädt das Whisper-Modell. Nur einmal aufrufen."""
        if self._model is not None:
            self._log.info("Whisper bereits geladen — überspringe.")
            return

        if sys.platform == "win32":
            for sp in sys.path:
                for pkg in ("cublas", "cudnn", "cuda_runtime", "cufft", "cusparse", "curand"):
                    dll_dir = Path(sp) / "nvidia" / pkg / "bin"
                    if dll_dir.exists():
                        os.add_dll_directory(str(dll_dir))
                        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ["PATH"]

        from faster_whisper import WhisperModel

        device = "cuda" if self._use_cuda else "cpu"
        compute = "float16" if self._use_cuda else "int8"

        self._log.info(f"Lade Whisper {self._model_name} auf {device.upper()}...")
        self._model = WhisperModel(self._model_name, device=device, compute_type=compute)
        self._log.success("Whisper geladen.")
        if self._model.model.device.startswith("cuda"):
            self._log.success("GPU-Beschleunigung aktiv.")
        else:
            self._log.info("GPU-Beschleunigung inaktiv (CPU).")

    def transcribe(self, video_path: Path, on_progress: Callable[[int], None] | None = None) -> TranscriptResult:
        """Transkribiert eine MP4-Datei."""
        if not self.ready:
            raise RuntimeError("Transcriber nicht initialisiert — init() aufrufen.")

        self._log.raw(f"\n{'='*50}")
        self._log.info(f"Verarbeite: {video_path.name}")
        self._log.info(f"Größe: {video_path.stat().st_size / (1024*1024):.1f} MB")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = Path(tmp.name)

        try:
            self._extract_audio(video_path, audio_path)
            return self._run_whisper(audio_path, on_progress)
        finally:
            if audio_path.exists():
                audio_path.unlink()

    def _extract_audio(self, video_path: Path, audio_path: Path) -> None:
        self._log.info("ffmpeg: Extrahiere Audio...")
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg Fehler:\n{result.stderr}")

        size_mb = audio_path.stat().st_size / (1024 * 1024)
        self._log.success(f"Audio extrahiert: {size_mb:.1f} MB")

    @staticmethod
    def to_text(segments: list[dict]) -> str:
        def ts(s: float) -> str:
            t = int(s)
            m, sec = (t % 3600) // 60, t % 60
            return f"{m:02d}:{sec:02d}" if t < 3600 else f"{t // 3600}:{m:02d}:{sec:02d}"

        return "\n".join(f"[{ts(seg['start'])}] {seg['text']}" for seg in segments)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        s = int(seconds)
        return f"{s // 60:02d}:{s % 60:02d}"

    def _run_whisper(self, audio_path: Path, on_progress: Callable[[int], None] | None = None) -> TranscriptResult:
        self._log.info("Whisper: Starte Transkription...")
        t_start = time.time()

        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language="de",
            beam_size=5,
        )

        self._log.info(f"Sprache: {info.language} ({info.language_probability:.0%})")

        segments = []
        for i, seg in enumerate(segments_iter, 1):
            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })
            elapsed = time.time() - t_start
            progress = seg.end / info.duration * 100
            # self._log.info(
            #     f"Whisper: {progress:.0f}% — "
            #     f"{self._fmt_time(seg.end)} / {self._fmt_time(info.duration)} "
            #     f"({elapsed:.0f}s Rechenzeit)..."
            # )
            if on_progress:
                on_progress(int(progress))

        elapsed = time.time() - t_start
        self._log.success(f"{len(segments)} Segmente in {elapsed:.1f}s")

        return TranscriptResult(
            text=" ".join(s["text"] for s in segments),
            language=info.language,
            segments=segments,
            duration_seconds=elapsed,
        )

    def download_audio(self, url: str) -> Path:
        """Lädt YouTube-Audio via yt-dlp als WAV herunter. Caller muss die Datei löschen."""
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            self._log.info("yt-dlp: Lade Audio herunter...")
            result = subprocess.run(
                [
                    "yt-dlp", "--extract-audio", "--audio-format", "wav",
                    "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
                    "-o", str(tmp_dir / "audio.%(ext)s"),
                    "--no-playlist", url,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"yt-dlp Fehler:\n{result.stderr}")

            wav_files = list(tmp_dir.glob("*.wav"))
            if not wav_files:
                raise RuntimeError("yt-dlp hat keine WAV-Datei erzeugt.")

            out = Path(tempfile.mktemp(suffix=".wav"))
            shutil.move(str(wav_files[0]), str(out))
            self._log.success(f"Audio heruntergeladen: {out.stat().st_size / (1024*1024):.1f} MB")
            return out
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def transcribe_audio(self, audio_path: Path, label: str = "",
                         on_progress: Callable[[int], None] | None = None) -> TranscriptResult:
        """Transkribiert eine bereits extrahierte WAV-Datei."""
        if not self.ready:
            raise RuntimeError("Transcriber nicht initialisiert — init() aufrufen.")
        self._log.raw(f"\n{'='*50}")
        if label:
            self._log.info(f"Verarbeite: {label}")
        self._log.info(f"Größe: {audio_path.stat().st_size / (1024*1024):.1f} MB")
        return self._run_whisper(audio_path, on_progress)
