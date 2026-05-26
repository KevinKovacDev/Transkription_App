"""
Logger-Modul: Gibt Nachrichten aus.
Ohne Callback: print() in die Konsole.
Mit Callback: z.B. in ein GUI-Fenster schreiben.

Nutzung:
    from logger import Logger, setup_file_log

    setup_file_log(Path("logs"))   # einmal beim Start aufrufen

    log = Logger()                    # Konsole
    log = Logger(callback=my_func)    # GUI oder anderes Ziel

    log.info("Starte...")
    log.success("Fertig.")
    log.error("Fehler!")
    log.exception("Fehler mit Traceback")   # im except-Block aufrufen
    log.progress(2, 5, "Video 2 von 5")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

_file_log: logging.Logger | None = None


def setup_file_log(log_dir: Path) -> Path:
    """Richtet File-Logging ein. Einmal beim App-Start aufrufen."""
    global _file_log

    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"transkript_{ts}.log"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    _file_log = logging.getLogger("transkript")
    _file_log.setLevel(logging.DEBUG)
    _file_log.addHandler(handler)

    def _excepthook(exc_type, exc_value, exc_tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            _file_log.critical("Unbehandelte Ausnahme", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    return log_path


def get_file_log() -> logging.Logger | None:
    return _file_log


class Logger:
    def __init__(self, callback: Callable[[str], None] | None = None):
        """
        Args:
            callback: Funktion die einen String entgegennimmt.
                      None → print() in die Konsole.
        """
        self._callback = callback

    def _write(self, msg: str) -> None:
        if self._callback:
            self._callback(msg)
        else:
            print(msg, flush=True)

    def _flog(self, level: int, msg: str) -> None:
        if _file_log:
            _file_log.log(level, msg)

    def info(self, msg: str) -> None:
        self._write(f"[INFO]  {msg}")
        self._flog(logging.INFO, msg)

    def success(self, msg: str) -> None:
        self._write(f"[OK]    {msg}")
        self._flog(logging.INFO, f"[OK] {msg}")

    def error(self, msg: str) -> None:
        self._write(f"[ERR]   {msg}")
        self._flog(logging.ERROR, msg)

    def exception(self, msg: str) -> None:
        """Im except-Block aufrufen — schreibt Fehlermeldung + vollen Traceback in die Log-Datei."""
        self._write(f"[ERR]   {msg}")
        if _file_log:
            _file_log.exception(msg)

    def progress(self, current: int, total: int, label: str = "") -> None:
        suffix = f"  {label}" if label else ""
        self._write(f"[{current}/{total}]{suffix}")
        self._flog(logging.DEBUG, f"Fortschritt {current}/{total} {label}".rstrip())

    def raw(self, msg: str) -> None:
        """Ohne Prefix — für freie Nachrichten."""
        self._write(msg)
        stripped = msg.strip()
        if stripped:
            self._flog(logging.DEBUG, stripped)
