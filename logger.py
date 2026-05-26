"""
Logger-Modul: Gibt Nachrichten aus.
Ohne Callback: print() in die Konsole.
Mit Callback: z.B. in ein GUI-Fenster schreiben.

Nutzung:
    from logger import Logger

    log = Logger()                    # Konsole
    log = Logger(callback=my_func)    # GUI oder anderes Ziel

    log.info("Starte...")
    log.success("Fertig.")
    log.error("Fehler!")
    log.progress(2, 5, "Video 2 von 5")
"""

from typing import Callable


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

    def info(self, msg: str) -> None:
        self._write(f"[INFO]  {msg}")

    def success(self, msg: str) -> None:
        self._write(f"[OK]    {msg}")

    def error(self, msg: str) -> None:
        self._write(f"[ERR]   {msg}")

    def progress(self, current: int, total: int, label: str = "") -> None:
        suffix = f"  {label}" if label else ""
        self._write(f"[{current}/{total}]{suffix}")

    def raw(self, msg: str) -> None:
        """Ohne Prefix — für freie Nachrichten."""
        self._write(msg)
