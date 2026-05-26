import logging
import platform
import sys
from pathlib import Path

import config
from logger import setup_file_log
from gui import App

_BASE_DIR = Path(__file__).parent


def main():
    log_path = setup_file_log(_BASE_DIR / "logs")
    flog = logging.getLogger("transkript")

    flog.info("=" * 60)
    flog.info("App gestartet")
    flog.info(f"Python {sys.version}")
    flog.info(f"Platform: {platform.platform()}")
    flog.info(f"Log-Datei: {log_path}")

    cfg = config.load()
    flog.info("Konfiguration beim Start:")
    for k, v in cfg.items():
        flog.info(f"  {k} = {v}")

    try:
        app = App()
        app.mainloop()
    finally:
        flog.info("App beendet.")


if __name__ == "__main__":
    main()
