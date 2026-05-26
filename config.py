import json
from pathlib import Path

CONFIG_PATH  = Path(__file__).parent / "transcript_config.json"
THEMES_PATH  = Path(__file__).parent / "themes.json"

_DEFAULTS: dict = {
    "default_folder": str(Path(__file__).parent),
    "use_default_folder": False,
    "output_folder_yt": str(Path(__file__).parent / "output_yt"),
    "model": "large-v3",
    "ui_scale": "medium",
    "theme": "discord_dark",
}


def load() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                data = json.load(f)
            return {**_DEFAULTS, **data}
        except Exception:
            pass
    return _DEFAULTS.copy()


def save(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


def load_theme(name: str = "discord_dark") -> dict:
    try:
        with open(THEMES_PATH, encoding="utf-8") as f:
            all_themes = json.load(f)
        return all_themes.get(name) or next(iter(all_themes.values()))
    except Exception:
        return {}


def list_themes() -> list[tuple[str, str]]:
    try:
        with open(THEMES_PATH, encoding="utf-8") as f:
            all_themes = json.load(f)
        return [(k, v.get("name", k)) for k, v in all_themes.items()]
    except Exception:
        return [("discord_dark", "Discord Dark")]
