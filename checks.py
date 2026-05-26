import subprocess


def cuda_available() -> bool:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if result.returncode != 0:
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
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def ytdlp_available() -> bool:
    try:
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False
