import logging
import queue
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
import tkinter.ttk as ttk
import webbrowser
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog

import config
from checks import cuda_available, ffmpeg_available, ytdlp_available
from logger import Logger, get_file_log
from transcriber import Transcriber

Theme: dict = {}


def _setup_scrollbar_style() -> None:
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "Flat.Vertical.TScrollbar",
        troughcolor=Theme["bg_secondary"],
        background=Theme["text_muted"],
        darkcolor=Theme["bg_secondary"],
        lightcolor=Theme["bg_secondary"],
        bordercolor=Theme["bg_secondary"],
        arrowcolor=Theme["bg_secondary"],
        arrowsize=12,
        width=12,
        relief="flat",
    )
    style.map(
        "Flat.Vertical.TScrollbar",
        background=[("active", Theme["text"]), ("!active", Theme["text_muted"])],
        troughcolor=[("active", Theme["bg_secondary"]), ("!active", Theme["bg_secondary"])],
    )

_SCALES = {
    "small":  {"font": 10, "font_sm": 9,  "font_load": 13, "padx": 20, "pady": 16},
    "medium": {"font": 12, "font_sm": 11, "font_load": 15, "padx": 32, "pady": 28},
    "large":  {"font": 15, "font_sm": 13, "font_load": 18, "padx": 44, "pady": 40},
}

_MODELS = [
    ("small",   "Small    —  schnell, schlechtere Qualität"),
    ("medium",  "Medium  —  ausgewogen"),
    ("large-v3","Large v3  —  langsam, beste Qualität"),
]

_WINDOW_SIZES = {
    "small":  "780x400",
    "medium": "1020x480",
    "large":  "1240x560",
}


class RoundedButton(tk.Canvas):
    def __init__(self, parent, text="", command=None, radius=5,
                 bg=None, fg=None, active_bg=None, disabled_fg=None,
                 font=("", 11), padx=10, pady=5, state="normal", cursor="", **kw):
        self._cmd      = command
        self._radius   = radius
        self._state    = state
        self._text     = text
        self._font     = font
        self._c_bg     = bg          or Theme["btn_bg"]
        self._c_fg     = fg          or Theme["text"]
        self._c_active = active_bg   or Theme["btn_active_bg"]
        self._c_dis_fg = disabled_fg or Theme["text_disabled"]
        self._hover    = False

        f  = tkfont.Font(font=font)
        w  = f.measure(text) + padx * 2 + 6
        h  = f.metrics("linespace") + pady * 2 + 2

        try:
            pbg = parent.cget("bg")
        except Exception:
            pbg = Theme["bg"]

        super().__init__(parent, width=w, height=h, bg=pbg,
                         highlightthickness=0, cursor=cursor, **kw)
        self._draw()
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>",    self._on_enter)
        self.bind("<Leave>",    self._on_leave)

    def _fill(self) -> str:
        if self._state == "disabled":
            return self._c_bg
        return self._c_active if self._hover else self._c_bg

    def _ink(self) -> str:
        return self._c_dis_fg if self._state == "disabled" else self._c_fg

    def _draw(self) -> None:
        self.delete("all")
        w, h, r = int(self["width"]), int(self["height"]), self._radius
        pts = [
            r, 0,   w-r, 0,
            w, 0,   w,   r,
            w, h-r, w,   h,
            w-r, h, r,   h,
            0, h,   0,   h-r,
            0, r,   0,   0,
        ]
        self.create_polygon(pts, smooth=True, fill=self._fill(), outline="")
        self.create_text(w // 2, h // 2, text=self._text,
                         font=self._font, fill=self._ink())

    def _on_click(self, _=None) -> None:
        if self._state != "disabled" and self._cmd:
            self._cmd()

    def _on_enter(self, _=None) -> None:
        if self._state != "disabled":
            self._hover = True
            self._draw()

    def _on_leave(self, _=None) -> None:
        self._hover = False
        self._draw()

    def config(self, **kw) -> None:
        redraw = False
        if "state" in kw:
            self._state = kw.pop("state")
            redraw = True
        if "text" in kw:
            self._text = kw.pop("text")
            redraw = True
        if "command" in kw:
            self._cmd = kw.pop("command")
        if kw:
            super().config(**kw)
        if redraw:
            self._draw()


def _btn(**kw) -> dict:
    return {
        "bg": Theme["btn_bg"],
        "fg": Theme["text"],
        "active_bg": Theme["btn_active_bg"],
        "disabled_fg": Theme["text_disabled"],
        **kw,
    }


def _flat_btn(**kw) -> dict:
    return {
        "bg": Theme["bg_secondary"],
        "fg": Theme["text_muted"],
        "activebackground": Theme["bg_elevated"],
        "activeforeground": Theme["text"],
        "relief": "flat", "highlightthickness": 0, "bd": 0,
        **kw,
    }


class App(tk.Tk):
    def __init__(self):
        global Theme
        super().__init__()
        self.title("Transkript")
        self.resizable(False, False)

        cfg_early = config.load()
        Theme = config.load_theme(cfg_early.get("theme", "discord_dark"))
        self.configure(bg=Theme["bg"])
        _setup_scrollbar_style()
        self.geometry(_WINDOW_SIZES.get(cfg_early.get("ui_scale", "medium"), "880x480"))

        if not ffmpeg_available():
            FfmpegMissingFrame(self).pack(fill="both", expand=True)
            return

        self._cfg = cfg_early
        self._use_cuda = cuda_available()
        self._ytdlp_ok = ytdlp_available()
        self._ffmpeg_ok = True

        flog = get_file_log()
        if flog:
            flog.info(f"ffmpeg: verfügbar, CUDA: {self._use_cuda}, yt-dlp: {self._ytdlp_ok}")

        self._transcriber = Transcriber(logger=Logger(), use_cuda=self._use_cuda, model=self._cfg.get("model", "large-v3"))
        self._queue: queue.Queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._done_count = 0
        self._total_count = 0

        self._loading_frame = LoadingFrame(self, cfg=self._cfg)
        self._main_frame = self._build_main_frame()

        self._loading_frame.pack(fill="both", expand=True)
        threading.Thread(target=self._load_whisper, daemon=True).start()
        self.after(100, self._poll_queue)
        self.after(0, self._apply_titlebar_theme)

    def _build_main_frame(self) -> "MainFrame":
        return MainFrame(
            self,
            cuda=self._use_cuda,
            ffmpeg=self._ffmpeg_ok,
            ytdlp=self._ytdlp_ok,
            cfg=self._cfg,
            on_start=self._start_transcription,
            on_cancel=self._cancel_transcription,
            on_settings=self._open_settings,
        )

    def _update_geometry(self) -> None:
        self.geometry(_WINDOW_SIZES.get(self._cfg.get("ui_scale", "medium"), "880x480"))

    def _apply_titlebar_theme(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())

            def to_colorref(hex_color: str) -> int:
                r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
                return r | (g << 8) | (b << 16)

            for attr, color_key in ((35, "bg"), (36, "text")):
                val = ctypes.c_int(to_colorref(Theme[color_key]))
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(val), ctypes.sizeof(val))
        except Exception:
            pass

    def _cancel_transcription(self) -> None:
        self._cancel_event.set()

    def _open_settings(self) -> None:
        SettingsWindow(self, self._cfg, on_save=self._on_settings_save)

    def _on_settings_save(self, model_changed: bool, theme_changed: bool = False) -> None:
        global Theme
        self._cfg = config.load()

        flog = get_file_log()
        if flog:
            flog.info(
                f"Einstellungen gespeichert — Modell: {self._cfg.get('model')}, "
                f"Theme: {self._cfg.get('theme')}, Skalierung: {self._cfg.get('ui_scale')}, "
                f"Modell neu geladen: {model_changed}"
            )

        if theme_changed:
            Theme = config.load_theme(self._cfg.get("theme", "discord_dark"))
            self.configure(bg=Theme["bg"])
            self._apply_titlebar_theme()
            _setup_scrollbar_style()
        if model_changed:
            self._apply_with_model_reload()
        else:
            self._apply_scale()

    def _apply_scale(self) -> None:
        self._main_frame.pack_forget()
        self._main_frame.destroy()
        self._main_frame = self._build_main_frame()
        self._main_frame.pack(fill="both", expand=True)
        self._update_geometry()

    def _apply_with_model_reload(self) -> None:
        self._main_frame.pack_forget()
        self._main_frame.destroy()
        self._loading_frame.pack_forget()
        self._loading_frame.destroy()

        self._loading_frame = LoadingFrame(self, cfg=self._cfg)
        self._main_frame = self._build_main_frame()
        self._loading_frame.pack(fill="both", expand=True)
        self.title("Transkript")
        self._update_geometry()

        self._transcriber = Transcriber(logger=Logger(), use_cuda=self._use_cuda, model=self._cfg.get("model", "large-v3"))
        threading.Thread(target=self._load_whisper, daemon=True).start()

    def _load_whisper(self) -> None:
        try:
            self._transcriber.init()
            self._queue.put(("ready", None))
        except Exception as e:
            flog = get_file_log()
            if flog:
                flog.exception("Whisper konnte nicht geladen werden")
            self._queue.put(("error", str(e)))

    def _start_transcription(self) -> None:
        self._cancel_event.clear()
        pending = self._main_frame.pending_items()
        self._done_count = 0
        self._total_count = len(pending)

        flog = get_file_log()
        if flog:
            flog.info(f"Transkription gestartet: {len(pending)} Element(e)")
            for item in pending:
                if item["section"] == "pc":
                    flog.info(f"  [PC]  {item['path']}")
                else:
                    flog.info(f"  [YT]  {item.get('title') or item['url']}")

        self._update_title()
        self._main_frame.set_all_waiting()
        threading.Thread(target=self._run_all, args=(pending,), daemon=True).start()

    def _run_all(self, pending: list) -> None:
        for item in pending:
            if self._cancel_event.is_set():
                break

            section = item["section"]
            idx     = item["idx"]

            def on_progress(pct: int, s=section, i=idx) -> None:
                if self._cancel_event.is_set():
                    raise InterruptedError
                self._queue.put(("progress", (s, i, pct)))

            label = str(item["path"]) if section == "pc" else (item.get("title") or item["url"])
            flog = get_file_log()

            try:
                if section == "pc":
                    result = self._transcriber.transcribe(item["path"], on_progress=on_progress)
                    output_path = item["path"].parent / f"{item['path'].stem}.txt"
                else:
                    audio_path = self._transcriber.download_audio(item["url"])
                    try:
                        result = self._transcriber.transcribe_audio(
                            audio_path, label=item.get("title", ""), on_progress=on_progress,
                        )
                    finally:
                        if audio_path.exists():
                            audio_path.unlink()
                    title = item.get("title") or "youtube_transcript"
                    safe  = "".join(c for c in title if c.isalnum() or c in " -_").strip() or "youtube_transcript"
                    folder = Path(self._cfg.get("output_folder_yt", Path(__file__).parent / "output_yt"))
                    folder.mkdir(parents=True, exist_ok=True)
                    output_path = folder / f"{safe}.txt"

                output_path.write_text(Transcriber.to_text(result.segments), encoding="utf-8")
                if flog:
                    flog.info(f"Fertig: {label} → {output_path}")
                self._queue.put(("video_done", (section, idx)))
            except InterruptedError:
                if flog:
                    flog.info(f"Abgebrochen: {label}")
                self._queue.put(("video_reverted", (section, idx)))
                break
            except Exception as e:
                if flog:
                    flog.exception(f"Fehler bei: {label}")
                self._queue.put(("video_error", (section, idx, str(e))))

        self._queue.put(("all_done", None))

    def _update_title(self) -> None:
        self.title(f"Transkript — {self._done_count} / {self._total_count}")

    def _poll_queue(self) -> None:
        try:
            event, data = self._queue.get_nowait()
            if event == "ready":
                self._loading_frame.pack_forget()
                self._main_frame.pack(fill="both", expand=True)
            elif event == "error":
                self._loading_frame.show_error(data)
            elif event == "progress":
                section, idx, pct = data
                self._main_frame.set_progress(section, idx, f"{pct}%", pct=pct)
            elif event == "video_done":
                section, idx = data
                self._main_frame.set_progress(section, idx, "fertig")
                self._done_count += 1
                self._update_title()
            elif event == "video_error":
                section, idx, _ = data
                self._main_frame.set_progress(section, idx, "Fehler")
            elif event == "video_reverted":
                section, idx = data
                self._main_frame.revert_progress(section, idx)
            elif event == "all_done":
                self._main_frame.on_transcription_done()
                self.title("Transkript")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)


class FfmpegMissingFrame(tk.Frame):
    _INSTALL_CMD = {
        "win32":  "winget install ffmpeg",
        "darwin": "brew install ffmpeg",
    }

    def __init__(self, parent: tk.Tk):
        super().__init__(parent, bg=Theme["bg"], padx=48, pady=40)

        tk.Label(self, text="ffmpeg nicht gefunden", font=("", 15, "bold"),
                 bg=Theme["bg"], fg=Theme["text"]).pack(pady=(0, 12))
        tk.Label(
            self,
            text="ffmpeg wird benötigt, um Audio aus Videos zu extrahieren.\nBitte installieren und die App neu starten.",
            font=("", 12), justify="center", bg=Theme["bg"], fg=Theme["text_muted"],
        ).pack(pady=(0, 20))

        cmd = self._INSTALL_CMD.get(sys.platform, "sudo apt install ffmpeg")
        hint = tk.Frame(self, bg=Theme["bg_tertiary"], padx=14, pady=10)
        hint.pack()
        tk.Label(hint, text=cmd, font=("Courier", 12),
                 bg=Theme["bg_tertiary"], fg=Theme["text"]).pack()

        RoundedButton(self, text="Schließen", command=parent.destroy,
                      font=("", 11), padx=10, pady=5, **_btn()).pack(pady=(24, 0))


class LoadingFrame(tk.Frame):
    def __init__(self, parent: tk.Tk, cfg: dict):
        scale = _SCALES.get(cfg.get("ui_scale", "medium"), _SCALES["medium"])
        super().__init__(parent, bg=Theme["bg"], padx=scale["padx"] * 2, pady=scale["pady"] * 2)
        model_label = next((lbl for val, lbl in _MODELS if val == cfg.get("model", "large-v3")), cfg.get("model", "large-v3"))
        self._label = tk.Label(self, text=f"Lade {model_label.split('—')[0].strip()}...",
                               font=("", scale["font_load"]), bg=Theme["bg"], fg=Theme["text"])
        self._label.pack()

    def show_error(self, msg: str) -> None:
        self._label.config(text=f"Fehler beim Laden:\n{msg}", fg=Theme["danger"])


class MainFrame(tk.Frame):
    _MAX_NAME_CHARS = 30

    def __init__(self, parent: tk.Tk, cuda: bool, ffmpeg: bool, ytdlp: bool, cfg: dict,
                 on_start: Callable, on_cancel: Callable, on_settings: Callable):
        scale = _SCALES.get(cfg.get("ui_scale", "medium"), _SCALES["medium"])
        super().__init__(parent, bg=Theme["bg"], padx=scale["padx"], pady=scale["pady"])

        self._F     = ("", scale["font"])
        self._F_SM  = ("", scale["font_sm"])
        self._F_BTN = ("", scale["font_sm"])
        self._cfg   = cfg
        self._ffmpeg = ffmpeg
        self._ytdlp  = ytdlp
        self._cuda   = cuda
        self._on_cancel    = on_cancel
        self._transcribing = False
        self._rows:    list[dict] = []
        self._rows_yt: list[dict] = []
        self._last_folder = Path(cfg.get("default_folder", Path.home()))

        f, px, py = self._F_BTN, 10, 5

        # ── Top bar ───────────────────────────────────────────────────────────
        top_bar = tk.Frame(self, bg=Theme["bg"])
        top_bar.pack(side="top", fill="x", pady=(0, 8))
        self._settings_btn = RoundedButton(top_bar, text="⚙ Einstellungen",
                                           font=self._F_BTN, padx=10, pady=4,
                                           command=on_settings, **_btn())
        self._settings_btn.pack(side="right")

        # ── Footer (packed first so columns can expand into remaining space) ──
        footer = tk.Frame(self, bg=Theme["bg"])
        footer.pack(side="bottom", fill="x", pady=(5, 0))

        dot_color = Theme["success"] if cuda else Theme["danger"]
        tk.Label(footer, text="●", fg=dot_color, font=("", 8),
                 bg=Theme["bg"]).pack(side="left")
        tk.Label(footer, text=" CUDA", font=("", 9),
                 bg=Theme["bg"], fg=Theme["text_muted"]).pack(side="left")
        tk.Button(footer, text="?", font=("", 8), cursor="hand2",
                  command=self._open_cuda_info, **_flat_btn()).pack(side="left", padx=(2, 14))

        model_name = next((lbl.split("—")[0].strip() for val, lbl in _MODELS
                           if val == cfg.get("model", "large-v3")), cfg.get("model", "large-v3"))
        tk.Label(footer, text=f"Modell: {model_name}", font=("", 9),
                 bg=Theme["bg"], fg=Theme["text_muted"]).pack(side="left")

        tk.Frame(self, height=1, bg=Theme["separator"]).pack(side="bottom", fill="x")

        # ── Shared bottom buttons ─────────────────────────────────────────────
        shared_btns = tk.Frame(self, bg=Theme["bg"])
        shared_btns.pack(side="bottom", fill="x", pady=(12, 0))
        self._start_btn = RoundedButton(shared_btns, text="Starten", state="disabled",
                                        font=f, padx=px, pady=py,
                                        command=on_start, **_btn())
        self._start_btn.pack(side="left")
        self._close_btn = RoundedButton(shared_btns, text="Abbrechen",
                                        font=f, padx=px, pady=py,
                                        command=self._on_cancel, **_btn())
        self._remove_done_btn = RoundedButton(shared_btns, text="Fertige entfernen", state="disabled",
                                              font=f, padx=px, pady=py,
                                              command=self._remove_done_rows, **_btn())
        self._remove_done_btn.pack(side="right")

        # ── Two-column body (expands into remaining space) ────────────────────
        columns = tk.Frame(self, bg=Theme["bg"])
        columns.pack(fill="both", expand=True)
        columns.columnconfigure(0, weight=1)
        columns.columnconfigure(2, weight=1)
        columns.rowconfigure(0, weight=1)

        # Left — PC
        left = tk.Frame(columns, bg=Theme["bg"])
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_propagate(False)

        tk.Label(left, text="Dateien (PC)", font=("", scale["font_sm"], "bold"),
                 bg=Theme["bg"], fg=Theme["text_muted"]).pack(anchor="w", pady=(0, 4))

        self._add_btn = RoundedButton(left, text="Hinzufügen",
                                      font=f, padx=px, pady=py,
                                      command=self._add_files, **_btn())
        self._add_btn.pack(anchor="w", pady=(0, 6))

        self._list_container = tk.Frame(left, bg=Theme["bg_secondary"],
                                        highlightbackground=Theme["separator"],
                                        highlightthickness=1)
        self._list_container.pack(fill="both", expand=True, pady=(0, 8))
        _canvas = tk.Canvas(self._list_container, bg=Theme["bg_secondary"], highlightthickness=0, bd=0)
        _sb = ttk.Scrollbar(self._list_container, orient="vertical", command=_canvas.yview, style="Flat.Vertical.TScrollbar")
        _canvas.configure(yscrollcommand=_sb.set)
        _sb.pack(side="right", fill="y")
        _canvas.pack(side="left", fill="both", expand=True)
        self._list_inner = tk.Frame(_canvas, bg=Theme["bg_secondary"])
        _cw = _canvas.create_window((0, 0), window=self._list_inner, anchor="nw")
        self._list_inner.bind("<Configure>", lambda e: _canvas.configure(scrollregion=_canvas.bbox("all")))
        _canvas.bind("<Configure>", lambda e: _canvas.itemconfig(_cw, width=e.width))
        _canvas.bind("<MouseWheel>", lambda e: _canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        self._list_inner.bind("<MouseWheel>", lambda e: _canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        self._empty_placeholder = tk.Frame(self._list_inner, bg=Theme["bg_secondary"], height=80)
        self._empty_placeholder.pack(fill="x", padx=12, pady=12)

        # Vertical separator
        tk.Frame(columns, width=1, bg=Theme["separator"]).grid(row=0, column=1, sticky="ns")

        # Right — YouTube
        right = tk.Frame(columns, bg=Theme["bg"])
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        right.grid_propagate(False)

        tk.Label(right, text="YouTube", font=("", scale["font_sm"], "bold"),
                 bg=Theme["bg"], fg=Theme["text_muted"]).pack(anchor="w", pady=(0, 4))

        url_row = tk.Frame(right, bg=Theme["bg"])
        url_row.pack(side="top", fill="x", pady=(0, 6))
        self._url_entry = tk.Entry(
            url_row, font=self._F_SM,
            bg=Theme["bg_secondary"], fg=Theme["text"],
            insertbackground=Theme["text"],
            relief="flat", highlightthickness=1,
            highlightbackground=Theme["separator"],
            highlightcolor=Theme["accent"],
            disabledbackground=Theme["bg_secondary"],
        )
        self._url_entry.pack(side="left", fill="x", expand=True, ipady=4)
        self._url_entry.bind("<Return>", lambda _: self._add_yt_url())

        self._add_yt_btn = RoundedButton(url_row, text="+", font=f, padx=px, pady=py,
                                         command=self._add_yt_url,
                                         state="normal" if ytdlp else "disabled",
                                         **_btn())
        self._add_yt_btn.pack(side="left", padx=(6, 0))

        if not ytdlp:
            tk.Label(right, text="yt-dlp nicht gefunden", font=self._F_SM,
                     bg=Theme["bg"], fg=Theme["danger"]).pack(anchor="w", pady=(0, 4))
            self._url_entry.config(state="disabled")

        self._list_container_yt = tk.Frame(right, bg=Theme["bg_secondary"],
                                           highlightbackground=Theme["separator"],
                                           highlightthickness=1)
        self._list_container_yt.pack(fill="both", expand=True, pady=(0, 8))
        _canvas_yt = tk.Canvas(self._list_container_yt, bg=Theme["bg_secondary"], highlightthickness=0, bd=0)
        _sb_yt = ttk.Scrollbar(self._list_container_yt, orient="vertical", command=_canvas_yt.yview, style="Flat.Vertical.TScrollbar")
        _canvas_yt.configure(yscrollcommand=_sb_yt.set)
        _sb_yt.pack(side="right", fill="y")
        _canvas_yt.pack(side="left", fill="both", expand=True)
        self._list_inner_yt = tk.Frame(_canvas_yt, bg=Theme["bg_secondary"])
        _cw_yt = _canvas_yt.create_window((0, 0), window=self._list_inner_yt, anchor="nw")
        self._list_inner_yt.bind("<Configure>", lambda e: _canvas_yt.configure(scrollregion=_canvas_yt.bbox("all")))
        _canvas_yt.bind("<Configure>", lambda e: _canvas_yt.itemconfig(_cw_yt, width=e.width))
        _canvas_yt.bind("<MouseWheel>", lambda e: _canvas_yt.yview_scroll(-1 * (e.delta // 120), "units"))
        self._list_inner_yt.bind("<MouseWheel>", lambda e: _canvas_yt.yview_scroll(-1 * (e.delta // 120), "units"))
        self._empty_placeholder_yt = tk.Frame(self._list_inner_yt, bg=Theme["bg_secondary"], height=80)
        self._empty_placeholder_yt.pack(fill="x", padx=12, pady=12)

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def video_paths(self) -> list[Path]:
        return [r["path"] for r in self._rows]

    def pending_items(self) -> list[dict]:
        items = []
        for i, r in enumerate(self._rows):
            if not r.get("done"):
                items.append({"section": "pc", "idx": i, "path": r["path"]})
        for i, r in enumerate(self._rows_yt):
            if not r.get("done"):
                items.append({"section": "yt", "idx": i, "url": r["url"], "title": r.get("title", "")})
        return items

    # ── PC section ────────────────────────────────────────────────────────────

    def _add_files(self) -> None:
        if self._cfg.get("use_default_folder", False):
            initial = str(Path(self._cfg.get("default_folder", Path.home())))
        else:
            initial = str(self._last_folder)

        paths = filedialog.askopenfilenames(
            title="Videos auswählen",
            initialdir=initial,
            filetypes=[("MP4-Dateien", "*.mp4")],
        )
        if not paths:
            return

        existing = set(self.video_paths)
        for p in paths:
            path = Path(p)
            if path not in existing:
                self._add_row(path)
                existing.add(path)

        self._last_folder = Path(paths[0]).parent
        self._update_start_btn()

    @staticmethod
    def _get_duration(path: Path) -> str:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, timeout=10,
            )
            total = int(float(result.stdout.strip()))
            h, m, s = total // 3600, (total % 3600) // 60, total % 60
            return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        except Exception:
            return "–"

    @staticmethod
    def _open_folder(path: Path) -> None:
        if sys.platform == "win32":
            subprocess.Popen(f'explorer /select,"{path}"', shell=True)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path.parent)])

    def _add_row(self, path: Path) -> None:
        if self._empty_placeholder.winfo_ismapped():
            self._empty_placeholder.pack_forget()

        wrapper = tk.Frame(self._list_inner, bg=Theme["bg_secondary"])
        wrapper.pack(fill="x", padx=12, pady=(8, 3))

        info_row = tk.Frame(wrapper, bg=Theme["bg_secondary"])
        info_row.pack(fill="x")

        num_lbl = tk.Label(info_row, text=f"{len(self._rows) + 1}.", width=3, anchor="e",
                           bg=Theme["bg_secondary"], fg=Theme["text_muted"], font=self._F_SM)
        num_lbl.pack(side="left")

        name = path.name if len(path.name) <= self._MAX_NAME_CHARS else path.name[:self._MAX_NAME_CHARS - 1] + "…"
        tk.Label(info_row, text=name, anchor="w", width=1, bg=Theme["bg_secondary"],
                 fg=Theme["text"], font=self._F).pack(side="left", padx=(10, 0), fill="x", expand=True)

        duration_lbl = tk.Label(info_row, text="...", width=7, anchor="e",
                                bg=Theme["bg_secondary"], fg=Theme["text_muted"], font=self._F_SM)
        duration_lbl.pack(side="right", padx=(0, 10))

        def _fetch(lbl=duration_lbl, p=path):
            dur = self._get_duration(p)
            lbl.after(0, lambda: lbl.config(text=dur))

        threading.Thread(target=_fetch, daemon=True).start()

        progress_lbl = tk.Label(info_row, text="", width=8, anchor="e",
                                bg=Theme["bg_secondary"], fg=Theme["text_muted"], font=self._F_SM)
        progress_lbl.pack(side="right")

        remove_btn = tk.Button(info_row, text="✕", width=2, font=self._F_SM,
                               command=lambda p=path: self._remove_row(p), **_flat_btn())
        remove_btn.pack(side="right", padx=(6, 6))

        folder_btn = tk.Button(info_row, text="↗", width=2, font=self._F_SM,
                               command=lambda p=path: self._open_folder(p), **_flat_btn())
        folder_btn.pack(side="right")

        bar_track = tk.Frame(wrapper, bg=Theme["bg_tertiary"], height=4)
        bar_track.pack(fill="x", pady=(5, 0))
        bar_track.pack_propagate(False)

        bar_fill = tk.Frame(bar_track, bg=Theme["accent"], height=4)
        bar_fill.place(x=0, y=0, relheight=1.0, relwidth=0.0)

        self._rows.append({
            "path": path,
            "wrapper": wrapper,
            "num_label": num_lbl,
            "progress_label": progress_lbl,
            "bar_fill": bar_fill,
            "remove_btn": remove_btn,
            "done": False,
        })

        output = path.parent / f"{path.stem}.txt"
        if output.exists():
            self.set_progress("pc", len(self._rows) - 1, "fertig")
            self._update_start_btn()

    def _remove_row(self, path: Path) -> None:
        idx = next(i for i, r in enumerate(self._rows) if r["path"] == path)
        self._rows[idx]["wrapper"].destroy()
        self._rows.pop(idx)
        for i, row in enumerate(self._rows):
            row["num_label"].config(text=f"{i + 1}.")
        if not self._rows:
            self._empty_placeholder.pack(fill="x", padx=12, pady=12)
        self._update_start_btn()

    def _remove_done_rows(self) -> None:
        for path in [r["path"] for r in self._rows if r.get("done")]:
            self._remove_row(path)
        for url in [r["url"] for r in self._rows_yt if r.get("done")]:
            self._remove_yt_row(url)
        self._remove_done_btn.config(state="disabled")

    # ── YouTube section ───────────────────────────────────────────────────────

    def _add_yt_url(self) -> None:
        url = self._url_entry.get().strip()
        if not url:
            return
        existing = {r["url"] for r in self._rows_yt}
        if url in existing:
            return
        self._url_entry.delete(0, tk.END)
        self._add_yt_row(url)
        self._update_start_btn()

    @staticmethod
    def _get_yt_info(url: str) -> tuple[str, str]:
        try:
            result = subprocess.run(
                ["yt-dlp", "--simulate", "--print", "%(title)s",
                 "--print", "%(duration_string)s", "--no-playlist", url],
                capture_output=True, text=True, timeout=20,
            )
            lines    = result.stdout.strip().splitlines()
            title    = lines[0] if lines else ""
            duration = lines[1] if len(lines) > 1 else "–"
            return title, duration
        except Exception:
            return "", "–"

    def _add_yt_row(self, url: str) -> None:
        if self._empty_placeholder_yt.winfo_ismapped():
            self._empty_placeholder_yt.pack_forget()

        wrapper = tk.Frame(self._list_inner_yt, bg=Theme["bg_secondary"])
        wrapper.pack(fill="x", padx=12, pady=(8, 3))

        info_row = tk.Frame(wrapper, bg=Theme["bg_secondary"])
        info_row.pack(fill="x")

        num_lbl = tk.Label(info_row, text=f"{len(self._rows_yt) + 1}.", width=3, anchor="e",
                           bg=Theme["bg_secondary"], fg=Theme["text_muted"], font=self._F_SM)
        num_lbl.pack(side="left")

        title_lbl = tk.Label(info_row, text="lädt...", anchor="w", width=1,
                             bg=Theme["bg_secondary"], fg=Theme["text_muted"], font=self._F)
        title_lbl.pack(side="left", padx=(10, 0), fill="x", expand=True)

        duration_lbl = tk.Label(info_row, text="...", width=7, anchor="e",
                                bg=Theme["bg_secondary"], fg=Theme["text_muted"], font=self._F_SM)
        duration_lbl.pack(side="right", padx=(0, 10))

        progress_lbl = tk.Label(info_row, text="", width=8, anchor="e",
                                bg=Theme["bg_secondary"], fg=Theme["text_muted"], font=self._F_SM)
        progress_lbl.pack(side="right")

        remove_btn = tk.Button(info_row, text="✕", width=2, font=self._F_SM,
                               command=lambda u=url: self._remove_yt_row(u), **_flat_btn())
        remove_btn.pack(side="right", padx=(6, 6))

        link_btn = tk.Button(info_row, text="↗", width=2, font=self._F_SM,
                             command=lambda u=url: webbrowser.open(u), **_flat_btn())
        link_btn.pack(side="right")

        bar_track = tk.Frame(wrapper, bg=Theme["bg_tertiary"], height=4)
        bar_track.pack(fill="x", pady=(5, 0))
        bar_track.pack_propagate(False)

        bar_fill = tk.Frame(bar_track, bg=Theme["accent"], height=4)
        bar_fill.place(x=0, y=0, relheight=1.0, relwidth=0.0)

        row_ref = {
            "url": url,
            "title": "",
            "wrapper": wrapper,
            "num_label": num_lbl,
            "title_label": title_lbl,
            "progress_label": progress_lbl,
            "bar_fill": bar_fill,
            "remove_btn": remove_btn,
            "done": False,
        }
        self._rows_yt.append(row_ref)

        def _fetch(row=row_ref, u=url, t_lbl=title_lbl, d_lbl=duration_lbl):
            title, duration = self._get_yt_info(u)
            display = title if len(title) <= self._MAX_NAME_CHARS else title[:self._MAX_NAME_CHARS - 1] + "…"
            def _apply():
                try:
                    row["title"] = title
                    t_lbl.config(text=display or u, fg=Theme["text"])
                    d_lbl.config(text=duration)
                except tk.TclError:
                    pass
            t_lbl.after(0, _apply)

        threading.Thread(target=_fetch, daemon=True).start()

    def _remove_yt_row(self, url: str) -> None:
        idx = next(i for i, r in enumerate(self._rows_yt) if r["url"] == url)
        self._rows_yt[idx]["wrapper"].destroy()
        self._rows_yt.pop(idx)
        for i, row in enumerate(self._rows_yt):
            row["num_label"].config(text=f"{i + 1}.")
        if not self._rows_yt:
            self._empty_placeholder_yt.pack(fill="x", padx=12, pady=12)
        self._update_start_btn()

    # ── Shared state ──────────────────────────────────────────────────────────

    def set_all_waiting(self) -> None:
        self._transcribing = True
        self._settings_btn.config(state="disabled")
        self._add_btn.config(state="disabled")
        self._add_yt_btn.config(state="disabled")
        self._url_entry.config(state="disabled")
        self._start_btn.config(state="disabled")
        self._remove_done_btn.config(state="disabled")
        self._close_btn.pack(side="right", padx=(10, 0))
        for row in self._rows:
            row["remove_btn"].config(state="disabled")
            if not row.get("done"):
                row["progress_label"].config(text="wartet...")
        for row in self._rows_yt:
            row["remove_btn"].config(state="disabled")
            if not row.get("done"):
                row["progress_label"].config(text="wartet...")

    def set_progress(self, section: str, idx: int, text: str, pct: int | None = None) -> None:
        rows = self._rows if section == "pc" else self._rows_yt
        rows[idx]["progress_label"].config(text=text)
        if pct is not None:
            rows[idx]["bar_fill"].place(relwidth=pct / 100)
        elif text == "fertig":
            rows[idx]["bar_fill"].place(relwidth=1.0)
            rows[idx]["done"] = True
            self._remove_done_btn.config(state="normal")

    def revert_progress(self, section: str, idx: int) -> None:
        rows = self._rows if section == "pc" else self._rows_yt
        rows[idx]["progress_label"].config(text="")
        rows[idx]["bar_fill"].place(relwidth=0.0)

    def on_transcription_done(self) -> None:
        self._transcribing = False
        self._settings_btn.config(state="normal")
        self._add_btn.config(state="normal")
        if self._ytdlp:
            self._add_yt_btn.config(state="normal")
            self._url_entry.config(state="normal")
        self._close_btn.pack_forget()
        has_done = any(r.get("done") for r in self._rows + self._rows_yt)
        self._remove_done_btn.config(state="normal" if has_done else "disabled")
        for row in self._rows:
            if not row.get("done"):
                row["remove_btn"].config(state="normal")
                if row["progress_label"].cget("text") == "wartet...":
                    row["progress_label"].config(text="")
        for row in self._rows_yt:
            if not row.get("done"):
                row["remove_btn"].config(state="normal")
                if row["progress_label"].cget("text") == "wartet...":
                    row["progress_label"].config(text="")
        self._update_start_btn()

    def _update_start_btn(self) -> None:
        has_pending = bool(self.pending_items())
        state = "normal" if (self._ffmpeg and has_pending and not self._transcribing) else "disabled"
        self._start_btn.config(state=state)

    def _open_cuda_info(self) -> None:
        CudaInfoWindow(self.master, self._cuda)


class CudaInfoWindow(tk.Toplevel):
    _CUDA_URL = "developer.nvidia.com/cuda-downloads"
    _INSTALL_CMD = {
        "win32":  "winget install cuda",
        "darwin": "brew install --cask cuda",
    }

    def __init__(self, parent: tk.Tk, cuda_active: bool):
        super().__init__(parent)
        self.title("CUDA — Info")
        self.resizable(False, False)
        self.grab_set()
        self.focus_set()
        self.configure(bg=Theme["bg"])

        outer = tk.Frame(self, bg=Theme["bg"], padx=32, pady=28)
        outer.pack()

        tk.Label(outer, text="Was ist CUDA?", font=("", 13, "bold"),
                 bg=Theme["bg"], fg=Theme["text"]).pack(anchor="w", pady=(0, 10))
        tk.Label(
            outer,
            text="CUDA ist NVIDIAs Technologie zur GPU-Beschleunigung.\n"
                 "Mit einer kompatiblen NVIDIA-Grafikkarte läuft\n"
                 "Whisper deutlich schneller als auf der CPU.",
            font=("", 11), justify="left", bg=Theme["bg"], fg=Theme["text_muted"],
        ).pack(anchor="w", pady=(0, 16))

        status_row = tk.Frame(outer, bg=Theme["bg"])
        status_row.pack(anchor="w", pady=(0, 16))
        tk.Label(status_row, text="Status:  ", font=("", 11),
                 bg=Theme["bg"], fg=Theme["text"]).pack(side="left")
        dot_color = Theme["success"] if cuda_active else Theme["danger"]
        status_text = "aktiv" if cuda_active else "inaktiv — läuft auf CPU"
        tk.Label(status_row, text="●", fg=dot_color, font=("", 10),
                 bg=Theme["bg"]).pack(side="left")
        tk.Label(status_row, text=f"  {status_text}", font=("", 11),
                 bg=Theme["bg"], fg=Theme["text"]).pack(side="left")

        if not cuda_active:
            tk.Label(outer, text="Download / Installation:", font=("", 11),
                     bg=Theme["bg"], fg=Theme["text"]).pack(anchor="w", pady=(0, 6))
            cmd = self._INSTALL_CMD.get(sys.platform, "sudo apt install nvidia-cuda-toolkit")
            for snippet in (cmd, self._CUDA_URL):
                box = tk.Frame(outer, bg=Theme["bg_tertiary"], padx=14, pady=8)
                box.pack(anchor="w", fill="x", pady=(0, 6))
                tk.Label(box, text=snippet, font=("Courier", 11),
                         bg=Theme["bg_tertiary"], fg=Theme["text"]).pack(anchor="w")

        RoundedButton(outer, text="Schließen", command=self.destroy,
                      font=("", 11), padx=10, pady=5,
                      **_btn()).pack(anchor="e", pady=(10, 0))


class SettingsWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, cfg: dict, on_save: Callable[[bool], None]):
        super().__init__(parent)
        self.title("Einstellungen")
        self.resizable(False, False)
        self.grab_set()
        self.focus_set()
        self.configure(bg=Theme["bg"])

        self._cfg = cfg
        self._on_save = on_save
        self._scale_var = tk.StringVar(value=cfg.get("ui_scale", "medium"))
        self._model_var = tk.StringVar(value=cfg.get("model", "large-v3"))
        self._theme_var = tk.StringVar(value=cfg.get("theme", "discord_dark"))
        self._folder_var = tk.StringVar(value=cfg.get("default_folder", str(Path(__file__).parent)))

        outer = tk.Frame(self, bg=Theme["bg"], padx=32, pady=28)
        outer.pack()

        rb = {
            "bg": Theme["bg"], "fg": Theme["text"], "font": ("", 12),
            "selectcolor": Theme["bg_elevated"],
            "activebackground": Theme["bg"], "activeforeground": Theme["text"],
            "highlightthickness": 0,
        }

        tk.Label(outer, text="UI-Größe", font=("", 13, "bold"),
                 bg=Theme["bg"], fg=Theme["text"]).pack(anchor="w", pady=(0, 8))
        for value, label in [("small", "Klein"), ("medium", "Mittel"), ("large", "Groß")]:
            tk.Radiobutton(outer, text=label, variable=self._scale_var, value=value, **rb).pack(anchor="w", pady=2)

        tk.Frame(outer, height=1, bg=Theme["separator"]).pack(fill="x", pady=(16, 16))

        tk.Label(outer, text="Whisper-Modell", font=("", 13, "bold"),
                 bg=Theme["bg"], fg=Theme["text"]).pack(anchor="w", pady=(0, 8))
        for value, label in _MODELS:
            tk.Radiobutton(outer, text=label, variable=self._model_var, value=value, **rb).pack(anchor="w", pady=2)

        tk.Frame(outer, height=1, bg=Theme["separator"]).pack(fill="x", pady=(16, 16))

        tk.Label(outer, text="Theme", font=("", 13, "bold"),
                 bg=Theme["bg"], fg=Theme["text"]).pack(anchor="w", pady=(0, 8))
        for value, label in config.list_themes():
            tk.Radiobutton(outer, text=label, variable=self._theme_var, value=value, **rb).pack(anchor="w", pady=2)

        tk.Frame(outer, height=1, bg=Theme["separator"]).pack(fill="x", pady=(16, 16))

        tk.Label(outer, text="Standardordner", font=("", 13, "bold"),
                 bg=Theme["bg"], fg=Theme["text"]).pack(anchor="w", pady=(0, 8))
        tk.Label(outer, text="Startordner beim Öffnen des Datei-Dialogs.",
                 font=("", 11), bg=Theme["bg"], fg=Theme["text_muted"]).pack(anchor="w", pady=(0, 8))
        folder_row = tk.Frame(outer, bg=Theme["bg"])
        folder_row.pack(fill="x")
        tk.Entry(folder_row, textvariable=self._folder_var, font=("", 11),
                 bg=Theme["bg_elevated"], fg=Theme["text"], insertbackground=Theme["text"],
                 relief="flat", width=34).pack(side="left", ipady=4)
        RoundedButton(folder_row, text="...", command=self._pick_folder,
                      font=("", 11), padx=8, pady=3, **_btn()).pack(side="left", padx=(8, 0))

        btn_frame = tk.Frame(outer, bg=Theme["bg"])
        btn_frame.pack(fill="x", pady=(20, 0))
        RoundedButton(btn_frame, text="Speichern", command=self._apply,
                      font=("", 11), padx=10, pady=5, **_btn()).pack(side="left")
        RoundedButton(btn_frame, text="Abbrechen", command=self.destroy,
                      font=("", 11), padx=10, pady=5, **_btn()).pack(side="left", padx=(10, 0))

    def _pick_folder(self) -> None:
        from tkinter import filedialog
        current = self._folder_var.get()
        initial = current if Path(current).exists() else str(Path(__file__).parent)
        folder = filedialog.askdirectory(initialdir=initial)
        if folder:
            self._folder_var.set(folder)

    def _apply(self) -> None:
        model_changed = self._cfg.get("model") != self._model_var.get()
        theme_changed = self._cfg.get("theme", "discord_dark") != self._theme_var.get()
        self._cfg["ui_scale"] = self._scale_var.get()
        self._cfg["model"] = self._model_var.get()
        self._cfg["theme"] = self._theme_var.get()
        self._cfg["default_folder"] = self._folder_var.get()
        config.save(self._cfg)
        self.destroy()
        self._on_save(model_changed, theme_changed)
