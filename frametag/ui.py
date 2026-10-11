"""Tk interface; worker callbacks are delivered through the main-thread queue."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageOps, ImageTk

from . import __version__
from .api import APIError, Cancelled, DEFAULT_MODELS, diagnose_connection, list_models
from .domain import is_gender_prefixed_filename, select_through
from .model_catalog import model_id, model_label, restore_model
from .engine import Engine
from .models import Photo
from .storage import KeyStore, SettingsStore


class FrameTagApp(tk.Tk):
    def __init__(self, background=False, settings_store=None, key_store=None,
                 engine_factory=Engine):
        super().__init__()
        self.title(f"FrameTag {__version__}")
        self.minsize(1050, 680)
        self.configure(background="#edf5ef")
        self.settings_store = settings_store or SettingsStore()
        self.key_store = key_store or KeyStore()
        self.events = queue.Queue()
        self.engine = engine_factory(lambda name, data: self.events.put((name, data)))
        self._engine_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-control")
        self.photos: list[Photo] = []
        self.by_id: dict[str, Photo] = {}
        self.reference = ""
        self._busy = False
        self._loading = False
        self._auto_pending = False
        self._diagnosing = False
        self._diagnostic_cancel = threading.Event()
        self._api_revision = 0
        self._models_pending = False
        self._loaded_folder = None
        self._closed = False
        self._tray = None
        self._poll_id = None
        self._save_id = None
        self._search_id = None
        self._preview_id = None
        self._preview_token = 0
        self._preview_image = None
        self._preview_tk = None
        self._preview_cache = OrderedDict()
        self._preview_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview")
        self._zoom = None
        self._editor = None
        self._provider_keys = {}
        self._saved_keys = {}
        self._key_dirty = False
        self._loading_key = False
        self._key_save_id = None
        try:
            saved = self.settings_store.load()
        except Exception:
            saved = {}
        self._saved = saved
        self._models = saved.get("models", {}) if isinstance(saved.get("models"), dict) else {}
        provider = saved.get("provider", "NVIDIA")
        if provider not in ("NVIDIA", "OpenAI"):
            provider = "NVIDIA"
        self.folder = tk.StringVar(value=saved.get("folder", ""))
        self.provider = tk.StringVar(value=provider)
        self.model = tk.StringVar(value=restore_model(provider, self._models.get(provider, "")))
        self.model_choice = tk.StringVar(value=model_label(provider, self.model.get()))
        self.api_key = tk.StringVar()
        self.recursive = tk.BooleanVar(value=bool(saved.get("recursive", False)))
        self.auto_watch = tk.BooleanVar(value=False)
        self.close_to_tray = tk.BooleanVar(value=bool(saved.get("close_to_tray", True)))
        self.show_completed = tk.BooleanVar(value=bool(saved.get("show_completed", False)))
        self.search = tk.StringVar()
        self.status = tk.StringVar(value="폴더와 API 키를 선택하세요. 설정은 다음 실행에도 유지됩니다.")
        self.loading_text = tk.StringVar()
        self.count = tk.StringVar(value="0 / 0")
        self.preview_label = tk.StringVar(value="사진을 선택하면 미리보기가 표시됩니다.")
        self.zoom_label = tk.StringVar(value="맞춤")
        self._apply_style()
        self._build()
        try:
            self.geometry(saved.get("geometry", "1450x900"))
        except tk.TclError:
            self.geometry("1450x900")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Control-f>", lambda event: self.search_entry.focus_set())
        self.search.trace_add("write", self._search_changed)
        self.api_key.trace_add("write", self._key_changed)
        self.folder.trace_add("write", self._folder_changed)
        self.model.trace_add("write", self._model_changed)
        self.model_choice.trace_add("write", self._model_choice_changed)
        self._active_provider = provider
        self._load_key(provider)
        self._setup_tray()
        self._poll_id = self.after(80, self._poll)
        self.after(120, self._restore_sash)
        self.after(200, lambda: self._startup(bool(saved.get("auto_watch", False)), background))

    @staticmethod
    def _defaults(provider):
        defaults = DEFAULT_MODELS.get(provider, [])
        if isinstance(defaults, str):
            return [defaults]
        return list(defaults)

    def _model_changed(self, *args):
        self._api_revision += 1
        label = model_label(self.provider.get(), self.model.get())
        if self.model_choice.get() != label:
            self.model_choice.set(label)
        self._schedule_save()

    def _model_choice_changed(self, *args):
        identifier = model_id(self.provider.get(), self.model_choice.get())
        if self.model.get() != identifier:
            self.model.set(identifier)

    def _dispatch_engine(self, action, *args, done=None):
        """Keep filesystem/engine locks off Tk's event thread, in FIFO order."""
        if self._closed:
            return
        def run():
            try:
                if not self._closed:
                    action(*args)
            except Exception as exc:
                self.events.put(("error", str(exc)))
            finally:
                if done:
                    self.events.put((done, None))
        self._engine_pool.submit(run)

    def _update_loading_indicator(self):
        active = self._loading or self._auto_pending or self._diagnosing
        self.loading_text.set("로딩 중 · 사진 목록과 저장된 결과를 불러오고 있습니다." if self._loading else
                              "처리 중 · 자동 감시 설정을 적용하고 있습니다." if self._auto_pending else
                              "API 연결 진단 중 · 서버 응답을 기다리고 있습니다." if self._diagnosing else "")
        if active and str(self.progress["mode"]) != "indeterminate":
            self.progress.configure(mode="indeterminate")
            self.progress.start(40)
        elif not active and str(self.progress["mode"]) == "indeterminate":
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.progress["value"] = 100 * sum(bool(p.analysis) for p in self.photos) / len(self.photos) if self.photos else 0

    def _apply_style(self):
        self.option_add("*Font", ("Segoe UI", 10))
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#edf5ef")
        style.configure("Header.TFrame", background="#e2f0e6")
        style.configure("TLabel", background="#edf5ef", foreground="#254739")
        style.configure("Header.TLabel", background="#e2f0e6")
        style.configure("TCheckbutton", background="#edf5ef", foreground="#254739")
        style.configure("TButton", padding=(9, 7), background="#ffffff", foreground="#315343")
        style.map("TButton", background=[("active", "#d4e8da")])
        style.configure("Accent.TButton", background="#2f6c51", foreground="white")
        style.map("Accent.TButton", background=[("active", "#25583f")])
        style.configure("Treeview", background="white", fieldbackground="white", rowheight=28)
        style.configure("Treeview.Heading", background="#e2eee5", foreground="#254739", padding=8)
        style.map("Treeview", background=[("selected", "#b9d9c5")], foreground=[("selected", "#153526")])

    def _build(self):
        header = ttk.Frame(self, style="Header.TFrame", padding=10)
        header.pack(fill="x", padx=10, pady=(10, 5))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="사진 폴더", style="Header.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.folder_entry = ttk.Entry(header, textvariable=self.folder)
        self.folder_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8), ipady=5)
        self.folder_entry.bind("<Return>", lambda event: self._load_folder())
        self.browse_button = ttk.Button(header, text="폴더 선택", command=self._browse)
        self.browse_button.grid(row=0, column=2, padx=(0, 8))
        self.recursive_check = ttk.Checkbutton(header, text="하위 폴더 포함", variable=self.recursive,
                                               command=self._recursive_changed)
        self.recursive_check.grid(row=0, column=3, sticky="w")
        self.diagnostic_button = ttk.Button(header, text="API 연결 진단", command=self._diagnose_api)
        self.diagnostic_button.grid(row=0, column=4, padx=(5, 0))
        self.api_label = ttk.Label(header, text="NVIDIA API 키", style="Header.TLabel")
        self.api_label.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.key_entry = ttk.Entry(header, textvariable=self.api_key, show="●")
        self.key_entry.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(8, 0), ipady=5)
        self.key_entry.bind("<FocusOut>", lambda event: self._save_key())
        self.provider_combo = ttk.Combobox(header, textvariable=self.provider, values=("NVIDIA", "OpenAI"),
                                           state="readonly", width=10)
        self.provider_combo.grid(row=1, column=2, pady=(8, 0), padx=(0, 8))
        self.provider_combo.bind("<<ComboboxSelected>>", self._provider_changed)
        self.model_combo = ttk.Combobox(header, textvariable=self.model_choice, width=38,
                                       values=[model_label(self.provider.get(), identifier)
                                               for identifier in self._defaults(self.provider.get())])
        self.model_combo.grid(row=1, column=3, pady=(8, 0), padx=(0, 5), sticky="ew")
        self.models_button = ttk.Button(header, text="사용 가능한 모델 확인", command=self._fetch_models)
        self.models_button.grid(row=1, column=4, pady=(8, 0))
        ttk.Button(header, text="설정 저장", command=lambda: self._save_settings(explicit=True)).grid(
            row=1, column=5, padx=(5, 0), pady=(8, 0))

        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=10, pady=4)
        self.load_button = ttk.Button(toolbar, text="목록 불러오기", command=self._load_folder)
        self.load_button.pack(side="left", padx=(0, 5))
        self.analyze_button = ttk.Button(toolbar, text="남은 사진 분석", style="Accent.TButton", command=self._analyze)
        self.analyze_button.pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="선택 셀 수정", command=self._edit_selected).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="선택 사진 기준 통일", command=self._unify).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="현재 사진까지 선택", command=self._select_through).pack(side="left", padx=(0, 5))
        self.apply_button = ttk.Button(toolbar, text="선택 파일명 적용", command=self._apply)
        self.apply_button.pack(side="left", padx=(0, 5))
        self.undo_button = ttk.Button(toolbar, text="마지막 변경 되돌리기", command=self._undo)
        self.undo_button.pack(side="left", padx=(0, 5))
        self.stop_button = ttk.Button(toolbar, text="분석 중지", command=self._stop_analysis, state="disabled")
        self.stop_button.pack(side="right")
        monitor = ttk.Frame(self)
        monitor.pack(fill="x", padx=10, pady=4)
        self.auto_check = ttk.Checkbutton(monitor, text="새 사진 자동 분석·이름 변경", variable=self.auto_watch,
                                          command=self._toggle_auto)
        self.auto_check.pack(side="left", padx=(0, 15))
        ttk.Checkbutton(monitor, text="창 닫을 때 트레이로 숨기기", variable=self.close_to_tray,
                        command=self._schedule_save).pack(side="left")
        ttk.Checkbutton(monitor, text="이름 변경 완료 사진도 표시", variable=self.show_completed,
                        command=self._completed_filter_changed).pack(side="left", padx=(15, 0))
        ttk.Button(monitor, text="백그라운드로 숨기기", command=self._hide).pack(side="right", padx=(5, 0))
        ttk.Button(monitor, text="완전히 종료", command=self._exit).pack(side="right")

        search_bar = ttk.Frame(self)
        search_bar.pack(fill="x", padx=10, pady=(5, 7))
        ttk.Label(search_bar, text="목록 검색").pack(side="left", padx=(0, 8))
        self.search_entry = ttk.Entry(search_bar, textvariable=self.search)
        self.search_entry.pack(side="left", fill="x", expand=True, ipady=5)
        ttk.Label(search_bar, textvariable=self.count).pack(side="left", padx=12)
        ttk.Button(search_bar, text="검색 초기화", command=lambda: self.search.set("")).pack(side="left")

        self.pane = ttk.Panedwindow(self, orient="horizontal")
        self.pane.pack(fill="both", expand=True, padx=10)
        listing = ttk.Frame(self.pane)
        self.pane.add(listing, weight=3)
        columns = ("status", "name", "gender", "clothing", "objects", "proposed_name")
        self.tree = ttk.Treeview(listing, columns=columns, show="headings", selectmode="extended")
        for column, title, width in zip(columns,
                ("상태", "현재 파일명", "성별", "의류", "정물", "새 파일명"),
                (95, 180, 70, 205, 210, 260)):
            self.tree.heading(column, text=title)
            self.tree.column(column, width=width, minwidth=55, stretch=True)
        vscroll = ttk.Scrollbar(listing, orient="vertical", command=self.tree.yview)
        hscroll = ttk.Scrollbar(listing, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll.grid(row=1, column=0, sticky="ew")
        listing.rowconfigure(0, weight=1)
        listing.columnconfigure(0, weight=1)
        self.tree.tag_configure("odd", background="#f3f8f5")
        self.tree.tag_configure("reference", foreground="#227346")
        self.tree.tag_configure("error", foreground="#a03737")
        self.tree.bind("<<TreeviewSelect>>", self._selection_changed)
        self.tree.bind("<Double-1>", self._edit_cell)
        self.tree.bind("<Button-3>", self._context_menu)
        self.tree.bind("<Return>", lambda event: self._edit_selected())
        preview = ttk.Frame(self.pane, padding=8)
        self.pane.add(preview, weight=2)
        preview_top = ttk.Frame(preview)
        preview_top.pack(fill="x", pady=(0, 8))
        ttk.Label(preview_top, text="미리보기", font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Label(preview_top, textvariable=self.zoom_label).pack(side="right", padx=5)
        for text, command in (("+", lambda: self._zoom_by(1.25)),
                              ("맞춤", self._fit), ("−", lambda: self._zoom_by(0.8))):
            ttk.Button(preview_top, text=text, width=5, command=command).pack(side="right", padx=2)
        canvas_frame = ttk.Frame(preview)
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, background="white", highlightthickness=0)
        pvscroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        phscroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=pvscroll.set, xscrollcommand=phscroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        pvscroll.grid(row=0, column=1, sticky="ns")
        phscroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas.bind("<Configure>", lambda event: self._draw_preview())
        self.canvas.bind("<MouseWheel>", lambda event: self._zoom_by(1.12 if event.delta > 0 else 1 / 1.12))
        self.canvas.bind("<Button-4>", lambda event: self._zoom_by(1.12))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_by(1 / 1.12))
        self.canvas.bind("<Double-1>", self._large_preview)
        ttk.Label(preview, textvariable=self.preview_label, wraplength=500).pack(fill="x", pady=(7, 0))
        self.progress = ttk.Progressbar(self, maximum=100, mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=(7, 4))
        ttk.Label(self, textvariable=self.loading_text, padding=(10, 0)).pack(fill="x")
        ttk.Label(self, textvariable=self.status, padding=(10, 4)).pack(fill="x")

    def _startup(self, auto, background):
        if self.folder.get().strip():
            self._load_folder(resume_auto=auto)
        if background:
            self._hide()

    def _restore_sash(self):
        try:
            self.pane.sashpos(0, int(self._saved.get("sash", self.winfo_width() * 0.61)))
        except (ValueError, tk.TclError):
            pass

    def _schedule_save(self, *args):
        if self._closed:
            return
        if self._save_id:
            self.after_cancel(self._save_id)
        self._save_id = self.after(700, self._save_settings)

    def _folder_changed(self, *args):
        if self._loaded_folder is not None:
            try:
                matches = Path(os.path.abspath(self.folder.get().strip())) == self._loaded_folder
            except (OSError, ValueError):
                matches = False
            if not matches and self.auto_watch.get():
                self._dispatch_engine(self.engine.set_auto, False)
                self.auto_watch.set(False)
        self._schedule_save()

    def _save_settings(self, explicit=False):
        if self._save_id:
            self.after_cancel(self._save_id)
        self._save_id = None
        if self._closed:
            return
        key_ok = self._save_key()
        self._configure_engine()
        self._models[self.provider.get()] = self.model.get().strip()
        data = dict(folder=self.folder.get().strip(), recursive=self.recursive.get(),
                    provider=self.provider.get(), models=self._models, auto_watch=self.auto_watch.get(),
                    close_to_tray=self.close_to_tray.get(), show_completed=self.show_completed.get(),
                    geometry=self.geometry(), sash=self.pane.sashpos(0))
        try:
            self.settings_store.save(data)
        except Exception:
            self.status.set("설정을 저장하지 못했습니다. 설정 폴더의 쓰기 권한을 확인하세요.")
            if explicit:
                messagebox.showerror("설정 저장", self.status.get(), parent=self)
            return
        if explicit:
            messagebox.showinfo("설정 저장", "폴더·모델 설정을 저장했습니다." +
                ("\nAPI 키는 Windows 자격 증명 저장소에 보관됩니다." if key_ok else
                 "\nAPI 키를 안전하게 저장하지 못했습니다. 현재 실행 중에는 사용할 수 있습니다."), parent=self)

    def _load_key(self, provider):
        self._loading_key = True
        try:
            if provider in self._provider_keys:
                key = self._provider_keys[provider]
            else:
                try:
                    key = self.key_store.get(provider)
                    self._saved_keys[provider] = key
                except Exception:
                    key = ""
                if not key:
                    key = os.environ.get("NVIDIA_API_KEY" if provider == "NVIDIA" else "OPENAI_API_KEY", "")
                self._provider_keys[provider] = key
            self.api_key.set(key)
            self.api_label.configure(text=f"{provider} API 키")
            self._key_dirty = False
        finally:
            self._loading_key = False

    def _key_changed(self, *args):
        if self._loading_key or self._closed:
            return
        self._key_dirty = True
        self._api_revision += 1
        self._provider_keys[self._active_provider] = self.api_key.get().strip()
        if self._key_save_id:
            self.after_cancel(self._key_save_id)
        self._key_save_id = self.after(900, self._save_key)

    def _save_key(self):
        if self._key_save_id:
            self.after_cancel(self._key_save_id)
            self._key_save_id = None
        if not self._key_dirty:
            key = self.api_key.get().strip()
            return not key or self._saved_keys.get(self._active_provider) == key
        key = self.api_key.get().strip()
        provider = self._active_provider
        # Use the current entry for this session even if the credential store fails.
        if self.provider.get() == provider:
            self._configure_engine()
        try:
            if key:
                self.key_store.set(provider, key)
            else:
                self.key_store.delete(provider)
        except Exception:
            self.status.set("API 키를 안전하게 저장할 수 없습니다. Windows 자격 증명 저장소를 확인하세요.")
            return False
        self._saved_keys[provider] = key
        self._key_dirty = False
        return True

    def _provider_changed(self, event=None):
        self._api_revision += 1
        self._save_key()
        self._models[self._active_provider] = self.model.get().strip()
        self._active_provider = self.provider.get()
        self.model.set(restore_model(self._active_provider, self._models.get(self._active_provider, "")))
        self.model_combo.configure(values=[model_label(self._active_provider, identifier)
                                          for identifier in self._defaults(self._active_provider)])
        self._load_key(self._active_provider)
        self._schedule_save()
        self._configure_engine()

    def _configure_engine(self):
        self._dispatch_engine(self.engine.configure, self.provider.get(), self.model.get().strip(), self.api_key.get().strip())

    def _fetch_models(self):
        if self._models_pending:
            return
        provider, key = self.provider.get(), self.api_key.get().strip()
        if not key:
            messagebox.showinfo("모델 목록", "선택한 공급자의 API 키를 먼저 입력하세요.", parent=self)
            return
        revision = self._api_revision
        self._models_pending = True
        self.models_button.configure(state="disabled", text="모델 확인 중…")
        self.status.set("계정에서 사용할 수 있는 API 모델 목록을 가져오는 중입니다.")
        def fetch():
            try:
                self.events.put(("models", (revision, provider, list_models(provider, key))))
            except APIError as exc:
                self.events.put(("models_error", (revision, str(exc))))
            except Exception:
                self.events.put(("models_error", (revision, "모델 목록을 가져오지 못했습니다. API 설정과 연결을 확인하세요.")))
        threading.Thread(target=fetch, daemon=True, name="model-list").start()

    def _diagnose_api(self):
        if self._diagnosing:
            return
        provider, model, key = self.provider.get(), self.model.get().strip(), self.api_key.get().strip()
        if not key or not model:
            messagebox.showinfo("API 연결 진단", "API 키와 모델을 먼저 입력하세요.", parent=self)
            return
        revision = self._api_revision
        self._diagnosing = True
        self._diagnostic_cancel = threading.Event()
        cancel = self._diagnostic_cancel
        self.diagnostic_button.configure(state="disabled", text="진단 중…")
        self._update_loading_indicator()
        self.status.set("사진 전송 없이 모델 목록과 짧은 텍스트 요청을 확인합니다.")
        def run():
            try:
                report = diagnose_connection(provider, model, key, cancel)
            except Cancelled:
                return
            except APIError as exc:
                report = str(exc)
            except Exception:
                report = "진단을 완료하지 못했습니다. API 공급자와 네트워크 상태를 확인하세요."
            self.events.put(("diagnostic", (revision, report)))
        threading.Thread(target=run, daemon=True, name="api-diagnostic").start()

    def _show_diagnostic(self, report):
        dialog = tk.Toplevel(self)
        dialog.title("FrameTag API 연결 진단")
        dialog.geometry("850x540")
        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill="both", expand=True)
        text = tk.Text(frame, wrap="word", font=("Segoe UI", 10))
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)
        text.insert("1.0", report)
        text.configure(state="disabled")
        def copy():
            self.clipboard_clear()
            self.clipboard_append(report)
            self.status.set("키를 숨긴 API 진단 결과를 복사했습니다.")
        buttons = ttk.Frame(dialog, padding=(10, 0, 10, 10))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="진단 결과 복사", command=copy).pack(side="left")
        ttk.Button(buttons, text="닫기", command=dialog.destroy).pack(side="right")

    def _browse(self):
        if self._busy or self._loading:
            return
        folder = filedialog.askdirectory(parent=self, initialdir=self.folder.get() or None)
        if folder:
            self.folder.set(folder)
            self._load_folder()

    def _recursive_changed(self):
        self._schedule_save()
        if self.folder.get().strip():
            self._load_folder()

    def _load_folder(self, resume_auto=None):
        if self._busy or self._loading:
            self.status.set("분석을 중지한 뒤 폴더를 불러오세요.")
            return
        path = Path(self.folder.get().strip()).expanduser()
        if not self.folder.get().strip():
            messagebox.showerror("사진 폴더", "사진이 있는 폴더를 선택하세요.", parent=self)
            return
        if resume_auto is None:
            resume_auto = self.auto_watch.get()
        self.auto_watch.set(False)
        self._loading = True
        self.reference = ""
        self._update_buttons()
        self.status.set("사진 목록과 저장된 결과를 불러오는 중입니다.")
        recursive = self.recursive.get()
        config = (self.provider.get(), self.model.get().strip(), self.api_key.get().strip())
        def load():
            try:
                self.engine.set_auto(False)
                self.engine.configure(*config)
                if self.engine.load_folder(path, recursive):
                    self.events.put(("loaded", (path.resolve(), resume_auto)))
                else:
                    self.events.put(("load_error", "폴더를 불러오지 못했습니다. 오류를 확인한 뒤 다시 시도하세요."))
            except Exception as exc:
                self.events.put(("load_error", str(exc)))
        self._dispatch_engine(load)

    def _analyze(self):
        if not self._ready():
            return
        self._save_settings()
        self._configure_engine()
        self._dispatch_engine(self.engine.start_analysis)

    def _ready(self):
        if self._loading:
            return False
        if not self.folder.get().strip():
            messagebox.showinfo("사진 폴더", "폴더를 먼저 선택하세요.", parent=self)
            return False
        if not self._folder_matches():
            messagebox.showinfo("사진 폴더", "입력한 폴더의 사진 목록을 먼저 불러오세요.", parent=self)
            return False
        if not self.api_key.get().strip() or not self.model.get().strip():
            messagebox.showinfo("AI 설정", "API 키와 API 모델 ID를 입력하세요.", parent=self)
            return False
        return True

    def _folder_matches(self):
        try:
            return self._loaded_folder is not None and Path(os.path.abspath(self.folder.get().strip())) == self._loaded_folder
        except (OSError, ValueError):
            return False

    def _toggle_auto(self):
        enabled = self.auto_watch.get()
        if enabled and not self._ready():
            self.auto_watch.set(False)
            return
        self._configure_engine()
        self._auto_pending = True
        self._update_buttons()
        self._dispatch_engine(self.engine.set_auto, enabled, done="auto_done")
        self._save_settings()

    def _stop_analysis(self):
        self.auto_watch.set(False)
        self._dispatch_engine(self.engine.stop_analysis)
        self._save_settings()

    def _search_changed(self, *args):
        if self._search_id:
            self.after_cancel(self._search_id)
        self._search_id = self.after(180, self._render_rows)

    def _completed_filter_changed(self):
        self._render_rows()
        self._schedule_save()

    @staticmethod
    def _row_values(photo):
        analysis = photo.analysis or {}
        status = ("이름 기준 제외" if is_gender_prefixed_filename(photo.path.name)
                  and not photo.confirmed and not photo.manual else photo.status)
        return (status, photo.path.name, analysis.get("gender", "-"),
                ", ".join(analysis.get("clothing", [])) or "-",
                ", ".join(analysis.get("objects", [])) or "-", photo.proposed_name or "-")

    def _render_rows(self):
        self._search_id = None
        # Never destroy an active editor when analysis refreshes another row.
        if self._editor is not None:
            return
        selected = set(self.tree.selection())
        focus = self.tree.focus()
        old_y = self.tree.yview()
        words = self.search.get().casefold().split()
        visible = []
        hidden_completed = 0
        self.by_id = {str(p.path): p for p in self.photos}
        for photo in self.photos:
            if (photo.confirmed or is_gender_prefixed_filename(photo.path.name)) and not self.show_completed.get():
                hidden_completed += 1
                continue
            values = self._row_values(photo)
            if all(word in " ".join(map(str, values)).casefold() for word in words):
                visible.append((photo, values))
        wanted = {str(photo.path) for photo, values in visible}
        old_order = self.tree.get_children("")
        order_changed = tuple(str(photo.path) for photo, _ in visible) != old_order
        for iid in old_order:
            if iid not in wanted:
                self.tree.detach(iid)
        # Remove stale rows, including detached rows from a previous folder.
        for iid in getattr(self, "_known_rows", set()) - set(self.by_id):
            if self.tree.exists(iid):
                self.tree.delete(iid)
        self._known_rows = set(self.by_id)
        for index, (photo, values) in enumerate(visible):
            iid = str(photo.path)
            tags = ["odd"] if index % 2 else []
            if iid == self.reference:
                tags.append("reference")
                values = ("★ 기준 · " + values[0], *values[1:])
            if photo.error:
                tags.append("error")
            if self.tree.exists(iid):
                if self.tree.item(iid, "values") != values or self.tree.item(iid, "tags") != tuple(tags):
                    self.tree.item(iid, values=values, tags=tags)
                if order_changed:
                    self.tree.move(iid, "", index)
            else:
                self.tree.insert("", "end", iid=iid, values=values, tags=tags)
        self.tree.selection_set([iid for iid in selected if iid in wanted])
        if focus in wanted:
            self.tree.focus(focus)
        else:
            self.tree.focus("")
            if focus:
                self._preview_token += 1
                self._preview_fingerprint = None
                self._preview_image = None
                self.canvas.delete("all")
                self.preview_label.set("사진을 선택하면 미리보기가 표시됩니다.")
        if old_y:
            self.tree.yview_moveto(old_y[0])
        count = f"{len(visible):,} / {len(self.photos) - hidden_completed:,}"
        if hidden_completed:
            count += f" · 완료 {hidden_completed:,}장 숨김"
        self.count.set(count)
        completed = sum(bool(p.analysis) for p in self.photos)
        if not self._loading and not self._auto_pending:
            self.progress["value"] = 100 * completed / len(self.photos) if self.photos else 0

    def _selection_changed(self, event=None):
        iid = self.tree.focus()
        if iid not in self.by_id:
            selection = self.tree.selection()
            iid = selection[-1] if selection else ""
        photo = self.by_id.get(iid)
        if not photo:
            return
        fingerprint = (str(photo.path), photo.size, photo.mtime_ns)
        if getattr(self, "_preview_fingerprint", None) == fingerprint:
            return
        self._preview_fingerprint = fingerprint
        self._preview_token += 1
        token = self._preview_token
        if self._preview_id:
            self.after_cancel(self._preview_id)
        self._preview_id = self.after(100, lambda: self._request_preview(photo.path, fingerprint, token))
        self.preview_label.set(photo.path.name + (" · " + photo.error if photo.error else ""))

    def _request_preview(self, path, fingerprint, token):
        self._preview_id = None
        if fingerprint in self._preview_cache:
            self._preview_cache.move_to_end(fingerprint)
            self._accept_preview(token, self._preview_cache[fingerprint])
            return
        def read():
            try:
                with Image.open(path) as raw:
                    raw.seek(0)
                    image = ImageOps.exif_transpose(raw).convert("RGB")
                    image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                self.events.put(("preview", (token, fingerprint, image)))
            except Exception:
                self.events.put(("preview_error", token))
        self._preview_pool.submit(read)

    def _accept_preview(self, token, image):
        if token != self._preview_token:
            return
        self._preview_image = image
        self._zoom = None
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self._draw_preview()

    def _fit(self):
        self._zoom = None
        self._draw_preview()

    def _fit_zoom(self):
        if not self._preview_image:
            return 1.0
        width, height = self._preview_image.size
        return min(max(1, self.canvas.winfo_width() - 16) / width,
                   max(1, self.canvas.winfo_height() - 16) / height, 1.0)

    def _zoom_by(self, factor):
        if not self._preview_image:
            return
        self._zoom = max(0.05, min(4.0, (self._zoom or self._fit_zoom()) * factor))
        self._draw_preview()

    def _draw_preview(self):
        if not self._preview_image or self._closed:
            return
        zoom = self._zoom or self._fit_zoom()
        width, height = self._preview_image.size
        width, height = max(1, round(width * zoom)), max(1, round(height * zoom))
        # Limit zoom allocations independently of unusually large source images.
        image = self._preview_image.resize((width, height), Image.Resampling.LANCZOS)
        self._preview_tk = ImageTk.PhotoImage(image, master=self)
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.delete("all")
        self.canvas.create_image(max(0, (canvas_w - width) // 2), max(0, (canvas_h - height) // 2),
                                 image=self._preview_tk, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, max(width, canvas_w), max(height, canvas_h)))
        self.zoom_label.set(f"{zoom:.0%}")

    def _large_preview(self, event=None):
        if not self._preview_image:
            return
        window = tk.Toplevel(self)
        window.title(self.preview_label.get())
        window.geometry("1000x800")
        label = ttk.Label(window, anchor="center")
        label.pack(fill="both", expand=True)
        image = self._preview_image.copy()
        def draw(event=None):
            fitted = image.copy()
            fitted.thumbnail((max(1, window.winfo_width() - 20), max(1, window.winfo_height() - 20)))
            label.image = ImageTk.PhotoImage(fitted, master=window)
            label.configure(image=label.image)
        window.bind("<Configure>", draw)
        window.bind("<Escape>", lambda event: window.destroy())
        window.after(50, draw)

    def _edit_selected(self):
        iid = self.tree.focus()
        if iid:
            self._start_editor(iid, "gender")

    def _edit_cell(self, event):
        iid = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        field = {"#3": "gender", "#4": "clothing", "#5": "objects", "#6": "proposed_name"}.get(column)
        if iid and field:
            self._start_editor(iid, field)

    def _start_editor(self, iid, field):
        photo = self.by_id.get(iid)
        if photo is None:
            return
        if self._editor is not None:
            self._editor.destroy()
            self._editor = None
        box = self.tree.bbox(iid, field)
        if not box:
            return
        analysis = photo.analysis or {}
        value = photo.proposed_name if field == "proposed_name" else analysis.get(field, "")
        if isinstance(value, list):
            value = ", ".join(value)
        text = tk.StringVar(value=value)
        if field == "gender":
            editor = ttk.Combobox(self.tree, textvariable=text,
                                 values=("남성", "여성", "불명", "인물없음"), state="readonly")
        else:
            editor = ttk.Entry(self.tree, textvariable=text)
        self._editor = editor
        editor.place(x=box[0], y=box[1], width=box[2], height=box[3])
        editor.focus_set()
        if isinstance(editor, ttk.Entry):
            editor.select_range(0, "end")
        def finish(commit):
            if self._editor is not editor:
                return
            self._editor = None
            new_value = text.get()
            editor.destroy()
            if commit:
                self._dispatch_engine(self.engine.update_photo, iid, field, new_value)
            self._render_rows()
        editor.bind("<Return>", lambda event: finish(True))
        editor.bind("<Escape>", lambda event: finish(False))
        editor.bind("<FocusOut>", lambda event: finish(True))

    def _context_menu(self, event):
        iid = self.tree.identify_row(event.y)
        if not iid:
            return
        self.tree.focus(iid)
        if iid not in self.tree.selection():
            self.tree.selection_set(iid)
        menu = tk.Menu(self, tearoff=False)
        menu.add_command(label="이 사진을 기준으로 지정", command=lambda: self._set_reference(iid))
        menu.add_command(label="현재 사진까지 선택", command=self._select_through)
        menu.add_separator()
        menu.add_command(label="사진 크게 보기", command=self._large_preview)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _set_reference(self, iid):
        photo = self.by_id.get(iid)
        if photo is None or not photo.analysis:
            messagebox.showinfo("기준 사진", "분석이 완료된 사진을 기준으로 지정하세요.", parent=self)
            return
        self.reference = iid
        self._render_rows()
        self.status.set("★ 기준 사진: " + photo.path.name)

    def _unify(self):
        paths = list(self.tree.selection())
        if len(paths) < 2 or not self.reference:
            messagebox.showinfo("기준 통일", "사진을 두 장 이상 선택하고 기준 사진을 우클릭해 지정하세요.", parent=self)
            return
        self._dispatch_engine(self.engine.unify_selected, paths, self.reference)

    def _select_through(self):
        current = self.tree.focus()
        if not current:
            return
        selected_photo = self.by_id.get(current)
        paths = [str(photo.path) for photo in select_through(self.photos, selected_photo)] if selected_photo else []
        visible = set(self.tree.get_children(""))
        selected = [path for path in paths if path in visible]
        self.tree.selection_set(selected)
        self.status.set(f"확인 완료 사진을 제외한 {len(selected):,}장을 선택했습니다.")

    def _apply(self):
        if not self._folder_matches():
            messagebox.showinfo("사진 폴더", "입력한 폴더의 목록을 먼저 불러오세요.", parent=self)
            return
        paths = list(self.tree.selection())
        if not paths:
            messagebox.showinfo("파일명 적용", "적용할 사진을 먼저 선택하세요.", parent=self)
            return
        self._run_file_action(self.engine.apply_selected, paths)

    def _undo(self):
        if not self._folder_matches():
            messagebox.showinfo("사진 폴더", "입력한 폴더의 목록을 먼저 불러오세요.", parent=self)
            return
        if messagebox.askyesno("마지막 변경 되돌리기", "이 폴더의 마지막 파일명 변경을 되돌릴까요?", parent=self):
            self._run_file_action(self.engine.undo)

    def _run_file_action(self, action, *args):
        if self._busy or self._loading:
            return
        self._busy = True
        self._update_buttons()
        def run():
            try:
                action(*args)
            except Exception as exc:
                self.events.put(("error", f"파일명 변경 오류: {exc}"))
            finally:
                self.events.put(("operation_done", self.engine.is_busy()))
        self._dispatch_engine(run)

    def _update_buttons(self):
        unavailable = self._busy or self._loading or self._auto_pending
        for widget in (self.load_button, self.browse_button, self.apply_button, self.undo_button,
                       self.folder_entry, self.recursive_check):
            widget.configure(state="disabled" if unavailable else "normal")
        self.analyze_button.configure(state="disabled" if unavailable else "normal")
        self.stop_button.configure(state="normal" if self._busy else "disabled")
        self.auto_check.configure(state="disabled" if self._loading or self._auto_pending else "normal")
        self._update_loading_indicator()

    def _poll(self):
        if self._closed:
            return
        latest_photos = None
        for _ in range(100):
            try:
                name, data = self.events.get_nowait()
            except queue.Empty:
                break
            if name == "photos":
                latest_photos = data
            elif name == "status":
                self.status.set(data)
            elif name == "busy":
                self._busy = bool(data)
                self._update_buttons()
            elif name == "operation_done":
                self._busy = bool(data)
                self._update_buttons()
            elif name == "auto_done":
                self._auto_pending = False
                self._update_buttons()
            elif name == "auto":
                self.auto_watch.set(bool(data))
                self._schedule_save()
            elif name == "error":
                self.status.set(data)
                if self.state() != "withdrawn":
                    messagebox.showerror("FrameTag", data, parent=self)
            elif name == "loaded":
                path, resume_auto = data
                self._loading = False
                self._loaded_folder = path.resolve()
                self.folder.set(str(path.resolve()))
                self._update_buttons()
                if resume_auto and self._ready():
                    self._configure_engine()
                    self._dispatch_engine(self.engine.set_auto, True)
                    self.auto_watch.set(True)
                self._save_settings()
            elif name == "load_error":
                self._loading = False
                self._update_buttons()
                self.status.set(data)
            elif name == "models":
                revision, provider, models = data
                self._models_pending = False
                self.models_button.configure(state="normal", text="사용 가능한 모델 확인")
                if revision == self._api_revision and provider == self.provider.get():
                    self.model_combo.configure(values=[model_label(provider, identifier) for identifier in models])
                    if models and self.model.get() not in models:
                        self.model.set(models[0])
                    self.status.set(f"계정 목록에서 사진 분석 후보 {len(models)}개를 확인했습니다." if models else
                                    "계정 목록에 추천 사진 모델이 없습니다. NVIDIA 계정의 모델 권한을 확인하세요.")
                else:
                    self.status.set("설정이 바뀌어 이전 모델 조회 결과를 적용하지 않았습니다. 다시 확인하세요.")
            elif name == "models_error":
                revision, message = data
                self._models_pending = False
                self.models_button.configure(state="normal", text="사용 가능한 모델 확인")
                if revision == self._api_revision:
                    self.status.set(message)
                    messagebox.showerror("모델 목록", message, parent=self)
            elif name == "diagnostic":
                revision, report = data
                self._diagnosing = False
                self.diagnostic_button.configure(state="normal", text="API 연결 진단")
                self._update_loading_indicator()
                if revision != self._api_revision:
                    report = "진단 도중 설정이 변경되었습니다. 아래 결과는 진단을 시작할 때의 설정 기준입니다.\n\n" + report
                self.status.set("API 진단이 끝났습니다. 결과 창에서 서버 설명을 확인하세요.")
                self._show_diagnostic(report)
            elif name == "preview":
                token, fingerprint, image = data
                self._preview_cache[fingerprint] = image
                while len(self._preview_cache) > 8:
                    self._preview_cache.popitem(last=False)
                self._accept_preview(token, image)
            elif name == "preview_error" and data == self._preview_token:
                self._preview_image = None
                self.canvas.delete("all")
                self.preview_label.set("사진을 읽을 수 없습니다. 파일이 복사 중인지 확인하세요.")
            elif name == "show":
                self._show()
            elif name == "hide":
                self._hide()
            elif name == "tray_pause":
                self._stop_analysis()
            elif name == "exit":
                self._exit()
                return
        if latest_photos is not None:
            self.photos = latest_photos
            self._render_rows()
        self._poll_id = self.after(80, self._poll)

    def _setup_tray(self):
        # Windows is the supported desktop target. No import-time display dependency.
        if os.name != "nt":
            return
        try:
            import pystray
            icon = Image.new("RGBA", (64, 64), "#2f6c51")
            draw = ImageDraw.Draw(icon)
            draw.rounded_rectangle((10, 17, 54, 48), radius=5, outline="white", width=4)
            draw.ellipse((24, 22, 43, 42), outline="white", width=4)
            menu = pystray.Menu(
                pystray.MenuItem("FrameTag 열기", lambda *args: self.events.put(("show", None)), default=True),
                pystray.MenuItem("자동 감시·분석 중지", lambda *args: self.events.put(("tray_pause", None))),
                pystray.MenuItem("완전히 종료", lambda *args: self.events.put(("exit", None))))
            self._tray = pystray.Icon("FrameTag", icon, "FrameTag · 사진 폴더 감시", menu)
            self._tray.run_detached()
        except Exception:
            self._tray = None
            self.status.set("트레이를 시작하지 못했습니다. 창을 열어 둔 상태로 사용할 수 있습니다.")

    def _show(self):
        self.deiconify()
        self.lift()
        self.focus_force()

    def _hide(self):
        if not self._tray:
            messagebox.showinfo("백그라운드 실행", "Windows 트레이를 사용할 수 없습니다. 창을 최소화해 사용하세요.", parent=self)
            self.iconify()
            return
        self._save_settings()
        self.withdraw()

    def _on_close(self):
        if self.close_to_tray.get() and self._tray:
            self._hide()
        else:
            self._exit()

    def _exit(self):
        if self._closed:
            return
        self._save_settings()
        self._closed = True
        self._engine_pool.shutdown(wait=False, cancel_futures=True)
        self._diagnostic_cancel.set()
        self.progress.stop()
        for after_id in (self._poll_id, self._save_id, self._search_id, self._preview_id, self._key_save_id):
            if after_id:
                try:
                    self.after_cancel(after_id)
                except tk.TclError:
                    pass
        self.engine.shutdown()
        self._preview_pool.shutdown(wait=False, cancel_futures=True)
        if self._tray:
            self._tray.stop()
        self.destroy()
