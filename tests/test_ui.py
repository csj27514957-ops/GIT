"""Desktop integration checks; skipped when no display is available."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image

from frametag.api import APIError, Cancelled, ModelDiscovery
from frametag.engine import Engine
from frametag.models import Photo
from frametag.model_catalog import NVIDIA_DISCOVERABLE, NVIDIA_UNVERIFIED_LABEL, model_label
from frametag.storage import Cache, SettingsStore
from frametag.ui import FrameTagApp


class FakeKeys:
    def __init__(self):
        self.keys = {}

    def get(self, provider):
        return self.keys.get(provider, "")

    def set(self, provider, key):
        self.keys[provider] = key

    def delete(self, provider):
        self.keys.pop(provider, None)


class FakeAnalyzer:
    calls = []

    def __init__(self, provider, model, key):
        pass

    def analyze(self, path, cancel):
        self.calls.append(path.name)
        return {"gender": "불명", "clothing": ["흰색반팔"], "objects": []}


@unittest.skipUnless(os.environ.get("DISPLAY") or os.name == "nt", "A desktop display is required")
class UITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = self.root / "photos"
        self.folder.mkdir()
        for number in range(1, 4):
            Image.new("RGB", (40, 60), (number * 50, 90, 130)).save(self.folder / f"photo{number}.jpg")
        self.settings = SettingsStore(self.root / "config")
        self.keys = FakeKeys()
        FakeAnalyzer.calls = []
        self.errors = []
        environment = patch.dict(os.environ, {"NVIDIA_API_KEY": "", "OPENAI_API_KEY": ""})
        environment.start()
        self.addCleanup(environment.stop)
        for target, arguments in (
            ("frametag.ui.FrameTagApp._setup_tray", {"return_value": None}),
            ("frametag.ui.messagebox.showerror", {"side_effect": lambda *a, **kw: self.errors.append(a)}),
            ("frametag.ui.messagebox.showinfo", {"return_value": None}),
            ("frametag.ui.messagebox.askyesno", {"return_value": True}),
        ):
            mock = patch(target, **arguments)
            mock.start()
            self.addCleanup(mock.stop)
        self.apps = []
        self.addCleanup(self._close_apps)
        self.app = self._new_app()
        # Let the automatic startup callback finish before setting the test folder.
        deadline = time.monotonic() + .25
        self.wait(lambda: time.monotonic() >= deadline)
        self.app.folder.set(str(self.folder))
        self.app.api_key.set("mock-nvidia-key-not-valid")
        self._verify_model_for_test()
        self.app._load_folder()
        self.wait(lambda: len(self.app.photos) == 3 and not self.app._loading)

    def _new_app(self):
        app = FrameTagApp(settings_store=self.settings, key_store=self.keys,
                         engine_factory=lambda callback: Engine(
                             callback, interval=.03, stable_scans=2, analyzer_factory=FakeAnalyzer))
        self.apps.append(app)
        return app

    def _close_apps(self):
        for app in self.apps:
            if not app._closed:
                app._exit()

    def _verify_model_for_test(self, model="test-vision-model"):
        self.app.model.set(model)
        self.app._verified_models[("NVIDIA", self.app.api_key.get().strip())] = {model}

    def wait(self, predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.update()
            if predicate():
                return
            time.sleep(.01)
        self.fail(f"GUI wait timed out: {self.app.status.get()}")

    def test_review_selection_apply_and_undo_preserve_photo_bytes(self):
        hashes = sorted(hashlib.sha256(path.read_bytes()).hexdigest() for path in self.folder.glob("*.jpg"))
        self.app._analyze()
        self.wait(lambda: all(photo.analysis for photo in self.app.photos) and not self.app._busy)
        first, second, third = [str(photo.path) for photo in self.app.photos]
        self.app.engine.update_photo(first, "objects", "우산, 종이(바닥)")
        self.wait(lambda: self.app.by_id[first].manual)
        self.app.tree.selection_set(first, second)
        self.app._set_reference(first)
        self.app._unify()
        self.wait(lambda: self.app.by_id[second].manual)
        self.app.tree.focus(second)
        self.app._select_through()
        self.assertEqual(set(self.app.tree.selection()), {first, second})
        self.app._apply()
        self.wait(lambda: sum(photo.confirmed for photo in self.app.photos) == 2)
        self.app.tree.focus(third)
        self.app._select_through()
        self.assertEqual(self.app.tree.selection(), (third,))
        self.app._undo()
        self.wait(lambda: not any(photo.confirmed for photo in self.app.photos))
        self.assertEqual(sorted(path.name for path in self.folder.glob("*.jpg")),
                         ["photo1.jpg", "photo2.jpg", "photo3.jpg"])
        self.assertEqual(sorted(hashlib.sha256(path.read_bytes()).hexdigest() for path in self.folder.glob("*.jpg")), hashes)
        self.app.search.set("우산 종이")
        self.wait(lambda: self.app.count.get() == "2 / 3")
        self.assertFalse(self.errors)

    def test_folder_provider_and_keys_restore_without_plaintext_settings(self):
        self.app._save_settings()
        self.app.provider.set("OpenAI")
        self.app._provider_changed()
        self.assertEqual(self.app.api_key.get(), "")
        self.app.api_key.set("mock-openai-key-not-valid")
        self.app.model.set("mock-vision-model")
        self.app._save_settings()
        self.app._exit()
        self.app = self._new_app()
        self.assertEqual(self.app.provider.get(), "OpenAI")
        self.assertEqual(self.app.model.get(), "mock-vision-model")
        self.assertEqual(self.app.api_key.get(), "mock-openai-key-not-valid")
        self.assertEqual(self.app.folder.get(), str(self.folder))
        self.wait(lambda: len(self.app.photos) == 3 and not self.app._loading)
        self.app.provider.set("NVIDIA")
        self.app._provider_changed()
        self.assertEqual(self.app.api_key.get(), "mock-nvidia-key-not-valid")
        settings_text = self.settings.path.read_text(encoding="utf-8")
        self.assertEqual(json.loads(settings_text)["folder"], str(self.folder))
        self.assertNotIn("mock-nvidia-key-not-valid", settings_text)
        self.assertNotIn("mock-openai-key-not-valid", settings_text)
        self.assertFalse(self.errors)

    def test_new_key_is_used_when_analysis_starts_before_save_debounce(self):
        new_key = "mock-replacement-nvidia-key-not-valid"
        captured = []

        def factory(provider, model, key):
            captured.append((provider, model, key))
            return FakeAnalyzer(provider, model, key)

        with patch.object(self.app.engine, "_factory", side_effect=factory):
            self.app.api_key.set(new_key)
            self._verify_model_for_test()
            self.app._analyze()
            self.wait(lambda: all(photo.analysis for photo in self.app.photos) and not self.app._busy)
        self.assertEqual(len(captured), 3)
        self.assertTrue(all(provider == "NVIDIA" and key == new_key
                            for provider, _model, key in captured))
        self.assertEqual(self.keys.get("NVIDIA"), new_key)
        self.assertFalse(self.errors)

    def test_model_discovery_uses_entered_key_before_save_debounce(self):
        new_key = "mock-discovery-nvidia-key-not-valid"
        available = list(NVIDIA_DISCOVERABLE)[:3]
        result = ModelDiscovery(available, [], available, [])
        preferred = self.app.model.get()
        with patch("frametag.ui.discover_photo_models", return_value=result) as lookup:
            self.app.api_key.set(new_key)
            self.app._fetch_models()
            self.wait(lambda: lookup.called and not self.app._models_pending)
        self.assertEqual(lookup.call_count, 1)
        self.assertEqual(lookup.call_args.args[:2], ("NVIDIA", new_key))
        self.assertEqual(lookup.call_args.args[2], preferred)
        self.assertIsInstance(lookup.call_args.args[3], threading.Event)
        self.assertTrue(callable(lookup.call_args.kwargs["progress"]))
        self.assertIn("excluded", lookup.call_args.kwargs)
        self.assertEqual(self.app._verified_models[("NVIDIA", new_key)], set(available))
        self.assertFalse(self.errors)

    def test_failed_key_persistence_still_updates_current_session_key(self):
        old_key = self.app.api_key.get()
        self.wait(lambda: self.app.engine._config[2] == old_key)
        # Exercise the key-only debounce, without an unrelated settings save
        # also configuring the engine and concealing a persistence failure.
        if self.app._save_id:
            self.app.after_cancel(self.app._save_id)
            self.app._save_id = None
        new_key = "mock-unsaved-nvidia-key-not-valid"
        with patch.object(self.keys, "set", side_effect=RuntimeError("Credential store unavailable")) as save:
            self.app.api_key.set(new_key)
            self.wait(lambda: save.called and self.app.engine._config[2] == new_key)
        save.assert_called_once_with("NVIDIA", new_key)
        self.assertEqual(self.keys.get("NVIDIA"), old_key)
        self.assertTrue(self.app._key_dirty)
        self.assertNotIn(new_key, self.settings.path.read_text(encoding="utf-8"))
        self.assertFalse(self.errors)

    def test_retired_model_restores_blank_and_verified_choices_save_actual_api_id(self):
        self.app._save_settings()
        self.app._exit()
        saved = self.settings.load()
        saved["models"]["NVIDIA"] = "qwen/qwen3.5-122b-a10b"
        self.settings.save(saved)
        self.app = self._new_app()
        self.assertEqual(self.app.model.get(), "")
        self.assertEqual(self.app.model_choice.get(), NVIDIA_UNVERIFIED_LABEL)
        self.wait(lambda: len(self.app.photos) == 3 and not self.app._loading)
        available = list(NVIDIA_DISCOVERABLE)[:3]
        result = ModelDiscovery(available, [], available, [])
        with patch("frametag.ui.discover_photo_models", return_value=result):
            self.app._fetch_models()
            self.wait(lambda: self.app.model.get() == available[0])
        self.assertEqual(list(self.app.model_combo["values"]), [model_label("NVIDIA", name) for name in available])
        self.app.model_choice.set(model_label("NVIDIA", available[1]))
        self.app._save_settings()
        self.assertEqual(self.settings.load()["models"]["NVIDIA"], available[1])
        self.assertFalse(self.errors)

    def test_nvidia_analysis_and_auto_watch_require_current_key_verification(self):
        self.app._verified_models.clear()
        with patch("frametag.ui.discover_photo_models", return_value=ModelDiscovery([], [], [], [])) as lookup:
            for selected in ("", next(iter(NVIDIA_DISCOVERABLE))):
                with self.subTest(selected=selected):
                    self.app.model.set(selected)
                    self.assertFalse(self.app._ready())
                    self.app._analyze()
                    self.app.auto_watch.set(True)
                    self.app._toggle_auto()
                    self.assertFalse(self.app.auto_watch.get())
            self.wait(lambda: not self.app._models_pending)
            self.assertGreaterEqual(lookup.call_count, 1)
        self.assertFalse(self.app.engine.is_busy())
        self.assertEqual(FakeAnalyzer.calls, [])
        self.assertFalse(self.errors)

    def test_discovery_offers_only_live_verified_models_before_photo_analysis(self):
        available = list(NVIDIA_DISCOVERABLE)[:2]
        result = ModelDiscovery([available[1]], [], available + ["text-only-model"], [])
        self.app._verified_models.clear()
        with patch("frametag.ui.discover_photo_models", return_value=result):
            self.app._fetch_models()
            self.wait(lambda: not self.app._models_pending and self.app.model.get() == available[1])
        self.assertEqual(list(self.app.model_combo["values"]), [model_label("NVIDIA", available[1])])
        self.assertEqual(self.app._verified_models[("NVIDIA", self.app.api_key.get())], {available[1]})
        self.assertEqual(FakeAnalyzer.calls, [])
        self.assertTrue(self.app._ready())
        self.app._analyze()
        self.wait(lambda: len(FakeAnalyzer.calls) == 3 and not self.app._busy)
        self.assertFalse(self.errors)

    def test_empty_verified_result_clears_stale_model_and_prevents_photo_calls(self):
        catalog = list(NVIDIA_DISCOVERABLE)[:2]
        result = ModelDiscovery([], [], catalog, ["테스트 이미지 확인 실패"])
        with patch("frametag.ui.discover_photo_models", return_value=result):
            self.app._fetch_models()
            self.wait(lambda: not self.app._models_pending)
        self.assertEqual(self.app.model.get(), "")
        self.assertEqual(self.app.model_choice.get(), NVIDIA_UNVERIFIED_LABEL)
        self.assertFalse(self.app._verified_models.get(("NVIDIA", self.app.api_key.get())))
        with patch("frametag.ui.discover_photo_models", return_value=result):
            self.assertFalse(self.app._ready())
            self.app._analyze()
            self.wait(lambda: not self.app._models_pending)
        self.assertEqual(FakeAnalyzer.calls, [])
        self.assertFalse(self.errors)

    def test_partial_model_verification_report_retains_later_failures_and_redacts_key(self):
        available = list(NVIDIA_DISCOVERABLE)[:3]
        key = self.app.api_key.get()
        notes = ["첫 번째 후보의 이미지 입력을 확인하지 못했습니다.",
                 "두 번째 후보의 테스트 색상 확인이 실패했습니다.",
                 f"마지막 후보 오류: unsupported image input; credential={key}"]
        result = ModelDiscovery([available[0]], [], available, notes)
        with patch("frametag.ui.discover_photo_models", return_value=result) as discovery:
            self.app._fetch_models()
            self.wait(lambda: not self.app._models_pending)
        dialog = next(widget for widget in self.app.winfo_children()
                      if widget.winfo_class() == "Toplevel"
                      and widget.title() == "FrameTag 사진 모델 확인 결과")
        self.assertFalse(dialog.grab_current())
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        text = next(widget for widget in descendants(dialog) if widget.winfo_class() == "Text")
        report = text.get("1.0", "end-1c")
        self.assertIn("공급자: NVIDIA", report)
        self.assertIn(available[0], report)
        self.assertIn(notes[0], report)
        self.assertIn(notes[1], report)
        self.assertIn("마지막 후보 오류: unsupported image input", report)
        self.assertIn("[API 키 숨김]", report)
        self.assertNotIn(key, report)
        self.assertNotIn(str(self.folder), report)
        self.assertNotIn("photo1.jpg", report)
        self.assertNotIn("마지막 후보 오류", self.app.status.get())
        copy_button = next(widget for widget in descendants(dialog)
                           if widget.winfo_class() == "TButton"
                           and widget.cget("text") == "모델 확인 결과 복사")
        copy_button.invoke()
        self.assertEqual(self.app.clipboard_get(), report)
        self.assertEqual(self.app.model.get(), available[0])
        self.assertEqual(list(self.app.model_combo["values"]), [model_label("NVIDIA", available[0])])
        self.assertTrue(self.app._ready())
        self.assertEqual(discovery.call_count, 1)
        self.assertEqual(FakeAnalyzer.calls, [])
        self.assertFalse(self.errors)
        dialog.destroy()

    def test_empty_model_verification_opens_copyable_report_without_modal_error(self):
        result = ModelDiscovery([], [], list(NVIDIA_DISCOVERABLE)[:2], [])
        with patch("frametag.ui.discover_photo_models", return_value=result) as discovery, \
                patch.object(self.app, "_show_diagnostic") as show:
            self.app._fetch_models()
            self.wait(lambda: not self.app._models_pending)
        self.assertEqual(show.call_count, 1)
        report = show.call_args.args[0]
        self.assertIn("공급자: NVIDIA", report)
        self.assertIn("통과한 모델이 없습니다.", report)
        self.assertEqual(show.call_args.kwargs["title"], "FrameTag 사진 모델 확인 결과")
        self.assertEqual(show.call_args.kwargs["copy_label"], "모델 확인 결과 복사")
        self.assertNotIn(self.app.api_key.get(), report)
        self.assertEqual(self.app.model.get(), "")
        self.assertEqual(list(self.app.model_combo["values"]), [])
        self.assertEqual(discovery.call_count, 1)
        self.assertEqual(FakeAnalyzer.calls, [])
        self.assertFalse(self.errors)

    def test_lookup_error_connection_failure_report_has_model_path_and_no_key(self):
        selected, key = self.app.model.get(), self.app.api_key.get()
        endpoint = "https://integrate.api.nvidia.com/v1/models"
        message = ("API 연결에 실패했습니다. 인터넷 연결과 서버 상태를 확인하세요.\n"
                   f"요청: GET {endpoint}\n연결 오류 종류: ConnectError; credential={key}")
        with patch("frametag.ui.discover_photo_models", side_effect=APIError(message)) as lookup, \
                patch.object(self.app, "_show_diagnostic") as show:
            self.app._fetch_models()
            self.wait(lambda: not self.app._models_pending)
        self.assertEqual(show.call_count, 1)
        report = show.call_args.args[0]
        self.assertIn("통과한 모델이 없습니다.", report)
        self.assertIn(selected, report)
        self.assertIn(endpoint, report)
        self.assertIn("ConnectError", report)
        self.assertNotIn(key, report)
        self.assertNotIn(str(self.folder), report)
        self.assertNotIn("photo1.jpg", report)
        self.assertEqual(show.call_args.kwargs["copy_label"], "모델 확인 결과 복사")
        self.assertEqual(lookup.call_count, 1)
        self.assertFalse(self.app._verified_models.get(("NVIDIA", key)))
        with patch.object(self.app, "_fetch_models"):
            self.assertFalse(self.app._ready())
        self.assertFalse(self.app.auto_watch.get())
        self.assertFalse(self.app.engine._auto)
        self.assertEqual(FakeAnalyzer.calls, [])
        self.assertFalse(self.errors)

    def test_lookup_error_after_key_change_does_not_open_a_report(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def lookup(*args, **kwargs):
            entered.set()
            release.wait(2)
            raise APIError("이전 키의 API 연결에 실패했습니다.")

        with patch("frametag.ui.discover_photo_models", side_effect=lookup), \
                patch.object(self.app, "_show_model_verification") as show:
            self.app._fetch_models()
            self.wait(entered.is_set)
            self.app.api_key.set("mock-changed-before-lookup-error")
            release.set()
            self.wait(lambda: not self.app._models_pending)
            show.assert_not_called()
        self.assertEqual(FakeAnalyzer.calls, [])
        self.assertFalse(self.errors)

    def test_lookup_error_cancelled_before_queue_poll_does_not_open_a_report(self):
        with patch("frametag.ui.discover_photo_models", side_effect=APIError("API 연결에 실패했습니다.")), \
                patch.object(self.app, "_show_model_verification") as show:
            self.app._fetch_models()
            deadline = time.monotonic() + 1
            while True:
                with self.app.events.mutex:
                    queued = any(name == "models_error" for name, _ in self.app.events.queue)
                if queued or time.monotonic() >= deadline:
                    break
                time.sleep(.005)
            self.assertTrue(queued)
            self.app._fetch_models()
            self.wait(lambda: not self.app._models_pending)
            show.assert_not_called()
        self.assertEqual(str(self.app.models_button["state"]), "normal")
        self.assertEqual(FakeAnalyzer.calls, [])
        self.assertFalse(self.errors)

    def test_startup_auto_watch_checks_saved_key_before_enable_and_new_photo_analysis(self):
        saved_key = self.app.api_key.get()
        self.app.auto_watch.set(True)
        self.app._save_settings()
        self.app._exit()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        available = ["test-vision-model"]

        def lookup(provider, key, preferred, cancel, **kwargs):
            entered.set()
            release.wait(3)
            return ModelDiscovery(available, [], available, [])

        with patch("frametag.ui.discover_photo_models", side_effect=lookup) as discovery:
            self.app = self._new_app()
            self.wait(entered.is_set)
            self.assertFalse(self.app.auto_watch.get())
            self.assertFalse(self.app.engine._auto)
            self.assertEqual(FakeAnalyzer.calls, [])
            self.assertEqual(discovery.call_args.args[:2], ("NVIDIA", saved_key))
            release.set()
            self.wait(lambda: self.app.auto_watch.get() and self.app.engine._auto)
        incoming = self.folder / "incoming.jpg"
        Image.new("RGB", (45, 65), "white").save(incoming)
        self.wait(lambda: sum(photo.confirmed for photo in self.app.photos) == 1
                  and not self.app._busy)
        self.assertEqual(FakeAnalyzer.calls, ["incoming.jpg"])
        self.assertTrue(all((self.folder / f"photo{number}.jpg").exists()
                            for number in range(1, 4)))
        self.assertFalse(self.errors)

    def test_startup_does_not_enable_auto_with_a_different_verified_model(self):
        self.app.auto_watch.set(True)
        self.app._save_settings()
        self.app._exit()
        replacement = next(iter(NVIDIA_DISCOVERABLE))
        result = ModelDiscovery([replacement], [], [replacement], [])
        with patch("frametag.ui.discover_photo_models", return_value=result) as discovery:
            self.app = self._new_app()
            self.wait(lambda: discovery.called and not self.app._models_pending
                      and self.app.model.get() == replacement)
        self.assertFalse(self.app.auto_watch.get())
        self.assertFalse(self.app.engine._auto)
        self.assertEqual(FakeAnalyzer.calls, [])
        self.assertTrue(all((self.folder / f"photo{number}.jpg").exists()
                            for number in range(1, 4)))
        self.assertFalse(self.errors)

    def test_server_retired_model_persists_and_is_not_offered_after_restart(self):
        available, retired = list(NVIDIA_DISCOVERABLE)[:2]
        self._verify_model_for_test(retired)
        result = ModelDiscovery([available], [retired], [available, retired], [])
        with patch("frametag.ui.discover_photo_models", return_value=result):
            self.app._fetch_models()
            self.wait(lambda: not self.app._models_pending and self.app.model.get() == available)
        self.app._save_settings()
        self.assertIn(retired, self.settings.load()["retired_nvidia_models"])
        self.app._exit()
        # A stale preference must not resurrect a model the server has retired.
        saved = self.settings.load()
        saved["models"]["NVIDIA"] = retired
        self.settings.save(saved)
        self.app = self._new_app()
        self.assertEqual(self.app.model.get(), "")
        self.assertEqual(self.app.model_choice.get(), NVIDIA_UNVERIFIED_LABEL)
        self.wait(lambda: len(self.app.photos) == 3 and not self.app._loading)
        with patch("frametag.ui.discover_photo_models", return_value=result) as discovery:
            self.app._fetch_models()
            self.wait(lambda: not self.app._models_pending and self.app.model.get() == available)
        self.assertIn(retired, discovery.call_args.kwargs["excluded"])
        self.assertNotIn(model_label("NVIDIA", retired), self.app.model_combo["values"])
        self.assertNotIn(retired, self.app._verified_models[("NVIDIA", self.app.api_key.get())])
        self.assertFalse(self.errors)

    def test_model_verification_animates_and_can_be_cancelled_without_blocking_gui(self):
        entered = threading.Event()

        def lookup(provider, key, preferred, cancel, **kwargs):
            entered.set()
            if not cancel.wait(3):
                raise AssertionError("Model verification was not cancelled")
            raise Cancelled()

        with patch("frametag.ui.discover_photo_models", side_effect=lookup):
            started = time.monotonic()
            self.app._fetch_models()
            self.assertLess(time.monotonic() - started, .25)
            self.wait(entered.is_set)
            self.assertTrue(self.app._models_pending)
            self.assertIn("모델 확인 중", self.app.loading_text.get())
            self.assertEqual(str(self.app.progress["mode"]), "indeterminate")
            self.assertEqual(self.app.models_button["text"], "모델 확인 중지")
            initial_progress = float(self.app.progress["value"])
            heartbeat = []
            self.app.after(30, lambda: heartbeat.append(True))
            self.wait(lambda: bool(heartbeat), timeout=.5)
            self.wait(lambda: float(self.app.progress["value"]) > initial_progress, timeout=.5)
            started = time.monotonic()
            self.app.models_button.invoke()
            self.assertLess(time.monotonic() - started, .25)
            self.assertTrue(self.app._model_cancel.is_set())
            self.assertEqual(str(self.app.models_button["state"]), "disabled")
            self.wait(lambda: not self.app._models_pending)
        self.assertEqual(str(self.app.models_button["state"]), "normal")
        self.assertEqual(str(self.app.progress["mode"]), "determinate")
        self.assertEqual(self.app.loading_text.get(), "")
        self.assertEqual(FakeAnalyzer.calls, [])
        self.assertFalse(self.errors)

    def test_cancelled_lookup_does_not_apply_a_result_already_waiting_in_queue(self):
        original = self.app.model.get()
        replacement = next(iter(NVIDIA_DISCOVERABLE))
        result = ModelDiscovery([replacement], [], [replacement], ["취소한 모델 확인의 실패 내용"])
        with patch("frametag.ui.discover_photo_models", return_value=result), \
                patch.object(self.app, "_show_model_verification") as show:
            self.app._fetch_models()
            deadline = time.monotonic() + 1
            while True:
                with self.app.events.mutex:
                    queued = any(name == "models" for name, _ in self.app.events.queue)
                if queued or time.monotonic() >= deadline:
                    break
                time.sleep(.005)
            self.assertTrue(queued)
            self.app._fetch_models()  # Cancel before Tk consumes the ready result.
            self.wait(lambda: not self.app._models_pending)
            show.assert_not_called()
        self.assertEqual(self.app.model.get(), original)
        self.assertFalse(self.app._verified_models.get(("NVIDIA", self.app.api_key.get())))
        self.assertEqual(FakeAnalyzer.calls, [])
        self.assertIn("중지", self.app.status.get())

    def test_model_retirement_during_real_analysis_clears_and_persists_selection(self):
        retired = next(iter(NVIDIA_DISCOVERABLE))
        self._verify_model_for_test(retired)
        class RetiredAnalyzer:
            def analyze(_self, path, cancel):
                raise APIError("The selected model has reached its end of life", fatal=True,
                               status_code=410, model_retired=retired)
        with patch.object(self.app.engine, "_factory", return_value=RetiredAnalyzer()):
            self.app._analyze()
            self.wait(lambda: self.app.model.get() == "" and not self.app._busy)
        self.app._save_settings()
        self.assertIn(retired, self.settings.load()["retired_nvidia_models"])
        self.assertNotIn(retired, self.app._verified_models[("NVIDIA", self.app.api_key.get())])
        self.assertTrue(all(photo.path.exists() for photo in self.app.photos))
        self.assertEqual(sum(photo.status == "오류" for photo in self.app.photos), 1)
        self.assertEqual(sum(photo.status == "대기" for photo in self.app.photos), 2)
        self.assertTrue(self.errors)

    def test_api_diagnostic_uses_new_key_and_gui_animates_while_server_waits(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        captured = []
        report = "모델 목록 HTTP 200\n분석 경로 HTTP 410\n서버 설명: Inference route disabled"
        def diagnostic(provider, model, key, cancel):
            captured.append((provider, model, key, cancel))
            entered.set()
            release.wait(2)
            return report
        new_key = "mock-diagnostic-new-key-not-valid"
        self.app.api_key.set(new_key)
        with patch("frametag.ui.diagnose_connection", side_effect=diagnostic), \
                patch.object(self.app, "_show_diagnostic") as show:
            start = time.monotonic()
            self.app._diagnose_api()
            self.assertLess(time.monotonic() - start, .25)
            self.wait(entered.is_set)
            self.assertEqual(captured[0][:3], ("NVIDIA", self.app.model.get(), new_key))
            self.assertTrue(self.app._diagnosing)
            self.assertIn("진단 중", self.app.loading_text.get())
            self.assertEqual(str(self.app.progress["mode"]), "indeterminate")
            initial_progress = float(self.app.progress["value"])
            heartbeat = []
            self.app.after(30, lambda: heartbeat.append(True))
            self.wait(lambda: bool(heartbeat), timeout=.5)
            self.wait(lambda: float(self.app.progress["value"]) > initial_progress, timeout=.5)
            self.assertFalse(release.is_set())
            release.set()
            self.wait(lambda: not self.app._diagnosing)
            show.assert_called_once_with(report)
        self.assertEqual(str(self.app.diagnostic_button["state"]), "normal")
        self.assertEqual(self.app.loading_text.get(), "")
        self.assertFalse(self.errors)

    def test_old_key_discovery_cannot_change_model_after_key_is_replaced(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        original_model = self.app.model.get()
        available = list(NVIDIA_DISCOVERABLE)[:1]
        def lookup(provider, key, preferred, cancel, **kwargs):
            entered.set()
            release.wait(2)
            return ModelDiscovery(available, [], available, ["이전 키의 모델 확인 실패"])
        with patch("frametag.ui.discover_photo_models", side_effect=lookup), \
                patch.object(self.app, "_show_model_verification") as show:
            self.app._fetch_models()
            self.wait(entered.is_set)
            self.app.api_key.set("mock-changed-during-discovery")
            release.set()
            self.wait(lambda: not self.app._models_pending)
            show.assert_not_called()
        self.assertEqual(self.app.model.get(), original_model)
        self.assertIn("중지", self.app.status.get())
        self.assertFalse(self.errors)

    def test_diagnostic_identifies_result_from_settings_changed_during_request(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def diagnostic(*args):
            entered.set()
            release.wait(2)
            return "이전 설정의 진단 결과"
        with patch("frametag.ui.diagnose_connection", side_effect=diagnostic), \
                patch.object(self.app, "_show_diagnostic") as show:
            self.app._diagnose_api()
            self.wait(entered.is_set)
            self.app.api_key.set("mock-changed-during-diagnostic")
            release.set()
            self.wait(lambda: not self.app._diagnosing)
            self.assertIn("설정이 변경되었습니다", show.call_args.args[0])
            self.assertIn("이전 설정의 진단 결과", show.call_args.args[0])

    def test_diagnostic_result_dialog_copies_sanitized_report(self):
        report = "HTTP 410\n서버 설명: Account route disabled\n[API 키 숨김]"
        self.app._show_diagnostic(report)
        dialog = next(widget for widget in self.app.winfo_children() if widget.winfo_class() == "Toplevel")
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        copy_button = next(widget for widget in descendants(dialog)
                           if widget.winfo_class() == "TButton" and widget.cget("text") == "진단 결과 복사")
        copy_button.invoke()
        self.assertEqual(self.app.clipboard_get(), report)
        dialog.destroy()

    def _hold_engine_lock(self):
        entered, release = threading.Event(), threading.Event()
        def hold():
            with self.app.engine._lock:
                entered.set()
                release.wait(2)
        thread = threading.Thread(target=hold, daemon=True)
        thread.start()
        self.assertTrue(entered.wait(1))
        timer = threading.Timer(1.2, release.set)
        timer.start()
        self.addCleanup(release.set)
        self.addCleanup(timer.cancel)
        return release, thread

    def test_folder_loading_animates_and_gui_stays_responsive_during_engine_lock(self):
        release, thread = self._hold_engine_lock()
        started = time.monotonic()
        self.app._load_folder()
        self.assertLess(time.monotonic() - started, .25)
        self.assertTrue(self.app._loading)
        self.assertIn("로딩 중", self.app.loading_text.get())
        self.assertEqual(str(self.app.progress["mode"]), "indeterminate")
        initial_progress = float(self.app.progress["value"])
        heartbeat = []
        self.app.after(30, lambda: heartbeat.append(True))
        self.wait(lambda: bool(heartbeat), timeout=.5)
        self.wait(lambda: float(self.app.progress["value"]) > initial_progress, timeout=.5)
        self.assertFalse(release.is_set())
        release.set()
        thread.join(1)
        self.wait(lambda: not self.app._loading and not self.app._busy)
        self.assertEqual(self.app.loading_text.get(), "")
        self.assertEqual(str(self.app.progress["mode"]), "determinate")

    def test_auto_watch_toggle_returns_immediately_and_shows_pending_state(self):
        release, thread = self._hold_engine_lock()
        self.app.auto_watch.set(True)
        started = time.monotonic()
        self.app._toggle_auto()
        self.assertLess(time.monotonic() - started, .25)
        self.assertTrue(self.app._auto_pending)
        self.assertIn("처리 중", self.app.loading_text.get())
        initial_progress = float(self.app.progress["value"])
        heartbeat = []
        self.app.after(30, lambda: heartbeat.append(True))
        self.wait(lambda: bool(heartbeat), timeout=.5)
        self.wait(lambda: float(self.app.progress["value"]) > initial_progress, timeout=.5)
        self.assertFalse(release.is_set())
        release.set()
        thread.join(1)
        self.wait(lambda: not self.app._auto_pending and self.app.auto_watch.get())
        self.assertEqual(self.app.loading_text.get(), "")
        self.assertFalse(self.errors)

    def test_queued_edit_is_applied_before_immediate_rename_action(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        timer = threading.Timer(1.2, release.set)
        self.addCleanup(timer.cancel)
        timer.start()
        def pending():
            entered.set()
            release.wait(2)
        self.app._dispatch_engine(pending)
        self.wait(entered.is_set)
        first = str(self.app.photos[0].path)
        original = Path(first).read_bytes()
        self.app._dispatch_engine(self.app.engine.update_photo, first, "proposed_name", "수정확인.jpg")
        self.app.tree.selection_set(first)
        self.app._apply()
        release.set()
        target = self.folder / "수정확인.jpg"
        self.wait(lambda: target.exists() and not self.app._busy)
        self.assertFalse(Path(first).exists())
        self.assertEqual(target.read_bytes(), original)
        self.assertFalse(self.errors)

    def test_changed_folder_entry_pauses_watch_and_refuses_wrong_folder_actions(self):
        self.app.auto_watch.set(True)
        self.app._toggle_auto()
        self.wait(lambda: self.app.auto_watch.get())
        other = self.root / "other"
        other.mkdir()
        self.app.folder.set(str(other))
        self.assertFalse(self.app.auto_watch.get())
        self.assertFalse(self.app._ready())
        self.app._analyze()
        self.app.auto_watch.set(True)
        self.app._toggle_auto()
        self.assertFalse(self.app.auto_watch.get())
        self.assertEqual(self.app.engine.get_root(), self.folder.resolve())
        self.assertEqual(FakeAnalyzer.calls, [])
        self.assertFalse(self.errors)

    def test_completed_photos_default_hidden_can_show_and_edit_without_losing_undo(self):
        first = str(self.app.photos[0].path)
        self.app.engine.update_photo(first, "proposed_name", "남성_검정반팔_파랑청바지.jpg")
        self.wait(lambda: bool(self.app.by_id[first].proposed_name))
        self.app.tree.selection_set(first)
        self.app._apply()
        self.wait(lambda: len(self.app.tree.get_children()) == 2 and not self.app._busy)
        renamed = str(self.folder / "남성_검정반팔_파랑청바지.jpg")
        self.assertTrue(self.app.by_id[renamed].confirmed)
        self.assertFalse(self.app.show_completed.get())
        self.assertNotIn(renamed, self.app.tree.get_children())
        self.assertIn("완료 1장 숨김", self.app.count.get())
        self.app.show_completed.set(True)
        self.app._completed_filter_changed()
        self.assertIn(renamed, self.app.tree.get_children())
        self.app._save_settings()
        self.assertTrue(self.settings.load()["show_completed"])
        self.app.show_completed.set(False)
        self.app._completed_filter_changed()
        self.app.engine.update_photo(renamed, "objects", "우산")
        self.wait(lambda: self.app.by_id[renamed].manual and not self.app.by_id[renamed].confirmed)
        self.assertNotIn(renamed, self.app.tree.get_children())
        self.assertFalse(self.app.by_id[renamed].confirmed)
        self.app._undo()
        self.wait(lambda: first in self.app.tree.get_children() and not self.app._busy)
        self.assertTrue(Path(first).exists())
        self.assertFalse(Path(renamed).exists())
        self.assertFalse(self.errors)

    def test_gender_prefixed_photos_are_hidden_even_with_old_pending_cache(self):
        self.app._save_settings()
        self.app._exit()
        names = (
            "여성_교복_곰인형.jpg", "여성_꽃무늬원피스_대걸레 (1).jpg",
            "여성_검정스트라이프긴팔_베이지바지.jpg",
            "남성+여성_회색반팔_파랑청바지.jpg", "남성_군복_가방_4.jpg",
        )
        for name in names:
            path = self.folder / name
            Image.new("RGB", (40, 60), "blue").save(path)
            Cache(self.folder).put(Photo.from_path(path))
        self.app = self._new_app()
        self.wait(lambda: len(self.app.photos) == 8 and not self.app._loading)
        self._verify_model_for_test()
        self.assertEqual(len(self.app.tree.get_children()), 3)
        self.assertIn("완료 5장 숨김", self.app.count.get())
        self.app._analyze()
        self.wait(lambda: len(FakeAnalyzer.calls) == 3 and not self.app._busy)
        self.assertEqual(set(FakeAnalyzer.calls), {"photo1.jpg", "photo2.jpg", "photo3.jpg"})
        self.app.show_completed.set(True)
        self.app._completed_filter_changed()
        self.assertEqual(len(self.app.tree.get_children()), 8)
        for name in names:
            identifier = str(self.folder / name)
            self.assertEqual(self.app.tree.item(identifier, "values")[0], "이름 기준 제외")
            self.assertTrue(Path(identifier).exists())
        self.app.show_completed.set(False)
        self.app._completed_filter_changed()
        self.assertEqual(len(self.app.tree.get_children()), 3)
        self.assertFalse(self.errors)

    def test_arrival_during_slow_manual_analysis_is_visible_then_renamed_once(self):
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        original_analyze = FakeAnalyzer.analyze
        first = str(self.folder / "photo1.jpg")
        incoming = self.folder / "incoming.jpg"

        def slow_analyze(analyzer, path, cancel):
            if path.name == "photo1.jpg":
                entered.set()
                while not release.wait(.01) and not cancel.is_set():
                    pass
            return original_analyze(analyzer, path, cancel)

        with patch.object(FakeAnalyzer, "analyze", slow_analyze):
            try:
                self.app.auto_watch.set(True)
                self.app._toggle_auto()
                self.app._analyze()
                self.wait(entered.is_set)
                Image.new("RGB", (45, 65), "white").save(incoming)
                self.wait(lambda: len(self.app.photos) == 4 and str(incoming) in self.app.by_id)
                self.assertFalse(release.is_set())
                self.assertTrue(self.app._busy)
                self.assertIsNone(self.app.by_id[str(incoming)].analysis)
                self.assertEqual(self.app.by_id[str(incoming)].status, "대기")
                revision = self.app.by_id[first].revision
                self.app.engine.update_photo(first, "objects", "직접수정우산")
                self.wait(lambda: self.app.by_id[first].revision > revision)
                release.set()
                self.wait(lambda: not self.app._busy and len(self.app.photos) == 4
                          and all(photo.analysis for photo in self.app.photos)
                          and sum(photo.confirmed for photo in self.app.photos) == 1)
                self.assertFalse(incoming.exists())
                self.assertTrue(all((self.folder / f"photo{number}.jpg").exists()
                                    for number in range(1, 4)))
                self.assertTrue(self.app.by_id[first].manual)
                self.assertEqual(self.app.by_id[first].analysis["objects"], ["직접수정우산"])
                self.assertEqual(self.app.by_id[first].analysis["clothing"], [])
                self.assertEqual(sorted(FakeAnalyzer.calls),
                                 ["incoming.jpg", "photo1.jpg", "photo2.jpg", "photo3.jpg"])
                self.assertEqual(str(self.app.apply_button["state"]), "normal")
                self.assertEqual(str(self.app.undo_button["state"]), "normal")
                self.app.tree.selection_set(first)
                self.app._apply()
                self.wait(lambda: not self.app._busy
                          and sum(photo.confirmed for photo in self.app.photos) == 2)
                self.app._undo()
                self.wait(lambda: not self.app._busy and first in self.app.by_id
                          and sum(photo.confirmed for photo in self.app.photos) == 1)
                self.assertEqual(self.app.by_id[first].analysis["objects"], ["직접수정우산"])
                deadline = time.monotonic() + .15
                self.wait(lambda: time.monotonic() >= deadline)
                self.assertEqual(FakeAnalyzer.calls.count("incoming.jpg"), 1)
                self.assertFalse(self.errors)
            finally:
                release.set()

    def test_mock_tray_hide_show_and_complete_exit_keep_worker_alive_until_exit(self):
        class FakeTray:
            stopped = False

            def stop(self):
                self.stopped = True

        tray = FakeTray()
        self.app._tray = tray
        self.app.close_to_tray.set(True)
        self.app._on_close()
        self.assertEqual(self.app.state(), "withdrawn")
        self.assertFalse(self.app._closed)
        self.assertTrue(self.app.engine._worker.is_alive())
        self.app._show()
        self.app.update()
        self.assertEqual(self.app.state(), "normal")
        self.app._exit()
        self.assertTrue(tray.stopped)
        self.assertFalse(self.app.engine._worker.is_alive())


if __name__ == "__main__":
    unittest.main()
