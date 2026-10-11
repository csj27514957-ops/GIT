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

from frametag.engine import Engine
from frametag.models import Photo
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
