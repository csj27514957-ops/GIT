import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from frametag.api import APIError, Cancelled
from frametag.engine import Engine
from frametag.models import Photo
from frametag.rename import _move_no_clobber
from frametag.storage import Cache


RESULT = {"gender": "남성", "clothing": ["검정반팔", "파랑청바지"],
          "objects": [{"name": "가방", "location": "손"}]}
TAGGED_NAME = "여성_흰색반팔_검정반바지_손-종이 (2).png"
TAGGED_ANALYSIS = {"gender": "여성", "clothing": ["흰색반팔", "검정반바지"],
                   "objects": ["종이(손)"]}


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("background operation did not reach expected state")


class FakeAPI:
    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.failure = None

    def factory(self, provider, model, key):
        outer = self

        class Analyzer:
            def analyze(self, path, cancel):
                outer.calls.append((path.name, provider, model))
                outer.entered.set()
                while not outer.release.wait(0.01):
                    if cancel.is_set():
                        raise Cancelled()
                if cancel.is_set():
                    raise Cancelled()
                if outer.failure:
                    raise outer.failure
                return RESULT

        return Analyzer()


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.events = []
        self.api = FakeAPI()
        self.engines = []
        self.addCleanup(self.close_all)

    def close_all(self):
        self.api.release.set()
        for engine in self.engines:
            engine.shutdown()

    def engine(self, interval=0.03, analyzer_factory=None):
        engine = Engine(lambda kind, value: self.events.append((kind, value)),
                        interval=interval, stable_scans=3,
                        analyzer_factory=analyzer_factory or self.api.factory)
        self.engines.append(engine)
        engine.configure("OpenAI", "fake-model", "test-placeholder")
        engine.load_folder(self.root)
        return engine

    def photo(self, name="original.png", color="red"):
        path = self.root / name
        Image.new("RGB", (16, 16), color).save(path)
        return path

    def idle(self, engine):
        wait_until(lambda: not engine._busy)

    def test_existing_tagged_filename_is_cached_confirmed_without_rename_history(self):
        path = self.photo(TAGGED_NAME)
        before = path.read_bytes()
        engine = self.engine()
        photo = engine.get_photos()[0]
        self.assertEqual(photo.analysis, TAGGED_ANALYSIS)
        self.assertEqual(photo.proposed_name, TAGGED_NAME)
        self.assertTrue(photo.confirmed)
        self.assertFalse(photo.manual)
        self.assertEqual(photo.status, "기존 이름 변경 완료")
        self.assertEqual(photo.model, "")
        self.assertEqual(path.read_bytes(), before)
        cached = Cache(self.root).get(path)
        self.assertTrue(cached.confirmed)
        self.assertEqual(cached.analysis, TAGGED_ANALYSIS)
        record = json.loads((self.root / ".frametag_watch_v2.json").read_text())
        self.assertEqual(record["entries"][TAGGED_NAME]["state"], "processed")
        self.assertFalse((self.root / "frametag_history_v2.json").exists())
        engine.start_analysis()
        self.idle(engine)
        engine.set_auto(True)
        time.sleep(0.15)
        self.assertEqual(self.api.calls, [])

    def test_manual_analysis_processes_original_only_in_mixed_folder(self):
        tagged = self.photo(TAGGED_NAME)
        original = self.photo("original.png")
        arbitrary = self.photo("여성_휴가사진.png")
        engine = self.engine()
        engine.start_analysis()
        self.idle(engine)
        self.assertEqual({call[0] for call in self.api.calls},
                         {original.name, arbitrary.name})
        photos = {photo.path.name: photo for photo in engine.get_photos()}
        self.assertTrue(photos[tagged.name].confirmed)
        self.assertFalse(photos[original.name].confirmed)
        self.assertFalse(photos[arbitrary.name].confirmed)
        self.assertTrue(tagged.exists())

    def test_user_numbered_tagged_names_skip_analysis_and_keep_existing_baseline(self):
        names = ["여성_흰색반팔_검정긴바지_3.jpg",
                 "남성_흰색반팔_검정반바지_16.jpg"]
        paths = [self.photo(name) for name in names]
        (self.root / ".frametag_watch_v2.json").write_text(json.dumps({
            "version": 2, "recursive": False, "entries": {
                path.name: {"state": "baseline", "size": path.stat().st_size,
                            "mtime_ns": path.stat().st_mtime_ns, "present": True}
                for path in paths}}), encoding="utf-8")
        engine = self.engine()
        self.assertTrue(all(photo.confirmed for photo in engine.get_photos()))
        self.assertTrue(all(engine._known[name]["state"] == "processed" for name in names))
        engine.start_analysis()
        self.idle(engine)
        engine.set_auto(True)
        time.sleep(0.15)
        self.assertEqual(self.api.calls, [])
        self.assertTrue(all(path.exists() for path in paths))
        for path in paths:
            self.assertTrue(Cache(self.root).get(path).confirmed)

    def test_arrival_is_visible_during_slow_auto_analysis_and_waits_for_complete_copy(self):
        engine = self.engine()
        engine.set_auto(True)
        self.api.release.clear()
        first = self.photo("first.png")
        self.assertTrue(self.api.entered.wait(2))
        buffer = io.BytesIO()
        Image.new("RGB", (48, 32), "blue").save(buffer, format="PNG")
        contents = buffer.getvalue()
        incoming = self.root / "incoming.png"
        incoming.write_bytes(contents[:20])
        wait_until(lambda: len(engine.get_photos()) == 2)
        photos = {photo.path.name: photo for photo in engine.get_photos()}
        self.assertTrue(engine.is_busy())
        self.assertEqual(photos[incoming.name].status, "대기")
        self.assertEqual(engine._known[incoming.name]["state"], "pending")
        self.assertEqual([call[0] for call in self.api.calls], [first.name])
        self.api.release.set()
        wait_until(lambda: any(photo.confirmed for photo in engine.get_photos()))
        self.idle(engine)
        time.sleep(0.12)
        self.assertEqual([call[0] for call in self.api.calls], [first.name])
        incoming.write_bytes(contents)
        wait_until(lambda: len(engine.get_photos()) == 2
                   and all(photo.confirmed for photo in engine.get_photos()))
        self.assertEqual([call[0] for call in self.api.calls], [first.name, incoming.name])
        self.assertFalse(incoming.exists())
        self.assertIn(contents, [photo.path.read_bytes() for photo in engine.get_photos()])

    def test_arrival_during_manual_batch_is_visible_and_processed_after_batch(self):
        originals = [self.photo("first.png"), self.photo("second.png")]
        engine = self.engine()
        engine.set_auto(True)
        self.api.release.clear()
        engine.start_analysis()
        self.assertTrue(self.api.entered.wait(2))
        incoming = self.photo("incoming.png", color="blue")
        wait_until(lambda: len(engine.get_photos()) == 3)
        self.assertTrue(engine.is_busy())
        self.assertEqual([call[0] for call in self.api.calls], [originals[0].name])
        self.api.release.set()
        wait_until(lambda: len(self.api.calls) == 3 and not engine.is_busy())
        self.assertEqual([call[0] for call in self.api.calls],
                         [originals[0].name, originals[1].name, incoming.name])
        self.assertTrue(all(path.exists() for path in originals))
        self.assertFalse(incoming.exists())
        self.assertEqual(sum(photo.confirmed for photo in engine.get_photos()), 1)

    def test_arrival_during_rename_transaction_is_scanned_after_commit_once(self):
        original = self.photo()
        original_contents = original.read_bytes()
        engine = self.engine()
        engine.set_auto(True)
        engine.update_photo(str(original), "proposed_name", "reviewed.png")
        staged, release = threading.Event(), threading.Event()
        blocked = False

        def pause_after_first_move(source, target, entry):
            nonlocal blocked
            _move_no_clobber(source, target, entry)
            if not blocked:
                blocked = True
                staged.set()
                if not release.wait(3):
                    raise AssertionError("rename was not released")

        operation = threading.Thread(target=engine.apply_selected,
                                     args=([str(original)],), daemon=True)
        with patch("frametag.rename._move_no_clobber", side_effect=pause_after_first_move):
            operation.start()
            try:
                self.assertTrue(staged.wait(2))
                incoming = self.photo("incoming.png", color="blue")
                incoming_contents = incoming.read_bytes()
                time.sleep(0.12)
                self.assertFalse(original.exists())  # Already staged to the temporary name.
                self.assertEqual(self.api.calls, [])
                # Scanning waits for the transaction lock; transient staging
                # state must never publish a disappearing original or temp row.
                snapshots = [value for kind, value in self.events if kind == "photos"]
                self.assertEqual([photo.path.name for photo in snapshots[-1]], [original.name])
            finally:
                release.set()
                operation.join(3)
            self.assertFalse(operation.is_alive())
            wait_until(lambda: len(engine.get_photos()) == 2
                       and all(photo.confirmed for photo in engine.get_photos()))
        self.assertEqual([call[0] for call in self.api.calls], [incoming.name])
        self.assertEqual((self.root / "reviewed.png").read_bytes(), original_contents)
        self.assertIn(incoming_contents, [photo.path.read_bytes() for photo in engine.get_photos()])
        history = json.loads((self.root / "frametag_history_v2.json").read_text())
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["entries"][0]["source"], original.name)
        self.assertEqual(history[0]["entries"][0]["target"], "reviewed.png")
        self.assertEqual(history[1]["entries"][0]["source"], incoming.name)
        self.assertFalse((self.root / ".frametag_pending_v2.json").exists())
        self.assertEqual(list(self.root.glob(".frametag-*.tmp")), [])

    def test_auto_disabled_still_queues_arrival_during_manual_analysis(self):
        original = self.photo()
        engine = self.engine()
        self.api.release.clear()
        engine.start_analysis()
        self.assertTrue(self.api.entered.wait(2))
        incoming = self.photo("incoming.png")
        wait_until(lambda: len(engine.get_photos()) == 2)
        self.assertEqual(engine._known[incoming.name]["state"], "pending")
        self.assertTrue(engine.is_busy())
        self.api.release.set()
        self.idle(engine)
        time.sleep(0.12)
        self.assertEqual([call[0] for call in self.api.calls], [original.name])
        self.assertTrue(incoming.exists())
        photos = {photo.path.name: photo for photo in engine.get_photos()}
        self.assertEqual(photos[incoming.name].status, "대기")
        self.assertIsNone(photos[incoming.name].analysis)
        engine.set_auto(True)
        wait_until(lambda: any(photo.confirmed for photo in engine.get_photos()))
        self.assertEqual([call[0] for call in self.api.calls], [original.name, incoming.name])

    def test_shutdown_cancels_request_and_stops_worker_and_watcher(self):
        engine = self.engine()
        engine.set_auto(True)
        self.api.release.clear()
        self.photo("first.png")
        self.assertTrue(self.api.entered.wait(2))
        self.photo("incoming.png")
        wait_until(lambda: len(engine.get_photos()) == 2)
        engine.shutdown()
        self.assertFalse(engine._worker.is_alive())
        self.assertFalse(engine._watcher.is_alive())
        self.assertEqual([call[0] for call in self.api.calls], ["first.png"])
        record = json.loads((self.root / ".frametag_watch_v2.json").read_text())
        self.assertEqual(record["entries"]["incoming.png"]["state"], "pending")

    def test_stale_request_error_does_not_poison_watcher_replaced_file(self):
        api = self.api

        def factory(provider, model, key):
            delegate = api.factory(provider, model, key)

            class Analyzer:
                def analyze(self, path, cancel):
                    first_call = not api.calls
                    result = delegate.analyze(path, cancel)
                    if first_call:
                        raise APIError("이전 파일 요청 오류", fatal=True)
                    return result

            return Analyzer()

        engine = self.engine(analyzer_factory=factory)
        engine.set_auto(True)
        api.release.clear()
        self.photo("first.png")
        self.assertTrue(api.entered.wait(2))
        previous = engine.get_photos()[0]
        self.photo("first.png", color="blue")
        wait_until(lambda: (engine.get_photos()[0].size, engine.get_photos()[0].mtime_ns)
                   != (previous.size, previous.mtime_ns))
        self.assertEqual(engine.get_photos()[0].status, "대기")
        api.release.set()
        wait_until(lambda: engine.get_photos()[0].confirmed)
        self.assertEqual([call[0] for call in api.calls], ["first.png", "first.png"])
        self.assertEqual(engine.get_photos()[0].error, "")
        self.assertTrue(engine._auto)

    def test_new_tagged_photo_skips_auto_while_original_is_processed(self):
        engine = self.engine()
        engine.set_auto(True)
        tagged = self.photo(TAGGED_NAME)
        original = self.photo("new.png")
        wait_until(lambda: len(engine.get_photos()) == 2
                   and all(photo.confirmed for photo in engine.get_photos()))
        self.assertEqual([call[0] for call in self.api.calls], [original.name])
        self.assertTrue(tagged.exists())
        self.assertFalse(original.exists())
        self.assertEqual(engine._known[tagged.name]["state"], "processed")

    def test_offline_tagged_arrival_is_recognized_after_restart_without_api(self):
        engine = self.engine()
        engine.shutdown()
        tagged = self.photo(TAGGED_NAME)
        restarted = self.engine()
        restarted.set_auto(True)
        time.sleep(0.15)
        self.assertTrue(restarted.get_photos()[0].confirmed)
        self.assertTrue(tagged.exists())
        self.assertEqual(self.api.calls, [])
        restarted.shutdown()
        reloaded = self.engine()
        self.assertTrue(reloaded.get_photos()[0].confirmed)
        self.assertEqual(reloaded.get_photos()[0].analysis, TAGGED_ANALYSIS)

    def test_tagged_existing_manifest_baseline_is_updated_to_processed(self):
        path = self.photo(TAGGED_NAME)
        stat = path.stat()
        (self.root / ".frametag_watch_v2.json").write_text(json.dumps({
            "version": 2, "recursive": False, "entries": {
                TAGGED_NAME: {"state": "baseline", "size": stat.st_size,
                              "mtime_ns": stat.st_mtime_ns, "present": True}}}),
            encoding="utf-8")
        engine = self.engine()
        self.assertTrue(engine.get_photos()[0].confirmed)
        self.assertEqual(engine._known[TAGGED_NAME]["state"], "processed")
        record = json.loads((self.root / ".frametag_watch_v2.json").read_text())
        self.assertEqual(record["entries"][TAGGED_NAME]["state"], "processed")

    def test_valid_unconfirmed_cache_takes_precedence_over_tagged_filename(self):
        path = self.photo(TAGGED_NAME)
        cached = Photo.from_path(path)
        # A valid awaiting-analysis record can come from an earlier version or
        # an undo. The filename heuristic must not replace that explicit state.
        Cache(self.root).put(cached)
        engine = self.engine()
        self.assertFalse(engine.get_photos()[0].confirmed)
        self.assertIsNone(engine.get_photos()[0].analysis)
        self.assertEqual(engine.get_photos()[0].status, "대기")
        engine.start_analysis()
        self.idle(engine)
        self.assertEqual([call[0] for call in self.api.calls], [TAGGED_NAME])
        self.assertFalse(engine.get_photos()[0].confirmed)

    def test_manual_edit_of_recognized_photo_survives_reload_and_undo(self):
        path = self.photo(TAGGED_NAME)
        engine = self.engine()
        engine.update_photo(str(path), "clothing", "직접정한의상")
        engine.shutdown()
        restarted = self.engine()
        photo = restarted.get_photos()[0]
        self.assertFalse(photo.confirmed)
        self.assertTrue(photo.manual)
        self.assertEqual(photo.analysis["clothing"], ["직접정한의상"])
        restarted.start_analysis()
        self.idle(restarted)
        self.assertEqual(self.api.calls, [])
        restarted.apply_selected([str(path)])
        self.assertTrue(restarted.get_photos()[0].confirmed)
        restarted.undo()
        self.assertTrue(path.exists())
        self.assertFalse(restarted.get_photos()[0].confirmed)
        restarted.shutdown()
        reloaded = self.engine()
        self.assertFalse(reloaded.get_photos()[0].confirmed)
        self.assertEqual(reloaded.get_photos()[0].analysis["clothing"], ["직접정한의상"])

    def test_replaced_tagged_file_ignores_stale_manual_cache_and_skips_api(self):
        path = self.photo(TAGGED_NAME)
        engine = self.engine()
        engine.update_photo(str(path), "clothing", "이전사진의상")
        self.assertTrue(engine.get_photos()[0].manual)
        self.photo(TAGGED_NAME, color="blue")
        wait_until(lambda: engine.get_photos()[0].confirmed)
        self.assertFalse(engine.get_photos()[0].manual)
        self.assertEqual(engine.get_photos()[0].analysis, TAGGED_ANALYSIS)
        engine.start_analysis()
        self.idle(engine)
        self.assertEqual(self.api.calls, [])

    def test_baseline_untouched_new_photo_renamed_once(self):
        original = self.photo()
        engine = self.engine()
        engine.set_auto(True)
        engine.set_auto(True)
        new = self.photo("new.png")
        wait_until(lambda: len(engine.get_photos()) == 2
                   and any(photo.confirmed for photo in engine.get_photos()))
        time.sleep(0.15)
        self.assertEqual(len(self.api.calls), 1)
        self.assertTrue(original.exists())
        self.assertFalse(new.exists())
        self.assertEqual(sum(photo.confirmed for photo in engine.get_photos()), 1)
        self.assertEqual(sum(event == "auto" and value for event, value in self.events), 1)

    def test_copy_waits_for_stability_and_decodable_contents(self):
        engine = self.engine()
        engine.set_auto(True)
        buffer = io.BytesIO()
        Image.new("RGB", (48, 32), "blue").save(buffer, format="PNG")
        contents = buffer.getvalue()
        incoming = self.root / "copy.png"
        incoming.write_bytes(contents[:20])
        time.sleep(0.15)
        self.assertEqual(self.api.calls, [])
        incoming.write_bytes(contents)
        time.sleep(0.035)
        self.assertEqual(self.api.calls, [])
        wait_until(lambda: any(p.confirmed for p in engine.get_photos()))
        self.assertEqual(len(self.api.calls), 1)

    def test_truncated_jpeg_is_not_sent_even_with_complete_header(self):
        engine = self.engine()
        engine.set_auto(True)
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), "blue").save(buffer, format="JPEG")
        contents = buffer.getvalue()
        incoming = self.root / "copy.jpg"
        incoming.write_bytes(contents[:-20])
        time.sleep(0.15)
        self.assertEqual(self.api.calls, [])
        incoming.write_bytes(contents)
        wait_until(lambda: any(p.confirmed for p in engine.get_photos()))
        self.assertEqual(len(self.api.calls), 1)

    def test_arrivals_while_stopped_are_processed_on_restart(self):
        baseline = self.photo("baseline.png")
        engine = self.engine()
        engine.shutdown()
        self.photo("offline.png")
        restarted = self.engine()
        restarted.set_auto(True)
        wait_until(lambda: any(p.confirmed for p in restarted.get_photos()))
        self.assertEqual([call[0] for call in self.api.calls], ["offline.png"])
        self.assertTrue(baseline.exists())

    def test_recursive_scope_expansion_baselines_existing_subfolder_photos(self):
        subfolder = self.root / "subfolder"
        subfolder.mkdir()
        existing = subfolder / "existing.png"
        Image.new("RGB", (16, 16), "red").save(existing)
        engine = self.engine()
        self.assertTrue(engine.load_folder(self.root, recursive=True))
        engine.set_auto(True)
        time.sleep(0.15)
        self.assertEqual(self.api.calls, [])
        self.assertTrue(existing.exists())
        added = subfolder / "new.png"
        Image.new("RGB", (16, 16), "blue").save(added)
        wait_until(lambda: any(p.confirmed for p in engine.get_photos()))
        self.assertEqual([call[0] for call in self.api.calls], ["new.png"])

    def test_scope_expansion_preserves_known_pending_and_top_level_offline_arrivals(self):
        subfolder = self.root / "subfolder"
        subfolder.mkdir()
        engine = self.engine()
        self.assertTrue(engine.load_folder(self.root, recursive=True))
        nested = subfolder / "waiting.png"
        Image.new("RGB", (16, 16), "red").save(nested)
        wait_until(lambda: len(engine.get_photos()) == 1)
        self.assertTrue(engine.load_folder(self.root, recursive=False))
        engine.shutdown()
        self.photo("offline.png")
        restarted = self.engine()
        self.assertTrue(restarted.load_folder(self.root, recursive=True))
        restarted.set_auto(True)
        wait_until(lambda: len(self.api.calls) == 2 and not restarted._busy)
        self.assertEqual({call[0] for call in self.api.calls}, {"offline.png", "waiting.png"})

    def test_pending_copy_survives_shutdown_and_restart(self):
        engine = self.engine(interval=0.08)
        engine.set_auto(True)
        self.photo("arriving.png")
        wait_until(lambda: len(engine.get_photos()) == 1)
        engine.shutdown()
        self.assertEqual(self.api.calls, [])
        restarted = self.engine()
        restarted.set_auto(True)
        wait_until(lambda: any(p.confirmed for p in restarted.get_photos()))
        self.assertEqual(len(self.api.calls), 1)

    def test_late_result_cannot_overwrite_manual_revision_or_rename(self):
        engine = self.engine()
        engine.set_auto(True)
        self.api.release.clear()
        path = self.photo("new.png")
        self.assertTrue(self.api.entered.wait(2))
        engine.update_photo(str(path), "clothing", "직접입력의상")
        self.api.release.set()
        self.idle(engine)
        photo = engine.get_photos()[0]
        self.assertEqual(photo.analysis["clothing"], ["직접입력의상"])
        self.assertTrue(photo.manual)
        self.assertFalse(photo.confirmed)
        self.assertTrue(path.exists())
        time.sleep(0.15)
        self.assertEqual(len(self.api.calls), 1)

    def test_file_changed_during_request_discards_result_and_reanalyzes(self):
        engine = self.engine()
        engine.set_auto(True)
        self.api.release.clear()
        self.photo("new.png")
        self.assertTrue(self.api.entered.wait(2))
        self.photo("new.png", color="blue")
        self.api.release.set()
        wait_until(lambda: any(p.confirmed for p in engine.get_photos()))
        self.assertEqual(len(self.api.calls), 2)

    def test_configured_model_is_snapshot_and_valid_results_survive_reload(self):
        path = self.photo()
        engine = self.engine()
        self.api.release.clear()
        engine.start_analysis()
        self.assertTrue(self.api.entered.wait(2))
        engine.configure("NVIDIA", "other-model", "test-placeholder")
        self.api.release.set()
        self.idle(engine)
        self.assertEqual(engine.get_photos()[0].model, "fake-model")
        engine.shutdown()
        restarted = self.engine()
        restarted.start_analysis()
        self.idle(restarted)
        self.assertEqual(len(self.api.calls), 1)
        self.assertTrue(path.exists())
        self.assertEqual(restarted.get_photos()[0].model, "fake-model")

    def test_fatal_pauses_auto_and_error_does_not_repeat_after_restart(self):
        self.api.failure = APIError("API 권한 오류", fatal=True)
        engine = self.engine()
        engine.set_auto(True)
        self.photo("bad.png")
        wait_until(lambda: any(event == "auto" and value is False for event, value in self.events))
        self.idle(engine)
        self.assertEqual(engine.get_photos()[0].status, "오류")
        engine.shutdown()
        restarted = self.engine()
        restarted.set_auto(True)
        time.sleep(0.2)
        self.assertEqual(len(self.api.calls), 1)
        self.assertEqual(restarted.get_photos()[0].error, "API 권한 오류")

    def test_fatal_manual_batch_leaves_remaining_pending_and_reports_halt(self):
        self.photo("first.png")
        self.photo("second.png")
        self.api.failure = APIError("API 권한 오류", fatal=True)
        engine = self.engine()
        engine.start_analysis()
        self.idle(engine)
        self.assertEqual([call[0] for call in self.api.calls], ["first.png"])
        photos = engine.get_photos()
        self.assertEqual(photos[0].status, "오류")
        self.assertEqual(photos[1].status, "대기")
        self.assertIsNone(photos[1].analysis)
        statuses = [value for kind, value in self.events if kind == "status"]
        self.assertEqual(statuses[-1], "오류로 분석을 중지했습니다. API 설정을 확인한 뒤 다시 시작하세요.")
        self.assertNotIn("남은 사진 분석을 마쳤습니다.", statuses)

    def test_three_nonfatal_errors_pause_without_retry_loop(self):
        self.api.failure = APIError("일시 오류")
        engine = self.engine()
        engine.set_auto(True)
        for number in range(4):
            self.photo(f"new{number}.png")
        wait_until(lambda: len(self.api.calls) == 3 and not engine._busy)
        time.sleep(0.15)
        self.assertFalse(engine._auto)
        self.assertEqual(len(self.api.calls), 3)
        self.assertEqual(sum(bool(p.error) for p in engine.get_photos()), 3)

    def test_cancel_stops_worker_and_preserves_pending_for_restart(self):
        engine = self.engine()
        self.api.release.clear()
        engine.set_auto(True)
        self.photo("new.png")
        self.assertTrue(self.api.entered.wait(2))
        engine.shutdown()
        self.assertFalse(engine._worker.is_alive())
        self.assertFalse(engine.get_photos()[0].confirmed)
        record = json.loads((self.root / ".frametag_watch_v2.json").read_text())
        self.assertEqual(record["entries"]["new.png"]["state"], "pending")
        self.api.release.set()
        restarted = self.engine()
        restarted.set_auto(True)
        wait_until(lambda: restarted.get_photos()[0].confirmed)
        self.assertEqual(len(self.api.calls), 2)

    def test_deep_snapshots_and_manual_text_preserved_through_cache(self):
        path = self.photo()
        engine = self.engine()
        engine.update_photo(str(path), "clothing", "내가정한의상, 무늬바지")
        engine.update_photo(str(path), "objects", "의자(바닥), 종이(바닥), 우산")
        external = engine.get_photos()
        external[0].analysis["clothing"].append("외부변경")
        self.assertNotIn("외부변경", engine.get_photos()[0].analysis["clothing"])
        engine.shutdown()
        restarted = self.engine()
        self.assertEqual(restarted.get_photos()[0].analysis["objects"], ["의자(바닥)", "종이(바닥)", "우산"])
        restarted.start_analysis()
        self.idle(restarted)
        self.assertEqual(self.api.calls, [])

    def test_busy_refuses_actual_rename_and_folder_switch(self):
        path = self.photo()
        engine = self.engine()
        self.api.release.clear()
        engine.start_analysis()
        self.assertTrue(self.api.entered.wait(2))
        engine.apply_selected([str(path)])
        engine.undo()
        engine.load_folder(self.root / "missing")
        self.assertTrue(path.exists())
        self.assertEqual(len([value for event, value in self.events if event == "error"]), 3)
        engine.stop_analysis()
        self.idle(engine)

    def test_failed_folder_load_rolls_back_root_and_pauses_auto(self):
        path = self.photo()
        engine = self.engine()
        engine.set_auto(True)
        other = self.root / "other"
        other.mkdir()
        with patch("frametag.engine.recover_pending", side_effect=ValueError("invalid journal")):
            self.assertFalse(engine.load_folder(other))
        self.assertEqual(engine.get_root(), self.root)
        self.assertEqual(engine.get_photos()[0].path, path)
        self.assertFalse(engine._auto)
        self.assertEqual(self.api.calls, [])

    def test_failed_scan_after_new_root_assignment_rolls_back(self):
        path = self.photo()
        engine = self.engine()
        other = self.root / "other"
        other.mkdir()
        with patch.object(engine, "_poll", side_effect=PermissionError("cannot scan")):
            self.assertFalse(engine.load_folder(other))
        self.assertEqual(engine.get_root(), self.root)
        self.assertEqual(engine.get_photos()[0].path, path)

    def test_cached_success_pending_rename_resumes_without_api_charge(self):
        engine = self.engine()
        path = self.photo("arriving.png")
        wait_until(lambda: len(engine.get_photos()) == 1)
        engine.shutdown()
        photo = engine.get_photos()[0]
        photo.analysis = {"gender": "여성", "clothing": ["빨강원피스"], "objects": []}
        photo.proposed_name = "여성_빨강원피스.png"
        photo.status = "분석 완료"
        Cache(self.root).put(photo)
        restarted = self.engine()
        restarted.set_auto(True)
        wait_until(lambda: restarted.get_photos()[0].confirmed)
        self.assertFalse(path.exists())
        self.assertEqual(self.api.calls, [])

    def test_reference_unifies_all_fields_and_select_through_skips_confirmed(self):
        paths = [self.photo(f"photo{number}.png") for number in (1, 2, 10)]
        engine = self.engine()
        engine.start_analysis()
        self.idle(engine)
        engine.update_photo(str(paths[0]), "clothing", "직접정한의상")
        engine.update_photo(str(paths[0]), "objects", "의자(바닥), 우산")
        engine.unify_selected([str(p) for p in paths[:2]], str(paths[0]))
        photos = engine.get_photos()
        self.assertEqual(photos[0].analysis, photos[1].analysis)
        self.assertTrue(photos[1].manual)
        engine.apply_selected([str(paths[0])])
        self.assertEqual(engine.select_through(str(paths[2])), [str(paths[1]), str(paths[2])])

    def test_manual_apply_and_undo_do_not_trigger_auto_again(self):
        path = self.photo()
        engine = self.engine()
        engine.start_analysis()
        self.idle(engine)
        engine.apply_selected([str(path)])
        self.assertTrue(engine.get_photos()[0].confirmed)
        engine.set_auto(True)
        time.sleep(0.15)
        engine.undo()
        self.assertTrue(path.exists())
        time.sleep(0.15)
        self.assertEqual(len(self.api.calls), 1)
        self.assertFalse(engine.get_photos()[0].confirmed)

    def test_undo_filename_swap_preserves_every_snapshot_and_manifest_entry(self):
        first = self.photo("a.png", color="red")
        second = self.photo("b.png", color="blue")
        before = {path.name: path.read_bytes() for path in (first, second)}
        engine = self.engine()
        # Hold the worker out of the operation to inspect the immediate result, before any rescan.
        with engine._lock:
            engine.update_photo(str(first), "proposed_name", "b.png")
            engine.update_photo(str(second), "proposed_name", "a.png")
            engine.apply_selected([str(first), str(second)])
            engine.undo()
            self.assertEqual({p.path.name for p in engine.get_photos()}, {"a.png", "b.png"})
            self.assertEqual(set(engine._known), {"a.png", "b.png"})
            self.assertEqual({path.name: path.read_bytes() for path in (first, second)}, before)

    def test_mixed_selection_applies_only_ready_photos_and_leaves_pending(self):
        ready = self.photo("ready.png")
        pending = self.photo("pending.png")
        engine = self.engine()
        engine.update_photo(str(ready), "proposed_name", "reviewed.png")
        engine.apply_selected([str(ready), str(pending)])
        photos = {photo.path.name: photo for photo in engine.get_photos()}
        self.assertTrue(photos["reviewed.png"].confirmed)
        self.assertTrue(pending.exists())
        self.assertFalse(photos["pending.png"].confirmed)
        self.assertEqual(photos["pending.png"].status, "대기")
        self.assertEqual(photos["pending.png"].proposed_name, "")
        self.assertEqual(self.api.calls, [])
        self.assertIn(("status", "사진 1장의 파일명을 적용했습니다."), self.events)

    def test_changed_cached_baseline_photo_is_processed(self):
        path = self.photo()
        engine = self.engine()
        engine.start_analysis()
        self.idle(engine)
        engine.set_auto(True)
        self.photo("original.png", color="blue")
        wait_until(lambda: engine.get_photos()[0].confirmed)
        self.assertEqual(len(self.api.calls), 2)
        self.assertFalse(path.exists())

    def test_successful_analysis_is_reused_when_auto_rename_failed(self):
        engine = self.engine()
        engine.set_auto(True)
        with patch("frametag.engine.apply_names", side_effect=PermissionError("locked")):
            self.photo("new.png")
            wait_until(lambda: len(engine.get_photos()) == 1 and engine.get_photos()[0].status == "파일명 오류")
        engine.start_analysis()
        self.idle(engine)
        self.assertEqual(len(self.api.calls), 1)
        engine.apply_selected([str(engine.get_photos()[0].path)])
        self.assertTrue(engine.get_photos()[0].confirmed)


if __name__ == "__main__":
    unittest.main()
