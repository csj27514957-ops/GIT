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
from frametag.storage import Cache


RESULT = {"gender": "남성", "clothing": ["검정반팔", "파랑청바지"],
          "objects": [{"name": "가방", "location": "손"}]}


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

    def engine(self, interval=0.03):
        engine = Engine(lambda kind, value: self.events.append((kind, value)),
                        interval=interval, stable_scans=3, analyzer_factory=self.api.factory)
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
