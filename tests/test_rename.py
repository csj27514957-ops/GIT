import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from frametag.models import Photo
from frametag import rename
from frametag.rename import RenameError, apply_names, recover_pending, undo_last
from frametag.storage import Cache


class RenameTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cache = Cache(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def photo(self, name, proposed, contents=b"image"):
        path = self.root / name
        path.write_bytes(contents)
        photo = Photo.from_path(path)
        photo.proposed_name = proposed
        photo.manual = True
        self.cache.put(photo)
        return photo

    def test_duplicates_respect_case_and_existing_numbered_names(self):
        (self.root / "PERSON.JPG").write_bytes(b"existing")
        (self.root / "person_2.jpg").write_bytes(b"existing 2")
        a = self.photo("a.jpg", "person.jpg", b"A")
        b = self.photo("b.jpg", "person.jpg", b"B")
        result = apply_names(self.root, [a, b], self.cache)
        self.assertEqual([new.name for old, new in result], ["person_3.jpg", "person_4.jpg"])
        self.assertEqual((self.root / "PERSON.JPG").read_bytes(), b"existing")
        self.assertEqual(a.path.read_bytes(), b"A")
        self.assertEqual(b.path.read_bytes(), b"B")
        self.assertTrue(self.cache.get(a.path).manual)
        self.assertTrue(self.cache.get(a.path).confirmed)
        self.assertIsNone(self.cache.get(self.root / "a.jpg"))

    def test_filename_swap_preserves_both_files_and_cache_then_undo(self):
        a = self.photo("a.jpg", "b.jpg", b"A")
        b = self.photo("b.jpg", "a.jpg", b"B")
        apply_names(self.root, [a, b], self.cache)
        self.assertEqual((self.root / "a.jpg").read_bytes(), b"B")
        self.assertEqual((self.root / "b.jpg").read_bytes(), b"A")
        self.assertEqual(self.cache.get(self.root / "a.jpg").proposed_name, "a.jpg")
        result = undo_last(self.root, self.cache)
        self.assertEqual([(old.name, new.name) for old, new in result], [("b.jpg", "a.jpg"), ("a.jpg", "b.jpg")])
        self.assertEqual((self.root / "a.jpg").read_bytes(), b"A")
        self.assertEqual((self.root / "b.jpg").read_bytes(), b"B")
        self.assertFalse(self.cache.get(self.root / "a.jpg").confirmed)
        self.assertEqual(undo_last(self.root, self.cache), [])

    def test_stale_analysis_and_invalid_manual_name_are_rejected(self):
        photo = self.photo("a.jpg", "person.jpg")
        photo.path.write_bytes(b"different contents")
        with self.assertRaises(RenameError):
            apply_names(self.root, [photo], self.cache)
        photo = Photo.from_path(photo.path)
        photo.proposed_name = "../escape.jpg"
        with self.assertRaises(RenameError):
            apply_names(self.root, [photo], self.cache)
        self.assertTrue((self.root / "a.jpg").exists())

    def test_ordinary_second_move_failure_rolls_back_without_history(self):
        a = self.photo("a.jpg", "new-a.jpg", b"A")
        b = self.photo("b.jpg", "new-b.jpg", b"B")
        original_move = rename._move_no_clobber
        failed = False

        def fail_once(source, target, entry):
            nonlocal failed
            if target.name == "new-b.jpg" and not failed:
                failed = True
                raise OSError("simulated locked file")
            return original_move(source, target, entry)

        with patch.object(rename, "_move_no_clobber", side_effect=fail_once):
            with self.assertRaises(RenameError):
                apply_names(self.root, [a, b], self.cache)
        self.assertEqual((self.root / "a.jpg").read_bytes(), b"A")
        self.assertEqual((self.root / "b.jpg").read_bytes(), b"B")
        self.assertFalse((self.root / "new-a.jpg").exists())
        self.assertFalse((self.root / rename._PENDING).exists())
        self.assertEqual(undo_last(self.root, self.cache), [])
        self.assertFalse(a.confirmed)

    def test_new_external_destination_is_preserved_and_operation_rolls_back(self):
        photo = self.photo("a.jpg", "new.jpg", b"ORIGINAL")
        original_move = rename._move_no_clobber

        def competing_file(source, target, entry):
            if target.name == "new.jpg":
                target.write_bytes(b"EXTERNAL")
            return original_move(source, target, entry)

        with patch.object(rename, "_move_no_clobber", side_effect=competing_file):
            with self.assertRaises(RenameError):
                apply_names(self.root, [photo], self.cache)
        self.assertEqual((self.root / "a.jpg").read_bytes(), b"ORIGINAL")
        self.assertEqual((self.root / "new.jpg").read_bytes(), b"EXTERNAL")

    def test_crash_after_first_stage_is_recovered_on_reopen(self):
        photo = self.photo("a.jpg", "new.jpg", b"ORIGINAL")
        original_move = rename._move_no_clobber

        def crash_after_stage(source, target, entry):
            original_move(source, target, entry)
            if target.name.endswith(".tmp"):
                raise KeyboardInterrupt("simulate process termination")

        with patch.object(rename, "_move_no_clobber", side_effect=crash_after_stage):
            with self.assertRaises(KeyboardInterrupt):
                apply_names(self.root, [photo], self.cache)
        self.assertFalse((self.root / "a.jpg").exists())
        self.assertTrue((self.root / rename._PENDING).exists())
        recover_pending(self.root, Cache(self.root))
        self.assertEqual((self.root / "a.jpg").read_bytes(), b"ORIGINAL")
        self.assertFalse((self.root / rename._PENDING).exists())

    def test_crash_mid_swap_restores_both_original_names(self):
        self.photo("a.jpg", "b.jpg", b"A")
        a = self.cache.get(self.root / "a.jpg")
        b = self.photo("b.jpg", "a.jpg", b"B")
        original_move = rename._move_no_clobber

        def crash_mid_commit(source, target, entry):
            original_move(source, target, entry)
            if target.name == "b.jpg":
                raise KeyboardInterrupt("simulate process termination during swap")

        with patch.object(rename, "_move_no_clobber", side_effect=crash_mid_commit):
            with self.assertRaises(KeyboardInterrupt):
                apply_names(self.root, [a, b], self.cache)
        recover_pending(self.root, self.cache)
        self.assertEqual((self.root / "a.jpg").read_bytes(), b"A")
        self.assertEqual((self.root / "b.jpg").read_bytes(), b"B")
        self.assertFalse((self.root / rename._PENDING).exists())

    def test_recovery_preserves_external_file_at_original_name(self):
        photo = self.photo("a.jpg", "new.jpg", b"ORIGINAL")
        original_move = rename._move_no_clobber

        def crash_after_stage(source, target, entry):
            original_move(source, target, entry)
            raise KeyboardInterrupt("simulate crash")

        with patch.object(rename, "_move_no_clobber", side_effect=crash_after_stage):
            with self.assertRaises(KeyboardInterrupt):
                apply_names(self.root, [photo], self.cache)
        (self.root / "a.jpg").write_bytes(b"EXTERNAL")
        with self.assertRaises(RenameError):
            recover_pending(self.root, self.cache)
        self.assertEqual((self.root / "a.jpg").read_bytes(), b"EXTERNAL")
        self.assertTrue((self.root / rename._PENDING).exists())
        self.assertEqual(next(self.root.glob(".frametag-*.tmp")).read_bytes(), b"ORIGINAL")
        (self.root / "a.jpg").unlink()
        recover_pending(self.root, self.cache)
        self.assertEqual((self.root / "a.jpg").read_bytes(), b"ORIGINAL")

    def test_metadata_failure_reports_incomplete_and_remains_recoverable(self):
        photo = self.photo("a.jpg", "new.jpg", b"ORIGINAL")
        with patch.object(self.cache, "relocate_many", side_effect=OSError("simulated database failure")):
            with self.assertRaises(RenameError):
                apply_names(self.root, [photo], self.cache)
        self.assertFalse(photo.confirmed)
        self.assertTrue((self.root / "new.jpg").exists())
        self.assertTrue((self.root / rename._PENDING).exists())
        recover_pending(self.root, self.cache)
        self.assertTrue(self.cache.get(self.root / "new.jpg").confirmed)
        undo_last(self.root, self.cache)
        self.assertEqual((self.root / "a.jpg").read_bytes(), b"ORIGINAL")

    def test_committed_files_finish_metadata_after_restart(self):
        photo = self.photo("a.jpg", "new.jpg", b"ORIGINAL")
        with patch.object(rename, "_finish", side_effect=KeyboardInterrupt("simulate crash")):
            with self.assertRaises(KeyboardInterrupt):
                apply_names(self.root, [photo], self.cache)
        self.assertTrue((self.root / "new.jpg").exists())
        recover_pending(self.root, Cache(self.root))
        self.assertTrue(self.cache.get(self.root / "new.jpg").confirmed)
        self.assertFalse((self.root / rename._PENDING).exists())
        undo_last(self.root, self.cache)
        self.assertEqual((self.root / "a.jpg").read_bytes(), b"ORIGINAL")

    def test_undo_refuses_external_original_and_modified_renamed_file(self):
        photo = self.photo("a.jpg", "new.jpg", b"ORIGINAL")
        apply_names(self.root, [photo], self.cache)
        (self.root / "A.JPG").write_bytes(b"EXTERNAL")
        with self.assertRaises(RenameError):
            undo_last(self.root, self.cache)
        (self.root / "A.JPG").unlink()
        photo.path.write_bytes(b"EDITED")
        with self.assertRaises(RenameError):
            undo_last(self.root, self.cache)
        self.assertEqual(photo.path.read_bytes(), b"EDITED")

    def test_long_unicode_duplicate_names_fit_filename_limit(self):
        proposed = "가" * 78 + ".jpg"  # 238 bytes before adding a duplicate suffix.
        a = self.photo("a.jpg", proposed, b"A")
        b = self.photo("b.jpg", proposed, b"B")
        apply_names(self.root, [a, b], self.cache)
        self.assertNotEqual(a.path, b.path)
        self.assertLessEqual(len(b.path.name.encode("utf-8")), 240)
        self.assertTrue(b.path.stem.endswith("_2"))

    def test_no_op_name_still_persists_confirmation(self):
        photo = self.photo("a.jpg", "a.jpg")
        self.assertEqual(apply_names(self.root, [photo], self.cache), [])
        self.assertTrue(photo.confirmed)
        self.assertTrue(self.cache.get(photo.path).confirmed)
        self.assertEqual(undo_last(self.root, self.cache), [])

    def test_case_only_rename_and_undo_preserves_exact_capitalization(self):
        photo = self.photo("photo.jpg", "PHOTO.jpg")
        apply_names(self.root, [photo], self.cache)
        self.assertIn("PHOTO.jpg", [path.name for path in self.root.iterdir()])
        undo_last(self.root, self.cache)
        self.assertIn("photo.jpg", [path.name for path in self.root.iterdir()])

    def test_selected_hardlinks_are_rejected_before_any_changes(self):
        a = self.photo("a.jpg", "new-a.jpg")
        os.link(a.path, self.root / "b.jpg")
        b = Photo.from_path(self.root / "b.jpg")
        b.proposed_name = "new-b.jpg"
        with self.assertRaises(RenameError):
            apply_names(self.root, [a, b], self.cache)
        self.assertTrue(a.path.exists())
        self.assertTrue(b.path.exists())
        self.assertFalse((self.root / rename._PENDING).exists())


if __name__ == "__main__":
    unittest.main()
