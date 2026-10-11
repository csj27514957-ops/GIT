import concurrent.futures
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from frametag.models import Photo
from frametag.storage import Cache, KeyStore, KeyStoreError, SettingsStore


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def photo(self, name="one.jpg"):
        path = self.root / name
        path.write_bytes(b"photo contents")
        return Photo.from_path(path)

    def test_cache_persists_manual_confirmation_without_touching_legacy(self):
        legacy = self.root / ".frametag_cache.sqlite"
        legacy.write_bytes(b"unknown legacy data")
        photo = self.photo()
        photo.analysis = {"gender": "여성"}
        photo.proposed_name = "직접 선택.jpg"
        photo.manual = photo.confirmed = True
        photo.model = "test-model"
        photo.status = "확인 완료"
        photo.error = "previous diagnostic"
        photo.revision = 3
        Cache(self.root).put(photo)
        loaded = Cache(self.root).get(photo.path)
        self.assertEqual(loaded, photo)
        self.assertEqual(legacy.read_bytes(), b"unknown legacy data")

    def test_changed_content_invalidates_manual_results(self):
        photo = self.photo()
        photo.manual = photo.confirmed = True
        cache = Cache(self.root)
        cache.put(photo)
        photo.path.write_bytes(b"other contents")
        os.utime(photo.path, ns=(photo.mtime_ns + 1, photo.mtime_ns + 1))
        self.assertIsNone(cache.get(photo.path))

    def test_cache_supports_worker_threads(self):
        photos = [self.photo(f"{index}.jpg") for index in range(12)]
        cache = Cache(self.root)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(cache.put, photos))
            loaded = list(executor.map(lambda photo: cache.get(photo.path), photos))
        self.assertEqual(loaded, photos)

    def test_bulk_cache_read_uses_one_connection_and_rejects_changed_fingerprints(self):
        photos = [self.photo(f"{index}.jpg") for index in range(20)]
        photos[0].manual, photos[0].revision = True, 5
        cache = Cache(self.root)
        cache.put_many(photos)
        paths = {photo.path: (photo.size, photo.mtime_ns) for photo in photos}
        paths[photos[-1].path] = (photos[-1].size + 1, photos[-1].mtime_ns)
        with patch.object(cache, "_connect", wraps=cache._connect) as connect:
            loaded = cache.get_many(paths)
        self.assertEqual(connect.call_count, 1)
        self.assertEqual(loaded, {str(photo.path): photo for photo in photos[:-1]})
        self.assertEqual(cache.get_many({}), {})

    def test_settings_are_persistent_and_recursively_exclude_secrets(self):
        store = SettingsStore(self.root / "config")
        store.save({"folder": "D:/사진", "provider": "OpenAI", "api_key": "test-only-key",
                    "nested": {"NVIDIA_API_KEY": "another-test-key", "model": "vision"},
                    "items": [{"token": "token-value", "value": 4}]})
        self.assertEqual(store.load(), {"folder": "D:/사진", "provider": "OpenAI",
                                      "nested": {"model": "vision"}, "items": [{"value": 4}]})
        self.assertNotIn("test-only-key", store.path.read_text(encoding="utf-8"))
        store.save({"folder": "D:/새폴더"})
        self.assertEqual(SettingsStore(store.directory).load(), {"folder": "D:/새폴더"})
        self.assertEqual(list(store.directory.glob("*.tmp")), [])

    def test_corrupted_settings_are_preserved(self):
        store = SettingsStore(self.root)
        store.path.write_text("unfinished json", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            store.load()
        self.assertEqual(store.path.read_text(), "unfinished json")

    def test_keystore_uses_secure_backend_and_normalizes_provider(self):
        values = {}
        backend = types.SimpleNamespace(
            get_password=lambda service, provider: values.get((service, provider)),
            set_password=lambda service, provider, value: values.__setitem__((service, provider), value),
            delete_password=lambda service, provider: values.pop((service, provider)),
        )
        store = KeyStore()
        with patch.object(KeyStore, "_backend", return_value=backend):
            self.assertEqual(store.get("OpenAI"), "")
            store.set("OpenAI", "fake-test-key")
            self.assertEqual(store.get("openai"), "fake-test-key")
            store.delete("OPENAI")
            self.assertEqual(store.get("OpenAI"), "")
            store.delete("OpenAI")

    def test_plaintext_keyring_backend_is_rejected(self):
        backend_type = type("PlaintextKeyring", (), {"__module__": "keyrings.alt.file", "priority": 1})
        fake = types.SimpleNamespace(get_keyring=lambda: backend_type())
        with patch.dict(sys.modules, {"keyring": fake}):
            with self.assertRaises(KeyStoreError):
                KeyStore().set("OpenAI", "fake-test-key")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_windows_loads_credential_manager_without_entrypoint_discovery(self):
        values = {}
        backend_type = type("WinVaultKeyring", (), {
            "priority": 5,
            "get_password": lambda self, service, provider: values.get((service, provider)),
            "set_password": lambda self, service, provider, value: values.__setitem__((service, provider), value),
            "delete_password": lambda self, service, provider: values.pop((service, provider)),
        })
        keyring_module = types.ModuleType("keyring")
        keyring_module.__path__ = []
        keyring_module.get_keyring = lambda: self.fail("Windows must not depend on entry-point discovery")
        backends_module = types.ModuleType("keyring.backends")
        backends_module.__path__ = []
        windows_module = types.ModuleType("keyring.backends.Windows")
        windows_module.WinVaultKeyring = backend_type
        modules = {"keyring": keyring_module, "keyring.backends": backends_module,
                   "keyring.backends.Windows": windows_module}
        with patch.dict(sys.modules, modules), patch.object(os, "name", "nt"):
            store = KeyStore()
            store.set("NVIDIA", "fake-test-key")
            self.assertEqual(store.get("nvidia"), "fake-test-key")
            store.delete("NVIDIA")
            self.assertEqual(store.get("NVIDIA"), "")


if __name__ == "__main__":
    unittest.main()
