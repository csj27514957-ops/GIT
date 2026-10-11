"""Local settings, secure credentials, and content-aware analysis cache."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Iterator

from .models import Photo


def _config_directory() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "FrameTag"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "FrameTag"


def atomic_json(path: Path, data: Any) -> None:
    """Replace a JSON document only after its complete contents have reached disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


_SECRET_FIELDS = {
    "apikey", "apikeys", "token", "tokens", "accesstoken", "refreshtoken",
    "password", "passwords", "secret", "secrets", "credential", "credentials",
    "openaiapikey", "geminiapikey", "googleapikey", "anthropicapikey",
}


def _without_secrets(data: Any) -> Any:
    if isinstance(data, dict):
        return {
            key: _without_secrets(value)
            for key, value in data.items()
            if isinstance(key, str)
            and "".join(char for char in key.casefold() if char.isalnum()) not in _SECRET_FIELDS
            and not "".join(char for char in key.casefold() if char.isalnum()).endswith("apikey")
        }
    if isinstance(data, list):
        return [_without_secrets(value) for value in data]
    return data


class SettingsStore:
    """Settings contain preferences only; API keys belong in KeyStore."""

    def __init__(self, config_dir: Path | None = None):
        self.directory = Path(config_dir) if config_dir is not None else _config_directory()
        self.path = self.directory / "settings.json"

    def load(self) -> dict[str, Any]:
        try:
            with self.path.open(encoding="utf-8") as stream:
                data = json.load(stream)
        except FileNotFoundError:
            return {}
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("설정 파일을 읽을 수 없습니다. 설정 파일을 백업한 뒤 다시 저장해 주세요.") from exc
        if not isinstance(data, dict):
            raise RuntimeError("설정 파일의 형식이 올바르지 않습니다.")
        return _without_secrets(data)

    def save(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise TypeError("settings must be a dictionary")
        atomic_json(self.path, _without_secrets(data))


class KeyStoreError(RuntimeError):
    """Secure credential storage is unavailable or rejected an operation."""


class KeyStore:
    service = "FrameTag.v2"

    @staticmethod
    def _backend() -> Any:
        try:
            import keyring

            if os.name == "nt":
                # Frozen applications may not retain package entry-point metadata.
                # Load Windows Credential Manager directly, without changing a global backend.
                from keyring.backends.Windows import WinVaultKeyring

                backend = WinVaultKeyring()
                if backend.priority > 0:
                    return backend
                raise KeyStoreError("Windows 자격 증명 관리자를 사용할 수 없습니다.")
            backend = keyring.get_keyring()
            candidates = getattr(backend, "backends", [backend])
            secure_modules = (
                "keyring.backends.Windows", "keyring.backends.macOS",
                "keyring.backends.SecretService", "keyring.backends.kwallet",
            )
            secure = [
                candidate for candidate in candidates
                if type(candidate).__module__.startswith(secure_modules)
                and candidate.priority > 0
            ]
            if secure:
                return max(secure, key=lambda candidate: candidate.priority)
        except Exception as exc:
            raise KeyStoreError(
                "안전한 API 키 저장소를 사용할 수 없습니다. Windows 자격 증명 관리자와 keyring 설치를 확인하거나 "
                "환경 변수로 API 키를 설정해 주세요. API 키는 평문 파일에 저장되지 않습니다."
            ) from exc
        raise KeyStoreError(
            "안전한 API 키 저장소가 없습니다. Windows에서는 자격 증명 관리자를 사용하며, Linux에서는 "
            "Secret Service 같은 시스템 키 저장소가 필요합니다. 환경 변수로 API 키를 설정할 수도 있습니다."
        )

    @staticmethod
    def _provider(provider: str) -> str:
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("API 제공자를 선택해 주세요.")
        return provider.strip().casefold()

    def get(self, provider: str) -> str:
        try:
            return self._backend().get_password(self.service, self._provider(provider)) or ""
        except KeyStoreError:
            raise
        except Exception as exc:
            raise KeyStoreError("저장된 API 키를 읽지 못했습니다. 시스템 자격 증명 저장소를 확인해 주세요.") from exc

    def set(self, provider: str, key: str) -> None:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("저장할 API 키를 입력해 주세요.")
        try:
            self._backend().set_password(self.service, self._provider(provider), key.strip())
        except KeyStoreError:
            raise
        except Exception as exc:
            raise KeyStoreError("API 키를 안전하게 저장하지 못했습니다. 시스템 자격 증명 저장소를 확인해 주세요.") from exc

    def delete(self, provider: str) -> None:
        try:
            backend = self._backend()
            account = self._provider(provider)
            if backend.get_password(self.service, account) is not None:
                backend.delete_password(self.service, account)
        except KeyStoreError:
            raise
        except Exception as exc:
            raise KeyStoreError("저장된 API 키를 삭제하지 못했습니다. 시스템 자격 증명 저장소를 확인해 주세요.") from exc


def photo_to_dict(photo: Photo) -> dict[str, Any]:
    return {
        "path": str(photo.path), "size": photo.size, "mtime_ns": photo.mtime_ns,
        "analysis": photo.analysis, "proposed_name": photo.proposed_name,
        "status": photo.status, "manual": photo.manual, "confirmed": photo.confirmed,
        "model": photo.model, "error": photo.error, "revision": photo.revision,
    }


def photo_from_dict(data: dict[str, Any]) -> Photo:
    return Photo(
        path=Path(data["path"]), size=int(data["size"]), mtime_ns=int(data["mtime_ns"]),
        analysis=data.get("analysis"), proposed_name=str(data.get("proposed_name", "")),
        status=str(data.get("status", "대기")), manual=bool(data.get("manual", False)),
        confirmed=bool(data.get("confirmed", False)), model=str(data.get("model", "")),
        error=str(data.get("error", "")), revision=int(data.get("revision", 0)),
    )


class Cache:
    """Use a separate versioned database without migrating or touching legacy data."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError("사진 폴더가 존재하지 않습니다.")
        self.path = self.root / ".frametag_v2.sqlite3"
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS photos (path TEXT PRIMARY KEY, size INTEGER NOT NULL, "
                "mtime_ns INTEGER NOT NULL, data TEXT NOT NULL)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Each operation owns its connection. Workers never share SQLite connection objects.
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _key(self, path: Path) -> str:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("사진 경로가 선택한 폴더 밖에 있습니다.") from exc
        return os.path.normcase(str(relative))

    def get(self, path: Path) -> Photo | None:
        path = Path(path).resolve()
        key = self._key(path)
        try:
            fingerprint = path.stat()
        except (FileNotFoundError, NotADirectoryError):
            return None
        with self._connect() as connection:
            row = connection.execute("SELECT size, mtime_ns, data FROM photos WHERE path=?", (key,)).fetchone()
        if not row or row[0] != fingerprint.st_size or row[1] != fingerprint.st_mtime_ns:
            return None
        photo = photo_from_dict(json.loads(row[2]))
        photo.path = path
        return photo

    def put(self, photo: Photo) -> None:
        self.put_many([photo])

    def put_many(self, photos: list[Photo]) -> None:
        rows = [
            (self._key(photo.path), photo.size, photo.mtime_ns, json.dumps(photo_to_dict(photo), ensure_ascii=False))
            for photo in photos
        ]
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO photos (path, size, mtime_ns, data) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET size=excluded.size, mtime_ns=excluded.mtime_ns, data=excluded.data",
                rows,
            )

    def relocate(self, old: Path, new: Path, photo: Photo) -> None:
        if Path(photo.path).resolve() != Path(new).resolve():
            raise ValueError("새 경로와 사진 기록이 일치하지 않습니다.")
        with self._connect() as connection:
            connection.execute("DELETE FROM photos WHERE path=?", (self._key(old),))
            connection.execute(
                "INSERT INTO photos (path, size, mtime_ns, data) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET size=excluded.size, mtime_ns=excluded.mtime_ns, data=excluded.data",
                (self._key(new), photo.size, photo.mtime_ns, json.dumps(photo_to_dict(photo), ensure_ascii=False)),
            )

    def relocate_many(self, records: list[tuple[Path, Path, Photo]]) -> None:
        """Delete all old keys first so filename swaps cannot discard another row."""
        with self._connect() as connection:
            for old, new, photo in records:
                if Path(photo.path).resolve() != Path(new).resolve():
                    raise ValueError("새 경로와 사진 기록이 일치하지 않습니다.")
                connection.execute("DELETE FROM photos WHERE path=?", (self._key(old),))
            for old, new, photo in records:
                connection.execute(
                    "INSERT INTO photos (path, size, mtime_ns, data) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(path) DO UPDATE SET size=excluded.size, mtime_ns=excluded.mtime_ns, data=excluded.data",
                    (self._key(new), photo.size, photo.mtime_ns, json.dumps(photo_to_dict(photo), ensure_ascii=False)),
                )
