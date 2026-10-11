"""No-clobber filename transactions with crash recovery and persistent undo."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterator
import uuid

from .domain import sanitize_filename
from .models import Photo
from .storage import Cache, atomic_json, photo_from_dict, photo_to_dict


class RenameError(RuntimeError):
    pass


_PENDING = ".frametag_pending_v2.json"
_HISTORY = "frametag_history_v2.json"
_LOCK = ".frametag_v2.lock"


@contextmanager
def _operation_lock(root: Path) -> Iterator[None]:
    """OS locks release automatically if the application terminates."""
    with (root / _LOCK).open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RenameError("다른 FrameTag 작업이 이 폴더의 이름을 변경하고 있습니다. 잠시 후 다시 시도해 주세요.") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _root(root: Path, cache: Cache) -> Path:
    root = Path(root).resolve()
    if not root.is_dir() or cache.root != root:
        raise RenameError("선택한 사진 폴더와 캐시 폴더가 일치하지 않습니다.")
    return root


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError as exc:
        raise RenameError("선택한 폴더 밖의 사진은 변경할 수 없습니다.") from exc


def _path(root: Path, value: str) -> Path:
    if not isinstance(value, str):
        raise RenameError("복구 기록의 경로 형식이 올바르지 않습니다.")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RenameError("복구 기록에 안전하지 않은 경로가 있습니다.")
    path = root / relative
    _relative(root, path.resolve())
    return path


def _fingerprint(path: Path) -> dict[str, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RenameError(f"사진 파일을 확인할 수 없습니다: {path.name}. 파일 존재 여부와 접근 권한을 확인해 주세요.") from exc
    if not stat.S_ISREG(info.st_mode):
        raise RenameError("일반 사진 파일만 이름을 변경할 수 있습니다.")
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns, "dev": info.st_dev, "ino": info.st_ino}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RenameError(f"사진 내용을 확인할 수 없습니다: {path.name}. 파일 사용 여부와 접근 권한을 확인해 주세요.") from exc
    return digest.hexdigest()


def _same_size_and_time(path: Path, entry: dict[str, Any]) -> bool:
    try:
        current = _fingerprint(path)
    except (OSError, RenameError):
        return False
    return all(current[key] == entry[key] for key in ("size", "mtime_ns"))


def _matches(path: Path, entry: dict[str, Any]) -> bool:
    """Is this the photo the entry recorded?

    The file ID settles it on NTFS. On FAT32/exFAT Windows derives that ID from
    the directory slot, so each rename changes it; there the recorded content
    hash decides. Size and modification time must agree in both cases.
    """
    try:
        current = _fingerprint(path)
        if any(current[key] != entry[key] for key in ("size", "mtime_ns")):
            return False
        if all(current[key] == entry[key] for key in ("dev", "ino")):
            return True
        digest = entry.get("sha256")
        return bool(digest) and _sha256(path) == digest
    except (OSError, RenameError):
        return False


def _same_file(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def recovery_pending(root: Path) -> bool:
    """True while an interrupted rename still waits for recovery in this folder."""
    return os.path.lexists(Path(root) / _PENDING)


def _exists(path: Path) -> bool:
    return os.path.lexists(path)


def _move_windows(source: Path, target: Path, entry: dict[str, Any]) -> None:
    try:
        # Unlike POSIX rename, Windows os.rename fails if destination exists.
        # exFAT/FAT32 drives have no hard links, so this is their only safe move.
        os.rename(source, target)
    except FileExistsError as exc:
        raise RenameError(f"같은 이름의 파일이 생겨 변경을 중단했습니다: {target.name}") from exc
    except OSError as exc:
        raise RenameError(f"파일 이름을 변경하지 못했습니다: {source.name}. 파일 사용 여부와 폴더 권한을 확인해 주세요.") from exc
    if not _matches(target, entry):
        raise RenameError(f"변경 중 파일 내용이 바뀌었습니다: {source.name}")


def _move_no_clobber(source: Path, target: Path, entry: dict[str, Any]) -> None:
    """Use Windows' exclusive rename or POSIX exclusive link; never replace a file."""
    if not _matches(source, entry):
        raise RenameError(f"파일이 변경되었습니다. 다시 확인해 주세요: {source.name}")
    if os.name == "nt":
        _move_windows(source, target, entry)
        return
    try:
        os.link(source, target, follow_symlinks=False)
    except FileExistsError as exc:
        raise RenameError(f"같은 이름의 파일이 생겨 변경을 중단했습니다: {target.name}") from exc
    except OSError as exc:
        raise RenameError(
            f"안전하게 파일 이름을 변경하지 못했습니다: {source.name}. "
            "파일 사용 여부, 폴더 권한, 하드 링크를 지원하는 파일시스템인지 확인해 주세요."
        ) from exc
    if not _matches(target, entry) or not _matches(source, entry):
        # Remove only the link we just created if its identity still matches the source.
        try:
            if os.path.samefile(source, target):
                target.unlink()
        except OSError:
            pass
        raise RenameError(f"변경 중 파일 내용이 바뀌었습니다: {source.name}")
    try:
        source.unlink()
    except OSError:
        # The original still exists. Discard only our own matching destination link.
        if _matches(source, entry) and _matches(target, entry):
            target.unlink()
        raise


def _read_json(path: Path, default: Any) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError:
        return default
    except (ValueError, UnicodeError) as exc:
        raise RenameError(f"기록 파일을 읽지 못했습니다. 파일을 보존하고 확인해 주세요: {path.name}") from exc


def _history(root: Path) -> list[dict[str, Any]]:
    history = _read_json(root / _HISTORY, [])
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise RenameError("이름 변경 기록의 형식이 올바르지 않습니다.")
    return history


def _validate_transaction(root: Path, transaction: Any) -> dict[str, Any]:
    if not isinstance(transaction, dict) or transaction.get("version") != 2:
        raise RenameError("복구 기록의 버전이 올바르지 않습니다.")
    transaction_id = transaction.get("id")
    try:
        if uuid.UUID(hex=transaction_id).hex != transaction_id:
            raise ValueError
    except (ValueError, TypeError, AttributeError) as exc:
        raise RenameError("복구 기록의 작업 번호가 올바르지 않습니다.") from exc
    if transaction.get("kind") not in {"rename", "undo"} or transaction.get("phase") not in {
        "prepared", "staging", "committing", "files_committed", "metadata_done", "rolling_back"
    }:
        raise RenameError("복구 기록의 작업 상태가 올바르지 않습니다.")
    entries = transaction.get("entries")
    if not isinstance(entries, list):
        raise RenameError("복구 기록의 파일 목록이 올바르지 않습니다.")
    seen: set[Path] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RenameError("복구 기록의 파일 정보가 올바르지 않습니다.")
        try:
            source = _path(root, entry["source"])
            target = _path(root, entry["target"])
            temporary = _path(root, entry["temporary"])
            if source.parent != target.parent or temporary != source.parent / f".frametag-{transaction_id}-{index}.tmp":
                raise RenameError("복구 기록의 임시 파일 경로가 올바르지 않습니다.")
            if source in seen or str(source) == str(target):
                raise RenameError("복구 기록에 중복된 파일이 있습니다.")
            seen.add(source)
            if any(not isinstance(entry[key], int) for key in ("size", "mtime_ns", "dev", "ino")):
                raise ValueError
            digest = entry.get("sha256")
            if digest is not None and not (isinstance(digest, str) and len(digest) == 64
                                           and all(char in "0123456789abcdef" for char in digest)):
                raise ValueError
            photo_from_dict(entry["before"])
            photo_from_dict(entry["after"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RenameError("복구 기록의 사진 정보가 올바르지 않습니다.") from exc
    return transaction


def _records(root: Path, transaction: dict[str, Any], reverse: bool = False) -> list[tuple[Path, Path, Photo]]:
    records = []
    for entry in transaction["entries"]:
        source = _path(root, entry["source"])
        target = _path(root, entry["target"])
        photo = photo_from_dict(entry["before"] if reverse else entry["after"])
        photo.path = source if reverse else target
        if reverse:
            records.append((target, source, photo))
        else:
            records.append((source, target, photo))
    if not reverse:
        for data in transaction.get("stationary", []):
            photo = photo_from_dict(data)
            _relative(root, photo.path.resolve())
            records.append((photo.path, photo.path, photo))
    return records


def _finish(root: Path, cache: Cache, transaction: dict[str, Any]) -> None:
    for entry in transaction["entries"]:
        if not _matches(_path(root, entry["target"]), entry):
            raise RenameError("변경된 사진이 기록과 다릅니다. 복구 기록을 보존하고 파일을 확인해 주세요.")
    history = _history(root)
    cache.relocate_many(_records(root, transaction))
    if transaction["kind"] == "rename":
        if not any(item.get("id") == transaction["id"] for item in history):
            history.append({"id": transaction["id"], "entries": transaction["entries"]})
    else:
        # Idempotent when a previous attempt saved history before terminating.
        history = [item for item in history if item.get("id") != transaction["undo_id"]]
    atomic_json(root / _HISTORY, history[-50:])
    transaction["phase"] = "metadata_done"
    atomic_json(root / _PENDING, transaction)
    (root / _PENDING).unlink()


def _rollback(root: Path, cache: Cache, transaction: dict[str, Any]) -> None:
    """Stage every original inode before restoring names, including name-swap cycles."""
    entries = transaction["entries"]
    for entry in entries:
        paths = {_path(root, entry[key]) for key in ("source", "temporary", "target")}
        if not any(_matches(path, entry) for path in paths):
            raise RenameError("원래 사진을 안전하게 찾지 못했습니다. 임시 파일과 복구 기록을 보존해 주세요.")
    # Never begin recovery if an unrelated file occupies an original name.
    for entry in entries:
        source = _path(root, entry["source"])
        if _exists(source) and not any(_matches(source, candidate) for candidate in entries):
            raise RenameError(f"원래 이름에 다른 파일이 있어 복구를 중단했습니다: {source.name}")
    transaction["phase"] = "rolling_back"
    atomic_json(root / _PENDING, transaction)
    for entry in entries:
        temporary = _path(root, entry["temporary"])
        if not _matches(temporary, entry):
            candidates = [_path(root, entry[key]) for key in ("source", "target")]
            source = next((path for path in candidates if _matches(path, entry)), None)
            if source is None:
                raise RenameError("사진 복구 위치를 찾을 수 없습니다.")
            _move_no_clobber(source, temporary, entry)
        # A crash between link and unlink can leave two names for the same inode.
        # A separate file with identical content is someone's copy: never remove it.
        for key in ("source", "target"):
            path = _path(root, entry[key])
            if path != temporary and _matches(path, entry) and _same_file(path, temporary):
                path.unlink()
    for entry in entries:
        source = _path(root, entry["source"])
        _move_no_clobber(_path(root, entry["temporary"]), source, entry)
    cache.relocate_many(_records(root, transaction, reverse=True))
    (root / _PENDING).unlink()


def _adopt_content_identity(root: Path, transaction: dict[str, Any]) -> None:
    """Give journals written before 2.0.7 the content hash later checks rely on.

    Those journals identify a photo by file ID alone, which FAT32/exFAT change
    on rename, leaving the staged photo unrecognised. The three names belong to
    this transaction, so one file among them with the recorded size and time is
    that photo; anything less certain is left for the existing refusal.
    """
    changed = False
    for entry in transaction["entries"]:
        if entry.get("sha256"):
            continue
        paths = list(dict.fromkeys(_path(root, entry[key]) for key in ("temporary", "source", "target")))
        found = [path for path in paths if _matches(path, entry)]
        if not found:
            found = [path for path in paths if _same_size_and_time(path, entry)]
        if len(found) == 1 or (found and all(_same_file(found[0], path) for path in found)):
            entry["sha256"] = _sha256(found[0])
            changed = True
    if changed:
        atomic_json(root / _PENDING, transaction)


def _recover_pending(root: Path, cache: Cache) -> None:
    transaction = _read_json(root / _PENDING, None)
    if transaction is None:
        return
    transaction = _validate_transaction(root, transaction)
    _adopt_content_identity(root, transaction)
    if transaction["phase"] in {"files_committed", "metadata_done"}:
        _finish(root, cache, transaction)
    else:
        _rollback(root, cache, transaction)


def recover_pending(root: Path, cache: Cache) -> None:
    """Complete committed metadata or safely undo an interrupted file transaction."""
    root = _root(root, cache)
    with _operation_lock(root):
        _recover_pending(root, cache)


def _execute(root: Path, cache: Cache, transaction: dict[str, Any]) -> None:
    atomic_json(root / _PENDING, transaction)
    try:
        transaction["phase"] = "staging"
        atomic_json(root / _PENDING, transaction)
        for entry in transaction["entries"]:
            _move_no_clobber(_path(root, entry["source"]), _path(root, entry["temporary"]), entry)
        transaction["phase"] = "committing"
        atomic_json(root / _PENDING, transaction)
        for entry in transaction["entries"]:
            _move_no_clobber(_path(root, entry["temporary"]), _path(root, entry["target"]), entry)
        transaction["phase"] = "files_committed"
        atomic_json(root / _PENDING, transaction)
    except Exception as exc:
        try:
            _rollback(root, cache, transaction)
        except Exception as rollback_error:
            raise RenameError(
                "이름 변경을 완료하지 못했고 자동 복구도 중단되었습니다. 사진과 임시 파일, "
                "복구 기록을 보존한 뒤 다시 시도해 주세요. " + str(rollback_error)
            ) from exc
        raise RenameError("이름 변경을 완료하지 못해 원래 이름으로 복구했습니다. " + str(exc)) from exc
    try:
        _finish(root, cache, transaction)
    except Exception as exc:
        raise RenameError(
            "사진 이름은 변경되었지만 기록 저장이 완료되지 않았습니다. "
            "복구 기록을 삭제하지 말고 폴더를 다시 열어 기록 저장을 재시도해 주세요."
        ) from exc


def _copy_photo(target: Photo, source: Photo) -> None:
    for key, value in vars(source).items():
        setattr(target, key, value)


def apply_names(root: Path, photos: list[Photo], cache: Cache) -> list[tuple[Path, Path]]:
    root = _root(root, cache)
    if not photos:
        return []
    with _operation_lock(root):
        _recover_pending(root, cache)
        _history(root)  # Validate undo history before touching photo files.
        sources: set[Path] = set()
        identities: set[tuple[int, int]] = set()
        prepared: list[tuple[Photo, Path, str, dict[str, int]]] = []
        for photo in photos:
            raw_path = Path(photo.path)
            if raw_path.is_symlink():
                raise RenameError("바로가기나 심볼릭 링크는 변경할 수 없습니다.")
            source = raw_path.resolve()
            _relative(root, source)
            if source in sources:
                raise RenameError("같은 사진이 두 번 선택되었습니다.")
            fingerprint = _fingerprint(source)
            identity = (fingerprint["dev"], fingerprint["ino"])
            if identity in identities:
                raise RenameError("같은 파일을 가리키는 하드 링크가 여러 개 선택되었습니다. 한 파일씩 선택해 주세요.")
            identities.add(identity)
            if fingerprint["size"] != photo.size or fingerprint["mtime_ns"] != photo.mtime_ns:
                raise RenameError(f"분석 후 파일 내용이 바뀌었습니다. 다시 분석해 주세요: {source.name}")
            try:
                filename = sanitize_filename(photo.proposed_name, source.suffix)
            except ValueError as exc:
                raise RenameError(str(exc)) from exc
            sources.add(source)
            prepared.append((photo, source, filename, fingerprint))
        occupied: dict[Path, set[str]] = {}
        for parent in {source.parent for source in sources}:
            occupied[parent] = {item.name.casefold() for item in parent.iterdir() if item not in sources}
        transaction_id = uuid.uuid4().hex
        entries: list[dict[str, Any]] = []
        after_photos: list[tuple[Photo, Photo]] = []
        stationary: list[dict[str, Any]] = []
        next_suffix: dict[tuple[Path, str], int] = {}
        for photo, source, filename, fingerprint in prepared:
            names = occupied[source.parent]
            candidate = filename
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            suffix_key = (source.parent, filename.casefold())
            duplicate = next_suffix.get(suffix_key, 2)
            while candidate.casefold() in names:
                ending = f"_{duplicate}{suffix}"
                budget = 240 - len(ending.encode("utf-8"))
                trimmed = stem.encode("utf-8")[:budget].decode("utf-8", errors="ignore").rstrip(" .")
                candidate = sanitize_filename(f"{trimmed}{ending}", source.suffix)
                duplicate += 1
            next_suffix[suffix_key] = duplicate
            names.add(candidate.casefold())
            target = source.with_name(candidate)
            after = replace(photo, path=target, proposed_name=candidate, confirmed=True, status="확인 완료", error="", revision=photo.revision + 1)
            after_photos.append((photo, after))
            if str(target) == str(source):
                stationary.append(photo_to_dict(after))
                continue
            digest = _sha256(source)
            if _fingerprint(source) != fingerprint:
                raise RenameError(f"분석 후 파일 내용이 바뀌었습니다. 다시 분석해 주세요: {source.name}")
            entries.append({
                "source": _relative(root, source), "target": _relative(root, target),
                "temporary": _relative(root, source.parent / f".frametag-{transaction_id}-{len(entries)}.tmp"),
                **fingerprint, "sha256": digest,
                "before": photo_to_dict(photo), "after": photo_to_dict(after),
            })
        if not entries:
            cache.put_many([after for _, after in after_photos])
        else:
            transaction = {"version": 2, "id": transaction_id, "kind": "rename", "phase": "prepared", "entries": entries, "stationary": stationary}
            _execute(root, cache, transaction)
        for original, after in after_photos:
            _copy_photo(original, after)
        return [(_path(root, entry["source"]), _path(root, entry["target"])) for entry in entries]


def undo_last(root: Path, cache: Cache) -> list[tuple[Path, Path]]:
    root = _root(root, cache)
    with _operation_lock(root):
        _recover_pending(root, cache)
        history = _history(root)
        if not history:
            return []
        previous = history[-1]
        if not isinstance(previous.get("entries"), list) or not previous.get("id"):
            raise RenameError("최근 이름 변경 기록의 형식이 올바르지 않습니다.")
        transaction_id = uuid.uuid4().hex
        entries = []
        for index, prior in enumerate(previous["entries"]):
            try:
                source = _path(root, prior["target"])
                target = _path(root, prior["source"])
                fingerprint = _fingerprint(source)
                if not _matches(source, prior):
                    raise RenameError(f"이름 변경 후 사진 내용이 바뀌어 되돌릴 수 없습니다: {source.name}")
                digest = prior.get("sha256") or _sha256(source)
                before = cache.get(source) or photo_from_dict(prior["after"])
                after = photo_from_dict(prior["before"])
                after.path = target
                after.revision = max(after.revision, before.revision) + 1
                if source.parent != target.parent:
                    raise RenameError("이름 변경 기록의 폴더가 올바르지 않습니다.")
                entries.append({
                    "source": _relative(root, source), "target": _relative(root, target),
                    "temporary": _relative(root, source.parent / f".frametag-{transaction_id}-{index}.tmp"),
                    **fingerprint, "sha256": digest,
                    "before": photo_to_dict(before), "after": photo_to_dict(after),
                })
            except (KeyError, TypeError, ValueError) as exc:
                raise RenameError("이름 변경 기록의 사진 정보가 올바르지 않습니다.") from exc
        sources = {_path(root, entry["source"]) for entry in entries}
        occupied = {
            parent: {item.name.casefold() for item in parent.iterdir() if item not in sources}
            for parent in {source.parent for source in sources}
        }
        for entry in entries:
            target = _path(root, entry["target"])
            if target.name.casefold() in occupied[target.parent]:
                raise RenameError(f"원래 이름에 다른 파일이 있어 되돌릴 수 없습니다: {target.name}")
        transaction = {
            "version": 2, "id": transaction_id, "kind": "undo", "phase": "prepared",
            "entries": entries, "undo_id": previous["id"], "stationary": [],
        }
        _validate_transaction(root, transaction)
        _execute(root, cache, transaction)
        return [(_path(root, entry["source"]), _path(root, entry["target"])) for entry in entries]
