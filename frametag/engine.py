"""Sequential analysis and persistent, copy-safe folder watching.

The callback receives immutable-by-convention snapshots and never touches a GUI.
All API calls run on one cancellable worker; rename operations share its lock.
"""
from __future__ import annotations

import copy
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .api import APIError, Analyzer, Cancelled
from .domain import (filename_from_analysis, list_images, natural_key,
                     normalize_analysis, sanitize_filename, select_through,
                     unify_runs)
from .models import Photo
from .rename import apply_names, recover_pending, undo_last
from .storage import Cache, atomic_json


class Engine:
    def __init__(self, callback: Callable[[str, Any], None], interval: float = 1.0,
                 stable_scans: int = 3, analyzer_factory: Callable[..., Any] = Analyzer):
        self._callback = callback
        self._interval = max(0.02, float(interval))
        self._stable_scans = max(2, int(stable_scans))
        self._factory = analyzer_factory
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._cancel = threading.Event()
        self._closed = False
        self._busy = False
        self._operation = ""
        self._manual_requested = False
        self._auto = False
        self._config = ("", "", "")
        self._root: Path | None = None
        self._recursive = False
        self._cache: Cache | None = None
        self._photos: dict[str, Photo] = {}
        self._known: dict[str, dict[str, Any]] = {}
        self._stable: dict[str, tuple[tuple[int, int], int]] = {}
        self._last_poll = 0.0
        self._consecutive_failures = 0
        self._manifest_error = ""
        self._worker = threading.Thread(target=self._run, name="FrameTag-worker", daemon=True)
        self._worker.start()

    def _emit(self, event: str, value: Any) -> None:
        self._callback(event, value)

    def _ordered(self) -> list[Photo]:
        return sorted(self._photos.values(), key=lambda p: (
            not p.confirmed, natural_key(str(p.path.parent)), natural_key(p.path.name)))

    def get_photos(self) -> list[Photo]:
        with self._lock:
            return copy.deepcopy(self._ordered())

    def get_root(self) -> Path | None:
        with self._lock:
            return self._root

    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def _publish(self) -> None:
        self._emit("photos", copy.deepcopy(self._ordered()))

    def _key(self, path: Path) -> str:
        assert self._root is not None
        return path.relative_to(self._root).as_posix()

    @staticmethod
    def _fingerprint(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns

    @staticmethod
    def _record_fingerprint(record: dict[str, Any]) -> tuple[int, int]:
        return record.get("size", -1), record.get("mtime_ns", -1)

    @staticmethod
    def _record(photo: Photo, state: str) -> dict[str, Any]:
        return {"size": photo.size, "mtime_ns": photo.mtime_ns,
                "state": state, "present": True}

    def _save_manifest(self) -> None:
        if self._root is None:
            return
        try:
            atomic_json(self._root / ".frametag_watch_v2.json", {
                "version": 2, "recursive": self._recursive, "entries": self._known})
            self._manifest_error = ""
        except OSError as exc:
            message = f"폴더 감시 기록을 저장하지 못했습니다: {exc}"
            if message != self._manifest_error:
                self._emit("error", message)
            self._manifest_error = message

    def _load_manifest(self, root: Path) -> tuple[dict[str, dict[str, Any]], bool, bool]:
        path = root / ".frametag_watch_v2.json"
        if not path.exists():
            return {}, True, False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") != 2 or not isinstance(data.get("entries"), dict):
                raise ValueError("알 수 없는 감시 기록 형식")
            entries = data["entries"]
            for name, record in entries.items():
                if not isinstance(name, str) or not isinstance(record, dict):
                    raise ValueError("손상된 감시 기록")
                if record.get("state") not in {"baseline", "pending", "error", "processed"}:
                    raise ValueError("손상된 감시 상태")
            return entries, False, bool(data.get("recursive", False))
        except (OSError, ValueError, TypeError) as exc:
            # Preserve a malformed local file before establishing a safe new baseline.
            backup = path.with_name(f"{path.name}.broken-{time.time_ns()}")
            try:
                path.rename(backup)
            except OSError:
                raise ValueError(f"감시 기록을 읽거나 보존할 수 없습니다: {exc}") from exc
            self._emit("error", f"감시 기록이 손상되어 원본을 {backup.name}에 보존했습니다.")
            return {}, True, False

    def load_folder(self, root: Path, recursive: bool = False) -> bool:
        with self._lock:
            if self._busy or self._closed:
                self._emit("error", "분석이나 파일 변경이 끝난 후 폴더를 불러오세요.")
                return False
            root = Path(root).expanduser().resolve()
            if not root.is_dir():
                self._emit("error", "사진 폴더가 존재하지 않습니다.")
                return False
            old_state = (self._root, self._recursive, self._cache, self._known,
                         self._photos, self._stable, self._last_poll,
                         self._consecutive_failures)
            self._busy = True
            self._emit("busy", True)
            try:
                cache = Cache(root)
                recover_pending(root, cache)
                known, first_baseline, previous_recursive = self._load_manifest(root)
                self._root, self._recursive, self._cache = root, bool(recursive), cache
                self._known, self._photos, self._stable = known, {}, {}
                self._last_poll = 0.0
                self._consecutive_failures = 0
                self._poll(first_baseline=first_baseline,
                           baseline_subfolders=bool(recursive) and not previous_recursive)
                self._emit("status", f"사진 {len(self._photos):,}장을 불러왔습니다.")
            except Exception as exc:
                (self._root, self._recursive, self._cache, self._known,
                 self._photos, self._stable, self._last_poll,
                 self._consecutive_failures) = old_state
                if self._auto:
                    self._auto = False
                    self._emit("auto", False)
                self._publish()
                self._emit("error", f"폴더를 불러오지 못했습니다: {exc}")
                return False
            finally:
                self._busy = False
                self._emit("busy", False)
        self._wake.set()
        return True

    def configure(self, provider: str, model: str, key: str) -> None:
        with self._lock:
            config = (provider, model, key)
            if config != self._config:
                self._config = config
                self._consecutive_failures = 0
                # A corrected provider/key/model can resume outstanding error photos.
                changed = False
                for record in self._known.values():
                    if record.get("state") == "error":
                        record["state"] = "pending"
                        changed = True
                if changed:
                    self._save_manifest()
        self._wake.set()

    def set_auto(self, enabled: bool) -> None:
        with self._lock:
            enabled = bool(enabled) and not self._closed
            if enabled == self._auto:
                return
            self._auto = enabled
            if not enabled and self._operation == "auto":
                self._cancel.set()
            self._emit("auto", enabled)
            self._emit("status", "새 사진 자동 감시를 시작합니다." if enabled else "자동 감시를 중지했습니다.")
        self._wake.set()

    def start_analysis(self) -> None:
        with self._lock:
            if self._closed or self._busy:
                return
            if self._root is None:
                self._emit("error", "먼저 사진 폴더를 선택하세요.")
                return
            if not all(self._config):
                self._emit("error", "API 공급자, 모델과 API 키를 설정하세요.")
                return
            self._cancel = threading.Event()
            self._busy, self._operation, self._manual_requested = True, "manual", True
            self._emit("busy", True)
        self._wake.set()

    def stop_analysis(self) -> None:
        with self._lock:
            self._cancel.set()
            if self._auto:
                self._auto = False
                self._emit("auto", False)
            self._emit("status", "분석 중지를 요청했습니다.")
        self._wake.set()

    @staticmethod
    def _needs_analysis(photo: Photo) -> bool:
        return not photo.confirmed and not photo.manual and (
            not photo.analysis or not photo.proposed_name
            or (bool(photo.error) and photo.status != "파일명 오류"))

    def _poll(self, first_baseline: bool = False, baseline_subfolders: bool = False) -> None:
        """Called under the shared lock; size/mtime stability is measured per scan."""
        if self._root is None or self._cache is None:
            return
        self._last_poll = time.monotonic()
        paths = list_images(self._root, self._recursive)
        present: set[str] = set()
        changed, manifest_changed = False, False
        for path in paths:
            path = path.resolve()
            try:
                fingerprint = self._fingerprint(path)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            name, identifier = self._key(path), str(path)
            present.add(name)
            previous_fingerprint, count = self._stable.get(name, (fingerprint, 0))
            self._stable[name] = (fingerprint, count + 1 if previous_fingerprint == fingerprint else 1)
            photo = self._photos.get(identifier)
            if photo is None or (photo.size, photo.mtime_ns) != fingerprint:
                photo = self._cache.get(path) or Photo(path, *fingerprint)
                self._photos[identifier] = photo
                changed = True
            previous = self._known.get(name)
            if (previous is None or self._record_fingerprint(previous) != fingerprint
                    or not previous.get("present", True)):
                added_scope_baseline = baseline_subfolders and previous is None and "/" in name
                state = "baseline" if first_baseline or added_scope_baseline else "pending"
                if photo.confirmed or photo.manual:
                    state = "processed"
                self._known[name] = self._record(photo, state)
                manifest_changed = True
        for identifier, photo in list(self._photos.items()):
            if self._key(photo.path) not in present:
                del self._photos[identifier]
                changed = True
        for name, record in self._known.items():
            # Excluded subfolders are outside this scan's scope, rather than deleted.
            if not self._recursive and "/" in name:
                continue
            if name not in present and record.get("present", True):
                record["present"] = False
                self._stable.pop(name, None)
                manifest_changed = True
        if manifest_changed:
            self._save_manifest()
        if changed:
            self._publish()

    def _ready_auto_photo(self) -> str | None:
        for photo in self._ordered():
            name = self._key(photo.path)
            record = self._known.get(name, {})
            if record.get("state") != "pending" or photo.confirmed or photo.manual:
                continue
            fingerprint, count = self._stable.get(name, ((-1, -1), 0))
            if count < self._stable_scans or fingerprint[0] <= 0:
                continue
            # A copy may pause without finishing: never charge for an undecodable file.
            try:
                with Image.open(photo.path) as image:
                    # JPEG verify() checks headers only; load() also rejects a paused/truncated copy.
                    image.load()
                if self._fingerprint(photo.path) != fingerprint:
                    continue
            except (OSError, ValueError, Image.DecompressionBombError):
                continue
            return str(photo.path)
        return None

    def _mark(self, photo: Photo, state: str) -> None:
        self._known[self._key(photo.path)] = self._record(photo, state)
        self._save_manifest()

    def _pause_for_failure(self, message: str) -> None:
        if self._auto:
            self._auto = False
            self._emit("auto", False)
        self._emit("error", message)

    def _process(self, identifier: str, automatic: bool) -> str:
        with self._lock:
            photo = self._photos.get(identifier)
            if photo is None or self._cancel.is_set():
                return "cancelled"
            snapshot = copy.deepcopy(photo)
            config, cancel = self._config, self._cancel
            cache = self._cache
            assert cache is not None
            reuse = automatic and bool(photo.analysis and photo.proposed_name
                                       and (not photo.error or photo.status == "파일명 오류"))
            if not reuse:
                photo.status = "분석 중"
                photo.error = ""
                self._publish()
                self._emit("status", f"{'자동 ' if automatic else ''}분석: {photo.path.name}")
        try:
            if cancel.is_set():
                raise Cancelled()
            if reuse:
                analysis = snapshot.analysis
            else:
                if not all(config):
                    raise APIError("API 공급자, 모델과 API 키를 설정하세요.", fatal=True)
                analysis = normalize_analysis(self._factory(*config).analyze(snapshot.path, cancel))
            if cancel.is_set():
                raise Cancelled()
            with self._lock:
                photo = self._photos.get(identifier)
                if photo is None:
                    return "stale"
                if photo.revision != snapshot.revision:
                    # A user edit made during the request owns both the result and filename.
                    self._mark(photo, "processed")
                    self._publish()
                    return "edited"
                try:
                    fingerprint = self._fingerprint(snapshot.path)
                except OSError:
                    return "stale"
                if fingerprint != (snapshot.size, snapshot.mtime_ns):
                    self._poll()
                    return "stale"
                if cancel.is_set():
                    raise Cancelled()
                if not reuse:
                    photo.analysis = analysis
                    photo.proposed_name = filename_from_analysis(analysis or {}, photo.path.suffix)
                    photo.status, photo.error, photo.model = "분석 완료", "", config[1]
                    cache.put(photo)
                self._consecutive_failures = 0
                if automatic:
                    assert self._root is not None
                    # Analysis is cached before renaming. A restart can finish this without a second API call.
                    try:
                        pairs = apply_names(self._root, [photo], cache)
                    except Exception as exc:
                        photo.status, photo.error = "파일명 오류", str(exc)
                        cache.put(photo)
                        self._mark(photo, "error")
                        self._publish()
                        self._emit("error", f"자동 파일명 변경 실패: {exc}")
                        return "rename_error"
                    self._remap(pairs)
                    self._mark(photo, "processed")
                else:
                    self._mark(photo, "processed")
                self._publish()
                return "success"
        except Cancelled:
            with self._lock:
                photo = self._photos.get(identifier)
                if photo is not None and photo.revision == snapshot.revision:
                    photo.status = snapshot.status if snapshot.status != "분석 중" else "대기"
                    photo.error = snapshot.error
                    cache.put(photo)
                    self._publish()
            return "cancelled"
        except Exception as exc:
            with self._lock:
                photo = self._photos.get(identifier)
                if photo is not None and photo.revision == snapshot.revision:
                    photo.status, photo.error = "오류", str(exc)
                    cache.put(photo)
                    self._mark(photo, "error")
                    self._publish()
                self._consecutive_failures += 1
                fatal = bool(getattr(exc, "fatal", False))
                if fatal or self._consecutive_failures >= 3:
                    self._pause_for_failure(str(exc) if fatal else "3장 연속 오류로 분석을 중지했습니다. " + str(exc))
                    return "halt"
                self._emit("status", f"분석 오류: {snapshot.path.name} — {exc}")
            return "error"

    def _run_manual(self) -> None:
        with self._lock:
            identifiers = [str(p.path) for p in self._ordered() if self._needs_analysis(p)]
        halted = False
        for identifier in identifiers:
            if self._cancel.is_set() or self._closed:
                break
            result = self._process(identifier, automatic=False)
            if result in {"halt", "cancelled"}:
                halted = result == "halt"
                break
            with self._lock:
                if time.monotonic() - self._last_poll >= self._interval:
                    self._poll()
        with self._lock:
            if self._cache is not None:
                changed = unify_runs(sorted(self._photos.values(), key=lambda p: (
                    natural_key(str(p.path.parent)), natural_key(p.path.name))))
                if changed:
                    self._cache.put_many(changed)
                    self._publish()
            if halted:
                message = "오류로 분석을 중지했습니다. API 설정을 확인한 뒤 다시 시작하세요."
            elif self._cancel.is_set():
                message = "분석을 중지했습니다."
            else:
                message = "남은 사진 분석을 마쳤습니다."
            self._emit("status", message)

    def _run(self) -> None:
        while True:
            self._wake.wait(self._interval)
            self._wake.clear()
            with self._lock:
                if self._closed:
                    return
                manual = self._manual_requested
                if manual:
                    self._manual_requested = False
                elif self._busy:
                    continue
                else:
                    try:
                        if time.monotonic() - self._last_poll >= self._interval:
                            self._poll()
                        identifier = self._ready_auto_photo() if self._auto else None
                    except Exception as exc:
                        self._pause_for_failure(f"폴더 감시 오류: {exc}")
                        continue
                    if identifier is None:
                        continue
                    self._busy, self._operation = True, "auto"
                    self._cancel = threading.Event()
                    self._emit("busy", True)
            try:
                if manual:
                    self._run_manual()
                else:
                    self._process(identifier, automatic=True)
            except Exception as exc:
                with self._lock:
                    self._pause_for_failure(f"분석 작업 오류: {exc}")
            finally:
                with self._lock:
                    self._busy, self._operation = False, ""
                    self._emit("busy", False)
                    if self._auto:
                        self._emit("status", "새 사진을 감시하고 있습니다.")
                self._wake.set()

    def update_photo(self, path: str, field: str, value: str) -> None:
        with self._lock:
            photo = self._photos.get(str(Path(path).resolve()))
            if photo is None:
                return
            try:
                if field == "proposed_name":
                    name = sanitize_filename(value.strip(), photo.path.suffix)
                    photo.proposed_name = name
                elif field in {"gender", "clothing", "objects"}:
                    photo.analysis = copy.deepcopy(photo.analysis or {"gender": "불명", "clothing": [], "objects": []})
                    if field == "gender":
                        photo.analysis[field] = value.strip() or "불명"
                    else:
                        photo.analysis[field] = [part.strip() for part in re.split(r"[,，\n;]+", value)
                                                 if part.strip() and part.strip() not in {"없음", "-"}]
                    photo.proposed_name = filename_from_analysis(photo.analysis, photo.path.suffix)
                else:
                    raise ValueError("수정할 수 없는 항목입니다.")
                photo.manual, photo.confirmed = True, False
                photo.status, photo.error = "수동 수정", ""
                photo.revision += 1
                assert self._cache is not None
                self._cache.put(photo)
                self._mark(photo, "processed")
                self._publish()
            except Exception as exc:
                self._emit("error", str(exc))

    def unify_selected(self, paths: list[str], reference: str) -> None:
        with self._lock:
            base = self._photos.get(str(Path(reference).resolve()))
            if base is None or not base.analysis:
                self._emit("error", "분석된 기준 사진을 선택하세요.")
                return
            selected = [self._photos[p] for p in dict.fromkeys(str(Path(p).resolve()) for p in paths)
                        if p in self._photos and self._photos[p].analysis
                        and not self._photos[p].error and self._photos[p].status != "분석 중"]
            if len(selected) < 2:
                self._emit("error", "분석을 완료한 사진을 두 장 이상 선택하세요.")
                return
            changed: list[Photo] = []
            reference_analysis = copy.deepcopy(base.analysis)
            for photo in selected:
                photo.analysis = copy.deepcopy(reference_analysis)
                photo.proposed_name = filename_from_analysis(photo.analysis, photo.path.suffix)
                photo.manual, photo.confirmed = True, False
                photo.status, photo.error = "수동 수정", ""
                photo.revision += 1
                self._known[self._key(photo.path)] = self._record(photo, "processed")
                changed.append(photo)
            if changed:
                assert self._cache is not None
                self._cache.put_many(changed)
                self._save_manifest()
                self._publish()

    def select_through(self, path: str) -> list[str]:
        with self._lock:
            photo = self._photos.get(str(Path(path).resolve()))
            return [str(p.path) for p in select_through(self._ordered(), photo)] if photo else []

    def _remap(self, pairs: list[tuple[Path, Path]]) -> None:
        # Remove all sources first: one operation may swap two existing names.
        for before, _ in pairs:
            old = self._key(Path(before))
            self._known.pop(old, None)
            self._stable.pop(old, None)
        for before, after in pairs:
            new = self._key(Path(after))
            try:
                fingerprint = self._fingerprint(Path(after))
            except OSError:
                continue
            self._known[new] = {"size": fingerprint[0], "mtime_ns": fingerprint[1],
                                "state": "processed", "present": True}
            self._stable[new] = (fingerprint, self._stable_scans)
        self._photos = {str(p.path): p for p in self._photos.values() if p.path.exists()}
        self._save_manifest()

    def _rename_action(self, paths: list[str] | None) -> None:
        with self._lock:
            if self._busy or self._closed:
                self._emit("error", "분석이 끝난 후 파일명을 적용하거나 되돌리세요.")
                return
            if self._root is None or self._cache is None:
                self._emit("error", "먼저 사진 폴더를 선택하세요.")
                return
            self._busy = True
            self._emit("busy", True)
            try:
                if paths is None:
                    pairs = undo_last(self._root, self._cache)
                    # The journal/cache restores pre-rename metadata, not the current snapshots.
                    # Remove every current name first so swapped filenames cannot erase restored rows.
                    for before, _ in pairs:
                        self._photos.pop(str(before), None)
                    for _, after in pairs:
                        photo = self._cache.get(after) or Photo.from_path(after)
                        self._photos[str(photo.path)] = photo
                    self._remap(pairs)
                    self._emit("status", f"사진 {len(pairs):,}장의 마지막 변경을 되돌렸습니다.")
                else:
                    photos = [self._photos[p] for p in dict.fromkeys(str(Path(p).resolve()) for p in paths)
                              if p in self._photos and not self._photos[p].confirmed
                              and self._photos[p].proposed_name]
                    pairs = apply_names(self._root, photos, self._cache)
                    self._remap(pairs)
                    for photo in photos:
                        if photo.confirmed:
                            self._known[self._key(photo.path)] = self._record(photo, "processed")
                    self._save_manifest()
                    self._emit("status", f"사진 {len(photos):,}장의 파일명을 적용했습니다.")
                self._publish()
            except Exception as exc:
                self._emit("error", f"파일명 변경 오류: {exc}")
            finally:
                self._busy = False
                self._emit("busy", False)
        self._wake.set()

    def apply_selected(self, paths: list[str]) -> None:
        self._rename_action(paths)

    def undo(self) -> None:
        self._rename_action(None)

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._auto = False
            self._cancel.set()
            self._save_manifest()
        self._wake.set()
        if threading.current_thread() is not self._worker:
            self._worker.join(timeout=5.0)
            if self._worker.is_alive():
                self._emit("error", "분석 작업의 종료를 기다리고 있습니다. 네트워크 요청은 취소되었습니다.")
