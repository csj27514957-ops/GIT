"""Pure naming and selection rules shared by the UI and background worker."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .models import Photo


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
COLORS = ("검정", "흰색", "회색", "빨강", "주황", "노랑", "초록", "파랑",
          "보라", "분홍", "갈색", "베이지", "카키", "다색")
TOPS = ("원피스", "우비", "자켓", "조끼", "후드", "체크셔츠", "셔츠", "긴팔", "반팔")
BOTTOMS = ("치마", "멜빵바지", "레깅스", "청바지", "반바지", "긴바지")
_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_DEVICE = re.compile(r"^(?:CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|COM[1-9¹²³]|LPT[1-9¹²³])(?:\s*\.|$)", re.I)
_MAX_NAME_BYTES = 240  # Leave room for the collision suffix and work on POSIX too.


def natural_key(value: str) -> tuple:
    """Compare digit sequences as numbers without mixing int and str tokens."""
    return tuple((1, int(token)) if token.isdigit() else (0, token.casefold())
                 for token in re.split(r"(\d+)", str(value)))


def list_images(root: Path, recursive: bool = False) -> list[Path]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    entries = root.rglob("*") if recursive else root.iterdir()
    # A symlink can lead outside the chosen folder; only real contained files
    # participate in renaming and monitoring.
    paths = [p for p in entries if not p.is_symlink() and p.is_file()
             and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths, key=lambda p: (natural_key(str(p.relative_to(root))),
                                       str(p.relative_to(root)).casefold()))


def _token(value: Any) -> str:
    return re.sub(r"[\s_-]+", "", str(value or "").strip().casefold())


_COLOR_ALIASES = {
    "검은색": "검정", "검정색": "검정", "검은": "검정", "black": "검정",
    "흰": "흰색", "하양": "흰색", "화이트": "흰색", "white": "흰색",
    "회색빛": "회색", "gray": "회색", "grey": "회색",
    "빨간색": "빨강", "빨강색": "빨강", "red": "빨강",
    "주황색": "주황", "orange": "주황", "노란색": "노랑", "노랑색": "노랑", "yellow": "노랑",
    "초록색": "초록", "녹색": "초록", "green": "초록",
    "파란색": "파랑", "파랑색": "파랑", "청색": "파랑", "남색": "파랑", "진청": "파랑",
    "연청": "파랑", "네이비": "파랑", "데님": "파랑", "blue": "파랑", "navy": "파랑",
    "denim": "파랑", "보라색": "보라", "purple": "보라", "분홍색": "분홍", "핑크": "분홍", "pink": "분홍",
    "brown": "갈색", "beige": "베이지", "khaki": "카키", "여러색": "다색", "멀티": "다색",
    "multicolor": "다색", "multicolour": "다색", "colorful": "다색",
}
_GARMENT_ALIASES = {
    "티셔츠": "반팔", "반팔티": "반팔", "반팔티셔츠": "반팔", "tshirt": "반팔",
    "shortsleeve": "반팔", "shortsleevedshirt": "반팔", "긴팔티": "긴팔", "긴팔티셔츠": "긴팔",
    "longsleeve": "긴팔", "longsleevedshirt": "긴팔", "shirt": "셔츠", "checkedshirt": "체크셔츠",
    "checkshirt": "체크셔츠", "plaidshirt": "체크셔츠", "후드티": "후드", "hoodie": "후드",
    "raincoat": "우비", "재킷": "자켓", "jacket": "자켓", "vest": "조끼", "dress": "원피스",
    "shorts": "반바지", "pants": "긴바지", "trousers": "긴바지", "longpants": "긴바지",
    "jeans": "청바지", "데님바지": "청바지", "skirt": "치마", "overalls": "멜빵바지",
    "dungarees": "멜빵바지", "leggings": "레깅스",
}


def _color(value: Any) -> str:
    token = _token(value)
    return token if token in COLORS else _COLOR_ALIASES.get(token, "")


def _garment(value: Any) -> str:
    token = _token(value)
    return token if token in TOPS + BOTTOMS else _GARMENT_ALIASES.get(token, "")


def _clothing_item(item: Any) -> tuple[str, str] | None:
    if isinstance(item, str):
        token = _token(item)
        for garment in sorted(TOPS + BOTTOMS, key=len, reverse=True):
            if token.endswith(garment):
                color = _color(token[:-len(garment)])
                return (color, garment) if color else None
        return None
    if not isinstance(item, dict):
        return None
    garment = _garment(item.get("type", item.get("kind", item.get("종류"))))
    color = _color(item.get("color", item.get("색상")))
    # Richer model output may supply visible construction details. Outer
    # garments keep their priority; details refine ordinary tops and pants.
    if garment in ("반팔", "긴팔", "셔츠", "체크셔츠", "후드"):
        if item.get("hood") is True or item.get("has_hood") is True:
            garment = "후드"
        elif _token(item.get("pattern")) in ("check", "checked", "plaid", "체크", "체크무늬"):
            garment = "체크셔츠"
        elif item.get("collar") is True or item.get("has_collar") is True:
            garment = "셔츠"
        elif garment in ("반팔", "긴팔"):
            sleeve = _token(item.get("sleeve_length", item.get("sleeve")))
            if sleeve in ("short", "짧음", "반팔"):
                garment = "반팔"
            elif sleeve in ("long", "김", "긴팔"):
                garment = "긴팔"
    if garment in ("긴바지", "청바지"):
        if item.get("knees_visible") is True or _token(item.get("length")) in ("short", "짧음"):
            garment = "반바지"
        elif _token(item.get("material")) in ("denim", "데님", "청"):
            garment = "청바지"
    return (color, garment) if color and garment else None


def _array(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, dict):
        return [value]
    return []


def normalize_analysis(raw: dict) -> dict:
    """Normalize AI output only. Manual edits must bypass this function."""
    gender_token = _token(raw.get("gender", raw.get("성별", "불명")))
    genders = {"남성": "남성", "남자": "남성", "male": "남성", "man": "남성",
               "여성": "여성", "여자": "여성", "female": "여성", "woman": "여성",
               "불명": "불명", "unknown": "불명", "인물없음": "인물없음", "없음": "인물없음",
               "noperson": "인물없음", "none": "인물없음"}
    gender = genders.get(gender_token, "불명")
    entries = _array(raw.get("clothing", raw.get("clothes", raw.get("의류", []))))
    for field in ("shirt", "pants", "top", "bottom"):
        entries.extend(_array(raw.get(field)))
    clothes = [value for value in (_clothing_item(item) for item in entries) if value]
    clothing: list[str] = []
    if gender != "인물없음":
        for group in (TOPS, BOTTOMS):
            found = next(((color, kind) for kind in group for color, garment in clothes if garment == kind), None)
            if found:
                clothing.append("".join(found))
                if found[1] == "원피스":
                    break

    objects: list[str] = []
    locations = {"손": "손", "hand": "손", "held": "손", "손에": "손",
                 "바닥": "바닥", "floor": "바닥", "ground": "바닥"}
    name_aliases = {"콜라병": "페트병", "음료병": "페트병", "생수병": "페트병", "플라스틱병": "페트병",
                    "plasticbottle": "페트병", "petbottle": "페트병", "유리병": "병", "glassbottle": "병",
                    "chair": "의자", "paper": "종이", "newspaper": "신문", "mat": "매트",
                    "shoes": "신발", "shoe": "신발", "phone": "휴대폰", "cellphone": "휴대폰",
                    "핸드폰": "휴대폰"}
    for item in _array(raw.get("objects", raw.get("정물", []))):
        if isinstance(item, dict):
            name = str(item.get("name", item.get("이름", ""))).strip()
            location = locations.get(_token(item.get("location", item.get("위치"))), "")
        elif isinstance(item, str):
            match = re.fullmatch(r"\s*(.+?)\s*\((손|바닥|hand|floor|ground)\)\s*", item, re.I)
            if not match:
                continue
            name, location_text = match.groups()
            location = locations.get(_token(location_text), "")
        else:
            continue
        name = name_aliases.get(_token(name), name)
        if not name or _token(name) in ("없음", "none", "-", "unknown") or not location:
            continue
        if location == "바닥" and name in ("의자", "종이", "신문", "매트", "신발", "휴대폰"):
            continue
        value = f"{name}({location})"
        if value not in objects:
            objects.append(value)
    objects.sort(key=lambda value: (0 if value.endswith("(손)") else 1, natural_key(value)))
    if gender == "인물없음":
        objects = []
    return {"gender": gender, "clothing": clothing, "objects": objects}


def _extension(suffix: str) -> str:
    if not re.fullmatch(r"\.[A-Za-z0-9]{1,12}", str(suffix)):
        raise ValueError("원본 파일 확장자가 올바르지 않습니다.")
    return suffix


def sanitize_filename(name: str, suffix: str) -> str:
    """Validate a manual basename, retaining the original image extension."""
    suffix = _extension(suffix)
    if not isinstance(name, str) or not name or _INVALID.search(name):
        raise ValueError("파일명에 사용할 수 없는 문자나 경로가 있습니다.")
    if name.endswith((".", " ")):
        raise ValueError("파일명은 점이나 공백으로 끝날 수 없습니다.")
    if name.casefold().endswith(suffix.casefold()):
        stem = name[:-len(suffix)]
    else:
        if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
            raise ValueError("사진의 원래 확장자는 변경할 수 없습니다.")
        stem = name
    if not stem.strip(" .") or stem.endswith((".", " ")) or _DEVICE.match(stem):
        raise ValueError("빈 파일명이나 Windows 예약 이름은 사용할 수 없습니다.")
    result = stem + suffix
    if len(result.encode("utf-8")) > _MAX_NAME_BYTES or len(result.encode("utf-16-le")) // 2 > 240:
        raise ValueError("파일명이 너무 깁니다. 이름을 짧게 입력하세요.")
    return result


def _safe_component(value: Any) -> str:
    # Manual vocabulary remains unchanged in the analysis/cache; filenames
    # omit whitespace, as documented for the original application's editor.
    return re.sub(r"\s+", "", _INVALID.sub("-", str(value))).strip(" ._-")


def filename_from_analysis(analysis: dict, suffix: str) -> str:
    """Build a filename; preserve manual vocabulary without AI normalization."""
    suffix = _extension(suffix)
    parts = [str(analysis.get("gender", "불명"))]
    parts.extend(str(value) for value in _array(analysis.get("clothing", [])))
    for value in _array(analysis.get("objects", [])):
        if isinstance(value, dict):
            name, location = value.get("name", ""), value.get("location", "")
            value = f"{name}({location})" if location else str(name)
        match = re.fullmatch(r"\s*(.+?)\s*\((손|바닥)\)\s*", str(value))
        parts.append(f"{match[2]}-{match[1]}" if match else str(value))
    stem = "_".join(part for part in map(_safe_component, parts) if part) or "사진"
    if _DEVICE.match(stem):
        stem = "사진_" + stem
    budget = _MAX_NAME_BYTES - len(suffix.encode("utf-8"))
    stem = stem.encode("utf-8")[:budget].decode("utf-8", errors="ignore").rstrip(" .") or "사진"
    return sanitize_filename(stem + suffix, suffix)


def _eligible(photo: Photo) -> bool:
    return (photo.analysis is not None and not photo.manual and not photo.confirmed
            and photo.status not in ("대기", "오류", "분석 중", "분석중", "분석 대기"))


def _runs(photos: list[Photo], key: Callable[[dict], tuple]) -> list[list[Photo]]:
    runs: list[list[Photo]] = []
    active: list[Photo] = []
    previous = None
    for photo in photos:
        current = (photo.path.parent, key(photo.analysis)) if _eligible(photo) else None
        if current is None or current != previous:
            if len(active) >= 3:
                runs.append(active)
            active = []
        if current is not None:
            active.append(photo)
        previous = current
    if len(active) >= 3:
        runs.append(active)
    return runs


def unify_runs(photos: list[Photo]) -> list[Photo]:
    """Mutate automatic contiguous runs and return only photos that changed.

    Call with the displayed natural order. Manual, confirmed and unfinished
    photos split runs, as do directory boundaries. Status/manual flags survive.
    """
    changed_ids: set[int] = set()
    for run in _runs(photos, lambda a: (a.get("gender"), tuple(a.get("clothing", [])))):
        values = [tuple(sorted(photo.analysis.get("objects", []), key=natural_key)) for photo in run]
        counts = Counter(values)
        winner = max(counts, key=lambda value: (counts[value], len(value), -values.index(value)))
        for photo in run:
            if tuple(sorted(photo.analysis.get("objects", []), key=natural_key)) != winner:
                photo.analysis = {**photo.analysis, "objects": list(winner)}
                changed_ids.add(id(photo))
    for run in _runs(photos, lambda a: (tuple(a.get("clothing", [])), tuple(sorted(a.get("objects", []), key=natural_key)))):
        values = [photo.analysis.get("gender", "불명") for photo in run]
        counts = Counter(values)
        winner = max(counts, key=lambda value: (counts[value], value != "불명", -values.index(value)))
        for photo in run:
            if photo.analysis.get("gender") != winner:
                photo.analysis = {**photo.analysis, "gender": winner}
                changed_ids.add(id(photo))
    changed = [photo for photo in photos if id(photo) in changed_ids]
    for photo in changed:
        photo.proposed_name = filename_from_analysis(photo.analysis, photo.path.suffix)
        photo.revision += 1
    return changed


def select_through(photos: list[Photo], current: Photo) -> list[Photo]:
    """Select unconfirmed rows from the start through the current object."""
    if current.confirmed:
        return []
    index = next((index for index, photo in enumerate(photos) if photo is current), None)
    if index is None:
        return []
    return [photo for photo in photos[:index + 1] if not photo.confirmed]
