from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Photo:
    path: Path
    size: int
    mtime_ns: int
    analysis: dict[str, Any] | None = None
    proposed_name: str = ""
    status: str = "대기"
    manual: bool = False
    confirmed: bool = False
    model: str = ""
    error: str = ""
    revision: int = 0

    @classmethod
    def from_path(cls, path: Path) -> "Photo":
        stat = path.stat()
        return cls(path.resolve(), stat.st_size, stat.st_mtime_ns)
