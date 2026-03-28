from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


COOKIE_SYNC_DIR_ENV = "GEMINI_FLOW_COOKIE_SYNC_DIR"
IMAGE_OUTPUT_DIR_ENV = "GEMINI_FLOW_IMAGE_DIR"


def _resolve_env_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def get_cookie_sync_dir() -> Optional[Path]:
    raw = os.environ.get(COOKIE_SYNC_DIR_ENV)
    if not raw or not raw.strip():
        return None
    return _resolve_env_path(raw.strip())


def get_image_output_dir() -> Path:
    raw = os.environ.get(IMAGE_OUTPUT_DIR_ENV)
    base = _resolve_env_path(raw.strip()) if raw and raw.strip() else (Path.cwd() / "output" / "image").resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def sync_cookie_exports(*, cookies_dir: Path, sync_dir: Optional[Path] = None) -> int:
    source_dir = sync_dir or get_cookie_sync_dir()
    if source_dir is None:
        return 0
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"cookie sync dir not found: {source_dir}")

    destination_dir = cookies_dir.expanduser()
    if not destination_dir.is_absolute():
        destination_dir = (Path.cwd() / destination_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for source_path in source_dir.iterdir():
        if not source_path.is_file() or source_path.suffix.lower() != ".json":
            continue

        target_path = destination_dir / source_path.name
        try:
            if source_path.resolve() == target_path.resolve():
                continue
        except FileNotFoundError:
            pass

        if target_path.exists():
            source_stat = source_path.stat()
            target_stat = target_path.stat()
            if (
                target_stat.st_size == source_stat.st_size
                and target_stat.st_mtime_ns >= source_stat.st_mtime_ns
            ):
                continue

        shutil.copy2(source_path, target_path)
        copied += 1

    return copied
