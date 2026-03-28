from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .config import sync_cookie_exports
from .types import Cookies, MissingAuthError


GOOGLE_COOKIE_DOMAIN = ".google.com"
REQUIRED_COOKIE_NAME = "__Secure-1PSID"


def _load_json(path: Path) -> object:
    with path.open("rb") as f:
        return json.load(f)


def _parse_exported_cookie_list(cookie_export: object) -> Dict[str, Cookies]:
    """Parse Chrome/extension exported JSON cookie list.

    Expected shape: a list of objects with at least `domain`, `name`, `value`.
    Returns: { domain -> { name -> value } }
    """
    if not isinstance(cookie_export, list):
        return {}

    by_domain: Dict[str, Cookies] = {}
    for item in cookie_export:
        if not isinstance(item, dict):
            continue
        domain = item.get("domain")
        name = item.get("name")
        value = item.get("value")
        if not domain or not name or value is None:
            continue
        by_domain.setdefault(str(domain), {})[str(name)] = str(value)
    return by_domain


def _load_cookies_from_dir(cookies_dir: Path) -> Dict[str, Cookies]:
    merged: Dict[str, Cookies] = {}
    for entry in cookies_dir.iterdir():
        if not entry.is_file() or entry.suffix.lower() != ".json":
            continue
        try:
            parsed = _parse_exported_cookie_list(_load_json(entry))
        except Exception:
            continue
        for domain, cookies in parsed.items():
            merged.setdefault(domain, {}).update(cookies)
    return merged


def _pick_google_cookies(cookies_by_domain: Dict[str, Cookies]) -> Cookies:
    if GOOGLE_COOKIE_DOMAIN in cookies_by_domain:
        return dict(cookies_by_domain[GOOGLE_COOKIE_DOMAIN])

    combined: Cookies = {}
    for domain, cookies in cookies_by_domain.items():
        if domain.endswith("google.com"):
            combined.update(cookies)
    return combined


def load_google_cookies(cookies_dir: Path) -> Cookies:
    sync_cookie_exports(cookies_dir=cookies_dir)

    if not cookies_dir.exists() or not cookies_dir.is_dir():
        raise FileNotFoundError(f"cookies dir not found: {cookies_dir}")

    cookies_by_domain = _load_cookies_from_dir(cookies_dir)
    cookies = _pick_google_cookies(cookies_by_domain)

    if not cookies.get(REQUIRED_COOKIE_NAME):
        raise MissingAuthError(f"Missing required cookie: {REQUIRED_COOKIE_NAME}")

    return cookies
