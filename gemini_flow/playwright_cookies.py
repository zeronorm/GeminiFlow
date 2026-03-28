from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .types import MissingAuthError


REQUIRED_COOKIE_NAME = "__Secure-1PSID"
GEMINI_URL = "https://gemini.google.com/"


@dataclass(frozen=True)
class PlaywrightCookieRefreshResult:
    ok: bool
    cookies_written: bool
    logged_in: bool
    cookie_count: int


def _has_required_cookie(cookie_export: Sequence[dict]) -> bool:
    for c in cookie_export:
        try:
            if c.get("name") == REQUIRED_COOKIE_NAME and c.get("value"):
                return True
        except Exception:
            continue
    return False


def _looks_like_login_redirect(url: str) -> bool:
    u = (url or "").lower()
    return (
        "accounts.google.com" in u
        or "servicelogin" in u
        or "/signin" in u
        or "oauth" in u
    )


async def export_gemini_cookies_with_playwright_async(
    *,
    cookies_path: Path,
    profile_dir: Path,
    headless: bool,
    browser_channel: Optional[str] = None,
    profile_directory: Optional[str] = None,
    navigate_url: Optional[str] = GEMINI_URL,
    require_login_check: bool = True,
    debug: bool = False,
) -> PlaywrightCookieRefreshResult:
    """Async version of cookie export.

    NOTE: Login may be blocked by Google in automated browsers in some environments.
    """

    try:
        from playwright.async_api import async_playwright  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Playwright is not installed. Install with `pip install playwright` and run "
            "`python -m playwright install chromium`."
        ) from e

    cookies_path.parent.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        launch_kwargs = {
            "user_data_dir": str(profile_dir),
            "headless": headless,
        }
        if browser_channel:
            launch_kwargs["channel"] = browser_channel
        if profile_directory:
            launch_kwargs["args"] = [f"--profile-directory={profile_directory}"]

        ctx = await p.chromium.launch_persistent_context(**launch_kwargs)
        try:
            page = await ctx.new_page()
            if navigate_url:
                await page.goto(navigate_url, wait_until="domcontentloaded")

            cookie_export = await ctx.cookies()
            has_cookie = _has_required_cookie(cookie_export)
            if require_login_check:
                logged_in = has_cookie and not _looks_like_login_redirect(page.url)
            else:
                logged_in = has_cookie

            if debug:
                print(
                    f"[debug] playwright headless={headless} url={page.url} cookies={len(cookie_export)} has_{REQUIRED_COOKIE_NAME}={has_cookie}"
                )

            if not logged_in:
                return PlaywrightCookieRefreshResult(
                    ok=False,
                    cookies_written=False,
                    logged_in=False,
                    cookie_count=len(cookie_export),
                )

            cookies_path.write_text(
                json.dumps(cookie_export, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return PlaywrightCookieRefreshResult(
                ok=True,
                cookies_written=True,
                logged_in=True,
                cookie_count=len(cookie_export),
            )
        finally:
            await ctx.close()


async def ensure_playwright_cookies(
    *,
    cookies_dir: Path,
    cookies_filename: str = "auth_Gemini.json",
    debug: bool = False,
) -> Path:
    cookies_path = cookies_dir / cookies_filename
    profile_dir = cookies_dir / ".pw-profile"

    # Pick preferred channel:
    browser_channel = "chrome"

    async def _try_export(*, headless: bool) -> Optional[PlaywrightCookieRefreshResult]:
        try:
            return await export_gemini_cookies_with_playwright_async(
                cookies_path=cookies_path,
                profile_dir=profile_dir,
                headless=headless,
                browser_channel=browser_channel,
                profile_directory=None,
                debug=debug,
            )
        except Exception as e:
            if debug:
                print(f"[debug] playwright launch failed channel={browser_channel!r}: {e}")
            raise

    # Headless only: in some environments Google blocks sign-in inside automated browsers.
    # So we only export cookies from an already-authenticated persistent profile.
    print("[info] Attempting headless refresh of Gemini cookies...")
    result = await _try_export(headless=True)
    if result.ok:
        print("[info] Gemini cookies refreshed successfully (headless).")
        return cookies_path

    raise MissingAuthError(
        "Missing required cookie: __Secure-1PSID. "
        "Please sign in to https://gemini.google.com/ using a normal Chrome/Edge profile at: "
        f"{profile_dir} (launch browser with --user-data-dir pointing to that folder), then rerun."
    )
