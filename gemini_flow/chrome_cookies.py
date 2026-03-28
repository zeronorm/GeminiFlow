from __future__ import annotations

import asyncio
import json
import shutil
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp

from .types import MissingAuthError


MACOS_CHROME_USER_DATA_DIR = (
    Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
)
MACOS_CHROME_EXECUTABLE = Path(
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)
DEFAULT_EXPORT_FILENAME = "auth_Gemini.json"
REQUIRED_COOKIE_NAME = "__Secure-1PSID"


@dataclass(frozen=True)
class ChromeProfileInfo:
    user_data_dir: Path
    profile_directory: str
    profile_path: Path


def _load_local_state(user_data_dir: Path) -> dict:
    local_state_path = user_data_dir / "Local State"
    if not local_state_path.exists():
        return {}
    try:
        return json.loads(local_state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def detect_active_chrome_profile(
    *,
    user_data_dir: Optional[Path] = None,
    profile_directory: Optional[str] = None,
) -> ChromeProfileInfo:
    resolved_user_data_dir = (user_data_dir or MACOS_CHROME_USER_DATA_DIR).expanduser()
    if not resolved_user_data_dir.is_absolute():
        resolved_user_data_dir = (Path.cwd() / resolved_user_data_dir).resolve()
    if not resolved_user_data_dir.exists() or not resolved_user_data_dir.is_dir():
        raise FileNotFoundError(f"Chrome user data dir not found: {resolved_user_data_dir}")

    if profile_directory and profile_directory.strip():
        chosen_profile = profile_directory.strip()
    else:
        local_state = _load_local_state(resolved_user_data_dir)
        profile_state = local_state.get("profile", {})
        last_active_profiles = profile_state.get("last_active_profiles")
        if isinstance(last_active_profiles, list):
            chosen_profile = next(
                (str(item).strip() for item in last_active_profiles if str(item).strip()),
                "Default",
            )
        else:
            chosen_profile = "Default"

    profile_path = resolved_user_data_dir / chosen_profile
    if not profile_path.exists() or not profile_path.is_dir():
        raise FileNotFoundError(f"Chrome profile dir not found: {profile_path}")

    return ChromeProfileInfo(
        user_data_dir=resolved_user_data_dir,
        profile_directory=chosen_profile,
        profile_path=profile_path,
    )


def resolve_export_cookies_path(*, output_dir: Path, filename: str = DEFAULT_EXPORT_FILENAME) -> Path:
    resolved_output_dir = output_dir.expanduser()
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = (Path.cwd() / resolved_output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    return resolved_output_dir / filename


def _copy_local_state(source_user_data_dir: Path, target_user_data_dir: Path) -> None:
    local_state_path = source_user_data_dir / "Local State"
    if local_state_path.exists():
        shutil.copy2(local_state_path, target_user_data_dir / "Local State")


def _copy_profile_tree(source_profile_path: Path, target_profile_path: Path) -> None:
    ignore = shutil.ignore_patterns(
        "Cache",
        "Code Cache",
        "GPUCache",
        "GrShaderCache",
        "GraphiteDawnCache",
        "DawnGraphiteCache",
        "DawnWebGPUCache",
        "ShaderCache",
        "Service Worker",
        "Session Storage",
        "shared_proto_db",
        "VideoDecodeStats",
        "blob_storage",
        "Blob Storage",
    )
    shutil.copytree(source_profile_path, target_profile_path, ignore=ignore)


def stage_chrome_profile_copy(profile: ChromeProfileInfo) -> tuple[Path, Path]:
    temp_root = Path(tempfile.mkdtemp(prefix="gemini-flow-chrome-"))
    staged_user_data_dir = temp_root / "user-data"
    staged_user_data_dir.mkdir(parents=True, exist_ok=True)

    _copy_local_state(profile.user_data_dir, staged_user_data_dir)
    staged_profile_path = staged_user_data_dir / profile.profile_directory
    _copy_profile_tree(profile.profile_path, staged_profile_path)
    return staged_user_data_dir, temp_root


def _find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


async def _wait_for_cdp_target(
    *,
    session: aiohttp.ClientSession,
    port: int,
    process: asyncio.subprocess.Process,
    debug: bool,
) -> str:
    endpoint = f"http://127.0.0.1:{port}/json"
    for _ in range(80):
        if process.returncode is not None:
            stderr = ""
            if process.stderr is not None:
                try:
                    stderr = (await process.stderr.read()).decode("utf-8", "ignore").strip()
                except Exception:
                    stderr = ""
            raise RuntimeError(
                "Chrome exited before DevTools became ready."
                + (f" stderr={stderr[:800]}" if stderr else "")
            )

        try:
            async with session.get(endpoint) as resp:
                if resp.status >= 400:
                    await asyncio.sleep(0.25)
                    continue
                pages = await resp.json()
        except Exception:
            await asyncio.sleep(0.25)
            continue

        if isinstance(pages, list) and pages:
            ws_url = pages[0].get("webSocketDebuggerUrl")
            if isinstance(ws_url, str) and ws_url:
                return ws_url

        await asyncio.sleep(0.25)

    if debug:
        print(f"[debug] DevTools endpoint not ready: {endpoint}")
    raise RuntimeError("Timed out waiting for Chrome DevTools endpoint.")


async def _cdp_get_all_cookies(
    *,
    session: aiohttp.ClientSession,
    ws_url: str,
) -> list[dict]:
    async with session.ws_connect(ws_url) as ws:
        await ws.send_json({"id": 1, "method": "Network.getAllCookies"})
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                payload = json.loads(msg.data)
            except Exception:
                continue
            if payload.get("id") != 1:
                continue
            result = payload.get("result", {})
            cookies = result.get("cookies", [])
            if isinstance(cookies, list):
                return [item for item in cookies if isinstance(item, dict)]
            return []
    return []


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return

    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except Exception:
        process.kill()
        await process.wait()


async def export_gemini_cookies_from_chrome_profile_async(
    *,
    output_dir: Path,
    output_filename: str = DEFAULT_EXPORT_FILENAME,
    chrome_user_data_dir: Optional[Path] = None,
    profile_directory: Optional[str] = None,
    debug: bool = False,
) -> Path:
    profile = detect_active_chrome_profile(
        user_data_dir=chrome_user_data_dir,
        profile_directory=profile_directory,
    )
    if not MACOS_CHROME_EXECUTABLE.exists():
        raise FileNotFoundError(f"Chrome executable not found: {MACOS_CHROME_EXECUTABLE}")

    cookies_path = resolve_export_cookies_path(
        output_dir=output_dir,
        filename=output_filename,
    )
    staged_user_data_dir, temp_root = stage_chrome_profile_copy(profile)
    port = _find_free_tcp_port()
    chrome_process: Optional[asyncio.subprocess.Process] = None

    try:
        chrome_process = await asyncio.create_subprocess_exec(
            str(MACOS_CHROME_EXECUTABLE),
            "--headless=new",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-sync",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={staged_user_data_dir}",
            f"--profile-directory={profile.profile_directory}",
            "about:blank",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        async with aiohttp.ClientSession() as session:
            ws_url = await _wait_for_cdp_target(
                session=session,
                port=port,
                process=chrome_process,
                debug=debug,
            )
            cookies = await _cdp_get_all_cookies(session=session, ws_url=ws_url)

        if debug:
            print(
                f"[debug] chrome profile={profile.profile_directory} exported_cookies={len(cookies)} output={cookies_path}"
            )

        if not any(item.get("name") == REQUIRED_COOKIE_NAME and item.get("value") for item in cookies):
            raise MissingAuthError(
                "Missing required cookie: __Secure-1PSID. "
                f"Chrome profile '{profile.profile_directory}' may not be logged in to Gemini."
            )

        cookies_path.write_text(
            json.dumps(cookies, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return cookies_path
    finally:
        if chrome_process is not None:
            await _terminate_process(chrome_process)
        shutil.rmtree(temp_root, ignore_errors=True)
