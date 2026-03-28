from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Union

from aiohttp import web

from .chrome_cookies import export_gemini_cookies_from_chrome_profile_async
from .config import get_cookie_sync_dir
from .server_app import create_app, serve


PathLike = Union[str, Path]


async def aexport_cookies(
    *,
    output_dir: Optional[PathLike] = None,
    output_filename: str = "auth_Gemini.json",
    chrome_user_data_dir: Optional[PathLike] = None,
    profile_directory: Optional[str] = None,
    debug: bool = False,
) -> Path:
    resolved_output_dir = Path(output_dir) if output_dir is not None else get_cookie_sync_dir()
    if resolved_output_dir is None:
        raise ValueError(
            "output_dir is required. Pass output_dir or set GEMINI_FLOW_COOKIE_SYNC_DIR."
        )

    resolved_chrome_user_data_dir = (
        Path(chrome_user_data_dir) if chrome_user_data_dir is not None else None
    )
    return await export_gemini_cookies_from_chrome_profile_async(
        output_dir=resolved_output_dir,
        output_filename=output_filename,
        chrome_user_data_dir=resolved_chrome_user_data_dir,
        profile_directory=profile_directory,
        debug=debug,
    )


def export_cookies(
    *,
    output_dir: Optional[PathLike] = None,
    output_filename: str = "auth_Gemini.json",
    chrome_user_data_dir: Optional[PathLike] = None,
    profile_directory: Optional[str] = None,
    debug: bool = False,
) -> Path:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            aexport_cookies(
                output_dir=output_dir,
                output_filename=output_filename,
                chrome_user_data_dir=chrome_user_data_dir,
                profile_directory=profile_directory,
                debug=debug,
            )
        )

    raise RuntimeError(
        "gemini_flow.export_cookies() cannot be called from within an active event loop. "
        "Use `await gemini_flow.aexport_cookies(...)` instead."
    )


def create_server_app() -> web.Application:
    return create_app()
