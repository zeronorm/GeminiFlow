from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Optional

from .chrome_cookies import export_gemini_cookies_from_chrome_profile_async
from .config import get_cookie_sync_dir
from .gemini.client import GeminiWebClient
from .gemini.protocol import MODEL_HEADERS
from .types import ChatSession


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gemini_flow", description="Gemini web(cookie) client")
    sub = p.add_subparsers(dest="cmd", required=True)

    chat = sub.add_parser("chat", help="Send a prompt and stream text output")
    chat.add_argument("prompt", nargs="?", default="", help="User prompt")
    chat.add_argument("-m", "--model", default="gemini-3-pro", choices=sorted(MODEL_HEADERS.keys()))
    chat.add_argument("-c", "--cookies-dir", type=Path, required=True)
    chat.add_argument(
        "--image",
        action="append",
        type=Path,
        default=None,
        help="Attach a local image (repeatable). Example: --image ./photo.png",
    )
    chat.add_argument("--lang", default="zh-TW")
    chat.add_argument("--proxy", default=None)
    chat.add_argument("--debug", action="store_true", help="Print debug diagnostics")

    export = sub.add_parser("export-cookies", help="Export Gemini cookies from the active Chrome profile")
    export.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write exported cookie JSON. Defaults to GEMINI_FLOW_COOKIE_SYNC_DIR.",
    )
    export.add_argument(
        "--output-filename",
        default="auth_Gemini.json",
        help="Exported cookie filename",
    )
    export.add_argument(
        "--chrome-user-data-dir",
        type=Path,
        default=None,
        help="Chrome user data dir. Defaults to macOS Chrome user data dir.",
    )
    export.add_argument(
        "--profile-directory",
        default=None,
        help="Chrome profile directory name such as Default or 'Profile 1'. Defaults to Chrome last_active_profiles.",
    )
    export.add_argument("--debug", action="store_true", help="Print debug diagnostics")

    return p


async def _run_chat(
    *,
    prompt: str,
    model: str,
    cookies_dir: Path,
    images: Optional[list[Path]],
    lang: str,
    proxy: Optional[str],
    debug: bool,
) -> int:
    client = GeminiWebClient()
    chat_session = ChatSession()

    current_prompt = prompt.strip()
    is_interactive = not current_prompt

    if not is_interactive:
        print(f"You: {current_prompt}")
    else:
        print("Starting interactive session. Type 'exit' or 'quit' to close.")

    try:
        while True:
            if not current_prompt:
                try:
                    current_prompt = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if current_prompt.lower() in ("exit", "quit"):
                    break
                if not current_prompt:
                    continue

            stream = await client.chat(
                prompt=current_prompt,
                model=model,
                language=lang,
                cookies_dir=cookies_dir,
                images=images if chat_session.conversation_id is None else None,
                proxy=proxy,
                debug=debug,
                chat_session=chat_session,
            )
            had_output = False
            async for chunk in stream:
                had_output = True
                print(chunk, end="", flush=True)
            print()
            if debug and not had_output:
                print("[debug] No text chunks were parsed from the response.")

            current_prompt = ""

        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


async def _run_export_cookies(
    *,
    output_dir: Optional[Path],
    output_filename: str,
    chrome_user_data_dir: Optional[Path],
    profile_directory: Optional[str],
    debug: bool,
) -> int:
    try:
        resolved_output_dir = output_dir or get_cookie_sync_dir()
        if resolved_output_dir is None:
            print("ERROR: output dir is required. Pass --output-dir or set GEMINI_FLOW_COOKIE_SYNC_DIR.")
            return 1

        cookies_path = await export_gemini_cookies_from_chrome_profile_async(
            output_dir=resolved_output_dir,
            output_filename=output_filename,
            chrome_user_data_dir=chrome_user_data_dir,
            profile_directory=profile_directory,
            debug=debug,
        )
        print(f"Exported cookies to: {cookies_path}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


def main() -> None:
    args = _build_parser().parse_args()
    if args.cmd == "chat":
        images = None
        if args.image:
            images = [Path(p) for p in args.image]
        raise SystemExit(
            asyncio.run(
                _run_chat(
                    prompt=args.prompt,
                    model=args.model,
                    cookies_dir=args.cookies_dir,
                    images=images,
                    lang=args.lang,
                    proxy=args.proxy,
                    debug=args.debug,
                )
            )
        )
    if args.cmd == "export-cookies":
        raise SystemExit(
            asyncio.run(
                _run_export_cookies(
                    output_dir=args.output_dir,
                    output_filename=args.output_filename,
                    chrome_user_data_dir=args.chrome_user_data_dir,
                    profile_directory=args.profile_directory,
                    debug=args.debug,
                )
            )
        )

    raise SystemExit(2)
