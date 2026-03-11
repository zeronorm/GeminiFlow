from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Allow running without installation:
#   cd projects/gemini_flow
#   python cli.py chat -c ../../user_cookies "hello"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gemini_flow.gemini.client import GeminiWebClient
from gemini_flow.types import ChatSession  # noqa: E402
from gemini_flow.gemini.protocol import MODEL_HEADERS  # noqa: E402


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
                images=images if chat_session.conversation_id is None else None, # only upload images first turn
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
            
            # Reset current prompt for the next turn
            current_prompt = ""

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

    raise SystemExit(2)


if __name__ == "__main__":
    main()
