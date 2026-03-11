from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from ..cookies import load_google_cookies
from ..playwright_cookies import ensure_playwright_cookies
from ..types import AsyncTextStream, ChatSession
from .provider import GeminiWebProvider


class GeminiWebClient:
    def __init__(self, *, provider: Optional[GeminiWebProvider] = None):
        self._provider = provider or GeminiWebProvider()

    @classmethod
    def from_cookies_dir(cls, cookies_dir: Path) -> "GeminiWebClient":
        client = cls()
        client._cookies_dir = cookies_dir
        return client

    async def chat(
        self,
        *,
        prompt: str,
        model: str,
        language: str = "zh-TW",
        cookies_dir: Path,
        images: Optional[Sequence[Union[Path, Tuple[bytes, str]]]] = None,
        proxy: Optional[str] = None,
        debug: bool = False,
        auto_refresh_cookies: bool = True,
        save_images: bool = True,
        chat_session: Optional[ChatSession] = None,
    ) -> AsyncTextStream:
        async def _refresh_cookies() -> None:
            await ensure_playwright_cookies(
                cookies_dir=cookies_dir,
                debug=debug,
            )

        async def _load_or_refresh() -> dict:
            try:
                return load_google_cookies(cookies_dir)
            except Exception:
                if not auto_refresh_cookies:
                    raise
                await _refresh_cookies()
                return load_google_cookies(cookies_dir)

        cookies = await _load_or_refresh()
        image_payload: Optional[list[Tuple[bytes, str]]] = None
        if images:
            image_payload = []
            for item in images:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], (bytes, bytearray)):
                    data = bytes(item[0])
                    name = str(item[1]) if item[1] else "image.bin"
                    image_payload.append((data, name))
                    continue

                path = Path(item)
                data = path.read_bytes()
                image_payload.append((data, path.name))

        try:
            return await self._provider.stream_chat(
                model=model,
                prompt=prompt,
                cookies=cookies,
                images=image_payload,
                language=language,
                proxy=proxy,
                debug=debug,
                save_images=save_images,
                chat_session=chat_session,
            )
        except Exception:
            if not auto_refresh_cookies:
                raise

            # Token fetch commonly fails when cookies expire. Refresh and retry once.
            await _refresh_cookies()
            cookies = load_google_cookies(cookies_dir)
            return await self._provider.stream_chat(
                model=model,
                prompt=prompt,
                cookies=cookies,
                images=image_payload,
                language=language,
                proxy=proxy,
                debug=debug,
                save_images=save_images,
                chat_session=chat_session,
            )
