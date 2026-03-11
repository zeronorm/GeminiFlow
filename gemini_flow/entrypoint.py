from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

from .gemini.client import GeminiWebClient
from .types import ChatSession


PathLike = Union[str, Path]
ImageInput = Union[PathLike, Tuple[bytes, str]]


class Gemini:
    def __init__(
        self,
        *,
        cookies_dir: PathLike = "user_cookies",
        model: str = "gemini-3-pro",
        language: str = "zh-TW",
        proxy: Optional[str] = None,
        debug: bool = False,
        auto_refresh_cookies: bool = True,
        client: Optional[GeminiWebClient] = None,
    ):
        self.cookies_dir = Path(cookies_dir)
        self.model = model
        self.language = language
        self.proxy = proxy
        self.debug = debug
        self.auto_refresh_cookies = auto_refresh_cookies
        self._client = client or GeminiWebClient()

    async def astream_chat(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        language: Optional[str] = None,
        proxy: Optional[str] = None,
        debug: Optional[bool] = None,
        save_images: Optional[bool] = None,
        chat_session: Optional[ChatSession] = None,
    ):
        image_inputs: Optional[list[ImageInput]] = None
        if images:
            image_inputs = []
            for item in images:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], (bytes, bytearray)):
                    image_inputs.append((bytes(item[0]), str(item[1])))
                else:
                    image_inputs.append(Path(item))
        return await self._client.chat(
            prompt=prompt,
            model=model or self.model,
            language=language or self.language,
            cookies_dir=self.cookies_dir,
            images=image_inputs,
            proxy=proxy if proxy is not None else self.proxy,
            debug=debug if debug is not None else self.debug,
            auto_refresh_cookies=self.auto_refresh_cookies,
            save_images=True if save_images is None else save_images,
            chat_session=chat_session,
        )

    async def achat(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        language: Optional[str] = None,
        proxy: Optional[str] = None,
        debug: Optional[bool] = None,
        save_images: Optional[bool] = None,
        chat_session: Optional[ChatSession] = None,
    ) -> str:
        stream = await self.astream_chat(
            prompt,
            model=model,
            images=images,
            language=language,
            proxy=proxy,
            debug=debug,
            save_images=save_images,
            chat_session=chat_session,
        )
        parts: list[str] = []
        async for chunk in stream:
            parts.append(chunk)
        return "".join(parts)

    def chat(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        images: Optional[Sequence[ImageInput]] = None,
        language: Optional[str] = None,
        proxy: Optional[str] = None,
        debug: Optional[bool] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        save_images: Optional[bool] = None,
        chat_session: Optional[ChatSession] = None,
    ) -> str:
        async def _run() -> str:
            stream = await self.astream_chat(
                prompt,
                model=model,
                images=images,
                language=language,
                proxy=proxy,
                debug=debug,
                save_images=save_images,
                chat_session=chat_session,
            )
            parts: list[str] = []
            async for chunk in stream:
                if on_chunk is not None:
                    on_chunk(chunk)
                parts.append(chunk)
            return "".join(parts)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())
        raise RuntimeError(
            "Gemini.chat() cannot be called from within an active event loop. "
            "Use `await Gemini.achat(...)` or `await Gemini.astream_chat(...)` instead."
        )
