from __future__ import annotations

import base64
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import aiohttp

from ..providers.base import ChatProvider
from ..types import AsyncTextStream, Cookies, MissingAuthError, RequestError, TokenFetchError, ChatSession
from .protocol import (
    DEFAULT_HEADERS,
    GEMINI_BASE_URL,
    REQUEST_URL,
    GeminiRequest,
    REQUIRED_COOKIE_NAME,
    extract_image_candidates_from_raw_line,
    extract_text_delta_from_raw_line,
    extract_tokens,
)
from .upload import upload_images


class GeminiWebProvider(ChatProvider):
    async def fetch_tokens(
        self,
        *,
        session: aiohttp.ClientSession,
        cookies: Cookies,
        proxy: Optional[str] = None,
        debug: bool = False,
    ):
        try:
            async with session.get(GEMINI_BASE_URL, cookies=cookies, proxy=proxy) as resp:
                if resp.status >= 400:
                    raise TokenFetchError(f"Token page fetch failed: HTTP {resp.status}")
                html = await resp.text()
        except aiohttp.ClientError as e:
            raise TokenFetchError(f"Token page fetch failed: {e}") from e

        tokens = extract_tokens(html)
        if not tokens:
            if debug:
                preview = html[:800].replace("\r", "")
                print(f"[debug] Token page preview (first 800 chars):\n{preview}\n")
            raise TokenFetchError("SNlM0e token not found; cookies likely invalid/expired")
        return tokens

    async def stream_chat(
        self,
        *,
        model: str,
        prompt: str,
        cookies: Cookies,
        images: Optional[Sequence[Tuple[bytes, str]]] = None,
        language: str = "zh-TW",
        proxy: Optional[str] = None,
        debug: bool = False,
        save_images: bool = True,
        chat_session: Optional[ChatSession] = None,
    ) -> AsyncTextStream:
        if REQUIRED_COOKIE_NAME not in cookies:
            raise MissingAuthError(f"Missing required cookie: {REQUIRED_COOKIE_NAME}")

        async with aiohttp.ClientSession(headers=DEFAULT_HEADERS) as token_session:
            tokens = await self.fetch_tokens(session=token_session, cookies=cookies, proxy=proxy, debug=debug)

        uploads = None
        if images:
            try:
                uploaded = await upload_images(images, proxy=proxy)
                uploads = [(u.upload_ref, u.name) for u in uploaded]
            except Exception as e:
                raise RequestError(f"Image upload failed: {e}") from e

        req = GeminiRequest(
            prompt=prompt,
            language=language,
            tokens=tokens,
            model=model,
            uploads=uploads,
            chat_session=chat_session,
        )

        normalized_model = model.strip().lower()
        is_image_model = (
            normalized_model.endswith("-image")
            or normalized_model.endswith("-image-preview")
            or "-image-" in normalized_model
        )

        _CONTROL_RE = re.compile(r"[\x00-\x1F\x7F\u200B\u200C\u200D\uFEFF]")

        def _normalize_candidate(value: str) -> str:
            value = value.strip()
            value = _CONTROL_RE.sub("", value)
            return value

        def _is_placeholder_or_input_image(url: str) -> bool:
            # Placeholder token, not a real downloadable image.
            if url.startswith("http://googleusercontent.com/image_generation_content/"):
                return True
            # Echoed input/uploaded image reference (not the generated output).
            if "lh3.googleusercontent.com/gg/" in url and "lh3.googleusercontent.com/gg-dl/" not in url:
                return True
            return False

        def _is_output_image_url(url: str) -> bool:
            if url.startswith("data:image/"):
                return True
            if "lh3.googleusercontent.com/gg-dl/" in url:
                return True
            return False

        def _is_noise_text_in_image_mode(text: str) -> bool:
            normalized = _normalize_candidate(text)
            if not normalized:
                return True
            if _is_placeholder_or_input_image(normalized):
                return True
            # Some image responses include media URLs in the text delta stream.
            if normalized.startswith("http://") or normalized.startswith("https://"):
                if any(
                    host in normalized
                    for host in [
                        "googleusercontent.com/image_generation_content/",
                        "lh3.googleusercontent.com/gg-dl/",
                        "lh3.googleusercontent.com/gg/",
                    ]
                ):
                    return True
            return False

        def _get_image_output_dir() -> Path:
            configured = os.environ.get("GEMINI_FLOW_IMAGE_DIR")
            base = Path(configured) if configured else Path("output") / "image"
            out = base.expanduser()
            if not out.is_absolute():
                out = (Path.cwd() / out).resolve()
            out.mkdir(parents=True, exist_ok=True)
            return out

        async def _save_image_candidate(
            *,
            client: aiohttp.ClientSession,
            candidate: str,
            out_dir: Path,
            suffix: str,
        ) -> Optional[Path]:
            if candidate.startswith("data:image/"):
                try:
                    header, b64 = candidate.split(",", 1)
                    mime = header.split(";", 1)[0].split(":", 1)[1]
                    ext = {
                        "image/png": "png",
                        "image/jpeg": "jpg",
                        "image/webp": "webp",
                        "image/svg+xml": "svg",
                    }.get(mime, "png")
                    data = base64.b64decode(b64)
                except Exception:
                    return None

                out_path = out_dir / f"{suffix}.{ext}"
                out_path.write_bytes(data)
                return out_path

            if not (candidate.startswith("https://") or candidate.startswith("http://")):
                return None

            try:
                async with client.get(candidate, proxy=proxy) as resp:
                    if resp.status >= 400:
                        return None
                    data = await resp.read()
                    content_type = (resp.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            except Exception:
                return None

            ext = {
                "image/png": "png",
                "image/jpeg": "jpg",
                "image/webp": "webp",
                "image/svg+xml": "svg",
            }.get(content_type, "png")
            out_path = out_dir / f"{suffix}.{ext}"
            out_path.write_bytes(data)
            return out_path

        async def gen():
            emitted_any = False
            preview = ""
            buffer = ""
            last_content = ""
            final_image_candidate: Optional[str] = None
            fallback_image_candidate: Optional[str] = None
            out_dir = _get_image_output_dir() if is_image_model else Path.cwd()
            out_prefix = f"gemini_{model}_{int(time.time())}"
            out_index = 0

            async with aiohttp.ClientSession(headers=DEFAULT_HEADERS, cookies=cookies) as client:
                try:
                    async with client.post(
                        REQUEST_URL,
                        params=req.params(),
                        data=req.data(),
                        headers=req.headers(),
                        proxy=proxy,
                    ) as resp:
                        if resp.status >= 400:
                            body = await resp.text()
                            raise RequestError(
                                f"StreamGenerate failed: HTTP {resp.status} body={body[:300]}"
                            )

                        async for chunk in resp.content.iter_any():
                            try:
                                part = chunk.decode("utf-8", errors="ignore")
                            except Exception:
                                continue

                            if debug and len(preview) < 800:
                                preview += part[: (800 - len(preview))].replace("\r", "")

                            buffer += part
                            while "\n" in buffer:
                                raw_line, buffer = buffer.split("\n", 1)
                                raw_line = raw_line.rstrip("\r")

                                if is_image_model:
                                    for candidate in extract_image_candidates_from_raw_line(raw_line):
                                        normalized = _normalize_candidate(candidate)
                                        if not normalized:
                                            continue
                                        if _is_placeholder_or_input_image(normalized):
                                            if fallback_image_candidate is None:
                                                fallback_image_candidate = normalized
                                            continue
                                        if _is_output_image_url(normalized):
                                            # Keep only the latest output candidate; save once at the end.
                                            final_image_candidate = normalized

                                delta, last_content, new_chat_session = extract_text_delta_from_raw_line(
                                    raw_line, last_content
                                )
                                if new_chat_session and chat_session is not None:
                                    chat_session.conversation_id = new_chat_session.conversation_id
                                    chat_session.response_id = new_chat_session.response_id
                                    chat_session.choice_id = new_chat_session.choice_id
                                if delta:
                                    if not is_image_model or not _is_noise_text_in_image_mode(delta):
                                        emitted_any = True
                                        yield delta

                except aiohttp.ClientError as e:
                    raise RequestError(f"StreamGenerate request failed: {e}") from e

            if buffer.strip():
                raw_line = buffer.rstrip("\r")

                if is_image_model:
                    for candidate in extract_image_candidates_from_raw_line(raw_line):
                        normalized = _normalize_candidate(candidate)
                        if not normalized:
                            continue
                        if _is_placeholder_or_input_image(normalized):
                            if fallback_image_candidate is None:
                                fallback_image_candidate = normalized
                            continue
                        if _is_output_image_url(normalized):
                            final_image_candidate = normalized

                delta, last_content, new_chat_session = extract_text_delta_from_raw_line(raw_line, last_content)
                if new_chat_session and chat_session is not None:
                    chat_session.conversation_id = new_chat_session.conversation_id
                    chat_session.response_id = new_chat_session.response_id
                    chat_session.choice_id = new_chat_session.choice_id
                if delta:
                    if not is_image_model or not _is_noise_text_in_image_mode(delta):
                        emitted_any = True
                        yield delta

            if is_image_model and final_image_candidate:
                if save_images:
                    out_index += 1
                    # NOTE: At this point the streaming client session has been closed.
                    # Use a fresh session for downloading the final image.
                    async with aiohttp.ClientSession(headers=DEFAULT_HEADERS, cookies=cookies) as download_client:
                        saved = await _save_image_candidate(
                            client=download_client,
                            candidate=final_image_candidate,
                            out_dir=out_dir,
                            suffix=f"{out_prefix}_{out_index}",
                        )
                    emitted_any = True
                    if saved:
                        yield f"[image saved] {saved}\n"
                        yield f"[image url] {final_image_candidate}\n"
                    else:
                        yield f"[image] {final_image_candidate}\n"
                else:
                    emitted_any = True
                    yield f"[image url] {final_image_candidate}\n"
            elif is_image_model and fallback_image_candidate:
                emitted_any = True
                yield f"[image] {fallback_image_candidate}\n"

            if not emitted_any:
                if debug and preview:
                    print(
                        f"[debug] StreamGenerate response preview (first 800 chars):\n{preview}\n",
                        file=sys.stderr,
                    )
                raise RequestError(
                    "No text could be parsed from StreamGenerate response. "
                    "Try --debug and share the preview; response format may have changed."
                )

        return gen()
