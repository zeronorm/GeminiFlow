from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
from pathlib import Path
from typing import Any, Optional, Tuple
from uuid import uuid4

import aiohttp
import aiohttp_cors
from aiohttp import web

from .entrypoint import Gemini
from .chrome_cookies import (
    detect_active_chrome_profile,
    export_gemini_cookies_from_chrome_profile_async,
)
from .config import get_cookie_sync_dir
from .cookies import load_google_cookies
from .types import ChatSession


def _json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _json_error(message: str, *, status: int = 400) -> web.Response:
    return web.json_response({"error": message}, status=status, dumps=_json_dumps)


async def _read_json_object(request: web.Request) -> dict[str, Any]:
    raw = await request.read()
    if not raw:
        raise ValueError("empty request body")

    last_error: Optional[Exception] = None
    for encoding in ("utf-8", "utf-8-sig", "cp950", "big5"):
        try:
            text = raw.decode(encoding)
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError("body must be a JSON object")
            return obj
        except Exception as e:
            last_error = e

    content_type = request.headers.get("Content-Type") or ""
    raise ValueError(
        f"invalid JSON body (Content-Type={content_type!r}, bytes={len(raw)}). "
        f"Tip: send UTF-8 JSON and set Content-Type: application/json"
    ) from last_error


async def _read_optional_json_object(request: web.Request) -> dict[str, Any]:
    raw = await request.read()
    if not raw or not raw.strip():
        return {}

    last_error: Optional[Exception] = None
    for encoding in ("utf-8", "utf-8-sig", "cp950", "big5"):
        try:
            text = raw.decode(encoding)
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError("body must be a JSON object")
            return obj
        except Exception as e:
            last_error = e

    content_type = request.headers.get("Content-Type") or ""
    raise ValueError(
        f"invalid JSON body (Content-Type={content_type!r}, bytes={len(raw)}). "
        f"Tip: send UTF-8 JSON and set Content-Type: application/json"
    ) from last_error


def _normalize_base64(data: str) -> str:
    compact = "".join(data.split())
    padding = (-len(compact)) % 4
    if padding:
        compact += "=" * padding
    return compact


def _decode_base64_image(value: str, *, index: int) -> Tuple[bytes, str]:
    if value.startswith("data:image/"):
        header, b64 = value.split(",", 1)
        mime = header.split(";", 1)[0].split(":", 1)[1].lower()
        ext = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/webp": "webp",
        }.get(mime, "png")
        payload = _normalize_base64(b64)
        try:
            data = base64.b64decode(payload, validate=False)
        except binascii.Error as e:
            raise ValueError(f"images[{index}] invalid base64 data URL") from e
        return data, f"upload_{index + 1}.{ext}"

    payload = _normalize_base64(value)
    try:
        data = base64.b64decode(payload, validate=False)
    except binascii.Error as e:
        raise ValueError(f"images[{index}] invalid base64 string") from e
    return data, f"upload_{index + 1}.png"


async def _image_url_to_base64(
    url: str,
    *,
    session: aiohttp.ClientSession,
) -> Optional[str]:
    if url.startswith("data:image/"):
        try:
            _, b64 = url.split(",", 1)
            return _normalize_base64(b64)
        except Exception:
            return None

    if not (url.startswith("http://") or url.startswith("https://")):
        return None

    try:
        async with session.get(url) as resp:
            if resp.status >= 400:
                return None
            data = await resp.read()
    except Exception:
        return None

    return base64.b64encode(data).decode("ascii")


def _load_download_cookies() -> Optional[dict[str, str]]:
    try:
        return load_google_cookies(Path("user_cookies"))
    except Exception:
        return None


def _parse_images(payload: dict[str, Any]) -> Optional[list[Tuple[bytes, str]]]:
    images = payload.get("images")
    if images is None:
        return None
    if not isinstance(images, list) or not all(isinstance(x, str) for x in images):
        raise ValueError("images must be a list of base64 strings")

    return [_decode_base64_image(value, index=i) for i, value in enumerate(images)]


def _payload_model(payload: dict[str, Any]) -> str:
    model = payload.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return "gemini-3-pro"


def _payload_has_images(payload: dict[str, Any]) -> bool:
    images = payload.get("images")
    return isinstance(images, list) and len(images) > 0


def _optional_payload_path(payload: dict[str, Any], key: str) -> Optional[Path]:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    stripped = value.strip()
    if not stripped:
        return None
    return Path(stripped)


def _optional_payload_text(payload: dict[str, Any], key: str) -> Optional[str]:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    stripped = value.strip()
    return stripped or None


def _payload_debug(payload: dict[str, Any]) -> bool:
    value = payload.get("debug", False)
    if not isinstance(value, bool):
        raise ValueError("debug must be a boolean")
    return value


async def _run_gemini_stream(
    *, payload: dict[str, Any], chat_session: Optional[ChatSession] = None
):
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt is required")

    model = payload.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError("model must be a string")
    if isinstance(model, str):
        model = model.strip() or None

    language = payload.get("language")
    if language is not None and not isinstance(language, str):
        raise ValueError("language must be a string")
    if isinstance(language, str):
        language = language.strip() or None

    auto_refresh_cookies = payload.get("auto_refresh_cookies", True)
    if not isinstance(auto_refresh_cookies, bool):
        raise ValueError("auto_refresh_cookies must be a boolean")

    images = _parse_images(payload)

    ai = Gemini(
        model=model or "gemini-3-pro",
        language=language or "zh-TW",
        auto_refresh_cookies=auto_refresh_cookies,
    )

    return await ai.astream_chat(
        prompt,
        model=model,
        images=images,
        language=language,
        save_images=False,
        chat_session=chat_session,
    )


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True}, dumps=_json_dumps)


async def sync_cookies(request: web.Request) -> web.Response:
    try:
        payload = await _read_optional_json_object(request)
        output_dir = _optional_payload_path(payload, "output_dir") or get_cookie_sync_dir()
        if output_dir is None:
            raise ValueError(
                "output_dir is required. Provide output_dir in request body or set GEMINI_FLOW_COOKIE_SYNC_DIR."
            )

        output_filename = _optional_payload_text(payload, "output_filename") or "auth_Gemini.json"
        chrome_user_data_dir = _optional_payload_path(payload, "chrome_user_data_dir")
        profile_directory = _optional_payload_text(payload, "profile_directory")
        debug = _payload_debug(payload)

        profile = detect_active_chrome_profile(
            user_data_dir=chrome_user_data_dir,
            profile_directory=profile_directory,
        )
        cookies_path = await export_gemini_cookies_from_chrome_profile_async(
            output_dir=output_dir,
            output_filename=output_filename,
            chrome_user_data_dir=chrome_user_data_dir,
            profile_directory=profile.profile_directory,
            debug=debug,
        )
    except Exception as e:
        return _json_error(str(e), status=400)

    return web.json_response(
        {
            "ok": True,
            "output_path": str(cookies_path),
            "output_dir": str(cookies_path.parent),
            "output_filename": cookies_path.name,
            "chrome_user_data_dir": str(profile.user_data_dir),
            "profile_directory": profile.profile_directory,
        },
        dumps=_json_dumps,
    )


async def chat(request: web.Request) -> web.Response:
    try:
        payload = await _read_json_object(request)
    except Exception as e:
        return _json_error(str(e))

    request_id = uuid4().hex[:8]
    try:
        print(
            f"[server] id={request_id} /chat recv model={_payload_model(payload)} has_images={_payload_has_images(payload)}"
        )
    except Exception:
        print(f"[server] id={request_id} /chat recv <unprintable>")

    try:
        chat_session = ChatSession(
            conversation_id=payload.get("conversation_id"),
            response_id=payload.get("response_id"),
            choice_id=payload.get("choice_id"),
        )
        stream = await _run_gemini_stream(payload=payload, chat_session=chat_session)
    except Exception as e:
        return _json_error(str(e), status=400)

    text_parts: list[str] = []
    images_saved: list[str] = []

    cookies = _load_download_cookies()
    async with aiohttp.ClientSession(cookies=cookies) as http_session:
        try:
            async for chunk in stream:
                if isinstance(chunk, str) and chunk.startswith("[image saved] "):
                    path = chunk[len("[image saved] ") :].strip()
                    if path:
                        images_saved.append(path)
                    continue
                if isinstance(chunk, str) and chunk.startswith("[image url] "):
                    url = chunk[len("[image url] ") :].strip()
                    if url:
                        b64 = await _image_url_to_base64(url, session=http_session)
                        if b64:
                            images_saved.append(b64)
                    continue
                if isinstance(chunk, str) and chunk.startswith("[image] "):
                    candidate = chunk[len("[image] ") :].strip()
                    if candidate:
                        b64 = await _image_url_to_base64(candidate, session=http_session)
                        if b64:
                            images_saved.append(b64)
                    continue
                text_parts.append(str(chunk))
        except Exception as e:
            return _json_error(str(e), status=500)

    response_payload = {
        "text": "".join(text_parts),
        "images": images_saved,
        "conversation_id": chat_session.conversation_id,
        "response_id": chat_session.response_id,
        "choice_id": chat_session.choice_id,
    }
    try:
        print(
            f"[server] id={request_id} /chat resp has_text={bool(response_payload['text'])} "
            f"has_images={bool(response_payload['images'])}"
        )
    except Exception:
        print(f"[server] id={request_id} /chat resp <unprintable>")

    return web.json_response(response_payload, dumps=_json_dumps)


def _sse_format(*, event: str, data: object) -> bytes:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


async def stream(request: web.Request) -> web.StreamResponse:
    try:
        payload = await _read_json_object(request)
    except Exception as e:
        return web.Response(status=400, text=str(e))

    request_id = uuid4().hex[:8]
    try:
        print(
            f"[server] id={request_id} /stream recv model={_payload_model(payload)} has_images={_payload_has_images(payload)}"
        )
    except Exception:
        print(f"[server] id={request_id} /stream recv <unprintable>")

    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await resp.prepare(request)

    try:
        chat_session = ChatSession(
            conversation_id=payload.get("conversation_id"),
            response_id=payload.get("response_id"),
            choice_id=payload.get("choice_id"),
        )
        gemini_stream = await _run_gemini_stream(payload=payload, chat_session=chat_session)
        cookies = _load_download_cookies()
        has_text = False
        has_images = False
        async with aiohttp.ClientSession(cookies=cookies) as http_session:
            async for chunk in gemini_stream:
                if isinstance(chunk, str) and chunk.startswith("[image saved] "):
                    path = chunk[len("[image saved] ") :].strip()
                    has_images = True
                    await resp.write(_sse_format(event="image", data={"path": path}))
                elif isinstance(chunk, str) and chunk.startswith("[image url] "):
                    url = chunk[len("[image url] ") :].strip()
                    b64 = await _image_url_to_base64(url, session=http_session) if url else None
                    if b64:
                        has_images = True
                        await resp.write(_sse_format(event="image", data={"base64": b64}))
                    else:
                        await resp.write(_sse_format(event="image", data={"base64": ""}))
                elif isinstance(chunk, str) and chunk.startswith("[image] "):
                    candidate = chunk[len("[image] ") :].strip()
                    b64 = await _image_url_to_base64(candidate, session=http_session) if candidate else None
                    if b64:
                        has_images = True
                        await resp.write(_sse_format(event="image", data={"base64": b64}))
                    else:
                        await resp.write(_sse_format(event="image", data={"base64": ""}))
                else:
                    if chunk:
                        has_text = True
                    await resp.write(_sse_format(event="text", data={"chunk": str(chunk)}))
        try:
            print(
                f"[server] id={request_id} /stream resp has_text={has_text} has_images={has_images}"
            )
        except Exception:
            pass
        await resp.write(
            _sse_format(
                event="done",
                data={
                    "conversation_id": chat_session.conversation_id,
                    "response_id": chat_session.response_id,
                    "choice_id": chat_session.choice_id,
                },
            )
        )
    except ConnectionResetError:
        return resp
    except Exception as e:
        try:
            await resp.write(_sse_format(event="error", data={"error": str(e)}))
        except Exception:
            pass

    return resp


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/sync-cookies", sync_cookies)
    app.router.add_post("/chat", chat)
    app.router.add_post("/stream", stream)
    cors = aiohttp_cors.setup(
        app,
        defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        },
    )
    for route in list(app.router.routes()):
        cors.add(route)
    return app


async def serve(*, host: str = "127.0.0.1", port: int = 8000) -> None:
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()

    print(f"[server] listening on http://{host}:{port}")
    print("[server] endpoints: GET /health, POST /sync-cookies, POST /chat, POST /stream (SSE)")

    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()


def main() -> None:
    p = argparse.ArgumentParser(description="gemini_flow HTTP server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    asyncio.run(serve(host=args.host, port=args.port))
