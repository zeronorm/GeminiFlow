from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

from ..types import GeminiTokens, ChatSession


GEMINI_BASE_URL = "https://gemini.google.com"
REQUEST_URL = (
    "https://gemini.google.com/_/BardChatUi/data/assistant.lamda."
    "BardFrontendService/StreamGenerate"
)
REQUEST_BL_PARAM = "boq_assistant-bard-web-server_20240519.16_p0"

DEFAULT_HEADERS = {
    "authority": "gemini.google.com",
    "origin": "https://gemini.google.com",
    "referer": "https://gemini.google.com/",
    "x-same-domain": "1",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

REQUIRED_COOKIE_NAME = "__Secure-1PSID"

MODEL_HEADERS: Dict[str, Dict[str, str]] = {
    "gemini-3-pro-thinking": {
        "x-goog-ext-525001261-jspb": '[1,null,null,null,"e051ce1aa80aa576",null,null,0,[4],null,null,2]'
    },
    "gemini-3-pro-new": {
        "x-goog-ext-525001261-jspb": '[1,null,null,null,"e6fa609c3fa255c0",null,null,0,[4],null,null,2]'
    },
    "gemini-3-pro": {
        "x-goog-ext-525001261-jspb": '[1,null,null,null,"9d8ca3786ebdfbea",null,null,0,[4]]'
    },
    "gemini-3-pro-image-preview": {
        "x-goog-ext-525001261-jspb": '[1,null,null,null,"56fdd199312815e2",null,null,0,[4],null,null,2]'
    },
    "gemini-3-flash": {
        "x-goog-ext-525001261-jspb": '[1,null,null,null,"56fdd199312815e2",null,null,0,[4],null,null,2]'
    },
    "gemini-2.5-pro": {
        "x-goog-ext-525001261-jspb": '[1,null,null,null,"61530e79959ab139",null,null,null,[4]]'
    },
    "gemini-2.5-flash": {
        "x-goog-ext-525001261-jspb": '[1,null,null,null,"9ec249fc9ad08861",null,null,null,[4]]'
    },
}


@dataclass(frozen=True)
class GeminiRequest:
    prompt: str
    language: str
    tokens: GeminiTokens
    model: str
    uploads: Optional[Sequence[Tuple[str, str]]] = None
    chat_session: Optional[ChatSession] = None

    def params(self) -> Dict[str, str]:
        return {
            "bl": REQUEST_BL_PARAM,
            "hl": self.language,
            "_reqid": str(random.randint(1111, 9999)),
            "rt": "c",
            "f.sid": "" if self.tokens.sid is None else self.tokens.sid,
        }

    def data(self) -> Dict[str, str]:
        inner = build_request(
            self.prompt,
            self.language,
            uploads=self.uploads,
            chat_session=self.chat_session,
        )
        return {
            "at": self.tokens.snlm0e,
            "f.req": json.dumps([None, json.dumps(inner)]),
        }

    def headers(self) -> Optional[Dict[str, str]]:
        return MODEL_HEADERS.get(self.model)


def extract_tokens(html: str) -> Optional[GeminiTokens]:
    snlm0e_match = re.search(r'SNlM0e\\\":\\\"(.*?)\\\"', html)
    if not snlm0e_match:
        snlm0e_match = re.search(r'SNlM0e":"(.*?)"', html)
    snlm0e = snlm0e_match.group(1) if snlm0e_match else None

    sid_match = re.search(r'"FdrFJe":"([\d-]+)"', html)
    sid = sid_match.group(1) if sid_match else None

    if not snlm0e:
        return None
    return GeminiTokens(snlm0e=snlm0e, sid=sid)


def build_request(
    prompt: str,
    language: str,
    *,
    uploads: Optional[Sequence[Tuple[str, str]]] = None,
    chat_session: Optional[ChatSession] = None,
) -> list:
    image_list = (
        [[[upload_ref, 1], image_name] for upload_ref, image_name in uploads]
        if uploads
        else []
    )
    conv_id = chat_session.conversation_id if chat_session else None
    resp_id = chat_session.response_id if chat_session else None
    choice_id = chat_session.choice_id if chat_session else None
    return [
        [prompt, 0, None, image_list, None, None, 0],
        [language],
        [conv_id, resp_id, choice_id, None, None, []],
        None,
        None,
        None,
        [1],
        0,
        [],
        [],
        1,
        0,
    ]


def iter_response_text_chunks(full_text: str) -> Iterator[str]:
    last_content = ""
    for raw_line in full_text.split("\n"):
        delta, last_content = extract_text_delta_from_raw_line(raw_line, last_content)
        if delta:
            yield delta


def extract_text_delta_from_raw_line(
    raw_line: str, last_content: str
) -> Tuple[Optional[str], str, Optional[ChatSession]]:
    """Extract incremental text delta and conversation IDs from one StreamGenerate response line.

    Returns (delta, new_last_content, chat_session). When the line doesn't contain text, returns (None, last_content, None).
    """

    def _flatten_strings(value):
        if isinstance(value, str):
            if value and not value.startswith("rc_"):
                yield value
            return
        if isinstance(value, list):
            for item in value:
                yield from _flatten_strings(item)

    def _extract_content(response_part):
        try:
            content = response_part[4][0][1][0]
            if isinstance(content, str):
                return content
        except Exception:
            pass

        try:
            content = response_part[4][0][1]
            if isinstance(content, str):
                return content
            if isinstance(content, list) and content and isinstance(content[0], str):
                return content[0]
        except Exception:
            pass

        try:
            candidates = list(_flatten_strings(response_part[4]))
            if candidates:
                return max(candidates, key=len)
        except Exception:
            pass

        return None

    try:
        # print(f"RAW_LINE: {raw_line}", file=__import__("sys").stderr)

        line = json.loads(raw_line)
    except Exception:
        return None, last_content, None
    if not isinstance(line, list) or not line:
        return None, last_content, None

    chat_session = None

    try:
        if len(line[0]) < 3 or not line[0][2]:
            return None, last_content, None
        response_part = json.loads(line[0][2])
        if not response_part or len(response_part) < 5:
            return None, last_content, None

        # Extract conversation IDs if available
        try:
            if isinstance(response_part[1], list) and len(response_part[1]) >= 2:
                conv_id = response_part[1][0]
                resp_id = response_part[1][1]
                choice_id = None
                if isinstance(response_part[4], list) and len(response_part[4]) > 0:
                    if isinstance(response_part[4][0], list) and len(response_part[4][0]) > 0:
                        choice_id = response_part[4][0][0]
                if conv_id and resp_id and choice_id:
                    chat_session = ChatSession(
                        conversation_id=conv_id,
                        response_id=resp_id,
                        choice_id=choice_id,
                    )
        except Exception:
            pass

        content = _extract_content(response_part)
        if not content:
            return None, last_content, chat_session
    except Exception:
        return None, last_content, chat_session

    if last_content and content.startswith(last_content):
        return content[len(last_content) :], content, chat_session
    return content, content, chat_session


def extract_image_candidates_from_raw_line(raw_line: str) -> Sequence[str]:
    """Extract image candidates (URLs or data URLs) from one StreamGenerate raw line."""

    def _walk_strings(value: Any) -> Iterator[str]:
        if isinstance(value, str):
            yield value
            return
        if isinstance(value, list):
            for item in value:
                yield from _walk_strings(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                yield from _walk_strings(item)

    def _is_likely_image_url(text: str) -> bool:
        if text.startswith("data:image/"):
            return True
        if not (text.startswith("https://") or text.startswith("http://")):
            return False
        lowered = text.lower()
        # Heuristics: Gemini web responses often reference these domains for media.
        if any(d in lowered for d in ["googleusercontent.com", "gstatic.com", "content-push.googleapis.com"]):
            return True
        if any(lowered.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
            return True
        return False

    try:
        line = json.loads(raw_line)
    except Exception:
        return []
    if not isinstance(line, list) or not line:
        return []

    try:
        if len(line[0]) < 3 or not line[0][2]:
            return []
        response_part = json.loads(line[0][2])
    except Exception:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for s in _walk_strings(response_part):
        if not s or s in seen:
            continue
        if _is_likely_image_url(s):
            seen.add(s)
            out.append(s)
    return out
