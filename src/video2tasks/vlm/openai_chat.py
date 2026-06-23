"""OpenAI-compatible chat completions backend for vision requests."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests

from .base import VLMBackend


def _encode_png_data_url(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    t = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    s = t.find("{")
    e = t.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(t[s : e + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def _is_xiaomi_mimo_openai_base(base_url: str) -> bool:
    """MiMo（小米）OpenAI 兼容网关：用于选择请求体字段等 MiMo 专用行为。"""
    return "xiaomimimo.com" in (base_url or "").lower()


def _openai_chat_uses_max_completion_tokens(base_url: str, model: str) -> bool:
    """MiMo 统一用 max_completion_tokens；AIHubMix 仅当模型名为 gpt* 时（OpenAI 系）改用该字段。"""
    u = (base_url or "").lower()
    m = (model or "").lower().strip()
    if "xiaomimimo.com" in u:
        return True
    if "aihubmix.com" in u and m.startswith("gpt"):
        return True
    return False


def _normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text", "")))
        return "\n".join(x for x in chunks if x)
    return str(content)


class OpenAIChatBackend(VLMBackend):
    """OpenAI-compatible backend, e.g. AIHubMix."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_sec: float = 120.0,
        max_tokens: int = 2048,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_sec = float(timeout_sec)
        self.max_tokens = int(max_tokens)
        self.headers = headers or {}

    @property
    def name(self) -> str:
        return "openai_chat"

    def infer(self, images: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            print("[OpenAIChat] Missing api_key")
            return {}
        if not self.model:
            print("[OpenAIChat] Missing model")
            return {}

        image_parts: List[Dict[str, Any]] = []
        for img in images:
            data_url = _encode_png_data_url(img)
            if not data_url:
                continue
            image_parts.append(
                {"type": "image_url", "image_url": {"url": data_url}}
            )
        text_part: Dict[str, Any] = {"type": "text", "text": prompt}
        # MiMo 图像理解文档（image-understanding）：示例中 user content 为「先 image_url、后 text」；
        # 且文档写明当前仅 mimo-v2.5、mimo-v2-omni 支持图像理解（非 Pro）。
        if _is_xiaomi_mimo_openai_base(self.base_url):
            content = image_parts + [text_part]
        else:
            content = [text_part] + image_parts

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        }
        # MiMo 全量；AIHubMix 仅 gpt* 模型：网关要求 max_completion_tokens 而非 max_tokens。
        if _openai_chat_uses_max_completion_tokens(self.base_url, self.model):
            payload["max_completion_tokens"] = self.max_tokens
        else:
            payload["max_tokens"] = self.max_tokens
        req_headers: Dict[str, str] = {
            "Content-Type": "application/json",
            **self.headers,
        }
        if not any(k.lower() == "authorization" for k in req_headers):
            req_headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        try:
            r = requests.post(url, json=payload, headers=req_headers, timeout=self.timeout_sec)
        except requests.RequestException as e:
            print(f"[OpenAIChat] Request failed: {e}")
            return {}

        if r.status_code != 200:
            print(f"[OpenAIChat] HTTP {r.status_code}: {r.text[:500]}")
            return {}

        try:
            data = r.json()
        except json.JSONDecodeError as e:
            print(f"[OpenAIChat] Invalid JSON response: {e}")
            return {}

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return {}

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            return {}

        raw = _normalize_message_content(message.get("content"))
        parsed = _extract_json(raw)
        if isinstance(parsed, dict) and ("transitions" in parsed or "instructions" in parsed):
            return parsed
        return {}
