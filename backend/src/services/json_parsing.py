"""Shared helpers for extracting structured JSON from LLM output."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from config import Configuration
from utils import strip_thinking_tokens

TOOL_CALL_PATTERN = re.compile(
    r"\[TOOL_CALL:(?P<tool>[^:]+):(?P<body>[^\]]+)\]",
    re.IGNORECASE,
)


def extract_json_payload(text: str) -> Optional[dict[str, Any] | list]:
    """Try to locate and parse a JSON object or array from the text."""

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    return None


def extract_tool_payload(text: str) -> Optional[dict[str, Any]]:
    """Parse the first TOOL_CALL expression in the output."""

    match = TOOL_CALL_PATTERN.search(text)
    if not match:
        return None

    body = match.group("body")

    try:
        payload = json.loads(body)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    parts = [segment.strip() for segment in body.split(",") if segment.strip()]
    payload: dict[str, Any] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        payload[key.strip()] = value.strip().strip('"').strip("'")

    return payload or None


def parse_structured_json(text: str, config: Configuration) -> Optional[dict[str, Any]]:
    """Strip thinking tokens then extract a JSON object from LLM output."""

    cleaned = text.strip()
    if config.strip_thinking_tokens:
        cleaned = strip_thinking_tokens(cleaned)

    payload = extract_json_payload(cleaned)
    return payload if isinstance(payload, dict) else None
