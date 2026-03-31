"""Shared JSON extraction utilities for the quorum debate system.

Uses a balanced-brace scanner instead of naive regex to correctly handle
nested JSON objects, curly braces inside string values, and markdown
code-fenced JSON blocks.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _find_balanced_braces(text: str) -> list[str]:
    """Extract all top-level ``{…}`` substrings respecting nesting and string escaping."""
    candidates: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            start = i
            depth = 0
            in_string = False
            escape_next = False
            j = i
            while j < len(text):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    j += 1
                    continue
                if ch == "\\":
                    escape_next = True
                    j += 1
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            candidates.append(text[start : j + 1])
                            i = j + 1
                            break
                j += 1
            else:
                i += 1
        else:
            i += 1
    return candidates


def extract_last_json(text: str) -> dict[str, Any] | None:
    """Find and parse the last valid JSON object in ``text``.

    Strategy:
    1. Try to find JSON inside a markdown code fence first (````` ```json ... ``` `````).
    2. Fall back to balanced-brace extraction over the entire text.
    3. Return the last successfully parsed candidate (models typically end with JSON).
    """
    fenced = re.search(r"```(?:json)?\s*(\{.+\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1), strict=False)
        except json.JSONDecodeError:
            pass

    candidates = _find_balanced_braces(text)
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate, strict=False)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None
