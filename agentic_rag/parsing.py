from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(cleaned[start : index + 1])
                    return value if isinstance(value, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def render_cited_claims(payload: dict[str, Any], hit_count: int) -> str | None:
    """Render only schema-valid claim/citation pairs from an LLM repair response."""
    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        return None
    lines: list[str] = []
    for item in claims:
        if not isinstance(item, dict):
            return None
        claim_text = str(item.get("text", "")).strip()
        citation = item.get("citation")
        if (
            not claim_text
            or "原始证据" in claim_text
            or not isinstance(citation, int)
            or not 1 <= citation <= hit_count
        ):
            return None
        lines.append(f"- {claim_text} [{citation}]")
    return "\n".join(lines)
