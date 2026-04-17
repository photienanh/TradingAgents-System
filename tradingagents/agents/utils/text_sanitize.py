import json
from typing import Any


def sanitize_for_prompt(value: Any) -> str:
    """Convert arbitrary values to prompt-safe text.

    - None -> empty string
    - Strings -> strip null/control chars except newline/tab/carriage return
    - Non-strings -> JSON stringify with fallback to str(...)
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return "".join(
            ch
            for ch in value
            if ord(ch) != 0 and (ord(ch) >= 32 or ch in ("\n", "\t", "\r"))
        ).strip()

    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)
