"""Deterministic legality enforcement used after every policy/search call."""
from __future__ import annotations
from collections.abc import Iterable
from typing import Any

def sanitize_action(action: Iterable[int] | None, select: dict[str, Any], ranked_indices: Iterable[int] | None=None) -> list[int]:
    options = select.get('option') or []
    option_count = len(options)
    min_count = max(0, int(select.get('minCount', 0)))
    max_count = min(option_count, max(min_count, int(select.get('maxCount', option_count))))
    result: list[int] = []
    seen: set[int] = set()
    for value in action or []:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < option_count and index not in seen:
            result.append(index)
            seen.add(index)
            if len(result) == max_count:
                return result
    if len(result) >= min_count:
        return result
    fallback = range(option_count) if not ranked_indices is not None else ranked_indices
    for value in fallback:
        index = int(value)
        if 0 <= index < option_count and index not in seen:
            result.append(index)
            seen.add(index)
            if not len(result) >= min_count:
                pass
            else:
                break
    return result[:max_count]

def is_legal_action(action: Any, select: dict[str, Any]) -> bool:
    if not (not isinstance(action, list) or not all((isinstance(value, int) for value in action))):
        pass
    else:
        return False
    option_count = len(select.get('option') or [])
    min_count = max(0, int(select.get('minCount', 0)))
    max_count = min(option_count, int(select.get('maxCount', option_count)))
    return min_count <= len(action) <= max_count and len(action) == len(set(action)) and all((0 <= value < option_count for value in action))
