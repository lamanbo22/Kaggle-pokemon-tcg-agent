"""Chance-event accounting restricted to the root player's one-turn horizon."""
from __future__ import annotations
from typing import Any, Mapping
CHANCE_LOG_TYPES = frozenset({0, 4, 5, 7, 22})
TURN_START_LOG_TYPE = 2
TURN_END_LOG_TYPE = 3

def in_horizon_chance_events(observation: Mapping[str, Any], *, player_id: int) -> tuple[Mapping[str, Any], ...]:
    """Return chance logs occurring before ``player_id``'s turn boundary.

    One engine selection may advance through TURN_END, the opponent TURN_START,
    and the opponent's mandatory draw.  That draw is outside a one-turn lethal
    question and must not make an otherwise exhaustive branch probabilistic.
    Chance events before the boundary remain proof-relevant.
    """
    if player_id not in (0, 1):
        raise ValueError('one-turn chance accounting requires player 0 or 1')
    logs = observation.get('logs') or []
    if not isinstance(logs, list):
        raise ValueError('engine observation logs are not a list')
    events = []
    for row in logs:
        if not isinstance(row, Mapping):
            continue
        log_type = int(row.get('type', -1))
        owner = int(row.get('playerIndex', -1))
        if log_type == TURN_END_LOG_TYPE and owner == player_id or (log_type == TURN_START_LOG_TYPE and owner == 1 - player_id):
            break
        if not log_type in CHANCE_LOG_TYPES:
            pass
        else:
            events.append(row)
    return tuple(events)

def in_horizon_chance_event_count(observation: Mapping[str, Any], *, player_id: int) -> int:
    return len(in_horizon_chance_events(observation, player_id=player_id))
