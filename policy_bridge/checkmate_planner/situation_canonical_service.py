"""Proof-conservative canonical keys for chance-free one-turn Search states."""
from __future__ import annotations
import hashlib
import json
from typing import Any, Mapping
CANONICAL_STATE_DOMAIN = b'ptcg-lethal-proof-state-v1\x00'
_EPHEMERAL_KEYS = frozenset({'logs', 'remainingOverageTime', 'search_begin_input', 'step', 'turnActionCount'})

def canonical_state_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete decision state relevant to a chance-free suffix.

    Search observations expose the acting player's private hand and any cards
    currently being looked at, plus all public state and legal options.  LTH-3
    never crosses shuffle, draw, facedown-move, or coin outcomes; hidden-order
    branches are therefore left UNKNOWN before this key may justify a proof.
    """
    current = observation.get('current')
    select = observation.get('select')
    if not isinstance(current, Mapping):
        raise ValueError('canonical Search state lacks current')
    if select is not None and (not isinstance(select, Mapping)):
        raise ValueError('canonical Search selection is not a mapping')
    return {'current': _canonical_value(current), 'select': _canonical_value(select)}

def canonical_state_key(observation: Mapping[str, Any]) -> str:
    payload = canonical_state_payload(observation)
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    digest = hashlib.sha256()
    digest.update(CANONICAL_STATE_DOMAIN)
    digest.update(encoded)
    return digest.hexdigest()

def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0])) if str(key) not in _EPHEMERAL_KEYS}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if not (isinstance(value, (str, int, float, bool)) or value is None):
        pass
    else:
        return value
    raise TypeError(f'canonical Search state contains unsupported {type(value).__name__}')
