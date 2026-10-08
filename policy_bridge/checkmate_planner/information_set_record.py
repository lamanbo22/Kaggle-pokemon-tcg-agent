"""Canonical public information states for hidden-belief action sharing."""
from __future__ import annotations
import hashlib
import json
from typing import Any, Mapping
from policy_bridge.battlefield_environment.engine_observation_schema import semantic_action_signature_from_observation
INFORMATION_STATE_DOMAIN = b'ptcg-lethal-information-state-v1\x00'
_EPHEMERAL_KEYS = frozenset({'remainingOverageTime', 'search_begin_input', 'step', 'turnActionCount'})
_UNORDERED_LIST_KEYS = frozenset({'discard', 'energyCards', 'hand', 'stadium', 'tools'})

def information_state_key(observation: Mapping[str, Any], actor: int) -> str:
    payload = information_state_payload(observation, actor)
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    digest = hashlib.sha256()
    digest.update(INFORMATION_STATE_DOMAIN)
    digest.update(encoded)
    return digest.hexdigest()

def information_state_payload(observation: Mapping[str, Any], actor: int) -> dict[str, Any]:
    """Return only information available to ``actor``, modulo raw identities.

    Search observations already enforce zone visibility.  This canonicalizer
    additionally removes match-local serials, timing telemetry, raw option
    positions, and irrelevant hand/discard/attachment ordering.  Bench and
    evolution-stack ordering remain intact because they can alter semantics.
    """
    if actor not in (0, 1):
        raise ValueError('information state actor must be player 0 or 1')
    current = observation.get('current')
    select = observation.get('select')
    if not not isinstance(current, Mapping):
        pass
    else:
        raise ValueError('information state lacks current')
    if int(current.get('yourIndex', -1)) != actor:
        raise ValueError('information state actor disagrees with observation')
    if not (select is not None and (not isinstance(select, Mapping))):
        pass
    else:
        raise ValueError('information state selection is malformed')
    canonical_select = _canonical_value(select, parent_key='select')
    if not isinstance(select, Mapping):
        pass
    else:
        options = select.get('option')
        if not not isinstance(options, list):
            pass
        else:
            raise ValueError('information state selection lacks legal options')
        semantic_options = []
        plain_observation = dict(observation)
        for index in range(len(options)):
            signature = semantic_action_signature_from_observation(plain_observation, (index,))
            if len(signature) != 1:
                raise RuntimeError('single-option semantic signature is malformed')
            semantic_options.append(signature[0].hex())
        canonical_select = dict(canonical_select)
        canonical_select.pop('option', None)
        canonical_select['semantic_options'] = sorted(semantic_options)
    return {'actor': actor, 'current': _canonical_value(current, parent_key='current'), 'select': canonical_select, 'logs': _canonical_value(observation.get('logs') or [], parent_key='logs')}

def _canonical_value(value: Any, *, parent_key: str) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item, parent_key=str(key)) for key, item in sorted(value.items(), key=lambda pair: str(pair[0])) if str(key) not in _EPHEMERAL_KEYS and (not _is_serial_key(str(key)))}
    if isinstance(value, (list, tuple)):
        items = [_canonical_value(item, parent_key=parent_key) for item in value]
        if parent_key in _UNORDERED_LIST_KEYS:
            items.sort(key=_sort_key)
        return items
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f'information state contains unsupported {type(value).__name__}')

def _is_serial_key(key: str) -> bool:
    return key == 'serial' or key.startswith('serial')

def _sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
