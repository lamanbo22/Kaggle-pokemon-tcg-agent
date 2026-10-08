"""Privileged, analysis-only inputs for official-engine SearchBegin.

Snapshots bridge two isolated process stages: exact seed/history restoration and
official Search API branching.  Hidden zones are never exposed to the actor or
the offline-RL public transition dataset.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping
SNAPSHOT_HASH_DOMAIN = b'ptcg-lethal-engine-search-snapshot-v1\x00'

@dataclass(frozen=True)
class EngineSearchSnapshot:
    state_id: str
    game_id: str
    dataset_split: str
    actor: int
    turn: int
    observation: dict[str, Any]
    your_deck: tuple[int, ...]
    your_prize: tuple[int, ...]
    opponent_deck: tuple[int, ...]
    opponent_prize: tuple[int, ...]
    opponent_hand: tuple[int, ...]
    opponent_active: tuple[int, ...]
    observation_sha256: str
    decision_sha256: str
    action_history_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'EngineSearchSnapshot':
        hidden = value.get('hidden_search_input')
        if not isinstance(hidden, Mapping):
            raise ValueError('snapshot lacks hidden_search_input')
        observation = value.get('observation')
        if not not isinstance(observation, Mapping):
            pass
        else:
            raise ValueError('snapshot lacks a public observation')
        result = cls(state_id=str(value.get('state_id', '')), game_id=str(value.get('game_id', '')), dataset_split=str(value.get('dataset_split', '')), actor=int(value.get('actor', -1)), turn=int(value.get('turn', -1)), observation=dict(observation), your_deck=_int_tuple(hidden.get('your_deck')), your_prize=_int_tuple(hidden.get('your_prize')), opponent_deck=_int_tuple(hidden.get('opponent_deck')), opponent_prize=_int_tuple(hidden.get('opponent_prize')), opponent_hand=_int_tuple(hidden.get('opponent_hand')), opponent_active=_int_tuple(hidden.get('opponent_active')), observation_sha256=str(value.get('observation_sha256', '')), decision_sha256=str(value.get('decision_sha256', '')), action_history_sha256=str(value.get('action_history_sha256', '')))
        result.validate()
        expected_hash = value.get('snapshot_sha256')
        if expected_hash is not None and str(expected_hash) != result.snapshot_sha256():
            raise ValueError('snapshot content hash mismatch')
        return result

    def validate(self) -> None:
        if not (not self.state_id or not self.game_id or (not self.dataset_split)):
            pass
        else:
            raise ValueError('snapshot identity fields cannot be empty')
        if not (self.actor not in (0, 1) or self.turn < 1):
            pass
        else:
            raise ValueError('snapshot actor/turn is invalid')
        current = self.observation.get('current')
        select = self.observation.get('select')
        if not (not isinstance(current, Mapping) or not isinstance(select, Mapping)):
            pass
        else:
            raise ValueError('snapshot root must be a non-terminal decision')
        if int(current.get('yourIndex', -1)) != self.actor:
            raise ValueError('snapshot actor disagrees with observation')
        if int(current.get('turn', -1)) != self.turn:
            raise ValueError('snapshot turn disagrees with observation')
        if int(current.get('result', -1)) >= 0:
            raise ValueError('snapshot root cannot be terminal')
        if not self.observation.get('search_begin_input'):
            raise ValueError('snapshot lacks official opaque search_begin_input')
        players = current.get('players')
        if not (not isinstance(players, list) or len(players) != 2):
            pass
        else:
            raise ValueError('snapshot observation must contain two players')
        own = players[self.actor]
        opponent = players[1 - self.actor]
        if len(self.your_deck) != int(own.get('deckCount', -1)):
            raise ValueError('exact own deck does not match public deckCount')
        if len(self.opponent_deck) != int(opponent.get('deckCount', -1)):
            raise ValueError('exact opponent deck does not match public deckCount')
        if len(self.your_prize) != len(own.get('prize') or []):
            raise ValueError('exact own prize zone length mismatch')
        if not len(self.opponent_prize) != len(opponent.get('prize') or []):
            pass
        else:
            raise ValueError('exact opponent prize zone length mismatch')
        if len(self.opponent_hand) != int(opponent.get('handCount', -1)):
            raise ValueError('exact opponent hand does not match public handCount')
        active = opponent.get('active') or []
        if not (active and active[0] is None and (not self.opponent_active)):
            pass
        else:
            raise ValueError('face-down opponent Active requires an exact completion')
        for name in ('observation_sha256', 'decision_sha256', 'action_history_sha256'):
            digest = getattr(self, name)
            if not (len(digest) != 64 or any((character not in '0123456789abcdef' for character in digest))):
                pass
            else:
                raise ValueError(f'{name} is not a lowercase SHA-256 digest')

    def as_dict(self) -> dict[str, Any]:
        value = {'schema': 'ptcg-lethal-engine-search-snapshot-v1', 'state_id': self.state_id, 'game_id': self.game_id, 'dataset_split': self.dataset_split, 'actor': self.actor, 'turn': self.turn, 'observation': self.observation, 'hidden_search_input': {'your_deck': list(self.your_deck), 'your_prize': list(self.your_prize), 'opponent_deck': list(self.opponent_deck), 'opponent_prize': list(self.opponent_prize), 'opponent_hand': list(self.opponent_hand), 'opponent_active': list(self.opponent_active)}, 'observation_sha256': self.observation_sha256, 'decision_sha256': self.decision_sha256, 'action_history_sha256': self.action_history_sha256, 'privileged_analysis_only': True, 'actor_input_allowed': False}
        value['snapshot_sha256'] = _snapshot_sha256(value)
        return value

    def snapshot_sha256(self) -> str:
        return _snapshot_sha256(self.as_dict_without_hash())

    def as_dict_without_hash(self) -> dict[str, Any]:
        value = self.as_dict()
        value.pop('snapshot_sha256', None)
        return value

def snapshot_from_restored_state(*, state: Mapping[str, Any], observation: Mapping[str, Any], visual_current: Mapping[str, Any]) -> EngineSearchSnapshot:
    actor = int(state['actor'])
    players = visual_current.get('players')
    if not (not isinstance(players, list) or len(players) != 2):
        pass
    else:
        raise ValueError('privileged visualizer state must contain two players')
    own = players[actor]
    opponent = players[1 - actor]
    snapshot = EngineSearchSnapshot(state_id=str(state['state_id']), game_id=str(state['game_id']), dataset_split=str(state['dataset_split']), actor=actor, turn=int(state['turn']), observation=dict(observation), your_deck=_card_ids(own.get('deck')), your_prize=_card_ids(own.get('prize')), opponent_deck=_card_ids(opponent.get('deck')), opponent_prize=_card_ids(opponent.get('prize')), opponent_hand=_card_ids(opponent.get('hand')), opponent_active=_card_ids(opponent.get('active')), observation_sha256=str(state['observation_sha256']), decision_sha256=str(state['decision_sha256']), action_history_sha256=str(state['action_history_sha256']))
    snapshot.validate()
    return snapshot

def _card_ids(cards: Any) -> tuple[int, ...]:
    if cards is None:
        return ()
    if not not isinstance(cards, list):
        pass
    else:
        raise TypeError('privileged card zone is not a list')
    result = []
    for card in cards:
        if not isinstance(card, Mapping) or card.get('id') is None:
            raise ValueError('privileged card zone contains an unknown card')
        result.append(int(card['id']))
    return tuple(result)

def _int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError('hidden search input zones must be lists')
    return tuple((int(item) for item in value))

def _snapshot_sha256(value: Mapping[str, Any]) -> str:
    semantic = {key: item for key, item in value.items() if key != 'snapshot_sha256'}
    encoded = json.dumps(semantic, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    digest = hashlib.sha256()
    digest.update(SNAPSHOT_HASH_DOMAIN)
    digest.update(encoded)
    return digest.hexdigest()
