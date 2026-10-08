"""Typed feature containers.

All arrays are already padded here so one representation can be used by replay BC,
self-play rollout collection, training, and submission inference.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True)
class FeatureSpec:
    max_entities: int = 96
    max_events: int = 48
    max_options: int = 128
    max_known_cards: int = 60
    max_actions: int = 64

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> 'FeatureSpec':
        if value is None:
            return cls()
        allowed = cls.__dataclass_fields__
        return cls(**{key: int(item) for key, item in value.items() if key in allowed})

@dataclass
class EncodedDecision:
    global_cat: np.ndarray
    global_num: np.ndarray
    own_deck: np.ndarray
    known_cards: np.ndarray
    entity_cat: np.ndarray
    entity_num: np.ndarray
    event_cat: np.ndarray
    event_num: np.ndarray
    option_cat: np.ndarray
    option_num: np.ndarray
    actions: np.ndarray
    entity_count: int
    event_count: int
    option_count: int
    action_count: int
    min_count: int
    max_count: int
    outcome: float = 0.0
    sample_weight: float = 1.0
    episode_id: str = ''
    deck_fingerprint: str = ''
    select_type: int = -1
    select_context: int = -1

    def arrays(self) -> dict[str, np.ndarray]:
        """Return only stackable training arrays."""
        return {'global_cat': self.global_cat, 'global_num': self.global_num, 'own_deck': self.own_deck, 'known_cards': self.known_cards, 'entity_cat': self.entity_cat, 'entity_num': self.entity_num, 'event_cat': self.event_cat, 'event_num': self.event_num, 'option_cat': self.option_cat, 'option_num': self.option_num, 'actions': self.actions, 'entity_count': np.asarray(self.entity_count, dtype=np.int16), 'event_count': np.asarray(self.event_count, dtype=np.int16), 'option_count': np.asarray(self.option_count, dtype=np.int16), 'action_count': np.asarray(self.action_count, dtype=np.int16), 'min_count': np.asarray(self.min_count, dtype=np.int16), 'max_count': np.asarray(self.max_count, dtype=np.int16), 'outcome': np.asarray(self.outcome, dtype=np.float32), 'sample_weight': np.asarray(self.sample_weight, dtype=np.float32)}
