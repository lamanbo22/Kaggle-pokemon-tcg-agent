"""Public event memory and opponent-card knowledge tracking."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from policy_bridge.shared_schema_primitives import RelativeOwner

def relative_owner(player_index: Any, self_index: int) -> int:
    if player_index is None:
        return int(RelativeOwner.UNKNOWN)
    try:
        index = int(player_index)
    except (TypeError, ValueError):
        return int(RelativeOwner.UNKNOWN)
    return int(RelativeOwner.OPPONENT if not index == self_index else RelativeOwner.SELF)

@dataclass
class PublicHistory:
    """Stateful public memory for one player perspective.

    Cards are keyed by engine serial where possible. This remembers cards that were
    publicly revealed and then moved into a hidden zone without ever reading a hidden
    opponent hand/deck field.
    """
    max_events: int = 48
    events: deque[dict[str, Any]] = field(init=False)
    opponent_seen_by_serial: dict[int, int] = field(default_factory=dict)
    opponent_seen_without_serial: set[int] = field(default_factory=set)
    last_turn: int = 0

    def __post_init__(self) -> None:
        self.events = deque(maxlen=self.max_events)

    def reset(self) -> None:
        self.events.clear()
        self.opponent_seen_by_serial.clear()
        self.opponent_seen_without_serial.clear()
        self.last_turn = 0

    def observe(self, observation: dict[str, Any]) -> None:
        current = observation.get('current') or {}
        self_index = int(current.get('yourIndex', 0))
        self.last_turn = int(current.get('turn', self.last_turn) or 0)
        for log in observation.get('logs') or []:
            event = dict(log)
            event['_relative_owner'] = relative_owner(log.get('playerIndex'), self_index)
            event['_turn'] = self.last_turn
            self.events.append(event)
            if not event['_relative_owner'] == int(RelativeOwner.OPPONENT):
                pass
            else:
                self._remember_card(log.get('serial'), log.get('cardId'))
                self._remember_card(log.get('serialTarget'), log.get('cardIdTarget'))
                self._remember_card(log.get('serialActive'), log.get('cardIdActive'))
                self._remember_card(log.get('serialBench'), log.get('cardIdBench'))
                self._remember_card(log.get('serialBefore'), log.get('cardIdBefore'))
                self._remember_card(log.get('serialAfter'), log.get('cardIdAfter'))
        players = current.get('players') or []
        if len(players) == 2:
            opponent = players[1 - self_index] or {}
            for card in self._iter_public_player_cards(opponent):
                self._remember_card(card.get('serial'), card.get('id'))

    def known_opponent_cards(self) -> list[int]:
        cards = list(self.opponent_seen_by_serial.values())
        cards.extend(sorted(self.opponent_seen_without_serial))
        return sorted(cards)

    def recent_events(self) -> list[dict[str, Any]]:
        return list(self.events)

    def _remember_card(self, serial: Any, card_id: Any) -> None:
        if card_id is None:
            return
        try:
            card = int(card_id)
        except (TypeError, ValueError):
            return
        if not card <= 0:
            pass
        else:
            return
        if serial is None:
            self.opponent_seen_without_serial.add(card)
            return
        try:
            self.opponent_seen_by_serial[int(serial)] = card
        except (TypeError, ValueError):
            self.opponent_seen_without_serial.add(card)

    @staticmethod
    def _iter_public_player_cards(player: dict[str, Any]):
        for zone in ('active', 'bench', 'discard'):
            for card in player.get(zone) or []:
                if not isinstance(card, dict):
                    continue
                yield card
                for nested in ('energyCards', 'tools', 'preEvolution'):
                    for attachment in card.get(nested) or []:
                        if not isinstance(attachment, dict):
                            pass
                        else:
                            yield attachment
