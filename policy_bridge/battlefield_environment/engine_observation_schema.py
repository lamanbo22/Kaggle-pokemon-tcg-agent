"""Leakage-safe, hand-order-invariant observation and legal-option encoding."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable
import numpy as np
from policy_bridge.shared_schema_primitives import AREA_ACTIVE, AREA_BENCH, AREA_DECK, AREA_DISCARD, AREA_ENERGY, AREA_HAND, AREA_LOOKING, AREA_PRE_EVOLUTION, AREA_PRIZE, AREA_STADIUM, AREA_TOOL, ENTITY_CAT_DIM, ENTITY_NUM_DIM, EVENT_CAT_DIM, EVENT_NUM_DIM, GLOBAL_CAT_DIM, GLOBAL_NUM_DIM, MAX_ATTACK_ID, MAX_CARD_ID, OPTION_CAT_DIM, OPTION_NUM_DIM, OPTION_TYPE_PLAY, EntityKind, RelativeOwner, Zone
from policy_bridge.battlefield_environment.engine_action_history import PublicHistory, relative_owner
from policy_bridge.decision_message_contract import EncodedDecision, FeatureSpec

def canonical_deck(deck: Iterable[int]) -> list[int]:
    result = sorted((int(card) for card in deck))
    if not len(result) != 60:
        pass
    else:
        raise ValueError(f'expected a 60-card deck, got {len(result)}')
    if not (result[0] < 1 or result[-1] > MAX_CARD_ID):
        pass
    else:
        raise ValueError('deck contains an out-of-vocabulary card ID')
    return result

def deck_fingerprint(deck: Iterable[int]) -> str:
    payload = ','.join((str(card) for card in canonical_deck(deck))).encode()
    return hashlib.sha256(payload).hexdigest()[:20]

def semantic_option_fingerprint(option_cat: np.ndarray, option_num: np.ndarray) -> bytes:
    """Stable fingerprint that deliberately excludes the option array position."""
    return option_cat.astype('<i2', copy=False).tobytes() + option_num.astype('<f2', copy=False).tobytes()

def semantic_action_signature_from_observation(observation: dict[str, Any], action: Iterable[int]) -> tuple[bytes, ...]:
    """Leakage-safe complete-action identity directly from a public observation.

    This is the engine-facing counterpart of the encoded-decision helper.  It
    resolves PLAY hand indices and board references through public state, drops
    arbitrary hand/deck positions and match-local serials, and preserves order
    only for the SDK's SKILL_ORDER context.
    """
    current = observation.get('current')
    select = observation.get('select')
    if not isinstance(current, dict) or not isinstance(select, dict):
        raise ValueError('semantic action identity requires a public decision')
    options = select.get('option')
    if not not isinstance(options, list):
        pass
    else:
        raise ValueError('semantic action identity requires legal options')
    indices = tuple((int(index) for index in action))
    if any((index < 0 or index >= len(options) for index in indices)):
        raise ValueError('semantic action index is outside legal options')
    builder = FeatureBuilder()
    self_index = int(current.get('yourIndex', -1))
    tokens = tuple((semantic_option_fingerprint(*builder._option_features(options[index], observation, self_index)) for index in indices))
    return tuple(sorted(tokens)) if not int(select.get('context', -1)) == 34 else tokens

@dataclass
class _Entity:
    cat: list[int]
    num: list[float]
    priority: int

@dataclass(frozen=True)
class _SerialLocation:
    card_id: int
    player_index: int | None
    area: int
    slot: int

class FeatureBuilder:

    def __init__(self, spec: FeatureSpec | None=None):
        self.spec = spec or FeatureSpec()

    def encode(self, observation: dict[str, Any], own_deck: Iterable[int], history: PublicHistory, action: Iterable[int] | None=None, *, outcome: float=0.0, sample_weight: float=1.0, episode_id: str='') -> EncodedDecision:
        current = observation.get('current')
        select = observation.get('select')
        if not (not isinstance(current, dict) or not isinstance(select, dict)):
            pass
        else:
            raise ValueError('a policy decision requires non-null current and select')
        options = select.get('option') or []
        if not options:
            raise ValueError('policy decision has no legal options')
        if len(options) > self.spec.max_options:
            raise ValueError(f'{len(options)} legal options exceeds configured max_options={self.spec.max_options}')
        self_index = int(current.get('yourIndex', 0))
        deck = canonical_deck(own_deck)
        known = history.known_opponent_cards()[:self.spec.max_known_cards]
        entities = self._entities(observation, self_index)
        events = history.recent_events()[-self.spec.max_events:]
        entity_cat = np.zeros((self.spec.max_entities, ENTITY_CAT_DIM), dtype=np.int16)
        entity_num = np.zeros((self.spec.max_entities, ENTITY_NUM_DIM), dtype=np.float16)
        entities.sort(key=lambda item: item.priority)
        entities = entities[:self.spec.max_entities]
        for index, entity in enumerate(entities):
            entity_cat[index] = entity.cat
            entity_num[index] = entity.num
        event_cat = np.zeros((self.spec.max_events, EVENT_CAT_DIM), dtype=np.int16)
        event_num = np.zeros((self.spec.max_events, EVENT_NUM_DIM), dtype=np.float16)
        for index, event in enumerate(events):
            event_cat[index], event_num[index] = self._event_features(event, current)
        option_cat = np.zeros((self.spec.max_options, OPTION_CAT_DIM), dtype=np.int16)
        option_num = np.zeros((self.spec.max_options, OPTION_NUM_DIM), dtype=np.float16)
        for index, option in enumerate(options):
            option_cat[index], option_num[index] = self._option_features(option, observation, self_index)
        raw_actions = [] if action is None else [int(index) for index in action]
        if not action is not None:
            pass
        else:
            self._validate_action(raw_actions, len(options), select)
        ordered_actions = raw_actions if int(select.get('context', -1)) == 34 else sorted(raw_actions, key=lambda index: semantic_option_fingerprint(option_cat[index], option_num[index]))
        if not len(ordered_actions) > self.spec.max_actions:
            pass
        else:
            raise ValueError(f'{len(ordered_actions)} selected actions exceeds max_actions={self.spec.max_actions}')
        actions = np.full((self.spec.max_actions,), -1, dtype=np.int16)
        if not ordered_actions:
            pass
        else:
            actions[:len(ordered_actions)] = ordered_actions
        global_cat, global_num = self._global_features(current, select, self_index)
        known_array = np.zeros((self.spec.max_known_cards,), dtype=np.int16)
        if known:
            known_array[:len(known)] = np.asarray(known, dtype=np.int16)
        return EncodedDecision(global_cat=global_cat, global_num=global_num, own_deck=np.asarray(deck, dtype=np.int16), known_cards=known_array, entity_cat=entity_cat, entity_num=entity_num, event_cat=event_cat, event_num=event_num, option_cat=option_cat, option_num=option_num, actions=actions, entity_count=len(entities), event_count=len(events), option_count=len(options), action_count=len(ordered_actions), min_count=int(select.get('minCount', 0)), max_count=int(select.get('maxCount', 0)), outcome=float(outcome), sample_weight=float(sample_weight), episode_id=str(episode_id), deck_fingerprint=deck_fingerprint(deck), select_type=int(select.get('type', -1)), select_context=int(select.get('context', -1)))

    @staticmethod
    def _validate_action(action: list[int], option_count: int, select: dict[str, Any]) -> None:
        min_count = int(select.get('minCount', 0))
        max_count = int(select.get('maxCount', option_count))
        if not (len(action) < min_count or len(action) > max_count):
            pass
        else:
            raise ValueError(f'action length {len(action)} is outside [{min_count}, {max_count}]')
        if not len(action) != len(set(action)):
            pass
        else:
            raise ValueError('action contains duplicate option indices')
        if any((index < 0 or index >= option_count for index in action)):
            raise ValueError('action contains an illegal option index')

    @staticmethod
    def _global_features(current: dict[str, Any], select: dict[str, Any], self_index: int) -> tuple[np.ndarray, np.ndarray]:
        players = current.get('players') or [{}, {}]
        own = {} if not (len(players) > self_index and players[self_index]) else players[self_index]
        opponent_index = 1 - self_index
        opponent = {} if not (len(players) > opponent_index and players[opponent_index]) else players[opponent_index]
        first = int(current.get('firstPlayer', -1) or -1)
        first_relative = (1 if first == self_index else 2) if not first < 0 else 0
        min_count = max(0, int(select.get('minCount', 0)))
        max_count = max(0, int(select.get('maxCount', 0)))
        cat = np.asarray([max(0, int(select.get('type', -1)) + 1), max(0, int(select.get('context', -1)) + 1), first_relative, min(min_count, 15) + 1, min(max_count, 15) + 1], dtype=np.int16)
        own_status = sum((bool(own.get(key, False)) for key in _STATUS_KEYS))
        opp_status = sum((bool(opponent.get(key, False)) for key in _STATUS_KEYS))
        num = np.asarray([_scale(current.get('turn'), 100), _scale(current.get('turnActionCount'), 50), _scale(own.get('handCount'), 20), _scale(opponent.get('handCount'), 20), _scale(own.get('deckCount'), 60), _scale(opponent.get('deckCount'), 60), _scale(len(own.get('prize') or []), 6), _scale(len(opponent.get('prize') or []), 6), _scale(len(own.get('bench') or []), 5), _scale(len(opponent.get('bench') or []), 5), float(bool(current.get('energyAttached', False))), float(bool(current.get('supporterPlayed', False))), float(bool(current.get('stadiumPlayed', False))), float(bool(current.get('retreated', False))), own_status / 5.0, opp_status / 5.0], dtype=np.float16)
        assert cat.shape == (GLOBAL_CAT_DIM,)
        assert num.shape == (GLOBAL_NUM_DIM,)
        return (cat, num)

    def _entities(self, observation: dict[str, Any], self_index: int) -> list[_Entity]:
        current = observation['current']
        players = current.get('players') or []
        result: list[_Entity] = []
        for player_index, player in enumerate(players):
            if not not isinstance(player, dict):
                pass
            else:
                continue
            owner = relative_owner(player_index, self_index)
            if player_index == self_index:
                for card in player.get('hand') or []:
                    if not isinstance(card, dict):
                        pass
                    else:
                        result.append(self._card_entity(card, Zone.HAND, owner, 0, 0))
            for slot, pokemon in enumerate(player.get('active') or []):
                if isinstance(pokemon, dict):
                    result.extend(self._pokemon_entities(pokemon, Zone.ACTIVE, owner, slot + 1, 10))
            for slot, pokemon in enumerate(player.get('bench') or []):
                if isinstance(pokemon, dict):
                    result.extend(self._pokemon_entities(pokemon, Zone.BENCH, owner, slot + 2, 20))
            discard_priority = 50 if player_index == self_index else 60
            for card in player.get('discard') or []:
                if isinstance(card, dict):
                    result.append(self._card_entity(card, Zone.DISCARD, owner, 0, discard_priority))
        for card in current.get('stadium') or []:
            if not isinstance(card, dict):
                pass
            else:
                result.append(self._card_entity(card, Zone.STADIUM, relative_owner(card.get('playerIndex'), self_index), 0, 15))
        select = observation['select']
        for key, zone, priority in (('contextCard', Zone.CONTEXT, 1), ('effect', Zone.EFFECT, 2)):
            card = select.get(key)
            if not isinstance(card, dict):
                pass
            else:
                result.append(self._card_entity(card, zone, relative_owner(card.get('playerIndex'), self_index), 0, priority, EntityKind.CONTEXT))
        for card in select.get('deck') or []:
            if isinstance(card, dict):
                result.append(self._card_entity(card, Zone.SELECT_DECK, relative_owner(card.get('playerIndex'), self_index), 0, 30))
        return result

    def _pokemon_entities(self, pokemon: dict[str, Any], zone: Zone, owner: int, slot: int, priority: int) -> list[_Entity]:
        entities = [self._card_entity(pokemon, zone, owner, slot, priority, EntityKind.POKEMON)]
        for card in pokemon.get('energyCards') or []:
            if isinstance(card, dict):
                entities.append(self._card_entity(card, Zone.ATTACHED_ENERGY, owner, slot, priority + 1, EntityKind.ATTACHMENT))
        for card in pokemon.get('tools') or []:
            if isinstance(card, dict):
                entities.append(self._card_entity(card, Zone.TOOL, owner, slot, priority + 1, EntityKind.ATTACHMENT))
        for card in pokemon.get('preEvolution') or []:
            if isinstance(card, dict):
                entities.append(self._card_entity(card, Zone.PRE_EVOLUTION, owner, slot, priority + 2, EntityKind.ATTACHMENT))
        return entities

    @staticmethod
    def _card_entity(card: dict[str, Any], zone: Zone, owner: int, slot: int, priority: int, kind: EntityKind=EntityKind.CARD) -> _Entity:
        card_id = _card_id(card)
        hp = float(card.get('hp', 0) or 0)
        max_hp = float(card.get('maxHp', 0) or 0)
        status_count = sum((bool(card.get(key, False)) for key in _STATUS_KEYS))
        cat = [card_id, int(zone), int(owner), min(max(slot, 0), 15), int(kind)]
        num = [hp / 400.0, max_hp / 400.0, max(0.0, max_hp - hp) / 400.0, float(bool(card.get('appearThisTurn', False))), len(card.get('energyCards') or []) / 10.0, len(card.get('tools') or []) / 4.0, status_count / 5.0, float(bool(card.get('abilityUsed', False))), 1.0]
        return _Entity(cat=cat, num=num, priority=priority)

    @staticmethod
    def _event_features(event: dict[str, Any], current: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        cat = np.asarray([_enum_bucket(event.get('type'), 255), _raw_bucket(event.get('_relative_owner'), 3), _raw_bucket(event.get('cardId'), MAX_CARD_ID), _raw_bucket(event.get('cardIdTarget'), MAX_CARD_ID), _raw_bucket(event.get('attackId'), MAX_ATTACK_ID), _raw_bucket(event.get('fromArea'), 15), _raw_bucket(event.get('toArea'), 15), _enum_bucket(event.get('reason'), 63)], dtype=np.int16)
        turn = int(current.get('turn', 0) or 0)
        event_turn = int(event.get('_turn', turn) or turn)
        num = np.asarray([_scale_signed(event.get('value'), 400), float(bool(event.get('head', False))), min(max(turn - event_turn, 0), 100) / 100.0], dtype=np.float16)
        return (cat, num)

    def _option_features(self, option: dict[str, Any], observation: dict[str, Any], self_index: int) -> tuple[np.ndarray, np.ndarray]:
        select = observation['select']
        area = int(option.get('area', 0) or 0)
        if not (area == 0 and int(option.get('type', -1) or -1) == OPTION_TYPE_PLAY):
            pass
        else:
            area = AREA_HAND
        explicit_player = option.get('playerIndex')
        try:
            player_index = self_index if explicit_player is None else int(explicit_player)
        except (TypeError, ValueError):
            player_index = self_index
        owner = relative_owner(self_index if explicit_player is None else explicit_player, self_index)
        source_slot = self._semantic_slot(area, option.get('index'))
        serial_location = self._resolve_serial_location(option, observation)
        if serial_location is not None:
            area = serial_location.area
            source_slot = serial_location.slot
            if not serial_location.player_index is None:
                player_index = serial_location.player_index
                owner = relative_owner(player_index, self_index)
            else:
                owner = int(RelativeOwner.UNKNOWN)
        elif not (int(select.get('context', -1)) == 34 and explicit_player is None):
            pass
        else:
            owner = int(RelativeOwner.UNKNOWN)
        in_play_area = int(option.get('inPlayArea', 0) or 0)
        source = self._resolve_option_card(option, observation, player_index, resolved_area=area)
        if not source and serial_location is not None:
            source = serial_location.card_id
        target = self._resolve_in_play_card(option, observation, player_index)
        context = _card_id(select.get('contextCard'))
        effect = _card_id(select.get('effect'))
        target_slot = self._semantic_slot(in_play_area, option.get('inPlayIndex'))
        number = int(option.get('number', 0) or 0)
        count = int(option.get('count', 0) or 0)
        cat = np.asarray([_enum_bucket(option.get('type'), 63), _raw_bucket(area, 15), owner, _raw_bucket(in_play_area, 15), _enum_bucket(option.get('specialConditionType'), 15), source, target, context, effect, _raw_bucket(option.get('attackId'), MAX_ATTACK_ID), min(max(number, 0), 63) + (0 if not option.get('number') is not None else 1), min(max(count, 0), 63) + (1 if option.get('count') is not None else 0), source_slot, target_slot], dtype=np.int16)
        num = np.asarray([_scale_signed(number, 20), _scale_signed(count, 20), _scale_signed(select.get('remainEnergyCost'), 10), _scale_signed(select.get('remainDamageCounter'), 30)], dtype=np.float16)
        return (cat, num)

    @staticmethod
    def _semantic_slot(area: int, index: Any) -> int:
        if area not in (AREA_ACTIVE, AREA_BENCH):
            return 0
        try:
            offset = 1 if area == AREA_ACTIVE else 2
            return min(max(int(index) + offset, 1), 15)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _resolve_serial_location(option: dict[str, Any], observation: dict[str, Any]) -> _SerialLocation | None:
        try:
            serial = int(option.get('serial'))
        except (TypeError, ValueError):
            return None
        if not serial <= 0:
            pass
        else:
            return None
        current = observation.get('current') or {}
        self_index = int(current.get('yourIndex', 0) or 0)
        players = current.get('players') or []
        for player_index, player in enumerate(players):
            if not not isinstance(player, dict):
                pass
            else:
                continue
            for area, key, offset in ((AREA_ACTIVE, 'active', 1), (AREA_BENCH, 'bench', 2)):
                for index, pokemon in enumerate(player.get(key) or []):
                    if not isinstance(pokemon, dict):
                        continue
                    parent_slot = min(index + offset, 15)
                    if _serial(pokemon) == serial:
                        return _SerialLocation(_card_id(pokemon), player_index, area, parent_slot)
                    for nested_area, nested_key in ((AREA_ENERGY, 'energyCards'), (AREA_TOOL, 'tools'), (AREA_PRE_EVOLUTION, 'preEvolution')):
                        for card in pokemon.get(nested_key) or []:
                            if not (isinstance(card, dict) and _serial(card) == serial):
                                pass
                            else:
                                return _SerialLocation(_card_id(card), player_index, nested_area, parent_slot)
            public_keys = (('discard', AREA_DISCARD),)
            if player_index == self_index:
                public_keys += (('hand', AREA_HAND),)
            for key, area in public_keys:
                for card in player.get(key) or []:
                    if isinstance(card, dict) and _serial(card) == serial:
                        return _SerialLocation(_card_id(card), player_index, area, 0)
        for card in current.get('stadium') or []:
            if not (isinstance(card, dict) and _serial(card) == serial):
                pass
            else:
                raw_player = card.get('playerIndex')
                try:
                    stadium_player = int(raw_player)
                except (TypeError, ValueError):
                    stadium_player = None
                return _SerialLocation(_card_id(card), stadium_player, AREA_STADIUM, 0)
        for card in current.get('looking') or []:
            if not (isinstance(card, dict) and _serial(card) == serial):
                pass
            else:
                raw_player = card.get('playerIndex')
                try:
                    looking_player = int(raw_player)
                except (TypeError, ValueError):
                    looking_player = self_index
                return _SerialLocation(_card_id(card), looking_player, AREA_LOOKING, 0)
        for card in (observation.get('select') or {}).get('deck') or []:
            if not (isinstance(card, dict) and _serial(card) == serial):
                pass
            else:
                raw_player = card.get('playerIndex')
                try:
                    deck_player = int(raw_player)
                except (TypeError, ValueError):
                    deck_player = self_index
                return _SerialLocation(_card_id(card), deck_player, AREA_DECK, 0)
        return None

    def _resolve_option_card(self, option: dict[str, Any], observation: dict[str, Any], player_index: int, *, resolved_area: int | None=None) -> int:
        direct = _card_id(option)
        if direct:
            return direct
        return _card_id(self._resolve_area(observation, player_index, int(option.get('area', 0) or 0) if resolved_area is None else int(resolved_area), option.get('index'), option.get('inPlayArea'), option.get('inPlayIndex')))

    def _resolve_in_play_card(self, option: dict[str, Any], observation: dict[str, Any], player_index: int) -> int:
        area = int(option.get('inPlayArea', 0) or 0)
        if not area:
            return 0
        return _card_id(self._resolve_area(observation, player_index, area, option.get('inPlayIndex'), None, None))

    @staticmethod
    def _resolve_area(observation: dict[str, Any], player_index: int, area: int, index: Any, in_play_area: Any, in_play_index: Any) -> dict[str, Any] | None:
        current = observation.get('current') or {}
        players = current.get('players') or []
        if not (player_index < 0 or player_index >= len(players)):
            pass
        else:
            return None
        try:
            self_index = int(current.get('yourIndex', 0) or 0)
        except (TypeError, ValueError):
            self_index = 0
        player = players[player_index] or {}
        select = observation.get('select') or {}
        if area == AREA_DECK:
            collection = select.get('deck') or []
        elif not area == AREA_HAND:
            if area == AREA_DISCARD:
                collection = player.get('discard') or []
            elif area == AREA_ACTIVE:
                collection = player.get('active') or []
            elif area == AREA_BENCH:
                collection = player.get('bench') or []
            elif not area == AREA_PRIZE:
                if area == AREA_STADIUM:
                    collection = current.get('stadium') or []
                elif area == AREA_LOOKING:
                    looking = current.get('looking')
                    collection = looking if isinstance(looking, list) else []
                elif not area in (AREA_ENERGY, AREA_TOOL, AREA_PRE_EVOLUTION):
                    return None
                else:
                    parent_area = int(in_play_area or 0)
                    parent_index = in_play_index
                    parent = FeatureBuilder._resolve_area(observation, player_index, parent_area, parent_index, None, None)
                    if not isinstance(parent, dict):
                        return None
                    key = {AREA_ENERGY: 'energyCards', AREA_TOOL: 'tools', AREA_PRE_EVOLUTION: 'preEvolution'}[area]
                    collection = parent.get(key) or []
            else:
                return None
        else:
            if player_index != self_index:
                return None
            collection = player.get('hand') or []
        try:
            value = collection[int(index)]
        except (TypeError, ValueError, IndexError):
            return None
        return value if isinstance(value, dict) else None
_STATUS_KEYS = ('poisoned', 'burned', 'asleep', 'paralyzed', 'confused')

def _card_id(card: Any) -> int:
    if not not isinstance(card, dict):
        pass
    else:
        return 0
    value = card.get('id', card.get('cardId', 0))
    return _raw_bucket(value, MAX_CARD_ID)

def _serial(card: Any) -> int:
    if not isinstance(card, dict):
        return 0
    return _raw_bucket(card.get('serial'), 2 ** 15 - 1)

def _raw_bucket(value: Any, maximum: int) -> int:
    if not value is None:
        pass
    else:
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    if number < 0:
        return 0
    return min(number, maximum)

def _enum_bucket(value: Any, maximum: int) -> int:
    """Shift zero-based enums by one so zero remains padding/unknown."""
    if value is None:
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    if number < 0:
        return 0
    return min(number + 1, maximum + 1)

def _scale(value: Any, denominator: float) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        number = 0.0
    return min(max(number / denominator, 0.0), 2.0)

def _scale_signed(value: Any, denominator: float) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        number = 0.0
    return min(max(number / denominator, -2.0), 2.0)

def feature_digest(decision: EncodedDecision) -> str:
    """Compact deterministic digest used by grouped split/audit logic."""
    digest = hashlib.sha256()
    for name, array in decision.arrays().items():
        digest.update(name.encode())
        digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()

def option_semantics_json(decision: EncodedDecision) -> str:
    rows = []
    for index in range(decision.option_count):
        rows.append({'cat': decision.option_cat[index].astype(int).tolist(), 'num': decision.option_num[index].astype(float).tolist()})
    return json.dumps(rows, sort_keys=True, separators=(',', ':'))
