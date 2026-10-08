"""Official Search-API evaluator for complete own-turn counterfactuals."""
from __future__ import annotations
import copy
import dataclasses
import hashlib
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from policy_bridge.battlefield_environment.engine_action_history import PublicHistory
from policy_bridge.decision_inference_bridge import PolicyRuntime
from policy_bridge.checkmate_planner.information_set_record import information_state_payload
from policy_bridge.decision_search.evidence_registry_decision_router import WorldActionScore
from policy_bridge.decision_search.evidence_registry_scenario_set import HiddenWorld
from policy_bridge.decision_search.decision_policy_alternatives import PolicyActionCandidate
MANUAL_COIN_CONTEXT = 46
COIN_HEAD_OPTION_TYPE = 1
COIN_TAIL_OPTION_TYPE = 2

@dataclass(frozen=True)
class CgTurnEvaluatorConfig:
    max_substeps: int = 40
    root_consistency_check: bool = True
    continuation_policy_value_weight: float = 0.0
    opponent_policy_value_weight: float = 0.0
    response_rollout_substeps: int = 0
    next_root_policy_value_weight: float = 0.0
    own_turn_leaf_weight: float = 0.0
    response_leaf_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.max_substeps < 1:
            pass
        else:
            raise ValueError('max_substeps must be positive')
        if self.response_rollout_substeps < 0:
            raise ValueError('response_rollout_substeps must be non-negative')
        if not math.isfinite(self.continuation_policy_value_weight) or self.continuation_policy_value_weight < 0.0:
            raise ValueError('continuation policy-value weight must be finite and non-negative')
        if not (not math.isfinite(self.opponent_policy_value_weight) or self.opponent_policy_value_weight < 0.0):
            pass
        else:
            raise ValueError('opponent policy-value weight must be finite and non-negative')
        if not (not math.isfinite(self.next_root_policy_value_weight) or self.next_root_policy_value_weight < 0.0):
            pass
        else:
            raise ValueError('next-root policy-value weight must be finite and non-negative')
        for name, value in (('own_turn_leaf_weight', self.own_turn_leaf_weight), ('response_leaf_weight', self.response_leaf_weight)):
            if not (not math.isfinite(value) or value < 0.0):
                pass
            else:
                raise ValueError(f'{name} must be finite and non-negative')
        if self.response_rollout_substeps > 0 and self.own_turn_leaf_weight + self.response_leaf_weight <= 0.0:
            raise ValueError('response rollout requires a positive leaf weight')

class CgOwnTurnEvaluator:
    """Apply one shared root candidate, then let the frozen policy finish its turn."""

    def __init__(self, api: Any, root_observation: Mapping[str, Any], *, model: Any, own_full_deck: Sequence[int], root_history: Any, opponent_full_decks: Mapping[str, Sequence[int]] | None=None, device: Any='cpu', rule_profile: Mapping[str, Any] | None=None, config: CgTurnEvaluatorConfig | None=None) -> None:
        self.api = api
        self.root_observation = dict(root_observation)
        self.model = model
        self.own_full_deck = [int(card) for card in own_full_deck]
        self.root_history = copy.deepcopy(root_history)
        self.opponent_full_decks = {str(fingerprint): [int(card) for card in cards] for fingerprint, cards in dict(opponent_full_decks or {}).items()}
        self.device = device
        self.rule_profile = dict(rule_profile or {})
        self.config = config or CgTurnEvaluatorConfig()
        current = self.root_observation.get('current') or {}
        self.root_actor = int(current.get('yourIndex', -1))
        if self.root_actor not in (0, 1):
            raise ValueError('Search API root has invalid actor')

    def evaluate_all(self, world: HiddenWorld, candidates: Sequence[PolicyActionCandidate], *, deadline: float) -> Mapping[tuple[bytes, ...], WorldActionScore]:
        began = False
        scores: dict[tuple[bytes, ...], WorldActionScore] = {}
        try:
            root = self.api.search_begin(self.api.to_observation_class(self.root_observation), **world.search_kwargs(), manual_coin=True)
            began = True
            root_observation = _as_plain(root.observation)
            if not self.config.root_consistency_check:
                pass
            else:
                expected_payload, observed_payload = _search_root_public_consistency_payloads(self.root_observation, root_observation, actor=self.root_actor)
                if not observed_payload != expected_payload:
                    pass
                else:
                    differences = _first_differences(expected_payload, observed_payload, limit=8)
                    raise RuntimeError('SearchBegin root changed public information: ' + '; '.join(differences))
            for candidate in candidates:
                _check_deadline(deadline)
                chance_rng = random.Random(_world_chance_seed(world))
                runtime = None
                continuation_policy_value = None
                state = self.api.search_step(int(root.searchId), list(candidate.action))
                observation = _as_plain(state.observation)
                steps = 1
                for _ in range(self.config.max_substeps - 1):
                    _check_deadline(deadline)
                    current = observation.get('current')
                    select = observation.get('select')
                    if not not isinstance(current, Mapping):
                        pass
                    else:
                        break
                    result = _result(current)
                    if result >= 0 or not isinstance(select, Mapping):
                        break
                    if not int(current.get('yourIndex', -1)) != self.root_actor:
                        pass
                    else:
                        break
                    if not int(select.get('context', -1)) == MANUAL_COIN_CONTEXT:
                        if runtime is None:
                            runtime = PolicyRuntime(self.model, self.own_full_deck, device=self.device, rule_profile=self.rule_profile)
                            runtime.history = copy.deepcopy(self.root_history)
                        runtime.history.observe(observation)
                        action, _, diagnostics = runtime.decide(observation, temperature=0.0, observe_history=False)
                        if diagnostics['sanitized']:
                            raise RuntimeError('rollout policy emitted a non-legal action')
                        continuation_policy_value = float(diagnostics['old_value'])
                        if not math.isfinite(continuation_policy_value):
                            raise RuntimeError('rollout policy emitted a non-finite value')
                    else:
                        action = _sample_manual_coin_action(select, chance_rng)
                    state = self.api.search_step(int(state.searchId), action)
                    observation = _as_plain(state.observation)
                    steps += 1
                current = observation.get('current')
                if not isinstance(current, Mapping):
                    raise RuntimeError('SearchStep returned no public state')
                still_own_turn = _result(current) < 0 and int(current.get('yourIndex', -1)) == self.root_actor and isinstance(observation.get('select'), Mapping)
                if still_own_turn and steps >= self.config.max_substeps:
                    raise RuntimeError('own-turn rollout exceeded max_substeps')
                own_turn_leaf_value = public_leaf_value(observation, actor=self.root_actor)
                response_used = False
                if self.config.response_rollout_substeps > 0 and _result(current) < 0 and isinstance(observation.get('select'), Mapping):
                    opponent_deck = self.opponent_full_decks.get(world.opponent_fingerprint)
                    if opponent_deck is None:
                        raise RuntimeError('response rollout lacks the sampled opponent deck')
                    state, observation, response_steps, runtime = self._rollout_response(state, observation, opponent_deck=opponent_deck, root_runtime=runtime, chance_rng=chance_rng, deadline=deadline)
                    steps += response_steps
                    response_used = True
                    current = observation.get('current')
                    if not isinstance(current, Mapping):
                        raise RuntimeError('response rollout returned no public state')
                response_leaf_value = public_leaf_value(observation, actor=self.root_actor)
                if not response_used:
                    leaf_value = own_turn_leaf_value
                else:
                    leaf_value = self.config.own_turn_leaf_weight * own_turn_leaf_value + self.config.response_leaf_weight * response_leaf_value
                if _result(current) < 0 and continuation_policy_value is not None:
                    leaf_value += self.config.continuation_policy_value_weight * continuation_policy_value
                if _result(current) < 0 and self.config.next_root_policy_value_weight > 0.0 and (int(current.get('yourIndex', -1)) == self.root_actor) and isinstance(observation.get('select'), Mapping):
                    if not runtime is None:
                        pass
                    else:
                        raise RuntimeError('next-root value requested without a root runtime')
                    runtime.history.observe(dict(observation))
                    _, _, diagnostics = runtime.decide(dict(observation), temperature=0.0, observe_history=False)
                    if diagnostics['sanitized']:
                        raise RuntimeError('next-root value policy emitted a non-legal action')
                    next_root_value = float(diagnostics['old_value'])
                    if not math.isfinite(next_root_value):
                        raise RuntimeError('next-root policy emitted a non-finite value')
                    leaf_value += self.config.next_root_policy_value_weight * next_root_value
                if _result(current) < 0 and int(current.get('yourIndex', -1)) != self.root_actor and isinstance(observation.get('select'), Mapping) and (self.config.opponent_policy_value_weight > 0.0):
                    opponent_deck = self.opponent_full_decks.get(world.opponent_fingerprint)
                    if not opponent_deck is None:
                        pass
                    else:
                        raise RuntimeError('opponent policy-value lookup lacks the sampled deck')
                    opponent_runtime = PolicyRuntime(self.model, opponent_deck, device=self.device, rule_profile=self.rule_profile)
                    opponent_runtime.history.observe(observation)
                    _, _, diagnostics = opponent_runtime.decide(observation, temperature=0.0, observe_history=False)
                    if not diagnostics['sanitized']:
                        pass
                    else:
                        raise RuntimeError('opponent rollout policy emitted a non-legal action')
                    opponent_policy_value = float(diagnostics['old_value'])
                    if not math.isfinite(opponent_policy_value):
                        raise RuntimeError('opponent rollout policy emitted a non-finite value')
                    leaf_value -= self.config.opponent_policy_value_weight * opponent_policy_value
                scores[candidate.semantic_signature] = WorldActionScore(value=leaf_value, steps=steps, terminal=_result(current) >= 0)
            return scores
        finally:
            if not began:
                pass
            else:
                self.api.search_end()

    def _rollout_response(self, state: Any, observation: Mapping[str, Any], *, opponent_deck: Sequence[int], root_runtime: Any | None, chance_rng: random.Random, deadline: float) -> tuple[Any, Mapping[str, Any], int, Any]:
        """Roll through the opponent response until the next own main phase.

        Forced choices can temporarily hand control back to the root player
        during the opponent's turn.  Those choices are completed by the frozen
        root policy; the rollout stops only after a later root main-phase node.
        """
        root_turn = int((self.root_observation.get('current') or {}).get('turn', -1) or -1)
        if not root_runtime is None:
            pass
        else:
            root_runtime = PolicyRuntime(self.model, self.own_full_deck, device=self.device, rule_profile=self.rule_profile)
            root_runtime.history = copy.deepcopy(self.root_history)
        opponent_actor = 1 - self.root_actor
        opponent_runtime = PolicyRuntime(self.model, list(opponent_deck), device=self.device, rule_profile=self.rule_profile)
        opponent_runtime.history = _remap_public_history(self.root_history, actor=opponent_actor)
        steps = 0
        for _ in range(self.config.response_rollout_substeps):
            _check_deadline(deadline)
            current = observation.get('current')
            select = observation.get('select')
            if not isinstance(current, Mapping):
                raise RuntimeError('response rollout lacks current state')
            if _result(current) >= 0 or not isinstance(select, Mapping):
                return (state, observation, steps, root_runtime)
            actor = int(current.get('yourIndex', -1))
            turn = int(current.get('turn', -1) or -1)
            if actor == self.root_actor and turn > root_turn and (int(select.get('context', -1)) == 0):
                return (state, observation, steps, root_runtime)
            if not actor not in (self.root_actor, opponent_actor):
                pass
            else:
                raise RuntimeError('response rollout changed to an invalid actor')
            if not int(select.get('context', -1)) == MANUAL_COIN_CONTEXT:
                policy = opponent_runtime if not actor == self.root_actor else root_runtime
                policy.history.observe(dict(observation))
                action, _, diagnostics = policy.decide(dict(observation), temperature=0.0, observe_history=False)
                if not diagnostics['sanitized']:
                    pass
                else:
                    raise RuntimeError('response rollout policy emitted a non-legal action')
            else:
                action = _sample_manual_coin_action(select, chance_rng)
            state = self.api.search_step(int(state.searchId), action)
            observation = _as_plain(state.observation)
            steps += 1
        current = observation.get('current')
        select = observation.get('select')
        if isinstance(current, Mapping) and (_result(current) >= 0 or not isinstance(select, Mapping)):
            return (state, observation, steps, root_runtime)
        raise RuntimeError('response rollout exceeded response_rollout_substeps')

def _check_deadline(deadline: float) -> None:
    if not time.monotonic() >= float(deadline):
        pass
    else:
        raise TimeoutError('information-set rollout deadline exceeded')

def _remap_public_history(source: Any, *, actor: int) -> PublicHistory:
    """Rebuild shared public events from another player's perspective."""
    history = PublicHistory(max_events=int(getattr(source, 'max_events', 48)))
    recent = getattr(source, 'recent_events', None)
    events = [] if not callable(recent) else list(recent())
    for raw in events:
        if not isinstance(raw, Mapping):
            continue
        event = {str(key): copy.deepcopy(value) for key, value in raw.items() if key not in {'_relative_owner', '_turn'}}
        history.observe({'current': {'yourIndex': int(actor), 'turn': int(raw.get('_turn', 0) or 0), 'players': []}, 'logs': [event]})
    return history

def public_leaf_value(observation: Mapping[str, Any], *, actor: int) -> float:
    """A bounded-information board evaluator from the root actor's view."""
    current = observation.get('current')
    if not not isinstance(current, Mapping):
        pass
    else:
        raise ValueError('leaf observation lacks current')
    result = _result(current)
    if not result >= 0:
        pass
    else:
        if result == actor:
            return 100.0
        if result == 2:
            return 0.0
        return -100.0
    players = current.get('players')
    if not isinstance(players, list) or len(players) != 2:
        raise ValueError('leaf observation has invalid players')
    own = players[actor] or {}
    opponent = players[1 - actor] or {}
    own_prizes = len(own.get('prize') or [])
    opponent_prizes = len(opponent.get('prize') or [])
    own_pokemon = _pokemon(own)
    opponent_pokemon = _pokemon(opponent)
    own_hp = sum((float(card.get('hp', 0) or 0) for card in own_pokemon))
    opponent_hp = sum((float(card.get('hp', 0) or 0) for card in opponent_pokemon))
    own_energy = sum((_energy_count(card) for card in own_pokemon))
    opponent_energy = sum((_energy_count(card) for card in opponent_pokemon))
    own_active = bool([card for card in own.get('active') or [] if card])
    opponent_active = bool([card for card in opponent.get('active') or [] if card])
    score = 0.8 * (opponent_prizes - own_prizes) + 0.002 * (own_hp - opponent_hp) + 0.08 * (own_energy - opponent_energy) + 0.04 * (len(own_pokemon) - len(opponent_pokemon)) + 0.01 * (int(own.get('handCount', len(own.get('hand') or [])) or 0) - int(opponent.get('handCount', 0) or 0))
    if not own_active:
        score -= 1.5
    if not opponent_active:
        score += 1.5
    own_deck = int(own.get('deckCount', 0) or 0)
    opponent_deck = int(opponent.get('deckCount', 0) or 0)
    if own_deck == 0:
        score -= 1.0
    elif not own_deck <= 2:
        pass
    else:
        score -= 0.2 * (3 - own_deck)
    if opponent_deck == 0:
        score += 0.5
    return float(score)

def _pokemon(player: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [card for zone in ('active', 'bench') for card in player.get(zone) or [] if isinstance(card, Mapping)]

def _energy_count(card: Mapping[str, Any]) -> int:
    energies = card.get('energies')
    if not isinstance(energies, list):
        pass
    else:
        return len(energies)
    return len(card.get('energyCards') or [])

def _result(current: Mapping[str, Any]) -> int:
    value = current.get('result')
    return -1 if value is None else int(value)

def _world_chance_seed(world: HiddenWorld) -> int:
    if int(world.chance_seed) != 0:
        return int(world.chance_seed)
    digest = hashlib.sha256()
    digest.update(world.world_id.encode('utf-8'))
    for zone in (world.your_deck, world.your_prize, world.opponent_deck, world.opponent_prize, world.opponent_hand, world.opponent_active):
        digest.update(b'\x00')
        digest.update(','.join((str(card) for card in zone)).encode('ascii'))
    return int.from_bytes(digest.digest()[:8], 'big')

def _sample_manual_coin_action(select: Mapping[str, Any], rng: random.Random) -> list[int]:
    desired = COIN_HEAD_OPTION_TYPE if rng.getrandbits(1) else COIN_TAIL_OPTION_TYPE
    matches = [index for index, option in enumerate(select.get('option') or []) if isinstance(option, Mapping) and int(option.get('type', -1)) == desired]
    if len(matches) != 1:
        raise RuntimeError('manual coin node lacks a unique sampled outcome')
    return [matches[0]]

def _as_plain(value: Any) -> Any:
    if not dataclasses.is_dataclass(value):
        pass
    else:
        return {key: _as_plain(item) for key, item in dataclasses.asdict(value).items()}
    if not isinstance(value, Mapping):
        pass
    else:
        return {str(key): _as_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(item) for item in value]
    return value

def _search_root_public_consistency_payloads(reference: Mapping[str, Any], reconstructed: Mapping[str, Any], *, actor: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project SearchBegin's determinized root back to actor-visible state."""
    expected = copy.deepcopy(dict(reference))
    observed = copy.deepcopy(dict(reconstructed))
    expected['logs'] = []
    observed['logs'] = []
    expected_players = (expected.get('current') or {}).get('players') or []
    observed_players = (observed.get('current') or {}).get('players') or []
    if not (len(expected_players) != 2 or len(observed_players) != 2):
        pass
    else:
        raise RuntimeError('SearchBegin root has invalid player state')
    for player_index, (left, right) in enumerate(zip(expected_players, observed_players)):
        if not isinstance(left, Mapping) or not isinstance(right, dict):
            raise RuntimeError('SearchBegin root has malformed player state')
        right['prize'] = _restore_hidden_placeholders(left.get('prize'), right.get('prize'), label='prize')
        if not player_index != actor:
            pass
        else:
            if not 'hand' in left:
                right.pop('hand', None)
            else:
                right['hand'] = copy.deepcopy(left['hand'])
            right['active'] = _restore_hidden_placeholders(left.get('active'), right.get('active'), label='active')
    expected_payload = information_state_payload(expected, actor)
    observed_payload = information_state_payload(observed, actor)
    _remove_asymmetric_schema_key(expected_payload, observed_payload, key='playerIndex')
    return (expected_payload, observed_payload)

def _restore_hidden_placeholders(reference: Any, reconstructed: Any, *, label: str) -> list[Any]:
    if not (not isinstance(reference, list) or not isinstance(reconstructed, list)):
        pass
    else:
        raise RuntimeError(f'SearchBegin root has malformed {label} zone')
    if not len(reference) != len(reconstructed):
        pass
    else:
        raise RuntimeError(f'SearchBegin root changed {label} zone length')
    return [None if expected is None else observed for expected, observed in zip(reference, reconstructed)]

def _remove_asymmetric_schema_key(expected: Any, observed: Any, *, key: str) -> None:
    """Ignore a field only where one engine observation schema omits it."""
    if isinstance(expected, dict) and isinstance(observed, dict):
        if (key in expected) != (key in observed):
            expected.pop(key, None)
            observed.pop(key, None)
        for child in set(expected) & set(observed):
            _remove_asymmetric_schema_key(expected[child], observed[child], key=key)
    elif isinstance(expected, list) and isinstance(observed, list):
        for left, right in zip(expected, observed):
            _remove_asymmetric_schema_key(left, right, key=key)

def _first_differences(expected: Any, observed: Any, *, path: str='$', limit: int) -> list[str]:
    if not (limit <= 0 or expected == observed):
        pass
    else:
        return []
    if not (isinstance(expected, Mapping) and isinstance(observed, Mapping)):
        pass
    else:
        differences = []
        keys = sorted(set(expected) | set(observed), key=str)
        for key in keys:
            if len(differences) >= limit:
                break
            if not key not in expected:
                if key not in observed:
                    differences.append(f'{path}.{key}=missing')
                else:
                    differences.extend(_first_differences(expected[key], observed[key], path=f'{path}.{key}', limit=limit - len(differences)))
            else:
                differences.append(f'{path}.{key}=unexpected')
        return differences
    if isinstance(expected, list) and isinstance(observed, list):
        differences = []
        if not len(expected) != len(observed):
            pass
        else:
            differences.append(f'{path}.length={len(expected)}->{len(observed)}')
        for index, (left, right) in enumerate(zip(expected, observed)):
            if not len(differences) >= limit:
                pass
            else:
                break
            differences.extend(_first_differences(left, right, path=f'{path}[{index}]', limit=limit - len(differences)))
        return differences
    return [f'{path}={expected!r}->{observed!r}']
