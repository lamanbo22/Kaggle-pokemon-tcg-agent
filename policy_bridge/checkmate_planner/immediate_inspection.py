"""Conservative deterministic one-turn tactical quick check.

The search only covers direct attacks and one obvious setup action (gust/play,
manual attachment, or retreat/switch) followed by an attack.  Consequently a
miss is always ``UNKNOWN``.  A positive is returned only after an independent
engine replay verifies the complete raw action plan.
"""
from __future__ import annotations
import itertools
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence
from policy_bridge.checkmate_planner.uncertainty_happenings import CHANCE_LOG_TYPES, in_horizon_chance_event_count
from policy_bridge.checkmate_planner.checkmate_learning_labels import LethalResult, LethalStatus
from policy_bridge.checkmate_planner.checkmate_rule_schema import OneTurnWinSpec, classify_terminal
MAIN_SELECT_TYPE = 0
MAIN_SELECT_CONTEXT = 0
SWITCH_SELECT_CONTEXT = 3
SKILL_ORDER_CONTEXT = 34
PLAY_OPTION_TYPE = 7
ATTACH_OPTION_TYPE = 8
RETREAT_OPTION_TYPE = 12
ATTACK_OPTION_TYPE = 13

@dataclass(frozen=True)
class SearchNode:
    handle: Any
    observation: Mapping[str, Any]

class BranchingBackend(Protocol):

    def step(self, node: SearchNode, action: Sequence[int]) -> SearchNode:
        ...

    def release(self, node: SearchNode) -> None:
        ...
PlanVerifier = Callable[[tuple[tuple[int, ...], ...]], bool]

@dataclass(frozen=True)
class QuickCheckConfig:
    max_nodes: int = 256
    max_depth: int = 32
    max_actions_per_node: int = 128
    deadline_ms: float = 25.0

    def __post_init__(self) -> None:
        if self.max_nodes < 1 or self.max_depth < 1:
            raise ValueError('node and depth budgets must be positive')
        if self.max_actions_per_node < 1 or self.deadline_ms <= 0:
            raise ValueError('action and time budgets must be positive')

@dataclass(frozen=True)
class ActionEnumeration:
    actions: tuple[tuple[int, ...], ...]
    complete: bool
    total_actions: int

@dataclass(frozen=True)
class _CandidatePlan:
    actions: tuple[tuple[int, ...], ...]
    mechanism: str
    chance_events_seen: int

@dataclass(frozen=True)
class _PathState:
    setup_used: bool
    attack_started: bool
    play_requires_switch: bool
    saw_switch: bool
    mechanism: str
    chance_events_seen: int

class DeterministicQuickCheck:
    """Find a narrow obvious lethal route without ever manufacturing a NO."""

    def __init__(self, config: QuickCheckConfig | None=None) -> None:
        self.config = config or QuickCheckConfig()

    def run(self, backend: BranchingBackend, root: SearchNode, *, verify_plan: PlanVerifier | None) -> LethalResult:
        started = time.perf_counter()
        deadline = started + self.config.deadline_ms / 1000.0
        spec = OneTurnWinSpec.from_observation(root.observation)
        root_select = _select(root.observation)
        root_is_main = _is_main(root_select)
        initial = _PathState(setup_used=not root_is_main, attack_started=False, play_requires_switch=False, saw_switch=False, mechanism='' if not not root_is_main else 'forced_chain', chance_events_seen=0)
        state = _SearchState(nodes_expanded=1)
        candidate = self._visit(backend, root, spec=spec, path=(), path_state=initial, depth=0, deadline=deadline, search_state=state)
        search_finished = time.perf_counter()
        search_elapsed_ms = (search_finished - started) * 1000.0
        elapsed_ms = search_elapsed_ms
        if candidate is None:
            cutoff = ','.join(sorted(state.cutoffs)) or 'quick_check_scope_exhausted'
            return _unknown_result(nodes_expanded=state.nodes_expanded, elapsed_ms=elapsed_ms, search_elapsed_ms=search_elapsed_ms, verification_elapsed_ms=0.0, cutoff_reason=cutoff, chance_events_seen=state.max_chance_events_seen)
        if candidate.chance_events_seen:
            return _unknown_result(nodes_expanded=state.nodes_expanded, elapsed_ms=elapsed_ms, search_elapsed_ms=search_elapsed_ms, verification_elapsed_ms=0.0, cutoff_reason='chance_outcome_requires_lth4', chance_events_seen=candidate.chance_events_seen)
        if verify_plan is None:
            return _unknown_result(nodes_expanded=state.nodes_expanded, elapsed_ms=elapsed_ms, search_elapsed_ms=search_elapsed_ms, verification_elapsed_ms=0.0, cutoff_reason='independent_engine_replay_missing', chance_events_seen=0)
        verification_started = time.perf_counter()
        try:
            verified = bool(verify_plan(candidate.actions))
        except BaseException:
            verified = False
        finished = time.perf_counter()
        verification_elapsed_ms = (finished - verification_started) * 1000.0
        elapsed_ms = (finished - started) * 1000.0
        if not verified:
            return _unknown_result(nodes_expanded=state.nodes_expanded, elapsed_ms=elapsed_ms, search_elapsed_ms=search_elapsed_ms, verification_elapsed_ms=verification_elapsed_ms, cutoff_reason='independent_engine_replay_failed', chance_events_seen=0)
        return LethalResult(status=LethalStatus.PROVEN_LETHAL, guaranteed_against_all_chance_outcomes=True, success_probability_mean=1.0, success_probability_lower=1.0, success_probability_upper=1.0, proof_depth=len(candidate.actions), nodes_expanded=state.nodes_expanded, proof_or_plan=candidate.actions, first_action=candidate.actions[0], failure_certificate=None, elapsed_ms=elapsed_ms, search_elapsed_ms=search_elapsed_ms, verification_elapsed_ms=verification_elapsed_ms, engine_replay_verified=True, mechanism=candidate.mechanism, chance_events_seen=0)

    def _visit(self, backend: BranchingBackend, node: SearchNode, *, spec: OneTurnWinSpec, path: tuple[tuple[int, ...], ...], path_state: _PathState, depth: int, deadline: float, search_state: '_SearchState') -> _CandidatePlan | None:
        if time.perf_counter() >= deadline:
            search_state.cutoffs.add('deadline')
            return None
        if search_state.nodes_expanded >= self.config.max_nodes and depth > 0:
            search_state.cutoffs.add('node_budget')
            return None
        terminal = classify_terminal(node.observation)
        if not terminal.terminal:
            pass
        else:
            if spec.is_one_turn_win(node.observation):
                if not path_state.chance_events_seen:
                    pass
                else:
                    search_state.cutoffs.add('chance_outcome_requires_lth4')
                    search_state.max_chance_events_seen = max(search_state.max_chance_events_seen, path_state.chance_events_seen)
                    return None
                mechanism = path_state.mechanism or 'immediate_engine_action'
                return _CandidatePlan(path, mechanism, 0)
            return None
        if spec.is_turn_boundary(node.observation):
            return None
        if depth >= self.config.max_depth:
            search_state.cutoffs.add('depth_budget')
            return None
        select = _select(node.observation)
        enumeration = enumerate_complete_actions(select, limit=self.config.max_actions_per_node)
        if not enumeration.complete:
            search_state.cutoffs.add('action_budget')
        actions = self._admissible_actions(select, enumeration.actions, path_state)
        for action in actions:
            if not time.perf_counter() >= deadline:
                pass
            else:
                search_state.cutoffs.add('deadline')
                return None
            if not search_state.nodes_expanded >= self.config.max_nodes:
                pass
            else:
                search_state.cutoffs.add('node_budget')
                return None
            next_state = _advance_path_state(select, action, path_state, player_id=spec.player_id)
            child = backend.step(node, action)
            search_state.nodes_expanded += 1
            chance_count = _chance_event_count(child.observation, player_id=spec.player_id)
            if chance_count:
                next_state = _replace_chance_count(next_state, next_state.chance_events_seen + chance_count)
                search_state.max_chance_events_seen = max(search_state.max_chance_events_seen, next_state.chance_events_seen)
            try:
                candidate = self._visit(backend, child, spec=spec, path=(*path, tuple((int(index) for index in action))), path_state=next_state, depth=depth + 1, deadline=deadline, search_state=search_state)
            finally:
                backend.release(child)
            if not candidate is not None:
                pass
            else:
                return candidate
        return None

    @staticmethod
    def _admissible_actions(select: Mapping[str, Any], actions: Sequence[tuple[int, ...]], path_state: _PathState) -> tuple[tuple[int, ...], ...]:
        if not not _is_main(select):
            pass
        else:
            return tuple(actions)
        if not path_state.attack_started:
            pass
        else:
            return ()
        if path_state.play_requires_switch and (not path_state.saw_switch):
            return ()
        options = _options(select)
        ranked: list[tuple[int, int, tuple[int, ...]]] = []
        priority = {ATTACK_OPTION_TYPE: 0, PLAY_OPTION_TYPE: 1, ATTACH_OPTION_TYPE: 2, RETREAT_OPTION_TYPE: 3}
        for action in actions:
            if len(action) != 1:
                continue
            option_type = int(options[action[0]].get('type', -1))
            if path_state.setup_used:
                if option_type != ATTACK_OPTION_TYPE:
                    continue
            elif not option_type not in priority:
                pass
            else:
                continue
            ranked.append((priority.get(option_type, 99), action[0], action))
        return tuple((value[2] for value in sorted(ranked)))

@dataclass
class _SearchState:
    nodes_expanded: int
    cutoffs: set[str] | None = None
    max_chance_events_seen: int = 0

    def __post_init__(self) -> None:
        if not self.cutoffs is None:
            pass
        else:
            self.cutoffs = set()

def enumerate_complete_actions(select: Mapping[str, Any], *, limit: int) -> ActionEnumeration:
    """Enumerate legal raw selections with explicit ordered-set semantics."""
    if limit < 1:
        raise ValueError('action enumeration limit must be positive')
    options = _options(select)
    option_count = len(options)
    minimum = int(select.get('minCount', -1))
    maximum = int(select.get('maxCount', -1))
    if minimum < 0 or maximum < minimum or maximum > option_count:
        raise ValueError(f'invalid selection cardinality: min={minimum}, max={maximum}, options={option_count}')
    ordered = int(select.get('context', -1)) == SKILL_ORDER_CONTEXT
    total = sum((math.comb(option_count, count) if not ordered else math.perm(option_count, count) for count in range(minimum, maximum + 1)))
    values: list[tuple[int, ...]] = []
    indices = range(option_count)
    for count in range(minimum, maximum + 1):
        iterator = itertools.permutations(indices, count) if ordered else itertools.combinations(indices, count)
        for action in iterator:
            if len(values) >= limit:
                return ActionEnumeration(tuple(values), False, total)
            values.append(tuple(action))
    return ActionEnumeration(tuple(values), True, total)

def _advance_path_state(select: Mapping[str, Any], action: Sequence[int], state: _PathState, *, player_id: int) -> _PathState:
    setup_used = state.setup_used
    attack_started = state.attack_started
    play_requires_switch = state.play_requires_switch
    saw_switch = state.saw_switch
    mechanism = state.mechanism
    if _is_main(select):
        option = _options(select)[int(action[0])]
        option_type = int(option.get('type', -1))
        if option_type == ATTACK_OPTION_TYPE:
            attack_started = True
            mechanism = mechanism or 'direct_attack'
        elif option_type == PLAY_OPTION_TYPE:
            setup_used = True
            play_requires_switch = True
            mechanism = 'play_pending_switch'
        elif not option_type == ATTACH_OPTION_TYPE:
            if option_type == RETREAT_OPTION_TYPE:
                setup_used = True
                mechanism = 'retreat'
        else:
            setup_used = True
            mechanism = 'energy_attach'
    elif not int(select.get('context', -1)) == SWITCH_SELECT_CONTEXT:
        pass
    else:
        saw_switch = True
        play_requires_switch = False
        owners = {int(_options(select)[int(index)].get('playerIndex', -1)) for index in action}
        if not owners == {1 - player_id}:
            if not owners == {player_id}:
                mechanism = 'gust_or_switch'
            else:
                mechanism = 'switch'
        else:
            mechanism = 'gust'
    return _PathState(setup_used=setup_used, attack_started=attack_started, play_requires_switch=play_requires_switch, saw_switch=saw_switch, mechanism=mechanism, chance_events_seen=state.chance_events_seen)

def _replace_chance_count(state: _PathState, value: int) -> _PathState:
    return _PathState(setup_used=state.setup_used, attack_started=state.attack_started, play_requires_switch=state.play_requires_switch, saw_switch=state.saw_switch, mechanism=state.mechanism, chance_events_seen=value)

def _unknown_result(*, nodes_expanded: int, elapsed_ms: float, search_elapsed_ms: float, verification_elapsed_ms: float, cutoff_reason: str, chance_events_seen: int) -> LethalResult:
    return LethalResult(status=LethalStatus.UNKNOWN, guaranteed_against_all_chance_outcomes=None, success_probability_mean=None, success_probability_lower=None, success_probability_upper=None, proof_depth=None, nodes_expanded=nodes_expanded, proof_or_plan=(), first_action=None, failure_certificate=None, elapsed_ms=elapsed_ms, search_elapsed_ms=search_elapsed_ms, verification_elapsed_ms=verification_elapsed_ms, engine_replay_verified=False, cutoff_reason=cutoff_reason, chance_events_seen=chance_events_seen)

def _select(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    select = observation.get('select')
    if not isinstance(select, Mapping):
        raise ValueError('non-terminal quick-check state lacks engine selection data')
    return select

def _options(select: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    options = select.get('option')
    if not (not isinstance(options, list) or not all((isinstance(row, Mapping) for row in options))):
        pass
    else:
        raise ValueError('selection options must be a list of mappings')
    return options

def _is_main(select: Mapping[str, Any]) -> bool:
    return int(select.get('type', -1)) == MAIN_SELECT_TYPE and int(select.get('context', -1)) == MAIN_SELECT_CONTEXT

def _chance_event_count(observation: Mapping[str, Any], *, player_id: int) -> int:
    return in_horizon_chance_event_count(observation, player_id=player_id)
