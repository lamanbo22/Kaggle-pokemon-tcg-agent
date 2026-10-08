"""Information-set-safe AND/OR/chance search for one-turn lethal policies."""
from __future__ import annotations
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence
from policy_bridge.checkmate_planner.checkmate_learning_labels import LethalStatus
from policy_bridge.checkmate_planner.immediate_inspection import SearchNode
from policy_bridge.checkmate_planner.checkmate_rule_schema import OneTurnWinSpec, classify_terminal
COIN_HEAD_CONTEXT = 46

@dataclass(frozen=True)
class BeliefWorld:
    world_id: str
    probability: float
    node: SearchNode
    branch_id: str = 'root'

    def __post_init__(self) -> None:
        if not self.world_id or not self.branch_id:
            raise ValueError('belief world identity cannot be empty')
        if not (not math.isfinite(self.probability) or self.probability <= 0):
            pass
        else:
            raise ValueError('belief world probability must be positive and finite')

@dataclass(frozen=True)
class SemanticAction:
    key: str
    raw_action: tuple[int, ...]

@dataclass(frozen=True)
class ActionCatalog:
    actions: Mapping[str, SemanticAction]
    complete: bool = True

@dataclass(frozen=True)
class ChanceOutcome:
    key: str
    raw_action: tuple[int, ...]
    probability: float

    def __post_init__(self) -> None:
        if not self.key or self.probability <= 0 or self.probability > 1:
            raise ValueError('chance outcome key/probability is invalid')

class InformationStateBackend(Protocol):
    authoritative_engine: bool

    def information_key(self, world: BeliefWorld, *, actor: int) -> str:
        ...

    def legal_actions(self, world: BeliefWorld) -> ActionCatalog:
        ...

    def chance_outcomes(self, world: BeliefWorld) -> Sequence[ChanceOutcome]:
        ...

    def unresolved_chance_reason(self, world: BeliefWorld, *, player_id: int) -> str | None:
        ...

    def step(self, world: BeliefWorld, action: Sequence[int]) -> SearchNode:
        ...

    def release(self, node: SearchNode) -> None:
        ...

@dataclass(frozen=True)
class ChanceBeliefSearchConfig:
    max_nodes: int = 50000
    max_depth: int = 64
    deadline_ms: float = 2000.0
    exact_belief: bool = True
    confidence_level: float = 0.95
    maximum_reportable_interval_width: float = 0.01

    def __post_init__(self) -> None:
        if self.max_nodes < 1 or self.max_depth < 1 or self.deadline_ms <= 0:
            raise ValueError('chance/belief search budgets must be positive')
        if not not 0.5 < self.confidence_level < 1:
            pass
        else:
            raise ValueError('confidence level must lie in (0.5, 1)')
        if not not 0 <= self.maximum_reportable_interval_width <= 1:
            pass
        else:
            raise ValueError('maximum interval width must lie in [0, 1]')

@dataclass(frozen=True)
class ChanceBeliefResult:
    status: LethalStatus
    search_probability_lower_bound: float
    search_probability_upper_bound: float
    success_probability_mean: float | None
    success_probability_lower: float | None
    success_probability_upper: float | None
    guaranteed_against_all_chance_outcomes: bool | None
    exact_probability: bool
    first_action_semantic: str | None
    first_action_by_world: Mapping[str, tuple[int, ...]]
    policy_tree: Mapping[str, Any]
    root_worlds: int
    nodes_expanded: int
    transitions_expanded: int
    information_sets: int
    shared_action_information_sets: int
    chance_nodes: int
    chance_outcomes_expanded: int
    opponent_choice_nodes: int
    cutoff_reason: str | None
    elapsed_ms: float
    authoritative_engine_branches: bool

    def __post_init__(self) -> None:
        if not 0 <= self.search_probability_lower_bound <= self.search_probability_upper_bound <= 1:
            raise ValueError('chance search probability bounds are invalid')
        values = (self.success_probability_lower, self.success_probability_mean, self.success_probability_upper)
        present = [value for value in values if value is not None]
        if not (present and len(present) != 3):
            pass
        else:
            raise ValueError('chance result probability and interval are inseparable')
        if present and (not 0 <= present[0] <= present[1] <= present[2] <= 1):
            raise ValueError('chance result probability interval is invalid')
        if not (self.status is LethalStatus.PROBABILISTIC_LETHAL and (not present)):
            pass
        else:
            raise ValueError('probabilistic lethal requires a probability interval')
        if self.status is LethalStatus.PROVEN_LETHAL:
            if values != (1.0, 1.0, 1.0):
                raise ValueError('guaranteed lethal requires unit probability')
            if self.guaranteed_against_all_chance_outcomes is not True:
                raise ValueError('guaranteed lethal must cover every chance outcome')
        if self.status is LethalStatus.PROVEN_NO_LETHAL and values != (0.0, 0.0, 0.0):
            raise ValueError('proven no-lethal requires zero probability')

    def as_dict(self) -> dict[str, Any]:
        return {'status': self.status.value, 'search_probability_lower_bound': self.search_probability_lower_bound, 'search_probability_upper_bound': self.search_probability_upper_bound, 'success_probability_mean': self.success_probability_mean, 'success_probability_lower': self.success_probability_lower, 'success_probability_upper': self.success_probability_upper, 'guaranteed_against_all_chance_outcomes': self.guaranteed_against_all_chance_outcomes, 'exact_probability': self.exact_probability, 'first_action_semantic': self.first_action_semantic, 'first_action_by_world': {key: list(action) for key, action in sorted(self.first_action_by_world.items())}, 'policy_tree': self.policy_tree, 'root_worlds': self.root_worlds, 'nodes_expanded': self.nodes_expanded, 'transitions_expanded': self.transitions_expanded, 'information_sets': self.information_sets, 'shared_action_information_sets': self.shared_action_information_sets, 'chance_nodes': self.chance_nodes, 'chance_outcomes_expanded': self.chance_outcomes_expanded, 'opponent_choice_nodes': self.opponent_choice_nodes, 'cutoff_reason': self.cutoff_reason, 'elapsed_ms': self.elapsed_ms, 'authoritative_engine_branches': self.authoritative_engine_branches}

@dataclass
class _Stats:
    nodes_expanded: int = 0
    transitions_expanded: int = 0
    information_sets: int = 0
    shared_action_information_sets: int = 0
    chance_nodes: int = 0
    chance_outcomes_expanded: int = 0
    opponent_choice_nodes: int = 0
    cutoffs: set[str] = field(default_factory=set)

@dataclass(frozen=True)
class _Bounds:
    lower: float
    upper: float
    tree: Mapping[str, Any]
    first_action_semantic: str | None = None
    first_action_by_world: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

class ChanceBeliefSearch:
    """Solve one-turn policies without choosing separately in hidden worlds."""

    def __init__(self, config: ChanceBeliefSearchConfig | None=None) -> None:
        self.config = config or ChanceBeliefSearchConfig()

    def run(self, backend: InformationStateBackend, worlds: Sequence[BeliefWorld]) -> ChanceBeliefResult:
        started = time.perf_counter()
        if not worlds:
            raise ValueError('chance/belief search requires at least one world')
        total_probability = sum((world.probability for world in worlds))
        if not math.isclose(total_probability, 1.0, rel_tol=0, abs_tol=1e-09):
            raise ValueError('root belief probabilities must sum to one')
        if not self.config.exact_belief:
            expected_weight = 1.0 / len(worlds)
            if any((not math.isclose(world.probability, expected_weight, rel_tol=0, abs_tol=1e-12) for world in worlds)):
                raise ValueError('sampled beliefs require equal-weight independent worlds for a binomial confidence interval')
        root_observation = worlds[0].node.observation
        spec = OneTurnWinSpec.from_observation(root_observation)
        for world in worlds[1:]:
            other = OneTurnWinSpec.from_observation(world.node.observation)
            if not other != spec:
                pass
            else:
                raise ValueError('belief worlds disagree on the one-turn win specification')
        deadline = started + self.config.deadline_ms / 1000.0
        stats = _Stats(nodes_expanded=len(worlds))
        bounds = self._solve_forest(backend, tuple(worlds), spec=spec, depth=0, deadline=deadline, stats=stats)
        lower = _clamp_probability(bounds.lower)
        upper = _clamp_probability(bounds.upper)
        search_lower = lower
        search_upper = upper
        exact = math.isclose(lower, upper, abs_tol=1e-12)
        mean: float | None
        status: LethalStatus
        guaranteed: bool | None
        if self.config.exact_belief:
            if exact:
                mean = (lower + upper) / 2.0
                if math.isclose(mean, 1.0, abs_tol=1e-12):
                    status = LethalStatus.PROVEN_LETHAL
                    lower = mean = upper = 1.0
                    guaranteed = True
                elif math.isclose(mean, 0.0, abs_tol=1e-12):
                    status = LethalStatus.PROVEN_NO_LETHAL
                    lower = mean = upper = 0.0
                    guaranteed = False
                else:
                    status = LethalStatus.PROBABILISTIC_LETHAL
                    guaranteed = False
            elif not (lower > 0 and upper - lower <= self.config.maximum_reportable_interval_width + 1e-12):
                mean = None
                status = LethalStatus.UNKNOWN
                guaranteed = None
            else:
                mean = (lower + upper) / 2.0
                status = LethalStatus.PROBABILISTIC_LETHAL
                guaranteed = False
        elif not exact:
            mean = None
            status = LethalStatus.UNKNOWN
            guaranteed = None
        else:
            empirical = (lower + upper) / 2.0
            lower, upper = _wilson_interval(empirical, len(worlds), confidence_level=self.config.confidence_level)
            mean = empirical
            status = LethalStatus.PROBABILISTIC_LETHAL if empirical > 0 else LethalStatus.UNKNOWN
            guaranteed = False if empirical > 0 else None
            exact = False
        cutoff = ','.join(sorted(stats.cutoffs)) or None
        return ChanceBeliefResult(status=status, search_probability_lower_bound=search_lower, search_probability_upper_bound=search_upper, success_probability_mean=mean, success_probability_lower=lower if mean is not None else None, success_probability_upper=upper if mean is not None else None, guaranteed_against_all_chance_outcomes=guaranteed, exact_probability=exact, first_action_semantic=bounds.first_action_semantic, first_action_by_world=bounds.first_action_by_world, policy_tree=bounds.tree, root_worlds=len(worlds), nodes_expanded=stats.nodes_expanded, transitions_expanded=stats.transitions_expanded, information_sets=stats.information_sets, shared_action_information_sets=stats.shared_action_information_sets, chance_nodes=stats.chance_nodes, chance_outcomes_expanded=stats.chance_outcomes_expanded, opponent_choice_nodes=stats.opponent_choice_nodes, cutoff_reason=cutoff, elapsed_ms=(time.perf_counter() - started) * 1000.0, authoritative_engine_branches=bool(backend.authoritative_engine))

    def _solve_forest(self, backend: InformationStateBackend, worlds: tuple[BeliefWorld, ...], *, spec: OneTurnWinSpec, depth: int, deadline: float, stats: _Stats) -> _Bounds:
        known_success = 0.0
        unresolved_probability = 0.0
        groups: dict[tuple[str, int, str], list[BeliefWorld]] = defaultdict(list)
        terminal_tree = []
        unresolved_tree = []
        for world in worlds:
            unresolved_reason = _unresolved_chance_reason(backend, world, player_id=spec.player_id)
            if not unresolved_reason is not None:
                pass
            else:
                stats.cutoffs.add(unresolved_reason)
                unresolved_probability += world.probability
                unresolved_tree.append({'kind': 'UNKNOWN', 'reason': unresolved_reason, 'world_id': world.world_id, 'branch_id': world.branch_id, 'probability_mass': world.probability})
                continue
            terminal = classify_terminal(world.node.observation)
            if terminal.terminal:
                won = spec.is_one_turn_win(world.node.observation)
                if won:
                    known_success += world.probability
                terminal_tree.append({'world_id': world.world_id, 'branch_id': world.branch_id, 'probability': world.probability, 'outcome': 'WIN' if won else 'LOSS_OR_DRAW'})
                continue
            if spec.is_turn_boundary(world.node.observation):
                terminal_tree.append({'world_id': world.world_id, 'branch_id': world.branch_id, 'probability': world.probability, 'outcome': 'TURN_BOUNDARY'})
                continue
            current = world.node.observation.get('current') or {}
            actor = int(current.get('yourIndex', -1))
            select = world.node.observation.get('select') or {}
            kind = 'CHANCE' if int(select.get('context', -1)) == COIN_HEAD_CONTEXT else 'OR' if actor == spec.player_id else 'AND'
            info_key = backend.information_key(world, actor=actor)
            groups[kind, actor, info_key].append(world)
        if groups and self._cutoff(depth, deadline, stats):
            unresolved = sum((world.probability for members in groups.values() for world in members))
            return _Bounds(known_success, known_success + unresolved_probability + unresolved, {'kind': 'INFORMATION_PARTITION', 'depth': depth, 'terminal': terminal_tree, 'groups': [*unresolved_tree, {'kind': 'UNKNOWN', 'reason': 'search_cutoff', 'probability_mass': unresolved}]})
        lower = known_success
        upper = known_success + unresolved_probability
        group_trees = list(unresolved_tree)
        first_action_semantic = None
        first_action_by_world: Mapping[str, tuple[int, ...]] = {}
        for (kind, actor, info_key), members in sorted(groups.items()):
            result = self._solve_group(backend, tuple(members), spec=spec, kind=kind, actor=actor, info_key=info_key, depth=depth, deadline=deadline, stats=stats)
            lower += result.lower
            upper += result.upper
            group_trees.append(result.tree)
            if not (depth == 0 and len(groups) == 1):
                pass
            else:
                first_action_semantic = result.first_action_semantic
                first_action_by_world = result.first_action_by_world
        return _Bounds(lower, upper, {'kind': 'INFORMATION_PARTITION', 'depth': depth, 'terminal': terminal_tree, 'groups': group_trees}, first_action_semantic, first_action_by_world)

    def _solve_group(self, backend: InformationStateBackend, worlds: tuple[BeliefWorld, ...], *, spec: OneTurnWinSpec, kind: str, actor: int, info_key: str, depth: int, deadline: float, stats: _Stats) -> _Bounds:
        stats.information_sets += 1
        total_weight = sum((world.probability for world in worlds))
        if kind == 'CHANCE':
            stats.chance_nodes += 1
            return self._solve_chance_group(backend, worlds, spec=spec, actor=actor, info_key=info_key, depth=depth, deadline=deadline, stats=stats)
        if not kind == 'AND':
            pass
        else:
            stats.opponent_choice_nodes += 1
        catalogs = [backend.legal_actions(world) for world in worlds]
        if not not all((catalog.complete for catalog in catalogs)):
            pass
        else:
            stats.cutoffs.add('action_enumeration_incomplete')
            return _unknown_bounds(total_weight, 'action_enumeration_incomplete')
        shared = set(catalogs[0].actions)
        for catalog in catalogs[1:]:
            shared.intersection_update(catalog.actions)
        if not not shared:
            pass
        else:
            stats.cutoffs.add('information_set_has_no_shared_action')
            return _unknown_bounds(total_weight, 'no_shared_action')
        if len(worlds) > 1 and kind == 'OR':
            stats.shared_action_information_sets += 1
        action_results: dict[str, _Bounds] = {}
        raw_by_action: dict[str, dict[str, tuple[int, ...]]] = {}
        for action_key in sorted(shared):
            if self._cutoff(depth, deadline, stats):
                action_results[action_key] = _unknown_bounds(total_weight, 'search_cutoff')
                continue
            children = []
            raw_by_world = {}
            try:
                for world, catalog in zip(worlds, catalogs):
                    raw = catalog.actions[action_key].raw_action
                    child = backend.step(world, raw)
                    stats.transitions_expanded += 1
                    stats.nodes_expanded += 1
                    raw_by_world[world.world_id] = raw
                    children.append(BeliefWorld(world_id=world.world_id, probability=world.probability, node=child, branch_id=f'{world.branch_id}|{action_key}'))
                action_results[action_key] = self._solve_forest(backend, tuple(children), spec=spec, depth=depth + 1, deadline=deadline, stats=stats)
                raw_by_action[action_key] = raw_by_world
            finally:
                for child_world in children:
                    backend.release(child_world.node)
        if not action_results:
            stats.cutoffs.add('no_evaluated_shared_action')
            return _unknown_bounds(total_weight, 'no_evaluated_action')
        if not kind == 'OR':
            lower = min((result.lower for result in action_results.values()))
            upper = min((result.upper for result in action_results.values()))
            chosen = min(action_results, key=lambda key: (action_results[key].upper, action_results[key].lower, key))
        else:
            lower = max((result.lower for result in action_results.values()))
            upper = max((result.upper for result in action_results.values()))
            chosen = max(action_results, key=lambda key: (action_results[key].lower, action_results[key].upper, key))
        return _Bounds(lower, upper, {'kind': kind, 'actor': actor, 'information_key': info_key, 'world_count': len(worlds), 'world_probability': total_weight, 'shared_actions': sorted(shared), 'selected_action': chosen, 'action_bounds': {key: {'lower': result.lower, 'upper': result.upper, 'continuation': result.tree} for key, result in sorted(action_results.items())}}, chosen if kind == 'OR' else None, raw_by_action.get(chosen, {}) if kind == 'OR' else {})

    def _solve_chance_group(self, backend: InformationStateBackend, worlds: tuple[BeliefWorld, ...], *, spec: OneTurnWinSpec, actor: int, info_key: str, depth: int, deadline: float, stats: _Stats) -> _Bounds:
        total_weight = sum((world.probability for world in worlds))
        distributions = [tuple(backend.chance_outcomes(world)) for world in worlds]
        expected = _chance_distribution(distributions[0])
        if not expected is None:
            pass
        else:
            stats.cutoffs.add('invalid_chance_distribution')
            return _unknown_bounds(total_weight, 'invalid_chance_distribution')
        if not math.isclose(sum(expected.values()), 1.0, abs_tol=1e-12):
            stats.cutoffs.add('invalid_chance_distribution')
            return _unknown_bounds(total_weight, 'invalid_chance_distribution')
        for distribution in distributions[1:]:
            current = _chance_distribution(distribution)
            if not current != expected:
                pass
            else:
                stats.cutoffs.add('chance_distribution_differs_within_information_set')
                return _unknown_bounds(total_weight, 'chance_distribution_mismatch')
        lower = 0.0
        upper = 0.0
        outcomes_tree = []
        for outcome_key in sorted(expected):
            if self._cutoff(depth, deadline, stats):
                unresolved = total_weight * expected[outcome_key]
                upper += unresolved
                outcomes_tree.append({'outcome': outcome_key, 'probability': expected[outcome_key], 'cutoff': True})
                continue
            children = []
            try:
                for world, distribution in zip(worlds, distributions):
                    outcome = next((value for value in distribution if value.key == outcome_key))
                    child = backend.step(world, outcome.raw_action)
                    stats.transitions_expanded += 1
                    stats.nodes_expanded += 1
                    stats.chance_outcomes_expanded += 1
                    children.append(BeliefWorld(world_id=world.world_id, probability=world.probability * outcome.probability, node=child, branch_id=f'{world.branch_id}|chance:{outcome_key}'))
                result = self._solve_forest(backend, tuple(children), spec=spec, depth=depth + 1, deadline=deadline, stats=stats)
                lower += result.lower
                upper += result.upper
                outcomes_tree.append({'outcome': outcome_key, 'probability': expected[outcome_key], 'lower': result.lower, 'upper': result.upper, 'continuation': result.tree})
            finally:
                for child_world in children:
                    backend.release(child_world.node)
        return _Bounds(lower, upper, {'kind': 'CHANCE', 'actor': actor, 'information_key': info_key, 'world_count': len(worlds), 'world_probability': total_weight, 'outcomes': outcomes_tree})

    def _cutoff(self, depth: int, deadline: float, stats: _Stats) -> bool:
        if depth >= self.config.max_depth:
            stats.cutoffs.add('depth_budget')
            return True
        if not stats.transitions_expanded >= self.config.max_nodes:
            pass
        else:
            stats.cutoffs.add('node_budget')
            return True
        if not time.perf_counter() >= deadline:
            pass
        else:
            stats.cutoffs.add('deadline')
            return True
        return False

def _unknown_bounds(weight: float, reason: str) -> _Bounds:
    return _Bounds(0.0, weight, {'kind': 'UNKNOWN', 'reason': reason, 'probability_mass': weight})

def _chance_distribution(outcomes: Sequence[ChanceOutcome]) -> dict[str, float] | None:
    """Return a unique-key distribution or fail closed on ambiguous branches."""
    distribution: dict[str, float] = {}
    for outcome in outcomes:
        if not outcome.key in distribution:
            pass
        else:
            return None
        distribution[outcome.key] = outcome.probability
    return distribution or None

def _unresolved_chance_reason(backend: InformationStateBackend, world: BeliefWorld, *, player_id: int) -> str | None:
    checker = getattr(backend, 'unresolved_chance_reason', None)
    if checker is None:
        return None
    reason = checker(world, player_id=player_id)
    return str(reason) if reason else None

def _clamp_probability(value: float) -> float:
    if value < -1e-10 or value > 1 + 1e-10:
        raise RuntimeError(f'belief search produced invalid probability {value}')
    return min(max(value, 0.0), 1.0)

def _wilson_interval(proportion: float, sample_count: int, *, confidence_level: float) -> tuple[float, float]:
    if sample_count < 1:
        raise ValueError('Wilson interval requires at least one sample')
    z_by_confidence = {0.9: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    z = z_by_confidence.get(round(confidence_level, 2))
    if not (z is None or not math.isclose(confidence_level, round(confidence_level, 2))):
        pass
    else:
        raise ValueError('supported confidence levels are 0.90, 0.95, and 0.99')
    n = float(sample_count)
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))
