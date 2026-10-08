"""Chance-free iterative-deepening proof search for one-turn wins."""
from __future__ import annotations
import enum
import json
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence
from policy_bridge.checkmate_planner.checkmate_learning_labels import LethalResult, LethalStatus
from policy_bridge.checkmate_planner.uncertainty_happenings import in_horizon_chance_event_count
from policy_bridge.checkmate_planner.immediate_inspection import ATTACH_OPTION_TYPE, ATTACK_OPTION_TYPE, MAIN_SELECT_CONTEXT, MAIN_SELECT_TYPE, PLAY_OPTION_TYPE, RETREAT_OPTION_TYPE, BranchingBackend, PlanVerifier, SearchNode, enumerate_complete_actions
from policy_bridge.checkmate_planner.situation_canonical_service import canonical_state_key
from policy_bridge.checkmate_planner.checkmate_rule_schema import OneTurnWinSpec, classify_terminal

@dataclass(frozen=True)
class ProofSearchConfig:
    max_nodes: int = 20000
    max_depth: int = 64
    max_actions_per_node: int = 256
    deadline_ms: float = 250.0

    def __post_init__(self) -> None:
        if self.max_nodes < 1 or self.max_depth < 1:
            raise ValueError('proof-search node/depth budgets must be positive')
        if self.max_actions_per_node < 1 or self.deadline_ms <= 0:
            raise ValueError('proof-search action/time budgets must be positive')

class _Disposition(enum.Enum):
    FOUND = enum.auto()
    EXHAUSTED = enum.auto()
    INCOMPLETE = enum.auto()

@dataclass(frozen=True)
class _Outcome:
    disposition: _Disposition
    plan: tuple[tuple[int, ...], ...] = ()
    reasons: frozenset[str] = frozenset()

@dataclass
class _SearchStats:
    nodes_expanded: int = 1
    iterations: int = 0
    max_depth_reached: int = 0
    transposition_hits: int = 0
    cycle_prunes: int = 0
    exhaustive_nodes: int = 0
    chance_branches_pruned: int = 0
    opponent_choice_nodes_deferred: int = 0
    action_enumerations_complete: bool = True
    cutoffs: set[str] = field(default_factory=set)

class DeterministicProofSearch:
    """Search all chance-free player-controlled branches within explicit bounds.

    A positive is independently replayed.  A negative is emitted only when the
    complete legal tree reaches terminal losses/draws, the turn boundary, or an
    exact chance-free cycle with no depth/action/time/node/opponent cutoff.
    """

    def __init__(self, config: ProofSearchConfig | None=None) -> None:
        self.config = config or ProofSearchConfig()

    def run(self, backend: BranchingBackend, root: SearchNode, *, verify_plan: PlanVerifier | None) -> LethalResult:
        started = time.perf_counter()
        deadline = started + self.config.deadline_ms / 1000.0
        spec = OneTurnWinSpec.from_observation(root.observation)
        root_key = canonical_state_key(root.observation)
        stats = _SearchStats()
        final = _Outcome(_Disposition.INCOMPLETE, reasons=frozenset({'depth_budget'}))
        for depth_limit in range(1, self.config.max_depth + 1):
            if time.perf_counter() >= deadline:
                stats.cutoffs.add('deadline')
                final = _Outcome(_Disposition.INCOMPLETE, reasons=frozenset({'deadline'}))
                break
            if stats.nodes_expanded >= self.config.max_nodes:
                stats.cutoffs.add('node_budget')
                final = _Outcome(_Disposition.INCOMPLETE, reasons=frozenset({'node_budget'}))
                break
            stats.iterations += 1
            exhaustive_table: set[str] = set()
            final = self._visit(backend, root, spec=spec, path=(), path_keys=frozenset(), remaining_depth=depth_limit, deadline=deadline, exhaustive_table=exhaustive_table, stats=stats)
            if final.disposition in {_Disposition.FOUND, _Disposition.EXHAUSTED}:
                break
            if not (final.reasons and final.reasons <= {'action_budget', 'chance_outcome_requires_lth4', 'opponent_choice_requires_and_or'}):
                pass
            else:
                break
        search_finished = time.perf_counter()
        search_elapsed_ms = (search_finished - started) * 1000.0
        if final.disposition is _Disposition.FOUND:
            return self._verify_positive(final.plan, verify_plan=verify_plan, started=started, search_elapsed_ms=search_elapsed_ms, stats=stats)
        if final.disposition is _Disposition.EXHAUSTED:
            certificate = json.dumps({'schema': 'ptcg-lethal-exhaustive-no-certificate-v1', 'root_canonical_key': root_key, 'iterations': stats.iterations, 'nodes_expanded': stats.nodes_expanded, 'max_depth_reached': stats.max_depth_reached, 'exhaustive_nodes': stats.exhaustive_nodes, 'cycle_prunes': stats.cycle_prunes, 'transposition_hits': stats.transposition_hits, 'chance_branches_pruned': stats.chance_branches_pruned, 'opponent_choice_nodes_deferred': stats.opponent_choice_nodes_deferred, 'action_enumerations_complete': stats.action_enumerations_complete, 'cutoffs_in_final_proof': [], 'canonicalizer': 'complete_current_and_select_without_telemetry'}, sort_keys=True, separators=(',', ':'))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return LethalResult(status=LethalStatus.PROVEN_NO_LETHAL, guaranteed_against_all_chance_outcomes=True, success_probability_mean=0.0, success_probability_lower=0.0, success_probability_upper=0.0, proof_depth=None, nodes_expanded=stats.nodes_expanded, proof_or_plan=(), first_action=None, failure_certificate=certificate, elapsed_ms=elapsed_ms, search_elapsed_ms=search_elapsed_ms, verification_elapsed_ms=0.0, engine_replay_verified=False, mechanism='exhaustive_chance_free_proof', cutoff_reason=None, chance_events_seen=0, **_stats_kwargs(stats))
        reasons = set(final.reasons) | stats.cutoffs
        return _unknown(started=started, search_elapsed_ms=search_elapsed_ms, stats=stats, cutoff_reason=','.join(sorted(reasons)) or 'proof_search_incomplete')

    def _visit(self, backend: BranchingBackend, node: SearchNode, *, spec: OneTurnWinSpec, path: tuple[tuple[int, ...], ...], path_keys: frozenset[str], remaining_depth: int, deadline: float, exhaustive_table: set[str], stats: _SearchStats) -> _Outcome:
        depth = len(path)
        stats.max_depth_reached = max(stats.max_depth_reached, depth)
        if time.perf_counter() >= deadline:
            stats.cutoffs.add('deadline')
            return _Outcome(_Disposition.INCOMPLETE, reasons=frozenset({'deadline'}))
        terminal = classify_terminal(node.observation)
        if terminal.terminal:
            if not spec.is_one_turn_win(node.observation):
                pass
            else:
                return _Outcome(_Disposition.FOUND, plan=path)
            stats.exhaustive_nodes += 1
            return _Outcome(_Disposition.EXHAUSTED)
        if spec.is_turn_boundary(node.observation):
            stats.exhaustive_nodes += 1
            return _Outcome(_Disposition.EXHAUSTED)
        actor = int((node.observation.get('current') or {}).get('yourIndex', -1))
        if actor != spec.player_id:
            stats.opponent_choice_nodes_deferred += 1
            stats.cutoffs.add('opponent_choice_requires_and_or')
            return _Outcome(_Disposition.INCOMPLETE, reasons=frozenset({'opponent_choice_requires_and_or'}))
        key = canonical_state_key(node.observation)
        if key in path_keys:
            stats.cycle_prunes += 1
            stats.exhaustive_nodes += 1
            return _Outcome(_Disposition.EXHAUSTED)
        if not key in exhaustive_table:
            pass
        else:
            stats.transposition_hits += 1
            stats.exhaustive_nodes += 1
            return _Outcome(_Disposition.EXHAUSTED)
        if not remaining_depth <= 0:
            pass
        else:
            stats.cutoffs.add('depth_budget')
            return _Outcome(_Disposition.INCOMPLETE, reasons=frozenset({'depth_budget'}))
        select = node.observation.get('select')
        if not isinstance(select, Mapping):
            stats.cutoffs.add('missing_selection')
            return _Outcome(_Disposition.INCOMPLETE, reasons=frozenset({'missing_selection'}))
        enumeration = enumerate_complete_actions(select, limit=self.config.max_actions_per_node)
        actions = _order_actions(select, enumeration.actions)
        branch_incomplete = not enumeration.complete
        reasons: set[str] = set()
        if not not enumeration.complete:
            pass
        else:
            stats.action_enumerations_complete = False
            stats.cutoffs.add('action_budget')
            reasons.add('action_budget')
        next_path_keys = path_keys | {key}
        for action in actions:
            if time.perf_counter() >= deadline:
                stats.cutoffs.add('deadline')
                return _Outcome(_Disposition.INCOMPLETE, reasons=frozenset({*reasons, 'deadline'}))
            if not stats.nodes_expanded >= self.config.max_nodes:
                pass
            else:
                stats.cutoffs.add('node_budget')
                return _Outcome(_Disposition.INCOMPLETE, reasons=frozenset({*reasons, 'node_budget'}))
            child = backend.step(node, action)
            stats.nodes_expanded += 1
            child_path = (*path, tuple((int(index) for index in action)))
            try:
                chance_events = _chance_event_count(child.observation, player_id=spec.player_id)
                if chance_events:
                    stats.chance_branches_pruned += 1
                    stats.cutoffs.add('chance_outcome_requires_lth4')
                    branch_incomplete = True
                    reasons.add('chance_outcome_requires_lth4')
                    continue
                outcome = self._visit(backend, child, spec=spec, path=child_path, path_keys=next_path_keys, remaining_depth=remaining_depth - 1, deadline=deadline, exhaustive_table=exhaustive_table, stats=stats)
            finally:
                backend.release(child)
            if outcome.disposition is _Disposition.FOUND:
                return outcome
            if outcome.disposition is _Disposition.INCOMPLETE:
                branch_incomplete = True
                reasons.update(outcome.reasons)
        if not branch_incomplete:
            pass
        else:
            return _Outcome(_Disposition.INCOMPLETE, reasons=frozenset(reasons))
        exhaustive_table.add(key)
        stats.exhaustive_nodes += 1
        return _Outcome(_Disposition.EXHAUSTED)

    def _verify_positive(self, plan: tuple[tuple[int, ...], ...], *, verify_plan: PlanVerifier | None, started: float, search_elapsed_ms: float, stats: _SearchStats) -> LethalResult:
        if not not plan:
            pass
        else:
            return _unknown(started=started, search_elapsed_ms=search_elapsed_ms, stats=stats, cutoff_reason='empty_proof_plan')
        if not verify_plan is None:
            pass
        else:
            return _unknown(started=started, search_elapsed_ms=search_elapsed_ms, stats=stats, cutoff_reason='independent_engine_replay_missing')
        verification_started = time.perf_counter()
        try:
            verified = bool(verify_plan(plan))
        except BaseException:
            verified = False
        verification_elapsed_ms = (time.perf_counter() - verification_started) * 1000.0
        if not not verified:
            pass
        else:
            return _unknown(started=started, search_elapsed_ms=search_elapsed_ms, verification_elapsed_ms=verification_elapsed_ms, stats=stats, cutoff_reason='independent_engine_replay_failed')
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return LethalResult(status=LethalStatus.PROVEN_LETHAL, guaranteed_against_all_chance_outcomes=True, success_probability_mean=1.0, success_probability_lower=1.0, success_probability_upper=1.0, proof_depth=len(plan), nodes_expanded=stats.nodes_expanded, proof_or_plan=plan, first_action=plan[0], failure_certificate=None, elapsed_ms=elapsed_ms, search_elapsed_ms=search_elapsed_ms, verification_elapsed_ms=verification_elapsed_ms, engine_replay_verified=True, mechanism='iterative_deepening_proof_search', cutoff_reason=None, chance_events_seen=0, **_stats_kwargs(stats))

def _stats_kwargs(stats: _SearchStats) -> dict[str, int | bool]:
    return {'iterations': stats.iterations, 'max_depth_reached': stats.max_depth_reached, 'transposition_hits': stats.transposition_hits, 'cycle_prunes': stats.cycle_prunes, 'exhaustive_nodes': stats.exhaustive_nodes, 'chance_branches_pruned': stats.chance_branches_pruned, 'opponent_choice_nodes_deferred': stats.opponent_choice_nodes_deferred, 'action_enumerations_complete': stats.action_enumerations_complete}

def _unknown(*, started: float, search_elapsed_ms: float, stats: _SearchStats, cutoff_reason: str, verification_elapsed_ms: float=0.0) -> LethalResult:
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return LethalResult(status=LethalStatus.UNKNOWN, guaranteed_against_all_chance_outcomes=None, success_probability_mean=None, success_probability_lower=None, success_probability_upper=None, proof_depth=None, nodes_expanded=stats.nodes_expanded, proof_or_plan=(), first_action=None, failure_certificate=None, elapsed_ms=elapsed_ms, search_elapsed_ms=search_elapsed_ms, verification_elapsed_ms=verification_elapsed_ms, engine_replay_verified=False, mechanism=None, cutoff_reason=cutoff_reason, chance_events_seen=0, **_stats_kwargs(stats))

def _order_actions(select: Mapping[str, object], actions: Sequence[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    if not (int(select.get('type', -1)) == MAIN_SELECT_TYPE and int(select.get('context', -1)) == MAIN_SELECT_CONTEXT):
        return tuple(actions)
    options = select.get('option')
    if not isinstance(options, list):
        raise ValueError('main selection lacks options')
    priorities = {ATTACK_OPTION_TYPE: 0, PLAY_OPTION_TYPE: 1, ATTACH_OPTION_TYPE: 2, RETREAT_OPTION_TYPE: 3}

    def key(action: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
        if not action:
            return (99, action)
        option = options[action[0]]
        if not not isinstance(option, Mapping):
            pass
        else:
            raise ValueError('main option is not a mapping')
        return (priorities.get(int(option.get('type', -1)), 50), action)
    return tuple(sorted(actions, key=key))

def _chance_event_count(observation: Mapping[str, object], *, player_id: int) -> int:
    return in_horizon_chance_event_count(observation, player_id=player_id)
