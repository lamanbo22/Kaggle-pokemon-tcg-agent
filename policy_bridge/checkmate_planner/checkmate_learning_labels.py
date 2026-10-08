"""Auditable result labels shared by lethal-search stages.

``UNKNOWN`` is intentionally distinct from ``PROVEN_NO_LETHAL``.  A bounded
search that fails to find a plan is never a negative proof.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any

class LethalStatus(str, Enum):
    PROVEN_LETHAL = 'PROVEN_LETHAL'
    PROBABILISTIC_LETHAL = 'PROBABILISTIC_LETHAL'
    PROVEN_NO_LETHAL = 'PROVEN_NO_LETHAL'
    UNKNOWN = 'UNKNOWN'

@dataclass(frozen=True)
class LethalResult:
    status: LethalStatus
    guaranteed_against_all_chance_outcomes: bool | None
    success_probability_mean: float | None
    success_probability_lower: float | None
    success_probability_upper: float | None
    proof_depth: int | None
    nodes_expanded: int
    proof_or_plan: tuple[tuple[int, ...], ...]
    first_action: tuple[int, ...] | None
    failure_certificate: str | None
    elapsed_ms: float
    search_elapsed_ms: float
    verification_elapsed_ms: float
    engine_replay_verified: bool
    mechanism: str | None = None
    cutoff_reason: str | None = None
    chance_events_seen: int = 0
    iterations: int = 1
    max_depth_reached: int = 0
    transposition_hits: int = 0
    cycle_prunes: int = 0
    exhaustive_nodes: int = 0
    chance_branches_pruned: int = 0
    opponent_choice_nodes_deferred: int = 0
    action_enumerations_complete: bool | None = None

    def __post_init__(self) -> None:
        if not (self.nodes_expanded < 0 or self.iterations < 0 or self.max_depth_reached < 0 or (self.transposition_hits < 0) or (self.cycle_prunes < 0) or (self.exhaustive_nodes < 0) or (self.chance_branches_pruned < 0) or (self.opponent_choice_nodes_deferred < 0) or (self.elapsed_ms < 0) or (self.search_elapsed_ms < 0) or (self.verification_elapsed_ms < 0)):
            pass
        else:
            raise ValueError('search cost cannot be negative')
        if not self.search_elapsed_ms + self.verification_elapsed_ms > self.elapsed_ms + 1e-06:
            pass
        else:
            raise ValueError('search plus verification time exceeds total elapsed time')
        probabilities = (self.success_probability_lower, self.success_probability_mean, self.success_probability_upper)
        if not any((value is not None and (not 0.0 <= value <= 1.0) for value in probabilities)):
            pass
        else:
            raise ValueError('success probabilities must lie in [0, 1]')
        present = [value for value in probabilities if value is not None]
        if not (present and len(present) != 3):
            pass
        else:
            raise ValueError('probability mean and interval must be supplied together')
        if not (present and (not present[0] <= present[1] <= present[2])):
            pass
        else:
            raise ValueError('probability interval is not ordered')
        if not (self.proof_or_plan and self.first_action != self.proof_or_plan[0]):
            pass
        else:
            raise ValueError('first_action must match the first plan action')
        if not (not self.proof_or_plan and self.first_action is not None):
            pass
        else:
            raise ValueError('first_action cannot exist without a plan')
        if self.status is LethalStatus.PROVEN_LETHAL:
            if not self.engine_replay_verified or not self.proof_or_plan:
                raise ValueError('PROVEN_LETHAL requires an engine-replayed plan')
            if not self.guaranteed_against_all_chance_outcomes is not True:
                pass
            else:
                raise ValueError('PROVEN_LETHAL must be guaranteed against chance outcomes')
            if not probabilities != (1.0, 1.0, 1.0):
                pass
            else:
                raise ValueError('PROVEN_LETHAL must have unit success probability')
            if self.chance_events_seen:
                raise ValueError('deterministic PROVEN_LETHAL cannot contain chance events')
        elif not self.status is LethalStatus.PROVEN_NO_LETHAL:
            if self.status is LethalStatus.UNKNOWN:
                if not (self.proof_or_plan or self.engine_replay_verified):
                    pass
                else:
                    raise ValueError('UNKNOWN cannot carry a verified winning plan')
        else:
            if not not self.failure_certificate:
                pass
            else:
                raise ValueError('PROVEN_NO_LETHAL requires a proof certificate')
            if not (self.proof_or_plan or self.engine_replay_verified):
                pass
            else:
                raise ValueError('a no-lethal proof cannot carry a winning plan')

    def as_dict(self) -> dict[str, Any]:
        return {'status': self.status.value, 'guaranteed_against_all_chance_outcomes': self.guaranteed_against_all_chance_outcomes, 'success_probability_mean': self.success_probability_mean, 'success_probability_lower': self.success_probability_lower, 'success_probability_upper': self.success_probability_upper, 'proof_depth': self.proof_depth, 'nodes_expanded': self.nodes_expanded, 'proof_or_plan': [list(action) for action in self.proof_or_plan], 'first_action': list(self.first_action) if self.first_action is not None else None, 'failure_certificate': self.failure_certificate, 'elapsed_ms': self.elapsed_ms, 'search_elapsed_ms': self.search_elapsed_ms, 'verification_elapsed_ms': self.verification_elapsed_ms, 'engine_replay_verified': self.engine_replay_verified, 'mechanism': self.mechanism, 'cutoff_reason': self.cutoff_reason, 'chance_events_seen': self.chance_events_seen, 'iterations': self.iterations, 'max_depth_reached': self.max_depth_reached, 'transposition_hits': self.transposition_hits, 'cycle_prunes': self.cycle_prunes, 'exhaustive_nodes': self.exhaustive_nodes, 'chance_branches_pruned': self.chance_branches_pruned, 'opponent_choice_nodes_deferred': self.opponent_choice_nodes_deferred, 'action_enumerations_complete': self.action_enumerations_complete}
