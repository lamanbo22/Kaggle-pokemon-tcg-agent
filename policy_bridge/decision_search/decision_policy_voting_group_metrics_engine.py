"""Common-random-number committee over independent continuation policies."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from policy_bridge.decision_search.evidence_registry_decision_router import WorldActionScore
from policy_bridge.decision_search.evidence_registry_scenario_set import HiddenWorld
from policy_bridge.decision_search.decision_policy_alternatives import PolicyActionCandidate

class PolicyCommitteeTurnEvaluator:
    """Average complete-action scores from orthogonal rollout evaluators.

    Every evaluator receives the identical hidden world, root candidates,
    chance seed and shared deadline. Committee members only evaluate; they
    never add candidates or receive direct action authority.
    """

    def __init__(self, evaluators: Sequence[Any], *, aggregation: str='mean') -> None:
        self.evaluators = tuple(evaluators)
        if len(self.evaluators) < 2:
            raise ValueError('policy committee requires at least two evaluators')
        if aggregation not in {'min', 'mean', 'max'}:
            raise ValueError('unknown policy committee aggregation')
        self.aggregation = aggregation

    def evaluate_all(self, world: HiddenWorld, candidates: Sequence[PolicyActionCandidate], *, deadline: float) -> Mapping[tuple[bytes, ...], WorldActionScore]:
        panels = [evaluator.evaluate_all(world, candidates, deadline=deadline) for evaluator in self.evaluators]
        expected = {candidate.semantic_signature for candidate in candidates}
        if any((set(panel) != expected for panel in panels)):
            raise RuntimeError('policy committee evaluator returned incomplete actions')
        result = {}
        for signature in expected:
            rows = [panel[signature] for panel in panels]
            values = [float(row.value) for row in rows]
            if self.aggregation == 'min':
                value = min(values)
            elif self.aggregation == 'max':
                value = max(values)
            else:
                value = sum(values) / len(values)
            result[signature] = WorldActionScore(value=value, steps=sum((int(row.steps) for row in rows)), terminal=all((bool(row.terminal) for row in rows)))
        return result
