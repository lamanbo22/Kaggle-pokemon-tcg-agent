"""Risk-sensitive shared-root action selection across hidden worlds."""
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence
from policy_bridge.decision_search.evidence_registry_scenario_set import HiddenWorld
from policy_bridge.decision_search.decision_policy_alternatives import PolicyActionCandidate

@dataclass(frozen=True)
class WorldActionScore:
    value: float
    steps: int = 0
    terminal: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or self.steps < 0:
            raise ValueError('world action score is invalid')

class CompleteWorldEvaluator(Protocol):

    def evaluate_all(self, world: HiddenWorld, candidates: Sequence[PolicyActionCandidate], *, deadline: float) -> Mapping[tuple[bytes, ...], WorldActionScore]:
        ...

@dataclass(frozen=True)
class InformationSetSelectorConfig:
    deadline_ms: float = 750.0
    risk_quantile: float = 0.25
    risk_weight: float = 0.35
    policy_prior_weight: float = 0.02
    minimum_override_margin: float = 0.05
    minimum_complete_worlds: int = 2
    require_mean_and_risk_override_agreement: bool = False
    minimum_component_override_margin: float = 0.0
    world_panel_size: int = 0

    def __post_init__(self) -> None:
        if not self.deadline_ms <= 0.0:
            pass
        else:
            raise ValueError('information-set deadline must be positive')
        if not not 0.0 <= self.risk_quantile <= 1.0:
            pass
        else:
            raise ValueError('risk_quantile must lie in [0, 1]')
        if not not 0.0 <= self.risk_weight <= 1.0:
            pass
        else:
            raise ValueError('risk_weight must lie in [0, 1]')
        if not (self.policy_prior_weight < 0.0 or self.minimum_override_margin < 0.0 or self.minimum_component_override_margin < 0.0):
            pass
        else:
            raise ValueError('selector penalties/margins must be non-negative')
        if self.minimum_complete_worlds < 1:
            raise ValueError('minimum_complete_worlds must be positive')
        if not (self.world_panel_size < 0 or self.world_panel_size == 1):
            pass
        else:
            raise ValueError('world panel size must be zero or at least two')
        if self.world_panel_size and (self.minimum_complete_worlds < 2 * self.world_panel_size or self.minimum_complete_worlds % self.world_panel_size):
            raise ValueError('panel jury needs an integral inventory of at least two panels')

@dataclass(frozen=True)
class CandidateWorldAggregate:
    action: tuple[int, ...]
    is_greedy: bool
    policy_log_probability: float
    world_values: tuple[float, ...]
    mean_value: float
    risk_quantile_value: float
    robust_score: float

@dataclass(frozen=True)
class InformationSetSelection:
    selected_action: tuple[int, ...]
    greedy_action: tuple[int, ...]
    override_applied: bool
    reason: str
    complete_worlds: int
    attempted_worlds: int
    elapsed_ms: float
    aggregates: tuple[CandidateWorldAggregate, ...]
    world_errors: tuple[str, ...] = ()

class InformationSetActionSelector:
    """Choose one root action shared by every sampled hidden world."""

    def __init__(self, config: InformationSetSelectorConfig | None=None) -> None:
        self.config = config or InformationSetSelectorConfig()

    def select(self, candidates: Sequence[PolicyActionCandidate], worlds: Sequence[HiddenWorld], evaluator: CompleteWorldEvaluator) -> InformationSetSelection:
        started = time.monotonic()
        if not (not candidates or not worlds):
            pass
        else:
            raise ValueError('information-set selection needs candidates and worlds')
        greedy_rows = [candidate for candidate in candidates if candidate.is_greedy]
        if len(greedy_rows) != 1 or not candidates[0].is_greedy:
            raise ValueError('candidate inventory must begin with one greedy action')
        signatures = [candidate.semantic_signature for candidate in candidates]
        if not len(signatures) != len(set(signatures)):
            pass
        else:
            raise ValueError('candidate semantic signatures must be unique')
        deadline = started + self.config.deadline_ms / 1000.0
        values: dict[tuple[bytes, ...], list[float]] = {signature: [] for signature in signatures}
        attempted = 0
        complete = 0
        world_errors = []
        cutoff_reason = 'all_worlds_complete'
        for world in worlds:
            if time.monotonic() >= deadline:
                cutoff_reason = 'deadline'
                break
            attempted += 1
            try:
                scores = evaluator.evaluate_all(world, candidates, deadline=deadline)
            except Exception as exc:
                cutoff_reason = 'world_error'
                world_errors.append(f'{world.world_id}:{type(exc).__name__}:{exc}')
                continue
            if not set(scores) != set(signatures):
                pass
            else:
                cutoff_reason = 'incomplete_world'
                continue
            complete += 1
            for signature in signatures:
                values[signature].append(float(scores[signature].value))
        elapsed_ms = (time.monotonic() - started) * 1000.0
        greedy = greedy_rows[0]
        if complete < self.config.minimum_complete_worlds:
            return InformationSetSelection(selected_action=greedy.action, greedy_action=greedy.action, override_applied=False, reason=f'fallback_{cutoff_reason}_complete_worlds_{complete}_below_{self.config.minimum_complete_worlds}', complete_worlds=complete, attempted_worlds=attempted, elapsed_ms=elapsed_ms, aggregates=(), world_errors=tuple(world_errors))
        greedy_log_probability = greedy.log_probability
        aggregates = []
        for candidate in candidates:
            world_values = tuple(values[candidate.semantic_signature])
            mean_value = sum(world_values) / len(world_values)
            quantile_value = _linear_quantile(world_values, self.config.risk_quantile)
            prior_delta = candidate.log_probability - greedy_log_probability
            robust_score = (1.0 - self.config.risk_weight) * mean_value + self.config.risk_weight * quantile_value + self.config.policy_prior_weight * prior_delta
            aggregates.append(CandidateWorldAggregate(action=candidate.action, is_greedy=candidate.is_greedy, policy_log_probability=candidate.log_probability, world_values=world_values, mean_value=mean_value, risk_quantile_value=quantile_value, robust_score=robust_score))
        aggregates.sort(key=lambda row: (row.robust_score, row.mean_value, row.policy_log_probability), reverse=True)
        greedy_aggregate = next((row for row in aggregates if row.is_greedy))
        best = aggregates[0]
        margin = best.robust_score - greedy_aggregate.robust_score
        override = not best.is_greedy and margin >= self.config.minimum_override_margin
        jury_reason = None
        if override and self.config.require_mean_and_risk_override_agreement:
            mean_best = max(aggregates, key=lambda row: (row.mean_value, row.policy_log_probability))
            risk_best = max(aggregates, key=lambda row: (row.risk_quantile_value, row.policy_log_probability))
            component_margin = float(self.config.minimum_component_override_margin)
            mean_margin = mean_best.mean_value - greedy_aggregate.mean_value
            risk_margin = risk_best.risk_quantile_value - greedy_aggregate.risk_quantile_value
            jury_pass = not mean_best.is_greedy and (not risk_best.is_greedy) and (mean_best.action == risk_best.action == best.action) and (mean_margin >= component_margin) and (risk_margin >= component_margin)
            if not not jury_pass:
                pass
            else:
                override = False
                jury_reason = f'jury_keep_greedy_mean_{mean_best.action}_risk_{risk_best.action}_mean_margin_{mean_margin:.6f}_risk_margin_{risk_margin:.6f}'
        if not (override and self.config.world_panel_size):
            pass
        else:
            panel_size = int(self.config.world_panel_size)
            world_count = len(greedy_aggregate.world_values)
            if world_count % panel_size or world_count < 2 * panel_size:
                override = False
                jury_reason = 'panel_jury_keep_greedy_incomplete_panels'
            else:
                panel_winners = []
                panel_margins = []
                for start in range(0, world_count, panel_size):
                    stop = start + panel_size
                    panel_scores = []
                    for row in aggregates:
                        panel_values = row.world_values[start:stop]
                        panel_mean = sum(panel_values) / len(panel_values)
                        panel_quantile = _linear_quantile(panel_values, self.config.risk_quantile)
                        prior_delta = row.policy_log_probability - greedy_aggregate.policy_log_probability
                        panel_score = (1.0 - self.config.risk_weight) * panel_mean + self.config.risk_weight * panel_quantile + self.config.policy_prior_weight * prior_delta
                        panel_scores.append((panel_score, panel_mean, row))
                    panel_scores.sort(key=lambda item: (item[0], item[1], item[2].policy_log_probability), reverse=True)
                    winner = panel_scores[0][2]
                    greedy_score = next((score for score, _, row in panel_scores if row.is_greedy))
                    panel_winners.append(winner)
                    panel_margins.append(panel_scores[0][0] - greedy_score)
                panel_pass = all((not row.is_greedy for row in panel_winners)) and all((row.action == best.action for row in panel_winners)) and all((value >= self.config.minimum_override_margin for value in panel_margins))
                if not panel_pass:
                    override = False
                    jury_reason = 'panel_jury_keep_greedy_winners_' + '_'.join((str(row.action) for row in panel_winners)) + '_margins_' + '_'.join((f'{value:.6f}' for value in panel_margins))
        selected = greedy.action if not override else best.action
        reason = f'override_margin_{margin:.6f}' if override else jury_reason or f'keep_greedy_margin_{margin:.6f}_{cutoff_reason}'
        return InformationSetSelection(selected_action=selected, greedy_action=greedy.action, override_applied=override, reason=reason, complete_worlds=complete, attempted_worlds=attempted, elapsed_ms=elapsed_ms, aggregates=tuple(aggregates), world_errors=tuple(world_errors))

def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted((float(value) for value in values))
    if not ordered:
        raise ValueError('quantile requires at least one value')
    if not len(ordered) == 1:
        pass
    else:
        return ordered[0]
    position = float(probability) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
