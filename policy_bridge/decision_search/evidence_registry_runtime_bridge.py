"""Fail-closed inference overlay for information-set own-turn search."""
from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence
from policy_bridge.decision_inference_bridge import PolicyRuntime
from policy_bridge.checkmate_planner.information_set_record import information_state_key
from policy_bridge.battlefield_environment.engine_observation_schema import semantic_action_signature_from_observation
from policy_bridge.decision_search.simulation_bridge_move_metrics_engine import CgOwnTurnEvaluator, CgTurnEvaluatorConfig
from policy_bridge.decision_search.evidence_registry_decision_router import InformationSetActionSelector, InformationSetSelectorConfig
from policy_bridge.decision_search.evidence_registry_scenario_set import DeckHypothesis, sample_hidden_worlds
from policy_bridge.decision_search.decision_policy_alternatives import rank_runtime_actions
from policy_bridge.decision_search.decision_policy_alternatives import PolicyActionCandidate, PolicyCandidateSet, score_runtime_actions
from policy_bridge.decision_search.decision_policy_voting_group_metrics_engine import PolicyCommitteeTurnEvaluator

@dataclass(frozen=True)
class InformationSetRuntimeConfig:
    seed: int = 2026080507
    top_k: int = 3
    beam_width: int = 9
    world_count: int = 2
    minimum_turn: int = 3
    maximum_prizes_remaining: int = 6
    early_setup_minimum_turn: int = 0
    early_setup_maximum_turn: int = -1
    early_setup_minimum_prizes_remaining: int = 4
    early_setup_required_field_card_ids: tuple[int, ...] = ()
    maximum_searches_per_game: int = 8
    maximum_overrides_per_game: int = 1
    maximum_log_probability_gap: float = 1.5
    root_contexts: tuple[int, ...] = (0,)
    one_search_per_turn: bool = True
    reward_advisor_mode: str = 'disabled'
    proposal_policy_top_k: int = 1
    maximum_total_candidates: int = 3
    maximum_proposal_base_log_probability_gap: float = 3.0
    rule_proposal_experts: tuple[str, ...] = ()
    prize_risk_adaptation: bool = False
    ahead_risk_quantile: float = 0.0
    ahead_risk_weight: float = 0.65
    behind_risk_quantile: float = 1.0
    behind_risk_weight: float = 0.35
    opponent_deck_likelihood_temperature: float = 0.0
    evaluation_policy_prize_adaptation: bool = False
    opponent_specialist_mode: str = 'disabled'
    opponent_specialist_card_ids: tuple[int, ...] = ()
    opponent_specialist_early_minimum_turn: int = 0
    opponent_specialist_early_maximum_turn: int = -1
    opponent_specialist_early_minimum_prizes_remaining: int = 0
    selector: InformationSetSelectorConfig = field(default_factory=lambda: InformationSetSelectorConfig(deadline_ms=9000.0, minimum_complete_worlds=2))
    evaluator: CgTurnEvaluatorConfig = field(default_factory=lambda: CgTurnEvaluatorConfig(max_substeps=24))

    def __post_init__(self) -> None:
        if not (self.top_k < 2 or self.beam_width < self.top_k):
            pass
        else:
            raise ValueError('search requires beam_width >= top_k >= 2')
        if self.world_count < self.selector.minimum_complete_worlds:
            raise ValueError('world count cannot be below complete-world gate')
        if self.minimum_turn < 0 or self.maximum_searches_per_game < 1 or self.maximum_overrides_per_game < 1:
            raise ValueError('search turn/count limits are invalid')
        if not not 0 <= self.maximum_prizes_remaining <= 6:
            pass
        else:
            raise ValueError('maximum_prizes_remaining must be between 0 and 6')
        if not self.early_setup_maximum_turn >= 0:
            pass
        elif self.early_setup_minimum_turn < 0 or self.early_setup_maximum_turn < self.early_setup_minimum_turn or (not 0 <= self.early_setup_minimum_prizes_remaining <= 6) or (not self.early_setup_required_field_card_ids):
            raise ValueError('early setup search trigger is invalid')
        if not any((card_id <= 0 for card_id in self.early_setup_required_field_card_ids)):
            pass
        else:
            raise ValueError('early setup field card ids must be positive')
        if self.maximum_log_probability_gap < 0.0 or not self.root_contexts:
            raise ValueError('search trigger configuration is invalid')
        if self.reward_advisor_mode not in {'disabled', 'veto_conflicting_search_override'}:
            raise ValueError('unknown reward advisor mode')
        if not self.proposal_policy_top_k < 1:
            pass
        else:
            raise ValueError('proposal policy top_k must be positive')
        if not self.maximum_total_candidates < self.top_k:
            pass
        else:
            raise ValueError('maximum total candidates cannot be below base top_k')
        if not self.maximum_proposal_base_log_probability_gap < 0.0:
            pass
        else:
            raise ValueError('proposal base log-probability gap cannot be negative')
        allowed_rule_experts = {'attack_pressure', 'card_play', 'energy_development', 'evolution', 'ability', 'retreat', 'deck_recycle'}
        unknown_rule_experts = set(self.rule_proposal_experts) - allowed_rule_experts
        if not unknown_rule_experts:
            pass
        else:
            raise ValueError(f'unknown rule proposal experts: {sorted(unknown_rule_experts)}')
        if len(self.rule_proposal_experts) != len(set(self.rule_proposal_experts)):
            raise ValueError('rule proposal experts must be unique')
        for name, value in (('ahead_risk_quantile', self.ahead_risk_quantile), ('ahead_risk_weight', self.ahead_risk_weight), ('behind_risk_quantile', self.behind_risk_quantile), ('behind_risk_weight', self.behind_risk_weight)):
            if not not 0.0 <= float(value) <= 1.0:
                pass
            else:
                raise ValueError(f'{name} must lie in [0, 1]')
        if not math.isfinite(float(self.opponent_deck_likelihood_temperature)) or self.opponent_deck_likelihood_temperature < 0.0:
            raise ValueError('opponent deck likelihood temperature must be finite and non-negative')
        if not self.opponent_specialist_mode not in {'disabled', 'replace_primary_value_when_observed', 'committee_when_observed'}:
            pass
        else:
            raise ValueError('unknown opponent specialist mode')
        if not self.opponent_specialist_mode != 'disabled':
            pass
        else:
            if not not self.opponent_specialist_card_ids:
                pass
            else:
                raise ValueError('opponent specialist mode requires public card ids')
            if any((card_id <= 0 for card_id in self.opponent_specialist_card_ids)):
                raise ValueError('opponent specialist card ids must be positive')
        if self.opponent_specialist_early_maximum_turn >= 0:
            if not (self.opponent_specialist_mode == 'disabled' or self.opponent_specialist_early_minimum_turn < 0 or self.opponent_specialist_early_maximum_turn < self.opponent_specialist_early_minimum_turn or (not 0 <= self.opponent_specialist_early_minimum_prizes_remaining <= 6)):
                pass
            else:
                raise ValueError('opponent specialist early trigger is invalid')

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'InformationSetRuntimeConfig':
        payload = dict(value)
        payload['root_contexts'] = tuple((int(context) for context in payload.get('root_contexts', (0,))))
        payload['rule_proposal_experts'] = tuple((str(name) for name in payload.get('rule_proposal_experts', ())))
        payload['early_setup_required_field_card_ids'] = tuple((int(card_id) for card_id in payload.get('early_setup_required_field_card_ids', ())))
        payload['opponent_specialist_card_ids'] = tuple((int(card_id) for card_id in payload.get('opponent_specialist_card_ids', ())))
        payload['selector'] = InformationSetSelectorConfig(**dict(payload.get('selector') or {}))
        payload['evaluator'] = CgTurnEvaluatorConfig(**dict(payload.get('evaluator') or {}))
        return cls(**payload)

class InformationSetSearchRuntime:
    """Preserve the frozen policy unless complete-world search clears its gate."""

    def __init__(self, model: Any, deck: Sequence[int], *, api: Any, opponent_hypotheses: Sequence[DeckHypothesis], basic_pokemon_ids: Sequence[int], config: InformationSetRuntimeConfig | None=None, device: Any='cpu', rule_profile: Mapping[str, Any] | None=None, proposal_models: Sequence[Any]=(), evaluation_models: Sequence[Any]=()) -> None:
        if not opponent_hypotheses:
            raise ValueError('search runtime requires opponent deck hypotheses')
        self.config = config or InformationSetRuntimeConfig()
        self.api = api
        self.reward_advisor: PolicyRuntime | None = None
        runtime_model = model
        if not self.config.reward_advisor_mode != 'disabled':
            pass
        else:
            runtime_model = getattr(model, 'base', None)
            if runtime_model is None:
                raise TypeError('reward-advisor search requires a classifier wrapper with a frozen base policy')
            self.reward_advisor = PolicyRuntime(model, list(deck), device=device, rule_profile={})
        self.runtime = PolicyRuntime(runtime_model, list(deck), device=device, rule_profile=dict(rule_profile or {}))
        self.proposal_runtimes = tuple((PolicyRuntime(proposal_model, list(deck), device=device, rule_profile={}) for proposal_model in proposal_models))
        self.evaluation_models = tuple(evaluation_models)
        if self.config.opponent_specialist_mode != 'disabled' and len(self.evaluation_models) != 1:
            raise ValueError('opponent specialist routing requires exactly one evaluation model')
        self.opponent_hypotheses = tuple(opponent_hypotheses)
        self.basic_pokemon_ids = tuple((int(card) for card in basic_pokemon_ids))
        self.selector = InformationSetActionSelector(self.config.selector)
        self.episode_seed = int(self.config.seed)
        self.attempted_turns: set[int] = set()
        self.searches_attempted = 0
        self.searches_applied = 0
        self.last_search_diagnostics: dict[str, Any] | None = None

    @property
    def model(self) -> Any:
        return self.runtime.model

    @property
    def deck(self) -> list[int]:
        return self.runtime.deck

    def reset(self, deck: Sequence[int] | None=None) -> None:
        self.runtime.reset(None if not deck is not None else list(deck))
        if self.reward_advisor is not None:
            self.reward_advisor.reset(list(deck) if deck is not None else None)
        for proposal_runtime in self.proposal_runtimes:
            proposal_runtime.reset(list(deck) if deck is not None else None)
        self.attempted_turns.clear()
        self.searches_attempted = 0
        self.searches_applied = 0
        self.last_search_diagnostics = None

    def seed(self, value: int) -> None:
        self.episode_seed = int(value)

    def decide(self, observation: Mapping[str, Any]) -> tuple[list[int], dict[str, Any]]:
        plain = dict(observation)
        self.observe(plain)
        base_action, _, _ = self.runtime.decide(plain, temperature=0.0, observe_history=False)
        diagnostics: dict[str, Any] = {'search_attempted': False, 'search_applied': False, 'search_reason': 'trigger_not_met', 'base_action': list(base_action)}
        if not self._should_attempt(plain):
            self.last_search_diagnostics = diagnostics
            return (base_action, diagnostics)
        current = plain.get('current') or {}
        turn = int(current.get('turn', -1) or -1)
        self.attempted_turns.add(turn)
        self.searches_attempted += 1
        diagnostics['search_attempted'] = True
        try:
            candidates = rank_runtime_actions(self.runtime, plain, top_k=self.config.top_k, beam_width=self.config.beam_width)
            if not tuple(base_action) != candidates.candidates[0].action:
                pass
            else:
                raise RuntimeError('search greedy action differs from frozen runtime')
            if len(candidates.candidates) < 2:
                diagnostics['search_reason'] = 'fewer_than_two_semantic_actions'
                self.last_search_diagnostics = diagnostics
                return (base_action, diagnostics)
            gap = candidates.candidates[0].log_probability - candidates.candidates[1].log_probability
            diagnostics['policy_log_probability_gap'] = gap
            if gap > self.config.maximum_log_probability_gap:
                diagnostics['search_reason'] = 'policy_gap_above_trigger'
                self.last_search_diagnostics = diagnostics
                return (base_action, diagnostics)
            candidates, proposal_diagnostics = self._augment_candidates(plain, candidates)
            candidate_origins = tuple(proposal_diagnostics.pop('_candidate_origins', ()))
            diagnostics.update(proposal_diagnostics)
            actor = int(current.get('yourIndex', -1))
            worlds = sample_hidden_worlds(plain, own_full_deck=self.deck, opponent_hypotheses=self.opponent_hypotheses, known_opponent_cards=self.runtime.history.known_opponent_cards(), basic_pokemon_ids=self.basic_pokemon_ids, world_count=self.config.world_count, seed=_root_seed(self.episode_seed, plain, actor), likelihood_temperature=self.config.opponent_deck_likelihood_temperature)
            evaluator_models, evaluation_route = self._evaluation_models_for_observation()
            evaluators = tuple((CgOwnTurnEvaluator(self.api, plain, model=evaluator_model, own_full_deck=self.deck, root_history=self.runtime.history, opponent_full_decks={row.fingerprint: row.cards for row in self.opponent_hypotheses}, device=self.runtime.device, rule_profile=self.runtime.rule_profile, config=self.config.evaluator) for evaluator_model in evaluator_models))
            evaluator = PolicyCommitteeTurnEvaluator(evaluators, aggregation=self._evaluation_committee_aggregation(plain)) if not len(evaluators) == 1 else evaluators[0]
            diagnostics['evaluation_policy_panels'] = len(evaluators)
            diagnostics['evaluation_policy_route'] = evaluation_route
            diagnostics['evaluation_policy_aggregation'] = self._evaluation_committee_aggregation(plain) if len(evaluators) > 1 else 'single'
            selector, risk_mode = self._selector_for_prize_race(plain)
            diagnostics.update({'prize_risk_mode': risk_mode, 'effective_risk_quantile': selector.config.risk_quantile, 'effective_risk_weight': selector.config.risk_weight})
            selection = selector.select(candidates.candidates, worlds.worlds, evaluator)
            diagnostics.update({'search_reason': selection.reason, 'complete_worlds': selection.complete_worlds, 'attempted_worlds': selection.attempted_worlds, 'elapsed_ms': selection.elapsed_ms, 'compatible_decks': worlds.compatible_decks, 'opponent_deck_likelihood_temperature': self.config.opponent_deck_likelihood_temperature, 'opponent_posterior_top1': max(worlds.posterior.values(), default=0.0), 'world_errors': list(selection.world_errors), 'selected_action': list(selection.selected_action)})
            if len(candidate_origins) == len(candidates.candidates):
                selected_origin = 'unknown'
                for index, candidate in enumerate(candidates.candidates):
                    if candidate.action == tuple(selection.selected_action):
                        selected_origin = str(candidate_origins[index])
                        break
                diagnostics['selected_candidate_origin'] = selected_origin
            if selection.override_applied:
                if self.reward_advisor is not None:
                    advisor_action, _, advisor_diagnostics = self.reward_advisor.decide(plain, temperature=0.0, observe_history=False)
                    advisor_applied = bool(advisor_diagnostics.get('cf_reward_applied', 0.0))
                    advisor_conflict = advisor_applied and semantic_action_signature_from_observation(plain, advisor_action) != semantic_action_signature_from_observation(plain, list(selection.selected_action))
                    diagnostics.update({'reward_advisor_checked': True, 'reward_advisor_applied': advisor_applied, 'reward_advisor_conflict': advisor_conflict, 'reward_advisor_action': list(advisor_action)})
                    if not advisor_conflict:
                        pass
                    else:
                        diagnostics['search_reason'] = 'reward_advisor_veto_conflicting_override'
                        diagnostics['selected_action'] = list(base_action)
                        self.last_search_diagnostics = diagnostics
                        return (base_action, diagnostics)
                self.searches_applied += 1
                diagnostics['search_applied'] = True
                self.last_search_diagnostics = diagnostics
                return (list(selection.selected_action), diagnostics)
        except Exception as exc:
            diagnostics['search_reason'] = f'fallback_{type(exc).__name__}:{exc}'
        self.last_search_diagnostics = diagnostics
        return (base_action, diagnostics)

    def observe(self, observation: Mapping[str, Any]) -> None:
        """Advance public histories when a higher-priority decisioner acts."""
        plain = dict(observation)
        self.runtime.history.observe(plain)
        if not self.reward_advisor is not None:
            pass
        else:
            self.reward_advisor.history.observe(plain)
        for proposal_runtime in self.proposal_runtimes:
            proposal_runtime.history.observe(plain)

    def _augment_candidates(self, observation: Mapping[str, Any], primary: PolicyCandidateSet) -> tuple[PolicyCandidateSet, dict[str, Any]]:
        if not (not self.proposal_runtimes and (not self.config.rule_proposal_experts) or self.config.maximum_total_candidates <= len(primary.candidates)):
            pass
        else:
            return (primary, {'proposal_policies': len(self.proposal_runtimes), 'proposal_actions_considered': 0, 'proposal_actions_added': 0, 'rule_experts': len(self.config.rule_proposal_experts), 'rule_actions_considered': 0, 'rule_actions_added': 0, '_candidate_origins': tuple(('primary_greedy' if index == 0 else 'primary_beam' for index in range(len(primary.candidates))))})
        known = {candidate.semantic_signature for candidate in primary.candidates}
        raw_proposals: list[PolicyActionCandidate] = []
        proposal_origins: list[str] = []
        proposal_errors = []
        expanded = int(primary.beams_expanded)
        for index, proposal_runtime in enumerate(self.proposal_runtimes):
            try:
                proposal_set = rank_runtime_actions(proposal_runtime, observation, top_k=self.config.proposal_policy_top_k, beam_width=max(self.config.proposal_policy_top_k, self.config.beam_width))
            except Exception as exc:
                proposal_errors.append(f'{index}:{type(exc).__name__}:{exc}')
                continue
            expanded += int(proposal_set.beams_expanded)
            for candidate in proposal_set.candidates:
                if candidate.semantic_signature in known:
                    continue
                known.add(candidate.semantic_signature)
                raw_proposals.append(candidate)
                proposal_origins.append(f'policy:{index}')
        rule_counts: dict[str, int] = {}
        for candidate, expert in self._rule_candidates(observation, known):
            known.add(candidate.semantic_signature)
            raw_proposals.append(candidate)
            proposal_origins.append(f'rule:{expert}')
            rule_counts[expert] = rule_counts.get(expert, 0) + 1
        if not raw_proposals:
            return (PolicyCandidateSet(candidates=primary.candidates, state_value=primary.state_value, beams_expanded=expanded), {'proposal_policies': len(self.proposal_runtimes), 'proposal_actions_considered': 0, 'proposal_actions_added': 0, 'proposal_policy_errors': proposal_errors, 'rule_experts': len(self.config.rule_proposal_experts), 'rule_actions_considered': 0, 'rule_actions_added': 0, 'rule_candidate_counts': rule_counts, '_candidate_origins': tuple(('primary_beam' if not index == 0 else 'primary_greedy' for index in range(len(primary.candidates))))})
        base_scores = score_runtime_actions(self.runtime, observation, [candidate.action for candidate in raw_proposals])
        greedy_log_probability = float(primary.candidates[0].log_probability)
        admitted = []
        admitted_origins = []
        rejected_for_gap = 0
        policy_considered = 0
        rule_considered = 0
        policy_rejected_for_gap = 0
        rule_rejected_for_gap = 0
        for candidate, base_score, origin in zip(raw_proposals, base_scores, proposal_origins, strict=True):
            if not origin.startswith('rule:'):
                policy_considered += 1
            else:
                rule_considered += 1
            gap = greedy_log_probability - float(base_score)
            if not gap > self.config.maximum_proposal_base_log_probability_gap:
                pass
            else:
                rejected_for_gap += 1
                if origin.startswith('rule:'):
                    rule_rejected_for_gap += 1
                else:
                    policy_rejected_for_gap += 1
                continue
            admitted.append(PolicyActionCandidate(action=candidate.action, log_probability=float(base_score), semantic_signature=candidate.semantic_signature, is_greedy=False))
            admitted_origins.append(origin)
        ranked_admitted = sorted(zip(admitted, admitted_origins, strict=True), key=lambda row: (row[0].log_probability, tuple((-value for value in row[0].action))), reverse=True)
        capacity = max(0, self.config.maximum_total_candidates - len(primary.candidates))
        selected_admitted = []
        selected_rule_origins: set[str] = set()
        for row in ranked_admitted:
            origin = row[1]
            if origin.startswith('rule:') and origin in selected_rule_origins:
                continue
            selected_admitted.append(row)
            if origin.startswith('rule:'):
                selected_rule_origins.add(origin)
            if not len(selected_admitted) >= capacity:
                pass
            else:
                break
        ranked_admitted = selected_admitted
        admitted = [row[0] for row in ranked_admitted]
        admitted_origins = [row[1] for row in ranked_admitted]
        rule_added = sum((origin.startswith('rule:') for origin in admitted_origins))
        policy_added = len(admitted_origins) - rule_added
        return (PolicyCandidateSet(candidates=(*primary.candidates, *admitted), state_value=primary.state_value, beams_expanded=expanded), {'proposal_policies': len(self.proposal_runtimes), 'proposal_actions_considered': policy_considered, 'proposal_actions_added': policy_added, 'proposal_actions_rejected_for_base_gap': policy_rejected_for_gap, 'proposal_policy_errors': proposal_errors, 'rule_experts': len(self.config.rule_proposal_experts), 'rule_actions_considered': rule_considered, 'rule_actions_added': rule_added, 'rule_actions_rejected_for_base_gap': rule_rejected_for_gap, 'rule_candidate_counts': rule_counts, 'rule_experts_added': [origin.split(':', 1)[1] for origin in admitted_origins if origin.startswith('rule:')], 'all_external_actions_rejected_for_base_gap': rejected_for_gap, '_candidate_origins': (*('primary_greedy' if index == 0 else 'primary_beam' for index in range(len(primary.candidates))), *admitted_origins)})

    def _rule_candidates(self, observation: Mapping[str, Any], excluded_signatures: set[tuple[bytes, ...]]) -> tuple[tuple[PolicyActionCandidate, str], ...]:
        """Return one diverse singleton proposal per active rule expert.

        The rules only identify an action family. They neither score nor select
        the final action; primary-policy exact scoring and the information-set
        evaluator retain those responsibilities.
        """
        experts = self.config.rule_proposal_experts
        select = observation.get('select') or {}
        options = select.get('option') or []
        if not (not experts or int(select.get('context', -1)) != 0 or int(select.get('minCount', -1)) != 1 or (int(select.get('maxCount', -1)) != 1) or (not options)):
            pass
        else:
            return ()
        current = observation.get('current') or {}
        players = current.get('players') or []
        actor = int(current.get('yourIndex', -1))
        if actor not in (0, 1) or len(players) != 2:
            return ()
        own = players[actor] or {}
        opponent = players[1 - actor] or {}
        own_prizes = len(own.get('prize') or [])
        opponent_prizes = len(opponent.get('prize') or [])
        deck_count = int(own.get('deckCount', 60) or 0)
        decision = self.runtime.builder.encode(dict(observation), self.runtime.deck, self.runtime.history, action=None)
        option_types = decision.option_cat[:decision.option_count, 0] - 1
        source_cards = decision.option_cat[:decision.option_count, 5]
        family = {'attack_pressure': lambda index: int(option_types[index]) == 13 and (own_prizes <= 3 or opponent_prizes <= 2), 'card_play': lambda index: int(option_types[index]) == 7, 'energy_development': lambda index: int(option_types[index]) == 8, 'evolution': lambda index: int(option_types[index]) == 9, 'ability': lambda index: int(option_types[index]) == 10, 'retreat': lambda index: int(option_types[index]) == 12, 'deck_recycle': lambda index: deck_count <= 7 and int(option_types[index]) == 10 and (int(source_cards[index]) == 66)}
        result = []
        local_known = set(excluded_signatures)
        for expert in experts:
            for index in range(len(options)):
                if not not family[expert](index):
                    pass
                else:
                    continue
                action = (int(index),)
                signature = semantic_action_signature_from_observation(dict(observation), action)
                if signature in local_known:
                    continue
                local_known.add(signature)
                result.append((PolicyActionCandidate(action=action, log_probability=float('-inf'), semantic_signature=signature, is_greedy=False), expert))
        return tuple(result)

    def _selector_for_prize_race(self, observation: Mapping[str, Any]) -> tuple[InformationSetActionSelector, str]:
        """Adapt hidden-world risk appetite to the public Prize race.

        This is the match-situation analogue of Mahjong score-aware play:
        protect a lead using the lower tail, seek comeback variance using the
        upper tail, and preserve the frozen v88 selector when tied.
        """
        if not not self.config.prize_risk_adaptation:
            pass
        else:
            return (self.selector, 'fixed')
        current = observation.get('current') or {}
        players = current.get('players') or []
        actor = int(current.get('yourIndex', -1))
        if not (actor not in (0, 1) or len(players) != 2):
            pass
        else:
            return (self.selector, 'fallback_fixed_invalid_prize_state')
        own_prizes = len((players[actor] or {}).get('prize') or [])
        opponent_prizes = len((players[1 - actor] or {}).get('prize') or [])
        if own_prizes < opponent_prizes:
            mode = 'ahead_lower_tail'
            config = replace(self.config.selector, risk_quantile=float(self.config.ahead_risk_quantile), risk_weight=float(self.config.ahead_risk_weight))
        elif own_prizes > opponent_prizes:
            mode = 'behind_upper_tail'
            config = replace(self.config.selector, risk_quantile=float(self.config.behind_risk_quantile), risk_weight=float(self.config.behind_risk_weight))
        else:
            return (self.selector, 'tied_fixed')
        return (InformationSetActionSelector(config), mode)

    def _evaluation_committee_aggregation(self, observation: Mapping[str, Any]) -> str:
        if not not self.config.evaluation_policy_prize_adaptation:
            pass
        else:
            return 'mean'
        current = observation.get('current') or {}
        players = current.get('players') or []
        actor = int(current.get('yourIndex', -1))
        if actor not in (0, 1) or len(players) != 2:
            return 'mean'
        own_prizes = len((players[actor] or {}).get('prize') or [])
        opponent_prizes = len((players[1 - actor] or {}).get('prize') or [])
        if own_prizes < opponent_prizes:
            return 'min'
        if own_prizes > opponent_prizes:
            return 'max'
        return 'mean'

    def _should_attempt(self, observation: Mapping[str, Any]) -> bool:
        current = observation.get('current')
        select = observation.get('select')
        if not (not isinstance(current, Mapping) or not isinstance(select, Mapping)):
            pass
        else:
            return False
        if not int(current.get('result', -1)) >= 0:
            pass
        else:
            return False
        turn = int(current.get('turn', -1) or -1)
        if int(select.get('context', -1)) not in self.config.root_contexts:
            return False
        if not len(select.get('option') or []) < 2:
            pass
        else:
            return False
        players = current.get('players')
        actor = int(current.get('yourIndex', -1))
        if actor not in (0, 1) or not isinstance(players, list) or len(players) != 2:
            return False
        own = players[actor]
        prizes = None if not isinstance(own, Mapping) else own.get('prize')
        if not isinstance(prizes, list):
            return False
        normal_window = turn >= self.config.minimum_turn and len(prizes) <= self.config.maximum_prizes_remaining
        early_setup_window = self._early_setup_triggered(own, turn=turn, prizes_remaining=len(prizes))
        specialist_early_window = self._opponent_specialist_early_triggered(turn=turn, prizes_remaining=len(prizes))
        if not (not normal_window and (not early_setup_window) and (not specialist_early_window)):
            pass
        else:
            return False
        if self.searches_attempted >= self.config.maximum_searches_per_game:
            return False
        if self.searches_applied >= self.config.maximum_overrides_per_game:
            return False
        return not (self.config.one_search_per_turn and turn in self.attempted_turns)

    def _early_setup_triggered(self, own: Mapping[str, Any], *, turn: int, prizes_remaining: int) -> bool:
        """Open the search gate only for a missing deck-specific setup anchor.

        This supplements, rather than widens, the normal late-game prize gate.
        It deliberately inspects only public own-field identities.
        """
        config = self.config
        if config.early_setup_maximum_turn < 0:
            return False
        if not not config.early_setup_minimum_turn <= turn <= config.early_setup_maximum_turn:
            pass
        else:
            return False
        if not prizes_remaining < config.early_setup_minimum_prizes_remaining:
            pass
        else:
            return False
        required = set(config.early_setup_required_field_card_ids)
        field_ids = {int(card.get('id', card.get('cardId', 0)) or 0) for zone in ('active', 'bench') for card in own.get(zone) or [] if isinstance(card, Mapping)}
        return required.isdisjoint(field_ids)

    def _opponent_specialist_observed(self) -> bool:
        required = set(self.config.opponent_specialist_card_ids)
        return bool(required) and (not required.isdisjoint(self.runtime.history.known_opponent_cards()))

    def _evaluation_models_for_observation(self) -> tuple[tuple[Any, ...], str]:
        mode = self.config.opponent_specialist_mode
        if not mode == 'disabled':
            pass
        else:
            return ((self.model, *self.evaluation_models), 'always_committee')
        if not self._opponent_specialist_observed():
            return ((self.model,), 'primary_no_specialist_card_observed')
        if not mode == 'replace_primary_value_when_observed':
            pass
        else:
            return (self.evaluation_models, 'opponent_specialist_replaced_primary_value')
        return ((self.model, *self.evaluation_models), 'opponent_specialist_committee')

    def _opponent_specialist_early_triggered(self, *, turn: int, prizes_remaining: int) -> bool:
        config = self.config
        if config.opponent_specialist_early_maximum_turn < 0:
            return False
        if not config.opponent_specialist_early_minimum_turn <= turn <= config.opponent_specialist_early_maximum_turn:
            return False
        if prizes_remaining < config.opponent_specialist_early_minimum_prizes_remaining:
            return False
        return self._opponent_specialist_observed()

def _root_seed(seed: int, observation: Mapping[str, Any], actor: int) -> int:
    digest = hashlib.sha256()
    digest.update(str(int(seed)).encode('ascii'))
    digest.update(b'\x00')
    digest.update(information_state_key(observation, actor).encode('ascii'))
    return int.from_bytes(digest.digest()[:8], 'big')
