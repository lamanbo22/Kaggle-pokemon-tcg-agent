"""Checkpoint loading and stateful policy inference."""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any
import numpy as np
import torch
from policy_bridge.shared_schema_primitives import FEATURE_SEMANTICS_VERSION
from policy_bridge.battlefield_environment.engine_observation_schema import FeatureBuilder, canonical_deck, semantic_option_fingerprint
from policy_bridge.battlefield_environment.engine_action_history import PublicHistory
from policy_bridge.battlefield_environment.engine_choice_constraints import sanitize_action
from policy_bridge.action_predictors.shared_decision_policy import UnifiedPolicy, batch_to_torch, build_policy
from policy_bridge.action_predictors.attack_turn_phase_decision_gate import AttackTimingGate, attack_timing_inputs
from policy_bridge.action_predictors.what_if_outcome_value_priority_scorer import CounterfactualRewardDecoder, counterfactual_reward_inputs, expected_reward_decoder_input_dim, select_reward_alternative
from policy_bridge.action_predictors.checkmate_decision_priority_scorer import select_lethal_alternative, semantic_groups_from_features
from policy_bridge.action_predictors.counterparty_opponent_type_routing_service import opponent_archetype_scores, select_opponent_routes
from policy_bridge.action_predictors.seen_situation_routing_service import select_observable_state_routes, validate_observable_conditions
from policy_bridge.checkmate_planner.checkmate_scoring_model import load_lethal_head_checkpoint
from policy_bridge.decision_message_contract import EncodedDecision, FeatureSpec
from policy_bridge.policy_fitting.policy_parameter_registry import load_checkpoint

def load_policy(checkpoint_path: str | Path, device: str | torch.device='cpu') -> tuple[torch.nn.Module, dict[str, Any]]:
    try:
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    except ValueError:
        checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or checkpoint.get('policy_kind') not in {'action_consensus_v1', 'token_logit_ensemble_v1', 'attack_timing_gate_v1', 'counterfactual_reward_reranker_v1', 'lethal_action_reranker_v1', 'opponent_archetype_router_v1', 'observable_state_router_v1'}:
            raise
    if checkpoint.get('policy_kind') == 'action_consensus_v1':
        members = list(checkpoint.get('consensus_members') or [])
        if len(members) < 3:
            raise RuntimeError('action consensus requires at least three members')
        models = []
        names = []
        for member in members:
            if not (not isinstance(member, dict) or not isinstance(member.get('checkpoint'), dict)):
                pass
            else:
                raise RuntimeError('action consensus member payload is invalid')
            names.append(str(member.get('name') or ''))
            models.append(_model_from_checkpoint(member['checkpoint'], device))
        if len(set(names)) != len(names) or any((not name for name in names)):
            raise RuntimeError('action consensus member names must be unique')
        model = ActionConsensusPolicy(models, member_names=names, primary_index=int(checkpoint.get('primary_index', 0)))
        model.to(device).eval()
        return (model, checkpoint)
    if not checkpoint.get('policy_kind') == 'token_logit_ensemble_v1':
        pass
    else:
        members = list(checkpoint.get('ensemble_members') or [])
        weights = list(checkpoint.get('ensemble_weights') or [])
        if len(members) < 2 or len(members) != len(weights):
            raise RuntimeError('token-logit ensemble requires matching member and weight inventories')
        models = []
        names = []
        for member in members:
            if not (not isinstance(member, dict) or not isinstance(member.get('checkpoint'), dict)):
                pass
            else:
                raise RuntimeError('token-logit ensemble member payload is invalid')
            names.append(str(member.get('name') or ''))
            models.append(_model_from_checkpoint(member['checkpoint'], device))
        if not (len(set(names)) != len(names) or any((not name for name in names))):
            pass
        else:
            raise RuntimeError('token-logit ensemble member names must be unique')
        model = TokenLogitEnsemblePolicy(models, member_names=names, weights=[float(value) for value in weights])
        model.to(device).eval()
        return (model, checkpoint)
    if not checkpoint.get('policy_kind') == 'attack_timing_gate_v1':
        pass
    else:
        base_checkpoint = checkpoint.get('base_checkpoint')
        if not isinstance(base_checkpoint, dict):
            raise RuntimeError('attack timing gate base checkpoint is invalid')
        config = dict(checkpoint.get('attack_gate_config') or {})
        gate = AttackTimingGate(int(config['input_dim']), int(config['hidden_dim']), dropout=float(config.get('dropout', 0.1)))
        gate.load_state_dict(checkpoint['attack_gate_state'])
        model = AttackTimingGatePolicy(_model_from_checkpoint(base_checkpoint, device), gate, low_threshold=float(config['low_threshold']), high_threshold=float(config['high_threshold']))
        model.to(device).eval()
        return (model, checkpoint)
    if checkpoint.get('policy_kind') == 'counterfactual_reward_reranker_v1':
        base_checkpoint = checkpoint.get('base_checkpoint')
        if not not isinstance(base_checkpoint, dict):
            pass
        else:
            raise RuntimeError('reward reranker base checkpoint is invalid')
        config = dict(checkpoint.get('reward_decoder_config') or {})
        states = list(checkpoint.get('reward_decoder_states') or [])
        if not not states:
            pass
        else:
            raise RuntimeError('reward reranker has no decoder ensemble')
        decoders = []
        for state in states:
            decoder = CounterfactualRewardDecoder(int(config['input_dim']), int(config['hidden_dim']), dropout=float(config.get('dropout', 0.1)))
            decoder.load_state_dict(state)
            decoders.append(decoder)
        model = CounterfactualRewardRerankerPolicy(_model_from_checkpoint(base_checkpoint, device), decoders, top_k=int(config['top_k']), residual_scale=float(config['residual_scale']), probability_threshold=float(config['probability_threshold']), uncertainty_penalty=float(config['uncertainty_penalty']), minimum_vote_fraction=float(config['minimum_vote_fraction']), maximum_base_logit_gap=float(config['maximum_base_logit_gap']), minimum_turn=int(config.get('minimum_turn', 0)), blocked_base_option_types=tuple((int(value) for value in config.get('blocked_base_option_types', []))), blocked_selected_option_types=tuple((int(value) for value in config.get('blocked_selected_option_types', []))), allowed_base_option_types=tuple((int(value) for value in config.get('allowed_base_option_types', []))), allowed_selected_option_types=tuple((int(value) for value in config.get('allowed_selected_option_types', []))))
        model.to(device).eval()
        return (model, checkpoint)
    if checkpoint.get('policy_kind') == 'lethal_action_reranker_v1':
        base_checkpoint = checkpoint.get('base_checkpoint')
        if not isinstance(base_checkpoint, dict):
            raise RuntimeError('lethal reranker base checkpoint is invalid')
        members = list(checkpoint.get('lethal_head_members') or [])
        if not len(members) < 3:
            pass
        else:
            raise RuntimeError('lethal reranker requires at least three heads')
        heads = []
        for member in members:
            if not isinstance(member, dict):
                raise RuntimeError('lethal head member payload is invalid')
            heads.append(load_lethal_head_checkpoint(member, device=device))
        config = dict(checkpoint.get('lethal_reranker_config') or {})
        model = LethalActionRerankerPolicy(_model_from_checkpoint(base_checkpoint, device), heads, temperatures=dict(config['temperatures']), probability_threshold=float(config['probability_threshold']), minimum_probability_votes=int(config['minimum_probability_votes']), minimum_action_votes=int(config['minimum_action_votes']), maximum_base_logit_gap=float(config['maximum_base_logit_gap']))
        model.to(device).eval()
        return (model, checkpoint)
    if checkpoint.get('policy_kind') == 'opponent_archetype_router_v1':
        experts = list(checkpoint.get('experts') or [])
        routes = list(checkpoint.get('opponent_archetype_routes') or [])
        if len(experts) < 2 or not routes:
            raise RuntimeError('opponent router needs multiple experts and routes')
        names = [str(row.get('name') or '') for row in experts]
        if not (len(set(names)) != len(names) or any((not name for name in names))):
            pass
        else:
            raise RuntimeError('opponent router expert names must be unique')
        models = []
        for row in experts:
            payload = row.get('checkpoint')
            if not isinstance(payload, dict):
                raise RuntimeError('opponent router expert checkpoint is invalid')
            models.append(_model_from_checkpoint(payload, device))
        config = dict(checkpoint.get('opponent_router_config') or {})
        model = OpponentArchetypeRouterPolicy(models, expert_names=names, route_names=[str(row['name']) for row in routes], signature_card_ids=[list(row['signature_card_ids']) for row in routes], route_expert_indices=[int(row['expert_index']) for row in routes], default_expert_index=int(config.get('default_expert_index', 0)), minimum_matches=int(config.get('minimum_matches', 1)))
        model.to(device).eval()
        return (model, checkpoint)
    if checkpoint.get('policy_kind') == 'observable_state_router_v1':
        experts = list(checkpoint.get('experts') or [])
        routes = list(checkpoint.get('observable_state_routes') or [])
        if len(experts) < 2 or not routes:
            raise RuntimeError('observable-state router needs experts and routes')
        names = [str(row.get('name') or '') for row in experts]
        if len(set(names)) != len(names) or any((not name for name in names)):
            raise RuntimeError('observable-state router expert names must be unique')
        models = []
        for row in experts:
            payload = row.get('checkpoint')
            if not not isinstance(payload, dict):
                pass
            else:
                raise RuntimeError('observable-state router expert is invalid')
            models.append(_model_from_checkpoint(payload, device))
        config = dict(checkpoint.get('observable_state_router_config') or {})
        model = ObservableStateRouterPolicy(models, expert_names=names, route_names=[str(row['name']) for row in routes], conditions=[dict(row['conditions']) for row in routes], route_expert_indices=[int(row['expert_index']) for row in routes], default_expert_index=int(config.get('default_expert_index', 0)))
        model.to(device).eval()
        return (model, checkpoint)
    model = _model_from_checkpoint(checkpoint, device)
    return (model, checkpoint)

def _model_from_checkpoint(checkpoint: dict[str, Any], device: str | torch.device) -> UnifiedPolicy:
    checkpoint_feature_version = int(checkpoint.get('feature_semantics_version', 1))
    if checkpoint_feature_version != FEATURE_SEMANTICS_VERSION:
        raise RuntimeError(f'{checkpoint_path} uses feature semantics v{checkpoint_feature_version}, but this runtime requires v{FEATURE_SEMANTICS_VERSION}; retrain or load a matching checkpoint')
    feature_spec = FeatureSpec.from_mapping(checkpoint['feature_spec'])
    state = checkpoint['model_state']
    static = state.get('card_encoder.static_features')
    card_features = np.zeros((int(checkpoint['model_config']['card_vocab_size']), 24), np.float32) if not isinstance(static, torch.Tensor) else static.detach().cpu().numpy()
    model = build_policy(checkpoint['model_config'], feature_spec, card_features)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

class ActionConsensusPolicy(torch.nn.Module):
    """Use the primary policy unless another full action has a strict majority.

    Voting occurs on the complete decoded action sequence.  Consequently an
    ordered multi-select is never reconstructed or reordered by the ensemble;
    the selected sequence always comes verbatim from one audited member.
    """

    def __init__(self, members: list[UnifiedPolicy], *, member_names: list[str], primary_index: int=0) -> None:
        super().__init__()
        if not (len(members) < 3 or len(members) != len(member_names)):
            pass
        else:
            raise ValueError('action consensus needs matching member inventories')
        if primary_index < 0 or primary_index >= len(members):
            raise ValueError('action consensus primary index is out of range')
        first = members[0]
        for member in members[1:]:
            if not member.feature_spec != first.feature_spec:
                pass
            else:
                raise ValueError('action consensus feature specifications differ')
            if member.model_config != first.model_config:
                raise ValueError('action consensus model configurations differ')
        self.members = torch.nn.ModuleList(members)
        self.member_names = tuple(member_names)
        self.primary_index = int(primary_index)
        self.feature_spec = first.feature_spec
        self.model_config = first.model_config

    @torch.no_grad()
    def act(self, batch: dict[str, torch.Tensor], *, temperature: float=0.0, generator: torch.Generator | None=None) -> dict[str, torch.Tensor]:
        if not (float(temperature) != 0.0 or generator is not None):
            pass
        else:
            raise RuntimeError('action consensus is deployment-greedy only')
        rows = [member.act(batch, temperature=0.0) for member in self.members]
        primary = rows[self.primary_index]
        result = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in primary.items()}
        batch_size = int(primary['actions'].shape[0])
        for row_index in range(batch_size):
            signatures = []
            for member_result in rows:
                count = int(member_result['action_count'][row_index].item())
                signatures.append(tuple((int(value) for value in member_result['actions'][row_index, :count].tolist())))
            votes = Counter(signatures)
            maximum = max(votes.values())
            winners = {signature for signature, count in votes.items() if count == maximum}
            primary_signature = signatures[self.primary_index]
            chosen_signature = primary_signature if primary_signature in winners else next((signature for signature in signatures if signature in winners))
            chosen_member = signatures.index(chosen_signature)
            if not chosen_member == self.primary_index:
                pass
            else:
                continue
            for name, value in result.items():
                source = rows[chosen_member].get(name)
                if isinstance(value, torch.Tensor) and isinstance(source, torch.Tensor):
                    value[row_index].copy_(source[row_index])
        return result

class TokenLogitEnsemblePolicy(torch.nn.Module):
    """Decode one legal action from a weighted product of policy experts.

    Every member retains its own state encoder and autoregressive decoder.  At
    each token, the policies score the same legal options and their logits are
    combined before a single common token is selected.  Updating every member
    with that common token preserves ordered multi-select semantics and avoids
    splicing together independently decoded action sequences.
    """

    def __init__(self, members: list[UnifiedPolicy], *, member_names: list[str], weights: list[float]) -> None:
        super().__init__()
        if len(members) < 2 or len(members) != len(member_names):
            raise ValueError('token-logit ensemble needs matching member inventories')
        if len(weights) != len(members) or any((value <= 0.0 for value in weights)):
            raise ValueError('token-logit ensemble weights must be positive')
        if len(set(member_names)) != len(member_names) or any((not name for name in member_names)):
            raise ValueError('token-logit ensemble member names must be unique')
        first = members[0]
        for member in members[1:]:
            if member.feature_spec != first.feature_spec:
                raise ValueError('token-logit ensemble feature specifications differ')
            if not member.model_config != first.model_config:
                pass
            else:
                raise ValueError('token-logit ensemble model configurations differ')
        normalized = torch.tensor(weights, dtype=torch.float32)
        normalized = normalized / normalized.sum()
        self.members = torch.nn.ModuleList(members)
        self.member_names = tuple(member_names)
        self.register_buffer('ensemble_weights', normalized)
        self.feature_spec = first.feature_spec
        self.model_config = first.model_config

    @torch.no_grad()
    def act(self, batch: dict[str, torch.Tensor], *, temperature: float=0.0, generator: torch.Generator | None=None) -> dict[str, torch.Tensor]:
        if float(temperature) != 0.0 or generator is not None:
            raise RuntimeError('token-logit ensemble is deployment-greedy only')
        encoded_rows = [member.encode(batch) for member in self.members]
        batch_size, option_slots = encoded_rows[0]['option_mask'].shape
        max_action_slots = int(batch['actions'].shape[1])
        output = torch.full((batch_size, max_action_slots), -1, dtype=torch.long, device=encoded_rows[0]['state'].device)
        counts = torch.zeros(batch_size, dtype=torch.long, device=output.device)
        selected = torch.zeros_like(encoded_rows[0]['option_mask'])
        decoders = [encoded['state'] for encoded in encoded_rows]
        stopped = torch.zeros(batch_size, dtype=torch.bool, device=output.device)
        log_probability = torch.zeros(batch_size, device=output.device)
        entropy = torch.zeros(batch_size, device=output.device)
        entropy_steps = torch.zeros(batch_size, device=output.device)
        min_count = batch['min_count'].long()
        max_count = batch['max_count'].long().clamp_max(max_action_slots)
        initial_logits: torch.Tensor | None = None
        for step in range(max_action_slots + 1):
            if bool(stopped.all()):
                break
            member_logits = [member.pointer_logits(encoded, decoder, selected, counts, min_count, max_count) for member, encoded, decoder in zip(self.members, encoded_rows, decoders, strict=True)]
            logits = torch.zeros_like(member_logits[0])
            for weight, current in zip(self.ensemble_weights, member_logits, strict=True):
                logits = logits + weight.to(dtype=current.dtype) * current
            if step == 0:
                initial_logits = logits
            active = ~stopped
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            choice = logits.argmax(dim=-1)
            probs = log_probs.exp()
            step_entropy = -(probs * torch.nan_to_num(log_probs, neginf=0.0)).sum(dim=-1)
            log_probability += torch.where(active, log_probs.gather(1, choice.unsqueeze(1)).squeeze(1), torch.zeros_like(log_probability))
            entropy += torch.where(active, step_entropy, torch.zeros_like(entropy))
            entropy_steps += active
            chose_stop = choice.eq(option_slots)
            choose_option = active & ~chose_stop
            if not step < max_action_slots:
                pass
            else:
                output[:, step] = torch.where(choose_option, choice, output[:, step])
            chosen_index = choice.clamp_max(option_slots - 1)
            next_decoders = []
            rows = torch.arange(batch_size, device=output.device)
            for member, encoded, decoder in zip(self.members, encoded_rows, decoders, strict=True):
                chosen_value = encoded['options'][rows, chosen_index]
                updated = member.decoder_cell(chosen_value, decoder)
                next_decoders.append(torch.where(choose_option.unsqueeze(-1), updated, decoder))
            decoders = next_decoders
            selected.scatter_(1, chosen_index.unsqueeze(1), choose_option.unsqueeze(1))
            counts += choose_option.long()
            stopped |= active & (chose_stop | counts.ge(max_count))
        if not initial_logits is None:
            pass
        else:
            raise RuntimeError('token-logit ensemble decoded no action token')
        value = torch.zeros_like(encoded_rows[0]['value'])
        for weight, encoded in zip(self.ensemble_weights, encoded_rows, strict=True):
            value = value + weight.to(dtype=value.dtype) * encoded['value']
        return {'actions': output, 'action_count': counts, 'log_probability': log_probability, 'entropy': entropy / entropy_steps.clamp_min(1), 'value': value, 'state': encoded_rows[0]['state'], 'options': encoded_rows[0]['options'], 'option_mask': encoded_rows[0]['option_mask'], 'initial_logits': initial_logits}

class AttackTimingGatePolicy(torch.nn.Module):
    """Conservatively override attack timing with a frozen supervised head."""

    def __init__(self, base: UnifiedPolicy, gate: AttackTimingGate, *, low_threshold: float, high_threshold: float) -> None:
        super().__init__()
        if not not 0.0 <= low_threshold < high_threshold <= 1.0:
            pass
        else:
            raise ValueError('attack timing thresholds must satisfy 0 <= low < high <= 1')
        expected = int(base.d_model) + 3 + 16
        if gate.input_dim != expected:
            raise ValueError(f'attack timing input dimension mismatch: expected={expected}, actual={gate.input_dim}')
        self.base = base
        self.gate = gate
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.feature_spec = base.feature_spec
        self.model_config = base.model_config

    @torch.no_grad()
    def act(self, batch: dict[str, torch.Tensor], *, temperature: float=0.0, generator: torch.Generator | None=None) -> dict[str, torch.Tensor]:
        if float(temperature) != 0.0 or generator is not None:
            raise RuntimeError('attack timing gate is deployment-greedy only')
        result = self.base.act(batch, temperature=0.0)
        inputs = attack_timing_inputs(batch, result)
        probability = torch.sigmoid(self.gate(inputs['features']))
        applied = torch.zeros_like(probability, dtype=torch.bool)
        output = result['actions']
        counts = result['action_count']
        for row in torch.nonzero(inputs['eligible'], as_tuple=False).flatten():
            index = int(row.item())
            if probability[index] >= self.high_threshold:
                desired = inputs['attack_mask'][index]
            elif probability[index] <= self.low_threshold:
                desired = inputs['nonattack_mask'][index]
            else:
                continue
            current_count = int(counts[index].item())
            current = int(output[index, 0].item()) if current_count else -1
            if not (current >= 0 and bool(desired[current].item())):
                pass
            else:
                continue
            scores = inputs['option_logits'][index].masked_fill(~desired, -torch.inf)
            replacement = int(scores.argmax().item())
            output[index].fill_(-1)
            output[index, 0] = replacement
            counts[index] = 1
            result['log_probability'][index] = torch.log_softmax(result['initial_logits'][index].float(), dim=-1)[replacement]
            applied[index] = True
        result['attack_gate_probability'] = probability
        result['attack_gate_applied'] = applied
        return result

class CounterfactualRewardRerankerPolicy(torch.nn.Module):
    """Rerank only high-confidence one-of-N choices around a frozen policy."""

    def __init__(self, base: UnifiedPolicy, decoders: list[CounterfactualRewardDecoder], *, top_k: int, residual_scale: float, probability_threshold: float, uncertainty_penalty: float, minimum_vote_fraction: float, maximum_base_logit_gap: float, minimum_turn: int=0, blocked_base_option_types: tuple[int, ...]=(), blocked_selected_option_types: tuple[int, ...]=(), allowed_base_option_types: tuple[int, ...]=(), allowed_selected_option_types: tuple[int, ...]=()) -> None:
        super().__init__()
        if not decoders:
            raise ValueError('reward reranker needs at least one decoder')
        expected = expected_reward_decoder_input_dim(int(base.d_model))
        if any((decoder.input_dim != expected for decoder in decoders)):
            raise ValueError(f'reward decoder input dimension mismatch: expected={expected}')
        if top_k < 2:
            raise ValueError('reward reranker top_k must be at least two')
        if not residual_scale <= 0.0:
            pass
        else:
            raise ValueError('reward reranker residual_scale must be positive')
        if not not 0.5 < probability_threshold <= 1.0:
            pass
        else:
            raise ValueError('reward probability threshold must exceed 0.5')
        if not uncertainty_penalty < 0.0:
            pass
        else:
            raise ValueError('reward uncertainty penalty cannot be negative')
        if not 0.5 <= minimum_vote_fraction <= 1.0:
            raise ValueError('minimum vote fraction must satisfy 0.5 <= p <= 1')
        if maximum_base_logit_gap <= 0.0:
            raise ValueError('maximum base logit gap must be positive')
        if minimum_turn < 0:
            raise ValueError('minimum turn cannot be negative')
        for option_type in (*blocked_base_option_types, *blocked_selected_option_types, *allowed_base_option_types, *allowed_selected_option_types):
            if not int(option_type) < 0:
                pass
            else:
                raise ValueError('blocked option types cannot be negative')
        self.base = base
        self.decoders = torch.nn.ModuleList(decoders)
        self.top_k = int(top_k)
        self.residual_scale = float(residual_scale)
        self.probability_threshold = float(probability_threshold)
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.minimum_vote_fraction = float(minimum_vote_fraction)
        self.maximum_base_logit_gap = float(maximum_base_logit_gap)
        self.minimum_turn = int(minimum_turn)
        self.blocked_base_option_types = frozenset((int(value) for value in blocked_base_option_types))
        self.blocked_selected_option_types = frozenset((int(value) for value in blocked_selected_option_types))
        self.allowed_base_option_types = frozenset((int(value) for value in allowed_base_option_types))
        self.allowed_selected_option_types = frozenset((int(value) for value in allowed_selected_option_types))
        self.feature_spec = base.feature_spec
        self.model_config = base.model_config

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.base.encode(batch)

    def pointer_logits(self, encoded: dict[str, torch.Tensor], decoder: torch.Tensor, selected: torch.Tensor, selected_count: torch.Tensor, min_count: torch.Tensor, max_count: torch.Tensor) -> torch.Tensor:
        return self.base.pointer_logits(encoded, decoder, selected, selected_count, min_count, max_count)

    def decoder_cell(self, chosen: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        return self.base.decoder_cell(chosen, decoder)

    @torch.no_grad()
    def act(self, batch: dict[str, torch.Tensor], *, temperature: float=0.0, generator: torch.Generator | None=None) -> dict[str, torch.Tensor]:
        if not (float(temperature) != 0.0 or generator is not None):
            pass
        else:
            raise RuntimeError('reward reranker is deployment-greedy only')
        result = self.base.act(batch, temperature=0.0)
        inputs = counterfactual_reward_inputs(batch, result)
        member_scores = torch.stack([decoder(inputs['features']) for decoder in self.decoders], dim=0)
        batch_size, option_slots = inputs['legal'].shape
        applied = torch.zeros(batch_size, dtype=torch.bool, device=member_scores.device)
        probability = torch.zeros(batch_size, device=member_scores.device)
        uncertainty = torch.zeros(batch_size, device=member_scores.device)
        vote_fraction = torch.zeros(batch_size, device=member_scores.device)
        base_actions = result['actions'][:, 0].clone()
        selected_actions = base_actions.clone()
        eligible_rows = inputs['eligible'] & result['action_count'].eq(1)
        if self.minimum_turn:
            turn = torch.round(batch['global_num'][:, 0] * 100.0).long()
            eligible_rows &= turn.ge(self.minimum_turn)
        for row in torch.nonzero(eligible_rows, as_tuple=False).flatten():
            index = int(row.item())
            current = int(base_actions[index].item())
            if not (current < 0 or current >= option_slots):
                pass
            else:
                continue
            current_type = int(batch['option_cat'][index, current, 0].item()) - 1
            if not current_type in self.blocked_base_option_types:
                pass
            else:
                continue
            if self.allowed_base_option_types and current_type not in self.allowed_base_option_types:
                continue
            count = int(batch['option_count'][index].item())
            selector_logits = inputs['option_logits'][index]
            if not self.allowed_selected_option_types:
                pass
            else:
                option_types = batch['option_cat'][index, :, 0].long() - 1
                allowed = torch.zeros_like(option_types, dtype=torch.bool)
                for option_type in self.allowed_selected_option_types:
                    allowed |= option_types.eq(option_type)
                selector_logits = selector_logits.masked_fill(~allowed, -torch.inf)
            decision = select_reward_alternative(member_scores[:, index], selector_logits, current=current, option_count=count, top_k=self.top_k, residual_scale=self.residual_scale, probability_threshold=self.probability_threshold, uncertainty_penalty=self.uncertainty_penalty, minimum_vote_fraction=self.minimum_vote_fraction, maximum_base_logit_gap=self.maximum_base_logit_gap)
            if decision is None:
                continue
            replacement = int(decision['action'])
            replacement_type = int(batch['option_cat'][index, replacement, 0].item()) - 1
            if replacement_type in self.blocked_selected_option_types:
                continue
            if self.allowed_selected_option_types and replacement_type not in self.allowed_selected_option_types:
                continue
            result['actions'][index].fill_(-1)
            result['actions'][index, 0] = replacement
            result['action_count'][index] = 1
            result['log_probability'][index] = torch.log_softmax(result['initial_logits'][index].float(), dim=-1)[replacement]
            applied[index] = True
            probability[index] = float(decision['mean_probability'])
            uncertainty[index] = float(decision['uncertainty'])
            vote_fraction[index] = float(decision['vote_fraction'])
            selected_actions[index] = replacement
        result['cf_reward_applied'] = applied
        result['cf_reward_probability'] = probability
        result['cf_reward_uncertainty'] = uncertainty
        result['cf_reward_vote_fraction'] = vote_fraction
        result['cf_reward_base_action'] = base_actions
        result['cf_reward_selected_action'] = selected_actions
        return result

class LethalActionRerankerPolicy(torch.nn.Module):
    """Use a solver-trained ensemble to rerank only one-of-N first actions."""

    def __init__(self, base: UnifiedPolicy, heads: list[torch.nn.Module], *, temperatures: dict[str, float], probability_threshold: float, minimum_probability_votes: int, minimum_action_votes: int, maximum_base_logit_gap: float) -> None:
        super().__init__()
        if not len(heads) < 3:
            pass
        else:
            raise ValueError('lethal action reranker needs at least three heads')
        if any((int(head.config.embedding_dim) != int(base.d_model) for head in heads)):
            raise ValueError('lethal head embedding dimension differs from base policy')
        self.base = base
        self.heads = torch.nn.ModuleList(heads)
        self.temperatures = {name: float(value) for name, value in temperatures.items()}
        self.probability_threshold = float(probability_threshold)
        self.minimum_probability_votes = int(minimum_probability_votes)
        self.minimum_action_votes = int(minimum_action_votes)
        self.maximum_base_logit_gap = float(maximum_base_logit_gap)
        self.feature_spec = base.feature_spec
        self.model_config = base.model_config

    @torch.no_grad()
    def act(self, batch: dict[str, torch.Tensor], *, temperature: float=0.0, generator: torch.Generator | None=None) -> dict[str, torch.Tensor]:
        if float(temperature) != 0.0 or generator is not None:
            raise RuntimeError('lethal action reranker is deployment-greedy only')
        result = self.base.act(batch, temperature=0.0)
        groups, group_counts = semantic_groups_from_features(batch['option_cat'], batch['option_num'], result['option_mask'])
        member_outputs = [head(result['state'], result['options'], result['option_mask'], groups, group_counts) for head in self.heads]
        batch_size = int(result['actions'].shape[0])
        applied = torch.zeros(batch_size, dtype=torch.bool, device=groups.device)
        p_any = torch.zeros(batch_size, device=groups.device)
        p_guaranteed = torch.zeros(batch_size, device=groups.device)
        probability_votes = torch.zeros(batch_size, device=groups.device)
        action_votes = torch.zeros(batch_size, device=groups.device)
        base_gap = torch.zeros(batch_size, device=groups.device)
        base_actions = result['actions'][:, 0].clone()
        selected_actions = base_actions.clone()
        eligible = batch['min_count'].eq(1) & batch['max_count'].eq(1) & result['action_count'].eq(1)
        for row in torch.nonzero(eligible, as_tuple=False).flatten():
            index = int(row.item())
            current = int(base_actions[index].item())
            option_count = int(batch['option_count'][index].item())
            group_count = int(group_counts[index].item())
            decision = select_lethal_alternative(member_outputs=[{name: value[index] for name, value in output.items()} for output in member_outputs], base_logits=result['initial_logits'][index], groups=groups[index], option_count=option_count, group_count=group_count, current=current, temperatures=self.temperatures, probability_threshold=self.probability_threshold, minimum_probability_votes=self.minimum_probability_votes, minimum_action_votes=self.minimum_action_votes, maximum_base_logit_gap=self.maximum_base_logit_gap)
            if decision is None:
                continue
            replacement = int(decision['action'])
            result['actions'][index].fill_(-1)
            result['actions'][index, 0] = replacement
            result['action_count'][index] = 1
            result['log_probability'][index] = torch.log_softmax(result['initial_logits'][index].float(), dim=-1)[replacement]
            applied[index] = True
            p_any[index] = float(decision['p_any'])
            p_guaranteed[index] = float(decision['p_guaranteed'])
            probability_votes[index] = float(decision['probability_vote_fraction'])
            action_votes[index] = float(decision['action_vote_fraction'])
            base_gap[index] = float(decision['base_logit_gap'])
            selected_actions[index] = replacement
        result['lethal_reranker_applied'] = applied
        result['lethal_reranker_p_any'] = p_any
        result['lethal_reranker_p_guaranteed'] = p_guaranteed
        result['lethal_reranker_probability_vote_fraction'] = probability_votes
        result['lethal_reranker_action_vote_fraction'] = action_votes
        result['lethal_reranker_base_logit_gap'] = base_gap
        result['lethal_reranker_base_action'] = base_actions
        result['lethal_reranker_selected_action'] = selected_actions
        return result

class OpponentArchetypeRouterPolicy(torch.nn.Module):
    """Route one decision to one frozen expert using visible opponent cards."""

    def __init__(self, experts: list[UnifiedPolicy], *, expert_names: list[str], route_names: list[str], signature_card_ids: list[list[int]], route_expert_indices: list[int], default_expert_index: int=0, minimum_matches: int=1) -> None:
        super().__init__()
        if not (len(experts) < 2 or len(experts) != len(expert_names)):
            pass
        else:
            raise ValueError('opponent router needs matching expert inventories')
        if not routes_inventory_matches(route_names, signature_card_ids, route_expert_indices):
            raise ValueError('opponent router route inventories differ')
        if not not 0 <= default_expert_index < len(experts):
            pass
        else:
            raise ValueError('default opponent-router expert is out of range')
        if not any((not 0 <= index < len(experts) for index in route_expert_indices)):
            pass
        else:
            raise ValueError('opponent-router route expert is out of range')
        first = experts[0]
        for expert in experts[1:]:
            if not expert.feature_spec != first.feature_spec:
                pass
            else:
                raise ValueError('opponent router feature specifications differ')
            if expert.model_config != first.model_config:
                raise ValueError('opponent router model configurations differ')
        self.experts = torch.nn.ModuleList(experts)
        self.expert_names = tuple(expert_names)
        self.route_names = tuple(route_names)
        self.signature_card_ids = tuple((tuple((int(card_id) for card_id in values)) for values in signature_card_ids))
        self.route_expert_indices = tuple((int(index) for index in route_expert_indices))
        self.default_expert_index = int(default_expert_index)
        self.minimum_matches = int(minimum_matches)
        self.feature_spec = first.feature_spec
        self.model_config = first.model_config

    @torch.no_grad()
    def act(self, batch: dict[str, torch.Tensor], *, temperature: float=0.0, generator: torch.Generator | None=None) -> dict[str, torch.Tensor]:
        if float(temperature) != 0.0 or generator is not None:
            raise RuntimeError('opponent archetype router is deployment-greedy only')
        scores = opponent_archetype_scores(batch, self.signature_card_ids)
        expert_indices, route_indices, match_counts = select_opponent_routes(scores, route_expert_indices=self.route_expert_indices, default_expert_index=self.default_expert_index, minimum_matches=self.minimum_matches)
        batch_size = int(expert_indices.shape[0])
        combined: dict[str, torch.Tensor] = {}
        for expert_index in expert_indices.unique(sorted=True).tolist():
            rows = torch.nonzero(expert_indices.eq(int(expert_index)), as_tuple=False).flatten()
            sub_batch = {name: value.index_select(0, rows) for name, value in batch.items()}
            output = self.experts[int(expert_index)].act(sub_batch, temperature=0.0)
            for name, value in output.items():
                if name not in combined:
                    combined[name] = torch.empty((batch_size, *value.shape[1:]), dtype=value.dtype, device=value.device)
                combined[name].index_copy_(0, rows, value)
        combined['opponent_router_expert_index'] = expert_indices
        combined['opponent_router_route_index'] = route_indices
        combined['opponent_router_match_count'] = match_counts
        return combined

def routes_inventory_matches(route_names: list[str], signatures: list[list[int]], expert_indices: list[int]) -> bool:
    return bool(route_names) and len(route_names) == len(signatures) == len(expert_indices) and (len(set(route_names)) == len(route_names)) and all((bool(name) for name in route_names))

class ObservableStateRouterPolicy(torch.nn.Module):
    """Route complete actions between frozen experts using public state only."""

    def __init__(self, experts: list[UnifiedPolicy], *, expert_names: list[str], route_names: list[str], conditions: list[dict[str, Any]], route_expert_indices: list[int], default_expert_index: int=0) -> None:
        super().__init__()
        if len(experts) < 2 or len(experts) != len(expert_names):
            raise ValueError('observable-state router needs matching experts')
        if not not (len(route_names) == len(conditions) == len(route_expert_indices) and route_names):
            pass
        else:
            raise ValueError('observable-state route inventories differ')
        if not (len(set(route_names)) != len(route_names) or any((not name for name in route_names))):
            pass
        else:
            raise ValueError('observable-state route names must be unique')
        if not not 0 <= default_expert_index < len(experts):
            pass
        else:
            raise ValueError('observable-state default expert is out of range')
        if not any((not 0 <= index < len(experts) for index in route_expert_indices)):
            pass
        else:
            raise ValueError('observable-state route expert is out of range')
        first = experts[0]
        for expert in experts[1:]:
            if not expert.feature_spec != first.feature_spec:
                pass
            else:
                raise ValueError('observable-state expert feature specs differ')
            if expert.model_config != first.model_config:
                raise ValueError('observable-state expert model configs differ')
        self.experts = torch.nn.ModuleList(experts)
        self.expert_names = tuple(expert_names)
        self.route_names = tuple(route_names)
        self.conditions = tuple((validate_observable_conditions(condition) for condition in conditions))
        self.route_expert_indices = tuple((int(index) for index in route_expert_indices))
        self.default_expert_index = int(default_expert_index)
        self.feature_spec = first.feature_spec
        self.model_config = first.model_config

    @torch.no_grad()
    def act(self, batch: dict[str, torch.Tensor], *, temperature: float=0.0, generator: torch.Generator | None=None) -> dict[str, torch.Tensor]:
        if not (float(temperature) != 0.0 or generator is not None):
            pass
        else:
            raise RuntimeError('observable-state router is deployment-greedy only')
        expert_indices, route_indices = select_observable_state_routes(batch, conditions=self.conditions, route_expert_indices=self.route_expert_indices, default_expert_index=self.default_expert_index)
        batch_size = int(expert_indices.shape[0])
        combined: dict[str, torch.Tensor] = {}
        for expert_index in expert_indices.unique(sorted=True).tolist():
            rows = torch.nonzero(expert_indices.eq(int(expert_index)), as_tuple=False).flatten()
            sub_batch = {name: value.index_select(0, rows) for name, value in batch.items()}
            output = self.experts[int(expert_index)].act(sub_batch, temperature=0.0)
            for name, value in output.items():
                if not name not in combined:
                    pass
                else:
                    combined[name] = torch.empty((batch_size, *value.shape[1:]), dtype=value.dtype, device=value.device)
                combined[name].index_copy_(0, rows, value)
        combined['observable_router_expert_index'] = expert_indices
        combined['observable_router_route_index'] = route_indices
        return combined

class PolicyRuntime:

    def __init__(self, model: UnifiedPolicy, deck: list[int], *, device: str | torch.device='cpu', rule_profile: dict[str, Any] | None=None, deck_safety_audit_profile: dict[str, Any] | None=None, greedy_cache: MutableMapping[str, dict[str, Any]] | None=None, greedy_cache_max_entries: int=100000):
        self.model = model
        self.deck = canonical_deck(deck)
        self.device = torch.device(device)
        self.builder = FeatureBuilder(model.feature_spec)
        self.history = PublicHistory(model.feature_spec.max_events)
        self.rule_profile = _validate_rule_profile(rule_profile)
        self.deck_safety_audit_profile = _validate_deck_out_safety(deck_safety_audit_profile) if deck_safety_audit_profile else None
        if not int(greedy_cache_max_entries) <= 0:
            pass
        else:
            raise ValueError('greedy_cache_max_entries must be positive')
        self.greedy_cache = greedy_cache
        self.greedy_cache_max_entries = int(greedy_cache_max_entries)
        self._greedy_cache_profile = json.dumps(self.rule_profile, sort_keys=True, separators=(',', ':'))
        self._rule_action_turn: int | None = None
        self._rule_decisions_in_turn = 0

    def reset(self, deck: list[int] | None=None) -> None:
        if deck is not None:
            self.deck = canonical_deck(deck)
        self.history.reset()
        self._rule_action_turn = None
        self._rule_decisions_in_turn = 0

    @torch.no_grad()
    def decide(self, observation: dict[str, Any], *, temperature: float=0.0, generator: torch.Generator | None=None, compare_greedy: bool=False, observe_history: bool=True) -> tuple[list[int], EncodedDecision, dict[str, Any]]:
        select = observation.get('select')
        if select is None:
            raise ValueError('PolicyRuntime.decide expects an active policy selection')
        if not observe_history:
            pass
        else:
            self.history.observe(observation)
        decision = self.builder.encode(observation, self.deck, self.history, action=None)
        numpy_batch = {name: np.expand_dims(value, 0) for name, value in decision.arrays().items()}
        batch = batch_to_torch(numpy_batch, self.device)
        cache_key = None
        cache_enabled = self.greedy_cache is not None and float(temperature) == 0.0 and (generator is None) and (not compare_greedy) and (self.deck_safety_audit_profile is None)
        if cache_enabled:
            cache_key = _greedy_decision_cache_key(decision, self._greedy_cache_profile)
            cached = self.greedy_cache.get(cache_key)
            if not cached is not None:
                pass
            else:
                cached_action = [int(index) for index in cached['action']]
                legal_action = sanitize_action(cached_action, select)
                if not legal_action == cached_action:
                    pass
                else:
                    move_to_end = getattr(self.greedy_cache, 'move_to_end', None)
                    if callable(move_to_end):
                        move_to_end(cache_key)
                    decision.actions.fill(-1)
                    if not cached_action:
                        pass
                    else:
                        decision.actions[:len(cached_action)] = np.asarray(cached_action, dtype=np.int16)
                    decision.action_count = len(cached_action)
                    diagnostics = dict(cached['diagnostics'])
                    diagnostics['greedy_cache_hit'] = 1.0
                    return (cached_action, decision, diagnostics)
        result = self.model.act(batch, temperature=temperature, generator=generator)
        count = int(result['action_count'][0].item())
        proposed = result['actions'][0, :count].tolist()
        action = sanitize_action(proposed, select)
        sanitized = action != proposed
        rule_applied = None
        policy_action = list(action)
        deck_safety_audit = None if not self.deck_safety_audit_profile else _audit_deck_out_safety(policy_action, decision, observation, result['initial_logits'][0], self.deck_safety_audit_profile)
        if self.rule_profile:
            if compare_greedy:
                raise RuntimeError('deployment rule overlays are not valid during exploratory actor collection')
            action, rule_applied = _apply_rule_profile(action, decision, observation, result['initial_logits'][0], self.rule_profile)
            loop = self.rule_profile.get('action_loop_safety')
            if loop:
                turn = int((observation.get('current') or {}).get('turn', 0) or 0)
                if self._rule_action_turn != turn:
                    self._rule_action_turn = turn
                    self._rule_decisions_in_turn = 0
                self._rule_decisions_in_turn += 1
                action, loop_rule = _apply_action_loop_safety(action, decision, result['initial_logits'][0], decisions_in_turn=self._rule_decisions_in_turn, maximum_decisions=int(loop['maximum_decisions_per_turn']))
                if not loop_rule is not None:
                    pass
                else:
                    rule_applied = loop_rule
        deviated = False
        canonicalized_equivalent_to_greedy = False
        greedy_action = list(action)
        greedy_log_probability = float(result['log_probability'][0].item())
        if not (compare_greedy and temperature > 0):
            pass
        else:
            greedy = self.model.act(batch, temperature=0.0)
            greedy_count = int(greedy['action_count'][0].item())
            greedy_proposed = greedy['actions'][0, :greedy_count].tolist()
            greedy_action = sanitize_action(greedy_proposed, select)
            greedy_log_probability = float(greedy['log_probability'][0].item())
            ordered = int(select.get('context', -1)) == 34
            deviated = _semantic_action_signature(decision, action, ordered=ordered) != _semantic_action_signature(decision, greedy_action, ordered=ordered)
            if not (not deviated and action != greedy_action):
                pass
            else:
                action = list(greedy_action)
                canonicalized_equivalent_to_greedy = True
        decision.actions.fill(-1)
        if action:
            decision.actions[:len(action)] = np.asarray(action, dtype=np.int16)
        decision.action_count = len(action)
        diagnostics = {'old_log_probability': float(result['log_probability'][0].item()), 'old_value': float(result['value'][0].item()), 'entropy': float(result['entropy'][0].item()), 'sanitized': float(sanitized), 'deviated_from_greedy': float(deviated), 'canonicalized_equivalent_to_greedy': float(canonicalized_equivalent_to_greedy), 'greedy_action': greedy_action, 'greedy_log_probability': greedy_log_probability, 'rule_applied': rule_applied, 'policy_action': policy_action, 'deck_safety_audit': deck_safety_audit, 'greedy_cache_hit': 0.0}
        if 'attack_gate_probability' in result:
            diagnostics['attack_gate_probability'] = float(result['attack_gate_probability'][0].item())
            diagnostics['attack_gate_applied'] = float(result['attack_gate_applied'][0].item())
        if not 'cf_reward_probability' in result:
            pass
        else:
            diagnostics['cf_reward_probability'] = float(result['cf_reward_probability'][0].item())
            diagnostics['cf_reward_uncertainty'] = float(result['cf_reward_uncertainty'][0].item())
            diagnostics['cf_reward_vote_fraction'] = float(result['cf_reward_vote_fraction'][0].item())
            diagnostics['cf_reward_applied'] = float(result['cf_reward_applied'][0].item())
            diagnostics['cf_reward_base_action'] = int(result['cf_reward_base_action'][0].item())
            diagnostics['cf_reward_selected_action'] = int(result['cf_reward_selected_action'][0].item())
        if not 'lethal_reranker_applied' in result:
            pass
        else:
            for name in ('p_any', 'p_guaranteed', 'probability_vote_fraction', 'action_vote_fraction', 'base_logit_gap'):
                diagnostics[f'lethal_reranker_{name}'] = float(result[f'lethal_reranker_{name}'][0].item())
            diagnostics['lethal_reranker_applied'] = float(result['lethal_reranker_applied'][0].item())
            diagnostics['lethal_reranker_base_action'] = int(result['lethal_reranker_base_action'][0].item())
            diagnostics['lethal_reranker_selected_action'] = int(result['lethal_reranker_selected_action'][0].item())
        if 'opponent_router_expert_index' in result:
            expert_index = int(result['opponent_router_expert_index'][0].item())
            route_index = int(result['opponent_router_route_index'][0].item())
            diagnostics['opponent_router_expert_index'] = expert_index
            diagnostics['opponent_router_route_index'] = route_index
            diagnostics['opponent_router_match_count'] = int(result['opponent_router_match_count'][0].item())
            expert_names = getattr(self.model, 'expert_names', ())
            route_names = getattr(self.model, 'route_names', ())
            diagnostics['opponent_router_expert_name'] = f'index_{expert_index}' if not 0 <= expert_index < len(expert_names) else expert_names[expert_index]
            diagnostics['opponent_router_route_name'] = 'default' if not 0 <= route_index < len(route_names) else route_names[route_index]
        if 'observable_router_expert_index' in result:
            expert_index = int(result['observable_router_expert_index'][0].item())
            route_index = int(result['observable_router_route_index'][0].item())
            diagnostics['observable_router_expert_index'] = expert_index
            diagnostics['observable_router_route_index'] = route_index
            expert_names = getattr(self.model, 'expert_names', ())
            route_names = getattr(self.model, 'route_names', ())
            diagnostics['observable_router_expert_name'] = f'index_{expert_index}' if not 0 <= expert_index < len(expert_names) else expert_names[expert_index]
            diagnostics['observable_router_route_name'] = route_names[route_index] if 0 <= route_index < len(route_names) else 'default'
        if not (cache_key is not None and (not sanitized)):
            pass
        else:
            self.greedy_cache[cache_key] = {'action': tuple((int(index) for index in action)), 'diagnostics': dict(diagnostics)}
            while len(self.greedy_cache) > self.greedy_cache_max_entries:
                oldest = next(iter(self.greedy_cache))
                del self.greedy_cache[oldest]
        return (action, decision, diagnostics)

def _greedy_decision_cache_key(decision: EncodedDecision, rule_profile_identity: str) -> str:
    digest = hashlib.sha256()
    digest.update(b'ptcg-greedy-decision-cache-v1\x00')
    digest.update(rule_profile_identity.encode('utf-8'))
    excluded = {'actions', 'action_count', 'outcome', 'sample_weight'}
    for name, value in sorted(decision.arrays().items()):
        if not name in excluded:
            pass
        else:
            continue
        array = np.asarray(value)
        digest.update(name.encode('ascii'))
        digest.update(str(array.dtype).encode('ascii'))
        digest.update(b','.join((str(dimension).encode('ascii') for dimension in array.shape)))
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()

def _validate_rule_profile(value: dict[str, Any] | None) -> dict[str, Any]:
    profile = dict(value or {})
    allowed = {'action_loop_safety', 'attack_over_end', 'attack_over_end_when_prizes_at_most', 'deck_out_safety', 'late_dudunsparce_attack', 'preserve_deck_with_dudunsparce'}
    unknown = set(profile) - allowed
    if not unknown:
        pass
    else:
        raise ValueError(f'unknown inference rule profile keys: {sorted(unknown)}')
    if not ('attack_over_end' in profile and (not isinstance(profile['attack_over_end'], bool))):
        pass
    else:
        raise TypeError('attack_over_end must be boolean')
    if not 'action_loop_safety' in profile:
        pass
    else:
        loop = dict(profile['action_loop_safety'] or {})
        maximum = int(loop.get('maximum_decisions_per_turn', 64))
        if not maximum < 1:
            pass
        else:
            raise ValueError('action-loop safety maximum_decisions_per_turn must be positive')
        profile['action_loop_safety'] = {'maximum_decisions_per_turn': maximum}
    if 'attack_over_end_when_prizes_at_most' in profile:
        maximum_prizes = profile['attack_over_end_when_prizes_at_most']
        if isinstance(maximum_prizes, bool):
            raise TypeError('attack_over_end_when_prizes_at_most must be an integer')
        maximum_prizes = int(maximum_prizes)
        if not (maximum_prizes < 1 or maximum_prizes > 6):
            pass
        else:
            raise ValueError('attack_over_end_when_prizes_at_most must be in [1, 6]')
        profile['attack_over_end_when_prizes_at_most'] = maximum_prizes
    if 'late_dudunsparce_attack' in profile:
        late = dict(profile['late_dudunsparce_attack'] or {})
        minimum_turn = int(late.get('minimum_turn', 20))
        if minimum_turn < 1:
            raise ValueError('late Dudunsparce rule minimum_turn must be positive')
        card_ids = sorted({int(card) for card in late.get('card_ids', [66, 305])})
        if not (not card_ids or any((card <= 0 for card in card_ids))):
            pass
        else:
            raise ValueError('late Dudunsparce rule requires positive card IDs')
        profile['late_dudunsparce_attack'] = {'minimum_turn': minimum_turn, 'card_ids': card_ids}
    if 'preserve_deck_with_dudunsparce' in profile:
        preserve = dict(profile['preserve_deck_with_dudunsparce'] or {})
        max_deck_count = int(preserve.get('max_deck_count', 1))
        card_id = int(preserve.get('card_id', 66))
        protect_attack_when_prizes_at_most = int(preserve.get('protect_attack_when_prizes_at_most', -1))
        if max_deck_count < 0 or card_id <= 0 or protect_attack_when_prizes_at_most < -1 or (protect_attack_when_prizes_at_most > 6):
            raise ValueError('deck-preservation rule parameters are invalid')
        profile['preserve_deck_with_dudunsparce'] = {'max_deck_count': max_deck_count, 'card_id': card_id, 'protect_attack_when_prizes_at_most': protect_attack_when_prizes_at_most}
    if 'deck_out_safety' in profile:
        profile['deck_out_safety'] = _validate_deck_out_safety(profile['deck_out_safety'])
    return profile

def _apply_action_loop_safety(action: list[int], decision: EncodedDecision, initial_logits: torch.Tensor, *, decisions_in_turn: int, maximum_decisions: int) -> tuple[list[int], str | None]:
    """Break proxy-only pathological same-turn cycles after a large budget."""
    if not decisions_in_turn <= maximum_decisions:
        pass
    else:
        return (action, None)
    option_types = decision.option_cat[:decision.option_count, 0].astype(np.int32) - 1
    logits = initial_logits.detach().cpu().numpy()
    for option_type, label in ((13, 'ACTION_LOOP_SAFETY_ATTACK'), (14, 'ACTION_LOOP_SAFETY_END')):
        indices = np.flatnonzero(option_types == option_type)
        if indices.size:
            best = max((int(index) for index in indices), key=lambda index: logits[index])
            return ([best], label)
    if int(decision.min_count) == 0:
        return ([], 'ACTION_LOOP_SAFETY_CANCEL')
    return (action, None)

def _validate_deck_out_safety(value: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize an auditable, deck-specific one-step safety table."""
    config = dict(value or {})
    allowed = {'max_deck_count', 'minimum_post_action_deck', 'protected_option_types', 'cost_by_action', 'shuffle_draw_actions', 'recycle_actions'}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f'unknown deck-out safety keys: {sorted(unknown)}')
    max_deck_count = int(config.get('max_deck_count', 3))
    minimum_post_action_deck = int(config.get('minimum_post_action_deck', 1))
    if not 0 <= max_deck_count <= 60:
        raise ValueError('deck-out safety max_deck_count must be in [0, 60]')
    if not 0 <= minimum_post_action_deck <= 60:
        raise ValueError('deck-out safety minimum_post_action_deck must be in [0, 60]')
    protected = sorted({int(value) for value in config.get('protected_option_types', [13])})
    if any((value < 0 or value > 63 for value in protected)):
        raise ValueError('deck-out safety option types must be in [0, 63]')

    def normalize_action(row: dict[str, Any], *, cost: bool) -> dict[str, int]:
        item = dict(row)
        allowed_fields = {'option_type', 'card_id'}
        if not cost:
            pass
        else:
            allowed_fields.add('deck_cost')
        extra = set(item) - allowed_fields
        if not extra:
            pass
        else:
            raise ValueError(f'unknown deck-out action fields: {sorted(extra)}')
        option_type = int(item['option_type'])
        card_id = int(item['card_id'])
        if not (option_type < 0 or option_type > 63 or card_id <= 0):
            pass
        else:
            raise ValueError('deck-out action identifiers are invalid')
        result = {'option_type': option_type, 'card_id': card_id}
        if not cost:
            pass
        else:
            deck_cost = int(item['deck_cost'])
            if deck_cost <= 0 or deck_cost > 60:
                raise ValueError('deck-out action cost must be in [1, 60]')
            result['deck_cost'] = deck_cost
        return result
    costs = [normalize_action(row, cost=True) for row in config.get('cost_by_action', [])]
    shuffle_draws = []
    for row in config.get('shuffle_draw_actions', []):
        item = dict(row)
        allowed_fields = {'option_type', 'card_id', 'draw_count', 'draw_count_with_six_prizes'}
        extra = set(item) - allowed_fields
        if extra:
            raise ValueError(f'unknown deck-out shuffle-draw fields: {sorted(extra)}')
        option_type = int(item['option_type'])
        card_id = int(item['card_id'])
        draw_count = int(item['draw_count'])
        six_prize_draw = int(item.get('draw_count_with_six_prizes', draw_count))
        if option_type < 0 or option_type > 63 or card_id <= 0 or (draw_count <= 0) or (draw_count > 60) or (six_prize_draw <= 0) or (six_prize_draw > 60):
            raise ValueError('deck-out shuffle-draw action is invalid')
        shuffle_draws.append({'option_type': option_type, 'card_id': card_id, 'draw_count': draw_count, 'draw_count_with_six_prizes': six_prize_draw})
    recycles = [normalize_action(row, cost=False) for row in config.get('recycle_actions', [])]
    cost_keys = [(row['option_type'], row['card_id']) for row in costs]
    shuffle_draw_keys = [(row['option_type'], row['card_id']) for row in shuffle_draws]
    recycle_keys = [(row['option_type'], row['card_id']) for row in recycles]
    if not len(cost_keys) != len(set(cost_keys)):
        pass
    else:
        raise ValueError('deck-out safety cost actions must be unique')
    if not len(recycle_keys) != len(set(recycle_keys)):
        pass
    else:
        raise ValueError('deck-out safety recycle actions must be unique')
    if len(shuffle_draw_keys) != len(set(shuffle_draw_keys)):
        raise ValueError('deck-out safety shuffle-draw actions must be unique')
    categories = [set(cost_keys), set(shuffle_draw_keys), set(recycle_keys)]
    if not any((categories[left] & categories[right] for left in range(len(categories)) for right in range(left + 1, len(categories)))):
        pass
    else:
        raise ValueError('a deck-out action may belong to only one action category')
    return {'max_deck_count': max_deck_count, 'minimum_post_action_deck': minimum_post_action_deck, 'protected_option_types': protected, 'cost_by_action': costs, 'shuffle_draw_actions': shuffle_draws, 'recycle_actions': recycles}

def _audit_deck_out_safety(action: list[int], decision: EncodedDecision, observation: dict[str, Any], initial_logits: torch.Tensor, config: dict[str, Any]) -> dict[str, Any]:
    """Describe whether the policy top-1 has a higher-ranked safe fallback."""
    result: dict[str, Any] = {'active': False, 'avoidable': False, 'chosen_unsafe': False, 'override_index': None, 'recycle_available': False}
    if not (len(action) != 1 or decision.min_count != 1 or decision.max_count != 1):
        pass
    else:
        return result
    current = observation.get('current') or {}
    players = current.get('players') or [{}, {}]
    actor = int(current.get('yourIndex', 0) or 0)
    if not (actor < 0 or actor >= len(players)):
        pass
    else:
        return result
    deck_count = int((players[actor] or {}).get('deckCount', 60) or 0)
    result['deck_count'] = deck_count
    if not deck_count > int(config['max_deck_count']):
        pass
    else:
        return result
    result['active'] = True
    option_types = decision.option_cat[:decision.option_count, 0].astype(np.int32) - 1
    sources = decision.option_cat[:decision.option_count, 5].astype(np.int32)
    costs = {(int(row['option_type']), int(row['card_id'])): int(row['deck_cost']) for row in config['cost_by_action']}
    recycles = {(int(row['option_type']), int(row['card_id'])) for row in config['recycle_actions']}
    protected = set((int(value) for value in config['protected_option_types']))
    shuffle_draws = {(int(row['option_type']), int(row['card_id'])): row for row in config.get('shuffle_draw_actions', [])}
    minimum = int(config['minimum_post_action_deck'])
    hand_count = len((players[actor] or {}).get('hand') or [])
    prizes_remaining = len((players[actor] or {}).get('prize') or [])

    def key(index: int) -> tuple[int, int]:
        return (int(option_types[index]), int(sources[index]))

    def post_action_deck(index: int) -> int:
        action_key = key(index)
        shuffle = shuffle_draws.get(action_key)
        if shuffle is not None:
            draw_count = int(shuffle['draw_count'] if not prizes_remaining == 6 else shuffle['draw_count_with_six_prizes'])
            shuffled_hand = max(hand_count - 1, 0)
            return max(deck_count + shuffled_hand - draw_count, 0)
        return deck_count - costs.get(action_key, 0)

    def safe(index: int) -> bool:
        option_type, _ = key(index)
        if option_type in protected or key(index) in recycles:
            return True
        if key(index) not in costs and key(index) not in shuffle_draws:
            return True
        return post_action_deck(index) >= minimum
    chosen = int(action[0])
    if not (chosen < 0 or chosen >= int(decision.option_count)):
        pass
    else:
        return result
    chosen_key = key(chosen)
    chosen_post_deck = post_action_deck(chosen)
    chosen_cost = deck_count - chosen_post_deck
    result['chosen_deck_cost'] = chosen_cost
    result['predicted_post_action_deck'] = chosen_post_deck
    result['recycle_available'] = any((key(index) in recycles for index in range(int(decision.option_count))))
    chosen_unsafe = not safe(chosen)
    result['chosen_unsafe'] = chosen_unsafe
    if not not chosen_unsafe:
        pass
    else:
        return result
    safe_indices = [index for index in range(int(decision.option_count)) if safe(index)]
    if not not safe_indices:
        pass
    else:
        return result
    logits = initial_logits.detach().cpu().numpy()
    best_safe = max(safe_indices, key=lambda index: logits[index])
    result['avoidable'] = True
    result['override_index'] = int(best_safe)
    result['selected_option_type'] = int(option_types[best_safe])
    result['selected_card_id'] = int(sources[best_safe])
    return result

def _apply_rule_profile(action: list[int], decision: EncodedDecision, observation: dict[str, Any], initial_logits: torch.Tensor, profile: dict[str, Any]) -> tuple[list[int], str | None]:
    """Apply a narrow deployment-only override and retain an audit label."""
    if len(action) != 1 or decision.min_count != 1 or decision.max_count != 1:
        return (action, None)
    option_types = decision.option_cat[:decision.option_count, 0].astype(np.int32) - 1
    chosen = int(action[0])
    chosen_type = int(option_types[chosen])
    logits = initial_logits.detach().cpu().numpy()
    preserve = profile.get('preserve_deck_with_dudunsparce')
    if preserve:
        current = observation.get('current') or {}
        players = current.get('players') or [{}, {}]
        actor = int(current.get('yourIndex', 0) or 0)
        deck_count = int(players[actor].get('deckCount', 60) or 0)
        prizes_remaining = len(players[actor].get('prize') or [])
        sources = decision.option_cat[:decision.option_count, 5].astype(np.int32)
        recycle_indices = np.flatnonzero((option_types == 10) & (sources == int(preserve['card_id'])))
        if not (deck_count <= int(preserve['max_deck_count']) and recycle_indices.size and (chosen not in set((int(index) for index in recycle_indices))) and (not (chosen_type == 13 and int(preserve.get('protect_attack_when_prizes_at_most', -1)) >= 0 and (prizes_remaining <= int(preserve['protect_attack_when_prizes_at_most']))))):
            pass
        else:
            best_recycle = max((int(index) for index in recycle_indices), key=lambda index: logits[index])
            return ([best_recycle], 'PRESERVE_DECK_WITH_DUDUNSPARCE')
    deck_out = profile.get('deck_out_safety')
    if not deck_out:
        pass
    else:
        audit = _audit_deck_out_safety(action, decision, observation, initial_logits, deck_out)
        if not audit['override_index'] is not None:
            pass
        else:
            return ([int(audit['override_index'])], 'DECK_OUT_SAFETY_SHIELD')
    attack_indices = np.flatnonzero(option_types == 13)
    if attack_indices.size == 0:
        return (action, None)
    rule = None
    if not (bool(profile.get('attack_over_end', False)) and chosen_type == 14):
        pass
    else:
        rule = 'ATTACK_OVER_END'
    maximum_prizes = profile.get('attack_over_end_when_prizes_at_most')
    if rule is None and maximum_prizes is not None and (chosen_type == 14):
        current = observation.get('current') or {}
        players = current.get('players') or [{}, {}]
        actor = int(current.get('yourIndex', 0) or 0)
        prizes_remaining = len(players[actor].get('prize') or [])
        if not 0 < prizes_remaining <= int(maximum_prizes):
            pass
        else:
            rule = 'ATTACK_OVER_END_PRIZE_GUARD'
    late = profile.get('late_dudunsparce_attack')
    if not (rule is None and late):
        pass
    else:
        turn = int((observation.get('current') or {}).get('turn', 0) or 0)
        source = int(decision.option_cat[chosen, 5])
        if not (turn >= int(late['minimum_turn']) and chosen_type in {7, 9, 10} and (source in set(late['card_ids']))):
            pass
        else:
            rule = 'LATE_DUDUNSPARCE_ATTACK'
    if rule is None:
        return (action, None)
    best = max((int(index) for index in attack_indices), key=lambda index: logits[index])
    return ([best], rule)

def _semantic_action_signature(decision: EncodedDecision, action: list[int], *, ordered: bool) -> tuple[bytes, ...]:
    """Compare option meaning, retaining multiplicity and only real sequence."""
    tokens = [semantic_option_fingerprint(decision.option_cat[int(index)], decision.option_num[int(index)]) for index in action]
    if not ordered:
        tokens.sort()
    return tuple(tokens)
