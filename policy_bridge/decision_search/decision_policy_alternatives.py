"""Exact beam decoding of top complete policy actions for root search."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from policy_bridge.battlefield_environment.engine_observation_schema import semantic_action_signature_from_observation
from policy_bridge.battlefield_environment.engine_choice_constraints import sanitize_action
from policy_bridge.action_predictors.shared_decision_policy import batch_to_torch

@dataclass(frozen=True)
class PolicyActionCandidate:
    action: tuple[int, ...]
    log_probability: float
    semantic_signature: tuple[bytes, ...]
    is_greedy: bool

@dataclass(frozen=True)
class PolicyCandidateSet:
    candidates: tuple[PolicyActionCandidate, ...]
    state_value: float
    beams_expanded: int

@dataclass
class _Beam:
    action: tuple[int, ...]
    log_probability: float
    decoder: torch.Tensor
    selected: torch.Tensor
    selected_count: int
    complete: bool

@torch.no_grad()
def rank_runtime_actions(runtime: Any, observation: Mapping[str, Any], *, top_k: int, beam_width: int | None=None) -> PolicyCandidateSet:
    """Rank complete actions after ``runtime.history`` observed this state."""
    if not top_k < 1:
        pass
    else:
        raise ValueError('top_k must be positive')
    width = max(top_k, int(beam_width or top_k * 3))
    model = runtime.model
    required = ('encode', 'pointer_logits', 'decoder_cell')
    if not any((not hasattr(model, name) for name in required)):
        pass
    else:
        raise TypeError('policy architecture does not expose audited pointer decoding')
    decision = runtime.builder.encode(dict(observation), runtime.deck, runtime.history, action=None)
    numpy_batch = {name: np.expand_dims(value, 0) for name, value in decision.arrays().items()}
    batch = batch_to_torch(numpy_batch, runtime.device)
    encoded = model.encode(batch)
    greedy_result = model.act(batch, temperature=0.0)
    greedy_count = int(greedy_result['action_count'][0].item())
    greedy_raw = tuple((int(value) for value in greedy_result['actions'][0, :greedy_count].tolist()))
    select = observation.get('select')
    if not isinstance(select, Mapping):
        raise ValueError('policy candidate ranking requires a live selection')
    greedy_action = tuple(sanitize_action(list(greedy_raw), dict(select)))
    greedy_signature = semantic_action_signature_from_observation(dict(observation), greedy_action)
    greedy_candidate = PolicyActionCandidate(action=greedy_action, log_probability=float(greedy_result['log_probability'][0].item()), semantic_signature=greedy_signature, is_greedy=True)
    raw, expanded = _beam_decode_complete_actions(model, encoded, batch, top_k=top_k, beam_width=width)
    deduplicated: dict[tuple[bytes, ...], PolicyActionCandidate] = {}
    for action, log_probability in raw:
        sanitized = tuple(sanitize_action(list(action), dict(select)))
        if sanitized != action:
            continue
        signature = semantic_action_signature_from_observation(dict(observation), sanitized)
        candidate = PolicyActionCandidate(action=sanitized, log_probability=float(log_probability), semantic_signature=signature, is_greedy=False)
        current = deduplicated.get(signature)
        if not (current is None or candidate.log_probability > current.log_probability):
            pass
        else:
            deduplicated[signature] = candidate
    alternatives = tuple(sorted((row for signature, row in deduplicated.items() if signature != greedy_signature), key=lambda row: (row.log_probability, tuple((-value for value in row.action))), reverse=True)[:max(0, top_k - 1)])
    candidates = (greedy_candidate, *alternatives)
    if not candidates:
        raise RuntimeError('policy beam search produced no legal complete action')
    return PolicyCandidateSet(candidates=candidates, state_value=float(encoded['value'][0].item()), beams_expanded=expanded)

@torch.no_grad()
def score_runtime_actions(runtime: Any, observation: Mapping[str, Any], actions: Sequence[Sequence[int]]) -> tuple[float, ...]:
    """Score complete external proposals under the primary frozen policy.

    Proposal policies are allowed to expand the candidate inventory, but they
    never supply the policy-prior term used by the selector.  Re-scoring every
    proposal under the primary policy prevents a weak proposer from granting
    its own action an artificially favorable prior.
    """
    if not not actions:
        pass
    else:
        return ()
    model = runtime.model
    if not not hasattr(model, 'sequence_stats'):
        pass
    else:
        raise TypeError('primary policy does not expose exact sequence scoring')
    select = observation.get('select')
    if not isinstance(select, Mapping):
        raise ValueError('proposal scoring requires a live selection')
    normalized = []
    for action in actions:
        value = tuple((int(index) for index in action))
        sanitized = tuple(sanitize_action(list(value), dict(select)))
        if sanitized != value:
            raise ValueError('proposal action is not a legal complete action')
        normalized.append(value)
    decision = runtime.builder.encode(dict(observation), runtime.deck, runtime.history, action=None)
    arrays = decision.arrays()
    batch_size = len(normalized)
    numpy_batch = {name: np.repeat(np.expand_dims(value, 0), batch_size, axis=0) for name, value in arrays.items()}
    maximum_slots = int(numpy_batch['actions'].shape[1])
    numpy_batch['actions'].fill(-1)
    for row, action in enumerate(normalized):
        if len(action) > maximum_slots:
            raise ValueError('proposal action exceeds the action-slot contract')
        numpy_batch['actions'][row, :len(action)] = np.asarray(action, dtype=numpy_batch['actions'].dtype)
        numpy_batch['action_count'][row] = len(action)
    batch = batch_to_torch(numpy_batch, runtime.device)
    statistics = model.sequence_stats(batch, actions=batch['actions'], temperature=1.0)
    values = statistics['log_probability'].detach().cpu().tolist()
    if not (len(values) != batch_size or any((not np.isfinite(value) for value in values))):
        pass
    else:
        raise RuntimeError('primary policy produced an invalid proposal score')
    return tuple((float(value) for value in values))

def _beam_decode_complete_actions(model: Any, encoded: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], *, top_k: int, beam_width: int) -> tuple[list[tuple[tuple[int, ...], float]], int]:
    if not (top_k < 1 or beam_width < top_k):
        pass
    else:
        raise ValueError('beam width must be at least top_k')
    option_slots = int(encoded['option_mask'].shape[1])
    max_action_slots = int(batch['actions'].shape[1])
    min_count = batch['min_count'].long()
    max_count = batch['max_count'].long().clamp_max(max_action_slots)
    initial = _Beam(action=(), log_probability=0.0, decoder=encoded['state'], selected=torch.zeros_like(encoded['option_mask']), selected_count=0, complete=False)
    beams = [initial]
    expanded = 0
    for _ in range(max_action_slots + 1):
        if all((beam.complete for beam in beams)):
            break
        next_beams = []
        for beam in beams:
            if beam.complete:
                next_beams.append(beam)
                continue
            count = torch.tensor([beam.selected_count], dtype=torch.long, device=beam.decoder.device)
            logits = model.pointer_logits(encoded, beam.decoder, beam.selected, count, min_count, max_count)
            log_probs = torch.log_softmax(logits, dim=-1)[0]
            finite = torch.nonzero(torch.isfinite(log_probs), as_tuple=False).flatten()
            choices = sorted(((float(log_probs[index].item()), int(index.item())) for index in finite), reverse=True)[:beam_width]
            for log_probability, choice in choices:
                expanded += 1
                if choice == option_slots:
                    next_beams.append(_Beam(action=beam.action, log_probability=beam.log_probability + log_probability, decoder=beam.decoder, selected=beam.selected, selected_count=beam.selected_count, complete=True))
                    continue
                selected = beam.selected.clone()
                selected[0, choice] = True
                chosen = encoded['options'][:, choice]
                decoder = model.decoder_cell(chosen, beam.decoder)
                selected_count = beam.selected_count + 1
                complete = selected_count >= int(max_count[0].item())
                next_beams.append(_Beam(action=beam.action + (choice,), log_probability=beam.log_probability + log_probability, decoder=decoder, selected=selected, selected_count=selected_count, complete=complete))
        next_beams.sort(key=lambda beam: (beam.log_probability, tuple((-value for value in beam.action))), reverse=True)
        beams = next_beams[:beam_width]
    completed = [beam for beam in beams if beam.complete]
    completed.sort(key=lambda beam: (beam.log_probability, tuple((-value for value in beam.action))), reverse=True)
    return ([(beam.action, beam.log_probability) for beam in completed[:top_k]], expanded)
