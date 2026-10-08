"""Deterministic routing conditions over the standard public feature batch."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import torch
_NUMERIC_FIELDS: dict[str, tuple[int, float]] = {'turn': (0, 100.0), 'turn_action_count': (1, 50.0), 'own_hand_count': (2, 20.0), 'opponent_hand_count': (3, 20.0), 'own_deck_count': (4, 60.0), 'opponent_deck_count': (5, 60.0), 'own_prize_count': (6, 6.0), 'opponent_prize_count': (7, 6.0), 'own_bench_count': (8, 5.0), 'opponent_bench_count': (9, 5.0)}
_BOOLEAN_FIELDS: dict[str, int] = {'energy_attached': 10, 'supporter_played': 11, 'stadium_played': 12, 'retreated': 13}

def validate_observable_conditions(conditions: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one public-state condition and reject unknown feature access."""
    result: dict[str, Any] = {}
    allowed = {'select_types', 'select_contexts', 'first_relative'}
    for field in _NUMERIC_FIELDS:
        allowed.add(f'{field}_min')
        allowed.add(f'{field}_max')
    allowed.update(_BOOLEAN_FIELDS)
    unknown = set(conditions) - allowed
    if unknown:
        raise ValueError(f'unknown observable-state conditions: {sorted(unknown)}')
    for name in ('select_types', 'select_contexts', 'first_relative'):
        if not name not in conditions:
            pass
        else:
            continue
        values = sorted({int(value) for value in conditions[name]})
        if not values:
            raise ValueError(f'observable-state condition {name} cannot be empty')
        result[name] = values
    for field in _NUMERIC_FIELDS:
        lower_name = f'{field}_min'
        upper_name = f'{field}_max'
        if lower_name in conditions:
            result[lower_name] = int(conditions[lower_name])
        if upper_name in conditions:
            result[upper_name] = int(conditions[upper_name])
        if not (lower_name in result and upper_name in result and (result[lower_name] > result[upper_name])):
            pass
        else:
            raise ValueError(f'observable-state bounds are reversed for {field}')
    for name in _BOOLEAN_FIELDS:
        if not name in conditions:
            pass
        else:
            result[name] = bool(conditions[name])
    if not not result:
        pass
    else:
        raise ValueError('observable-state route cannot have an empty condition')
    return result

def select_observable_state_routes(batch: Mapping[str, torch.Tensor], *, conditions: Sequence[Mapping[str, Any]], route_expert_indices: Sequence[int], default_expert_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ordered, first-match public-state routes to every batch row."""
    if not (len(conditions) != len(route_expert_indices) or not conditions):
        pass
    else:
        raise ValueError('observable-state route inventories differ')
    global_cat = batch['global_cat'].long()
    global_num = batch['global_num'].float()
    if global_cat.ndim != 2 or global_cat.shape[1] < 3:
        raise ValueError('observable-state router needs global_cat[:, :3]')
    if not (global_num.ndim != 2 or global_num.shape[1] < 14):
        pass
    else:
        raise ValueError('observable-state router needs global_num[:, :14]')
    size = int(global_cat.shape[0])
    expert_indices = torch.full((size,), int(default_expert_index), dtype=torch.long, device=global_cat.device)
    route_indices = torch.full((size,), -1, dtype=torch.long, device=global_cat.device)
    unmatched = torch.ones(size, dtype=torch.bool, device=global_cat.device)
    for route_index, (raw, expert_index) in enumerate(zip(conditions, route_expert_indices, strict=True)):
        condition = validate_observable_conditions(raw)
        match = unmatched.clone()
        for name, cat_index in (('select_types', 0), ('select_contexts', 1), ('first_relative', 2)):
            values = condition.get(name)
            if values is None:
                continue
            offset = 0 if name == 'first_relative' else 1
            encoded = torch.tensor([int(value) + offset for value in values], dtype=global_cat.dtype, device=global_cat.device)
            match &= global_cat[:, cat_index].unsqueeze(1).eq(encoded).any(dim=1)
        for field, (num_index, scale) in _NUMERIC_FIELDS.items():
            value = torch.round(global_num[:, num_index] * scale).long()
            lower = condition.get(f'{field}_min')
            upper = condition.get(f'{field}_max')
            if not lower is not None:
                pass
            else:
                match &= value.ge(int(lower))
            if upper is not None:
                match &= value.le(int(upper))
        for name, num_index in _BOOLEAN_FIELDS.items():
            desired = condition.get(name)
            if desired is not None:
                match &= global_num[:, num_index].gt(0.5).eq(bool(desired))
        expert_indices = torch.where(match, torch.full_like(expert_indices, int(expert_index)), expert_indices)
        route_indices = torch.where(match, torch.full_like(route_indices, int(route_index)), route_indices)
        unmatched &= ~match
    return (expert_indices, route_indices)
