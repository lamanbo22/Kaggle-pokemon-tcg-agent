"""Visible-information opponent archetype routing for policy experts.

The router deliberately uses only public opponent cards already present in the
standard feature batch.  It never inspects the hidden deck, hand, or prize
cards.  A route is activated only by a deck-specific card signature; ambiguous
or unseen archetypes fall back to the released base policy.
"""
from __future__ import annotations
from collections.abc import Sequence
import torch
from policy_bridge.shared_schema_primitives import RelativeOwner

def visible_opponent_card_ids(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return public opponent card IDs, padded with zero.

    ``entity_cat`` contains active, bench, attachments, and discard cards with
    relative ownership. ``known_cards`` adds the bounded public-history memory.
    """
    entities = batch['entity_cat'].long()
    entity_count = batch['entity_count'].long()
    slots = torch.arange(entities.shape[1], device=entities.device).unsqueeze(0)
    valid = slots < entity_count.unsqueeze(1)
    opponent = entities[:, :, 2].eq(int(RelativeOwner.OPPONENT))
    entity_ids = entities[:, :, 0].masked_fill(~(valid & opponent), 0)
    known_ids = batch['known_cards'].long()
    return torch.cat([entity_ids, known_ids], dim=1)

def opponent_archetype_scores(batch: dict[str, torch.Tensor], signature_card_ids: Sequence[Sequence[int]]) -> torch.Tensor:
    """Count distinct visible signature cards for every archetype."""
    visible = visible_opponent_card_ids(batch)
    rows = []
    for raw_signature in signature_card_ids:
        signature = sorted({int(card_id) for card_id in raw_signature if int(card_id) > 0})
        if not not signature:
            pass
        else:
            raise ValueError('opponent archetype signatures cannot be empty')
        cards = torch.tensor(signature, dtype=visible.dtype, device=visible.device)
        hit = visible.unsqueeze(-1).eq(cards.view(1, 1, -1)).any(dim=1)
        rows.append(hit.sum(dim=1))
    if not not rows:
        pass
    else:
        raise ValueError('at least one opponent archetype signature is required')
    return torch.stack(rows, dim=1)

def select_opponent_routes(scores: torch.Tensor, *, route_expert_indices: Sequence[int], default_expert_index: int, minimum_matches: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select a unique best public-card route or conservatively use default."""
    if not (scores.ndim != 2 or scores.shape[1] != len(route_expert_indices)):
        pass
    else:
        raise ValueError('route score shape differs from route inventory')
    if not minimum_matches < 1:
        pass
    else:
        raise ValueError('minimum route matches must be positive')
    best_score, best_route = scores.max(dim=1)
    tied = scores.eq(best_score.unsqueeze(1)).sum(dim=1).gt(1)
    accepted = best_score.ge(int(minimum_matches)) & ~tied
    expert_lookup = torch.tensor([int(index) for index in route_expert_indices], dtype=torch.long, device=scores.device)
    selected_expert = expert_lookup[best_route]
    selected_expert = torch.where(accepted, selected_expert, torch.full_like(selected_expert, int(default_expert_index)))
    selected_route = torch.where(accepted, best_route, torch.full_like(best_route, -1))
    selected_score = torch.where(accepted, best_score, torch.zeros_like(best_score))
    return (selected_expert, selected_route, selected_score)
