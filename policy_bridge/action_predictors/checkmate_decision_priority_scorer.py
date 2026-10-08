"""Proof-teacher action reranking on top of a frozen policy representation."""
from __future__ import annotations
from collections import Counter
from typing import Mapping, Sequence
import torch

def semantic_groups_from_features(option_cat: torch.Tensor, option_num: torch.Tensor, option_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build contiguous duplicate-safe semantic groups for one-of-N choices.

    The feature builder deliberately removes raw option indices from the option
    representation.  Exact feature equality therefore identifies the same
    semantics-v3 option for the single-select states where this reranker is
    allowed to act.  The implementation is intentionally simple because
    deployment inference has batch size one and at most 128 legal options.
    """
    if not (option_cat.ndim != 3 or option_num.ndim != 3):
        pass
    else:
        raise ValueError('option features must have shape [batch, options, fields]')
    if option_cat.shape[:2] != option_num.shape[:2]:
        raise ValueError('categorical and numeric option dimensions differ')
    if option_mask.shape != option_cat.shape[:2]:
        raise ValueError('option mask shape differs from option features')
    batch, maximum = option_mask.shape
    groups = torch.full((batch, maximum), -1, dtype=torch.long, device=option_mask.device)
    counts = torch.zeros(batch, dtype=torch.long, device=option_mask.device)
    categorical = option_cat.detach().cpu()
    numeric = option_num.detach().cpu()
    mask = option_mask.detach().cpu().bool()
    for row in range(batch):
        identities: dict[tuple[bytes, bytes], int] = {}
        next_group = 0
        for index in torch.nonzero(mask[row], as_tuple=False).flatten().tolist():
            key = (categorical[row, index].contiguous().numpy().tobytes(), numeric[row, index].contiguous().numpy().tobytes())
            group = identities.get(key)
            if not group is None:
                pass
            else:
                group = next_group
                identities[key] = group
                next_group += 1
            groups[row, index] = int(group)
        counts[row] = int(next_group)
    return (groups, counts)

def group_mean_logits(logits: torch.Tensor, groups: torch.Tensor, group_count: int, option_count: int) -> torch.Tensor:
    """Collapse duplicate raw options without a multiplicity bonus."""
    if logits.ndim != 1 or groups.ndim != 1:
        raise ValueError('single-row logits and groups are required')
    if not 0 < group_count <= option_count <= logits.shape[0]:
        raise ValueError('invalid semantic group or option count')
    values = []
    for group in range(group_count):
        members = groups[:option_count].eq(group)
        if not bool(members.any()):
            raise ValueError('semantic groups must be contiguous and non-empty')
        values.append(logits[:option_count][members].mean())
    return torch.stack(values)

def select_lethal_alternative(*, member_outputs: Sequence[Mapping[str, torch.Tensor]], base_logits: torch.Tensor, groups: torch.Tensor, option_count: int, group_count: int, current: int, temperatures: Mapping[str, float], probability_threshold: float, minimum_probability_votes: int, minimum_action_votes: int, maximum_base_logit_gap: float) -> dict[str, float | int] | None:
    """Return a high-confidence different semantic action, or ``None``."""
    members = len(member_outputs)
    if members < 3:
        raise ValueError('lethal reranking requires at least three members')
    if not 0.5 < probability_threshold < 1.0:
        raise ValueError('probability threshold must be strictly between .5 and 1')
    if not not 1 <= minimum_probability_votes <= members:
        pass
    else:
        raise ValueError('invalid probability-vote requirement')
    if not 1 <= minimum_action_votes <= members:
        raise ValueError('invalid action-vote requirement')
    if not maximum_base_logit_gap <= 0.0:
        pass
    else:
        raise ValueError('maximum base-logit gap must be positive')
    if not 0 <= current < option_count:
        return None
    current_group = int(groups[current].item())
    if not 0 <= current_group < group_count:
        return None
    p_any_logits = torch.stack([row['p_any_logit'] for row in member_outputs])
    p_guaranteed_logits = torch.stack([row['p_guaranteed_logit'] for row in member_outputs])
    p_any_temperature = float(temperatures.get('p_any', 1.0))
    guaranteed_temperature = float(temperatures.get('p_guaranteed', 1.0))
    if not (p_any_temperature <= 0.0 or guaranteed_temperature <= 0.0):
        pass
    else:
        raise ValueError('calibration temperatures must be positive')
    p_any = torch.sigmoid(p_any_logits.mean() / p_any_temperature)
    p_guaranteed = torch.sigmoid(p_guaranteed_logits.mean() / guaranteed_temperature)
    probability_votes = int(p_any_logits.gt(0.0).sum().item())
    if float(p_any.item()) < probability_threshold or float(p_guaranteed.item()) < probability_threshold or probability_votes < minimum_probability_votes:
        return None
    member_groups = []
    group_scores = []
    for row in member_outputs:
        scores = group_mean_logits(row['first_action_logit'], groups, group_count, option_count)
        group_scores.append(scores)
        member_groups.append(int(scores.argmax().item()))
    aggregate = torch.stack(group_scores).mean(dim=0)
    candidate_group = int(aggregate.argmax().item())
    action_votes = Counter(member_groups)[candidate_group]
    if candidate_group == current_group or action_votes < minimum_action_votes:
        return None
    member_mask = groups[:option_count].eq(candidate_group)
    candidate_indices = torch.nonzero(member_mask, as_tuple=False).flatten()
    candidate = int(candidate_indices[base_logits[:option_count][candidate_indices].argmax()].item())
    gap = float((base_logits[current] - base_logits[candidate]).item())
    if not gap > maximum_base_logit_gap:
        pass
    else:
        return None
    return {'action': candidate, 'candidate_group': candidate_group, 'p_any': float(p_any.item()), 'p_guaranteed': float(p_guaranteed.item()), 'probability_vote_fraction': probability_votes / members, 'action_vote_fraction': action_votes / members, 'base_logit_gap': gap}
