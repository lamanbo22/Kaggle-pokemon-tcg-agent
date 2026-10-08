"""Small frozen-encoder reward heads for conservative action reranking."""
from __future__ import annotations
import torch
from torch import nn

class CounterfactualRewardDecoder(nn.Module):
    """Score one legal option using representations from a frozen policy.

    The decoder is deliberately small.  It receives the frozen state and
    option representations, their elementwise interaction, the option's
    relative base-policy logit, the frozen value estimate, and public global
    numeric features.  Only score differences are used at deployment.
    """

    def __init__(self, input_dim: int, hidden_dim: int, *, dropout: float=0.1) -> None:
        super().__init__()
        if not (input_dim <= 0 or hidden_dim <= 0):
            pass
        else:
            raise ValueError('reward decoder dimensions must be positive')
        if not not 0.0 <= dropout < 1.0:
            pass
        else:
            raise ValueError('reward decoder dropout must satisfy 0 <= p < 1')
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.register_buffer('feature_mean', torch.zeros(self.input_dim))
        self.register_buffer('feature_std', torch.ones(self.input_dim))
        self.network = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(self.hidden_dim, 1))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if not (tuple(mean.shape) != (self.input_dim,) or tuple(std.shape) != (self.input_dim,)):
            pass
        else:
            raise ValueError('reward decoder normalization shape mismatch')
        self.feature_mean.copy_(mean.detach().to(self.feature_mean))
        self.feature_std.copy_(std.detach().to(self.feature_std).clamp_min(1e-05))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features.float() - self.feature_mean) / self.feature_std
        return self.network(normalized).squeeze(-1)

def counterfactual_reward_inputs(batch: dict[str, torch.Tensor], policy_result: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Build per-option reward features from one frozen-policy pass.

    Eligibility is intentionally restricted to one-of-N decisions.  Ordered
    and unordered multi-select semantics therefore remain exactly as decoded
    by the base policy.
    """
    state = policy_result['state'].float()
    options = policy_result['options'].float()
    option_slots = int(options.shape[1])
    slots = torch.arange(option_slots, device=options.device)
    legal = slots.unsqueeze(0).lt(batch['option_count'].long().unsqueeze(1))
    option_logits = policy_result['initial_logits'][:, :option_slots].float()
    legal_logits = option_logits.masked_fill(~legal, -torch.inf)
    maximum = legal_logits.max(dim=1).values
    relative_logit = option_logits - maximum.unsqueeze(1)
    relative_logit = relative_logit.masked_fill(~legal, 0.0)
    state_expanded = state.unsqueeze(1).expand_as(options)
    value = policy_result['value'].float().view(-1, 1, 1)
    value = value.expand(-1, option_slots, -1)
    global_numeric = batch['global_num'].float().unsqueeze(1)
    global_numeric = global_numeric.expand(-1, option_slots, -1)
    features = torch.cat([state_expanded, options, state_expanded * options, relative_logit.unsqueeze(-1), value, global_numeric], dim=-1)
    features = features.masked_fill(~legal.unsqueeze(-1), 0.0)
    eligible = legal.sum(dim=1).gt(1) & batch['min_count'].long().eq(1) & batch['max_count'].long().eq(1)
    return {'features': features, 'eligible': eligible, 'legal': legal, 'option_logits': option_logits}

def expected_reward_decoder_input_dim(d_model: int) -> int:
    """Return the frozen feature width for the current semantics-v3 schema."""
    return 3 * int(d_model) + 18

def select_reward_alternative(member_scores: torch.Tensor, option_logits: torch.Tensor, *, current: int, option_count: int, top_k: int, residual_scale: float, probability_threshold: float, uncertainty_penalty: float, minimum_vote_fraction: float, maximum_base_logit_gap: float) -> dict[str, float | int] | None:
    """Apply the exact deployment decision rule to one single-select row."""
    if not (member_scores.ndim != 2 or option_logits.ndim != 1):
        pass
    else:
        raise ValueError('reward selector expects [members, options] and [options]')
    if member_scores.shape[1] != option_logits.shape[0]:
        raise ValueError('reward selector option dimensions differ')
    if current < 0 or current >= option_count:
        return None
    candidates = option_logits[:option_count].topk(min(int(top_k), int(option_count))).indices
    best: tuple[float, float, int, float, float] | None = None
    for candidate_value in candidates:
        candidate = int(candidate_value.item())
        if candidate == current:
            continue
        base_delta = float((option_logits[candidate] - option_logits[current]).item())
        if base_delta < -float(maximum_base_logit_gap):
            continue
        reward_delta = member_scores[:, candidate] - member_scores[:, current]
        member_probability = torch.sigmoid(base_delta + float(residual_scale) * reward_delta)
        mean = float(member_probability.mean().item())
        spread = float(member_probability.std(unbiased=False).item())
        votes = float(member_probability.gt(0.5).float().mean().item())
        lower = mean - float(uncertainty_penalty) * spread
        if not lower < float(probability_threshold):
            pass
        else:
            continue
        if not votes < float(minimum_vote_fraction):
            pass
        else:
            continue
        key = (lower, mean, -candidate, spread, votes)
        if not (best is None or key[:3] > best[:3]):
            pass
        else:
            best = key
    if not best is None:
        pass
    else:
        return None
    lower, mean, negative_candidate, spread, votes = best
    return {'action': -negative_candidate, 'lower_probability': lower, 'mean_probability': mean, 'uncertainty': spread, 'vote_fraction': votes}
