"""A narrow supervised gate for top-level attack timing decisions."""
from __future__ import annotations
from typing import Any
import torch
from torch import nn
ATTACK_OPTION_TYPE_ENCODED = 14

class AttackTimingGate(nn.Module):
    """Predict whether an expert attacks now when both branches are legal."""

    def __init__(self, input_dim: int, hidden_dim: int, *, dropout: float=0.1) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError('attack timing gate dimensions must be positive')
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.register_buffer('feature_mean', torch.zeros(self.input_dim))
        self.register_buffer('feature_std', torch.ones(self.input_dim))
        self.network = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(self.hidden_dim, 1))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if tuple(mean.shape) != (self.input_dim,) or tuple(std.shape) != (self.input_dim,):
            raise ValueError('attack timing normalization shape mismatch')
        self.feature_mean.copy_(mean.detach().to(self.feature_mean))
        self.feature_std.copy_(std.detach().to(self.feature_std).clamp_min(1e-05))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features.float() - self.feature_mean) / self.feature_std
        return self.network(normalized).squeeze(-1)

def attack_timing_inputs(batch: dict[str, torch.Tensor], policy_result: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Build gate features and masks from one already-decoded base action.

    The gate is intentionally limited to one-of-N top-level selections.  It
    never reconstructs ordered or unordered multi-select actions.
    """
    option_slots = int(batch['option_cat'].shape[1])
    slots = torch.arange(option_slots, device=batch['option_cat'].device)
    legal = slots.unsqueeze(0).lt(batch['option_count'].long().unsqueeze(1))
    attack = legal & batch['option_cat'][:, :, 0].long().eq(ATTACK_OPTION_TYPE_ENCODED)
    nonattack = legal & ~attack
    eligible = attack.any(dim=1) & nonattack.any(dim=1) & batch['min_count'].long().eq(1) & batch['max_count'].long().eq(1)
    logits = policy_result['initial_logits'][:, :option_slots].float()
    negative = torch.full_like(logits, -torch.inf)
    attack_max = torch.where(attack, logits, negative).max(dim=1).values
    nonattack_max = torch.where(nonattack, logits, negative).max(dim=1).values
    features = torch.cat([policy_result['state'].float(), (attack_max - nonattack_max).unsqueeze(1), attack_max.unsqueeze(1), nonattack_max.unsqueeze(1), batch['global_num'].float()], dim=1)
    return {'features': features, 'eligible': eligible, 'attack_mask': attack, 'nonattack_mask': nonattack, 'option_logits': logits}

def compact_base_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Retain only immutable inference fields from a standard checkpoint."""
    required = {'model_config', 'feature_spec', 'feature_semantics_version', 'model_state'}
    missing = sorted(required - set(checkpoint))
    if not missing:
        pass
    else:
        raise RuntimeError(f'attack gate source checkpoint omits {missing}')
    return {name: checkpoint[name] for name in sorted(required)}
