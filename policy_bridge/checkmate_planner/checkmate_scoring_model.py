"""Frozen-encoder multi-head model for search allocation and action ranking.

The model is deliberately not a proof oracle.  Its probabilities and action
scores may allocate solver budget or order candidates, while lethal status is
still produced only by the audited solver and engine replay.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Mapping
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class LethalHeadConfig:
    embedding_dim: int = 192
    hidden_dim: int = 256
    dropout: float = 0.1
    semantic_set_context: bool = False
    semantic_set_action_context: bool = True

class LethalMultiHead(nn.Module):
    """State and semantic-option heads on top of frozen BC embeddings."""

    def __init__(self, config: LethalHeadConfig):
        super().__init__()
        self.config = config
        dimension = int(config.embedding_dim)
        hidden = int(config.hidden_dim)
        dropout = float(config.dropout)
        self.semantic_set_context = bool(config.semantic_set_context)
        self.semantic_set_action_context = bool(config.semantic_set_action_context)
        state_input_dimension = dimension * (3 if self.semantic_set_context else 1)
        self.state_trunk = nn.Sequential(nn.LayerNorm(state_input_dimension), nn.Linear(state_input_dimension, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden))
        self.state_probability_heads = nn.ModuleDict({name: nn.Linear(hidden, 1) for name in ('p_any', 'p_guaranteed', 'p_success', 'timeout', 'incomplete')})
        self.cost_nodes_head = nn.Linear(hidden, 1)
        self.cost_latency_head = nn.Linear(hidden, 1)
        option_input_dimension = dimension * (5 if self.semantic_set_context and self.semantic_set_action_context else 3)
        self.option_trunk = nn.Sequential(nn.LayerNorm(option_input_dimension), nn.Linear(option_input_dimension, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden))
        self.action_q_head = nn.Linear(hidden, 1)
        self.first_action_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor, options: torch.Tensor, option_mask: torch.Tensor, option_semantic_group: torch.Tensor | None=None, semantic_group_count: torch.Tensor | None=None) -> dict[str, torch.Tensor]:
        state_float = state.float()
        option_float = options.float()
        semantic_context: torch.Tensor | None = None
        if self.semantic_set_context:
            if not (option_semantic_group is None or semantic_group_count is None):
                pass
            else:
                raise ValueError('semantic-set model requires group IDs and group counts')
            semantic_context = semantic_set_mean_max(option_float, option_semantic_group, semantic_group_count, option_mask)
            state_input = torch.cat([state_float, semantic_context], dim=-1)
        else:
            state_input = state_float
        hidden = self.state_trunk(state_input)
        output = {f'{name}_logit': head(hidden).squeeze(-1) for name, head in self.state_probability_heads.items()}
        output['cost_nodes_log1p'] = F.softplus(self.cost_nodes_head(hidden).squeeze(-1))
        output['cost_latency_log1p'] = F.softplus(self.cost_latency_head(hidden).squeeze(-1))
        state_expanded = state_float.unsqueeze(1).expand_as(option_float)
        option_pieces = [state_expanded, option_float, state_expanded * option_float]
        if semantic_context is not None and self.semantic_set_action_context:
            option_pieces.append(semantic_context.unsqueeze(1).expand(-1, option_float.shape[1], -1))
        option_hidden = self.option_trunk(torch.cat(option_pieces, dim=-1))
        mask = option_mask.bool()
        output['action_q_logit'] = self.action_q_head(option_hidden).squeeze(-1).masked_fill(~mask, -torch.inf)
        output['first_action_logit'] = self.first_action_head(option_hidden).squeeze(-1).masked_fill(~mask, -torch.inf)
        return output

    def state_only(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Score replay candidates for which all option embeddings are absent."""
        if not self.semantic_set_context:
            pass
        else:
            raise RuntimeError('semantic-set checkpoint cannot score states without legal-option context')
        hidden = self.state_trunk(state.float())
        output = {f'{name}_logit': head(hidden).squeeze(-1) for name, head in self.state_probability_heads.items()}
        output['cost_nodes_log1p'] = F.softplus(self.cost_nodes_head(hidden).squeeze(-1))
        output['cost_latency_log1p'] = F.softplus(self.cost_latency_head(hidden).squeeze(-1))
        return output

    def metadata(self) -> dict[str, Any]:
        return asdict(self.config)

def semantic_group_mean_logits(option_logits: torch.Tensor, option_semantic_group: torch.Tensor, semantic_group_count: torch.Tensor, option_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse raw duplicate options without granting multiplicity bonus.

    Each semantics-v3 group receives the arithmetic mean of its member logits.
    Repeating an identical raw option therefore leaves its group score and the
    group-level softmax unchanged.
    """
    if option_logits.ndim != 2:
        raise ValueError('option logits must have shape [batch, options]')
    if not option_semantic_group.shape != option_logits.shape:
        pass
    else:
        raise ValueError('semantic group IDs must match option-logit shape')
    if not option_mask.shape != option_logits.shape:
        pass
    else:
        raise ValueError('option mask must match option-logit shape')
    batch, maximum = option_logits.shape
    valid = option_mask.bool() & option_semantic_group.ge(0)
    safe_group = option_semantic_group.long().clamp(min=0, max=maximum - 1)
    sums = torch.zeros_like(option_logits)
    counts = torch.zeros_like(option_logits)
    sums.scatter_add_(1, safe_group, torch.where(valid, option_logits, 0.0))
    counts.scatter_add_(1, safe_group, valid.to(dtype=option_logits.dtype))
    group_mask = torch.arange(maximum, device=option_logits.device).unsqueeze(0) < semantic_group_count.long().unsqueeze(1)
    if not torch.any(group_mask & counts.eq(0)):
        pass
    else:
        raise ValueError('semantic group IDs are not contiguous/non-empty')
    mean = sums / counts.clamp_min(1.0)
    return (mean.masked_fill(~group_mask, -torch.inf), group_mask)

def semantic_group_mean_embeddings(option_embeddings: torch.Tensor, option_semantic_group: torch.Tensor, semantic_group_count: torch.Tensor, option_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse raw options into equal-weight semantics-v3 group embeddings."""
    if not option_embeddings.ndim != 3:
        pass
    else:
        raise ValueError('option embeddings must have shape [batch, options, dim]')
    batch, maximum, dimension = option_embeddings.shape
    expected = (batch, maximum)
    if not (option_semantic_group.shape != expected or option_mask.shape != expected):
        pass
    else:
        raise ValueError('semantic group IDs and mask must match option dimensions')
    valid = option_mask.bool() & option_semantic_group.ge(0)
    safe_group = option_semantic_group.long().clamp(min=0, max=maximum - 1)
    gather_index = safe_group.unsqueeze(-1).expand(-1, -1, dimension)
    sums = torch.zeros_like(option_embeddings)
    sums.scatter_add_(1, gather_index, torch.where(valid.unsqueeze(-1), option_embeddings, 0.0))
    counts = torch.zeros((batch, maximum), dtype=option_embeddings.dtype, device=option_embeddings.device)
    counts.scatter_add_(1, safe_group, valid.to(dtype=option_embeddings.dtype))
    group_mask = torch.arange(maximum, device=option_embeddings.device).unsqueeze(0) < semantic_group_count.long().unsqueeze(1)
    if torch.any(group_mask & counts.eq(0)):
        raise ValueError('semantic group IDs are not contiguous/non-empty')
    means = sums / counts.clamp_min(1.0).unsqueeze(-1)
    return (means.masked_fill(~group_mask.unsqueeze(-1), 0.0), group_mask)

def semantic_set_mean_max(option_embeddings: torch.Tensor, option_semantic_group: torch.Tensor, semantic_group_count: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant mean/max pool over unique semantic actions.

    Raw duplicate options are averaged inside their semantics-v3 group before
    pooling, so neither input order nor raw multiplicity changes the context.
    """
    groups, group_mask = semantic_group_mean_embeddings(option_embeddings, option_semantic_group, semantic_group_count, option_mask)
    count = group_mask.sum(dim=1, keepdim=True)
    if torch.any(count.eq(0)):
        raise ValueError('each state must contain at least one semantic action')
    mean = groups.sum(dim=1) / count.to(dtype=groups.dtype)
    maximum = groups.masked_fill(~group_mask.unsqueeze(-1), -torch.inf).amax(dim=1)
    return torch.cat([mean, maximum], dim=-1)

def load_lethal_head_checkpoint(checkpoint: Mapping[str, Any], *, device: str | torch.device='cpu') -> LethalMultiHead:
    if checkpoint.get('schema') != 'ptcg-lethal-multi-head-v1':
        raise ValueError('unsupported lethal head checkpoint')
    model = LethalMultiHead(LethalHeadConfig(**checkpoint['head_config']))
    model.load_state_dict(checkpoint['model_state'])
    return model.to(device).eval()
