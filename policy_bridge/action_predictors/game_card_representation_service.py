from __future__ import annotations
import torch
from torch import nn

class CardEncoder(nn.Module):
    """Card ID plus structured metadata, with ID 0 reserved for padding."""

    def __init__(self, vocab_size: int, embedding_dim: int, output_dim: int, static_features: torch.Tensor):
        super().__init__()
        if static_features.ndim != 2 or static_features.shape[0] != vocab_size:
            raise ValueError(f'static card features must be [{vocab_size}, F], got {tuple(static_features.shape)}')
        self.id_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.register_buffer('static_features', static_features.float(), persistent=True)
        self.static_projection = nn.Sequential(nn.Linear(static_features.shape[1], embedding_dim), nn.GELU(), nn.Linear(embedding_dim, embedding_dim))
        self.output = nn.Sequential(nn.Linear(embedding_dim * 2, output_dim), nn.LayerNorm(output_dim))

    def forward(self, card_ids: torch.Tensor) -> torch.Tensor:
        ids = card_ids.to(dtype=torch.long).clamp(0, self.id_embedding.num_embeddings - 1)
        learned = self.id_embedding(ids)
        structured = self.static_projection(self.static_features[ids])
        output = self.output(torch.cat([learned, structured], dim=-1))
        return output * ids.ne(0).unsqueeze(-1)

class SetPool(nn.Module):
    """Permutation-invariant DeepSets mean/max pooling."""

    def __init__(self, dimension: int):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension), nn.LayerNorm(dimension))
        self.rho = nn.Sequential(nn.Linear(dimension * 2 + 1, dimension), nn.GELU(), nn.Linear(dimension, dimension), nn.LayerNorm(dimension))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        transformed = self.phi(values)
        valid = mask.unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1)
        mean = (transformed * valid).sum(dim=1) / count
        minimum = torch.finfo(transformed.dtype).min
        maximum = transformed.masked_fill(~valid, minimum).max(dim=1).values
        maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        log_count = count.to(transformed.dtype).log1p() / 5.0
        return self.rho(torch.cat([mean, maximum, log_count], dim=-1))
