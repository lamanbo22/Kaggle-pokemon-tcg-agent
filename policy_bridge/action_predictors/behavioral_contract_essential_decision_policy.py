"""Semantics-complete policy with explicit small set/sequence inductive biases."""
from __future__ import annotations
from dataclasses import asdict
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from policy_bridge.shared_schema_primitives import FEATURE_SEMANTICS_VERSION, RelativeOwner, Zone
from policy_bridge.episode_archive.card_repository_index import CARD_FEATURE_DIM
from policy_bridge.action_predictors.game_card_representation_service import CardEncoder, SetPool
from policy_bridge.action_predictors.shared_decision_policy import UnifiedPolicy, _embedding, _length_mask
from policy_bridge.decision_message_contract import FeatureSpec

class GatedStateFusion(nn.Module):
    """Fuse six semantic summaries through one short gated MLP."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = int(dimension)
        width = self.dimension * 2
        self.content = nn.Sequential(nn.Linear(self.dimension * 6, width), nn.GELU(), nn.Linear(width, self.dimension))
        self.gate = nn.Linear(self.dimension * 6, self.dimension)
        self.norm = nn.LayerNorm(self.dimension)

    def forward(self, *tokens: torch.Tensor) -> torch.Tensor:
        if len(tokens) != 6:
            raise ValueError('semantic state fusion requires six token groups')
        joined = torch.cat(tokens, dim=-1)
        anchor = torch.stack(tokens, dim=0).mean(dim=0)
        return self.norm(anchor + torch.sigmoid(self.gate(joined)) * self.content(joined))

class IndependentOptionConditioner(nn.Module):
    """Condition every legal option independently on the shared public state."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.option_mlp = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.state_projection = nn.Linear(dimension, dimension, bias=False)
        self.gate = nn.Linear(dimension * 2, dimension)
        self.norm = nn.LayerNorm(dimension)

    def forward(self, options: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        transformed = self.option_mlp(options)
        expanded = state.unsqueeze(1).expand_as(transformed)
        gate = torch.sigmoid(self.gate(torch.cat([transformed, expanded], dim=-1)))
        return self.norm(transformed + gate * self.state_projection(expanded))

class SemanticMinimalPolicy(UnifiedPolicy):
    """A small DeepSets/GRU legal-option pointer using every semantics-v3 field.

    The class inherits only the battle-tested autoregressive decoder, equivalent
    option likelihood, legality masks, STOP handling, and deployment ``act``
    method.  Its encoder intentionally has no Transformer or multi-head
    attention module.
    """
    ARCHITECTURE = 'semantic_minimal_v1'

    def __init__(self, model_config: dict[str, Any], feature_spec: FeatureSpec, card_features: np.ndarray | torch.Tensor | None=None) -> None:
        nn.Module.__init__(self)
        self.model_config = dict(model_config)
        self.feature_spec = feature_spec
        architecture = str(self.model_config.get('architecture', ''))
        if architecture != self.ARCHITECTURE:
            raise ValueError(f'SemanticMinimalPolicy requires architecture={self.ARCHITECTURE}')
        d_model = int(self.model_config.get('d_model', 64))
        card_dim = int(self.model_config.get('card_embedding_dim', 32))
        dropout = float(self.model_config.get('dropout', 0.05))
        card_vocab = int(self.model_config.get('card_vocab_size', 1268))
        attack_vocab = int(self.model_config.get('attack_vocab_size', 1557))
        if d_model != 64:
            raise ValueError('the pre-registered semantic-minimal profile uses d_model=64')
        if int(self.model_config.get('event_gru_layers', 1)) != 1:
            raise ValueError('semantic-minimal uses exactly one event GRU layer')
        forbidden = {'transformer_layers', 'attention_heads', 'ffn_dim', 'option_layers', 'hand_adapter_dim'}
        configured_forbidden = sorted(forbidden & self.model_config.keys())
        if configured_forbidden:
            raise ValueError(f'semantic-minimal config contains Transformer-only fields: {configured_forbidden}')
        if not card_features is None:
            pass
        else:
            card_features = torch.zeros(card_vocab, CARD_FEATURE_DIM)
        static = torch.as_tensor(card_features, dtype=torch.float32)
        self.d_model = d_model
        self.card_encoder = CardEncoder(card_vocab, card_dim, d_model, static)
        self.deck_pool = SetPool(d_model)
        self.known_pool = SetPool(d_model)
        self.hand_pool = SetPool(d_model)
        self.board_pool = SetPool(d_model)
        self.global_embeddings = nn.ModuleList([_embedding(size, d_model) for size in (16, 65, 4, 18, 18)])
        self.global_numeric = nn.Sequential(nn.Linear(16, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.global_norm = nn.LayerNorm(d_model)
        self.entity_embeddings = nn.ModuleList([nn.Identity(), _embedding(14, d_model), _embedding(4, d_model), _embedding(16, d_model), _embedding(5, d_model)])
        self.entity_numeric = nn.Sequential(nn.Linear(9, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.entity_norm = nn.LayerNorm(d_model)
        self.event_embeddings = nn.ModuleList([_embedding(257, d_model), _embedding(4, d_model), nn.Identity(), nn.Identity(), _embedding(attack_vocab, d_model), _embedding(16, d_model), _embedding(16, d_model), _embedding(65, d_model)])
        self.event_numeric = nn.Linear(3, d_model)
        self.event_norm = nn.LayerNorm(d_model)
        self.event_gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True, dropout=0.0)
        self.state_fusion = GatedStateFusion(d_model)
        self.option_embeddings = nn.ModuleList([_embedding(65, d_model), _embedding(16, d_model), _embedding(4, d_model), _embedding(16, d_model), _embedding(17, d_model), nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(), _embedding(attack_vocab, d_model), _embedding(65, d_model), _embedding(65, d_model), _embedding(16, d_model), _embedding(16, d_model)])
        self.option_card_roles = nn.Parameter(torch.ones(4, d_model))
        self.option_numeric = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.option_norm = nn.LayerNorm(d_model)
        self.option_conditioner = IndependentOptionConditioner(d_model)
        self.base_scorer = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.decoder_query = nn.Linear(d_model, d_model, bias=False)
        self.option_key = nn.Linear(d_model, d_model, bias=False)
        self.decoder_cell = nn.GRUCell(d_model, d_model)
        self.stop_head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Tanh())

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        global_cat = batch['global_cat'].long()
        global_token = self.global_numeric(batch['global_num'].float())
        for index, embedding in enumerate(self.global_embeddings):
            global_token = global_token + embedding(global_cat[:, index])
        global_token = self.global_norm(global_token)
        own_deck_ids = batch['own_deck'].long()
        deck_token = self.deck_pool(self.card_encoder(own_deck_ids), own_deck_ids.ne(0))
        known_ids = batch['known_cards'].long()
        known_token = self.known_pool(self.card_encoder(known_ids), known_ids.ne(0))
        entity_cat = batch['entity_cat'].long()
        entity_token = self.card_encoder(entity_cat[:, :, 0])
        for index in range(1, len(self.entity_embeddings)):
            entity_token = entity_token + self.entity_embeddings[index](entity_cat[:, :, index])
        entity_token = self.entity_norm(entity_token + self.entity_numeric(batch['entity_num'].float()))
        entity_mask = _length_mask(batch['entity_count'], entity_cat.shape[1])
        entity_token = entity_token.masked_fill(~entity_mask.unsqueeze(-1), 0.0)
        hand_mask = entity_mask & entity_cat[:, :, 1].eq(int(Zone.HAND)) & entity_cat[:, :, 2].eq(int(RelativeOwner.SELF))
        hand_token = self.hand_pool(entity_token, hand_mask)
        board_token = self.board_pool(entity_token, entity_mask & ~hand_mask)
        event_cat = batch['event_cat'].long()
        event_token = self.card_encoder(event_cat[:, :, 2])
        event_token = event_token + self.card_encoder(event_cat[:, :, 3])
        for index in (0, 1, 4, 5, 6, 7):
            event_token = event_token + self.event_embeddings[index](event_cat[:, :, index])
        event_token = self.event_norm(event_token + self.event_numeric(batch['event_num'].float()))
        event_output, _ = self.event_gru(event_token)
        event_count = batch['event_count'].long()
        gather_index = (event_count - 1).clamp_min(0)
        history_token = event_output[torch.arange(event_output.shape[0], device=event_output.device), gather_index]
        history_token = history_token * event_count.gt(0).unsqueeze(-1)
        state = self.state_fusion(global_token, board_token, history_token, deck_token, known_token, hand_token)
        option_cat = batch['option_cat'].long()
        option_token = torch.zeros(*option_cat.shape[:2], self.d_model, device=option_cat.device, dtype=state.dtype)
        for index in (0, 1, 2, 3, 4, 9, 10, 11, 12, 13):
            option_token = option_token + self.option_embeddings[index](option_cat[:, :, index])
        for role, index in enumerate((5, 6, 7, 8)):
            option_token = option_token + self.card_encoder(option_cat[:, :, index]) * self.option_card_roles[role]
        option_token = self.option_norm(option_token + self.option_numeric(batch['option_num'].float()))
        option_token = self.option_conditioner(option_token, state)
        option_mask = _length_mask(batch['option_count'], option_cat.shape[1])
        option_token = option_token.masked_fill(~option_mask.unsqueeze(-1), 0.0)
        return {'state': state, 'options': option_token, 'option_mask': option_mask, 'value': self.value_head(state).squeeze(-1)}

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {'model_config': self.model_config, 'feature_spec': asdict(self.feature_spec), 'feature_semantics_version': FEATURE_SEMANTICS_VERSION, 'policy_architecture': self.ARCHITECTURE}
