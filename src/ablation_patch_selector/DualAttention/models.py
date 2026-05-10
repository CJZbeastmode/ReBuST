"""Raza dual attention models adapted for embedding inputs."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class SoftAttentionEmbedding(nn.Module):
    """Soft attention over patch embeddings."""

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, N, D]
        mask: [B, N] bool
        """
        scores = self.mlp(embeddings).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=1)
        return attn


class RazaHardAttention(nn.Module):
    """Hard attention policy with LSTM and classification head."""

    def __init__(
        self,
        embed_dim: int = 512,
        coord_dim: int = 2,
        hidden_dim: int = 256,
        num_classes: int = 2,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.coord_dim = int(coord_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)

        self.cand_mlp = nn.Sequential(
            nn.Linear(self.embed_dim + self.coord_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.context_mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + self.coord_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.glimpse_mlp = nn.Sequential(
            nn.Linear(self.embed_dim + self.coord_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.rnn = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.value_head = nn.Linear(self.hidden_dim, 1)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
        self.register_buffer("logit_bias", torch.zeros(self.num_classes))

    def build_candidate_features(
        self,
        embeddings: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        return self.cand_mlp(torch.cat([embeddings, coords], dim=-1))

    def init_state(self, embeddings: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_embed = embeddings.mean(dim=1)
        std_embed = embeddings.std(dim=1)
        mean_coord = coords.mean(dim=1)
        std_coord = coords.std(dim=1)
        context = self.context_mlp(torch.cat([mean_embed, std_embed, mean_coord, std_coord], dim=-1))
        h0 = context
        c0 = torch.zeros_like(h0)
        return h0, c0

    def policy_logits(
        self,
        hidden: torch.Tensor,
        candidate_features: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_candidates, _ = candidate_features.shape
        hidden_expand = hidden.unsqueeze(1).expand(batch_size, num_candidates, -1)
        logits = self.policy_head(torch.cat([candidate_features, hidden_expand], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(selected_mask, torch.finfo(logits.dtype).min)
        return logits

    def step(
        self,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        glimpse_embed: torch.Tensor,
        glimpse_coord: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        glimpse_feat = self.glimpse_mlp(torch.cat([glimpse_embed, glimpse_coord], dim=-1))
        hidden, cell = self.rnn(glimpse_feat, (hidden, cell))
        return hidden, cell

    def set_logit_adjustment(self, class_counts: torch.Tensor, tau: float) -> None:
        counts = class_counts.float().clamp(min=1.0)
        priors = counts / counts.sum()
        self.logit_bias = (-tau * torch.log(priors)).to(self.classifier.weight.device)

    def classify(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden) + self.logit_bias

    def value(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden).squeeze(-1)
