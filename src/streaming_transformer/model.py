"""Module for model."""

import torch
import torch.nn as nn
from typing import List, Tuple


class StreamingMILTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        model_dim: int,
        num_classes: int,
        patch_chunk_size: int = 128,
        local_num_heads: int = 8,
        local_num_layers: int = 2,
        local_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.patch_chunk_size = patch_chunk_size
        self.patch_proj = nn.Linear(embed_dim, model_dim)

        self.coord_mlp = nn.Sequential(
            nn.Linear(3, model_dim), nn.ReLU(), nn.Linear(model_dim, model_dim)
        )

        # Gated attention pooling (Ilse et al.)
        self.attn_v = nn.Linear(model_dim, model_dim)
        self.attn_u = nn.Linear(model_dim, model_dim)
        self.attn_w = nn.Linear(model_dim, 1)
        self.attn_dropout = nn.Dropout(local_dropout)

        local_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=local_num_heads,
            dim_feedforward=model_dim * 4,
            dropout=local_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.local_encoder = nn.TransformerEncoder(
            local_layer, num_layers=local_num_layers
        )

        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(local_dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.classifier = nn.Linear(model_dim, num_classes)
        self.register_buffer("logit_bias", torch.zeros(num_classes))

    def set_logit_adjustment(self, class_counts: torch.Tensor, tau: float) -> None:
        counts = class_counts.float()
        counts = torch.clamp(counts, min=1.0)
        priors = counts / counts.sum()
        bias = -tau * torch.log(priors)
        self.logit_bias = bias

    def _pool_single_wsi(
        self,
        patches: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        if patches.dim() == 1:
            patches = patches.unsqueeze(0)
        if patches.numel() == 0:
            patches = torch.zeros(
                1, self.embed_dim, device=patches.device, dtype=patches.dtype
            )
        if coords.numel() == 0:
            coords = torch.zeros(1, 3, device=patches.device)

        coords_norm = coords.clone()
        coords_norm[:, 0] /= coords_norm[:, 0].max().clamp(min=1e-6)
        coords_norm[:, 1] /= coords_norm[:, 1].max().clamp(min=1e-6)
        coords_norm[:, 2] /= coords_norm[:, 2].max().clamp(min=1e-6)

        patches = patches.float()

        num_patches = patches.shape[0]
        max_score = None
        denom = torch.tensor(0.0, device=patches.device)
        numer = torch.zeros(self.model_dim, device=patches.device)

        for start in range(0, num_patches, self.patch_chunk_size):
            end = min(start + self.patch_chunk_size, num_patches)
            chunk = patches[start:end]

            chunk_feats = self.patch_proj(chunk)  # [chunk, D]
            chunk_pos = self.coord_mlp(coords_norm[start:end])  # [chunk, D]
            chunk_tokens = chunk_feats + 0.5 * chunk_pos

            encoded = self.local_encoder(chunk_tokens.unsqueeze(0)).squeeze(0)
            encoded = self.attn_dropout(encoded)

            v = torch.tanh(self.attn_v(encoded))
            u = torch.sigmoid(self.attn_u(encoded))
            scores = self.attn_w(v * u).squeeze(-1)  # [chunk]

            chunk_max = scores.max()
            if max_score is None:
                max_score = chunk_max
            else:
                if chunk_max > max_score:
                    scale = torch.exp(max_score - chunk_max)
                    denom = denom * scale
                    numer = numer * scale
                    max_score = chunk_max

            weights = torch.exp(scores - max_score)
            denom = denom + weights.sum()
            numer = numer + (weights.unsqueeze(-1) * encoded).sum(dim=0)

        if denom.item() == 0.0:
            return torch.zeros(self.model_dim, device=patches.device)
        return numer / denom

    def encode_wsi_batch(
        self,
        patch_batches: List[torch.Tensor],
        coord_batches: List[torch.Tensor],
    ) -> torch.Tensor:
        local_tokens = []
        for patches, coords in zip(patch_batches, coord_batches):
            token = self._pool_single_wsi(patches, coords)
            local_tokens.append(token)

        tokens = torch.stack(local_tokens, dim=0)
        return tokens

    def forward(
        self,
        patch_batches: List[torch.Tensor],
        coord_batches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encode_wsi_batch(patch_batches, coord_batches)
        features = self.head(self.norm(tokens))
        logits = self.classifier(features) + self.logit_bias
        return logits, tokens
