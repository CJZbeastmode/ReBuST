"""
Model definition for the streaming MIL transformer architecture.
"""

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
        """
        Initialize the StreamingMILTransformer.

        Args:
            embed_dim: Dimension of input patch embeddings.
            model_dim: Dimension of model features and tokens.
            num_classes: Number of output classes for classification.
            patch_chunk_size: Number of patches to process at once for memory efficiency.
            local_num_heads: Number of attention heads in the local transformer encoder.
            local_num_layers: Number of layers in the local transformer encoder.
            local_dropout: Dropout rate for local transformer and attention.
        """

        super().__init__()

        # model hyperparameters
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.patch_chunk_size = patch_chunk_size

        # projection and positional encoding
        self.patch_proj = nn.Linear(embed_dim, model_dim)
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, model_dim), nn.ReLU(), nn.Linear(model_dim, model_dim)
        )

        # gated attention pooling
        self.attn_v = nn.Linear(model_dim, model_dim)
        self.attn_u = nn.Linear(model_dim, model_dim)
        self.attn_w = nn.Linear(model_dim, 1)
        self.attn_dropout = nn.Dropout(local_dropout)

        # local transformer encoder
        local_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,  # projected embedding size
            nhead=local_num_heads,  # numbers of attention heads
            dim_feedforward=model_dim * 4,  # *4 default transformer heuristic
            dropout=local_dropout,
            activation="gelu",
            batch_first=True,  # input: [batch, seq, dim]
        )
        self.local_encoder = nn.TransformerEncoder(
            local_layer, num_layers=local_num_layers
        )

        # final classification head
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(local_dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.classifier = nn.Linear(model_dim, num_classes)

        # logit adjustment buffer
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
        """
        Pool a single WSI's worth of patches into a single bag-level token using gated attention.

        Args:
            patches: [num_patches, embed_dim] tensor of patch embeddings
            coords: [num_patches, 3] tensor of patch coordinates (x,y,level)

        Returns:
            pooled_token: [model_dim] tensor representing the pooled WSI token
        """

        # Preprocess
        if patches.dim() == 1:
            patches = patches.unsqueeze(0)
        if patches.numel() == 0:
            patches = torch.zeros(
                1, self.embed_dim, device=patches.device, dtype=patches.dtype
            )
        if coords.numel() == 0:
            coords = torch.zeros(1, 3, device=patches.device)

        # Normalize coordinates to [0,1]
        coords_norm = coords.clone()
        coords_norm[:, 0] /= coords_norm[:, 0].max().clamp(min=1e-6)
        coords_norm[:, 1] /= coords_norm[:, 1].max().clamp(min=1e-6)
        coords_norm[:, 2] /= coords_norm[:, 2].max().clamp(min=1e-6)

        # Project patches and pool with gated attention
        patches = patches.float()
        num_patches = patches.shape[0]
        max_score = None
        denom = torch.tensor(0.0, device=patches.device)
        numer = torch.zeros(self.model_dim, device=patches.device)

        # loop over patch chunks
        for start in range(0, num_patches, self.patch_chunk_size):
            # get chunk
            end = min(start + self.patch_chunk_size, num_patches)
            chunk = patches[start:end]

            # project and add positional encoding
            chunk_feats = self.patch_proj(chunk)  # [chunk, D]
            chunk_pos = self.coord_mlp(coords_norm[start:end])  # [chunk, D]
            chunk_tokens = chunk_feats + 0.5 * chunk_pos

            # encode with local transformer
            encoded = self.local_encoder(chunk_tokens.unsqueeze(0)).squeeze(0)
            encoded = self.attn_dropout(encoded)

            # gated attention pooling
            v = torch.tanh(self.attn_v(encoded))
            u = torch.sigmoid(self.attn_u(encoded))
            scores = self.attn_w(v * u).squeeze(-1)  # [chunk]

            # update numer and denom with attention scores
            # keep maxscore for numerical stability in softmax computation across chunks
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

        # finalize pooled token
        if denom.item() == 0.0:
            return torch.zeros(self.model_dim, device=patches.device)
        return numer / denom

    def encode_wsi_batch(
        self,
        patch_batches: List[torch.Tensor],
        coord_batches: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode a batch of WSIs into bag-level tokens.

        Args:
            patch_batches: List of [num_patches, embed_dim] tensors of patch embeddings
            coord_batches: List of [num_patches, 3] tensors of patch coordinates (x,y,level)

        Returns:
            tokens: [batch_size, model_dim] tensor of pooled WSI tokens
        """

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
        """
        Forward pass for a batch of WSIs.

        Args:
            patch_batches: List of [num_patches, embed_dim] tensors of patch embeddings
            coord_batches: List of [num_patches, 3] tensors of patch coordinates (x,y,level)

        Returns:
            logits: [batch_size, num_classes] tensor of class logits
            tokens: [batch_size, model_dim] tensor of pooled WSI tokens (for potential auxiliary use)
        """

        tokens = self.encode_wsi_batch(patch_batches, coord_batches)
        features = self.head(self.norm(tokens))
        logits = self.classifier(features) + self.logit_bias
        return logits, tokens
