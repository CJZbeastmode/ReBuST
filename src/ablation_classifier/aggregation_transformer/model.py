"""Aggregation transformer model with configurable CLAM, ABMIL, or WiKG pooling."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLAMAttentionPooling(nn.Module):
    def __init__(self, model_dim: int, attn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.attn_v = nn.Sequential(
            nn.Linear(model_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attn_u = nn.Sequential(
            nn.Linear(model_dim, attn_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attn_w = nn.Linear(attn_dim, 1)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v = self.attn_v(tokens)
        u = self.attn_u(tokens)
        scores = self.attn_w(v * u).squeeze(-1)

        masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(masked_scores, dim=1)
        attn = attn * mask.float()
        denom = attn.sum(dim=1, keepdim=True).clamp(min=1e-8)
        attn = attn / denom

        pooled = torch.bmm(attn.unsqueeze(1), tokens).squeeze(1)
        return pooled, attn


class ABMILPooling(nn.Module):
    def __init__(self, model_dim: int, attn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(model_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn(tokens).squeeze(-1)
        masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(masked_scores, dim=1)
        attn = attn * mask.float()
        denom = attn.sum(dim=1, keepdim=True).clamp(min=1e-8)
        attn = attn / denom

        pooled = torch.bmm(attn.unsqueeze(1), tokens).squeeze(1)
        return pooled, attn


class WiKGPooling(nn.Module):
    def __init__(
        self,
        model_dim: int,
        k_neighbors: int = 16,
        num_steps: int = 2,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.k_neighbors = int(k_neighbors)
        self.num_steps = int(num_steps)
        self.temperature = float(max(1e-6, temperature))

        self.gate = nn.Sequential(
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.Sigmoid(),
        )
        self.readout = nn.Linear(self.model_dim, 1)

    def _message_passing_once(self, x: torch.Tensor) -> torch.Tensor:
        n_tokens = int(x.shape[0])
        if n_tokens <= 1:
            return x

        k = min(self.k_neighbors, n_tokens)
        normalized = F.normalize(x, p=2, dim=-1)
        sim = normalized @ normalized.T
        top_values, top_indices = torch.topk(sim, k=k, dim=-1)

        neighbor_tokens = x[top_indices]
        weights = torch.softmax(top_values / self.temperature, dim=-1).unsqueeze(-1)
        aggregated = (weights * neighbor_tokens).sum(dim=1)

        gate = self.gate(torch.cat([x, aggregated], dim=-1))
        return gate * x + (1.0 - gate) * aggregated

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_len, _ = tokens.shape
        pooled_out = []
        attn_out = []

        for batch_index in range(batch_size):
            valid_mask = mask[batch_index]
            valid_count = int(valid_mask.sum().item())

            if valid_count <= 0:
                pooled_out.append(torch.zeros(self.model_dim, device=tokens.device, dtype=tokens.dtype))
                attn_out.append(torch.zeros(max_len, device=tokens.device, dtype=tokens.dtype))
                continue

            token_subset = tokens[batch_index, valid_mask]
            for _ in range(self.num_steps):
                token_subset = self._message_passing_once(token_subset)

            node_scores = self.readout(token_subset).squeeze(-1)
            node_attn = torch.softmax(node_scores, dim=0)
            pooled = (node_attn.unsqueeze(-1) * token_subset).sum(dim=0)

            full_attn = torch.zeros(max_len, device=tokens.device, dtype=tokens.dtype)
            full_attn[valid_mask] = node_attn

            pooled_out.append(pooled)
            attn_out.append(full_attn)

        pooled_tensor = torch.stack(pooled_out, dim=0)
        attn_tensor = torch.stack(attn_out, dim=0)
        return pooled_tensor, attn_tensor


class PureMIL(nn.Module):
    """True ABMIL / CLAM: linear projection → attention pooling → classifier.

    No transformer encoder — patches are treated as an unordered set, exactly
    as in Ilse et al. (2018) and Lu et al. (2021).
    """

    def __init__(
        self,
        embed_dim: int,
        model_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_coords: bool = True,
        attn_dim: int = 128,
        pooling_type: str = "abmil",
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.model_dim = int(model_dim)
        self.num_classes = int(num_classes)
        self.use_coords = bool(use_coords)
        self.pooling_type = str(pooling_type).lower()
        if self.pooling_type not in {"abmil", "clam"}:
            raise ValueError("pooling_type must be 'abmil' or 'clam'")

        self.patch_proj = nn.Linear(self.embed_dim, self.model_dim)
        if self.use_coords:
            self.coord_proj = nn.Sequential(
                nn.Linear(3, self.model_dim),
                nn.ReLU(),
                nn.Linear(self.model_dim, self.model_dim),
            )

        if self.pooling_type == "abmil":
            self.pool = ABMILPooling(self.model_dim, attn_dim=attn_dim, dropout=dropout)
        else:
            self.pool = CLAMAttentionPooling(self.model_dim, attn_dim=attn_dim, dropout=dropout)

        self.norm = nn.LayerNorm(self.model_dim)
        self.classifier = nn.Linear(self.model_dim, self.num_classes)
        self.register_buffer("logit_bias", torch.zeros(self.num_classes))

    def set_logit_adjustment(self, class_counts: torch.Tensor, tau: float) -> None:
        counts = class_counts.float().clamp(min=1.0)
        priors = counts / counts.sum()
        self.logit_bias = (-tau * torch.log(priors)).to(self.classifier.weight.device)

    @staticmethod
    def _pad_batch(
        patch_batches: List[torch.Tensor],
        coord_batches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(patch_batches)
        max_len = max(int(p.shape[0]) for p in patch_batches)
        emb_dim = int(patch_batches[0].shape[-1])

        device = patch_batches[0].device
        patch_tensor = torch.zeros(batch_size, max_len, emb_dim, device=device)
        coord_tensor = torch.zeros(batch_size, max_len, 3, device=device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

        for i, (patches, coords) in enumerate(zip(patch_batches, coord_batches)):
            n = int(patches.shape[0])
            patch_tensor[i, :n] = patches
            coord_tensor[i, :n] = coords
            mask[i, :n] = True

        return patch_tensor, coord_tensor, mask

    def forward(
        self,
        patch_batches: List[torch.Tensor],
        coord_batches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        patch_batches = [p.float() for p in patch_batches]
        coord_batches = [c.float() for c in coord_batches]

        patch_tensor, coord_tensor, mask = self._pad_batch(patch_batches, coord_batches)

        x = self.patch_proj(patch_tensor)
        if self.use_coords:
            x = x + 0.5 * self.coord_proj(coord_tensor)

        pooled, attn = self.pool(x, mask)
        logits = self.classifier(self.norm(pooled)) + self.logit_bias

        extras = {
            "attention": attn,
            "mask": mask,
            "token_embeddings": x,
            "slide_embedding": pooled,
        }
        return logits, extras


class AggregationTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        model_dim: int,
        num_classes: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_coords: bool = True,
        attn_dim: int = 128,
        pooling_type: str = "abmil",
        wikg_k: int = 16,
        wikg_steps: int = 2,
        wikg_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.model_dim = int(model_dim)
        self.num_classes = int(num_classes)
        self.use_coords = bool(use_coords)
        self.pooling_type = str(pooling_type).lower()
        if self.pooling_type not in {"abmil", "clam", "wikg"}:
            raise ValueError("pooling_type must be one of {'abmil', 'clam', 'wikg'}")

        self.patch_proj = nn.Linear(self.embed_dim, self.model_dim)
        if self.use_coords:
            self.coord_proj = nn.Sequential(
                nn.Linear(3, self.model_dim),
                nn.ReLU(),
                nn.Linear(self.model_dim, self.model_dim),
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=self.model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if self.pooling_type == "abmil":
            self.pool = ABMILPooling(self.model_dim, attn_dim=attn_dim, dropout=dropout)
        elif self.pooling_type == "wikg":
            self.pool = WiKGPooling(
                self.model_dim,
                k_neighbors=wikg_k,
                num_steps=wikg_steps,
                temperature=wikg_temperature,
            )
        else:
            self.pool = CLAMAttentionPooling(self.model_dim, attn_dim=attn_dim, dropout=dropout)

        self.norm = nn.LayerNorm(self.model_dim)
        self.classifier = nn.Linear(self.model_dim, self.num_classes)
        self.register_buffer("logit_bias", torch.zeros(self.num_classes))

    def set_logit_adjustment(self, class_counts: torch.Tensor, tau: float) -> None:
        counts = class_counts.float().clamp(min=1.0)
        priors = counts / counts.sum()
        self.logit_bias = (-tau * torch.log(priors)).to(self.classifier.weight.device)

    @staticmethod
    def _pad_batch(
        patch_batches: List[torch.Tensor],
        coord_batches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(patch_batches)
        max_len = max(int(p.shape[0]) for p in patch_batches)
        emb_dim = int(patch_batches[0].shape[-1])

        device = patch_batches[0].device
        patch_tensor = torch.zeros(batch_size, max_len, emb_dim, device=device)
        coord_tensor = torch.zeros(batch_size, max_len, 3, device=device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

        for i, (patches, coords) in enumerate(zip(patch_batches, coord_batches)):
            n = int(patches.shape[0])
            patch_tensor[i, :n] = patches
            coord_tensor[i, :n] = coords
            mask[i, :n] = True

        return patch_tensor, coord_tensor, mask

    def forward(
        self,
        patch_batches: List[torch.Tensor],
        coord_batches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        patch_batches = [p.float() for p in patch_batches]
        coord_batches = [c.float() for c in coord_batches]

        patch_tensor, coord_tensor, mask = self._pad_batch(patch_batches, coord_batches)

        x = self.patch_proj(patch_tensor)
        if self.use_coords:
            x = x + 0.5 * self.coord_proj(coord_tensor)

        x = self.encoder(x, src_key_padding_mask=~mask)
        pooled, attn = self.pool(x, mask)

        logits = self.classifier(self.norm(pooled)) + self.logit_bias
        extras = {
            "attention": attn,
            "mask": mask,
            "token_embeddings": x,
            "slide_embedding": pooled,
        }
        return logits, extras
