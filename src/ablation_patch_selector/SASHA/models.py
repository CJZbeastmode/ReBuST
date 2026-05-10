"""SASHA models: hierarchical attention classifier (HAFED-like) + RL policy/value."""

from __future__ import annotations

import torch
import torch.nn as nn


class HAFEDClassifier(nn.Module):
    """
    Lightweight hierarchical attention classifier.

    Input: token embeddings [B, N, D] + mask [B, N]
    Output: logits [B, C], attention [B, N], pooled [B, D]
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)

        self.token_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )

        self.head_attn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(self.hidden_dim, 1),
                )
                for _ in range(self.num_heads)
            ]
        )

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, int(num_classes))
        self.register_buffer("logit_bias", torch.zeros(int(num_classes)))

    def set_logit_adjustment(self, class_counts: torch.Tensor, tau: float) -> None:
        counts = class_counts.float().clamp(min=1.0)
        priors = counts / counts.sum()
        self.logit_bias = (-tau * torch.log(priors)).to(self.classifier.weight.device)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        # tokens: [B, N, D], mask: [B, N]
        x = self.token_proj(tokens)
        attn_list = []
        pooled_list = []

        for block in self.head_attn:
            score = block(x).squeeze(-1)  # [B, N]
            score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
            attn = torch.softmax(score, dim=1)
            attn = attn * mask.float()
            denom = attn.sum(dim=1, keepdim=True).clamp(min=1e-8)
            attn = attn / denom
            pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # [B, H]

            attn_list.append(attn)
            pooled_list.append(pooled)

        stacked_attn = torch.stack(attn_list, dim=1)  # [B, heads, N]
        stacked_pooled = torch.stack(pooled_list, dim=1)  # [B, heads, H]

        mean_attn = stacked_attn.mean(dim=1)
        pooled = stacked_pooled.mean(dim=1)
        pooled = self.norm(pooled)

        logits = self.classifier(pooled) + self.logit_bias
        return logits, mean_attn, pooled


class SashaPolicyValue(nn.Module):
    """
    Policy + Value network for patch selection.

    At each step, it receives candidate patch embeddings, selection mask, and
    global context vector. It returns logits over N candidates plus one STOP action.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)

        # per-candidate score head
        self.token_mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.token_logit = nn.Linear(self.hidden_dim, 1)

        # stop and value heads from global state
        self.global_mlp = nn.Sequential(
            nn.Linear(self.embed_dim + 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.stop_logit = nn.Linear(self.hidden_dim, 1)
        self.value_head = nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        candidate_embeddings: torch.Tensor,
        selected_mask: torch.Tensor,
        global_context: torch.Tensor,
        step_frac: torch.Tensor,
    ):
        """
        candidate_embeddings: [B, N, D]
        selected_mask: [B, N] bool
        global_context: [B, D]
        step_frac: [B, 1] in [0,1]
        """
        batch_size, num_candidates, _ = candidate_embeddings.shape

        global_expand = global_context.unsqueeze(1).expand(batch_size, num_candidates, -1)
        selected_flag = selected_mask.float().unsqueeze(-1)
        step_expand = step_frac.unsqueeze(1).expand(batch_size, num_candidates, -1)

        token_input = torch.cat(
            [candidate_embeddings, global_expand, selected_flag, step_expand],
            dim=-1,
        )
        token_feat = self.token_mlp(token_input)
        token_logits = self.token_logit(token_feat).squeeze(-1)  # [B, N]

        global_input = torch.cat([global_context, step_frac, selected_mask.float().mean(dim=1, keepdim=True)], dim=-1)
        global_feat = self.global_mlp(global_input)
        stop = self.stop_logit(global_feat).squeeze(-1).unsqueeze(1)  # [B,1]
        value = self.value_head(global_feat).squeeze(-1)

        logits = torch.cat([token_logits, stop], dim=1)  # [B, N+1]
        return logits, value
