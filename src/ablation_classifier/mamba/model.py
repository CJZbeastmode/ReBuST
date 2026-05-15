"""Mamba MIL model for bag-level WSI classification.

Uses the real S6 selective state space model (Gu & Dao, 2023) implemented in
pure PyTorch (sequential scan — correct math, no custom CUDA kernels required).
"""

import math
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


def _parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive parallel prefix scan for the linear recurrence h[t] = a[t]*h[t-1] + b[t].

    Uses the Hillis-Steele algorithm with the associative operator:
        (a2, b2) ⊕ (a1, b1)  =  (a2·a1,  a2·b1 + b2)

    O(log L) rounds of parallel tensor ops — fast on CUDA where each op is truly parallel
    across B×D×N. On CPU, the large [B, L, D, N] tensor allocations make this slower than
    the sequential loop; use _sequential_scan on CPU instead.

    Args:
        a: [B, L, D, N]  — per-step decay factors (A_bar from ZOH discretisation)
        b: [B, L, D, N]  — per-step input contributions (B_bar · x)
    Returns:
        h: [B, L, D, N]  — state sequence where h[:, t] = accumulated state at step t
    """
    L = a.shape[1]
    d = 1
    while d < L:
        a_left = a[:, : L - d]
        b_left = b[:, : L - d]
        a_right_new = a[:, d:] * a_left
        b_right_new = a[:, d:] * b_left + b[:, d:]
        a = torch.cat([a[:, :d], a_right_new], dim=1)
        b = torch.cat([b[:, :d], b_right_new], dim=1)
        d *= 2
    return b


@torch.jit.script
def _sequential_scan(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """JIT-compiled sequential S6 scan — eliminates Python loop overhead on CPU.

    Args:
        a: [B, L, D, N]  — A_bar (decay factors)
        b: [B, L, D, N]  — Bu (B_bar · x, input contributions)
        c: [B, L, N]     — C (output matrix)
    Returns:
        y: [B, L, D]
    """
    B = a.shape[0]
    L = a.shape[1]
    D = a.shape[2]
    N = a.shape[3]
    h = a.new_zeros(B, D, N)
    y = a.new_zeros(B, L, D)
    for i in range(L):
        h = a[:, i] * h + b[:, i]
        y[:, i] = (h * c[:, i].unsqueeze(1)).sum(-1)
    return y


class SelectiveSSM(nn.Module):
    """S6: Selective State Space Model core from Mamba (Gu & Dao, 2023).

    Input-dependent A_bar, B, C and discretization step Δ — the key property
    that distinguishes Mamba from linear RNNs with fixed decay.

    Implemented as a sequential scan (O(N) in sequence length, correct S6
    math) without custom CUDA kernels.

    Reference: https://arxiv.org/abs/2312.00752
    """

    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_inner = int(d_inner)
        self.d_state = int(d_state)
        # dt_rank: rank of the Δ projection (heuristic from the paper)
        self.dt_rank = math.ceil(d_inner / 16)

        # Projects each token to (dt_rank + 2*d_state) for [Δ_raw, B, C]
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        # Expands Δ from dt_rank back to d_inner (bias initialised for target dt range)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # A: diagonal SSM matrix, log-parameterised so A = -exp(A_log) stays negative.
        # Initialised as [1, 2, …, d_state] for each channel (HiPPO-inspired).
        A_init = (
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(d_inner, -1)
        )
        self.A_log = nn.Parameter(torch.log(A_init.clone()))

        # D: per-channel skip-connection scalar
        self.D = nn.Parameter(torch.ones(d_inner))

        # dt_proj weight: uniform init matching dt_rank^{-0.5} scale
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank**-0.5, self.dt_rank**-0.5)

        # dt_proj bias: initialise so softplus(bias) is uniform in [dt_min, dt_max]
        dt_init = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        # inverse-softplus: softplus(inv_dt) == dt_init
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_inner]  (padded positions should already be zeroed)
        Returns:
            y: [B, L, d_inner]
        """
        B, L, _ = x.shape
        N = self.d_state

        # A stays negative (dissipative system)
        A = -torch.exp(self.A_log.float())  # [d_inner, N]

        # Input-dependent projections
        xbc = self.x_proj(x)  # [B, L, dt_rank + 2N]
        dt_raw, B_sel, C = xbc.split([self.dt_rank, N, N], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))  # [B, L, d_inner], positive

        # ZOH discretisation
        # A_bar[b,l,d,n] = exp(dt[b,l,d] * A[d,n])
        # Bu[b,l,d,n]    = dt[b,l,d] * B_sel[b,l,n] * x[b,l,d]  (combined input)
        A_bar = torch.exp(dt.unsqueeze(-1) * A)  # [B, L, d_inner, N]
        Bu = (
            dt.unsqueeze(-1)
            * B_sel.unsqueeze(2)  # B_bar [B, L, d_inner, N]
            * x.unsqueeze(-1)
        )  # * x   → Bu [B, L, d_inner, N]

        # GPU (CUDA/MPS): parallel scan — O(log L) rounds, truly parallel across B×D×N.
        # CPU:            JIT-compiled sequential scan — avoids large tensor allocations.
        if x.is_cuda:
            h = _parallel_scan(A_bar, Bu)  # [B, L, d_inner, N]
            y = (h * C.unsqueeze(2)).sum(-1)  # [B, L, d_inner]
        else:
            y = _sequential_scan(A_bar, Bu, C)  # [B, L, d_inner]
        y = y + x * self.D  # skip connection
        return y


class MambaLikeMixer(nn.Module):
    """Mamba mixer block: causal conv → S6 selective SSM → SiLU gate → project out.

    Interface is identical to the previous placeholder so MambaEncoderBlock is unchanged.
    """

    def __init__(
        self,
        model_dim: int,
        expand_factor: int = 2,
        conv_kernel_size: int = 3,
        d_state: int = 8,
    ):
        super().__init__()
        self.model_dim = int(model_dim)
        self.inner_dim = int(model_dim * expand_factor)

        # Project input to inner_dim (SSM branch) + inner_dim (gate branch)
        self.in_proj = nn.Linear(self.model_dim, self.inner_dim * 2, bias=False)

        # Short causal depthwise conv applied to the SSM branch before the scan
        self.conv1d = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=self.inner_dim,
            bias=True,
        )

        # Real S6 selective state space model
        self.ssm = SelectiveSSM(self.inner_dim, d_state=d_state)

        self.out_proj = nn.Linear(self.inner_dim, self.model_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [B, L, model_dim]
            mask: [B, L]  bool, True = valid token
        Returns:
            y: [B, L, model_dim]
        """
        B, L, _ = x.shape

        xz = self.in_proj(x)  # [B, L, 2*inner_dim]
        x_in, z = xz.chunk(2, dim=-1)  # each [B, L, inner_dim]

        # Causal depthwise conv (pad right so output length == L)
        x_in = self.conv1d(x_in.transpose(1, 2)).transpose(1, 2)[:, :L, :]
        x_in = F.silu(x_in)

        # Zero padded positions so they do not contaminate the SSM state
        x_in = x_in * mask.unsqueeze(-1).float()

        # Selective SSM
        y = self.ssm(x_in)  # [B, L, inner_dim]

        # Gating (SiLU of the parallel branch) + output projection
        y = y * F.silu(z)
        y = self.out_proj(y)

        # Mask output padding
        y = y * mask.unsqueeze(-1).float()
        return y


class MambaEncoderBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        dropout: float = 0.1,
        expand_factor: int = 2,
        conv_kernel_size: int = 3,
        d_state: int = 8,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.mixer = MambaLikeMixer(
            model_dim=model_dim,
            expand_factor=expand_factor,
            conv_kernel_size=conv_kernel_size,
            d_state=d_state,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mixed = self.mixer(self.norm1(x), mask)
        x = x + self.drop1(mixed)
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class MambaMIL(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        model_dim: int,
        num_classes: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_coords: bool = True,
        attn_dim: int = 128,
        expand_factor: int = 2,
        conv_kernel_size: int = 3,
        d_state: int = 8,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.model_dim = int(model_dim)
        self.num_classes = int(num_classes)
        self.use_coords = bool(use_coords)

        # patch and coordinate projections
        self.patch_proj = nn.Linear(self.embed_dim, self.model_dim)
        if self.use_coords:
            self.coord_proj = nn.Sequential(
                nn.Linear(3, self.model_dim),
                nn.ReLU(),
                nn.Linear(self.model_dim, self.model_dim),
            )

        # Mamba encoder blocks
        self.blocks = nn.ModuleList(
            [
                MambaEncoderBlock(
                    model_dim=self.model_dim,
                    dropout=dropout,
                    expand_factor=expand_factor,
                    conv_kernel_size=conv_kernel_size,
                    d_state=d_state,
                )
                for _ in range(int(num_layers))
            ]
        )
        # attention pooling
        self.pool = CLAMAttentionPooling(
            self.model_dim, attn_dim=attn_dim, dropout=dropout
        )
        # classifier and logit adjustment buffer
        self.norm = nn.LayerNorm(self.model_dim)
        self.classifier = nn.Linear(self.model_dim, self.num_classes)
        self.register_buffer("logit_bias", torch.zeros(self.num_classes))

    # Set logit adjustment bias from class counts and tau
    def set_logit_adjustment(self, class_counts: torch.Tensor, tau: float) -> None:
        counts = class_counts.float().clamp(min=1.0)
        priors = counts / counts.sum()
        self.logit_bias = (-tau * torch.log(priors)).to(self.classifier.weight.device)

    # Pad a list of per-WSI patch tensors into a batch tensor with mask
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

        for index, (patches, coords) in enumerate(zip(patch_batches, coord_batches)):
            count = int(patches.shape[0])
            patch_tensor[index, :count] = patches
            coord_tensor[index, :count] = coords
            mask[index, :count] = True

        return patch_tensor, coord_tensor, mask

    def forward(
        self,
        patch_batches: List[torch.Tensor],
        coord_batches: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        patch_batches = [p.float() for p in patch_batches]
        coord_batches = [c.float() for c in coord_batches]

        patch_tensor, coord_tensor, mask = self._pad_batch(patch_batches, coord_batches)

        tokens = self.patch_proj(patch_tensor)
        if self.use_coords:
            tokens = tokens + 0.5 * self.coord_proj(coord_tensor)

        for block in self.blocks:
            tokens = block(tokens, mask)

        pooled, attention = self.pool(tokens, mask)
        logits = self.classifier(self.norm(pooled)) + self.logit_bias

        extras = {
            "attention": attention,
            "mask": mask,
            "token_embeddings": tokens,
            "slide_embedding": pooled,
        }
        return logits, extras
