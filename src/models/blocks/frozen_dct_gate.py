"""
FrozenDCTGate — Frozen DCT Frequency Routing (Idea G / FDCR).

Each channel learns a global frequency preference (data-independent).
The DCT basis is frozen (never trained); only the per-channel frequency
mask is learnable.  This separates periodic actions (walking, ~0.5-2 Hz)
from impulsive ones (punching, high-frequency burst) at zero per-sample
adaptive compute.

Unlike LAST-Base's FreqTemporalGate (Idea A, per-sample adaptive),
FDCR has the same mask for every sample in the batch — no self-attention,
no per-sample sigmoid, just a fixed learnable filter.

Output uses a residual connection: x + x_back.
This ensures gradient flows even when the mask is near 0 (all frequencies
suppressed → x_back ≈ 0 → gradient from x path is unobstructed).

Cost: C × T learnable params, zero per-sample adaptive compute.

Reference: Experiment-LAST-Lite.md — Sections 1, 4.
"""

import numpy as np
import torch
import torch.nn as nn


class FrozenDCTGate(nn.Module):
    """
    Frequency-domain channel gating with a frozen DCT basis.

    Args:
        channels:  Number of channels (C).
        T:         Temporal length (must match input's T dimension).
    """

    def __init__(self, channels: int, T: int):
        super().__init__()

        # ── Frozen DCT-II basis (T × T) ─────────────────────────────────
        # scipy.fft.dct(eye(T), type=2, norm='ortho') gives the orthonormal
        # DCT-II matrix D such that x_freq = x @ D^T and x_back = x_freq @ D.
        try:
            from scipy.fft import dct as scipy_dct
            dct_np = scipy_dct(np.eye(T), type=2, norm='ortho').astype(np.float32)
        except ImportError:
            # Fallback: build DCT-II matrix manually
            n = np.arange(T, dtype=np.float32)
            k = n[:, None]
            dct_np = np.cos(np.pi / T * (n + 0.5) * k).astype(np.float32)
            # Orthonormal scaling
            dct_np[0] *= np.sqrt(1.0 / T)
            dct_np[1:] *= np.sqrt(2.0 / T)

        dct_matrix = torch.tensor(dct_np)   # (T, T)
        self.register_buffer('dct',  dct_matrix)      # analysis
        self.register_buffer('idct', dct_matrix.T)    # synthesis (DCT-II inverse = DCT-II^T)

        # ── Learnable frequency mask ─────────────────────────────────────
        # Shape (1, C, T, 1): one logit per (channel, frequency bin).
        # Init -2.0 → sigmoid(-2) ≈ 0.119 → x_back ≈ 0.119x → output ≈ 1.119x.
        # Near-identity init: avoids 1.5x BN instability from zero-init (sigmoid(0)=0.5).
        # Learning rate to mask: sigmoid'(-2) ≈ 0.105 (slower but not dead).
        # Excluded from weight decay (see trainer.py no_decay list).
        self.freq_mask = nn.Parameter(torch.full((1, channels, T, 1), -2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        Returns:
            out: (B, C, T, V) — x + frequency-filtered version of x
        """
        # Cast to fp32 for DCT matmul: under AMP (fp16), summing T=64 products in fp16
        # accumulates ~1-2% relative error per coefficient.  fp32 cast is free on CPU
        # and adds <0.1% overhead on GPU (64×64 matmul).  Cast result back to input dtype.
        orig_dtype = x.dtype
        xf = x.float()
        dct  = self.dct.float()
        idct = self.idct.float()

        # Transform to DCT domain along T axis
        # xf: (B, C, T, V) → transpose T↔V → (B, C, V, T)
        # matmul with dct (T, T): (B, C, V, T) @ (T, T) → (B, C, V, T)
        # transpose back → (B, C, T, V)
        x_freq = torch.matmul(xf.transpose(2, 3), dct).transpose(2, 3)

        # Apply data-independent learnable mask (same for all samples)
        mask = torch.sigmoid(self.freq_mask.float())   # (1, C, T, 1) — broadcasts over B, V
        x_gated = x_freq * mask

        # Transform back to time domain
        x_back = torch.matmul(x_gated.transpose(2, 3), idct).transpose(2, 3).to(orig_dtype)

        # Residual: preserves gradient flow when mask suppresses all frequencies
        return x + x_back
