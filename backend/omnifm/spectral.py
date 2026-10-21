"""FFT Spectral Encoder (`backend/omnifm/spectral.py`, Task 2.1).

Extracts a low-frequency 2D Fourier magnitude spectrum from an input image tensor
`(B, C, H, W)`, applies a centered low-pass spatial-frequency mask, projects the
filtered spectrum through a two-stage learned frequency-mixing MLP:
    s = Pool(sigma(W2 * sigma(W1 * FreqMix(S_LP))))
and normalizes the resulting spectral token to unit L2 length `(B, D)`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralEncoder(nn.Module):
    """Frequency-domain encoder producing L2-normalized spectral tokens `(B, D)`.

    Args:
        embedding_dim: Output spectral token dimension `D` (default: 128).
        low_pass_ratio: Fraction of spatial frequencies retained around the DC center
            along each spatial axis, in `(0.0, 1.0]` (default: 0.25).
        freq_grid_size: Fixed spatial grid size after adaptive pooling of the masked
            low-frequency magnitude spectrum (default: 8).
        hidden_dim: Intermediate hidden dimension in the spectral projection MLP (default: 256).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        low_pass_ratio: float = 0.25,
        freq_grid_size: int = 8,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if embedding_dim < 8:
            raise ValueError(f"embedding_dim must be >= 8, got {embedding_dim}.")
        if not (0.0 < low_pass_ratio <= 1.0):
            raise ValueError(f"low_pass_ratio must be in (0.0, 1.0], got {low_pass_ratio}.")
        if freq_grid_size < 2:
            raise ValueError(f"freq_grid_size must be >= 2, got {freq_grid_size}.")

        self.embedding_dim = int(embedding_dim)
        self.low_pass_ratio = float(low_pass_ratio)
        self.freq_grid_size = int(freq_grid_size)
        self.hidden_dim = int(hidden_dim)

        freq_flat_dim = self.freq_grid_size * self.freq_grid_size

        # Learned frequency mixing + two-layer projection: sigma(W2 * sigma(W1 * FreqMix(S_LP)))
        self.freq_mix = nn.Linear(freq_flat_dim, freq_flat_dim)
        self.w1 = nn.Linear(freq_flat_dim, self.hidden_dim)
        self.act1 = nn.GELU()
        self.w2 = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.act2 = nn.GELU()
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    @staticmethod
    def build_low_pass_mask(
        height: int,
        width: int,
        low_pass_ratio: float,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Build a centered binary low-pass mask of shape `(1, 1, H, W)`.

        Retains a centered window of height `max(1, round(H * low_pass_ratio))`
        and width `max(1, round(W * low_pass_ratio))` around `(H // 2, W // 2)`.
        Works for both odd and even spatial dimensions.
        """
        mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
        half_h = max(1, int(round(height * low_pass_ratio)) // 2)
        half_w = max(1, int(round(width * low_pass_ratio)) // 2)

        center_h = height // 2
        center_w = width // 2

        h_start = max(0, center_h - half_h)
        h_end = min(height, center_h + half_h + (1 if height % 2 == 1 else 0))
        w_start = max(0, center_w - half_w)
        w_end = min(width, center_w + half_w + (1 if width % 2 == 1 else 0))

        mask[:, :, h_start:h_end, w_start:w_end] = 1.0
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image tensor `x` of shape `(B, C, H, W)` into spectral token `(B, D)`.

         FFT is always computed in `float32` to prevent FP16 overflow/underflow
         under AMP autocast.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor (B, C, H, W), got shape {tuple(x.shape)}.")

        bsz, _channels, height, width = x.shape
        if bsz < 1 or height < 2 or width < 2:
            raise ValueError(f"Invalid spatial/batch dimensions: {tuple(x.shape)}.")

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            # 2D FFT over spatial axes (-2, -1) and shift DC to center
            fft_complex = torch.fft.fft2(x_fp32, dim=(-2, -1), norm="ortho")
            fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))

            # Stabilized magnitude spectrum: sqrt(Re^2 + Im^2 + eps), log1p compressed
            mag = torch.sqrt(fft_shifted.real.square() + fft_shifted.imag.square() + 1e-8)
            mag = torch.log1p(mag)

            # Channel-agnostic mean spectrum: (B, C, H, W) -> (B, 1, H, W)
            mag_mean = mag.mean(dim=1, keepdim=True)

            # Apply centered low-pass filter mask
            lp_mask = self.build_low_pass_mask(
                height=height,
                width=width,
                low_pass_ratio=self.low_pass_ratio,
                device=x.device,
                dtype=torch.float32,
            )
            s_lp = mag_mean * lp_mask  # (B, 1, H, W)

            # Pool masked spectrum to fixed frequency grid: (B, 1, G, G) -> (B, G*G)
            pooled_grid = F.adaptive_avg_pool2d(
                s_lp, (self.freq_grid_size, self.freq_grid_size)
            ).flatten(start_dim=1)

            # Learned spectral mixing & two-stage projection
            mixed = self.freq_mix(pooled_grid)
            h1 = self.act1(self.w1(mixed))
            h2 = self.act2(self.w2(h1))
            token = self.layer_norm(self.out_proj(h2) + h2)

            # L2-normalize each spectral token along D
            token_norm = F.normalize(token, p=2.0, dim=-1, eps=1e-8)

        return token_norm
