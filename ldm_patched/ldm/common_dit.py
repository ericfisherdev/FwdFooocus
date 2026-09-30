"""Shared DiT helpers: RoPE tables, rotary application, latent padding, fp16 clamp.

Mirrors ComfyUI's `comfy/ldm/common_dit.py` + `comfy/ldm/flux/math.py`. The
NextDiT (Lumina2 / Z-Image) and SingleStreamDiT (Krea 2) backbones both build on
these. `patchify`/`unpatchify` are deliberately NOT shared: NextDiT flattens
patch tokens in (ph, pw, c) order while Krea 2 uses (c, ph, pw), so each model
family keeps its own.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def clamp_fp16(x):
    # SwiGLU/attention branches of these DiTs can overflow fp16 range; the
    # upstream reference clamps at these same points to keep fp16 inference
    # numerically stable instead of producing NaNs.
    if x.dtype == torch.float16:
        return torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


def pad_to_patch_size(x, patch_size):
    pad_h = (-x.shape[-2]) % patch_size
    pad_w = (-x.shape[-1]) % patch_size
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="circular")


def rope_freqs(pos, dim, theta):
    """Per-axis rotation-matrix RoPE table (Flux-style), shape (..., n, dim//2, 2, 2)."""
    assert dim % 2 == 0, "rope axis dim must be even, got {}".format(dim)
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device="cpu")
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float64, device="cpu"), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.view(*out.shape[:-1], 2, 2)
    return out.to(dtype=torch.float32, device=pos.device)


class EmbedND(nn.Module):
    """Concatenates per-axis RoPE rotation tables into one head_dim-sized table."""

    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat([rope_freqs(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


def apply_rope1(x, freqs_cis):
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, :x_.shape[2]]
    x_out = freqs_cis[..., 0] * x_[..., 0] + freqs_cis[..., 1] * x_[..., 1]
    return x_out.reshape(*x.shape).type_as(x)


def apply_rope(xq, xk, freqs_cis):
    return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)
