"""SingleStreamDiT: the Krea 2 diffusion backbone (12.8B single-stream MMDiT).

Port of ComfyUI's `comfy/ldm/krea2/model.py::SingleStreamDiT` (PR #14589). It is
trained from scratch and is NOT Flux-compatible: different block structure,
grouped-query attention, sigmoid-gated attention, AdaLN-single timestep
modulation and RoPE theta 1000.

State-dict key names below are matched 1:1 against the safetensors header of
Comfy-Org/Krea-2 `diffusion_models/krea2_turbo_bf16.safetensors` (430 tensors,
12,820,073,036 parameters, flat keys with no `model.diffusion_model.` prefix);
`tests/fixtures/krea2_turbo_bf16_header.json` is a copy of that header.

Only the plain text-to-image forward path is implemented. The reference's
editing-only paths (`ref_latents`, `timestep_zero_index`, 5-D temporal input,
`attn1_patch` hooks) are deliberately not ported. The `*_bf16` checkpoints
carry the plain key set this module loads, and `*_fp8_scaled` checkpoints are
folded back into plain weights at load (`dequantize_comfy_scaled_fp8`). The
`*_int8_convrot`, `*_mxfp8` and `*_nvfp4` variants add `weight_scale` tensors
plus quantization metadata in formats `ldm_patched` has no loader for, so they
are rejected at load with a ValueError naming the layer.

Wiring into `supported_models`/`model_base` (FWDF-132), the text encoder
(FWDF-133) and the pipeline (FWDF-152) build on this module.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import ldm_patched.modules.ops
from ldm_patched.ldm.common_dit import (
    EmbedND,
    apply_rope,
    clamp_fp16,
    pad_to_patch_size,
)
from ldm_patched.ldm.modules.attention import optimized_attention
from ldm_patched.ldm.modules.diffusionmodules.util import timestep_embedding

ops = ldm_patched.modules.ops.disable_weight_init

MODULATION_CHUNKS_PER_BLOCK = 6


def patchify(x, patch_size):
    """(B, C, H, W) -> (B, h*w, C*ph*pw) with token features ordered (c, ph, pw).

    NextDiT orders them (ph, pw, c); the orderings are not interchangeable.
    `reshape`, not `view`: the incoming latent may be a non-contiguous slice.
    """
    B, C, H, W = x.shape
    h_tokens, w_tokens = H // patch_size, W // patch_size
    x = x.reshape(B, C, h_tokens, patch_size, w_tokens, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, h_tokens * w_tokens, C * patch_size * patch_size)
    return x, h_tokens, w_tokens


def unpatchify(x, h_tokens, w_tokens, patch_size, channels):
    """Inverse of `patchify`: (B, h*w, C*ph*pw) -> (B, C, h*ph, w*pw)."""
    B = x.shape[0]
    x = x.reshape(B, h_tokens, w_tokens, channels, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, channels, h_tokens * patch_size, w_tokens * patch_size)


def build_position_ids(txt_len, h_tokens, w_tokens, batch_size, device):
    """3-axis RoPE position ids. Text tokens sit at the origin (all zeros); image
    tokens are (0, row, col).
    """
    txt_ids = torch.zeros(batch_size, txt_len, 3, dtype=torch.float32, device=device)

    img_ids = torch.zeros(h_tokens, w_tokens, 3, dtype=torch.float32, device=device)
    img_ids[..., 1] = torch.arange(h_tokens, dtype=torch.float32, device=device).view(-1, 1)
    img_ids[..., 2] = torch.arange(w_tokens, dtype=torch.float32, device=device).view(1, -1)
    img_ids = img_ids.reshape(1, h_tokens * w_tokens, 3).repeat(batch_size, 1, 1)
    return txt_ids, img_ids


def expand_kv_heads(x, n_rep):
    """Grouped-query expansion (B, L, Hkv, D) -> (B, L, Hkv*n_rep, D).

    Head ordering is kv0 x n_rep, kv1 x n_rep, ... i.e. `repeat_interleave(n_rep)`
    over the head axis, which is the layout the checkpoint was trained with.
    """
    if n_rep == 1:
        return x
    return x.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)


class ScaledRMSNorm(nn.Module):
    """RMSNorm whose learned parameter is `scale` and is applied as `1 + scale`.

    Not interchangeable with `ops.RMSNorm` (parameter is named `weight` there and
    has no +1 convention). The parameter is cast to float32 on every forward so
    the norm also works under `manual_cast` and with low-precision storage.
    """

    def __init__(self, features, eps=1e-5, dtype=None, device=None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(features, dtype=dtype, device=device))

    def forward(self, x):
        weight = self.scale.to(device=x.device, dtype=torch.float32) + 1.0
        return F.rms_norm(x.float(), (self.features,), weight=weight, eps=self.eps).to(x.dtype)


class QKNorm(nn.Module):
    """Per-head query/key normalization, applied on (B, L, H, D) before RoPE."""

    def __init__(self, head_dim, dtype=None, device=None):
        super().__init__()
        self.qnorm = ScaledRMSNorm(head_dim, dtype=dtype, device=device)
        self.knorm = ScaledRMSNorm(head_dim, dtype=dtype, device=device)

    def forward(self, q, k):
        return self.qnorm(q), self.knorm(k)


class SwiGLU(nn.Module):
    def __init__(self, features, multiplier=4, multiple=128,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        hidden = int(2 * features / 3) * multiplier
        hidden = multiple * math.ceil(hidden / multiple)
        self.gate = operations.Linear(features, hidden, bias=False, dtype=dtype, device=device)
        self.up = operations.Linear(features, hidden, bias=False, dtype=dtype, device=device)
        self.down = operations.Linear(hidden, features, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return self.down(clamp_fp16(F.silu(self.gate(x)) * self.up(x)))


class GatedAttention(nn.Module):
    """Grouped-query attention with a sigmoid output gate; `freqs=None` skips RoPE."""

    def __init__(self, dim, heads, kvheads, dtype=None, device=None, operations=ops):
        super().__init__()
        assert dim % heads == 0 and heads % kvheads == 0, \
            "dim ({}) must split into heads ({}) and heads into kvheads ({})".format(dim, heads, kvheads)
        self.heads = heads
        self.kvheads = kvheads
        self.n_rep = heads // kvheads
        self.head_dim = dim // heads

        kv_dim = kvheads * self.head_dim
        self.wq = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.wk = operations.Linear(dim, kv_dim, bias=False, dtype=dtype, device=device)
        self.wv = operations.Linear(dim, kv_dim, bias=False, dtype=dtype, device=device)
        self.gate = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.wo = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.qknorm = QKNorm(self.head_dim, dtype=dtype, device=device)

    def forward(self, x, freqs=None):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.heads, self.head_dim)
        k = self.wk(x).view(B, L, self.kvheads, self.head_dim)
        v = self.wv(x).view(B, L, self.kvheads, self.head_dim)

        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = apply_rope(q, k, freqs)

        k = expand_kv_heads(k, self.n_rep)
        v = expand_kv_heads(v, self.n_rep)

        out = optimized_attention(
            q.reshape(B, L, -1), k.reshape(B, L, -1), v.reshape(B, L, -1), self.heads, mask=None,
        )
        return self.wo(out * torch.sigmoid(self.gate(x)))


class DoubleSharedModulation(nn.Module):
    """AdaLN-single: one shared timestep vector plus a per-block learned offset,
    chunked into (prescale, preshift, pregate, postscale, postshift, postgate).
    """

    def __init__(self, features, dtype=None, device=None):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(MODULATION_CHUNKS_PER_BLOCK * features, dtype=dtype, device=device))

    def forward(self, vec):
        return (vec + self.lin.to(dtype=vec.dtype, device=vec.device)).chunk(MODULATION_CHUNKS_PER_BLOCK, dim=-1)


class SimpleModulation(nn.Module):
    """Final-layer modulation: (B, 1, F) timestep vector -> (scale, shift), each (B, 1, F)."""

    def __init__(self, features, dtype=None, device=None):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, features, dtype=dtype, device=device))

    def forward(self, vec):
        scale, shift = (vec + self.lin.to(dtype=vec.dtype, device=vec.device).unsqueeze(0)).chunk(2, dim=1)
        return scale, shift


class TextFusionBlock(nn.Module):
    """Plain pre/post-norm transformer block (no timestep modulation, no RoPE)."""

    def __init__(self, features, heads, multiplier, kvheads, dtype=None, device=None, operations=ops):
        super().__init__()
        self.prenorm = ScaledRMSNorm(features, dtype=dtype, device=device)
        self.postnorm = ScaledRMSNorm(features, dtype=dtype, device=device)
        self.attn = GatedAttention(features, heads, kvheads, dtype=dtype, device=device, operations=operations)
        self.mlp = SwiGLU(features, multiplier, dtype=dtype, device=device, operations=operations)

    def forward(self, x):
        x = x + self.attn(self.prenorm(x))
        return x + self.mlp(self.postnorm(x))


class TextFusionTransformer(nn.Module):
    """Collapses the stacked per-layer text-encoder taps into one text sequence.

    Input (B, seq, num_txt_layers, txt_dim): the layerwise blocks attend across
    the layer taps of each token, a learned projector merges the layer axis, and
    the refiner blocks attend across tokens.
    """

    def __init__(self, num_txt_layers=12, txt_dim=2560, heads=20, multiplier=4, kvheads=20,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        block_args = dict(dtype=dtype, device=device, operations=operations)
        self.layerwise_blocks = nn.ModuleList([
            TextFusionBlock(txt_dim, heads, multiplier, kvheads, **block_args) for _ in range(2)
        ])
        self.projector = operations.Linear(num_txt_layers, 1, bias=False, dtype=dtype, device=device)
        self.refiner_blocks = nn.ModuleList([
            TextFusionBlock(txt_dim, heads, multiplier, kvheads, **block_args) for _ in range(2)
        ])

    def forward(self, x):
        B, seq, n_layers, dim = x.shape
        x = x.reshape(B * seq, n_layers, dim)
        for block in self.layerwise_blocks:
            x = block(x)
        x = x.reshape(B, seq, n_layers, dim).transpose(2, 3)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features, heads, multiplier, kvheads, dtype=None, device=None, operations=ops):
        super().__init__()
        self.mod = DoubleSharedModulation(features, dtype=dtype, device=device)
        self.prenorm = ScaledRMSNorm(features, dtype=dtype, device=device)
        self.postnorm = ScaledRMSNorm(features, dtype=dtype, device=device)
        self.attn = GatedAttention(features, heads, kvheads, dtype=dtype, device=device, operations=operations)
        self.mlp = SwiGLU(features, multiplier, dtype=dtype, device=device, operations=operations)

    def forward(self, x, vec, freqs):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs)
        return x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)


class LastLayer(nn.Module):
    def __init__(self, features, patch, channels, dtype=None, device=None, operations=ops):
        super().__init__()
        self.norm = ScaledRMSNorm(features, dtype=dtype, device=device)
        self.linear = operations.Linear(features, patch * patch * channels, bias=True, dtype=dtype, device=device)
        self.modulation = SimpleModulation(features, dtype=dtype, device=device)

    def forward(self, x, tvec):
        scale, shift = self.modulation(tvec)
        return self.linear((1 + scale) * self.norm(x) + shift)


class SingleStreamDiT(nn.Module):
    """Krea 2 backbone. Constructor defaults are the published 12.8B config.

    Text and image tokens are concatenated into one stream with 3-axis RoPE
    (axes [32, 48, 48] for head_dim 128) and modulated by a single shared
    timestep vector (AdaLN-single).
    """

    def __init__(
        self,
        features=6144,
        tdim=256,
        txtdim=2560,
        heads=48,
        kvheads=12,
        multiplier=4,
        layers=28,
        patch=2,
        channels=16,
        theta=1000.0,
        txtlayers=12,
        txtheads=20,
        txtkvheads=20,
        time_factor=1000.0,
        image_model=None,
        device=None,
        dtype=None,
        operations=ops,
        **kwargs,
    ):
        super().__init__()
        head_dim = features // heads
        axes_dim = [head_dim - 12 * (head_dim // 16), 6 * (head_dim // 16), 6 * (head_dim // 16)]
        assert sum(axes_dim) == head_dim, \
            "rope axes {} must sum to head_dim {}".format(axes_dim, head_dim)

        self.dtype = dtype
        self.features = features
        self.tdim = tdim
        self.patch = patch
        self.channels = channels
        self.out_channels = channels
        self.txtlayers = txtlayers
        self.txtdim = txtdim
        self.time_factor = time_factor

        common = dict(dtype=dtype, device=device)
        self.pe_embedder = EmbedND(head_dim, theta=int(theta), axes_dim=axes_dim)
        self.first = operations.Linear(channels * patch * patch, features, bias=True, **common)
        self.blocks = nn.ModuleList([
            SingleStreamBlock(features, heads, multiplier, kvheads, operations=operations, **common)
            for _ in range(layers)
        ])
        self.tmlp = nn.Sequential(
            operations.Linear(tdim, features, bias=True, **common),
            nn.GELU(approximate="tanh"),
            operations.Linear(features, features, bias=True, **common),
        )
        self.txtfusion = TextFusionTransformer(
            txtlayers, txtdim, txtheads, multiplier, txtkvheads, operations=operations, **common,
        )
        self.txtmlp = nn.Sequential(
            ScaledRMSNorm(txtdim, **common),
            operations.Linear(txtdim, features, bias=True, **common),
            nn.GELU(approximate="tanh"),
            operations.Linear(features, features, bias=True, **common),
        )
        self.last = LastLayer(features, patch, channels, operations=operations, **common)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"),
            operations.Linear(features, MODULATION_CHUNKS_PER_BLOCK * features, bias=True, **common),
        )
        self._freqs_cache = None

    def _unpack_context(self, context):
        """(B, seq, txtlayers*txtdim) -> (B, seq, txtlayers, txtdim).

        Raises:
            ValueError: the last dimension is not `txtlayers * txtdim`.
        """
        expected = self.txtlayers * self.txtdim
        if context.shape[-1] != expected:
            raise ValueError(
                "Krea 2 context must have {} features ({} layers x {}), got {}".format(
                    expected, self.txtlayers, self.txtdim, context.shape[-1])
            )
        return context.reshape(context.shape[0], context.shape[1], self.txtlayers, self.txtdim)

    def _rope_tables(self, txt_len, h_tokens, w_tokens, batch_size, device):
        """Joint (text + image) RoPE table, cached on its shape key.

        The table is deterministic given (text length, token grid, batch, device)
        but costs a float64 CPU einsum per call; a sampling loop re-enters
        forward with identical shapes every step, so a one-slot cache avoids
        recomputing and re-uploading it per step.
        """
        key = (txt_len, h_tokens, w_tokens, batch_size, str(device))
        if self._freqs_cache is not None and self._freqs_cache[0] == key:
            return self._freqs_cache[1]
        txt_ids, img_ids = build_position_ids(txt_len, h_tokens, w_tokens, batch_size, device)
        freqs = self.pe_embedder(torch.cat([txt_ids, img_ids], dim=1)).movedim(1, 2).to(device)
        self._freqs_cache = (key, freqs)
        return freqs

    def forward(self, x, timesteps, context, transformer_options=None, **kwargs):
        """
        x: (B, channels, H, W) noised latent
        timesteps: (B,) raw flow sigma in [0, 1]
        context: (B, seq, txtlayers * txtdim) stacked text-encoder layer taps
        transformer_options: accepted and ignored (no block-patch hooks yet).

        Returns the prediction as-is (not negated), cropped to (B, channels, H, W).

        Raises:
            ValueError: `context` has the wrong fused feature width.
        """
        if transformer_options is None:
            transformer_options = {}
        B, _, H, W = x.shape
        x = pad_to_patch_size(x, self.patch)

        time_embedding = timestep_embedding(timesteps * self.time_factor, self.tdim)
        t = self.tmlp(time_embedding.unsqueeze(1).to(x.dtype))
        tvec = self.tproj(t)

        img_tokens, h_tokens, w_tokens = patchify(x, self.patch)
        img = self.first(img_tokens)

        txt = self.txtmlp(self.txtfusion(self._unpack_context(context).to(img.dtype)))
        txt_len = txt.shape[1]

        freqs = self._rope_tables(txt_len, h_tokens, w_tokens, B, img.device)
        combined = torch.cat([txt, img], dim=1)
        for block in self.blocks:
            combined = block(combined, tvec, freqs)

        img_out = self.last(combined, t)[:, txt_len:]
        img_out = unpatchify(img_out, h_tokens, w_tokens, self.patch, self.out_channels)
        return img_out[:, :, :H, :W]
