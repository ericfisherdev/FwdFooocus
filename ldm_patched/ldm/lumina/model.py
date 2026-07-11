"""NextDiT (Lumina2) diffusion backbone, used by Z-Image.

Port of Alpha-VLLM's Lumina-Image-2.0 `NextDiT`, parameterized for Tongyi-MAI's
Z-Image ("ZImageTransformer2DModel", dim=3840). State-dict key names below are
matched 1:1 against the real checkpoint layout, verified against ComfyUI's
`comfy/ldm/lumina/model.py` and `comfy/model_detection.py` (the latter reads
these exact keys off real downloaded Z-Image checkpoints to auto-configure the
model, which is why its literal key strings are trustworthy ground truth).

Only the plain (non pixel-space, non "omni"/ref-latents editing) forward path
is implemented -- x_embedder, cap_embedder, context_refiner, noise_refiner,
the main transformer `layers`, and `final_layer`. Nothing else builds on this
module yet; wiring it up to `supported_models`/`model_base` is a follow-up.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import ldm_patched.modules.ops
from ldm_patched.ldm.modules.attention import optimized_attention
from ldm_patched.ldm.modules.diffusionmodules.util import timestep_embedding

ops = ldm_patched.modules.ops.disable_weight_init


def clamp_fp16(x):
    # NextDiT's SwiGLU/attention branches can overflow fp16 range; the
    # upstream reference clamps at these same points to keep fp16 inference
    # numerically stable instead of producing NaNs.
    if x.dtype == torch.float16:
        return torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))


def pad_to_patch_size(x, patch_size):
    pad_h = (-x.shape[-2]) % patch_size
    pad_w = (-x.shape[-1]) % patch_size
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="circular")


def patchify(x, patch_size):
    B, C, H, W = x.shape
    ph = pw = patch_size
    h_tokens = H // ph
    w_tokens = W // pw
    x = x.view(B, C, h_tokens, ph, w_tokens, pw)
    x = x.permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
    return x, h_tokens, w_tokens


def unpatchify(x, h_tokens, w_tokens, patch_size, out_channels):
    B = x.shape[0]
    ph = pw = patch_size
    x = x.view(B, h_tokens, w_tokens, ph, pw, out_channels)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(B, out_channels, h_tokens * ph, w_tokens * pw)


def build_position_ids(cap_len, h_tokens, w_tokens, batch_size, device):
    """3-axis RoPE position ids: axis0 is a caption/frame-order axis, axis1/2
    are the 2D spatial (row, col) axes. Caption tokens get sequential axis0
    positions; image tokens all share one axis0 value (placing the whole
    image block right after the caption) and vary over axis1/axis2 instead.
    """
    cap_pos_ids = torch.zeros(batch_size, cap_len, 3, dtype=torch.float32, device=device)
    cap_pos_ids[:, :, 0] = torch.arange(cap_len, dtype=torch.float32, device=device) + 1.0

    img_pos_ids = torch.zeros(batch_size, h_tokens * w_tokens, 3, dtype=torch.float32, device=device)
    img_pos_ids[:, :, 0] = cap_len + 1
    img_pos_ids[:, :, 1] = torch.arange(h_tokens, dtype=torch.float32, device=device).view(-1, 1).repeat(1, w_tokens).flatten()
    img_pos_ids[:, :, 2] = torch.arange(w_tokens, dtype=torch.float32, device=device).view(1, -1).repeat(h_tokens, 1).flatten()
    return cap_pos_ids, img_pos_ids


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


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, output_size=None, max_period=10000,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        output_size = hidden_size if output_size is None else output_size
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size, output_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    def forward(self, t, dtype):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, max_period=self.max_period).to(dtype)
        return self.mlp(t_freq)


class JointAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, qk_norm, out_bias=False,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads

        self.qkv = operations.Linear(
            dim,
            (self.n_heads + self.n_kv_heads + self.n_kv_heads) * self.head_dim,
            bias=False, dtype=dtype, device=device,
        )
        self.out = operations.Linear(self.n_heads * self.head_dim, dim, bias=out_bias, dtype=dtype, device=device)

        if qk_norm:
            self.q_norm = operations.RMSNorm(self.head_dim, elementwise_affine=True, dtype=dtype, device=device)
            self.k_norm = operations.RMSNorm(self.head_dim, elementwise_affine=True, dtype=dtype, device=device)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x, freqs_cis, mask=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = torch.split(
            self.qkv(x),
            [self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = apply_rope(xq, xk, freqs_cis)

        if self.n_rep > 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, self.n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, self.n_rep, 1).flatten(2, 3)

        xq = xq.reshape(bsz, seqlen, self.n_heads * self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_heads * self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_heads * self.head_dim)

        out = optimized_attention(xq, xk, xv, self.n_heads, mask=mask)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)
        self.w2 = operations.Linear(hidden_dim, dim, bias=False, dtype=dtype, device=device)
        self.w3 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return self.w2(clamp_fp16(F.silu(self.w1(x)) * self.w3(x)))


class JointTransformerBlock(nn.Module):
    """Pre-norm + post-norm ("sandwich") transformer block. `modulation=False`
    (used by `context_refiner`) drops the adaLN-zero timestep conditioning
    entirely -- only `noise_refiner` and the main `layers` are time-modulated.
    """

    def __init__(self, dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, qk_norm,
                 modulation=True, mod_dim=256, attn_out_bias=False,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.attention = JointAttention(dim, n_heads, n_kv_heads, qk_norm, out_bias=attn_out_bias,
                                         dtype=dtype, device=device, operations=operations)
        self.feed_forward = FeedForward(dim, dim, multiple_of, ffn_dim_multiplier,
                                         dtype=dtype, device=device, operations=operations)

        self.attention_norm1 = operations.RMSNorm(dim, eps=norm_eps, elementwise_affine=True, dtype=dtype, device=device)
        self.ffn_norm1 = operations.RMSNorm(dim, eps=norm_eps, elementwise_affine=True, dtype=dtype, device=device)
        self.attention_norm2 = operations.RMSNorm(dim, eps=norm_eps, elementwise_affine=True, dtype=dtype, device=device)
        self.ffn_norm2 = operations.RMSNorm(dim, eps=norm_eps, elementwise_affine=True, dtype=dtype, device=device)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                operations.Linear(mod_dim, 4 * dim, bias=True, dtype=dtype, device=device),
            )

    def forward(self, x, freqs_cis, adaln_input=None, mask=None):
        if self.modulation:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)
            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                clamp_fp16(self.attention(modulate(self.attention_norm1(x), scale_msa), freqs_cis, mask=mask))
            )
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                clamp_fp16(self.feed_forward(modulate(self.ffn_norm1(x), scale_mlp)))
            )
        else:
            x = x + self.attention_norm2(
                clamp_fp16(self.attention(self.attention_norm1(x), freqs_cis, mask=mask))
            )
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_channels, mod_dim=256,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.norm_final = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(dim, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(mod_dim, dim, bias=True, dtype=dtype, device=device),
        )

    def forward(self, x, c):
        scale = self.adaLN_modulation(c)
        return self.linear(modulate(self.norm_final(x), scale))


class NextDiT(nn.Module):
    """Diffusion Transformer backbone shared by Lumina-Image-2.0 and Z-Image.

    Constructor defaults reflect Tongyi-MAI's Z-Image config (dim=3840,
    n_heads=n_kv_heads=30, axes_dims=[32,48,48], axes_lens=[1536,512,512],
    rope_theta=256.0, ffn_dim_multiplier=8/3, z_image_modulation=True,
    time_scale=1000.0) -- verified directly against ComfyUI's
    `model_detection.py`, which derives this same config from a real
    downloaded checkpoint's tensor shapes.
    """

    def __init__(
        self,
        patch_size=2,
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        multiple_of=256,
        ffn_dim_multiplier=8.0 / 3.0,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        axes_dims=(32, 48, 48),
        axes_lens=(1536, 512, 512),
        rope_theta=256.0,
        z_image_modulation=True,
        time_scale=1000.0,
        image_model=None,
        device=None,
        dtype=None,
        operations=ops,
        **kwargs,
    ):
        super().__init__()
        assert (dim // n_heads) == sum(axes_dims), \
            "head_dim ({}) must equal sum(axes_dims) ({})".format(dim // n_heads, sum(axes_dims))

        self.dtype = dtype
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.time_scale = time_scale
        self.dim = dim
        self.n_heads = n_heads

        mod_dim = min(dim, 256) if z_image_modulation else min(dim, 1024)
        t_hidden_size = min(dim, 1024)

        self.x_embedder = operations.Linear(
            patch_size * patch_size * in_channels, dim, bias=True, dtype=dtype, device=device,
        )

        self.t_embedder = TimestepEmbedder(
            t_hidden_size, output_size=mod_dim if z_image_modulation else None,
            dtype=dtype, device=device, operations=operations,
        )

        self.cap_embedder = nn.Sequential(
            operations.RMSNorm(cap_feat_dim, eps=norm_eps, elementwise_affine=True, dtype=dtype, device=device),
            operations.Linear(cap_feat_dim, dim, bias=True, dtype=dtype, device=device),
        )

        self.noise_refiner = nn.ModuleList([
            JointTransformerBlock(
                dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, qk_norm,
                modulation=True, mod_dim=mod_dim, dtype=dtype, device=device, operations=operations,
            )
            for _ in range(n_refiner_layers)
        ])
        self.context_refiner = nn.ModuleList([
            JointTransformerBlock(
                dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, qk_norm,
                modulation=False, dtype=dtype, device=device, operations=operations,
            )
            for _ in range(n_refiner_layers)
        ])
        self.layers = nn.ModuleList([
            JointTransformerBlock(
                dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, qk_norm,
                modulation=True, mod_dim=mod_dim, attn_out_bias=False,
                dtype=dtype, device=device, operations=operations,
            )
            for _ in range(n_layers)
        ])

        self.final_layer = FinalLayer(
            dim, patch_size, self.out_channels, mod_dim=mod_dim,
            dtype=dtype, device=device, operations=operations,
        )

        self.axes_dims = list(axes_dims)
        self.axes_lens = list(axes_lens)
        self.rope_embedder = EmbedND(dim // n_heads, theta=rope_theta, axes_dim=list(axes_dims))

    def forward(self, x, timesteps, context, **kwargs):
        """
        x: (B, in_channels, H, W) noised latent
        timesteps: (B,) flow-matching timestep in [0, 1]
        context: (B, cap_len, cap_feat_dim) caption hidden states (e.g. Qwen3-4B)
        """
        B, _, H, W = x.shape
        x = pad_to_patch_size(x, self.patch_size)

        # Z-Image's flow convention feeds (1 - t) into the embedder and the
        # backbone predicts -x0-velocity; both quirks are inherited unchanged
        # from the reference implementation so downstream samplers see the
        # same sign/direction they already expect from this checkpoint family.
        t = 1.0 - timesteps
        adaln_input = self.t_embedder(t * self.time_scale, dtype=x.dtype)

        cap_feats = self.cap_embedder(context)
        bsz, cap_len, _ = cap_feats.shape

        img_tokens, h_tokens, w_tokens = patchify(x, self.patch_size)
        img = self.x_embedder(img_tokens)

        cap_pos_ids, img_pos_ids = build_position_ids(cap_len, h_tokens, w_tokens, bsz, x.device)
        cap_freqs_cis = self.rope_embedder(cap_pos_ids).movedim(1, 2).to(img.device)
        img_freqs_cis = self.rope_embedder(img_pos_ids).movedim(1, 2).to(img.device)

        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_freqs_cis)

        for layer in self.noise_refiner:
            img = layer(img, img_freqs_cis, adaln_input=adaln_input)

        combined = torch.cat([cap_feats, img], dim=1)
        combined_freqs_cis = torch.cat([cap_freqs_cis, img_freqs_cis], dim=1)

        for layer in self.layers:
            combined = layer(combined, combined_freqs_cis, adaln_input=adaln_input)

        img_out = self.final_layer(combined[:, cap_len:], adaln_input)
        img_out = unpatchify(img_out, h_tokens, w_tokens, self.patch_size, self.out_channels)
        return -img_out[:, :, :H, :W]
