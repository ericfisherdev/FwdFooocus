"""Wan 2.1 3D causal VAE, the Krea 2 / Qwen Image / Anima latent codec.

Ported from ComfyUI's `comfy/ldm/wan/vae.py` (Apache-2.0; see
`ldm_patched/licenses-3rd/comfyui`), itself derived from
https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py
(Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.).
State-dict key names match `qwen_image_vae.safetensors` 1:1.

`WanVAE` is the native 5-D (B, C, T, H, W) module. `ImageModeWanVAE` narrows it
to the 4-D (B, C, H, W) contract used by `ldm_patched.modules.sd.VAE`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import ldm_patched.modules.ops
from ldm_patched.ldm.modules.diffusionmodules.model import vae_attention

ops = ldm_patched.modules.ops.disable_weight_init

CACHE_T = 2


def _frames_to_batch(x):
    """(B, C, T, H, W) -> (B*T, C, H, W): fold frames into the batch for 2-D ops."""
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)


def _batch_to_frames(x, t):
    """Inverse of `_frames_to_batch`: (B*T, C, H, W) -> (B, C, T, H, W)."""
    bt, c, h, w = x.shape
    return x.reshape(bt // t, t, c, h, w).permute(0, 2, 1, 3, 4)


def _cat_present(tensors, dim):
    """torch.cat that skips the None entries (absent feature cache)."""
    return torch.cat([t for t in tensors if t is not None], dim=dim)


class CausalConv3d(ops.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = 2 * self.padding[0]
        self.padding = (0, self.padding[1], self.padding[2])

    def forward(self, x, cache_x=None, cache_list=None, cache_idx=None):
        if cache_list is not None:
            cache_x = cache_list[cache_idx]
            cache_list[cache_idx] = None

        if self._padding > 0:
            padding_needed = self._padding
            if cache_x is not None:
                cache_x = cache_x.to(x.device)
                padding_needed = max(0, padding_needed - cache_x.shape[2])
            padding_shape = list(x.shape)
            padding_shape[2] = padding_needed
            padding = torch.zeros(padding_shape, device=x.device, dtype=x.dtype)
            x = _cat_present([padding, cache_x, x], dim=2)
            del cache_x

        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma.to(x) + (self.bias.to(x) if self.bias is not None else 0)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                ops.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                ops.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                ops.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                ops.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0], final=False):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :]
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = _frames_to_batch(x)
        x = self.resample(x)
        x = _batch_to_frames(x, t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x
                else:

                    cache_x = x[:, :, -1:, :, :]
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x

                    deferred_x = feat_cache[idx + 1]
                    if deferred_x is not None:
                        x = torch.cat([deferred_x, x], 2)
                        feat_cache[idx + 1] = None

                    if x.shape[2] == 1 and not final:
                        feat_cache[idx + 1] = x
                        x = None

                feat_idx[0] += 2
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0], final=False):
        old_x = x
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]
                x = layer(x, cache_list=feat_cache, cache_idx=idx)
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + self.shortcut(old_x)


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = ops.Conv2d(dim, dim * 3, 1)
        self.proj = ops.Conv2d(dim, dim, 1)
        self.optimized_attention = vae_attention()

    def forward(self, x, feat_cache=None, feat_idx=[0], final=False):
        identity = x
        b, c, t, h, w = x.size()
        x = _frames_to_batch(x)
        x = self.norm(x)
        # compute query, key, value

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        x = self.optimized_attention(q, k, v)

        # output
        x = self.proj(x)
        x = _batch_to_frames(x, t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 input_channels=3,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(input_channels, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0], final=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, final=final)
                if x is None:
                    return None
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, final=final)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 output_channels=3,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, output_channels, 3, padding=1))

    def run_up(self, layer_idx, x_ref, feat_cache, feat_idx, out_chunks):
        x = x_ref[0]
        x_ref[0] = None
        if layer_idx >= len(self.upsamples):
            for layer in self.head:
                if isinstance(layer, CausalConv3d) and feat_cache is not None:
                    cache_x = x[:, :, -CACHE_T:, :, :]
                    x = layer(x, feat_cache[feat_idx[0]])
                    feat_cache[feat_idx[0]] = cache_x
                    feat_idx[0] += 1
                else:
                    x = layer(x)
            out_chunks.append(x)
            return

        layer = self.upsamples[layer_idx]
        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

        if isinstance(layer, Resample) and layer.mode == 'upsample3d' and x.shape[2] > 2:
            for frame_idx in range(0, x.shape[2], 2):
                self.run_up(
                    layer_idx + 1,
                    [x[:, :, frame_idx:frame_idx + 2, :, :]],
                    feat_cache,
                    feat_idx.copy(),
                    out_chunks,
                )
            del x
            return

        next_x_ref = [x]
        del x
        self.run_up(layer_idx + 1, next_x_ref, feat_cache, feat_idx, out_chunks)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        out_chunks = []

        self.run_up(0, [x], feat_cache, feat_idx, out_chunks)
        return out_chunks


def count_cache_layers(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d) or (isinstance(m, Resample) and m.mode == 'downsample3d'):
            count += 1
    return count


class WanVAE(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 image_channels=3,
                 conv_out_channels=3,
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, image_channels, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, conv_out_channels, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

    def encode(self, x):
        conv_idx = [0]
        ## cache
        t = x.shape[2]
        t = 1 + ((t - 1) // 4) * 4
        iter_ = 1 + (t - 1) // 2
        feat_map = None
        if iter_ > 1:
            feat_map = [None] * count_cache_layers(self.encoder)
        ## split the input along time into chunks of 1, 2, 2, 2, ... frames (frame count floored to 4N+1 first)
        for i in range(iter_):
            conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=feat_map,
                    feat_idx=conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 2 * (i - 1):1 + 2 * i, :, :],
                    feat_cache=feat_map,
                    feat_idx=conv_idx,
                    final=(i == (iter_ - 1)))
                if out_ is None:
                    continue
                out = torch.cat([out, out_], 2)

        mu, log_var = self.conv1(out).chunk(2, dim=1)
        return mu

    def decode(self, z):
        # z: [b,c,t,h,w]
        iter_ = 1 + z.shape[2] // 2
        feat_map = None
        if iter_ > 1:
            feat_map = [None] * count_cache_layers(self.decoder)
        x = self.conv2(z)
        for i in range(iter_):
            conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=feat_map,
                    feat_idx=conv_idx)
            else:
                out_ = self.decoder(
                    x[:, :, 1 + 2 * (i - 1):1 + 2 * i, :, :],
                    feat_cache=feat_map,
                    feat_idx=conv_idx)
                out += out_
        return torch.cat(out, 2)


class ImageModeWanVAE(WanVAE):
    """`WanVAE` exposed through the 4-D image contract of `sd.VAE`.

    A still image is a single-frame video (T=1): one encoder/decoder iteration
    and no temporal feature cache, so the existing 4-D batching and tiling code
    drives it unchanged. Subclassed (rather than wrapped) so the state-dict keys
    stay identical to the checkpoint and to `WanVAE`'s.
    """

    def encode(self, x):
        """(B, C, H, W) pixels in [-1, 1] -> (B, z_dim, H/8, W/8) latent mean."""
        return super().encode(x.unsqueeze(2)).squeeze(2)

    def decode(self, z):
        """(B, z_dim, h, w) latent -> (B, C, h*8, w*8) pixels in [-1, 1]."""
        return super().decode(z.unsqueeze(2)).squeeze(2)


WAN21_VAE_DETECTION_KEY = "decoder.middle.0.residual.0.gamma"
WAN22_VAE_LAYOUT_KEY = "decoder.upsamples.0.upsamples.0.residual.2.weight"
WAN21_LATENT_CHANNELS = 16


def is_wan21_vae_state_dict(sd):
    """True for the Wan 2.1 layout (Krea 2 / Qwen Image / Anima VAE).

    The Wan 2.2 VAE shares the detection key but restructures the decoder
    upsamples; it is not supported and must not be mistaken for 2.1.
    """
    return WAN21_VAE_DETECTION_KEY in sd and WAN22_VAE_LAYOUT_KEY not in sd


def wan21_vae_config(sd):
    """`ImageModeWanVAE` kwargs derived from a Wan 2.1 VAE state dict.

    Shape-derived so RGB and RGBA variants both load: `dim` from the decoder
    head norm gamma, input/output channels from `encoder.conv1` / `decoder.head.2`.
    """
    return {
        "dim": sd["decoder.head.0.gamma"].shape[0],
        "z_dim": WAN21_LATENT_CHANNELS,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_scales": [],
        "temperal_downsample": [False, True, True],
        "image_channels": sd["encoder.conv1.weight"].shape[1],
        "conv_out_channels": sd["decoder.head.2.weight"].shape[0],
        "dropout": 0.0,
    }
