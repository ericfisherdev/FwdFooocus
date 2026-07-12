"""Z-Image DiT ControlNet backbone (FWDF-156).

Backport of ComfyUI's `comfy/ldm/lumina/controlnet.py` (`ZImage_Control`),
itself a from-scratch reimplementation of the block-injection mechanism used
by Alibaba/Tongyi-MAI's `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`
checkpoint (Apache 2.0). Verified directly against ComfyUI's real published
source (`Comfy-Org/ComfyUI` on GitHub) and the real checkpoint's safetensors
header (`control_layers.0..14`, `control_noise_refiner.0..1`,
`control_all_x_embedder.2-1.*` -- 15 control layers + 2 refiner layers, no
base-model weights bundled in), which is why the class shapes/names below
are trustworthy ground truth rather than guesses.

The checkpoint injects a Canny (or Depth/Pose/MLSD/HED/Scribble) hint into a
subset of NextDiT's transformer blocks via a small parallel stack of
"control blocks": each is a `JointTransformerBlock` clone with a
zero-initialized `after_proj` linear head (and, at block 0 only, a
zero-initialized `before_proj` that seeds the control stream from the main
model's own residual stream) -- the same "zero convolution" idea SDXL's
UNet ControlNet uses, translated to a transformer block, so an untrained/
unloaded control stack contributes exactly zero and cannot perturb
non-ControlNet generations.

Only the 15-control-layer + 2-refiner-layer variant (`refiner_control=True`)
matching the "-2.1" Union checkpoint this ticket targets is exercised by
FwdFooocus. The 3/6-layer "lite"/v1 variants are supported structurally
(mirroring ComfyUI, for config-driven fast-follows) but unverified here --
out of scope for this ticket's Canny-only rollout.
"""

import torch.nn as nn

import ldm_patched.modules.ops
from ldm_patched.ldm.lumina.model import JointTransformerBlock, pad_to_patch_size

ops = ldm_patched.modules.ops.disable_weight_init


class ZImageControlTransformerBlock(JointTransformerBlock):
    """One control-stack block: a plain `JointTransformerBlock` plus a
    zero-init `after_proj` (and, at block_id==0, `before_proj`) linear head.

    `forward(c, x, ...)` returns `(c_skip, c)`: `c_skip` is the
    zero-initialized residual ("hint") to inject into the main model at
    this block's target index, and `c` is the running control-stream state
    passed to the next control block.
    """

    def __init__(self, dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, qk_norm,
                 modulation=True, mod_dim=256, block_id=0,
                 dtype=None, device=None, operations=ops):
        super().__init__(dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, qk_norm,
                          modulation=modulation, mod_dim=mod_dim, dtype=dtype, device=device, operations=operations)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = operations.Linear(dim, dim, bias=True, dtype=dtype, device=device)
        self.after_proj = operations.Linear(dim, dim, bias=True, dtype=dtype, device=device)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        return c_skip, c


class ZImage_Control(nn.Module):
    """Holds the control-only weights from a Z-Image-Turbo-Fun-Controlnet
    checkpoint and exposes them as per-block hooks
    (`forward_control_block`/`forward_noise_refiner_block`) that
    `ldm_patched.modules.controlnet.ZImageControlNetPatch` drives one main
    NextDiT block at a time -- mirroring ComfyUI's
    `comfy_extras/nodes_model_patch.py:ZImageControlPatch` calling
    `comfy/ldm/lumina/controlnet.py:ZImage_Control`.

    `n_control_layers`/`refiner_control`/`additional_in_dim` are detected
    from the checkpoint's own key layout by
    `ldm_patched.modules.controlnet.load_controlnet_zimage`, not guessed.
    """

    def __init__(self, dim=3840, n_heads=30, n_kv_heads=30, multiple_of=256,
                 ffn_dim_multiplier=8.0 / 3.0, norm_eps=1e-5, qk_norm=True,
                 n_control_layers=6, control_in_dim=16, additional_in_dim=0,
                 refiner_control=False, mod_dim=None, dtype=None, device=None, operations=ops, **kwargs):
        super().__init__()
        self.control_in_dim = control_in_dim
        self.additional_in_dim = additional_in_dim
        self.n_control_layers = n_control_layers
        self.refiner_control = refiner_control
        n_refiner_layers = 2

        # NextDiT's own adaLN_modulation input width, given z_image_modulation
        # is always True for this checkpoint family (ldm_patched/ldm/lumina/model.py:
        # `mod_dim = min(dim, 256) if z_image_modulation else ...`). The
        # control blocks share the main model's adaln_input tensor directly
        # (see ZImageControlNetPatch), so their adaLN_modulation Linears must
        # be built with the exact same width or every real (dim=3840 ->
        # mod_dim=256) and tiny-test (dim<256 -> mod_dim=dim) case would
        # mismatch identically.
        if mod_dim is None:
            mod_dim = min(dim, 256)

        self.control_layers = nn.ModuleList([
            ZImageControlTransformerBlock(
                dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, qk_norm,
                mod_dim=mod_dim, block_id=i, dtype=dtype, device=device, operations=operations,
            )
            for i in range(n_control_layers)
        ])

        patch_size, f_patch_size = 2, 1
        x_embedder = operations.Linear(
            f_patch_size * patch_size * patch_size * (control_in_dim + additional_in_dim),
            dim, bias=True, dtype=dtype, device=device,
        )
        self.control_all_x_embedder = nn.ModuleDict({"{}-{}".format(patch_size, f_patch_size): x_embedder})

        if refiner_control:
            self.control_noise_refiner = nn.ModuleList([
                ZImageControlTransformerBlock(
                    dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, qk_norm,
                    modulation=True, mod_dim=mod_dim, block_id=layer_id, dtype=dtype, device=device, operations=operations,
                )
                for layer_id in range(n_refiner_layers)
            ])
        else:
            self.control_noise_refiner = nn.ModuleList([
                JointTransformerBlock(
                    dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, qk_norm,
                    modulation=True, mod_dim=mod_dim, dtype=dtype, device=device, operations=operations,
                )
                for _ in range(n_refiner_layers)
            ])

    def embed_hint(self, control_context, freqs_cis=None, adaln_input=None):
        """Patchify + embed the (VAE-encoded) control hint latent.

        `control_context` is padded to `patch_size` first, mirroring
        `NextDiT.forward()`'s own `x = pad_to_patch_size(x, self.patch_size)`
        call (`ldm_patched/ldm/lumina/model.py`) exactly -- this hint latent
        and the main model's noise latent are both VAE-encoded from the same
        pixel-space width/height (`modules/async_worker.py`'s
        `apply_control_nets()` resizes the hint image to the generation's own
        width/height before encoding), so padding both with the identical
        function/patch_size keeps their resulting (h_tokens, w_tokens) grids
        identical. Without this, a control latent whose H/W are not already
        multiples of `patch_size` (e.g. from an odd `overwrite_width`/
        `overwrite_height` override) crashes `.view()` below outright, or --
        with a naive `.reshape()` instead -- would silently produce a
        misaligned token grid that no longer lines up with NextDiT's own.

        For `refiner_control=False` checkpoints, the embedded tokens are
        fully refined right here via the plain `control_noise_refiner`
        blocks (which require `freqs_cis`/`adaln_input`, matching the
        real Z-Image-Turbo generation's own tokens). For
        `refiner_control=True` (this ticket's target checkpoint), refinement
        instead happens block-by-block via `forward_noise_refiner_block`,
        interleaved with the main model's own noise_refiner loop.
        """
        patch_size, f_patch_size = 2, 1
        control_context = pad_to_patch_size(control_context, patch_size)
        pH = pW = patch_size
        B, C, H, W = control_context.shape
        # reshape, not view: mirrors ldm_patched.ldm.lumina.model.patchify()'s
        # own comment -- the padded/concatenated latent may be a
        # non-contiguous strided tensor, which view() rejects at runtime.
        tokens = (
            control_context.reshape(B, C, H // pH, pH, W // pW, pW)
            .permute(0, 2, 4, 3, 5, 1)
            .flatten(3)
            .flatten(1, 2)
        )
        control_context = self.control_all_x_embedder["{}-{}".format(patch_size, f_patch_size)](tokens)
        if not self.refiner_control:
            for layer in self.control_noise_refiner:
                control_context = layer(control_context, freqs_cis=freqs_cis, adaln_input=adaln_input, mask=None)
        return control_context

    def _sliced_freqs(self, freqs_cis, control_context):
        if freqs_cis is None:
            return None
        return freqs_cis[:control_context.shape[0], :control_context.shape[1]]

    def forward_noise_refiner_block(self, layer_id, control_context, x, freqs_cis, adaln_input):
        """Advance one `control_noise_refiner` block. Returns `(hint, next_control_context)`;
        `hint` is `None` for `refiner_control=False` checkpoints (no per-refiner-block
        injection in that variant -- see `embed_hint`).
        """
        if not self.refiner_control:
            return None, control_context
        fc = self._sliced_freqs(freqs_cis, control_context)
        return self.control_noise_refiner[layer_id](control_context, x, freqs_cis=fc, adaln_input=adaln_input, mask=None)

    def forward_control_block(self, layer_id, control_context, x, freqs_cis, adaln_input):
        """Advance one `control_layers` block. Returns `(hint, next_control_context)`."""
        fc = self._sliced_freqs(freqs_cis, control_context)
        return self.control_layers[layer_id](control_context, x, freqs_cis=fc, adaln_input=adaln_input, mask=None)
