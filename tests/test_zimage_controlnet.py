"""Unit tests for the Z-Image DiT ControlNet backport (FWDF-156):
- ldm_patched/ldm/lumina/model.py's NextDiT.forward() double_block/noise_refiner hooks
- ldm_patched/ldm/lumina/controlnet.py's ZImageControlTransformerBlock/ZImage_Control
- ldm_patched/modules/controlnet.py's ZImageControlNetPatch/load_controlnet_zimage
"""

import copy
import inspect
import unittest

import torch

from ldm_patched.ldm.lumina.model import EmbedND, JointTransformerBlock, NextDiT
from ldm_patched.ldm.lumina.controlnet import ZImage_Control, ZImageControlTransformerBlock
from ldm_patched.modules.controlnet import (
    ZImageControlNetPatch,
    _detect_zimage_controlnet_config,
    _z_image_controlnet_convert,
)
from ldm_patched.modules.model_patcher import ModelPatcher


def make_tiny_dit_config():
    """Structurally faithful but tiny NextDiT config, sized to match a tiny
    ZImage_Control below (same dim/n_heads/n_refiner_layers=2 -- the real
    checkpoint's noise_refiner count is fixed at 2 regardless of model size,
    see ZImage_Control.__init__).
    """
    return dict(
        patch_size=2, in_channels=4, dim=64, n_layers=4, n_refiner_layers=2,
        n_heads=2, n_kv_heads=2, multiple_of=8, ffn_dim_multiplier=2.0,
        norm_eps=1e-5, qk_norm=True, cap_feat_dim=32,
        axes_dims=(8, 12, 12), axes_lens=(16, 8, 8), rope_theta=256.0,
        z_image_modulation=True, time_scale=1000.0,
    )


def make_tiny_control_config(**overrides):
    cfg = dict(
        dim=64, n_heads=2, n_kv_heads=2, multiple_of=8, ffn_dim_multiplier=2.0,
        norm_eps=1e-5, qk_norm=True, n_control_layers=4, control_in_dim=4,
        additional_in_dim=0, refiner_control=True,
    )
    cfg.update(overrides)
    return cfg


def make_identity_freqs_cis(batch_size, seq_len, head_dim=32):
    """An all-zero-position RoPE table -- an identity rotation regardless of
    theta/axes breakdown (see test_lumina_nextdit.py's own
    test_apply_rope_identity_at_zero_position) -- for tests that need a
    structurally valid freqs_cis argument but aren't testing RoPE math
    itself.
    """
    embedder = EmbedND(head_dim, theta=256.0, axes_dim=[head_dim // 4, head_dim * 3 // 8, head_dim * 3 // 8])
    ids = torch.zeros(batch_size, seq_len, 3)
    return embedder(ids).movedim(1, 2)  # (B, seq, 1, head_dim // 2, 2, 2)


def init_small_weights(model):
    """See test_lumina_nextdit.py's identical helper: ldm_patched's ops
    intentionally skip random init (real usage always loads a trained
    checkpoint immediately), so tests running a bare forward pass must
    initialize weights themselves.
    """
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(mean=0.0, std=0.02)


class _StubVAE:
    """Deterministic VAE stand-in: encode() maps a pixel image to a small
    latent of the requested channel count at a fixed spatial size, so
    ZImageControlNetPatch's hint-encoding path has realistic,
    patch-size-compatible shapes to operate on without a real VAE.
    """

    def __init__(self, control_in_dim, latent_hw):
        self.control_in_dim = control_in_dim
        self.latent_hw = latent_hw

    def encode(self, pixels):
        b = pixels.shape[0]
        return torch.randn(b, self.control_in_dim, self.latent_hw, self.latent_hw)


class TestZImageControlTransformerBlock(unittest.TestCase):
    def test_block_zero_has_before_proj_others_do_not(self):
        block0 = ZImageControlTransformerBlock(64, 2, 2, 8, 2.0, 1e-5, True, block_id=0)
        block1 = ZImageControlTransformerBlock(64, 2, 2, 8, 2.0, 1e-5, True, block_id=1)
        self.assertTrue(hasattr(block0, "before_proj"))
        self.assertFalse(hasattr(block1, "before_proj"))
        self.assertTrue(hasattr(block0, "after_proj"))
        self.assertTrue(hasattr(block1, "after_proj"))

    def test_forward_returns_hint_and_running_context_with_matching_shapes(self):
        torch.manual_seed(0)
        # mod_dim defaults to 256 (matching the real dim=3840 checkpoint,
        # where min(3840, 256) == 256) even though dim=64 here.
        block0 = ZImageControlTransformerBlock(64, 2, 2, 8, 2.0, 1e-5, True, block_id=0)
        init_small_weights(block0)
        block0.eval()
        c = torch.randn(1, 6, 64)
        x = torch.randn(1, 6, 64)
        adaln = torch.randn(1, 256)
        freqs_cis = make_identity_freqs_cis(1, 6)
        with torch.no_grad():
            hint, c_out = block0(c, x, freqs_cis=freqs_cis, adaln_input=adaln)
        self.assertEqual(hint.shape, c.shape)
        self.assertEqual(c_out.shape, c.shape)

    def test_block_zero_uses_x_as_residual_seed(self):
        # Changing `x` (only used at block_id==0) must change the output;
        # otherwise before_proj's contribution is silently dropped.
        torch.manual_seed(0)
        block0 = ZImageControlTransformerBlock(64, 2, 2, 8, 2.0, 1e-5, True, block_id=0)
        init_small_weights(block0)
        block0.eval()
        c = torch.randn(1, 6, 64)
        adaln = torch.randn(1, 256)
        freqs_cis = make_identity_freqs_cis(1, 6)
        with torch.no_grad():
            hint_a, _ = block0(c, torch.randn(1, 6, 64), freqs_cis=freqs_cis, adaln_input=adaln)
            hint_b, _ = block0(c, torch.randn(1, 6, 64), freqs_cis=freqs_cis, adaln_input=adaln)
        self.assertFalse(torch.allclose(hint_a, hint_b))


class TestZImageControlModule(unittest.TestCase):
    def test_construction_layer_counts(self):
        control = ZImage_Control(**make_tiny_control_config())
        self.assertEqual(len(control.control_layers), 4)
        self.assertEqual(len(control.control_noise_refiner), 2)
        self.assertIsInstance(control.control_noise_refiner[0], ZImageControlTransformerBlock)

    def test_refiner_control_false_uses_plain_joint_transformer_blocks(self):
        control = ZImage_Control(**make_tiny_control_config(refiner_control=False))
        self.assertIsInstance(control.control_noise_refiner[0], JointTransformerBlock)
        self.assertNotIsInstance(control.control_noise_refiner[0], ZImageControlTransformerBlock)

    def test_embed_hint_patchifies_and_embeds(self):
        control = ZImage_Control(**make_tiny_control_config())
        init_small_weights(control)
        control.eval()
        control_context = torch.randn(1, 4, 4, 4)  # (B, control_in_dim, H, W)
        with torch.no_grad():
            embedded = control.embed_hint(control_context)
        # patch_size=2 over a 4x4 grid -> 2*2 = 4 tokens.
        self.assertEqual(embedded.shape, (1, 4, 64))

    def test_forward_control_block_returns_hint_and_context_with_matching_shapes(self):
        control = ZImage_Control(**make_tiny_control_config())
        init_small_weights(control)
        control.eval()
        control_context = torch.randn(1, 4, 64)
        x = torch.randn(1, 4, 64)
        # dim=64 -> mod_dim = min(64, 256) = 64, per ZImage_Control's own
        # z_image_modulation-matching default (see its __init__ docstring).
        adaln = torch.randn(1, 64)
        freqs_cis = make_identity_freqs_cis(1, 4)
        with torch.no_grad():
            hint, next_ctx = control.forward_control_block(0, control_context, x, freqs_cis, adaln)
        self.assertEqual(hint.shape, control_context.shape)
        self.assertEqual(next_ctx.shape, control_context.shape)

    def test_forward_noise_refiner_block_is_noop_when_refiner_control_false(self):
        control = ZImage_Control(**make_tiny_control_config(refiner_control=False))
        init_small_weights(control)
        control.eval()
        control_context = torch.randn(1, 4, 64)
        x = torch.randn(1, 4, 64)
        with torch.no_grad():
            hint, next_ctx = control.forward_noise_refiner_block(0, control_context, x, None, None)
        self.assertIsNone(hint)
        self.assertIs(next_ctx, control_context)

    def test_embed_hint_with_additional_in_dim_matches_real_union_2_1_shape(self):
        # additional_in_dim=17 is the real "-2.1" Union checkpoint's shape
        # (mask(1ch) + inpaint_latent(16ch), see
        # ldm_patched.modules.controlnet._encode_zimage_control_hint) --
        # the x_embedder's input width must grow by exactly that much
        # beyond control_in_dim's base channel count.
        control = ZImage_Control(**make_tiny_control_config(control_in_dim=16, additional_in_dim=17))
        init_small_weights(control)
        control.eval()
        control_context = torch.randn(1, 16 + 17, 4, 4)  # (B, control_in_dim + additional_in_dim, H, W)
        with torch.no_grad():
            embedded = control.embed_hint(control_context)
        # patch_size=2 over a 4x4 grid -> 2*2 = 4 tokens, at dim=64 regardless
        # of the input channel count (x_embedder projects into dim).
        self.assertEqual(embedded.shape, (1, 4, 64))

    def test_forward_control_block_after_additional_in_dim_embedding(self):
        # Exercises the extra-channel path through embed_hint() and into a
        # real control-block forward pass, not just the embedding shape.
        control = ZImage_Control(**make_tiny_control_config(control_in_dim=16, additional_in_dim=17))
        init_small_weights(control)
        control.eval()
        control_context = torch.randn(1, 16 + 17, 4, 4)
        x = torch.randn(1, 4, 64)
        adaln = torch.randn(1, 64)
        freqs_cis = make_identity_freqs_cis(1, 4)
        with torch.no_grad():
            embedded = control.embed_hint(control_context)
            hint, next_ctx = control.forward_control_block(0, embedded, x, freqs_cis, adaln)
        self.assertEqual(hint.shape, embedded.shape)
        self.assertEqual(next_ctx.shape, embedded.shape)

    def test_embed_hint_pads_non_patch_aligned_shape_instead_of_crashing(self):
        # A control latent whose H/W are not already multiples of patch_size
        # (e.g. from an odd overwrite_width/overwrite_height override
        # producing a non-8-aligned VAE encode) used to crash embed_hint()'s
        # .view() outright. embed_hint() now pads first via
        # ldm_patched.ldm.lumina.model.pad_to_patch_size(), mirroring
        # NextDiT.forward()'s own call, so this must not raise and must
        # produce the ceil-to-even token grid instead.
        control = ZImage_Control(**make_tiny_control_config())
        init_small_weights(control)
        control.eval()
        control_context = torch.randn(1, 4, 5, 5)  # odd H/W, patch_size=2
        with torch.no_grad():
            embedded = control.embed_hint(control_context)
        # pad_to_patch_size pads 5 -> 6 on each axis (circular, patch_size=2),
        # then patch_size=2 over a 6x6 grid -> 3*3 = 9 tokens.
        self.assertEqual(embedded.shape, (1, 9, 64))

    def test_embed_hint_is_a_noop_pad_for_already_aligned_shapes(self):
        # Regression guard: the padding call must not change behavior for
        # the common (already-aligned) case existing checkpoints hit today.
        control = ZImage_Control(**make_tiny_control_config())
        init_small_weights(control)
        control.eval()
        control_context = torch.randn(1, 4, 4, 4)
        with torch.no_grad():
            embedded = control.embed_hint(control_context)
        self.assertEqual(embedded.shape, (1, 4, 64))


class TestNextDiTControlNetHooks(unittest.TestCase):
    """Verifies NextDiT.forward()'s double_block/noise_refiner patch hooks in
    isolation, with plain spy functions (not a real ZImage_Control) so the
    model.py wiring itself is pinned down independently of the patch class.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.config = make_tiny_dit_config()
        self.model = NextDiT(**self.config)
        init_small_weights(self.model)
        self.model.eval()

    def _inputs(self, batch_size=1, h=8, w=8, cap_len=3):
        x = torch.randn(batch_size, self.config["in_channels"], h, w)
        timesteps = torch.rand(batch_size)
        context = torch.randn(batch_size, cap_len, self.config["cap_feat_dim"])
        return x, timesteps, context

    def test_no_patches_is_a_strict_noop(self):
        x, t, ctx = self._inputs()
        with torch.no_grad():
            out_bare = self.model(x, t, ctx)
            out_empty_options = self.model(x, t, ctx, transformer_options={})
            out_empty_patches = self.model(x, t, ctx, transformer_options={"patches": {}})
        torch.testing.assert_close(out_bare, out_empty_options)
        torch.testing.assert_close(out_bare, out_empty_patches)

    def test_transformer_options_default_is_none_not_a_shared_mutable_dict(self):
        # Structural regression for Ruff B006: forward() writes a
        # "total_blocks" key into transformer_options a few lines in, so a
        # `transformer_options={}` default argument would bind ONE dict
        # object (evaluated once at function-definition time) to every call
        # that omits this argument, shared and mutated across all of them.
        # The default must be the immutable sentinel None, normalized to a
        # fresh {} inside the function body instead.
        sig = inspect.signature(NextDiT.forward)
        self.assertIsNone(sig.parameters["transformer_options"].default)

    def test_repeated_bare_calls_produce_identical_deterministic_output(self):
        # Behavioral sanity check for the omitted-argument path: each bare
        # call must build its own {} internally and still produce the same
        # (patch-free) output as any other, rather than erroring or drifting
        # from some leftover state in a shared default dict.
        x, t, ctx = self._inputs()
        with torch.no_grad():
            out_a = self.model(x, t, ctx)
            out_b = self.model(x, t, ctx)
        torch.testing.assert_close(out_a, out_b)

    def test_double_block_patch_called_once_per_main_layer_in_order(self):
        x, t, ctx = self._inputs()
        calls = []

        def spy(kwargs):
            calls.append(kwargs["block_index"])
            self.assertEqual(kwargs["block_type"], "double")
            return {}

        with torch.no_grad():
            self.model(x, t, ctx, transformer_options={"patches": {"double_block": [spy]}})

        self.assertEqual(calls, list(range(self.config["n_layers"])))

    def test_noise_refiner_patch_called_once_per_refiner_layer_in_order(self):
        x, t, ctx = self._inputs()
        calls = []

        def spy(kwargs):
            calls.append(kwargs["block_index"])
            self.assertEqual(kwargs["block_type"], "noise_refiner")
            return {}

        with torch.no_grad():
            self.model(x, t, ctx, transformer_options={"patches": {"noise_refiner": [spy]}})

        self.assertEqual(calls, list(range(self.config["n_refiner_layers"])))

    def test_double_block_patch_receives_correctly_shaped_img_and_txt(self):
        h_tokens = w_tokens = 4  # h=w=8, patch_size=2
        img_len = h_tokens * w_tokens
        cap_len = 3
        x, t, ctx = self._inputs(h=8, w=8, cap_len=cap_len)
        seen = []

        def spy(kwargs):
            seen.append((tuple(kwargs["img"].shape), tuple(kwargs["txt"].shape), tuple(kwargs["pe"].shape)))
            return {}

        with torch.no_grad():
            self.model(x, t, ctx, transformer_options={"patches": {"double_block": [spy]}})

        self.assertEqual(len(seen), self.config["n_layers"])
        for img_shape, txt_shape, pe_shape in seen:
            self.assertEqual(img_shape, (1, img_len, self.config["dim"]))
            self.assertEqual(txt_shape, (1, cap_len, self.config["dim"]))
            self.assertEqual(pe_shape[:2], (1, img_len))  # (B, seq, ...) image-only RoPE table

    def test_double_block_patch_img_input_is_frozen_across_calls(self):
        x, t, ctx = self._inputs()
        seen_inputs = []

        def spy(kwargs):
            seen_inputs.append(kwargs["img_input"].clone())
            return {}

        with torch.no_grad():
            self.model(x, t, ctx, transformer_options={"patches": {"double_block": [spy]}})

        for tensor in seen_inputs[1:]:
            torch.testing.assert_close(tensor, seen_inputs[0])

    def test_returned_img_replaces_running_tensor_and_changes_output(self):
        x, t, ctx = self._inputs()

        def add_one(kwargs):
            return {"img": kwargs["img"] + 1.0}

        with torch.no_grad():
            baseline = self.model(x, t, ctx)
            patched = self.model(x, t, ctx, transformer_options={"patches": {"double_block": [add_one]}})

        self.assertEqual(patched.shape, baseline.shape)
        self.assertFalse(torch.allclose(baseline, patched))

    def test_txt_replacement_is_honored(self):
        x, t, ctx = self._inputs()

        def zero_txt(kwargs):
            return {"txt": torch.zeros_like(kwargs["txt"])}

        with torch.no_grad():
            baseline = self.model(x, t, ctx)
            patched = self.model(x, t, ctx, transformer_options={"patches": {"double_block": [zero_txt]}})

        self.assertFalse(torch.allclose(baseline, patched))


class TestZImageControlNetPatchIntegration(unittest.TestCase):
    """End-to-end: a real ZImage_Control + ZImageControlNetPatch driving a
    real (tiny) NextDiT through its own forward() hooks -- the same call
    shape production code uses (core.apply_controlnet_zimage ->
    ModelPatcher.set_model_double_block_patch/set_model_noise_refiner_patch
    -> NextDiT.forward()'s patches dict).
    """

    def setUp(self):
        torch.manual_seed(0)
        self.dit_config = make_tiny_dit_config()
        self.model = NextDiT(**self.dit_config)
        init_small_weights(self.model)
        self.model.eval()

        self.control = ZImage_Control(**make_tiny_control_config())
        init_small_weights(self.control)
        self.control.eval()

        self.vae = _StubVAE(control_in_dim=4, latent_hw=4)
        self.image = torch.rand(1, 8, 8, 3)

    def _inputs(self):
        x = torch.randn(1, self.dit_config["in_channels"], 8, 8)
        timesteps = torch.rand(1)
        context = torch.randn(1, 3, self.dit_config["cap_feat_dim"])
        return x, timesteps, context

    def _patched_transformer_options(self, strength=1.0):
        patch = ZImageControlNetPatch(self.control, self.vae, self.image, strength=strength)
        return patch, {"patches": {"double_block": [patch], "noise_refiner": [patch]}}

    def test_patched_output_differs_from_unpatched_and_has_correct_shape(self):
        x, t, ctx = self._inputs()
        with torch.no_grad():
            baseline = self.model(x, t, ctx)

        _, transformer_options = self._patched_transformer_options(strength=1.0)
        with torch.no_grad():
            patched = self.model(x, t, ctx, transformer_options=transformer_options)

        self.assertEqual(patched.shape, baseline.shape)
        self.assertFalse(torch.isnan(patched).any().item())
        self.assertFalse(torch.isinf(patched).any().item())
        self.assertFalse(torch.allclose(baseline, patched))

    def test_zero_strength_matches_unpatched_baseline(self):
        x, t, ctx = self._inputs()
        _, transformer_options = self._patched_transformer_options(strength=0.0)
        with torch.no_grad():
            baseline = self.model(x, t, ctx)
            zero_strength = self.model(x, t, ctx, transformer_options=transformer_options)
        torch.testing.assert_close(baseline, zero_strength)

    def test_state_fully_drains_after_one_forward_pass(self):
        x, t, ctx = self._inputs()
        patch, transformer_options = self._patched_transformer_options(strength=1.0)
        with torch.no_grad():
            self.model(x, t, ctx, transformer_options=transformer_options)
        self.assertIsNone(patch._state)

    def test_repeated_forward_passes_do_not_error_or_leak_state(self):
        # Simulates multiple denoising timesteps reusing the same patch
        # object (a real sampling loop calls NextDiT.forward() once per step
        # without recreating the ControlNet patch).
        patch, transformer_options = self._patched_transformer_options(strength=1.0)
        outputs = []
        with torch.no_grad():
            for _ in range(3):
                x, t, ctx = self._inputs()
                outputs.append(self.model(x, t, ctx, transformer_options=transformer_options))
        for out in outputs:
            self.assertFalse(torch.isnan(out).any().item())
        self.assertIsNone(patch._state)

    def test_non_patch_aligned_hint_latent_does_not_crash_the_call_path(self):
        # End-to-end regression for the embed_hint() padding fix: a VAE
        # whose encode() returns a latent not already aligned to patch_size
        # (e.g. from an odd overwrite_width/overwrite_height override) used
        # to crash inside ZImageControlNetPatch.__call__ -> embed_hint()'s
        # .view(). Drives the same call path production code uses
        # (core.apply_controlnet_zimage's patch, through NextDiT.forward()'s
        # own hooks) with such a VAE and asserts it completes cleanly.
        odd_vae = _StubVAE(control_in_dim=4, latent_hw=5)
        patch = ZImageControlNetPatch(self.control, odd_vae, self.image, strength=1.0)
        transformer_options = {"patches": {"double_block": [patch], "noise_refiner": [patch]}}

        x, t, ctx = self._inputs()
        with torch.no_grad():
            out = self.model(x, t, ctx, transformer_options=transformer_options)

        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any().item())
        self.assertFalse(torch.isinf(out).any().item())


class TestZImageControlNetPatchDeepcopy(unittest.TestCase):
    """ModelPatcher.clone() (ldm_patched/modules/model_patcher.py) deep-copies
    model_options wholesale, which would otherwise deep-copy a registered
    ZImageControlNetPatch too -- duplicating its control_model (a full
    nn.Module) and _encoded_hint tensor on every clone. __deepcopy__ makes
    this patch return itself instead.
    """

    def setUp(self):
        self.control = ZImage_Control(**make_tiny_control_config())
        self.vae = _StubVAE(control_in_dim=4, latent_hw=4)
        self.image = torch.rand(1, 8, 8, 3)
        self.patch = ZImageControlNetPatch(self.control, self.vae, self.image, strength=1.0)

    def test_deepcopy_returns_the_same_instance(self):
        copied = copy.deepcopy(self.patch)
        self.assertIs(copied, self.patch)

    def test_deepcopy_inside_a_container_also_returns_the_same_instance(self):
        # ModelPatcher.clone() deep-copies model_options as a whole nested
        # structure (dicts/lists containing this patch), not the patch
        # object directly -- exercise that shape.
        container = {"patches": {"double_block": [self.patch]}}
        copied_container = copy.deepcopy(container)
        self.assertIs(copied_container["patches"]["double_block"][0], self.patch)
        self.assertIsNot(copied_container, container)
        self.assertIsNot(copied_container["patches"]["double_block"], container["patches"]["double_block"])

    def test_cloning_a_patched_model_patcher_shares_the_control_module(self):
        model = torch.nn.Linear(2, 2)
        device = torch.device("cpu")
        original = ModelPatcher(model, load_device=device, offload_device=device)
        original.set_model_double_block_patch(self.patch)
        original.set_model_noise_refiner_patch(self.patch)

        cloned = original.clone()

        cloned_double_block_patch = cloned.model_options["transformer_options"]["patches"]["double_block"][0]
        cloned_noise_refiner_patch = cloned.model_options["transformer_options"]["patches"]["noise_refiner"][0]
        self.assertIs(cloned_double_block_patch, self.patch)
        self.assertIs(cloned_noise_refiner_patch, self.patch)
        self.assertIs(cloned_double_block_patch.control_model, self.control)
        # The two hook lists themselves must still be independent containers
        # (clone()'s deepcopy of model_options is otherwise unaffected --
        # only the patch object inside is shared, not everything around it).
        self.assertIsNot(
            cloned.model_options["transformer_options"]["patches"],
            original.model_options["transformer_options"]["patches"],
        )


class TestZImageControlNetPatchTo(unittest.TestCase):
    """ModelPatcher.model_patches_to() (ldm_patched/modules/model_patcher.py)
    calls .to() on every registered patch with either a torch.device or --
    for the model's own dtype pass, ldm_patched/modules/model_management.py's
    `model_patches_to(self.model.model_dtype())` -- a torch.dtype. Before
    this fix, only the torch.device branch was handled; a dtype-only or
    string-device move was a silent no-op.
    """

    def setUp(self):
        self.control = ZImage_Control(**make_tiny_control_config())
        self.vae = _StubVAE(control_in_dim=4, latent_hw=4)
        self.image = torch.rand(1, 8, 8, 3)
        self.patch = ZImageControlNetPatch(self.control, self.vae, self.image, strength=1.0)

    def test_dtype_argument_casts_encoded_hint_and_is_not_a_noop(self):
        self.assertEqual(self.patch._encoded_hint.dtype, torch.float32)

        result = self.patch.to(torch.float64)

        self.assertIs(result, self.patch)
        self.assertEqual(self.patch._encoded_hint.dtype, torch.float64)

    def test_dtype_argument_does_not_clear_in_flight_state(self):
        self.patch._state = (0, (None, torch.zeros(1, 4, 64)))

        self.patch.to(torch.float64)

        self.assertIsNotNone(self.patch._state)

    def test_string_device_argument_moves_encoded_hint_and_resets_state(self):
        self.patch._state = (0, (None, torch.zeros(1, 4, 64)))

        result = self.patch.to("cpu")

        self.assertIs(result, self.patch)
        self.assertEqual(self.patch._encoded_hint.device, torch.device("cpu"))
        self.assertIsNone(self.patch._state)

    def test_torch_device_argument_resets_state(self):
        self.patch._state = (0, (None, torch.zeros(1, 4, 64)))

        self.patch.to(torch.device("cpu"))

        self.assertIsNone(self.patch._state)


class TestZImageControlNetConvert(unittest.TestCase):
    def test_convert_fuses_qkv_in_qkv_order_and_renames_attention_keys(self):
        sd = {
            "control_layers.0.attention.to_q.weight": torch.zeros(4, 4),
            "control_layers.0.attention.to_k.weight": torch.ones(4, 4),
            "control_layers.0.attention.to_v.weight": torch.full((4, 4), 2.0),
            "control_layers.0.attention.to_out.0.weight": torch.full((4, 4), 3.0),
            "control_layers.0.attention.to_out.0.bias": torch.full((4,), 4.0),
            "control_layers.0.attention.norm_q.weight": torch.full((4,), 5.0),
            "control_layers.0.attention.norm_k.weight": torch.full((4,), 6.0),
        }

        out = _z_image_controlnet_convert(sd)

        self.assertIn("control_layers.0.attention.qkv.weight", out)
        self.assertNotIn("control_layers.0.attention.to_q.weight", out)
        self.assertNotIn("control_layers.0.attention.to_k.weight", out)
        self.assertNotIn("control_layers.0.attention.to_v.weight", out)

        qkv = out["control_layers.0.attention.qkv.weight"]
        self.assertEqual(tuple(qkv.shape), (12, 4))
        torch.testing.assert_close(qkv[0:4], torch.zeros(4, 4))
        torch.testing.assert_close(qkv[4:8], torch.ones(4, 4))
        torch.testing.assert_close(qkv[8:12], torch.full((4, 4), 2.0))

        self.assertIn("control_layers.0.attention.out.weight", out)
        self.assertIn("control_layers.0.attention.out.bias", out)
        self.assertIn("control_layers.0.attention.q_norm.weight", out)
        self.assertIn("control_layers.0.attention.k_norm.weight", out)


class TestDetectZImageControlnetConfig(unittest.TestCase):
    """Pure, cheap unit tests for the checkpoint-key-layout detection
    load_controlnet_zimage uses -- kept separate from model construction
    (ZImage_Control's real-scale defaults are dim=3840/30 heads, too large
    to instantiate repeatedly in a fast unit test) so this branching logic
    is exercised directly against tiny synthetic key sets.
    """

    def test_fifteen_layer_union_checkpoint(self):
        sd = {
            "control_layers.4.adaLN_modulation.0.weight": None,
            "control_layers.14.adaLN_modulation.0.weight": None,
        }
        config = _detect_zimage_controlnet_config(sd)
        self.assertEqual(config, {"n_control_layers": 15, "additional_in_dim": 17, "refiner_control": True})

    def test_three_layer_lite_checkpoint(self):
        sd = {"control_layers.0.adaLN_modulation.0.weight": None}
        config = _detect_zimage_controlnet_config(sd)
        self.assertEqual(config, {"n_control_layers": 3, "additional_in_dim": 17, "refiner_control": True})

    def test_six_layer_v1_checkpoint_falls_back_to_class_defaults(self):
        # Has layer 4 (so not "lite") but not layer 14 (so not the 15-layer
        # Union) -- the real v1.0 "Union" (non-"-2.1") checkpoint's shape,
        # which ZImage_Control's own defaults (n_control_layers=6,
        # refiner_control=False) already match.
        sd = {"control_layers.4.adaLN_modulation.0.weight": None}
        config = _detect_zimage_controlnet_config(sd)
        self.assertEqual(config, {})


if __name__ == "__main__":
    unittest.main()
