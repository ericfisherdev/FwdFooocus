import math
import unittest
import unittest.mock

import torch

import ldm_patched.modules.ops
from ldm_patched.ldm.lumina.model import (
    EmbedND,
    NextDiT,
    apply_rope,
    build_position_ids,
    rope_freqs,
)


def make_tiny_config():
    """A structurally faithful but tiny NextDiT: same axis/layer relationships
    as the real Z-Image config (dim=3840, n_heads=n_kv_heads=30, axes_dims
    summing to head_dim, patch_size=2), scaled down so tests run in
    milliseconds on CPU with no real weights.
    """
    return dict(
        patch_size=2,
        in_channels=4,
        dim=64,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        multiple_of=8,
        ffn_dim_multiplier=2.0,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=32,
        axes_dims=(8, 12, 12),
        axes_lens=(16, 8, 8),
        rope_theta=256.0,
        z_image_modulation=True,
        time_scale=1000.0,
    )


def init_small_weights(model):
    """`ldm_patched`'s ops.Linear/RMSNorm/LayerNorm intentionally override
    `reset_parameters` to a no-op -- real usage always loads a trained
    checkpoint immediately after construction, so skipping random init is a
    deliberate perf optimization there. That means a freshly constructed
    module here holds raw (possibly huge/pathological) uninitialized memory,
    so tests that run a forward pass without loading a checkpoint must
    initialize the weights themselves.
    """
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(mean=0.0, std=0.02)


class TestNextDiTForward(unittest.TestCase):
    """Shape/conditioning behavior of the tiny NextDiT forward pass."""

    def setUp(self):
        torch.manual_seed(0)
        self.config = make_tiny_config()
        self.model = NextDiT(**self.config)
        init_small_weights(self.model)
        self.model.eval()

    def _sample_inputs(self, batch_size=2, h=8, w=8, cap_len=5):
        x = torch.randn(batch_size, self.config["in_channels"], h, w)
        timesteps = torch.rand(batch_size)
        context = torch.randn(batch_size, cap_len, self.config["cap_feat_dim"])
        return x, timesteps, context

    def test_forward_output_shape_matches_input_latent(self):
        x, timesteps, context = self._sample_inputs(batch_size=2, h=8, w=8)
        with torch.no_grad():
            out = self.model(x, timesteps, context)
        self.assertEqual(out.shape, x.shape)

    def test_forward_accepts_non_contiguous_latent(self):
        """patchify must handle strided (non-contiguous) latents — a shape-
        valid slice would crash a view()-based implementation."""
        config = make_tiny_config()
        model = NextDiT(**config)
        init_small_weights(model)
        wide = torch.randn(1, config["in_channels"], 8, 16)
        x = wide[:, :, :, ::2]
        self.assertFalse(x.is_contiguous())
        with torch.no_grad():
            out = model(x, torch.full((1,), 0.5), torch.randn(1, 6, config["cap_feat_dim"]))
        self.assertEqual(out.shape, x.shape)

    def test_forward_crops_padding_for_non_multiple_of_patch_size(self):
        # H=W=9 isn't a multiple of patch_size=2; the backbone pads internally
        # for patchify/unpatchify but must crop back to the original size.
        x, timesteps, context = self._sample_inputs(batch_size=1, h=9, w=9)
        with torch.no_grad():
            out = self.model(x, timesteps, context)
        self.assertEqual(out.shape, x.shape)

    def test_forward_output_shape_in_fp16_and_bf16(self):
        x, timesteps, context = self._sample_inputs(batch_size=1, h=8, w=8)
        for dtype in (torch.float16, torch.bfloat16):
            model = self.model.to(dtype)
            with torch.no_grad():
                out = model(x.to(dtype), timesteps, context.to(dtype))
            self.assertEqual(out.shape, x.shape)
            self.assertEqual(out.dtype, dtype)
            self.assertFalse(torch.isnan(out).any().item(), "{} forward produced NaN".format(dtype))
            self.assertFalse(torch.isinf(out).any().item(), "{} forward produced Inf".format(dtype))
        self.model.to(torch.float32)

    def test_forward_works_with_manual_cast_operations(self):
        # manual_cast casts weights to the *input's* dtype on every forward
        # call (see ops.cast_bias_weight) -- keep the model's stored weights
        # in float32 and feed fp16/bf16 inputs so this test actually exercises
        # that cast (a same-dtype "cast" would be a no-op and wouldn't catch
        # a broken conversion). final_layer.norm_final is a LayerNorm with
        # elementwise_affine=False (matching the real checkpoint, which has
        # no norm_final.weight/bias), so this also covers the weight=None
        # case in ops.cast_bias_weight.
        config = dict(self.config, operations=ldm_patched.modules.ops.manual_cast)
        model = NextDiT(**config)
        init_small_weights(model)
        model.eval()

        self.assertEqual(model.x_embedder.weight.dtype, torch.float32)

        x, timesteps, context = self._sample_inputs(batch_size=1, h=8, w=8)
        for dtype in (torch.float16, torch.bfloat16):
            with torch.no_grad():
                out = model(x.to(dtype), timesteps, context.to(dtype))
            # Weights must stay float32 -- only the activations are cast.
            self.assertEqual(model.x_embedder.weight.dtype, torch.float32)
            self.assertEqual(out.shape, x.shape)
            self.assertEqual(out.dtype, dtype)
            self.assertFalse(torch.isnan(out).any().item(), "{} forward produced NaN".format(dtype))
            self.assertFalse(torch.isinf(out).any().item(), "{} forward produced Inf".format(dtype))

    def test_accepts_timestep_and_caption_conditioning(self):
        x, timesteps, context = self._sample_inputs(batch_size=1, h=8, w=8, cap_len=3)
        with torch.no_grad():
            out_a = self.model(x, timesteps, context)
            out_b = self.model(x, torch.rand_like(timesteps), context)
            out_c = self.model(x, timesteps, torch.randn_like(context))

        # Changing the timestep or the caption features must actually change
        # the output -- i.e. both conditioning paths are wired into the
        # backbone, not silently dropped.
        self.assertFalse(torch.allclose(out_a, out_b))
        self.assertFalse(torch.allclose(out_a, out_c))

    def test_different_caption_lengths_are_supported(self):
        x = torch.randn(1, self.config["in_channels"], 8, 8)
        timesteps = torch.rand(1)
        for cap_len in (1, 4, 7):
            context = torch.randn(1, cap_len, self.config["cap_feat_dim"])
            with torch.no_grad():
                out = self.model(x, timesteps, context)
            self.assertEqual(out.shape, x.shape)

    def test_axes_dims_must_sum_to_head_dim(self):
        bad_config = dict(self.config)
        bad_config["axes_dims"] = (8, 8, 8)  # sums to 24, not head_dim=32
        with self.assertRaises(AssertionError):
            NextDiT(**bad_config)


class TestRopeAxesMath(unittest.TestCase):
    """The 3-axis RoPE (caption-order axis + 2D spatial axes) shape contract."""

    def setUp(self):
        self.config = make_tiny_config()
        self.head_dim = self.config["dim"] // self.config["n_heads"]

    def test_rope_freqs_shape_per_axis(self):
        pos = torch.arange(6, dtype=torch.float32)
        axis_dim = self.config["axes_dims"][0]
        table = rope_freqs(pos, axis_dim, self.config["rope_theta"])
        self.assertEqual(table.shape, (6, axis_dim // 2, 2, 2))

    def test_rope_freqs_matches_hand_computed_rotation_at_nonzero_position(self):
        # dim=4, theta=100 -> scale=[0, 0.5] -> omega=[1, 0.1], clean values
        # that are easy to hand-verify independently of rope_freqs' own code.
        dim, theta, pos_value = 4, 100.0, 3.0
        table = rope_freqs(torch.tensor([pos_value]), dim, theta)

        omega = [1.0 / (theta ** s) for s in (0.0, 0.5)]
        for k, o in enumerate(omega):
            angle = pos_value * o
            expected = torch.tensor([
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ])
            torch.testing.assert_close(table[0, k], expected, atol=1e-6, rtol=1e-6)

    def test_rope_freqs_identity_at_zero_position(self):
        # pos=0 -> every rotation angle is 0 regardless of omega/theta, so
        # each 2x2 block must be the identity matrix.
        dim = 6
        table = rope_freqs(torch.zeros(3), dim, self.config["rope_theta"])
        identity = torch.eye(2).expand(3, dim // 2, 2, 2)
        torch.testing.assert_close(table, identity)

    def test_embednd_concatenates_axes_to_head_dim(self):
        embedder = EmbedND(self.head_dim, theta=self.config["rope_theta"], axes_dim=list(self.config["axes_dims"]))
        batch_size, seq_len = 2, 7
        ids = torch.zeros(batch_size, seq_len, 3)
        ids[:, :, 1] = torch.arange(seq_len, dtype=torch.float32)
        freqs_cis = embedder(ids)
        # (b, 1, seq, head_dim // 2, 2, 2) before the movedim done in NextDiT.forward
        self.assertEqual(freqs_cis.shape, (batch_size, 1, seq_len, self.head_dim // 2, 2, 2))

    def test_build_position_ids_shapes_and_axis_semantics(self):
        cap_len, h_tokens, w_tokens, batch_size = 5, 3, 4, 2
        cap_pos_ids, img_pos_ids = build_position_ids(cap_len, h_tokens, w_tokens, batch_size, torch.device("cpu"))

        self.assertEqual(cap_pos_ids.shape, (batch_size, cap_len, 3))
        self.assertEqual(img_pos_ids.shape, (batch_size, h_tokens * w_tokens, 3))

        # Caption tokens get sequential positions on axis 0; spatial axes are 0.
        self.assertTrue(torch.equal(cap_pos_ids[:, :, 0], (torch.arange(cap_len, dtype=torch.float32) + 1.0).expand(batch_size, -1)))
        self.assertTrue(torch.all(cap_pos_ids[:, :, 1:] == 0))

        # Image tokens all share one axis-0 value, placed right after the caption.
        self.assertTrue(torch.all(img_pos_ids[:, :, 0] == cap_len + 1))
        # Row/col axes must cover the exact Cartesian product of the
        # h_tokens x w_tokens grid -- not just each axis' value set
        # independently, which wouldn't catch a broken row/col pairing
        # (e.g. rows and cols transposed or not fully crossed).
        actual_pairs = set(zip(img_pos_ids[0, :, 1].tolist(), img_pos_ids[0, :, 2].tolist()))
        expected_pairs = {(float(r), float(c)) for r in range(h_tokens) for c in range(w_tokens)}
        self.assertEqual(actual_pairs, expected_pairs)
        self.assertEqual(len(actual_pairs), h_tokens * w_tokens)

    def test_apply_rope_preserves_shape(self):
        embedder = EmbedND(self.head_dim, theta=self.config["rope_theta"], axes_dim=list(self.config["axes_dims"]))
        batch_size, seq_len, n_heads = 2, 6, self.config["n_heads"]
        ids = torch.zeros(batch_size, seq_len, 3)
        ids[:, :, 1] = torch.arange(seq_len, dtype=torch.float32)
        freqs_cis = embedder(ids).movedim(1, 2)  # (b, seq, 1, head_dim // 2, 2, 2)

        xq = torch.randn(batch_size, seq_len, n_heads, self.head_dim)
        xk = torch.randn(batch_size, seq_len, n_heads, self.head_dim)
        rq, rk = apply_rope(xq, xk, freqs_cis)

        self.assertEqual(rq.shape, xq.shape)
        self.assertEqual(rk.shape, xk.shape)

    def test_apply_rope_matches_hand_computed_rotation_at_nonzero_position(self):
        # Isolate a single 2-wide axis (omega=1 regardless of theta) so the
        # rotation angle for that pair is exactly pos_value -- lets us assert
        # the exact rotated (cos, sin) components without re-deriving the
        # frequency formula.
        pos_value = 1.3
        axis_dims = [2, self.head_dim - 2]
        embedder = EmbedND(self.head_dim, theta=100.0, axes_dim=axis_dims)
        ids = torch.zeros(1, 1, 2)
        ids[:, :, 0] = pos_value
        freqs_cis = embedder(ids).movedim(1, 2)

        x = torch.zeros(1, 1, 1, self.head_dim)
        x[..., 0] = 1.0  # unit vector on the isolated pair's first component
        rx, _ = apply_rope(x, x, freqs_cis)

        self.assertAlmostEqual(rx[0, 0, 0, 0].item(), math.cos(pos_value), places=5)
        self.assertAlmostEqual(rx[0, 0, 0, 1].item(), math.sin(pos_value), places=5)

    def test_apply_rope_identity_at_zero_position(self):
        embedder = EmbedND(self.head_dim, theta=self.config["rope_theta"], axes_dim=list(self.config["axes_dims"]))
        batch_size, seq_len, n_heads = 1, 4, self.config["n_heads"]
        ids = torch.zeros(batch_size, seq_len, 3)  # all-zero ids -> identity rotation
        freqs_cis = embedder(ids).movedim(1, 2)

        xq = torch.randn(batch_size, seq_len, n_heads, self.head_dim)
        xk = torch.randn(batch_size, seq_len, n_heads, self.head_dim)
        rq, rk = apply_rope(xq, xk, freqs_cis)

        torch.testing.assert_close(rq, xq)
        torch.testing.assert_close(rk, xk)


class TestStateDictKeyLayout(unittest.TestCase):
    """Canonical state-dict key layout that FWDF-124's checkpoint detection
    can rely on. These prefixes are verified against the real Z-Image
    checkpoint's key names via ComfyUI's `model_detection.py` (which reads
    `cap_embedder.1.weight` and `noise_refiner.0.attention.k_norm.weight`
    directly off a downloaded checkpoint to auto-configure the model).
    """

    def setUp(self):
        self.config = make_tiny_config()
        self.model = NextDiT(**self.config)
        self.keys = set(self.model.state_dict().keys())

    def test_top_level_submodule_keys_present(self):
        expected = {
            "x_embedder.weight", "x_embedder.bias",
            "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias",
            "t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias",
            "cap_embedder.0.weight",  # RMSNorm, no bias
            "cap_embedder.1.weight", "cap_embedder.1.bias",
            "final_layer.linear.weight", "final_layer.linear.bias",
            "final_layer.adaLN_modulation.1.weight", "final_layer.adaLN_modulation.1.bias",
        }
        self.assertTrue(expected.issubset(self.keys), expected - self.keys)
        # final_layer.norm_final has elementwise_affine=False -> no learnable weight/bias.
        self.assertNotIn("final_layer.norm_final.weight", self.keys)

    def test_context_refiner_blocks_have_no_timestep_modulation(self):
        # context_refiner is constructed with modulation=False: it refines
        # caption tokens without any timestep conditioning.
        for i in range(self.config["n_refiner_layers"]):
            prefix = "context_refiner.{}.".format(i)
            self._assert_block_norm_and_attention_keys(prefix)
            self.assertFalse(
                any(k.startswith(prefix + "adaLN_modulation") for k in self.keys),
                "context_refiner block {} must not have adaLN_modulation keys".format(i),
            )

    def test_noise_refiner_and_layers_have_timestep_modulation(self):
        for prefix_base, count in (
            ("noise_refiner.", self.config["n_refiner_layers"]),
            ("layers.", self.config["n_layers"]),
        ):
            for i in range(count):
                prefix = "{}{}.".format(prefix_base, i)
                self._assert_block_norm_and_attention_keys(prefix)
                self.assertIn(prefix + "adaLN_modulation.0.weight", self.keys)
                self.assertIn(prefix + "adaLN_modulation.0.bias", self.keys)

    def _assert_block_norm_and_attention_keys(self, prefix):
        expected = {
            prefix + "attention.qkv.weight",
            prefix + "attention.out.weight",
            prefix + "attention.q_norm.weight",
            prefix + "attention.k_norm.weight",
            prefix + "attention_norm1.weight",
            prefix + "attention_norm2.weight",
            prefix + "ffn_norm1.weight",
            prefix + "ffn_norm2.weight",
            prefix + "feed_forward.w1.weight",
            prefix + "feed_forward.w2.weight",
            prefix + "feed_forward.w3.weight",
        }
        self.assertTrue(expected.issubset(self.keys), expected - self.keys)
        # qkv/out/feed_forward Linears are bias=False in the real checkpoint.
        self.assertNotIn(prefix + "attention.qkv.bias", self.keys)
        self.assertNotIn(prefix + "attention.out.bias", self.keys)


class TestRopeFreqsCache(unittest.TestCase):
    """The RoPE tables are cached per (cap_len, grid, batch, device) so a
    sampling loop with constant shapes does not redo the float64 CPU einsum
    and host-to-device upload on every diffusion step."""

    def _model(self):
        config = make_tiny_config()
        model = NextDiT(**config)
        init_small_weights(model)
        return model, config

    def test_second_forward_with_same_shapes_reuses_cached_tables(self):
        model, config = self._model()
        x = torch.randn(1, config["in_channels"], 8, 8)
        timesteps = torch.full((1,), 0.5)
        context = torch.randn(1, 6, config["cap_feat_dim"])
        with torch.no_grad():
            model(x, timesteps, context)
            with unittest.mock.patch(
                "ldm_patched.ldm.lumina.model.rope_freqs",
                side_effect=AssertionError("rope_freqs must not be recomputed"),
            ):
                out = model(x, timesteps, context)
        self.assertEqual(out.shape, x.shape)

    def test_shape_change_invalidates_cache(self):
        model, config = self._model()
        timesteps = torch.full((1,), 0.5)
        context = torch.randn(1, 6, config["cap_feat_dim"])
        with torch.no_grad():
            out_small = model(torch.randn(1, config["in_channels"], 8, 8), timesteps, context)
            out_large = model(torch.randn(1, config["in_channels"], 12, 12), timesteps, context)
        self.assertEqual(out_small.shape[-2:], (8, 8))
        self.assertEqual(out_large.shape[-2:], (12, 12))


if __name__ == "__main__":
    unittest.main()
