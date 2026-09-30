import json
import os
import unittest
import unittest.mock
from pathlib import Path

import torch
import torch.nn as nn

import ldm_patched.modules.ops
from ldm_patched.ldm import common_dit
from ldm_patched.ldm.krea2.model import (
    DoubleSharedModulation,
    GatedAttention,
    ScaledRMSNorm,
    SingleStreamDiT,
    TextFusionTransformer,
    build_position_ids,
    expand_kv_heads,
    patchify,
    unpatchify,
)

HEADER_FIXTURE = Path(__file__).parent / "fixtures" / "krea2_turbo_bf16_header.json"
PUBLISHED_PARAMETER_COUNT = 12_820_073_036
KREA2_MODEL_ENV = "KREA2_DIFFUSION_MODEL"


def make_tiny_config():
    """A structurally faithful but tiny SingleStreamDiT (head_dim 16 -> rope axes
    [4, 6, 6]; GQA 4 query heads over 2 kv heads), so tests run in milliseconds
    on CPU with no real weights.
    """
    return dict(
        features=64,
        tdim=32,
        txtdim=32,
        heads=4,
        kvheads=2,
        multiplier=4,
        layers=2,
        patch=2,
        channels=4,
        txtlayers=3,
        txtheads=2,
        txtkvheads=2,
    )


def init_small_weights(model):
    """`ldm_patched`'s ops layers skip random init (real usage loads a checkpoint
    right away), and the raw Krea 2 parameters start at zero, so tests that run a
    forward pass without a checkpoint initialize every parameter themselves.
    """
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(mean=0.0, std=0.02)


def fp8_storage_supported():
    try:
        torch.zeros(1).to(torch.float8_e4m3fn).to(torch.float16)
    except (AttributeError, RuntimeError, TypeError):
        return False
    return True


class TinyModelTestCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = make_tiny_config()
        self.model = SingleStreamDiT(**self.config)
        init_small_weights(self.model)
        self.model.eval()

    def fused_width(self):
        return self.config["txtlayers"] * self.config["txtdim"]

    def sample_inputs(self, batch_size=2, h=8, w=8, seq=5):
        x = torch.randn(batch_size, self.config["channels"], h, w)
        timesteps = torch.rand(batch_size)
        context = torch.randn(batch_size, seq, self.fused_width())
        return x, timesteps, context

    def assert_finite_output(self, out, x, dtype):
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, dtype)
        self.assertFalse(torch.isnan(out).any().item(), "{} forward produced NaN".format(dtype))
        self.assertFalse(torch.isinf(out).any().item(), "{} forward produced Inf".format(dtype))


class TestForward(TinyModelTestCase):
    def test_output_shape_matches_input_latent(self):
        x, timesteps, context = self.sample_inputs(batch_size=2, h=8, w=8)
        with torch.no_grad():
            out = self.model(x, timesteps, context)
        self.assert_finite_output(out, x, torch.float32)

    def test_odd_spatial_size_is_padded_then_cropped(self):
        x, timesteps, context = self.sample_inputs(batch_size=1, h=9, w=7)
        with torch.no_grad():
            out = self.model(x, timesteps, context)
        self.assertEqual(out.shape, x.shape)

    def test_non_contiguous_latent_is_accepted(self):
        wide = torch.randn(1, self.config["channels"], 8, 16)
        x = wide[:, :, :, ::2]
        self.assertFalse(x.is_contiguous())
        with torch.no_grad():
            out = self.model(x, torch.full((1,), 0.5), torch.randn(1, 4, self.fused_width()))
        self.assertEqual(out.shape, x.shape)

    def test_fp16_and_bf16_outputs_are_finite(self):
        x, timesteps, context = self.sample_inputs(batch_size=1)
        for dtype in (torch.float16, torch.bfloat16):
            model = SingleStreamDiT(**self.config, dtype=dtype)
            init_small_weights(model)
            with torch.no_grad():
                out = model(x.to(dtype), timesteps, context.to(dtype))
            self.assert_finite_output(out, x, dtype)

    def test_manual_cast_with_float32_weights_and_low_precision_inputs(self):
        model = SingleStreamDiT(**self.config, operations=ldm_patched.modules.ops.manual_cast)
        init_small_weights(model)
        model.eval()
        x, timesteps, context = self.sample_inputs(batch_size=1)
        for dtype in (torch.float16, torch.bfloat16):
            with torch.no_grad():
                out = model(x.to(dtype), timesteps, context.to(dtype))
            self.assertEqual(model.first.weight.dtype, torch.float32)
            self.assertEqual(model.last.modulation.lin.dtype, torch.float32)
            self.assert_finite_output(out, x, dtype)

    @unittest.skipUnless(fp8_storage_supported(), "torch build lacks float8_e4m3fn CPU casts")
    def test_manual_cast_with_fp8_stored_linear_weights(self):
        model = SingleStreamDiT(**self.config, operations=ldm_patched.modules.ops.manual_cast)
        init_small_weights(model)
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
        x, timesteps, context = self.sample_inputs(batch_size=1)
        with torch.no_grad():
            out = model(x.to(torch.float16), timesteps, context.to(torch.float16))
        self.assert_finite_output(out, x, torch.float16)

    def test_transformer_options_are_accepted_and_ignored(self):
        x, timesteps, context = self.sample_inputs(batch_size=1)
        with torch.no_grad():
            plain = self.model(x, timesteps, context)
            with_options = self.model(x, timesteps, context, transformer_options={"patches": {}}, extra=1)
        torch.testing.assert_close(plain, with_options)


class TestConditioning(TinyModelTestCase):
    def test_timestep_and_context_change_the_output(self):
        x, timesteps, context = self.sample_inputs(batch_size=1, seq=3)
        with torch.no_grad():
            base = self.model(x, timesteps, context)
            other_t = self.model(x, torch.rand_like(timesteps), context)
            other_ctx = self.model(x, timesteps, torch.randn_like(context))
        self.assertFalse(torch.allclose(base, other_t))
        self.assertFalse(torch.allclose(base, other_ctx))

    def test_mis_sized_context_raises_value_error(self):
        x, timesteps, _ = self.sample_inputs(batch_size=1)
        bad_context = torch.randn(1, 4, self.fused_width() + 1)
        with self.assertRaisesRegex(ValueError, str(self.fused_width())):
            self.model(x, timesteps, bad_context)

    def test_different_text_lengths_are_supported(self):
        x, timesteps, _ = self.sample_inputs(batch_size=1)
        for seq in (1, 4, 7):
            with torch.no_grad():
                out = self.model(x, timesteps, torch.randn(1, seq, self.fused_width()))
            self.assertEqual(out.shape, x.shape)


class TestArchitectureGuards(unittest.TestCase):
    def test_sigmoid_gate_scales_attention_output(self):
        torch.manual_seed(0)
        attn = GatedAttention(dim=32, heads=4, kvheads=2)
        init_small_weights(attn)
        x = torch.randn(1, 6, 32)
        with torch.no_grad():
            attn.gate.weight.zero_()  # sigmoid(0) == 0.5
            half_gated = attn(x)
            attn.gate = _ConstantGate(30.0)  # sigmoid(30) ~= 1
            fully_open = attn(x)
        torch.testing.assert_close(half_gated, 0.5 * fully_open, atol=1e-6, rtol=1e-5)

    def test_gqa_expansion_matches_repeat_interleave(self):
        kv = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
        expanded = expand_kv_heads(kv, 3)
        torch.testing.assert_close(expanded, kv.repeat_interleave(3, dim=2))

    def test_gqa_expansion_is_noop_without_grouping(self):
        kv = torch.randn(1, 2, 4, 8)
        self.assertIs(expand_kv_heads(kv, 1), kv)

    def test_scaled_rms_norm_with_zero_scale_is_plain_rms_norm(self):
        norm = ScaledRMSNorm(16)
        x = torch.randn(2, 5, 16)
        expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)
        torch.testing.assert_close(norm(x), expected, atol=1e-6, rtol=1e-5)

    def test_scaled_rms_norm_applies_one_plus_scale(self):
        norm = ScaledRMSNorm(16)
        with torch.no_grad():
            norm.scale.fill_(1.0)
        x = torch.randn(1, 3, 16)
        torch.testing.assert_close(norm(x), 2.0 * ScaledRMSNorm(16)(x), atol=1e-6, rtol=1e-5)

    def test_scaled_rms_norm_preserves_input_dtype(self):
        out = ScaledRMSNorm(16)(torch.randn(1, 2, 16).to(torch.bfloat16))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_double_shared_modulation_chunk_order(self):
        features = 4
        modulation = DoubleSharedModulation(features)
        vec = torch.arange(6 * features, dtype=torch.float32).reshape(1, 1, 6 * features)
        chunks = modulation(vec)
        self.assertEqual(len(chunks), 6)
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk.shape, (1, 1, features))
            torch.testing.assert_close(chunk, vec[..., index * features:(index + 1) * features])

    def test_text_fusion_projector_collapses_the_layer_axis(self):
        torch.manual_seed(0)
        fusion = TextFusionTransformer(num_txt_layers=3, txt_dim=32, heads=2, multiplier=4, kvheads=2)
        init_small_weights(fusion)
        self.assertEqual(tuple(fusion.projector.weight.shape), (1, 3))
        out = fusion(torch.randn(2, 5, 3, 32))
        self.assertEqual(out.shape, (2, 5, 32))

    def test_attention_rejects_heads_not_divisible_by_kvheads(self):
        with self.assertRaises(AssertionError):
            GatedAttention(dim=30, heads=5, kvheads=2)

    def test_patchify_roundtrip_and_token_feature_order(self):
        x = torch.randn(2, 4, 6, 8)
        tokens, h_tokens, w_tokens = patchify(x, 2)
        self.assertEqual((h_tokens, w_tokens), (3, 4))
        self.assertEqual(tokens.shape, (2, 12, 16))
        torch.testing.assert_close(unpatchify(tokens, h_tokens, w_tokens, 2, 4), x)
        # Feature order is (c, ph, pw): channel varies slowest within a token.
        first_token = tokens[0, 0].reshape(4, 2, 2)
        torch.testing.assert_close(first_token, x[0, :, :2, :2])

    def test_position_ids_place_text_at_origin_and_image_on_grid(self):
        txt_ids, img_ids = build_position_ids(3, 2, 4, 2, torch.device("cpu"))
        self.assertEqual(txt_ids.shape, (2, 3, 3))
        self.assertTrue(torch.all(txt_ids == 0))
        self.assertEqual(img_ids.shape, (2, 8, 3))
        self.assertTrue(torch.all(img_ids[..., 0] == 0))
        pairs = set(zip(img_ids[0, :, 1].tolist(), img_ids[0, :, 2].tolist()))
        self.assertEqual(pairs, {(float(r), float(c)) for r in range(2) for c in range(4)})


class _ConstantGate(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full_like(x, self.value)


class TestKeyLayout(unittest.TestCase):
    """The module's state dict must match the real checkpoint header exactly."""

    @classmethod
    def setUpClass(cls):
        header = json.loads(HEADER_FIXTURE.read_text())
        cls.expected_shapes = {key: tuple(entry["shape"]) for key, entry in header.items()}
        with torch.device("meta"):
            cls.model = SingleStreamDiT()

    def test_key_set_matches_reference_header(self):
        self.assertEqual(set(self.model.state_dict().keys()), set(self.expected_shapes))

    def test_every_shape_matches_reference_header(self):
        for key, tensor in self.model.state_dict().items():
            self.assertEqual(tuple(tensor.shape), self.expected_shapes[key], key)

    def test_parameter_count_is_the_published_figure(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), PUBLISHED_PARAMETER_COUNT)

    def test_text_fusion_sentinel_key_is_present(self):
        self.assertIn("txtfusion.projector.weight", self.model.state_dict())


@unittest.skipUnless(
    os.environ.get(KREA2_MODEL_ENV),
    "set {} to a local krea2_{{raw,turbo}}_bf16.safetensors to run".format(KREA2_MODEL_ENV),
)
class TestRealCheckpoint(unittest.TestCase):
    def test_bf16_checkpoint_loads_strictly(self):
        import safetensors.torch

        state_dict = safetensors.torch.load_file(os.environ[KREA2_MODEL_ENV])
        with torch.device("meta"):
            model = SingleStreamDiT()
        model.load_state_dict(state_dict, strict=True, assign=True)


class TestRopeCache(TinyModelTestCase):
    def test_second_forward_with_same_shapes_reuses_cached_tables(self):
        x, timesteps, context = self.sample_inputs(batch_size=1)
        with torch.no_grad():
            self.model(x, timesteps, context)
            with unittest.mock.patch(
                "ldm_patched.ldm.common_dit.rope_freqs",
                side_effect=AssertionError("rope_freqs must not be recomputed"),
            ):
                out = self.model(x, timesteps, context)
        self.assertEqual(out.shape, x.shape)

    def test_shape_change_invalidates_cache(self):
        timesteps = torch.full((1,), 0.5)
        context = torch.randn(1, 4, self.fused_width())
        with torch.no_grad():
            small = self.model(torch.randn(1, self.config["channels"], 8, 8), timesteps, context)
            large = self.model(torch.randn(1, self.config["channels"], 12, 12), timesteps, context)
            with unittest.mock.patch(
                "ldm_patched.ldm.common_dit.rope_freqs", wraps=common_dit.rope_freqs
            ) as spy:
                self.model(torch.randn(1, self.config["channels"], 8, 8), timesteps, context)
        self.assertEqual(small.shape[-2:], (8, 8))
        self.assertEqual(large.shape[-2:], (12, 12))
        self.assertGreater(spy.call_count, 0)


if __name__ == "__main__":
    unittest.main()
