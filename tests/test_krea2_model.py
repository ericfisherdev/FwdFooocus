"""
Unit tests for Krea 2 detection and model wiring (FWDF-132): the "krea2"
detector registered against FWDF-116's registry, the `Krea2` supported_models
config class, and the `model_base.Krea2` wrapper around the SingleStreamDiT
backbone (FWDF-131).

State dicts are synthetic (torch tensors of only the shapes detection reads).
The one real-checkpoint test reads safetensors *headers* only -- it never loads
tensor data -- and is skipped when no local Krea 2 checkpoint exists.
"""

import glob
import json
import math
import os
import struct
import sys
import unittest
from unittest import mock

import torch

# args_manager calls parse_args() at import time, which chokes on pytest's
# argv (modules.config imports args_manager). Patch sys.argv before importing
# project modules that pull modules.config in. Mirrors
# tests/test_model_family_detection.py.
_original_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    import modules.config  # noqa: E402
finally:
    sys.argv = _original_argv

import ldm_patched.ldm.krea2.model  # noqa: E402
from tests.test_krea2_single_stream_dit import fp8_storage_supported  # noqa: E402
from ldm_patched.modules import conds  # noqa: E402
from ldm_patched.modules import latent_formats  # noqa: E402
from ldm_patched.modules import model_base  # noqa: E402
from ldm_patched.modules import model_detection  # noqa: E402
from ldm_patched.modules import supported_models  # noqa: E402
from ldm_patched.modules import supported_models_base  # noqa: E402
from ldm_patched.modules.model_base import ModelType  # noqa: E402
from ldm_patched.modules.model_sampling import ModelSamplingDiscreteFlow  # noqa: E402

PREFIX = "model.diffusion_model."

# Published Krea 2 dims.
REAL_DIMS = dict(features=6144, channels=16, layers=28, heads=48, kvheads=12, txtlayers=12, txtdim=2560)


def _krea2_state_dict(prefix, features, channels, layers, heads, kvheads, txtlayers, txtdim):
    """A minimal synthetic Krea 2 state dict: only the keys matches_krea2() /
    detect_krea2_config() read, at the given shapes. Head counts are expressed
    via the architectural head width (KREA2_HEAD_DIM) the detector divides by.
    """
    head_dim = model_detection.KREA2_HEAD_DIM
    state_dict = {
        prefix + "first.weight": torch.zeros(features, 4 * channels),
        prefix + "txtfusion.projector.weight": torch.zeros(1, txtlayers),
        prefix + "txtfusion.layerwise_blocks.0.prenorm.scale": torch.zeros(txtdim),
    }
    for i in range(layers):
        state_dict["{}blocks.{}.attn.wq.weight".format(prefix, i)] = torch.zeros(heads * head_dim, features)
        state_dict["{}blocks.{}.attn.wk.weight".format(prefix, i)] = torch.zeros(kvheads * head_dim, features)
    return state_dict


def _real_krea2_state_dict(prefix=PREFIX):
    return _krea2_state_dict(prefix, **REAL_DIMS)


def _tiny_krea2_state_dict(prefix=PREFIX):
    return _krea2_state_dict(prefix, features=256, channels=8, layers=3, heads=4, kvheads=2, txtlayers=6, txtdim=40)


class TestMatchesKrea2(unittest.TestCase):
    def test_matches_when_all_required_keys_present(self):
        sd = _tiny_krea2_state_dict()
        self.assertTrue(model_detection.matches_krea2(list(sd.keys()), PREFIX))

    def test_each_required_key_missing_does_not_match(self):
        required = (
            "txtfusion.projector.weight",
            "first.weight",
            "blocks.0.attn.wq.weight",
            "blocks.0.attn.wk.weight",
            "txtfusion.layerwise_blocks.0.prenorm.scale",
        )
        for missing in required:
            with self.subTest(missing=missing):
                sd = _tiny_krea2_state_dict()
                del sd[PREFIX + missing]
                self.assertFalse(model_detection.matches_krea2(list(sd.keys()), PREFIX))
                # Graceful: no KeyError out of the config builder, just no match.
                self.assertIsNone(model_detection.model_config_from_unet(sd, PREFIX, torch.float32))

    def test_respects_key_prefix(self):
        sd = _tiny_krea2_state_dict("other.")
        keys = list(sd.keys())
        self.assertFalse(model_detection.matches_krea2(keys, PREFIX))
        self.assertTrue(model_detection.matches_krea2(keys, "other."))

    def test_unet_keys_do_not_match(self):
        self.assertFalse(model_detection.matches_krea2([PREFIX + "input_blocks.0.0.weight"], PREFIX))


class TestDetectKrea2Config(unittest.TestCase):
    def test_shape_inferred_values_on_tiny_dict(self):
        config = model_detection.detect_krea2_config(_tiny_krea2_state_dict(), PREFIX, torch.float32)
        self.assertEqual(config["features"], 256)
        self.assertEqual(config["channels"], 8)
        self.assertEqual(config["layers"], 3)
        self.assertEqual(config["heads"], 4)
        self.assertEqual(config["kvheads"], 2)
        self.assertEqual(config["txtlayers"], 6)
        self.assertEqual(config["txtdim"], 40)

    def test_published_dims_on_full_size_dict(self):
        config = model_detection.detect_krea2_config(_real_krea2_state_dict(), PREFIX, torch.bfloat16)
        for key, expected in REAL_DIMS.items():
            with self.subTest(key=key):
                self.assertEqual(config[key], expected)

    def test_fixed_constants(self):
        config = model_detection.detect_krea2_config(_tiny_krea2_state_dict(), PREFIX, torch.float16)
        self.assertEqual(config["image_model"], "krea2")
        self.assertEqual(config["patch"], 2)
        self.assertEqual(config["dtype"], torch.float16)

    def test_non_inferable_text_head_split_is_not_emitted(self):
        config = model_detection.detect_krea2_config(_tiny_krea2_state_dict(), PREFIX, torch.float32)
        for key in ("txtheads", "txtkvheads", "tdim", "theta"):
            with self.subTest(key=key):
                self.assertNotIn(key, config)

    def test_detected_config_builds_the_backbone(self):
        # The emitted kwargs must all be accepted by the constructor.
        config = model_detection.detect_krea2_config(_tiny_krea2_state_dict(), PREFIX, torch.float32)
        config.update(txtheads=2, txtkvheads=2)
        model = ldm_patched.ldm.krea2.model.SingleStreamDiT(**config, device="meta")
        self.assertEqual(len(model.blocks), 3)


class TestMalformedKrea2Checkpoints(unittest.TestCase):
    """A checkpoint with Krea 2's discriminant keys but impossible shapes must
    fail with the documented domain error, not a truncated head count, a
    KeyError, or an IndexError from the config builder (FWDF-160)."""

    def _route_with(self, key, shape):
        sd = _tiny_krea2_state_dict()
        sd[PREFIX + key] = torch.zeros(*shape)
        return model_detection.model_config_from_unet(sd, PREFIX, torch.float32)

    def test_impossible_shapes_raise_malformed_architecture_error(self):
        malformed = {
            "projector with more than one output row": ("txtfusion.projector.weight", (3, 7)),
            "projector that is a vector": ("txtfusion.projector.weight", (0,)),
            "query projection not a whole number of heads": ("blocks.0.attn.wq.weight", (500, 256)),
            "query projection with no rows": ("blocks.0.attn.wq.weight", (0, 256)),
            "key projection not a whole number of heads": ("blocks.0.attn.wk.weight", (200, 256)),
            "query heads not a multiple of kv heads": ("blocks.0.attn.wk.weight", (3 * 128, 256)),
            "patchify projection that is a vector": ("first.weight", (256,)),
            "patchify projection not a whole number of patches": ("first.weight", (256, 66)),
            "text norm scale that is a matrix": ("txtfusion.layerwise_blocks.0.prenorm.scale", (4, 10)),
        }
        for description, (key, shape) in malformed.items():
            with self.subTest(description):
                with self.assertRaises(model_detection.MalformedArchitectureError) as raised:
                    self._route_with(key, shape)
                self.assertEqual(raised.exception.architecture_name, "krea2")

    def test_malformed_error_is_a_value_error_and_distinct_from_unsupported(self):
        self.assertTrue(issubclass(model_detection.MalformedArchitectureError, ValueError))
        self.assertFalse(issubclass(model_detection.MalformedArchitectureError,
                                    model_detection.UnsupportedArchitectureError))

    def test_valid_published_and_tiny_shapes_are_not_rejected(self):
        for sd in (_real_krea2_state_dict(), _tiny_krea2_state_dict()):
            self.assertIsInstance(model_detection.model_config_from_unet(sd, PREFIX, torch.float32),
                                  supported_models.Krea2)


class TestKrea2DetectorTableOrdering(unittest.TestCase):
    """Krea 2 registers ahead of the terminal UNet fallback (FWDF-160)."""

    def test_krea2_is_tried_before_the_unet_fallback(self):
        names = [detector.name for detector in model_detection._DETECTOR_TABLE]
        self.assertEqual(names[-1], "unet")
        self.assertIn("krea2", names[:-1])

    def test_krea2_checkpoint_that_also_carries_unet_keys_still_routes_to_krea2(self):
        sd = _tiny_krea2_state_dict()
        sd[PREFIX + "input_blocks.0.0.weight"] = torch.zeros(320, 4, 3, 3)
        self.assertIsInstance(model_detection.model_config_from_unet(sd, PREFIX, torch.float32),
                              supported_models.Krea2)


class TestKrea2Routing(unittest.TestCase):
    def test_full_size_dict_routes_to_krea2_config(self):
        model_config = model_detection.model_config_from_unet(_real_krea2_state_dict(), PREFIX, torch.bfloat16)
        self.assertIsInstance(model_config, supported_models.Krea2)
        self.assertIsInstance(model_config.latent_format, latent_formats.QwenImage)
        self.assertEqual(model_config.unet_config["features"], 6144)

    def test_unet_config_carries_no_unet_injected_keys(self):
        model_config = model_detection.model_config_from_unet(_tiny_krea2_state_dict(), PREFIX, torch.float32)
        self.assertNotIn("num_heads", model_config.unet_config)
        self.assertNotIn("num_head_channels", model_config.unet_config)

    def test_krea2_does_not_steal_z_image_or_unet_configs(self):
        z_image_config = {"image_model": "z_image", "dim": 3840, "in_channels": 16}
        self.assertNotIsInstance(
            model_detection.model_config_from_unet_config(z_image_config), supported_models.Krea2)
        self.assertFalse(supported_models.Krea2.matches({"context_dim": 768, "model_channels": 320}))

    def test_z_image_shaped_checkpoint_still_routes_to_z_image(self):
        # Published Z-Image dims: dim 3840, 30 layers, 2 refiner layers, head_dim 128.
        state_dict = {
            PREFIX + "x_embedder.weight": torch.zeros(3840, 64),
            PREFIX + "cap_embedder.1.weight": torch.zeros(3840, 2560),
        }
        for i in range(30):
            state_dict["{}layers.{}.attention.q_norm.weight".format(PREFIX, i)] = torch.zeros(128)
        for i in range(2):
            state_dict["{}noise_refiner.{}.attention.q_norm.weight".format(PREFIX, i)] = torch.zeros(128)
        model_config = model_detection.model_config_from_unet(state_dict, PREFIX, torch.float32)
        self.assertIsInstance(model_config, supported_models.ZImage)

    def test_sdxl_shaped_config_still_resolves_to_sdxl(self):
        unet_config = {
            "use_checkpoint": False, "image_size": 32, "use_spatial_transformer": True, "legacy": False,
            "dtype": torch.float32, "num_classes": "sequential", "adm_in_channels": 2816,
            "in_channels": 4, "out_channels": 4, "model_channels": 320,
            "num_res_blocks": [2, 2, 2], "transformer_depth": [0, 0, 2, 2, 10, 10],
            "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
            "channel_mult": [1, 2, 4], "transformer_depth_middle": 10,
            "use_linear_in_transformer": True, "context_dim": 2048,
            "use_temporal_attention": False, "use_temporal_resblock": False,
        }
        self.assertIsInstance(model_detection.model_config_from_unet_config(unet_config), supported_models.SDXL)

    def test_sd15_shaped_checkpoint_still_resolves_to_sd15(self):
        state_dict = {
            PREFIX + "input_blocks.0.0.weight": torch.zeros(320, 4, 3, 3),
            PREFIX + "input_blocks.1.0.in_layers.0.weight": torch.zeros(1),
            PREFIX + "input_blocks.1.0.out_layers.3.weight": torch.zeros(320),
            PREFIX + "input_blocks.1.1.proj_in.weight": torch.zeros(320, 320, 1, 1),
            PREFIX + "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": torch.zeros(320, 768),
        }
        model_config = model_detection.model_config_from_unet(state_dict, PREFIX, torch.float32)
        self.assertIsInstance(model_config, supported_models.SD15)


def _flux_time_shift(mu, sigma, t):
    """ComfyUI's ModelSamplingFlux schedule (comfy/model_sampling.py)."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class TestKrea2SamplingSettings(unittest.TestCase):
    def setUp(self):
        self.model_config = model_detection.model_config_from_unet(
            _tiny_krea2_state_dict(), PREFIX, torch.float32)
        self.sampling = ModelSamplingDiscreteFlow(self.model_config)

    def test_shift_is_exp_of_mu_not_mu(self):
        self.assertAlmostEqual(self.sampling.shift, math.exp(1.15))
        self.assertNotAlmostEqual(self.sampling.shift, 1.15)

    def test_schedule_matches_comfyui_flux_time_shift(self):
        for t in (0.1, 0.25, 0.5, 0.75, 0.9):
            with self.subTest(t=t):
                got = float(self.sampling.sigma(torch.tensor(t, dtype=torch.float64)))
                self.assertAlmostEqual(got, _flux_time_shift(1.15, 1.0, t), places=6)

    def test_midpoint_value(self):
        expected = math.exp(1.15) / (math.exp(1.15) + 1)
        self.assertAlmostEqual(expected, 0.7595, places=3)
        self.assertAlmostEqual(float(self.sampling.sigma(torch.tensor(0.5))), expected, places=5)

    def test_sigma_table_matches_comfyui_10000_step_resolution(self):
        # karras / exponential schedules start from sigma_min, so the discrete
        # table must be tabulated at ComfyUI's ModelSamplingFlux resolution.
        self.assertEqual(len(self.sampling.sigmas), 10000)
        self.assertAlmostEqual(float(self.sampling.sigma_min), _flux_time_shift(1.15, 1.0, 1e-4), places=6)
        self.assertAlmostEqual(float(self.sampling.sigma_min), 0.0003158, places=6)
        self.assertAlmostEqual(float(self.sampling.sigma_max), 1.0, places=6)

    def test_z_image_sigma_table_resolution_is_unchanged(self):
        z_image_config = supported_models.ZImage({"image_model": "z_image"})
        self.assertEqual(len(ModelSamplingDiscreteFlow(z_image_config).sigmas), 1000)

    def test_timestep_is_unscaled(self):
        self.assertEqual(self.sampling.multiplier, 1.0)
        sigma = torch.tensor([0.3])
        self.assertTrue(torch.equal(self.sampling.timestep(sigma), sigma))


class _Krea2ModelTestCase(unittest.TestCase):
    """Shared tiny-backbone fixture mirroring tests/test_zimage_model.py."""

    class _FakeKrea2Config:
        memory_usage_factor = 2.2

        def __init__(self, unet_config):
            self.unet_config = unet_config
            self.latent_format = latent_formats.QwenImage()
            self.manual_cast_dtype = None
            self.sampling_settings = {"shift": math.exp(1.15), "multiplier": 1.0}

        def process_unet_state_dict(self, state_dict):
            return state_dict

    @staticmethod
    def tiny_unet_config():
        return dict(
            image_model="krea2", dtype=torch.float32,
            features=256, tdim=32, txtdim=16, heads=2, kvheads=1, multiplier=2,
            layers=2, patch=2, channels=16, txtlayers=3, txtheads=2, txtkvheads=2,
        )

    def tiny_model(self):
        return model_base.Krea2(self._FakeKrea2Config(self.tiny_unet_config()), device="cpu")


class TestKrea2ModelBaseWiring(_Krea2ModelTestCase):
    def test_diffusion_model_is_single_stream_dit(self):
        self.assertIsInstance(self.tiny_model().diffusion_model, ldm_patched.ldm.krea2.model.SingleStreamDiT)

    def test_model_type_is_flow(self):
        self.assertEqual(self.tiny_model().model_type, ModelType.FLOW)

    def test_model_sampling_is_discrete_flow_with_unscaled_multiplier(self):
        model = self.tiny_model()
        self.assertIsInstance(model.model_sampling, ModelSamplingDiscreteFlow)
        self.assertEqual(model.model_sampling.multiplier, 1.0)
        self.assertAlmostEqual(model.model_sampling.shift, math.exp(1.15))

    def test_unet_model_creation_is_disabled(self):
        unet_config = self.tiny_unet_config()
        model_base.Krea2(self._FakeKrea2Config(unet_config), device="cpu")
        self.assertTrue(unet_config["disable_unet_model_creation"])

    def test_does_not_override_apply_model(self):
        self.assertIs(model_base.Krea2.apply_model, model_base.BaseModel.apply_model)


@unittest.skipUnless(fp8_storage_supported(), "torch build lacks float8_e4m3fn CPU casts")
class TestKrea2LoadFlatFp8Weights(_Krea2ModelTestCase):
    """The Lustify checkpoint stores every tensor (norm scales, biases, modulation
    included) as float8_e4m3fn under flat keys; loading must restore them into the
    model's own dtype with nothing missing or unexpected."""

    def test_flat_all_fp8_state_dict_loads_into_model_dtype(self):
        model = self.tiny_model()
        dtypes_before = {name: p.dtype for name, p in model.diffusion_model.named_parameters()}
        flat_fp8 = {
            name: tensor.detach().to(torch.float8_e4m3fn)
            for name, tensor in model.diffusion_model.state_dict().items()
        }

        load_results = []
        real_load = model.diffusion_model.load_state_dict

        def recording_load(*args, **kwargs):
            load_results.append(real_load(*args, **kwargs))
            return load_results[-1]

        with mock.patch.object(model.diffusion_model, "load_state_dict", side_effect=recording_load):
            model.load_model_weights(flat_fp8, "")

        (missing, unexpected), = load_results
        self.assertEqual((missing, unexpected), ([], []))
        self.assertEqual(flat_fp8, {})
        self.assertEqual({name: p.dtype for name, p in model.diffusion_model.named_parameters()}, dtypes_before)


class TestKrea2ExtraConds(_Krea2ModelTestCase):
    def test_cross_attn_is_cond_regular_not_cond_cross_attn(self):
        out = self.tiny_model().extra_conds(cross_attn=torch.randn(1, 5, 48), device="cpu")
        self.assertIsInstance(out["c_crossattn"], conds.CONDRegular)
        self.assertNotIsInstance(out["c_crossattn"], conds.CONDCrossAttn)

    def test_no_guidance_tensor_is_produced(self):
        out = self.tiny_model().extra_conds(cross_attn=torch.randn(1, 5, 48), device="cpu")
        self.assertNotIn("guidance", out)
        self.assertEqual(set(out), {"c_crossattn"})

    def test_unequal_length_conds_refuse_to_concat(self):
        model = self.tiny_model()
        cond = model.extra_conds(cross_attn=torch.randn(1, 5, 48), device="cpu")["c_crossattn"]
        uncond = model.extra_conds(cross_attn=torch.randn(1, 7, 48), device="cpu")["c_crossattn"]
        self.assertFalse(cond.can_concat(uncond))

    def test_missing_cross_attn_yields_no_cross_attn_entry(self):
        out = self.tiny_model().extra_conds(device="cpu")
        self.assertNotIn("c_crossattn", out)


class TestKrea2ApplyModel(_Krea2ModelTestCase):
    def test_dit_receives_unscaled_flow_time_and_context(self):
        model = self.tiny_model()
        captured = {}

        def fake_forward(x, timesteps, context, **kwargs):
            captured["timesteps"] = timesteps
            captured["context"] = context
            return torch.zeros_like(x)

        model.diffusion_model.forward = fake_forward
        sigma = torch.tensor([0.3])
        context = torch.randn(1, 5, 48)

        model.apply_model(torch.randn(1, 16, 8, 8), sigma, c_crossattn=context)

        self.assertTrue(torch.allclose(captured["timesteps"], sigma))
        self.assertTrue(torch.equal(captured["context"], context))

    def test_output_shape_matches_input(self):
        torch.manual_seed(0)
        model = self.tiny_model()
        model.diffusion_model.eval()
        x = torch.randn(2, 16, 8, 8)
        with torch.no_grad():
            out = model.apply_model(x, torch.tensor([0.3, 0.6]), c_crossattn=torch.randn(2, 5, 48))
        self.assertEqual(out.shape, x.shape)


class TestMemoryUsageFactor(_Krea2ModelTestCase):
    INPUT_SHAPE = [2, 16, 64, 64]

    def _generic_estimate(self):
        area = self.INPUT_SHAPE[0] * self.INPUT_SHAPE[2] * self.INPUT_SHAPE[3]
        return (((area * 0.6) / 0.9) + 1024) * (1024 * 1024)

    def _memory_required(self, model):
        with mock.patch("ldm_patched.modules.model_management.xformers_enabled", return_value=False), \
                mock.patch("ldm_patched.modules.model_management.pytorch_attention_flash_attention", return_value=False):
            return model.memory_required(self.INPUT_SHAPE)

    def test_default_factor_is_one_for_existing_families(self):
        for config_class in (supported_models.SD15, supported_models.SDXL, supported_models.ZImage):
            with self.subTest(config_class=config_class.__name__):
                self.assertEqual(config_class.memory_usage_factor, 1.0)

    def test_default_factor_leaves_estimate_bit_identical(self):
        config = self._FakeKrea2Config({"disable_unet_model_creation": True})
        config.memory_usage_factor = supported_models_base.BASE.memory_usage_factor
        model = model_base.BaseModel(config, model_type=ModelType.FLOW)
        self.assertEqual(self._memory_required(model), self._generic_estimate())

    def test_krea2_scales_the_estimate_by_2_2(self):
        self.assertEqual(supported_models.Krea2.memory_usage_factor, 2.2)
        self.assertAlmostEqual(self._memory_required(self.tiny_model()), self._generic_estimate() * 2.2)


def _find_real_krea2_checkpoint():
    for directory in modules.config.paths_checkpoints:
        matches = sorted(glob.glob(os.path.join(directory, "krea2_*_bf16.safetensors")))
        if matches:
            return matches[0]
    return None


def _read_safetensors_header(path):
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(header_len))


_REAL_CHECKPOINT = _find_real_krea2_checkpoint()
# The published bf16 header and the Lustify Krea 2 fp8 header (all F8_E4M3):
# identical key set, so the second pins that the dtype is irrelevant to detection.
_HEADER_FIXTURES = {
    name: os.path.join(os.path.dirname(__file__), "fixtures", name)
    for name in ("krea2_turbo_bf16_header.json", "krea2_lustify_fp8_header.json")
}


class _HeaderDetectionAssertions:
    def assert_header_detects_published_dims(self, header):
        # Empty meta tensors of the header shapes: tensor data is never loaded.
        state_dict = {
            name: torch.empty(info["shape"], device="meta")
            for name, info in header.items() if name != "__metadata__"
        }

        model_config = model_detection.model_config_from_unet(state_dict, "", torch.bfloat16)

        self.assertIsInstance(model_config, supported_models.Krea2)
        for key, expected in REAL_DIMS.items():
            with self.subTest(key=key):
                self.assertEqual(model_config.unet_config[key], expected)


class TestKrea2HeaderFixture(_HeaderDetectionAssertions, unittest.TestCase):
    """CI-safe: the committed tensor-name/shape header of the published bf16 checkpoint."""

    def test_fixture_header_detects_published_dims(self):
        for name, path in _HEADER_FIXTURES.items():
            with self.subTest(fixture=name), open(path) as f:
                self.assert_header_detects_published_dims(json.load(f))

    def test_fp8_fixture_is_entirely_float8_e4m3(self):
        with open(_HEADER_FIXTURES["krea2_lustify_fp8_header.json"]) as f:
            header = json.load(f)
        self.assertEqual({info["dtype"] for info in header.values()}, {"F8_E4M3"})


@unittest.skipIf(_REAL_CHECKPOINT is None, "no local krea2_*_bf16.safetensors checkpoint")
class TestRealKrea2CheckpointHeader(_HeaderDetectionAssertions, unittest.TestCase):
    def test_header_shapes_detect_published_dims(self):
        self.assert_header_detects_published_dims(_read_safetensors_header(_REAL_CHECKPOINT))


if __name__ == "__main__":
    unittest.main()
