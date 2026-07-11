"""
Unit tests for Z-Image detection and model wiring (FWDF-124): the "z-image"
detector registered against FWDF-116's registry, the `ZImage` supported_models
config class (FWDF-120 latent format, FWDF-119 flow sampling), and the
`model_base.ZImage` wrapper around the NextDiT backbone (FWDF-123).

All state dicts are synthetic (torch tensors of only the shapes detection
reads) -- no real checkpoint files are used.
"""

import unittest

import torch

import ldm_patched.ldm.lumina.model
from ldm_patched.modules import latent_formats
from ldm_patched.modules import model_detection
from ldm_patched.modules import model_base
from ldm_patched.modules import supported_models
from ldm_patched.modules.model_base import ModelType
from ldm_patched.modules.model_sampling import ModelSamplingDiscreteFlow


def _z_image_state_dict(prefix, dim, in_channels, n_layers, n_refiner_layers, n_heads, cap_feat_dim):
    """A minimal synthetic Z-Image-shaped state dict: only the keys
    matches_z_image()/detect_z_image_config() actually read, at the given
    shapes. head_dim is derived the same way detect_z_image_config() does
    (dim // n_heads), keeping caller and detector in sync.
    """
    head_dim = dim // n_heads
    patch_size = 2
    state_dict = {
        prefix + "x_embedder.weight": torch.zeros(dim, patch_size * patch_size * in_channels),
        prefix + "cap_embedder.1.weight": torch.zeros(dim, cap_feat_dim),
    }
    for i in range(n_layers):
        state_dict["{}layers.{}.attention.q_norm.weight".format(prefix, i)] = torch.zeros(head_dim)
    for i in range(n_refiner_layers):
        state_dict["{}noise_refiner.{}.attention.q_norm.weight".format(prefix, i)] = torch.zeros(head_dim)
    return state_dict


def _real_z_image_state_dict(prefix=""):
    """Full-scale Z-Image dims (dim=3840, n_layers=30, n_refiner_layers=2,
    head_dim=sum(axes_dims)=128, cap_feat_dim=2560, in_channels=16) so
    detection/routing tests exercise the exact published checkpoint config.
    """
    return _z_image_state_dict(
        prefix, dim=3840, in_channels=16, n_layers=30, n_refiner_layers=2, n_heads=30, cap_feat_dim=2560,
    )


class TestMatchesZImage(unittest.TestCase):
    def test_matches_when_x_embedder_and_cap_embedder_present(self):
        prefix = "model.diffusion_model."
        keys = [prefix + "x_embedder.weight", prefix + "cap_embedder.1.weight"]
        self.assertTrue(model_detection.matches_z_image(keys, prefix))

    def test_does_not_match_without_x_embedder(self):
        prefix = "model.diffusion_model."
        keys = [prefix + "cap_embedder.1.weight"]
        self.assertFalse(model_detection.matches_z_image(keys, prefix))

    def test_does_not_match_without_cap_embedder(self):
        prefix = "model.diffusion_model."
        keys = [prefix + "x_embedder.weight"]
        self.assertFalse(model_detection.matches_z_image(keys, prefix))

    def test_does_not_match_unet_shaped_keys(self):
        prefix = "model.diffusion_model."
        keys = [prefix + "input_blocks.0.0.weight"]
        self.assertFalse(model_detection.matches_z_image(keys, prefix))

    def test_respects_key_prefix(self):
        keys = ["other.x_embedder.weight", "other.cap_embedder.1.weight"]
        self.assertFalse(model_detection.matches_z_image(keys, "model.diffusion_model."))
        self.assertTrue(model_detection.matches_z_image(keys, "other."))


class TestPartialCheckpointsFailDetectionGracefully(unittest.TestCase):
    """A partial or unrelated DiT checkpoint must not match the z-image
    detector: matches() guarantees every key detect_z_image_config() indexes
    unconditionally, per the ArchitectureDetector contract."""

    def _base_keys(self, prefix='model.diffusion_model.'):
        return {
            prefix + 'x_embedder.weight': torch.zeros(64, 16),
            prefix + 'cap_embedder.1.weight': torch.zeros(64, 32),
            prefix + 'layers.0.attention.q_norm.weight': torch.zeros(8),
        }

    def test_missing_cap_embedder_weight_does_not_match(self):
        prefix = 'model.diffusion_model.'
        sd = self._base_keys()
        del sd[prefix + 'cap_embedder.1.weight']
        sd[prefix + 'cap_embedder.0.bias'] = torch.zeros(1)
        self.assertFalse(model_detection.matches_z_image(list(sd.keys()), prefix))
        self.assertIsNone(model_detection.model_config_from_unet(sd, prefix, torch.float32))

    def test_missing_q_norm_weight_does_not_match(self):
        prefix = 'model.diffusion_model.'
        sd = self._base_keys()
        del sd[prefix + 'layers.0.attention.q_norm.weight']
        self.assertFalse(model_detection.matches_z_image(list(sd.keys()), prefix))
        self.assertIsNone(model_detection.model_config_from_unet(sd, prefix, torch.float32))


class TestDetectZImageConfig(unittest.TestCase):
    """detect_z_image_config() must infer shape-derived keys from tensor
    shapes and fill fixed architectural constants for the rest.
    """

    def setUp(self):
        self.prefix = "model.diffusion_model."
        self.state_dict = _z_image_state_dict(
            self.prefix, dim=64, in_channels=8, n_layers=3, n_refiner_layers=2, n_heads=4, cap_feat_dim=40,
        )
        self.config = model_detection.detect_z_image_config(self.state_dict, self.prefix, torch.float32)

    def test_shape_inferred_keys(self):
        self.assertEqual(self.config["dim"], 64)
        self.assertEqual(self.config["n_layers"], 3)
        self.assertEqual(self.config["n_refiner_layers"], 2)
        self.assertEqual(self.config["n_heads"], 4)
        self.assertEqual(self.config["n_kv_heads"], 4)
        self.assertEqual(self.config["in_channels"], 8)
        self.assertEqual(self.config["cap_feat_dim"], 40)
        self.assertEqual(self.config["dtype"], torch.float32)

    def test_fixed_architectural_constants(self):
        self.assertEqual(self.config["patch_size"], 2)
        self.assertEqual(self.config["axes_dims"], [32, 48, 48])
        self.assertEqual(self.config["axes_lens"], [1536, 512, 512])
        self.assertEqual(self.config["rope_theta"], 256.0)
        self.assertEqual(self.config["norm_eps"], 1e-05)
        self.assertTrue(self.config["qk_norm"])
        self.assertEqual(self.config["multiple_of"], 256)
        self.assertAlmostEqual(self.config["ffn_dim_multiplier"], 8.0 / 3.0)
        self.assertTrue(self.config["z_image_modulation"])
        self.assertEqual(self.config["time_scale"], 1000.0)
        self.assertEqual(self.config["image_model"], "z_image")

    def test_real_checkpoint_dims_produce_published_z_image_config(self):
        state_dict = _real_z_image_state_dict(self.prefix)
        config = model_detection.detect_z_image_config(state_dict, self.prefix, torch.float32)
        self.assertEqual(config["dim"], 3840)
        self.assertEqual(config["n_layers"], 30)
        self.assertEqual(config["n_refiner_layers"], 2)
        self.assertEqual(config["n_heads"], 30)
        self.assertEqual(config["n_kv_heads"], 30)
        self.assertEqual(config["in_channels"], 16)
        self.assertEqual(config["cap_feat_dim"], 2560)


class TestZImageRouting(unittest.TestCase):
    """model_config_from_unet() must route a Z-Image-shaped checkpoint to
    supported_models.ZImage with no KeyError, with SDXL/SD15 detection
    unaffected by ZImage's presence in the models list.
    """

    def test_z_image_shaped_checkpoint_routes_to_z_image_model_config(self):
        prefix = "model.diffusion_model."
        state_dict = _real_z_image_state_dict(prefix)

        model_config = model_detection.model_config_from_unet(state_dict, prefix, torch.float32)

        self.assertIsInstance(model_config, supported_models.ZImage)
        self.assertIsInstance(model_config.latent_format, latent_formats.Flux)
        self.assertEqual(model_config.sampling_settings["shift"], 3.0)
        self.assertEqual(model_config.unet_config["dim"], 3840)
        self.assertEqual(model_config.unet_config["in_channels"], 16)

    def test_sdxl_shaped_config_still_resolves_to_sdxl(self):
        # Exercises model_config_from_unet_config() directly (the function
        # BASE.matches() was fixed in) against SDXL's exact declared config,
        # rather than hand-reconstructing detect_unet_config()'s block-count
        # inference -- this is the precise path the disjoint-keys fix
        # (supported_models_base.BASE.matches()) must not regress.
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

        model_config = model_detection.model_config_from_unet_config(unet_config)

        self.assertIsInstance(model_config, supported_models.SDXL)

    def test_sd15_shaped_checkpoint_still_resolves_to_sd15(self):
        prefix = "model.diffusion_model."
        state_dict = {
            prefix + "input_blocks.0.0.weight": torch.zeros(320, 4, 3, 3),
            prefix + "input_blocks.1.0.in_layers.0.weight": torch.zeros(1),
            prefix + "input_blocks.1.0.out_layers.3.weight": torch.zeros(320),
            prefix + "input_blocks.1.1.proj_in.weight": torch.zeros(320, 320, 1, 1),
            prefix + "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": torch.zeros(320, 768),
        }

        model_config = model_detection.model_config_from_unet(state_dict, prefix, torch.float32)

        self.assertIsInstance(model_config, supported_models.SD15)


class TestZImageModelBaseWiring(unittest.TestCase):
    """model_base.ZImage must instantiate NextDiT (not UNetModel) and select
    the FLOW/ModelSamplingDiscreteFlow sampling path via ZImage's
    sampling_settings.
    """

    class _FakeZImageConfig:
        def __init__(self, unet_config, shift=3.0):
            self.unet_config = unet_config
            self.latent_format = latent_formats.Flux()
            self.manual_cast_dtype = None
            self.sampling_settings = {"shift": shift}

    def _tiny_unet_config(self):
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

    def test_diffusion_model_is_nextdit_not_unet(self):
        config = self._FakeZImageConfig(self._tiny_unet_config())
        model = model_base.ZImage(config, device="cpu")
        self.assertIsInstance(model.diffusion_model, ldm_patched.ldm.lumina.model.NextDiT)

    def test_model_type_is_flow(self):
        config = self._FakeZImageConfig(self._tiny_unet_config())
        model = model_base.ZImage(config, device="cpu")
        self.assertEqual(model.model_type, ModelType.FLOW)

    def test_model_sampling_is_discrete_flow_with_configured_shift(self):
        config = self._FakeZImageConfig(self._tiny_unet_config(), shift=3.0)
        model = model_base.ZImage(config, device="cpu")
        self.assertIsInstance(model.model_sampling, ModelSamplingDiscreteFlow)
        self.assertEqual(model.model_sampling.shift, 3.0)

    def test_unet_config_marks_unet_model_creation_disabled(self):
        unet_config = self._tiny_unet_config()
        config = self._FakeZImageConfig(unet_config)
        model_base.ZImage(config, device="cpu")
        self.assertTrue(unet_config["disable_unet_model_creation"])


class TestZImageApplyModel(unittest.TestCase):
    """apply_model() must route caption conditioning (c_crossattn) to
    NextDiT's `context` argument and pass the raw flow-matching sigma as
    `timesteps` (not ModelSamplingDiscreteFlow.timestep()'s sigma * multiplier
    rescale), rather than silently dropping or misrouting either.
    """

    def _tiny_model(self, shift=3.0):
        config = TestZImageModelBaseWiring._FakeZImageConfig(
            TestZImageModelBaseWiring()._tiny_unet_config(), shift=shift,
        )
        return model_base.ZImage(config, device="cpu")

    def test_context_and_raw_sigma_reach_diffusion_model_forward(self):
        model = self._tiny_model()
        captured = {}

        def fake_forward(x, timesteps, context, **kwargs):
            captured["x"] = x
            captured["timesteps"] = timesteps
            captured["context"] = context
            return torch.zeros_like(x)

        model.diffusion_model.forward = fake_forward

        x = torch.randn(1, 4, 8, 8)
        sigma = torch.tensor([0.3])
        context = torch.randn(1, 5, 32)

        model.apply_model(x, sigma, c_crossattn=context)

        self.assertIn("context", captured)
        self.assertTrue(torch.equal(captured["context"], context))
        self.assertTrue(torch.allclose(captured["timesteps"], sigma.float()))

    def test_timestep_is_not_rescaled_by_model_sampling_timestep(self):
        # ModelSamplingDiscreteFlow.timestep(sigma) = sigma * multiplier
        # (multiplier defaults to 1000) -- that value must NOT be what
        # reaches NextDiT, since NextDiT expects a raw [0, 1] sigma.
        model = self._tiny_model()
        captured = {}

        def fake_forward(x, timesteps, context, **kwargs):
            captured["timesteps"] = timesteps
            return torch.zeros_like(x)

        model.diffusion_model.forward = fake_forward

        x = torch.randn(1, 4, 8, 8)
        sigma = torch.tensor([0.3])
        context = torch.randn(1, 5, 32)

        model.apply_model(x, sigma, c_crossattn=context)

        rescaled = model.model_sampling.timestep(sigma).float()
        self.assertFalse(torch.allclose(captured["timesteps"], rescaled))
        self.assertTrue(torch.allclose(captured["timesteps"], sigma.float()))

    def test_caption_conditioning_is_not_silently_dropped(self):
        # End-to-end through the real (tiny) NextDiT forward: changing the
        # caption conditioning must actually change apply_model()'s output.
        torch.manual_seed(0)
        model = self._tiny_model()
        with torch.no_grad():
            for p in model.diffusion_model.parameters():
                p.normal_(mean=0.0, std=0.02)
        model.diffusion_model.eval()

        x = torch.randn(1, 4, 8, 8)
        sigma = torch.tensor([0.3])
        context_a = torch.randn(1, 5, 32)
        context_b = torch.randn(1, 5, 32)

        with torch.no_grad():
            out_a = model.apply_model(x, sigma, c_crossattn=context_a)
            out_b = model.apply_model(x, sigma, c_crossattn=context_b)

        self.assertEqual(out_a.shape, x.shape)
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_apply_model_has_no_c_concat_path(self):
        # Z-Image never sets inpaint_model=True, so extra_conds() (inherited
        # unmodified from BaseModel) never produces a c_concat entry; passing
        # None explicitly must not raise or alter channel count.
        model = self._tiny_model()
        with torch.no_grad():
            for p in model.diffusion_model.parameters():
                p.normal_(mean=0.0, std=0.02)
        model.diffusion_model.eval()

        x = torch.randn(1, 4, 8, 8)
        sigma = torch.tensor([0.3])
        context = torch.randn(1, 5, 32)

        with torch.no_grad():
            out = model.apply_model(x, sigma, c_concat=None, c_crossattn=context)
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
