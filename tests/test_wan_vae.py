"""Tests for FWDF-203: the Wan 2.1 3D causal VAE (Krea 2 / Qwen Image / Anima
latent codec) loaded through `ldm_patched.modules.sd.VAE`.

The synthetic tests build a tiny `WanVAE` (dim=8) and feed its state dict through
`VAE(sd=...)`, so they run in CI without model files. The real-weights tests
are skipped unless `modules.config.qwen_image_vae_path()` exists on disk.
"""

import contextlib
import io
import os
import sys
import unittest
from unittest.mock import patch

import safetensors.torch
import torch

from ldm_patched.ldm.models.autoencoder import AutoencoderKL
from ldm_patched.ldm.wan.vae import (
    WAN21_VAE_DETECTION_KEY,
    WAN22_VAE_LAYOUT_KEY,
    ImageModeWanVAE,
    WanVAE,
    is_wan21_vae_state_dict,
)
from ldm_patched.modules import sd as sd_module
from ldm_patched.modules.sd import VAE

# modules.config imports args_manager, which calls parse_args() against the
# real sys.argv at import time and chokes on pytest's own CLI args. Patch
# sys.argv around the first import (same pattern as tests/test_vae_parameterization.py).
_original_argv = sys.argv
sys.argv = [sys.argv[0]]
import modules.config as config  # noqa: E402
sys.argv = _original_argv

WAN21_KWARGS = dict(
    z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
    temperal_downsample=[False, True, True], image_channels=3, conv_out_channels=3,
)


def _wan_state_dict(dim=8, **overrides):
    torch.manual_seed(0)
    return WanVAE(dim=dim, **{**WAN21_KWARGS, **overrides}).state_dict()


def _build_vae(sd, dtype=torch.float32):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        vae = VAE(sd=sd, device=torch.device('cpu'), dtype=dtype)
    return vae, buf.getvalue()


class TestWan21Detection(unittest.TestCase):
    def test_wan21_state_dict_is_detected(self):
        self.assertTrue(is_wan21_vae_state_dict(_wan_state_dict()))

    def test_autoencoder_kl_state_dict_is_not_detected(self):
        ddconfig = {'double_z': True, 'z_channels': 16, 'resolution': 256, 'in_channels': 3,
                    'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2,
                    'attn_resolutions': [], 'dropout': 0.0}
        sd = AutoencoderKL(ddconfig=ddconfig, embed_dim=16).state_dict()
        self.assertFalse(is_wan21_vae_state_dict(sd))

    def test_wan22_layout_is_not_detected_as_wan21(self):
        sd = {WAN21_VAE_DETECTION_KEY: torch.zeros(1), WAN22_VAE_LAYOUT_KEY: torch.zeros(1)}
        self.assertFalse(is_wan21_vae_state_dict(sd))

    def test_wan22_layout_raises_instead_of_building_a_random_autoencoder(self):
        sd = {WAN21_VAE_DETECTION_KEY: torch.zeros(1), WAN22_VAE_LAYOUT_KEY: torch.zeros(1)}
        with self.assertRaises(ValueError) as ctx:
            VAE(sd=sd, device=torch.device('cpu'), dtype=torch.float32)
        self.assertIn("Wan 2.2", str(ctx.exception))


class TestWan21VAELoading(unittest.TestCase):
    def test_loads_as_wan21_with_no_missing_or_leftover_keys(self):
        vae, output = _build_vae(_wan_state_dict())

        self.assertIsInstance(vae.first_stage_model, ImageModeWanVAE)
        self.assertEqual(vae.latent_channels, 16)
        self.assertEqual(vae.downscale_ratio, 8)
        self.assertNotIn("Missing VAE keys", output)
        self.assertNotIn("Leftover VAE keys", output)

    def test_get_sd_round_trips_the_checkpoint_keys(self):
        sd = _wan_state_dict()
        vae, _ = _build_vae(sd)
        self.assertEqual(set(vae.get_sd().keys()), set(sd.keys()))

    def test_non_rgb_variant_is_rejected_at_load(self):
        """sd.VAE encodes/decodes 3-channel images, so an RGBA checkpoint must
        fail loudly at construction rather than mid-pipeline."""
        for overrides in ({'image_channels': 4}, {'conv_out_channels': 4}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError) as ctx:
                    VAE(sd=_wan_state_dict(**overrides), device=torch.device('cpu'), dtype=torch.float32)
                self.assertIn("RGB", str(ctx.exception))

    def test_memory_estimators_match_the_comfyui_single_frame_constants(self):
        vae, _ = _build_vae(_wan_state_dict())
        self.assertEqual(vae.memory_used_encode((1, 3, 32, 32), torch.float16), 1500 * 32 * 32 * 2)
        self.assertEqual(vae.memory_used_decode((1, 16, 4, 4), torch.float32), 2200 * 4 * 4 * 64 * 4)

    def test_dtype_defaults_to_model_management_vae_dtype(self):
        with patch.object(sd_module.model_management, 'vae_dtype', return_value=torch.bfloat16):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                vae = VAE(sd=_wan_state_dict(), device=torch.device('cpu'))
        self.assertEqual(vae.vae_dtype, torch.bfloat16)
        self.assertEqual(next(vae.first_stage_model.parameters()).dtype, torch.bfloat16)


class TestWan21ImageModeMechanics(unittest.TestCase):
    """Shape and equivalence checks on a randomly initialised model. Untrained
    residual stacks may produce non-finite activations, so only shapes (and
    equivalence to the 5-D path) are asserted."""

    def setUp(self):
        self.vae, _ = _build_vae(_wan_state_dict())

    def test_decode_returns_4d_channels_last_image(self):
        with torch.no_grad():
            pixels = self.vae.decode(torch.randn(1, 16, 4, 4))
        self.assertEqual(tuple(pixels.shape), (1, 32, 32, 3))

    def test_encode_returns_4d_latent(self):
        with torch.no_grad():
            latent = self.vae.encode(torch.rand(1, 32, 32, 3))
        self.assertEqual(tuple(latent.shape), (1, 16, 4, 4))

    def test_batched_decode_and_encode(self):
        with torch.no_grad():
            self.assertEqual(tuple(self.vae.decode(torch.randn(2, 16, 4, 4)).shape), (2, 32, 32, 3))
            self.assertEqual(tuple(self.vae.encode(torch.rand(2, 32, 32, 3)).shape), (2, 16, 4, 4))

    def test_tiled_decode_returns_expected_shape(self):
        with torch.no_grad():
            pixels = self.vae.decode_tiled(torch.randn(1, 16, 8, 8), tile_x=4, tile_y=4, overlap=1)
        self.assertEqual(tuple(pixels.shape), (1, 64, 64, 3))

    def test_tiled_encode_returns_expected_shape(self):
        # encode_tiled_ accumulates in place on tiled_scale's (inference-mode)
        # output, so callers run it under inference_mode, as the pipeline does.
        with torch.inference_mode():
            latent = self.vae.encode_tiled(torch.rand(1, 64, 64, 3), tile_x=32, tile_y=32, overlap=8)
        self.assertEqual(tuple(latent.shape), (1, 16, 8, 8))

    def test_image_mode_matches_single_frame_video_path(self):
        model = self.vae.first_stage_model
        z = torch.randn(1, 16, 4, 4)
        x = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(model.decode(z), WanVAE.decode(model, z.unsqueeze(2))[:, :, 0],
                                       equal_nan=True)
            torch.testing.assert_close(model.encode(x), WanVAE.encode(model, x.unsqueeze(2))[:, :, 0],
                                       equal_nan=True)

    def test_multi_frame_video_encode_decode_shapes(self):
        """The native 5-D path (feature cache, T>1) still works: 5 frames ->
        2 latent frames -> 5 frames (1 + 4N temporal rule)."""
        model = WanVAE(dim=8, **WAN21_KWARGS).eval()
        with torch.no_grad():
            latent = model.encode(torch.rand(1, 3, 5, 16, 16))
            video = model.decode(latent)
        self.assertEqual(tuple(latent.shape), (1, 16, 2, 2, 2))
        self.assertEqual(tuple(video.shape), (1, 3, 5, 16, 16))


_REAL_VAE_PATH = config.qwen_image_vae_path()
# Measured ~54 dB on a solid colour with the real weights; 35 leaves headroom
# for platform/dtype differences while still catching a broken port.
PSNR_THRESHOLD_DB = 35.0


@unittest.skipUnless(os.path.isfile(_REAL_VAE_PATH), "qwen_image_vae.safetensors not present locally")
class TestRealQwenImageVAE(unittest.TestCase):
    """Real-weights checks; skipped unless the FWDF-151 companion file exists."""

    @classmethod
    def setUpClass(cls):
        sd = safetensors.torch.load_file(_REAL_VAE_PATH)
        cls.vae, cls.output = _build_vae(sd)

    def test_loads_with_no_missing_or_leftover_keys(self):
        self.assertIsInstance(self.vae.first_stage_model, ImageModeWanVAE)
        self.assertEqual(self.vae.latent_channels, 16)
        self.assertEqual(self.vae.downscale_ratio, 8)
        self.assertNotIn("Missing VAE keys", self.output)
        self.assertNotIn("Leftover VAE keys", self.output)

    def test_fixed_seed_latent_decodes_to_finite_image(self):
        generator = torch.Generator().manual_seed(1234)
        latent = torch.randn(1, 16, 8, 8, generator=generator)
        with torch.no_grad():
            pixels = self.vae.decode(latent)
        self.assertEqual(tuple(pixels.shape), (1, 64, 64, 3))
        self.assertTrue(torch.isfinite(pixels).all())

    def test_solid_colour_round_trip_stays_within_psnr_threshold(self):
        colour = torch.tensor([0.8, 0.4, 0.2])
        image = colour.expand(1, 64, 64, 3).contiguous()
        with torch.no_grad():
            latent = self.vae.encode(image)
            reconstructed = self.vae.decode(latent)
        self.assertEqual(tuple(latent.shape), (1, 16, 8, 8))
        mse = torch.mean((reconstructed - image) ** 2).item()
        psnr = 10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item()
        self.assertGreater(psnr, PSNR_THRESHOLD_DB)


if __name__ == '__main__':
    unittest.main()
