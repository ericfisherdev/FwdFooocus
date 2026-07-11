"""
Unit tests for FWDF-121: parameterized VAE latent channel / downscale ratio
derivation, and the clear-error path for checkpoints with no embedded VAE.

All state dicts are synthetic (built from freshly-constructed AutoencoderKL
instances), no real model weights are downloaded or required.
"""

import contextlib
import io
import sys
import unittest
from unittest.mock import patch

import torch

from ldm_patched.ldm.models.autoencoder import AutoencoderKL
from ldm_patched.modules.sd import VAE, MissingVAEError, require_embedded_vae_state_dict

# modules.config imports args_manager, which calls parse_args() against the
# real sys.argv at import time and chokes on pytest's own CLI args. Patch
# sys.argv before the first import of modules.core, matching the pattern
# already used in tests/test_new_ui_app.py.
_original_argv = sys.argv
sys.argv = [sys.argv[0]]
import modules.core as core  # noqa: E402
sys.argv = _original_argv


def _build_autoencoder_state_dict(z_channels, ch_mult=(1, 2, 4, 4)):
    ddconfig = {
        'double_z': True, 'z_channels': z_channels, 'resolution': 256,
        'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': list(ch_mult),
        'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0,
    }
    reference = AutoencoderKL(ddconfig=ddconfig, embed_dim=z_channels)
    return reference.state_dict()


class TestVAELatentChannelDerivation(unittest.TestCase):
    """Covers the shape-derived latent_channels/downscale_ratio behavior."""

    def _construct_vae_without_key_warnings(self, sd):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            vae = VAE(sd=sd, device=torch.device('cpu'), dtype=torch.float32)
        output = buf.getvalue()
        return vae, output

    def test_four_channel_state_dict_is_regression_safe(self):
        """A standard SD1.x/SDXL-shaped (4-channel) state dict must keep the
        pre-existing hardcoded behavior: latent_channels=4, downscale_ratio=8,
        and load cleanly with no missing/unexpected keys."""
        sd = _build_autoencoder_state_dict(z_channels=4)
        vae, output = self._construct_vae_without_key_warnings(sd)

        self.assertEqual(vae.latent_channels, 4)
        self.assertEqual(vae.downscale_ratio, 8)
        self.assertNotIn("Missing VAE keys", output)
        self.assertNotIn("Leftover VAE keys", output)

    def test_sixteen_channel_state_dict_derives_flux_qwen_shape(self):
        """Flux's ae.safetensors and the Qwen Image VAE are both 16-channel
        AutoencoderKL-shaped checkpoints; the VAE class must derive
        latent_channels=16 from the state dict instead of hardcoding 4."""
        sd = _build_autoencoder_state_dict(z_channels=16)
        vae, output = self._construct_vae_without_key_warnings(sd)

        self.assertEqual(vae.latent_channels, 16)
        self.assertEqual(vae.downscale_ratio, 8)
        self.assertNotIn("Missing VAE keys", output)
        self.assertNotIn("Leftover VAE keys", output)

    def test_x4_upscaler_variant_is_unaffected(self):
        """The pre-existing SD x4-upscaler VAE special case (ch_mult=[1,2,4],
        downscale_ratio=4) must remain unchanged by the new derivation logic."""
        sd = _build_autoencoder_state_dict(z_channels=4, ch_mult=(1, 2, 4))
        vae, output = self._construct_vae_without_key_warnings(sd)

        self.assertEqual(vae.latent_channels, 4)
        self.assertEqual(vae.downscale_ratio, 4)
        self.assertNotIn("Missing VAE keys", output)
        self.assertNotIn("Leftover VAE keys", output)

    def test_missing_decoder_conv_in_key_falls_back_to_four_channels(self):
        """If a state dict lacks the discriminant key entirely, fall back to
        the historical hardcoded default rather than raising or guessing."""
        sd = _build_autoencoder_state_dict(z_channels=4)
        del sd['decoder.conv_in.weight']

        vae = VAE(sd=sd, device=torch.device('cpu'), dtype=torch.float32)

        self.assertEqual(vae.latent_channels, 4)
        self.assertEqual(vae.downscale_ratio, 8)


class TestSixteenChannelEncodeDecodeMechanics(unittest.TestCase):
    """Proves the 16-channel path is wired end to end at the shape/dtype
    level. The VAE under test is randomly initialized (no real Flux/Qwen
    Image weights), so untrained residual stacks can legitimately produce
    non-finite activations -- that is an expected property of random
    weights, not a correctness signal, so these tests assert shape/dtype
    only. Real-weight decode fidelity is a stretch goal per the ticket's
    acceptance criteria and requires the actual trained checkpoint."""

    def setUp(self):
        sd = _build_autoencoder_state_dict(z_channels=16)
        self.vae = VAE(sd=sd, device=torch.device('cpu'), dtype=torch.float32)

    def test_decode_produces_correctly_shaped_image(self):
        latent = torch.randn(1, 16, 4, 4)
        with torch.no_grad():
            pixels = self.vae.decode(latent)
        self.assertEqual(tuple(pixels.shape), (1, 32, 32, 3))
        self.assertEqual(pixels.dtype, torch.float32)

    def test_encode_produces_correctly_shaped_latent(self):
        image = torch.rand(1, 32, 32, 3)
        with torch.no_grad():
            samples = self.vae.encode(image)
        self.assertEqual(tuple(samples.shape), (1, 16, 4, 4))
        self.assertEqual(samples.dtype, torch.float32)

    def test_encode_decode_roundtrip_preserves_shape(self):
        image = torch.rand(1, 32, 32, 3)
        with torch.no_grad():
            samples = self.vae.encode(image)
            reconstructed = self.vae.decode(samples)
        self.assertEqual(reconstructed.shape, image.shape)


class TestMissingVAEGuard(unittest.TestCase):
    """Covers the no-embedded-VAE clear-error path (FWDF-121 gap b)."""

    def test_empty_state_dict_raises_missing_vae_error(self):
        with self.assertRaises(MissingVAEError) as ctx:
            require_embedded_vae_state_dict({}, "z_image_dit.safetensors")
        message = str(ctx.exception)
        self.assertIn("z_image_dit.safetensors", message)
        self.assertIn("no embedded vae weights", message.lower())

    def test_non_empty_state_dict_does_not_raise(self):
        try:
            require_embedded_vae_state_dict({'decoder.conv_in.weight': torch.zeros(1)}, "sdxl.safetensors")
        except MissingVAEError:
            self.fail("require_embedded_vae_state_dict raised for a non-empty state dict")

    def test_core_load_model_reraises_with_configured_vae_directory(self):
        """modules.core.load_model() must translate the generic ldm_patched
        error into an app-level message naming the configured VAE directory,
        since ldm_patched has no knowledge of modules.config."""
        with patch('modules.core.load_checkpoint_guess_config',
                   side_effect=MissingVAEError("Checkpoint 'dit_only.safetensors' has no embedded VAE weights.")):
            with self.assertRaises(MissingVAEError) as ctx:
                core.load_model('dit_only.safetensors')

        message = str(ctx.exception)
        self.assertIn("dit_only.safetensors", message)
        self.assertIn(core.path_vae, message)


if __name__ == '__main__':
    unittest.main()
