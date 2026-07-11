"""Tests for modules.core.generate_empty_latent channel-count threading (FWDF-120)
and modules.core.get_previewer's latent-format branching (FWDF-127)."""

import sys
import unittest
from pathlib import Path

import torch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# args_manager calls parse_args() at import time, which chokes on pytest's
# argv. Patch sys.argv before any project modules are imported.
_original_argv = sys.argv
sys.argv = [sys.argv[0]]

import modules.core as core  # noqa: E402
from ldm_patched.contrib.external import EmptyLatentImage  # noqa: E402

sys.argv = _original_argv


class TestGenerateEmptyLatentDefault(unittest.TestCase):
    """Default (unspecified) channel count must remain 4 -- byte-identical SDXL behavior."""

    def test_default_channel_count_is_four(self):
        latent = core.generate_empty_latent(width=64, height=64, batch_size=1)
        self.assertEqual(latent["samples"].shape, (1, 4, 8, 8))

    def test_default_respects_batch_size(self):
        latent = core.generate_empty_latent(width=64, height=64, batch_size=3)
        self.assertEqual(latent["samples"].shape, (3, 4, 8, 8))


class TestGenerateEmptyLatentExplicitChannels(unittest.TestCase):
    """Explicit latent_channels (e.g. Flux/QwenImage's 16) must be honored."""

    def test_explicit_sixteen_channels(self):
        latent = core.generate_empty_latent(width=64, height=64, batch_size=1, latent_channels=16)
        self.assertEqual(latent["samples"].shape, (1, 16, 8, 8))

    def test_explicit_four_channels_matches_default(self):
        latent = core.generate_empty_latent(width=64, height=64, batch_size=1, latent_channels=4)
        self.assertEqual(latent["samples"].shape, (1, 4, 8, 8))

    def test_spatial_downscale_by_eight(self):
        latent = core.generate_empty_latent(width=128, height=256, batch_size=1, latent_channels=16)
        self.assertEqual(latent["samples"].shape, (1, 16, 32, 16))


class TestEmptyLatentImageGenerate(unittest.TestCase):
    """The underlying ldm_patched node must also accept latent_channels directly."""

    def setUp(self):
        self.op = EmptyLatentImage()

    def test_default_is_four_channels(self):
        latent, = self.op.generate(width=64, height=64, batch_size=1)
        self.assertEqual(latent["samples"].shape, (1, 4, 8, 8))

    def test_explicit_sixteen_channels(self):
        latent, = self.op.generate(width=64, height=64, batch_size=1, latent_channels=16)
        self.assertEqual(latent["samples"].shape, (1, 16, 8, 8))




class TestPreviewerGuard(unittest.TestCase):
    """get_previewer() must never route a non-4-channel latent through
    VAEApprox (both checkpoints are 4-channel approximators and would
    crash the preview callback). Formats with latent_rgb_factors (Flux,
    Z-Image) get the FWDF-127 rgb-factors preview path instead; formats
    with neither get a None previewer, which ksampler treats as
    previews-disabled."""

    def test_returns_working_previewer_for_16_channel_flux_latent_format(self):
        from ldm_patched.modules import latent_formats

        class FakeInner:
            latent_format = latent_formats.Flux()

        class FakeModel:
            model = FakeInner()

        previewer = core.get_previewer(FakeModel())
        self.assertIsNotNone(previewer)

        x0 = torch.randn(1, 16, 4, 4)
        preview = previewer(x0, 1, 9)

        self.assertEqual(preview.shape, (4, 4, 3))
        self.assertEqual(preview.dtype.name, 'uint8')
        self.assertTrue((preview >= 0).all())
        self.assertTrue((preview <= 255).all())

    def test_returns_none_for_16_channel_format_with_no_rgb_factors(self):
        class FakeLatentFormat:
            latent_channels = 16
            latent_rgb_factors = None

        class FakeInner:
            latent_format = FakeLatentFormat()

        class FakeModel:
            model = FakeInner()

        self.assertIsNone(core.get_previewer(FakeModel()))


if __name__ == "__main__":
    unittest.main()
