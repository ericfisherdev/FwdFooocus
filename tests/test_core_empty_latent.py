"""Tests for modules.core.generate_empty_latent channel-count threading (FWDF-120)."""

import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()


class TestPreviewerGuard(unittest.TestCase):
    """get_previewer() must refuse non-4-channel latent formats: both
    VAEApprox checkpoints are 4-channel approximators, and ksampler treats
    a None previewer as previews-disabled (FWDF-127 adds the real
    16-channel preview path)."""

    def test_returns_none_for_16_channel_latent_format(self):
        from ldm_patched.modules import latent_formats

        class FakeInner:
            latent_format = latent_formats.Flux()

        class FakeModel:
            model = FakeInner()

        self.assertIsNone(core.get_previewer(FakeModel()))
